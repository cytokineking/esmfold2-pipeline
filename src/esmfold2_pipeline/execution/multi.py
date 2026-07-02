from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import time
from typing import Mapping, Sequence, TextIO

from esmfold2_pipeline.db import CampaignStore, connect_database
from esmfold2_pipeline.execution.recovery import (
    resolve_stale_after_seconds,
    stale_before_timestamp,
)


@dataclass(frozen=True)
class MultiWorkerResult:
    worker_id: str
    gpu_id: str
    pid: int | None
    returncode: int
    completed_shards: int
    recovered_shards: int
    log_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class RunMultiCampaignResult:
    campaign_dir: Path
    run_id: str
    worker_results: list[MultiWorkerResult]
    startup_recovered_shards: int = 0

    @property
    def completed_shards(self) -> int:
        return sum(worker.completed_shards for worker in self.worker_results)

    @property
    def recovered_shards(self) -> int:
        return self.startup_recovered_shards + sum(
            worker.recovered_shards for worker in self.worker_results
        )

    @property
    def failed_workers(self) -> int:
        return sum(1 for worker in self.worker_results if not worker.ok)

    @property
    def ok(self) -> bool:
        return self.failed_workers == 0


@dataclass
class _RunningWorker:
    worker_id: str
    gpu_id: str
    process: subprocess.Popen[bytes]
    log_handle: TextIO
    log_path: Path
    returncode: int | None = None
    recovered_shards: int = 0


def run_multi_campaign(
    campaign_dir: str | Path,
    *,
    gpu_ids: Sequence[str],
    esm_repo: str | Path | None = None,
    worker_prefix: str = "local-gpu",
    max_shards_per_worker: int | None = None,
    poll_interval_seconds: float = 2.0,
    heartbeat_interval_seconds: float = 30.0,
    stale_after_seconds: float | None = None,
    python_executable: str | Path | None = None,
    disable_hf_xet: bool = True,
    disable_local_runtime_cache: bool = False,
    extra_env: Mapping[str, str] | None = None,
    worker_subcommand: str = "run",
) -> RunMultiCampaignResult:
    """Run one local campaign worker process per GPU id."""

    root = Path(campaign_dir)
    normalized_gpu_ids = _normalize_gpu_ids(gpu_ids)
    if max_shards_per_worker is not None and max_shards_per_worker <= 0:
        raise ValueError("max_shards_per_worker must be positive")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be positive")
    resolved_stale_after_seconds = resolve_stale_after_seconds(
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        stale_after_seconds=stale_after_seconds,
    )
    if worker_subcommand not in {"run", "run-mock"}:
        raise ValueError("worker_subcommand must be 'run' or 'run-mock'")

    run_id = _run_id()
    log_dir = root / "logs" / "design_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    startup_recovered_shards = _recover_stale_shards(
        root,
        stale_after_seconds=resolved_stale_after_seconds,
    )

    executable = str(python_executable or sys.executable)
    workers: list[_RunningWorker] = []
    try:
        for index, gpu_id in enumerate(normalized_gpu_ids):
            worker_id = _worker_id(worker_prefix, gpu_id, index=index, run_id=run_id)
            log_path = log_dir / f"{worker_id}.log"
            command = _worker_command(
                executable=executable,
                worker_subcommand=worker_subcommand,
                campaign_dir=root,
                worker_id=worker_id,
                gpu_id=gpu_id,
                esm_repo=esm_repo,
                max_shards_per_worker=max_shards_per_worker,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                disable_hf_xet=disable_hf_xet,
                disable_local_runtime_cache=disable_local_runtime_cache,
            )
            env = _worker_env(
                gpu_id=gpu_id,
                disable_hf_xet=disable_hf_xet,
                extra_env=extra_env,
            )
            handle = log_path.open("w", encoding="utf-8", buffering=1)
            handle.write(f"$ {shlex.join(command)}\n")
            handle.flush()
            try:
                process = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            except Exception:
                handle.close()
                raise
            workers.append(
                _RunningWorker(
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    process=process,
                    log_handle=handle,
                    log_path=log_path,
                )
            )

        _wait_for_workers(root, workers, poll_interval_seconds=poll_interval_seconds)
    except BaseException:
        _terminate_workers(workers)
        raise
    finally:
        for worker in workers:
            if not worker.log_handle.closed:
                worker.log_handle.close()

    return RunMultiCampaignResult(
        campaign_dir=root,
        run_id=run_id,
        startup_recovered_shards=startup_recovered_shards,
        worker_results=[
            MultiWorkerResult(
                worker_id=worker.worker_id,
                gpu_id=worker.gpu_id,
                pid=worker.process.pid,
                returncode=int(worker.returncode or 0),
                completed_shards=_completed_shards_for_worker(root, worker.worker_id),
                recovered_shards=worker.recovered_shards,
                log_path=worker.log_path,
            )
            for worker in workers
        ],
    )


def _wait_for_workers(
    root: Path,
    workers: list[_RunningWorker],
    *,
    poll_interval_seconds: float,
) -> None:
    pending = set(range(len(workers)))
    while pending:
        finished: list[int] = []
        for index in sorted(pending):
            worker = workers[index]
            returncode = worker.process.poll()
            if returncode is None:
                continue

            worker.returncode = int(returncode)
            worker.log_handle.flush()
            worker.log_handle.close()
            if returncode != 0:
                worker.recovered_shards = _recover_worker_shards(root, worker)
            finished.append(index)

        for index in finished:
            pending.remove(index)

        if pending:
            time.sleep(poll_interval_seconds)


def _recover_worker_shards(root: Path, worker: _RunningWorker) -> int:
    conn = connect_database(root / "campaign.sqlite")
    try:
        store = CampaignStore(conn)
        return store.recover_failed_worker_shards(
            worker_id=worker.worker_id,
            pid=worker.process.pid,
            exit_code=worker.returncode,
            error_message=(
                f"worker process exited with status {worker.returncode}; "
                f"see {worker.log_path}"
            ),
        )
    finally:
        conn.close()


def _recover_stale_shards(root: Path, *, stale_after_seconds: float) -> int:
    conn = connect_database(root / "campaign.sqlite")
    try:
        store = CampaignStore(conn)
        return store.recover_stale_shards(
            stale_before=stale_before_timestamp(stale_after_seconds),
            error_message=(
                "running shard heartbeat exceeded "
                f"{stale_after_seconds:g}s; recovering for resume"
            ),
        )
    finally:
        conn.close()


def _completed_shards_for_worker(root: Path, worker_id: str) -> int:
    conn = connect_database(root / "campaign.sqlite")
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM attempts
            WHERE worker_id = ?
              AND stage = 'shard'
              AND status = 'completed'
            """,
            (worker_id,),
        ).fetchone()
        return int(row["count"])
    finally:
        conn.close()


def _terminate_workers(workers: list[_RunningWorker]) -> None:
    for worker in workers:
        if worker.process.poll() is None:
            worker.process.terminate()
    deadline = time.monotonic() + 30.0
    for worker in workers:
        while worker.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if worker.process.poll() is None:
            worker.process.kill()
            worker.process.wait()


def _worker_command(
    *,
    executable: str,
    worker_subcommand: str,
    campaign_dir: Path,
    worker_id: str,
    gpu_id: str,
    esm_repo: str | Path | None,
    max_shards_per_worker: int | None,
    heartbeat_interval_seconds: float,
    disable_hf_xet: bool,
    disable_local_runtime_cache: bool,
) -> list[str]:
    command = [
        executable,
        "-m",
        "esmfold2_pipeline",
        worker_subcommand,
        str(campaign_dir),
        "--worker-id",
        worker_id,
        "--gpu-id",
        gpu_id,
    ]
    if worker_subcommand == "run":
        if esm_repo is not None:
            command.extend(["--esm-repo", str(esm_repo)])
        if max_shards_per_worker is not None:
            command.extend(["--max-shards", str(max_shards_per_worker)])
        command.extend(["--heartbeat-interval", str(heartbeat_interval_seconds)])
        if not disable_hf_xet:
            command.append("--enable-hf-xet")
        if disable_local_runtime_cache:
            command.append("--disable-local-runtime-cache")
    return command


def _worker_env(
    *,
    gpu_id: str,
    disable_hf_xet: bool,
    extra_env: Mapping[str, str] | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    if disable_hf_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    if extra_env:
        env.update(extra_env)
    return env


def _normalize_gpu_ids(gpu_ids: Sequence[str]) -> list[str]:
    if _requests_all_gpus(gpu_ids):
        normalized = _discover_available_gpu_ids()
    else:
        normalized = []
        for raw in gpu_ids:
            for value in str(raw).split(","):
                gpu_id = value.strip()
                if not gpu_id:
                    continue
                normalized.extend(_expand_gpu_id_token(gpu_id))

    if not normalized:
        raise ValueError("at least one GPU id is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("GPU ids must be unique")
    return normalized


def _requests_all_gpus(gpu_ids: Sequence[str]) -> bool:
    tokens = [
        value.strip().lower()
        for raw in gpu_ids
        for value in str(raw).split(",")
        if value.strip()
    ]
    if "all" not in tokens:
        return False
    if tokens != ["all"]:
        raise ValueError("--gpus all cannot be combined with explicit GPU ids")
    return True


def _expand_gpu_id_token(gpu_id: str) -> list[str]:
    range_match = re.fullmatch(r"(\d+)-(\d+)", gpu_id)
    if range_match is None:
        return [gpu_id]

    start = int(range_match.group(1))
    end = int(range_match.group(2))
    if end < start:
        raise ValueError(f"GPU range must be ascending: {gpu_id}")
    return [str(index) for index in range(start, end + 1)]


def _discover_available_gpu_ids() -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip().lower() != "all":
        ids = [
            value.strip()
            for value in visible_devices.split(",")
            if value.strip() and value.strip() not in {"-1", "none", "void"}
        ]
        if not ids:
            raise ValueError("CUDA_VISIBLE_DEVICES does not expose any GPUs")
        return ids

    torch_ids = _discover_gpu_ids_from_torch()
    if torch_ids:
        return torch_ids

    nvidia_smi_ids = _discover_gpu_ids_from_nvidia_smi()
    if nvidia_smi_ids:
        return nvidia_smi_ids

    raise ValueError("could not discover any GPUs for --gpus all")


def _discover_gpu_ids_from_torch() -> list[str]:
    try:
        torch = importlib.import_module("torch")
        count = int(torch.cuda.device_count())
    except Exception:
        return []
    return [str(index) for index in range(count)]


def _discover_gpu_ids_from_nvidia_smi() -> list[str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    normalized: list[str] = []
    for line in result.stdout.splitlines():
        gpu_id = line.strip()
        if gpu_id:
            normalized.append(gpu_id)
    return normalized


def _worker_id(worker_prefix: str, gpu_id: str, *, index: int, run_id: str) -> str:
    safe_gpu = re.sub(r"[^A-Za-z0-9_.-]+", "_", gpu_id).strip("_") or str(index)
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", worker_prefix).strip("_")
    if not safe_prefix:
        safe_prefix = "local-gpu"
    return f"{safe_prefix}-gpu{safe_gpu}-{run_id}"


def _run_id() -> str:
    stamp = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )
    return f"{stamp}-{os.getpid()}"
