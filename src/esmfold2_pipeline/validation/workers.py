from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
MIN_STALE_RECOVERY_SECONDS = 90.0
STALE_RECOVERY_HEARTBEAT_MULTIPLIER = 3.0


@dataclass(frozen=True)
class ValidationWorkerResult:
    worker_id: str
    gpu_id: str
    pid: int | None
    returncode: int
    completed_tasks: int
    recovered_tasks: int
    log_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class RunMultiValidationResult:
    campaign_dir: Path
    run_id: str
    worker_results: list[ValidationWorkerResult]
    startup_recovered_tasks: int = 0

    @property
    def completed_tasks(self) -> int:
        return sum(worker.completed_tasks for worker in self.worker_results)

    @property
    def recovered_tasks(self) -> int:
        return self.startup_recovered_tasks + sum(
            worker.recovered_tasks for worker in self.worker_results
        )

    @property
    def failed_workers(self) -> int:
        return sum(1 for worker in self.worker_results if not worker.ok)

    @property
    def ok(self) -> bool:
        return self.failed_workers == 0


@dataclass
class _RunningValidationWorker:
    worker_id: str
    gpu_id: str
    process: subprocess.Popen[bytes]
    log_handle: TextIO
    log_path: Path
    cleanup_scratch_on_failure: bool
    returncode: int | None = None
    recovered_tasks: int = 0


def run_multi_validation(
    campaign_dir: str | Path,
    *,
    gpu_ids: Sequence[str],
    worker_prefix: str = "validation-gpu",
    max_tasks_per_worker: int | None = None,
    poll_interval_seconds: float = 2.0,
    heartbeat_interval_seconds: float = 30.0,
    stale_after_seconds: float | None = None,
    python_executable: str | Path | None = None,
    worker_args: Sequence[str] = (),
    extra_env: Mapping[str, str] | None = None,
) -> RunMultiValidationResult:
    root = Path(campaign_dir)
    normalized_gpu_ids = _normalize_gpu_ids(gpu_ids)
    if max_tasks_per_worker is not None and max_tasks_per_worker <= 0:
        raise ValueError("max_tasks_per_worker must be positive")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be positive")
    resolved_stale_after_seconds = resolve_stale_after_seconds(
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        stale_after_seconds=stale_after_seconds,
    )

    run_id = _run_id()
    log_dir = root / "logs" / "validation_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    startup_recovered_tasks = _recover_stale_validation_tasks(
        root,
        stale_after_seconds=resolved_stale_after_seconds,
    )

    executable = str(python_executable or sys.executable)
    workers: list[_RunningValidationWorker] = []
    try:
        for index, gpu_id in enumerate(normalized_gpu_ids):
            worker_id = _worker_id(worker_prefix, gpu_id, index=index, run_id=run_id)
            log_path = log_dir / f"{worker_id}.log"
            command = _validation_worker_command(
                executable=executable,
                campaign_dir=root,
                worker_id=worker_id,
                gpu_id=gpu_id,
                max_tasks_per_worker=max_tasks_per_worker,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                worker_args=worker_args,
            )
            env = _worker_env(
                gpu_id=gpu_id,
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
                _RunningValidationWorker(
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    process=process,
                    log_handle=handle,
                    log_path=log_path,
                    cleanup_scratch_on_failure=_should_cleanup_failed_worker_scratch(
                        worker_args
                    ),
                )
            )

        _wait_for_validation_workers(root, workers, poll_interval_seconds=poll_interval_seconds)
    except BaseException:
        _terminate_validation_workers(workers)
        raise
    finally:
        for worker in workers:
            if not worker.log_handle.closed:
                worker.log_handle.close()

    return RunMultiValidationResult(
        campaign_dir=root,
        run_id=run_id,
        startup_recovered_tasks=startup_recovered_tasks,
        worker_results=[
            ValidationWorkerResult(
                worker_id=worker.worker_id,
                gpu_id=worker.gpu_id,
                pid=worker.process.pid,
                returncode=int(worker.returncode or 0),
                completed_tasks=_completed_validation_tasks_for_worker(
                    root,
                    worker.worker_id,
                ),
                recovered_tasks=worker.recovered_tasks,
                log_path=worker.log_path,
            )
            for worker in workers
        ],
    )


def _validation_worker_command(
    *,
    executable: str,
    campaign_dir: Path,
    worker_id: str,
    gpu_id: str,
    max_tasks_per_worker: int | None,
    heartbeat_interval_seconds: float,
    worker_args: Sequence[str],
) -> list[str]:
    command = [
        executable,
        "-m",
        "esmfold2_pipeline",
        "validate-run",
        str(campaign_dir),
        "--worker-id",
        worker_id,
        "--gpu-id",
        gpu_id,
        "--heartbeat-interval",
        str(heartbeat_interval_seconds),
    ]
    if max_tasks_per_worker is not None:
        command.extend(["--max-tasks", str(max_tasks_per_worker)])
    command.extend(str(arg) for arg in worker_args)
    return command


def _wait_for_validation_workers(
    root: Path,
    workers: list[_RunningValidationWorker],
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
                worker.recovered_tasks = _recover_worker_validation_tasks(root, worker)
            finished.append(index)

        for index in finished:
            pending.remove(index)

        if pending:
            time.sleep(poll_interval_seconds)


def _recover_worker_validation_tasks(
    root: Path,
    worker: _RunningValidationWorker,
) -> int:
    conn = connect_database(root / "campaign.sqlite")
    try:
        store = CampaignStore(conn)
        recovered = store.recover_failed_worker_validation_tasks(
            worker_id=worker.worker_id,
            pid=worker.process.pid,
            exit_code=worker.returncode,
            error_message=(
                f"validation worker exited with status {worker.returncode}; "
                f"see {worker.log_path}"
            ),
        )
        if worker.cleanup_scratch_on_failure:
            _cleanup_failed_worker_scratch(root, worker.worker_id)
        return recovered
    finally:
        conn.close()


def _recover_stale_validation_tasks(root: Path, *, stale_after_seconds: float) -> int:
    conn = connect_database(root / "campaign.sqlite")
    try:
        store = CampaignStore(conn)
        return store.recover_stale_validation_tasks(
            stale_before=stale_before_timestamp(stale_after_seconds),
            error_message=(
                "running validation heartbeat exceeded "
                f"{stale_after_seconds:g}s; recovering for resume"
            ),
        )
    finally:
        conn.close()


def _completed_validation_tasks_for_worker(root: Path, worker_id: str) -> int:
    conn = connect_database(root / "campaign.sqlite")
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM validation_tasks
            WHERE claim_worker_id = ?
              AND status = 'completed'
            """,
            (worker_id,),
        ).fetchone()
        return int(row["count"])
    finally:
        conn.close()


def _terminate_validation_workers(workers: list[_RunningValidationWorker]) -> None:
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


def _should_cleanup_failed_worker_scratch(worker_args: Sequence[str]) -> bool:
    return "--keep-validation-debug" not in {str(arg) for arg in worker_args}


def _cleanup_failed_worker_scratch(root: Path, worker_id: str) -> None:
    scratch_dir = (
        root
        / ".scratch"
        / "protenix_validation"
        / _safe_identifier(worker_id, max_len=80)
    )
    shutil.rmtree(scratch_dir, ignore_errors=True)
    current = scratch_dir.parent
    stop = root.resolve()
    while True:
        try:
            resolved = current.resolve()
        except FileNotFoundError:
            resolved = current
        if resolved == stop:
            break
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _safe_identifier(value: str, *, max_len: int) -> str:
    chars = [
        char.lower() if char.isalnum() else "_"
        for char in str(value).strip()
    ]
    text = "".join(chars).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    if not text:
        text = "worker"
    return text[:max_len].strip("_") or "worker"


def _worker_env(
    *,
    gpu_id: str,
    extra_env: Mapping[str, str] | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
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
        safe_prefix = "validation-gpu"
    return f"{safe_prefix}-gpu{safe_gpu}-{run_id}"


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{os.getpid()}"


def resolve_stale_after_seconds(
    *,
    heartbeat_interval_seconds: float,
    stale_after_seconds: float | None,
) -> float:
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be positive")
    if stale_after_seconds is None:
        return max(
            MIN_STALE_RECOVERY_SECONDS,
            STALE_RECOVERY_HEARTBEAT_MULTIPLIER * heartbeat_interval_seconds,
        )
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    return stale_after_seconds


def stale_before_timestamp(stale_after_seconds: float) -> str:
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    stale_before = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
    return stale_before.isoformat(timespec="milliseconds").replace("+00:00", "Z")
