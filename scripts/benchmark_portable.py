#!/usr/bin/env python3
"""Run one isolated ESM design benchmark in an already-qualified GPU runtime.

Source the image's env.sh before invoking this script. Give each run a fresh
output directory; a baseline uses the image's installed source, while a
candidate can set --source-root to a source-only overlay.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_gpu_memory(done: threading.Event, samples: list[int]) -> None:
    while not done.wait(1):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                try:
                    samples.append(int(line.strip()))
                except ValueError:
                    continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("off", "portable"), required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)
    env = os.environ.copy()
    env["ESMFOLD2_PIPELINE_OPTIMIZATIONS"] = args.mode
    if args.source_root is not None:
        env["PYTHONPATH"] = str(args.source_root / "src")
    else:
        env.pop("PYTHONPATH", None)

    source = subprocess.check_output(
        [str(args.executable.parent / "python"), "-c",
         "import esmfold2_pipeline.esm_adapter.binder_design as b; print(b.__file__)"],
        env=env, text=True,
    ).strip()
    for command in (
        [str(args.executable), "check", str(args.config)],
        [str(args.executable), "plan", str(args.config), "--out", str(args.out / "campaign")],
    ):
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        with (args.out / "setup.log").open("a") as log:
            log.write(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"setup command failed ({result.returncode}): {command[:2]}")

    samples: list[int] = []
    done = threading.Event()
    sampler = threading.Thread(target=_peak_gpu_memory, args=(done, samples), daemon=True)
    sampler.start()
    started_at = _utc()
    start = time.perf_counter()
    command = [str(args.executable), "run", str(args.out / "campaign"), "--max-shards", "1"]
    try:
        with (args.out / "run.log").open("w") as log:
            process = subprocess.Popen(
                command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(f"{_utc()} {line}")
                log.flush()
            returncode = process.wait()
    finally:
        done.set()
        sampler.join(timeout=3)
    elapsed = time.perf_counter() - start
    receipt = {
        "mode": args.mode,
        "source": source,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "started_at": started_at,
        "ended_at": _utc(),
        "run_wall_seconds": elapsed,
        "peak_gpu_memory_mib": max(samples) if samples else None,
        "exit_code": returncode,
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
