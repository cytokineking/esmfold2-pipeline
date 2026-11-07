#!/usr/bin/env python3
"""Launch one local campaign with synchronized timing and memory sampling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import threading
import time

import psutil
import torch

from esmfold2_pipeline import cli
from esmfold2_pipeline.design import loop
from esmfold2_pipeline.esm_adapter import binder_design
from esmfold2_pipeline.esm_adapter.mps import int64_scatter_fallback_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--esm-repo", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="fresh campaign directory for a config")
    parser.add_argument("--checkpoint-dir", type=Path, help="save and resume design progress")
    args = parser.parse_args()
    if os.environ.get("ESMFOLD2_PIPELINE_DEVICE") != "mps":
        raise RuntimeError("set ESMFOLD2_PIPELINE_DEVICE=mps")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process")
    if args.checkpoint_dir is not None:
        os.environ["ESMFOLD2_PIPELINE_CHECKPOINT_DIR"] = str(args.checkpoint_dir.resolve())

    process = psutil.Process()
    done = threading.Event()
    metrics = {"measurement_method": {
        "timing": "time.perf_counter, synchronized with torch.mps.synchronize at phase and step boundaries",
        "rss": "process RSS sampled with psutil at 1 second plus macOS getrusage.ru_maxrss in bytes",
        "mps_driver": "torch.mps.driver_allocated_memory sampled at 1 second; includes allocator and MPS/MPSGraph allocations",
    }, "torch_version": torch.__version__, "precision": "FP32 unquantized weights; upstream attention may cast selected operations to bfloat16",
        "peak_rss_sampled_bytes": 0, "peak_mps_driver_sampled_bytes": 0,
        "completed_steps": 0, "steps": []}
    lock = threading.Lock()
    args.measurements.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        temporary = args.measurements.with_suffix(".tmp")
        temporary.write_text(json.dumps(metrics, indent=2) + "\n")
        temporary.replace(args.measurements)

    def sample():
        while not done.is_set():
            try:
                rss = process.memory_info().rss
                driver = torch.mps.driver_allocated_memory()
                with lock:
                    metrics["peak_rss_sampled_bytes"] = max(metrics["peak_rss_sampled_bytes"], rss)
                    metrics["peak_mps_driver_sampled_bytes"] = max(metrics["peak_mps_driver_sampled_bytes"], driver)
            except Exception as exc:
                with lock:
                    metrics["sampling_error"] = repr(exc)
            done.wait(1.0)

    sampler = threading.Thread(target=sample, daemon=True)
    original_load = binder_design._get_or_load_local_design_runtime
    original_loop = loop.run_design_loop

    def sync():
        torch.mps.synchronize()

    def timed_load(*a, **kw):
        sync()
        start = time.perf_counter()
        try:
            return original_load(*a, **kw)
        finally:
            sync()
            with lock:
                metrics["model_setup_seconds"] = time.perf_counter() - start
                persist()
            print(f"[measurement] model setup {metrics['model_setup_seconds']:.2f}s", flush=True)

    def timed_loop(*a, **kw):
        run_step, score_sequence = kw["run_step"], kw["score_sequence"]
        sync()
        started = time.perf_counter()

        def step(*step_args, **step_kw):
            sync()
            start = time.perf_counter()
            result = run_step(*step_args, **step_kw)
            sync()
            elapsed = time.perf_counter() - start
            step_index = step_kw.get("step", step_args[0] if step_args else None)
            confidence = step_kw.get("calculate_confidence", step_args[2] if len(step_args) > 2 else False)
            with lock:
                if not metrics["steps"]:
                    metrics["start_step"] = step_index
                metrics["completed_steps"] = step_index + 1
                metrics["steps"].append({"step": step_index + 1, "seconds": elapsed,
                                          "confidence": confidence})
                metrics["int64_scatter_cpu_fallbacks"] = int64_scatter_fallback_count()
                persist()
            print(f"[measurement] step {step_index+1}/{kw['steps']} {elapsed:.2f}s confidence={confidence}", flush=True)
            return result

        def critic(*critic_args, **critic_kw):
            sync()
            start = time.perf_counter()
            print("[measurement] critic started", flush=True)
            try:
                return score_sequence(*critic_args, **critic_kw)
            finally:
                sync()
                with lock:
                    metrics["critic_seconds"] = time.perf_counter() - start
                    persist()
                print(f"[measurement] critic {metrics['critic_seconds']:.2f}s", flush=True)

        kw["run_step"], kw["score_sequence"] = step, critic
        try:
            return original_loop(*a, **kw)
        finally:
            sync()
            with lock:
                metrics["design_and_critic_seconds"] = time.perf_counter() - started
                metrics["design_steps_seconds"] = sum(x["seconds"] for x in metrics["steps"])
                persist()

    binder_design._get_or_load_local_design_runtime = timed_load
    loop.run_design_loop = timed_loop
    sampler.start()
    sync()
    started = time.perf_counter()
    exit_code = 1
    try:
        command = ["launch", str(args.config), "--esm-repo", str(args.esm_repo)]
        if args.out is not None:
            command.extend(["--out", str(args.out)])
        exit_code = cli.main(command)
        return exit_code
    finally:
        sync()
        done.set()
        sampler.join()
        with lock:
            metrics["total_elapsed_seconds"] = time.perf_counter() - started
            metrics["peak_rss_getrusage_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metrics["int64_scatter_cpu_fallbacks"] = int64_scatter_fallback_count()
            metrics["exit_code"] = exit_code
            persist()


if __name__ == "__main__":
    raise SystemExit(main())
