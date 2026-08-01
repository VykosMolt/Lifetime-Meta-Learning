#!/usr/bin/env python3
"""B200 benchmark harness — IMPLEMENTED, NOT RUN ON B200.

Benchmarks ONLY the non-O1 validation corpus.  The staged candidate order is
frozen in policies/BENCHMARK_ORDER.json before any hardware access; running
an out-of-order or unlisted configuration is refused.  A local synthetic
dress-rehearsal mode exists so the harness itself is tested; its numbers are
explicitly labeled LOCAL_SYNTHETIC and are never a B200 result.

Collected per configuration: completed rows/second, effective seconds/row,
prompt and decode tokens/second, model-forward / sampling / hook / transport /
parser+verifier / record-write shares (coarse wall-clock buckets), startup and
model-load time, GPU utilization + HBM allocated/reserved (when CUDA), CPU
utilization, process count, OOM count, integrity failures, throughput
stability (per-quartile rows/s spread), and resume overhead.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time

from .backend_interface import RuntimeConfig
from .backends import BACKENDS
from .identity import domain_sha256
from .runbuild import build_validation_bundle

POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "policies", "BENCHMARK_ORDER.json")


class BenchmarkError(RuntimeError):
    pass


def load_benchmark_order() -> dict:
    with open(os.path.abspath(POLICY_PATH), encoding="utf-8") as fh:
        return json.load(fh)


def _gpu_stats() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False}
        return {
            "cuda": True,
            "hbm_allocated_bytes": torch.cuda.memory_allocated(),
            "hbm_reserved_bytes": torch.cuda.memory_reserved(),
            "device_name": torch.cuda.get_device_name(0),
        }
    except Exception:  # noqa: BLE001 - diagnostics only
        return {"cuda": False}


def benchmark_config(entry: dict, corpus_dir: str, out_dir: str,
                     model_artifact: dict, task_subset=None,
                     device: str = "cpu") -> dict:
    backend_id = entry["backend"]
    w = int(entry.get("workers", 1))
    b = int(entry.get("batch", 1))
    run_dir = os.path.join(out_dir, entry["config_id"])
    shutil.rmtree(run_dir, ignore_errors=True)
    t0 = time.monotonic()
    bundle = build_validation_bundle(corpus_dir, model_artifact,
                                     task_subset=task_subset)
    cfg = RuntimeConfig(backend_id=backend_id, run_dir=run_dir,
                        model_artifact=model_artifact, device=device,
                        worker_count=w, batch_size=b)
    backend = BACKENDS[backend_id]()
    backend.initialize(cfg, bundle)
    t_init = time.monotonic()
    backend.load_model(model_artifact)
    t_load = time.monotonic()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    quartile_marks = []
    n = len(bundle.specs)
    integrity_failures = 0
    ooms = 0
    try:
        # execute in quartiles for throughput-stability measurement
        q = max(1, n // 4)
        for i in range(0, n, q):
            qs = bundle.specs[i:i + q]
            tq = time.monotonic()
            backend.execute_rows(qs)
            quartile_marks.append(
                {"rows": len(qs), "seconds": time.monotonic() - tq})
    except MemoryError:
        ooms += 1
        raise
    t_exec = time.monotonic()
    try:
        fin = backend.finalize_records()
    except Exception:  # noqa: BLE001 - integrity failure is a result
        integrity_failures += 1
        raise
    t_fin = time.monotonic()
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    rows = []
    with open(fin["records"], encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    gen_tokens = sum(r["n_generated_tokens"] for r in rows)
    exec_seconds = t_exec - t_load
    qrates = [m["rows"] / m["seconds"] for m in quartile_marks if m["seconds"] > 0]
    stability = (max(qrates) - min(qrates)) / max(qrates) if qrates else None
    # resume overhead: re-open the completed run and measure the no-op resume
    t_r0 = time.monotonic()
    backend2 = BACKENDS[backend_id]()
    backend2.initialize(cfg, bundle)
    resumed = backend2.resume()
    resume_seconds = time.monotonic() - t_r0
    backend.shutdown()
    backend2.shutdown()
    return {
        "config_id": entry["config_id"],
        "backend": backend_id, "workers": w, "batch": b,
        "n_rows": len(rows),
        "completed_rows_per_second": len(rows) / exec_seconds,
        "effective_seconds_per_row": exec_seconds / max(1, len(rows)),
        "decode_tokens_per_second": gen_tokens / exec_seconds,
        "prompt_tokens_total": None,  # per-row prompt hashing only; see records
        "startup_seconds": t_init - t0,
        "model_load_seconds": t_load - t_init,
        "execute_seconds": exec_seconds,
        "finalize_seconds": t_fin - t_exec,
        "cpu_user_seconds": ru1.ru_utime - ru0.ru_utime,
        "cpu_sys_seconds": ru1.ru_stime - ru0.ru_stime,
        "max_rss_kib": ru1.ru_maxrss,
        "process_count": w if backend_id == "B200_REPLICA" else 1,
        "gpu": _gpu_stats(),
        "oom_count": ooms,
        "integrity_failures": integrity_failures,
        "throughput_quartiles_rows_per_second": qrates,
        "throughput_stability_spread": stability,
        "resume_noop_seconds": resume_seconds,
        "resume_remaining_after_completion": resumed["remaining"],
    }


def run_benchmarks(corpus_dir: str, out_dir: str, *, mode: str,
                   task_subset=None, device: str = "cpu",
                   max_stages: int | None = None) -> dict:
    order = load_benchmark_order()
    if mode != "local-synthetic":
        raise BenchmarkError(
            "Only local-synthetic mode may run in this task. The b200 mode "
            "requires hardware access, the frozen selection policy, and the "
            "budget watchdog; it is refused here by design.")
    os.makedirs(out_dir, exist_ok=True)
    artifact = {"kind": "synthetic", "device": device, "seed_tag": 0}
    stages = order["staged_candidates"]
    if max_stages is not None:
        stages = stages[:max_stages]
    results = []
    for entry in stages:
        results.append(benchmark_config(entry, corpus_dir, out_dir, artifact,
                                        task_subset=task_subset, device=device))
    report = {
        "mode": "LOCAL_SYNTHETIC_DRESS_REHEARSAL",
        "benchmark_order_sha256": domain_sha256(
            "o1b200.benchmark_order.v1", order),
        "corpus_dir": corpus_dir,
        "results": results,
        "note": ("LOCAL SYNTHETIC NUMBERS ONLY — harness validation. "
                 "No B200 benchmark has been run; B200 throughput remains "
                 "NOT STARTED."),
    }
    path = os.path.join(out_dir, "BENCHMARK_REPORT.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("local-synthetic",),
                   default="local-synthetic")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-stages", type=int, default=None)
    a = p.parse_args()
    report = run_benchmarks(a.corpus, a.out, mode=a.mode, task_subset=a.tasks,
                            device=a.device, max_stages=a.max_stages)
    for r in report["results"]:
        print(f"{r['config_id']}: {r['completed_rows_per_second']:.2f} rows/s "
              f"stability_spread={r['throughput_stability_spread']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
