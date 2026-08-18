#!/usr/bin/env python3
"""Run every B200-runner test module; write the aggregate test report.

Modules are independent (own scratch dir, own process) so the default mode
runs them in parallel: wall time becomes the slowest module rather than the
sum of all of them.  ``--jobs 1`` forces the original single-process
sequential path; the emitted report is identical either way.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _h  # noqa: E402  (sets repo root on sys.path)

MODULES = [
    "test_row_rng",
    "test_records",
    "test_persistence",
    "test_intervention_transport",
    "test_engine_batched",
    "test_backends",
    "test_corpus_and_policies",
    "test_budget_watchdog",
    "test_state_machine",
    "test_templates_and_integrity",
    "test_runpod_adapter",
    "test_runpod_interlock",
    "test_runpod_lifecycle",
    "test_runpod_zero_touch",
    "test_hardware_gate",
    "test_launch_path",
    "test_preemption",
    "test_review_repairs",
    "test_credential_isolation",
]

# Longest first: with a bounded pool the long poles must start immediately or
# they serialise behind the short ones and set the wall clock on their own.
_SLOW_FIRST = ["test_engine_batched", "test_credential_isolation",
               "test_backends", "test_intervention_transport",
               "test_corpus_and_policies", "test_records",
               "test_persistence"]

_RESULT_MARKER = "@@MODULE_RESULT@@ "


def _run_module(name: str) -> dict:
    """In-process execution of one module (used by --jobs 1 and by --child)."""
    mod = importlib.import_module(name)
    runner = mod.run()
    p, f, fails = runner.summary()
    return {"module": name, "passed": p, "failed": f,
            "failures": [[label, tb] for label, tb in fails],
            "checks": [label for label, ok, _ in runner.results]}


def _run_module_subprocess(name: str) -> dict:
    import subprocess
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Without this every concurrent torch process claims all cores and they
    # thrash.  Safe for the recorded values: the synthetic runtime is
    # deliberately all-elementwise (no reductions), so thread count cannot
    # change a bit of its output.
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    # Per-module scratch: modules pick fixed fresh_dir() names, so a shared
    # scratch root would let concurrent modules delete each other's fixtures.
    env["O1_B200_TEST_SCRATCH"] = os.path.join(_h.SCRATCH, "par", name)
    os.makedirs(env["O1_B200_TEST_SCRATCH"], exist_ok=True)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", name],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    tail = (proc.stdout + proc.stderr)[-4000:]
    return {"module": name, "passed": 0, "failed": 1,
            "failures": [[f"{name}: module process died (rc={proc.returncode})",
                          tail]],
            "checks": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel module processes (0 = auto, 1 = in-process "
                         "sequential)")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        res = _run_module(args.child)
        sys.stdout.write(_RESULT_MARKER + json.dumps(res) + "\n")
        return 1 if res["failed"] else 0

    jobs = args.jobs or min(len(MODULES), (os.cpu_count() or 1), 10)
    t0 = time.monotonic()
    timings: dict[str, float] = {}

    if jobs == 1:
        results = {}
        for name in MODULES:
            t = time.monotonic()
            results[name] = _run_module(name)
            timings[name] = time.monotonic() - t
    else:
        order = _SLOW_FIRST + [m for m in MODULES if m not in _SLOW_FIRST]

        def _timed(name: str) -> tuple[str, dict, float]:
            t = time.monotonic()
            return name, _run_module_subprocess(name), time.monotonic() - t

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            done = list(pool.map(_timed, order))
        results = {n: r for n, r, _ in done}
        timings = {n: d for n, _, d in done}

    total_pass = total_fail = 0
    module_reports = []
    for name in MODULES:                      # fixed order => stable report
        res = results[name]
        total_pass += res["passed"]
        total_fail += res["failed"]
        status = "PASS" if res["failed"] == 0 else "FAIL"
        print(f"{status:4s}  {name}: {res['passed']} passed, "
              f"{res['failed']} failed ({timings[name]:.1f}s)")
        for label, tb in res["failures"]:
            print(f"      FAIL {label}")
            print("      " + tb.replace("\n", "\n      "))
        module_reports.append({
            "module": name, "passed": res["passed"], "failed": res["failed"],
            "failures": [label for label, _ in res["failures"]],
            "checks": res["checks"],
        })
    verdict = "ALL O1 B200 RUNNER TESTS PASSED" if total_fail == 0 \
        else f"{total_fail} FAILURES"
    print("=" * 70)
    print(f"{verdict}  ({total_pass} checks, "
          f"{time.monotonic() - t0:.1f}s wall, {jobs} job(s))")
    report = {
        "schema": "o1b200.test_report.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_passed": total_pass,
        "total_failed": total_fail,
        "verdict": verdict,
        "modules": module_reports,
        "scope_note": ("Hardware-independent local validation only: synthetic "
                       "runtime + non-O1 corpus + mocked provider/clock. "
                       "No B200 hardware claim is made or implied."),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "reports", "TEST_REPORT.json")
    with open(os.path.abspath(out), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
