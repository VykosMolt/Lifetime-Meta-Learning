#!/usr/bin/env python3
"""Run every B200-runner test module; write the aggregate test report."""
from __future__ import annotations

import importlib
import json
import os
import sys
import time

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
]


def main() -> int:
    t0 = time.monotonic()
    total_pass = total_fail = 0
    module_reports = []
    for name in MODULES:
        t = time.monotonic()
        mod = importlib.import_module(name)
        runner = mod.run()
        p, f, fails = runner.summary()
        total_pass += p
        total_fail += f
        status = "PASS" if f == 0 else "FAIL"
        print(f"{status:4s}  {name}: {p} passed, {f} failed "
              f"({time.monotonic() - t:.1f}s)")
        for label, tb in fails:
            print(f"      FAIL {label}")
            print("      " + tb.replace("\n", "\n      "))
        module_reports.append({
            "module": name, "passed": p, "failed": f,
            "failures": [label for label, _ in fails],
            "checks": [label for label, ok, _ in runner.results],
        })
    verdict = "ALL O1 B200 RUNNER TESTS PASSED" if total_fail == 0 \
        else f"{total_fail} FAILURES"
    print("=" * 70)
    print(f"{verdict}  ({total_pass} checks, "
          f"{time.monotonic() - t0:.1f}s)")
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
