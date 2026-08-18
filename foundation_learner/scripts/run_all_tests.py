#!/usr/bin/env python3
"""Aggregate local validation runner (contract §18, §21b).

Three suites, in order, each reported separately:

1. **unit + hostile** — ``pytest`` over ``foundation_learner/tests`` (the
   hostile fixtures live in ``tests/hostile/`` and are permanent regressions);
2. **dress rehearsal** — the offline miniature campaign through the REAL
   supervisor and scheduler;
3. **package checksums** — ``SHA256SUMS`` exact-coverage verification, run only
   when a package has actually been built (otherwise reported as ``SKIPPED``
   with the reason, never as a pass).

``--fast`` deselects the measured slow tests (the long model walkers); the
report then names every deselected node, so a fast run can never be mistaken
for a full one.

Writes ``foundation_learner/reports/TEST_REPORT.json``
(``flb200.test_report.v1``).  Everything here is hardware-independent: tiny
nonscientific model, tiny pools, stub O1 command, CPU.  **No B200 claim is made
or implied.**
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

REPORT_SCHEMA = "flb200.test_report.v1"
PACKAGE = os.path.join(_PKG_PARENT, "foundation_learner")
REPORTS_DIR = os.path.join(PACKAGE, "reports")
TESTS_DIR = os.path.join(PACKAGE, "tests")

#: The MEASURED slow modules: every module that contains at least one test
#: taking >= 20 s on CPU in the full run (``pytest --durations``; full suite
#: 1478 tests / ~28 min, of which these modules are the great majority).
#: ``--fast`` ignores exactly these, and the report NAMES them, so a fast run
#: can never be mistaken for a full one.  No test is weakened or deleted.
SLOW_NODE_IDS: tuple[str, ...] = (
    # walks the WHOLE ladder through the production entry (Amendment 12
    # item 1); minutes on CPU, and the dress rehearsal covers the same walk
    "tests/test_campaign_entry.py",
    "tests/test_training_trainer.py",
    "tests/test_training_checkpointing.py",
    "tests/test_evaluation_fl0_base.py",
    "tests/test_evaluation_learning_curve.py",
    "tests/test_evaluation_context_reset.py",
    "tests/test_evaluation_interference.py",
    "tests/test_evaluation_poison_eval.py",
    "tests/test_evaluation_remap_eval.py",
    "tests/test_evaluation_generation.py",
    "tests/test_mechanisms_integration.py",
    "tests/test_mechanisms_value_head.py",
    "tests/test_mechanisms_fl5_training.py",
    "tests/test_mechanisms_fast_adapter.py",
)

_SUMMARY_RE = re.compile(
    r"(?:(?P<passed>\d+) passed)?(?:, )?(?:(?P<failed>\d+) failed)?"
    r"(?:, )?(?:(?P<errors>\d+) error)?(?:, )?(?:(?P<skipped>\d+) skipped)?"
    r"(?:, )?(?:(?P<deselected>\d+) deselected)?")


def _parse_pytest_summary(text: str) -> dict:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0,
              "deselected": 0}
    for line in reversed(text.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            for key in counts:
                match = re.search(rf"(\d+) {key[:-1] if key == 'errors' else key}",
                                  line)
                if match:
                    counts[key] = int(match.group(1))
            break
    return counts


def run_pytest(*, fast: bool, extra_args: list[str] | None = None) -> dict:
    args = [sys.executable, "-m", "pytest", TESTS_DIR, "-q", "--no-header",
            "-p", "no:cacheprovider"]
    deselected: list[str] = []
    if fast:
        for node in SLOW_NODE_IDS:
            path = os.path.join(PACKAGE, node)
            if os.path.exists(path):
                args += ["--ignore", path]
                deselected.append(node)
    args += list(extra_args or [])
    started = time.monotonic()
    proc = subprocess.run(args, cwd=_PKG_PARENT, capture_output=True, text=True)
    seconds = time.monotonic() - started
    counts = _parse_pytest_summary(proc.stdout + proc.stderr)
    failures = [ln for ln in (proc.stdout + proc.stderr).splitlines()
                if ln.startswith("FAILED") or ln.startswith("ERROR")]
    return {
        "suite": "unit_and_hostile",
        "command": " ".join(args[1:]),
        "returncode": proc.returncode,
        "seconds": round(seconds, 2),
        "counts": counts,
        "failures": failures,
        "skipped_modules": deselected,
        "verdict": "PASS" if proc.returncode == 0 else "FAIL",
        "tail": (proc.stdout + proc.stderr)[-4000:],
    }


def run_dress_rehearsal(out_dir: str | None = None) -> dict:
    from foundation_learner.scripts import dress_rehearsal as dr

    target = out_dir or os.path.join(REPORTS_DIR, "local_runs",
                                     "dress_rehearsal")
    started = time.monotonic()
    try:
        report = dr.run_rehearsal(target, keep=True, log=lambda *a, **k: None)
        verdict = report["verdict"]
        error = None
    except BaseException as exc:  # noqa: BLE001 - recorded, never swallowed
        report = {}
        verdict = "FAIL"
        error = repr(exc)
    return {
        "suite": "dress_rehearsal",
        "seconds": round(time.monotonic() - started, 2),
        "verdict": verdict,
        "error": error,
        "checks": report.get("checks"),
        "states_walked": (report.get("supervisor") or {}).get("states_walked"),
        "ladder_states": (report.get("ladder") or {}).get("states"),
        "report_path": report.get("report_path"),
    }


def run_package_checksums(*, pregen_root: str | None) -> dict:
    from foundation_learner.scripts import package_release as pr

    sums = os.path.join(PACKAGE, pr.SUMS_NAME)
    if not os.path.isfile(sums):
        return {"suite": "package_checksums", "verdict": "SKIPPED",
                "reason": (f"{pr.SUMS_NAME} does not exist; the release has "
                           "not been built yet")}
    listing = open(sums, encoding="utf-8").read()
    covers_pregen = "artifacts_fl/" in listing
    pregen_present = bool(pregen_root and os.path.isdir(pregen_root))
    if covers_pregen and not pregen_present:
        return {"suite": "package_checksums", "verdict": "SKIPPED",
                "reason": (f"{pr.SUMS_NAME} covers artifacts_fl/pregen, which "
                           "is not present in this checkout (it is git-ignored "
                           "and bitwise reproducible from "
                           "scripts/pregenerate_all.py); regenerate the data "
                           "or rebuild the package to verify coverage")}
    started = time.monotonic()
    try:
        include_pregen = covers_pregen and pregen_present
        coverage = pr.verify_exact_coverage(sums, _PKG_PARENT,
                                            pregen_root=pregen_root,
                                            include_pregen=include_pregen)
        return {"suite": "package_checksums", "verdict": "PASS",
                "seconds": round(time.monotonic() - started, 2),
                "coverage": coverage, "include_pregen": include_pregen}
    except BaseException as exc:  # noqa: BLE001
        return {"suite": "package_checksums", "verdict": "FAIL",
                "seconds": round(time.monotonic() - started, 2),
                "error": repr(exc)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the whole local Foundation Learner validation")
    parser.add_argument("--fast", action="store_true",
                        help="deselect the measured slow model walkers")
    parser.add_argument("--no-rehearsal", action="store_true")
    parser.add_argument("--pregen",
                        default=os.path.join(_PKG_PARENT, "artifacts_fl", "pregen"))
    parser.add_argument("--out", default=os.path.join(REPORTS_DIR,
                                                      "TEST_REPORT.json"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    started = time.monotonic()
    suites = [run_pytest(fast=args.fast)]
    print(f"{suites[0]['verdict']:7s} unit+hostile: {suites[0]['counts']} "
          f"({suites[0]['seconds']}s)")
    for line in suites[0]["failures"]:
        print("   ", line)

    if not args.no_rehearsal:
        rehearsal = run_dress_rehearsal()
        suites.append(rehearsal)
        print(f"{rehearsal['verdict']:7s} dress rehearsal "
              f"({rehearsal['seconds']}s)")
        if rehearsal["error"]:
            print("   ", rehearsal["error"])

    checksums = run_package_checksums(pregen_root=args.pregen)
    suites.append(checksums)
    print(f"{checksums['verdict']:7s} package checksums "
          f"({checksums.get('reason') or ''})")

    failed = [s for s in suites if s["verdict"] == "FAIL"]
    verdict = "ALL FOUNDATION LEARNER LOCAL CHECKS PASSED" if not failed \
        else f"{len(failed)} SUITE(S) FAILED"
    report = {
        "schema": REPORT_SCHEMA,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.monotonic() - started, 2),
        "fast_mode": bool(args.fast),
        "suites": suites,
        "verdict": verdict,
        "scope_note": (
            "Hardware-independent local validation only: the tiny "
            "NONSCIENTIFIC Ouro model, tiny pools, a stub O1 command, a mocked "
            "clock and CPU execution. No B200 hardware claim, no throughput "
            "claim, and no scientific claim is made or implied. The real "
            "checkpoint is touched only by scripts/local_smoke_test.py, which "
            "is reported separately and is also nonscientific."),
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, args.out)
    print("=" * 70)
    print(f"{verdict}  ({report['seconds']}s) -> {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
