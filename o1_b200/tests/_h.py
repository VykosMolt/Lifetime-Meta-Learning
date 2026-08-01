"""Tiny self-contained test harness (no external test dependency).

Mirrors the sealed package's style: each test module exposes run() -> (pass,
fail, details); run_all_b200_tests.py aggregates and writes the test report.
"""
from __future__ import annotations

import os
import sys
import traceback

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SCRATCH = os.environ.get(
    "O1_B200_TEST_SCRATCH",
    os.path.join(_ROOT, "o1_b200", "reports", "local_runs", "test_scratch"))
os.makedirs(SCRATCH, exist_ok=True)

CORPUS_DIR = os.path.join(_ROOT, "o1_b200", "corpus")

FAST_SUBSET = [
    "b200val-000-commit_a", "b200val-004-malformed_eos",
    "b200val-008-stoch_commit", "b200val-010-stoch_eos",
]


class Runner:
    def __init__(self, name: str):
        self.name = name
        self.results: list[tuple[str, bool, str]] = []

    def check(self, label: str, fn) -> None:
        try:
            fn()
            self.results.append((label, True, ""))
        except BaseException:  # noqa: BLE001
            self.results.append((label, False, traceback.format_exc(limit=8)))

    def expect_raises(self, label: str, exc_types, fn) -> None:
        def _inner():
            try:
                fn()
            except exc_types:
                return
            raise AssertionError(f"{label}: expected {exc_types} not raised")
        self.check(label, _inner)

    def summary(self) -> tuple[int, int, list]:
        p = sum(1 for _, ok, _ in self.results if ok)
        f = len(self.results) - p
        fails = [(l, tb) for l, ok, tb in self.results if not ok]
        return p, f, fails

    def report(self) -> int:
        p, f, fails = self.summary()
        print(f"[{self.name}] {p} passed, {f} failed")
        for label, tb in fails:
            print(f"  FAIL {label}\n{tb}")
        return 1 if f else 0


def fresh_dir(name: str) -> str:
    import shutil
    d = os.path.join(SCRATCH, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d
