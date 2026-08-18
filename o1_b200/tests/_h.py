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

# The B200_REPLICA backend spawns worker_count processes and each one imports
# torch fresh, which by default claims every core.  Several workers plus the
# parent oversubscribe the machine by an order of magnitude and the wall clock
# collapses (test_backends measured 84s alone versus 2515s under that
# contention).  The env vars are what the spawned children read; the runtime
# call fixes this process, whose torch may already be imported.  Safe for the
# recorded values: the synthetic runtime is deliberately all-elementwise, so
# no reduction ordering — and therefore no bit — depends on thread count.
_THREADS = os.environ.setdefault("O1_B200_TEST_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)
if "torch" in sys.modules:
    sys.modules["torch"].set_num_threads(int(_THREADS))

SCRATCH = os.environ.get(
    "O1_B200_TEST_SCRATCH",
    os.path.join(_ROOT, "o1_b200", "reports", "local_runs", "test_scratch"))
os.makedirs(SCRATCH, exist_ok=True)

CORPUS_DIR = os.path.join(_ROOT, "o1_b200", "corpus")


def hermetic_mock_credentials() -> str:
    """Force synthetic provider credentials for local mock tests.

    Hard-sets RUNPOD_API_KEY to the mock server's synthetic key (never
    setdefault: an exported real operator key must not survive into a mock
    test process) and removes RUNPOD_API_KEY_FILE so no code path can read
    a real key from disk.  Returns the synthetic key so tests can pass it
    explicitly to adapters/transports.
    """
    from o1_b200.provider.runpod.mock_server import MOCK_API_KEY
    os.environ["RUNPOD_API_KEY"] = MOCK_API_KEY
    os.environ.pop("RUNPOD_API_KEY_FILE", None)
    return MOCK_API_KEY

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
