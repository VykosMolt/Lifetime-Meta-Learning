#!/usr/bin/env python3
"""Run the Foundation Learner test suite file-by-file across processes.

The suite has no conftest, no session fixtures and no shared on-disk state,
so per-file pytest processes are independent.  Wall time becomes the slowest
file rather than the sum of all of them.

    python -m foundation_learner.tests.run_parallel [--jobs N] [-k EXPR]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

_COUNTS = re.compile(r"(\d+) (passed|failed|skipped|error[s]?|xfailed|xpassed)")


def test_files() -> list[str]:
    out = []
    for base, _dirs, names in os.walk(_HERE):
        if "__pycache__" in base:
            continue
        for n in sorted(names):
            if n.startswith("test_") and n.endswith(".py"):
                out.append(os.path.relpath(os.path.join(base, n), _ROOT))
    return sorted(out)


def run_file(path: str, extra: list[str], env: dict) -> dict:
    t = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         path, *extra],
        capture_output=True, text=True, cwd=_ROOT, env=env)
    text = proc.stdout + proc.stderr
    counts: dict[str, int] = {}
    for line in text.splitlines()[::-1]:
        found = _COUNTS.findall(line)
        if found:
            for num, kind in found:
                if kind.startswith("error"):
                    kind = "errors"
                counts[kind] = int(num)
            break
    return {"file": path, "rc": proc.returncode, "counts": counts,
            "seconds": time.monotonic() - t, "output": text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("extra", nargs="*", help="extra pytest args (e.g. -k EXPR)")
    args = ap.parse_args()

    files = test_files()
    jobs = args.jobs or min(len(files), (os.cpu_count() or 1))
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda f: run_file(f, args.extra, env), files))

    totals: dict[str, int] = {}
    failed = []
    for res in sorted(results, key=lambda r: -r["seconds"]):
        for kind, num in res["counts"].items():
            totals[kind] = totals.get(kind, 0) + num
        flag = "" if res["rc"] == 0 else "  <-- FAILED"
        summary = " ".join(f"{n} {k}" for k, n in sorted(res["counts"].items()))
        print(f"{res['seconds']:7.1f}s  {res['file']:<62} {summary}{flag}")
        if res["rc"] != 0:
            failed.append(res)
    print("=" * 100)
    for res in failed:
        print(f"--- {res['file']} ---")
        print(res["output"][-4000:])
    line = ", ".join(f"{n} {k}" for k, n in sorted(totals.items()))
    print(f"{line}  in {time.monotonic() - t0:.1f}s wall ({jobs} jobs, "
          f"{len(files)} files)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
