#!/usr/bin/env python3
"""Generate every pre-rental Foundation Learner artefact (contract §16).

One deterministic CLI.  Everything derives from the frozen root seed 20260809
(``--seed`` also accepts the predeclared second seed 20260810).  ``--tiny``
divides every pool by 100 for tests and the dress rehearsal.

Re-running is idempotent: the generation contains no wall-clock value and
writes a file only when its bytes would change, so a second run leaves the
output byte-identical and reports zero rewrites.  ``--verify-only`` re-hashes
an existing output root against its ``PREGEN_MANIFEST.json``.

Usage::

    python -m foundation_learner.scripts.pregenerate_all --tiny
    python -m foundation_learner.scripts.pregenerate_all --out artifacts_fl/pregen
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:  # allow direct execution from a checkout
    sys.path.insert(0, _PKG_PARENT)

from foundation_learner.data.generate_shards import (  # noqa: E402
    DEFAULT_TOKENIZER_PATH, PREGEN_MANIFEST_NAME, generate_all, verify_manifest)
from foundation_learner.data.pools import (  # noqa: E402
    DEFAULT_DIFFICULTY, FULL_POOLS, MAX_SEQ_LEN, ROOT_SEED, SECOND_SEED,
    tiny_pools)

DEFAULT_OUT = os.path.join(_PKG_PARENT, "artifacts_fl", "pregen")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic Foundation Learner data pre-generation")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"output root (default {DEFAULT_OUT})")
    parser.add_argument("--seed", type=int, default=ROOT_SEED,
                        choices=(ROOT_SEED, SECOND_SEED),
                        help="root seed; only the two predeclared seeds")
    parser.add_argument("--tiny", action="store_true",
                        help="pools divided by 100 (tests and rehearsal)")
    parser.add_argument("--difficulty", type=int, default=DEFAULT_DIFFICULTY,
                        help="frozen campaign difficulty level")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_PATH,
                        help="Ouro tokenizer directory (tokenizer files only)")
    parser.add_argument("--max-tokens", type=int, default=MAX_SEQ_LEN,
                        help="hard per-episode token budget")
    parser.add_argument("--verify-only", action="store_true",
                        help="re-hash an existing output root and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_root = os.path.abspath(args.out)

    if args.verify_only:
        report = verify_manifest(out_root)
        print(f"verified {report['checked']} files; "
              f"{'OK' if report['ok'] else 'MISMATCH'}")
        for path, reason in report["bad"]:
            print(f"  BAD {path}: {reason}")
        return 0 if report["ok"] else 1

    pools = tiny_pools() if args.tiny else FULL_POOLS
    started = time.monotonic()
    manifest = generate_all(
        out_root, root_seed=args.seed, pools=pools,
        difficulty=args.difficulty, tokenizer_path=args.tokenizer,
        max_tokens=args.max_tokens,
        log=None if args.quiet else (lambda m: print(m, flush=True)))
    elapsed = time.monotonic() - started

    files = manifest["files"]
    rewritten = len(manifest["_run"]["rewritten"])
    total_bytes = sum(f.get("bytes", 0) for f in files)
    episodes = sum(sum(v["episodes"].values())
                   for k, v in manifest["per_family"].items()
                   if "episodes" in v)
    print("-" * 70)
    print(f"root                {out_root}")
    print(f"mode                {'TINY' if args.tiny else 'FULL'} "
          f"(seed {manifest['root_seed']})")
    print(f"episodes            {episodes}")
    print(f"instances           {manifest['counts']['instances']}")
    print(f"latent rules        {manifest['counts']['rules']}")
    print(f"files               {len(files)} ({rewritten} rewritten)")
    print(f"bytes               {total_bytes}")
    print(f"max episode tokens  {manifest['token_budget']['max_tokens_seen']} "
          f"(limit {manifest['token_budget']['max_tokens_allowed']})")
    print(f"mean episode tokens {manifest['token_budget']['mean_tokens']:.1f}")
    print(f"wall time           {elapsed:.1f}s")
    print(f"manifest            {os.path.join(out_root, PREGEN_MANIFEST_NAME)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
