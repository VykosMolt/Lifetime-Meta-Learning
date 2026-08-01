#!/usr/bin/env python3
"""LOCAL_NONSCIENTIFIC_SMOKE_TEST — optional real-checkpoint smoke.

Permitted scope (all enforced here): uses ONLY non-O1 validation-corpus
tasks; changes no environment; installs nothing; targets <= 15 minutes;
produces NO scientific result.  It exists to verify that the batched engine
speaks the real Ouro-RLTT model's API (current_ut hook kwargs,
per_loop_hidden_states, cache reuse, growing attention masks) on the LOCAL
GPU, and to OBSERVE (not certify) mixed-length left-padded batching on the
real architecture.  Any divergence found here is a finding for the future
B200 equivalence session, not a scientific verdict.
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import numpy as np

from . import sealed_import
from .engine_batched import generate_batch
from .runbuild import build_validation_bundle
from .validation_corpus import load_corpus

sealed_import.ensure_sealed_path()

CHECKPOINT = "/home/moloch/ouro_project/models/ouro_rltt_local"
TIME_BUDGET_SECONDS = 15 * 60
SMOKE_MAX_NEW_TOKENS = 48  # engine cap for the smoke; sealed constant untouched


def main() -> int:
    t0 = time.monotonic()

    def left(label):
        used = time.monotonic() - t0
        print(f"[{used:6.1f}s] {label}")
        if used > TIME_BUDGET_SECONDS:
            raise SystemExit("time budget exceeded; aborting smoke test")

    from run_o1_v2_generation import _load_model, generate_branch
    tasks, _ = load_corpus(os.path.join(
        sealed_import.WORKTREE_ROOT, "o1_b200", "corpus"))
    by_id = {t["task_id"]: t for t in tasks}
    short_task = by_id["b200val-000-commit_a"]
    long_task = by_id["b200val-003-commit_c_period"]  # long prompt variant
    for t in (short_task, long_task):
        assert t["task_id"].startswith("b200val-"), "non-corpus task refused"

    left("loading real Ouro-RLTT checkpoint (local, offline)")
    model, tok = _load_model(pathlib.Path(CHECKPOINT))
    left("model loaded")

    report = {"label": "LOCAL_NONSCIENTIFIC_SMOKE_TEST",
              "scientific_result": None, "checks": {}}

    # 1) engine(B=1) vs sealed generate_branch — same task, same seed.
    #    The sealed path has no token cap parameter, so run the engine at the
    #    sealed MAX and compare prefixes up to the smoke horizon.
    left("engine B=1 vs sealed serial (baseline row)")
    sealed = generate_branch(model, tok, short_task, 4242, None, 0.0)
    eng = generate_batch(model, tok, [short_task], [4242], [None], [0.0])[0]
    same_tokens = sealed["generated_token_ids"] == eng["generated_token_ids"]
    same_boundary = np.array_equal(
        np.asarray(sealed["_prefill_l4_47"]).reshape(-1),
        np.asarray(eng["_prefill_l4_47"]).reshape(-1))
    report["checks"]["engine_b1_matches_sealed_tokens"] = bool(same_tokens)
    report["checks"]["engine_b1_matches_sealed_boundary"] = bool(same_boundary)
    left(f"  tokens identical: {same_tokens}; boundary bitwise: {same_boundary}")

    # 2) equal-length batch (same prompt twice, different seeds): pure batch
    #    machinery, no padding involved.
    left("equal-length batch B=2 (no padding)")
    solo = [generate_batch(model, tok, [short_task], [s], [None], [0.0],
                           max_new_tokens=SMOKE_MAX_NEW_TOKENS)[0]
            for s in (7, 8)]
    duo = generate_batch(model, tok, [short_task, short_task], [7, 8],
                         [None, None], [0.0, 0.0],
                         max_new_tokens=SMOKE_MAX_NEW_TOKENS)
    eq = [solo[i]["generated_token_ids"] == duo[i]["generated_token_ids"]
          for i in range(2)]
    report["checks"]["equal_length_batch_token_identity"] = eq
    left(f"  per-row token identity vs solo: {eq}")

    # 3) mixed-length left-padded batch: OBSERVE real-model behavior.
    left("mixed-length left-padded batch B=2 (observational)")
    solo_long = generate_batch(model, tok, [long_task], [9], [None], [0.0],
                               max_new_tokens=SMOKE_MAX_NEW_TOKENS)[0]
    mixed = generate_batch(model, tok, [short_task, long_task], [7, 9],
                           [None, None], [0.0, 0.0],
                           max_new_tokens=SMOKE_MAX_NEW_TOKENS)
    obs = {
        "short_row_tokens_match_solo":
            solo[0]["generated_token_ids"] == mixed[0]["generated_token_ids"],
        "long_row_tokens_match_solo":
            solo_long["generated_token_ids"] == mixed[1]["generated_token_ids"],
        "long_row_boundary_bitwise": bool(np.array_equal(
            np.asarray(solo_long["_prefill_l4_47"]).reshape(-1),
            np.asarray(mixed[1]["_prefill_l4_47"]).reshape(-1))),
    }
    report["checks"]["mixed_length_observations"] = obs
    left(f"  observations: {obs}")

    # 4) intervention row in a batch: hook fires once, per-row delta recorded.
    left("batched intervention row (structured axis, alpha 0.04)")
    from .runbuild import load_frozen_axes
    axes = load_frozen_axes()
    duo_int = generate_batch(
        model, tok, [short_task, short_task], [7, 8],
        [np.asarray(axes["structured"][0], dtype=np.float64), None],
        [0.04, 0.0], max_new_tokens=SMOKE_MAX_NEW_TOKENS)
    report["checks"]["batched_intervention"] = {
        "intervened_row_injected_rms_positive": duo_int[0]["injected_rms"] > 0,
        "coresident_baseline_untouched":
            duo_int[1]["generated_token_ids"] == duo[1]["generated_token_ids"],
    }
    left(f"  {report['checks']['batched_intervention']}")

    report["elapsed_seconds"] = time.monotonic() - t0
    out = os.path.join(sealed_import.WORKTREE_ROOT, "o1_b200", "reports",
                       "LOCAL_NONSCIENTIFIC_SMOKE_TEST.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    print(f"LOCAL_NONSCIENTIFIC_SMOKE_TEST complete in "
          f"{report['elapsed_seconds']:.0f}s; report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
