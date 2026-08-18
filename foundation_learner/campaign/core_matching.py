"""Production core-comparison audit (contract §8, §1, §15).

Three checks decide whether the FL1/FL2/FL3 comparison means anything:

1. ``training.arms.assert_core_arms_matched`` — the arm CONFIGURATIONS are
   compute-matched (same updates, same per-update token budget, same learning
   rate, same trainable-parameter mode, same seed);
2. ``training.compute_accounting.compare_core_ledgers`` — the REALIZED ledgers
   agree on the matched quantities, and the intrinsically differing ones (loss
   tokens above all) are reported rather than hidden;
3. :func:`assert_core_arms_share_a_base` — every arm's checkpoint manifest
   records the SAME base identity hash and the same frozen checkpoint tree
   hash, i.e. the arms really started from one identical frozen backbone.

All three existed before this module; none of them was invoked anywhere except
in the test suite (verification finding V-Gap4), so a real campaign could have
produced an unmatched comparison and nothing would have refused it.  The
``CORE_MATCHING`` stage calls :func:`build_core_matching_report` immediately
after the three core arms complete, and a violation raises — the scheduler
records the stage as FAILED with its predeclared fallback rather than letting
an unmatched comparison become the headline.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from . import o1_isolation

__all__ = [
    "CoreMatchingError",
    "CORE_MATCHING_SCHEMA",
    "assert_core_arms_share_a_base",
    "load_arm_checkpoint_manifests",
    "build_core_matching_report",
]

CORE_MATCHING_SCHEMA = "flb200.core_matching_report.v1"


class CoreMatchingError(RuntimeError):
    """The core arms are not matched, or did not share one frozen base."""


def assert_core_arms_share_a_base(manifests: Sequence[Mapping[str, Any]]) -> dict:
    """Refuse a comparison whose arms started from different checkpoints.

    Contract §1/§13: "every training arm starts from a fresh load of this
    identical frozen checkpoint".  The base identity hash covers the config
    digest and the recurrence pinning as well as the tree hash, so an arm that
    ran with, say, ``total_ut_steps`` unpinned is a DIFFERENT base and is
    caught here too.
    """
    if not manifests:
        raise CoreMatchingError(
            "no checkpoint manifests supplied to the base-identity check")
    identities = {}
    trees = {}
    scientific = {}
    for manifest in manifests:
        arm = str((manifest.get("arm_config") or {}).get("arm_id")
                  or manifest.get("arm_id") or "?")
        identities[arm] = manifest.get("base_identity_hash")
        trees[arm] = manifest.get("base_checkpoint_tree_sha256")
        scientific[arm] = bool((manifest.get("base_identity") or {}).get(
            "scientific", False))
    if None in identities.values():
        raise CoreMatchingError(
            "REFUSED: a core-arm checkpoint records no base identity hash; the "
            f"arms cannot be shown to share a base (identities {identities})")
    if len(set(identities.values())) != 1 or len(set(trees.values())) != 1:
        raise CoreMatchingError(
            "REFUSED: core arms did not start from one identical frozen "
            f"checkpoint; identity hashes {identities}, tree hashes {trees}")
    tree = next(iter(trees.values()))
    if tree is None and any(scientific.values()):
        # A SCIENTIFIC arm must be able to name the frozen checkpoint it came
        # from; only the tiny NONSCIENTIFIC mechanics model has no tree hash.
        raise CoreMatchingError(
            "REFUSED: the core arms record no base checkpoint tree hash but "
            f"claim to be scientific ({scientific}); contract §1/§15 bind every "
            "scientific arm to the frozen checkpoint tree hash")
    record = {"base_identity_hash": next(iter(identities.values())),
              "base_checkpoint_tree_sha256": tree,
              "scientific": scientific,
              "arms": sorted(identities)}
    if tree is None:
        record["note"] = (
            "no checkpoint tree hash: every arm ran on a NONSCIENTIFIC model "
            "(a dress rehearsal). The arms still shared ONE identical base "
            "identity, which is what this check certifies.")
    return record


def load_arm_checkpoint_manifests(ctx: Any, arm_ids: Sequence[str], *,
                                  tag: str = "final") -> dict[str, dict]:
    """Read each core arm's checkpoint manifest through the isolation guard."""
    out: dict[str, dict] = {}
    for arm_id in arm_ids:
        out_dir = (ctx.results.get("_arm_out_dirs", {}).get(arm_id)
                   or os.path.join(ctx.out_dir, arm_id.lower(), "arm"))
        path = ctx.guard.guard(os.path.join(out_dir, f"ckpt_{tag}.manifest.json"),
                               o1_isolation.MODE_READ)
        if not os.path.isfile(path):
            raise CoreMatchingError(
                f"{arm_id}: no {tag!r} checkpoint manifest at {path!r}; the "
                "core comparison cannot be audited without it")
        out[arm_id] = ctx.guard.read_json(path)
    return out


def build_core_matching_report(ctx: Any, arm_ids: Sequence[str], *,
                               tag: str = "final") -> dict:
    """Run all three checks and return the ``flb200.core_matching_report.v1``."""
    from ..training.arms import assert_core_arms_matched
    from ..training.compute_accounting import compare_core_ledgers

    arm_ids = [str(a) for a in arm_ids]
    missing = [a for a in arm_ids if a not in ctx.results]
    if missing:
        raise CoreMatchingError(
            f"the core comparison is incomplete: {missing} produced no result; "
            "an unmatched or partial comparison is never presented as the "
            "headline (contract §8)")
    configs = ctx.results.get("_arm_configs", {})
    absent = [a for a in arm_ids if a not in configs]
    if absent:
        raise CoreMatchingError(
            f"no in-process arm configuration recorded for {absent}; the "
            "matching check consumes the real ArmConfig objects the arms ran "
            "with, never a reconstruction")
    assert_core_arms_matched([configs[a] for a in arm_ids])

    ledgers = []
    for arm_id in arm_ids:
        ledger = (ctx.results[arm_id].get("arm_result") or {}).get("ledger")
        if not ledger:
            raise CoreMatchingError(
                f"{arm_id}: no compute ledger in its arm result; contract §8 "
                "requires the realized compute of every core arm")
        ledgers.append(ledger)
    comparison = compare_core_ledgers(ledgers)

    manifests = load_arm_checkpoint_manifests(ctx, arm_ids, tag=tag)
    base = assert_core_arms_share_a_base(list(manifests.values()))

    return {
        "schema": CORE_MATCHING_SCHEMA,
        "arms": arm_ids,
        "checkpoint_tag": tag,
        "arm_config_hashes": {
            a: ctx.results[a].get("arm_result", {}).get("arm_config_hash")
            for a in arm_ids},
        "compute_comparison": comparison,
        "base_identity": base,
        "checkpoint_manifest_hashes": {
            a: m.get("manifest_hash") for a, m in sorted(manifests.items())},
        "checks": [
            "training.arms.assert_core_arms_matched (configurations)",
            "training.compute_accounting.compare_core_ledgers (realized compute)",
            "campaign.core_matching.assert_core_arms_share_a_base (base identity)",
        ],
        "policy": ("loss-token counts differ across objectives by construction "
                   "and are REPORTED, never hidden or equalised (contract §8, "
                   "option B)"),
    }
