"""Frozen deterministic confirmatory-cohort allocation for O1 v2.1.

This module is the executable form of the preregistered allocation rule.  It
is pure stdlib and pure-functional: identical inputs always produce identical
allocations, leaving zero operator discretion after calibration.

ALLOCATION RULE (frozen before calibration):

 1. Settings are assigned to the hard-medium and medium strata by calibration
    only (half-open baseline any-correct-of-8 bands).
 2. The precommitted candidate pool is filtered to those settings; whole
    settings are eligible, never individual tasks.
 3. Desired G starts from G_power, then the wall-clock G_max cap, then the
    eligible-pool capacity cap (G_confirm = min of the three, applied
    upstream; this module receives the final G_confirm).
 4. Stratum split targets 50/50; an odd remainder goes to hard_medium (the
    lexicographically first stratum name).
 5. If one stratum's eligible supply is below its target, it takes its whole
    supply and the shortfall moves to the other stratum (bounded by supply;
    G_confirm <= total supply is guaranteed upstream).
 6. Within each stratum, tasks are distributed over eligible settings by
    single-unit round robin over the lexicographically sorted setting IDs,
    skipping exhausted settings — "as even as possible, integer remainders to
    lexicographically first settings".
 7. Within each setting, only the hash-ordered prefix is taken: sealed ranks
    0..quota-1 in sealed_rank_within_generator_setting order.
 8. sealed_rank_within_stratum is assigned by enumerating each stratum's
    selected tasks ordered by (setting_id lexicographic, sealed rank within
    setting), starting at 0.
"""
from __future__ import annotations

ALLOCATION_RULE_TEXT = (
    "50/50 stratum target, odd remainder to hard_medium; undersupplied stratum "
    "takes its whole eligible supply and the shortfall moves to the other "
    "stratum; within a stratum, single-unit round robin over lexicographically "
    "sorted eligible settings with remaining capacity; within a setting, "
    "hash-ordered prefix by sealed_rank_within_generator_setting; stratum "
    "ranks enumerate (setting lex, setting rank) order from 0."
)

STRATA = ("hard_medium", "medium")


class AllocationError(RuntimeError):
    """The frozen allocation rule cannot be satisfied."""


def allocate(selected_settings: dict[str, list[str]],
             pool_counts: dict[str, int], G_confirm: int) -> dict:
    """Return the deterministic allocation for ``G_confirm`` tasks.

    ``selected_settings``: stratum -> list of eligible setting IDs (from
    calibration banding).  ``pool_counts``: setting ID -> candidate-pool count
    (eligible settings only).  ``G_confirm``: the final capped cohort size.
    """
    if isinstance(G_confirm, bool) or not isinstance(G_confirm, int) or G_confirm < 1:
        raise AllocationError(f"G_confirm must be a positive int, got {G_confirm!r}")
    by_stratum: dict[str, list[str]] = {}
    seen: set[str] = set()
    for st in STRATA:
        settings = sorted(selected_settings.get(st, []))
        if not settings:
            raise AllocationError(f"stratum {st!r} has no eligible settings")
        for s in settings:
            if s in seen:
                raise AllocationError(f"setting {s!r} appears in both strata")
            seen.add(s)
            if pool_counts.get(s, 0) < 0:
                raise AllocationError(f"negative pool count for {s!r}")
        by_stratum[st] = settings
    supply = {st: sum(int(pool_counts.get(s, 0)) for s in by_stratum[st])
              for st in STRATA}
    total_supply = sum(supply.values())
    if G_confirm > total_supply:
        raise AllocationError(
            f"G_confirm {G_confirm} exceeds eligible pool supply {total_supply}; "
            f"the upstream pool cap was not applied")

    # Steps 4-5: stratum targets with deterministic waterfall.
    targets = {"hard_medium": G_confirm - G_confirm // 2,
               "medium": G_confirm // 2}
    for st in STRATA:
        if targets[st] > supply[st]:
            shortfall = targets[st] - supply[st]
            targets[st] = supply[st]
            other = "medium" if st == "hard_medium" else "hard_medium"
            targets[other] += shortfall
    for st in STRATA:
        if targets[st] > supply[st]:
            raise AllocationError(
                f"stratum {st!r} target {targets[st]} exceeds supply "
                f"{supply[st]} after waterfall")

    # Step 6: single-unit round robin within each stratum.
    quota: dict[str, int] = {}
    for st in STRATA:
        remaining = targets[st]
        caps = {s: int(pool_counts.get(s, 0)) for s in by_stratum[st]}
        q = {s: 0 for s in by_stratum[st]}
        while remaining > 0:
            progressed = False
            for s in by_stratum[st]:
                if remaining == 0:
                    break
                if q[s] < caps[s]:
                    q[s] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise AllocationError(
                    f"stratum {st!r}: round robin stalled with {remaining} "
                    f"unassigned")
        quota.update(q)

    return {
        "expected_counts": {st: int(targets[st]) for st in STRATA},
        "per_setting_quota": {s: int(quota[s]) for s in sorted(quota)},
        "settings_per_stratum": {st: list(by_stratum[st]) for st in STRATA},
        "G_confirm": int(G_confirm),
    }


def materialize(pool: list[dict], allocation: dict) -> list[dict]:
    """Materialize the confirmatory task rows from the sealed pool.

    Steps 7-8: hash-ordered prefixes per setting; stratum ranks enumerate
    (setting lex, sealed setting rank) order.  Rows keep the full task content
    (options, symbolic, prompts) so gate G22 can recompute scoring, and gain
    ``stratum`` and ``sealed_rank_within_stratum``.
    """
    by_setting: dict[str, list[dict]] = {}
    for row in pool:
        by_setting.setdefault(row["generator_setting_id"], []).append(row)
    for s, rows in by_setting.items():
        rows.sort(key=lambda r: int(r["sealed_rank_within_generator_setting"]))
        ranks = [int(r["sealed_rank_within_generator_setting"]) for r in rows]
        if ranks != list(range(len(rows))):
            raise AllocationError(
                f"setting {s!r}: sealed ranks are not a 0..n-1 permutation")
    out: list[dict] = []
    for st in STRATA:
        rank = 0
        for s in allocation["settings_per_stratum"][st]:
            q = int(allocation["per_setting_quota"].get(s, 0))
            rows = by_setting.get(s, [])
            if q > len(rows):
                raise AllocationError(
                    f"setting {s!r}: quota {q} exceeds pool rows {len(rows)}")
            for src in rows[:q]:
                row = dict(src)
                row["stratum"] = st
                row["sealed_rank_within_stratum"] = rank
                rank += 1
                out.append(row)
        if rank != int(allocation["expected_counts"][st]):
            raise AllocationError(
                f"stratum {st!r}: materialized {rank} rows, expected "
                f"{allocation['expected_counts'][st]}")
    return out
