"""Development-grid selection and the freezing of the development decisions.

Contract §8/§9: the grid is exactly TWO learning rates for the selected
trainable-parameter scope ({1e-4, 3e-4} for ``PEFT_MODE``, {1e-5, 3e-5} for
``FULL_MODEL_MODE``), run on FL3 ONLY, at the frozen shortened step count of
25 % of the core U.  Selection is ``max DEV macro-AULC``; ties break to the
LOWER learning rate.  The winner is then used for FL1, FL2 and FL3 with the
same scope, and no new candidate may be proposed after dev performance has
been seen.

Two refusals are load-bearing here:

* every episode this module reads passes through
  :func:`campaign.sealed_gate.loader_guard`, so a SEALED_TEST shard offered to
  the selector is REFUSED rather than read (contract §12a, hostile-tested);
* :class:`~campaign.promotion.DevMetrics` cannot be constructed from sealed
  records, so a selection cannot be computed from them either.

The output, ``DEV_DECISIONS_FROZEN.json``, is the prerequisite the sealed gate
checks before it will derive ``K_seal`` at all.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import o1_isolation, sealed_gate
from .affordability import SCOPES, grid_updates
from .promotion import DevMetrics

__all__ = [
    "DevSelectorError",
    "GRID_SIZE",
    "GRID_ARM",
    "frozen_grid",
    "expected_grid_updates",
    "GridRun",
    "SelectionResult",
    "assert_grid_is_frozen",
    "select_learning_rate",
    "load_dev_episodes",
    "freeze_dev_decisions",
]

GRID_SIZE = 2
GRID_ARM = "FL3"


class DevSelectorError(RuntimeError):
    """The development grid or its selection violated the frozen design."""


def frozen_grid(scope: str) -> tuple[float, ...]:
    """The frozen two-learning-rate grid for a scope (from ``training.arms``)."""
    from ..training.arms import LEARNING_RATE_GRID

    if scope not in SCOPES:
        raise DevSelectorError(f"unknown scope {scope!r}; expected one of {SCOPES}")
    grid = tuple(sorted(float(lr) for lr in LEARNING_RATE_GRID[scope]))
    if len(grid) != GRID_SIZE:
        raise DevSelectorError(
            f"the frozen grid for {scope} has {len(grid)} learning rates; "
            f"contract §8 freezes exactly {GRID_SIZE}")
    return grid


def expected_grid_updates(updates_U: int) -> int:
    """25 % of the core U (contract §8), shared with the affordability math."""
    return grid_updates(int(updates_U))


@dataclass(frozen=True)
class GridRun:
    """One completed grid configuration and its DEVELOPMENT evidence."""

    learning_rate: float
    scope: str
    updates: int
    dev: DevMetrics
    arm_id: str = GRID_ARM
    arm_config_hash: str = ""
    checkpoint_tags: tuple[str, ...] = ()
    records_sha256: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise DevSelectorError(f"unknown scope {self.scope!r}")
        if self.dev.split != "DEVELOPMENT":  # pragma: no cover - DevMetrics guards
            raise DevSelectorError("grid evidence must be DEVELOPMENT")
        if self.dev.macro_aulc is None:
            raise DevSelectorError(
                f"grid run lr={self.learning_rate} has no DEV macro-AULC; the "
                "selection rule cannot be applied to it")

    def to_dict(self) -> dict:
        return {
            "learning_rate": float(self.learning_rate),
            "scope": self.scope,
            "updates": int(self.updates),
            "arm_id": self.arm_id,
            "arm_config_hash": self.arm_config_hash,
            "checkpoint_tags": list(self.checkpoint_tags),
            "records_sha256": self.records_sha256,
            "dev": self.dev.to_dict(),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class SelectionResult:
    chosen_learning_rate: float
    scope: str
    updates_grid: int
    ranking: tuple[dict, ...]
    tie_broken: bool
    rule: str = ("max DEV macro-AULC over the frozen FL3 25 %-step grid; "
                 "tie -> lower learning rate")

    def to_dict(self) -> dict:
        return {
            "schema": "flb200.dev_selection.v1",
            "chosen_learning_rate": float(self.chosen_learning_rate),
            "scope": self.scope,
            "updates_grid": int(self.updates_grid),
            "ranking": [dict(r) for r in self.ranking],
            "tie_broken": bool(self.tie_broken),
            "rule": self.rule,
        }


def assert_grid_is_frozen(runs: Sequence[GridRun], *, scope: str,
                          updates_U: int | None = None) -> None:
    """Refuse anything but the frozen grid: 2 LRs, FL3, one scope, 25 % U."""
    if len(runs) != GRID_SIZE:
        raise DevSelectorError(
            f"the development grid must contain exactly {GRID_SIZE} runs "
            f"(contract §8), got {len(runs)}")
    scopes = {r.scope for r in runs}
    if scopes != {scope}:
        raise DevSelectorError(
            f"grid runs mix scopes {sorted(scopes)}; the grid runs in the "
            "SELECTED scope only")
    arms = {r.arm_id for r in runs}
    if arms != {GRID_ARM}:
        raise DevSelectorError(
            f"the grid runs on {GRID_ARM} only (contract §8), got {sorted(arms)}")
    lrs = tuple(sorted(float(r.learning_rate) for r in runs))
    expected = frozen_grid(scope)
    if lrs != expected:
        raise DevSelectorError(
            f"grid learning rates {lrs} are not the frozen grid {expected}; "
            "hyperparameter invention outside the frozen grid is forbidden "
            "(contract §8/§20)")
    if updates_U is not None:
        want = expected_grid_updates(updates_U)
        wrong = [r.updates for r in runs if int(r.updates) != want]
        if wrong:
            raise DevSelectorError(
                f"grid runs must use the frozen shortened step count {want} "
                f"(25 % of U={updates_U}); found {wrong}")


def select_learning_rate(runs: Sequence[GridRun], *, scope: str | None = None,
                         updates_U: int | None = None) -> SelectionResult:
    """Apply the frozen selection rule to a complete grid."""
    runs = list(runs)
    if not runs:
        raise DevSelectorError("no grid runs supplied")
    scope = scope or runs[0].scope
    assert_grid_is_frozen(runs, scope=scope, updates_U=updates_U)
    ranked = sorted(
        runs, key=lambda r: (-float(r.dev.macro_aulc), float(r.learning_rate)))
    best = ranked[0]
    top_score = float(best.dev.macro_aulc)
    tied = [r for r in ranked if float(r.dev.macro_aulc) == top_score]
    ranking = tuple({
        "learning_rate": float(r.learning_rate),
        "macro_aulc": float(r.dev.macro_aulc),
        "slope": r.dev.slope,
        "n_episodes": int(r.dev.n_episodes),
        "families": list(r.dev.family_ids),
        "arm_config_hash": r.arm_config_hash,
    } for r in ranked)
    return SelectionResult(
        chosen_learning_rate=float(best.learning_rate),
        scope=scope,
        updates_grid=int(best.updates),
        ranking=ranking,
        tie_broken=len(tied) > 1,
    )


def load_dev_episodes(shard_paths: Sequence[str], *,
                      guard: o1_isolation.IsolationGuard | None = None
                      ) -> list[dict]:
    """Read DEVELOPMENT shards through the loader guard.

    A SEALED_TEST path is refused here — the selector holds no unlock and never
    will (contract §12a).  The isolation guard runs on the same call.
    """
    from ..data.shards import read_shard

    guard = guard or o1_isolation.default_guard()
    records: list[dict] = []
    for path in shard_paths:
        resolved = sealed_gate.loader_guard(path, unlock=None, guard=guard,
                                            context="dev_selector")
        records.extend(read_shard(resolved))
    return records


def freeze_dev_decisions(out_path: str, *, selection: SelectionResult,
                         updates_U: int | None,
                         stage_states: Mapping[str, Any],
                         checkpoint_tags: Mapping[str, Any],
                         hashes: Mapping[str, Any],
                         runs: Sequence[GridRun] = (),
                         notes: Sequence[str] = (),
                         guard: o1_isolation.IsolationGuard | None = None) -> dict:
    """Write ``DEV_DECISIONS_FROZEN.json`` — the sealed gate's prerequisite.

    Refuses to overwrite an existing record: the development decisions are
    frozen once, and the sealed opening ledger binds their hash.
    """
    guard = guard or o1_isolation.default_guard()
    resolved = guard.guard(out_path, o1_isolation.MODE_WRITE)
    if os.path.exists(resolved):
        raise DevSelectorError(
            f"{out_path!r} already exists; the development decisions are FROZEN "
            "once (the sealed opening ledger binds their hash)")
    payload = sealed_gate.build_dev_decisions(
        chosen_learning_rate=selection.chosen_learning_rate,
        chosen_scope=selection.scope,
        updates_U=updates_U,
        stage_states=stage_states,
        checkpoint_tags=checkpoint_tags,
        hashes=hashes,
        grid=[r.to_dict() for r in runs],
        selection_rule=selection.rule,
        notes=list(notes) + [
            "SEALED_TEST was not read by the selector: every shard passed "
            "through campaign.sealed_gate.loader_guard without an unlock.",
        ],
    )
    payload["selection"] = selection.to_dict()
    # the digest covers the whole document, so it is recomputed after the
    # selection block is attached
    payload.pop("dev_decisions_sha256")
    from ..ecology.base import domain_sha256

    payload["dev_decisions_sha256"] = domain_sha256("FL_V0_DEV_DECISIONS", payload)
    sealed_gate.validate_dev_decisions(payload, source=out_path)
    guard.write_json(resolved, payload)
    return payload
