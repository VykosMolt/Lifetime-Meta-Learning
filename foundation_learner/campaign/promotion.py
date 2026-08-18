"""Promotion ladder (contract §10) — SEALED_TEST is never consulted.

The rules are exactly §10:

* FL3 -> extensions when training was stable AND
  ``DEV macro-AULC(FL3) >= DEV macro-AULC(FL1) + 0.02`` AND the DEV macro
  improvement slope is positive (not merely an ``R_0`` gain);
* FL4 when FL3 finished and >= 200 realized episodes carry >= 2 scoreable
  feedback items each;
* FL5 whenever the core comparison is complete and the scheduler admits it;
* FL6 requires FL4 DEV pairwise ranking accuracy >= 0.55;
* FL7 requires scheduler admission AND a stable fast-update implementation
  (the local dress-rehearsal flag); it is independently predeclared and may run
  even when FL5 is null, but is skipped when its only remaining variant is the
  FL6-dependent gated one and FL4 failed;
* FL8 requires context-reset persistence > 0 on DEV with a CI excluding 0.

**Structural sealed exclusion.**  Every promotion function consumes
:class:`DevMetrics`, whose constructor refuses any split tag other than
``DEVELOPMENT`` and any family that the frozen split assigns to
``SEALED_TEST``.  A sealed record therefore cannot be carried into a promotion
decision even by a caller that wants to: the object cannot be built.  The same
guard is applied to the FL8 persistence evidence and to the FL4 value-head
metrics.

Fallbacks are the frozen §10 predeclared work: no objective is ever invented
live, and all remaining time goes to predeclared work only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..ecology.split import compute_split

__all__ = [
    "PromotionError",
    "SealedEvidenceRefused",
    "DEV_SPLIT",
    "TRAIN_SPLIT",
    "SEALED_SPLIT",
    "FL3_AULC_MARGIN",
    "FL4_MIN_EPISODES",
    "FL4_MIN_SCOREABLE_ITEMS",
    "FL6_MIN_PAIRWISE_ACCURACY",
    "DevMetrics",
    "PromotionDecision",
    "assert_no_sealed",
    "promote_fl3_to_extensions",
    "fl4_data_sufficiency",
    "promote_fl4",
    "promote_fl5",
    "promote_fl6",
    "promote_fl7",
    "promote_fl8",
    "select_best_dev_checkpoint",
    "FALLBACKS",
    "fallback_for",
]

DEV_SPLIT = "DEVELOPMENT"
TRAIN_SPLIT = "TRAIN"
SEALED_SPLIT = "SEALED_TEST"

#: Contract §10 frozen thresholds.
FL3_AULC_MARGIN = 0.02
FL4_MIN_EPISODES = 200
FL4_MIN_SCOREABLE_ITEMS = 2
FL6_MIN_PAIRWISE_ACCURACY = 0.55


class PromotionError(RuntimeError):
    """Malformed promotion evidence (never silently tolerated)."""


class SealedEvidenceRefused(PromotionError):
    """Sealed-test material was offered to a promotion decision."""


def _sealed_families() -> frozenset[str]:
    return frozenset(compute_split()[SEALED_SPLIT])


def assert_no_sealed(records: Iterable[Mapping[str, Any]], *,
                     context: str = "promotion evidence") -> None:
    """Refuse any record tagged with a sealed split or a sealed family."""
    sealed = _sealed_families()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PromotionError(
                f"{context}: record {index} is {type(record).__name__}, not a mapping")
        split = record.get("split")
        if split is None:
            raise PromotionError(
                f"{context}: record {index} carries no split tag; untagged "
                "records are refused because their provenance cannot be checked")
        if str(split) != DEV_SPLIT:
            raise SealedEvidenceRefused(
                f"{context}: record {index} has split {split!r}; promotion "
                f"consumes {DEV_SPLIT} records only (contract §10: SEALED_TEST "
                "is NEVER used for promotion)")
        family = record.get("family_id")
        if family is not None and str(family) in sealed:
            raise SealedEvidenceRefused(
                f"{context}: record {index} belongs to sealed family "
                f"{family!r}; promotion from sealed outcomes is forbidden")


@dataclass(frozen=True)
class DevMetrics:
    """DEVELOPMENT-split metrics for one stage.

    The constructor is the structural guard: a sealed split tag or a sealed
    family id makes the object unconstructable, so no promotion rule can be
    reached with sealed evidence.
    """

    stage: str
    macro_aulc: float | None
    slope: float | None = None
    r_0: float | None = None
    n_episodes: int = 0
    family_ids: tuple[str, ...] = ()
    split: str = DEV_SPLIT
    stability_events: tuple[str, ...] = ()
    source: str = "unspecified"
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split != DEV_SPLIT:
            raise SealedEvidenceRefused(
                f"DevMetrics for stage {self.stage!r} was given split "
                f"{self.split!r}; only {DEV_SPLIT} metrics may enter a "
                "promotion decision (contract §10)")
        sealed = _sealed_families()
        offending = sorted(set(self.family_ids) & sealed)
        if offending:
            raise SealedEvidenceRefused(
                f"DevMetrics for stage {self.stage!r} contains sealed families "
                f"{offending}; promotion from sealed outcomes is forbidden")
        for name in ("macro_aulc", "slope", "r_0"):
            value = getattr(self, name)
            if value is not None:
                float(value)  # raises TypeError/ValueError on non-numeric input

    @property
    def stable(self) -> bool:
        return not self.stability_events

    @staticmethod
    def from_records(records: Sequence[Mapping[str, Any]], *, stage: str,
                     stability_events: Sequence[str] = (),
                     source: str = "episode_records") -> "DevMetrics":
        """Build from W4 episode records, refusing anything non-DEVELOPMENT."""
        records = list(records)
        assert_no_sealed(records, context=f"DevMetrics({stage})")
        from ..evaluation import metrics as _metrics

        families = tuple(sorted({str(r.get("family_id")) for r in records
                                 if r.get("family_id") is not None}))
        return DevMetrics(
            stage=str(stage),
            macro_aulc=_metrics.macro_aulc(records),
            slope=_metrics.improvement_slope(records),
            r_0=_metrics.r_0(records),
            n_episodes=len(records),
            family_ids=families,
            stability_events=tuple(str(e) for e in stability_events),
            source=source,
        )

    def to_dict(self) -> dict:
        return {
            "schema": "flb200.dev_metrics.v1",
            "stage": self.stage,
            "split": self.split,
            "macro_aulc": self.macro_aulc,
            "slope": self.slope,
            "r_0": self.r_0,
            "n_episodes": int(self.n_episodes),
            "family_ids": list(self.family_ids),
            "stability_events": list(self.stability_events),
            "source": self.source,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class PromotionDecision:
    stage: str
    admitted: bool
    reasons: tuple[str, ...]
    evidence: dict = field(default_factory=dict)
    fallback: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "schema": "flb200.promotion_decision.v1",
            "stage": self.stage,
            "admitted": bool(self.admitted),
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
            "fallback": list(self.fallback),
        }


def _decide(stage: str, checks: Sequence[tuple[str, bool]], evidence: dict
            ) -> PromotionDecision:
    failed = [label for label, ok in checks if not ok]
    admitted = not failed
    reasons = tuple(
        f"{'PASS' if ok else 'FAIL'}: {label}" for label, ok in checks)
    return PromotionDecision(
        stage=stage, admitted=admitted, reasons=reasons, evidence=evidence,
        fallback=() if admitted else tuple(fallback_for(stage)))


# --------------------------------------------------------------------------
# the frozen rules
# --------------------------------------------------------------------------

def promote_fl3_to_extensions(fl3: DevMetrics, fl1: DevMetrics, *,
                              stability_ok: bool | None = None,
                              margin: float = FL3_AULC_MARGIN
                              ) -> PromotionDecision:
    """FL3 -> extensions gate (contract §10)."""
    if fl3.stage.upper().startswith("FL1") or fl1.stage.upper().startswith("FL3"):
        raise PromotionError(
            "promote_fl3_to_extensions(fl3, fl1) received the arms in the "
            f"wrong order: fl3={fl3.stage!r}, fl1={fl1.stage!r}")
    if fl3.macro_aulc is None or fl1.macro_aulc is None:
        raise PromotionError(
            "FL3 promotion needs DEV macro-AULC for both FL3 and FL1")
    if fl3.slope is None:
        raise PromotionError(
            "FL3 promotion needs DEV macro improvement-slope evidence; an "
            "R_0 gain alone can never satisfy the gate")
    stable = fl3.stable if stability_ok is None else bool(stability_ok)
    delta = float(fl3.macro_aulc) - float(fl1.macro_aulc)
    checks = [
        ("training stable (no unresolved stability events)", stable),
        (f"DEV macro-AULC(FL3) >= DEV macro-AULC(FL1) + {margin}",
         delta >= float(margin)),
        ("positive DEV macro improvement slope", float(fl3.slope) > 0.0),
    ]
    evidence = {
        "fl3": fl3.to_dict(), "fl1": fl1.to_dict(),
        "delta_macro_aulc": delta, "margin": float(margin),
        "slope": float(fl3.slope), "stable": bool(stable),
    }
    return _decide("FL3_TO_EXTENSIONS", checks, evidence)


def fl4_data_sufficiency(scoreable_items_per_episode: Sequence[int], *,
                         min_episodes: int = FL4_MIN_EPISODES,
                         min_items: int = FL4_MIN_SCOREABLE_ITEMS) -> dict:
    """Count realized episodes carrying enough scoreable feedback items."""
    counts = [int(n) for n in scoreable_items_per_episode]
    qualifying = sum(1 for n in counts if n >= int(min_items))
    return {
        "episodes_seen": len(counts),
        "episodes_with_enough_items": qualifying,
        "min_episodes": int(min_episodes),
        "min_scoreable_items": int(min_items),
        "sufficient": qualifying >= int(min_episodes),
    }


def promote_fl4(*, fl3_finished: bool,
                scoreable_items_per_episode: Sequence[int],
                min_episodes: int = FL4_MIN_EPISODES,
                min_items: int = FL4_MIN_SCOREABLE_ITEMS) -> PromotionDecision:
    """FL4 runs when FL3 finished and the target data exists (contract §10)."""
    sufficiency = fl4_data_sufficiency(
        scoreable_items_per_episode, min_episodes=min_episodes,
        min_items=min_items)
    checks = [
        ("FL3 finished", bool(fl3_finished)),
        (f">= {min_episodes} realized episodes with >= {min_items} scoreable "
         "feedback items each", bool(sufficiency["sufficient"])),
    ]
    return _decide("FL4", checks, {"data_sufficiency": sufficiency})


def promote_fl5(*, core_comparison_complete: bool,
                scheduler_admits: bool) -> PromotionDecision:
    """FL5 runs whenever the core comparison is complete and time admits it."""
    checks = [
        ("core comparison complete", bool(core_comparison_complete)),
        ("scheduler admits the stage", bool(scheduler_admits)),
    ]
    return _decide("FL5", checks, {})


def promote_fl6(fl4_dev: DevMetrics | None = None, *,
                pairwise_ranking_accuracy: float | None = None,
                minimum: float = FL6_MIN_PAIRWISE_ACCURACY
                ) -> PromotionDecision:
    """FL6 requires FL4 DEV pairwise ranking accuracy >= 0.55 (contract §10)."""
    accuracy = pairwise_ranking_accuracy
    if accuracy is None and fl4_dev is not None:
        accuracy = fl4_dev.extra.get("pairwise_ranking_accuracy")
    if accuracy is None:
        raise PromotionError(
            "FL6 promotion needs the FL4 DEV pairwise ranking accuracy")
    if fl4_dev is not None and fl4_dev.split != DEV_SPLIT:  # pragma: no cover
        raise SealedEvidenceRefused("FL4 evidence must be DEVELOPMENT")
    checks = [(f"FL4 DEV pairwise ranking accuracy >= {minimum}",
               float(accuracy) >= float(minimum))]
    return _decide("FL6", checks, {
        "pairwise_ranking_accuracy": float(accuracy),
        "minimum": float(minimum),
        "fl4": None if fl4_dev is None else fl4_dev.to_dict(),
    })


def promote_fl7(*, scheduler_admits: bool, fast_update_stable: bool,
                fl4_succeeded: bool,
                only_remaining_variant_is_fl6_gated: bool = False
                ) -> PromotionDecision:
    """FL7 gate (contract §10).

    FL7 is independently predeclared: it may run even when FL5 is null.  It is
    skipped only when its ONLY remaining variant is the FL6-dependent gated one
    and FL4 failed.
    """
    checks = [
        ("scheduler admits the stage", bool(scheduler_admits)),
        ("stable fast-update implementation (dress-rehearsal flag)",
         bool(fast_update_stable)),
        ("not reduced to the FL6-gated variant alone after an FL4 failure",
         not (bool(only_remaining_variant_is_fl6_gated) and not bool(fl4_succeeded))),
    ]
    return _decide("FL7", checks, {
        "fl4_succeeded": bool(fl4_succeeded),
        "only_remaining_variant_is_fl6_gated":
            bool(only_remaining_variant_is_fl6_gated),
    })


def promote_fl8(*, persistence_point: float, persistence_ci: Sequence[float],
                split: str = DEV_SPLIT) -> PromotionDecision:
    """FL8 requires DEV context-reset persistence > 0 with a CI excluding 0."""
    if str(split) != DEV_SPLIT:
        raise SealedEvidenceRefused(
            f"FL8 persistence evidence must come from {DEV_SPLIT}, got {split!r}")
    ci = [float(v) for v in persistence_ci]
    if len(ci) != 2:
        raise PromotionError("persistence_ci must be a (low, high) pair")
    low, high = min(ci), max(ci)
    checks = [
        ("DEV context-reset persistence > 0", float(persistence_point) > 0.0),
        ("persistence CI excludes 0", low > 0.0),
    ]
    return _decide("FL8", checks, {
        "persistence_point": float(persistence_point),
        "persistence_ci": [low, high],
        "split": DEV_SPLIT,
    })


def select_best_dev_checkpoint(candidates: Sequence[tuple[str, DevMetrics]]
                               ) -> dict:
    """Contract §15 best-DEV checkpoint rule: max DEV macro-AULC.

    Checkpoint selection is a development decision, so it obeys the same
    structural exclusion as promotion: the evidence is :class:`DevMetrics`,
    which cannot be built from sealed records.  Ties break to the EARLIER
    scheduled evaluation, so the rule is deterministic and never depends on
    iteration order.
    """
    entries = list(candidates)
    if not entries:
        raise PromotionError("no checkpoint candidates supplied")
    scored = []
    for index, (tag, metrics) in enumerate(entries):
        if not isinstance(metrics, DevMetrics):
            raise PromotionError(
                f"checkpoint {tag!r}: evidence must be a DevMetrics object "
                "(the structural guard against sealed selection), got "
                f"{type(metrics).__name__}")
        if metrics.split != DEV_SPLIT:  # pragma: no cover - DevMetrics guards
            raise SealedEvidenceRefused(
                f"checkpoint {tag!r}: selection evidence is not DEVELOPMENT")
        if metrics.macro_aulc is None:
            raise PromotionError(
                f"checkpoint {tag!r} has no DEV macro-AULC; the §15 rule cannot "
                "be applied to it")
        scored.append((-float(metrics.macro_aulc), index, str(tag), metrics))
    scored.sort()
    _, _, best_tag, best_metrics = scored[0]
    return {
        "schema": "flb200.best_dev_checkpoint.v1",
        "tag": best_tag,
        "macro_aulc": float(best_metrics.macro_aulc),
        "rule": "max DEV macro-AULC at scheduled evaluations; tie -> earlier",
        "candidates": [{"tag": tag, "macro_aulc": float(m.macro_aulc)}
                       for _, _, tag, m in scored],
        "sealed_consulted": False,
    }


# --------------------------------------------------------------------------
# frozen fallbacks (contract §10)
# --------------------------------------------------------------------------

FALLBACKS: dict[str, tuple[str, ...]] = {
    "FL3_TO_EXTENSIONS": (
        "complete the matched FL1/FL2/FL3 baselines and publish the null",
        "run the second predeclared seed (20260810) if affordable",
        "run the frozen context / surface / feedback diagnostics",
        "never invent an objective live",
    ),
    "FL4": (
        "preserve the FL4 null",
        "skip FL6",
        "proceed to FL5",
    ),
    "FL5": (
        "skip the FL5-dependent parts of FL6",
        "FL7 only as independently predeclared",
    ),
    "FL6": (
        "record the gating null; no gated variant runs",
    ),
    "FL7": (
        "skip FL8",
    ),
    "FL8": (
        "record the absence of persistence evidence; consolidation does not run",
    ),
}

#: Every stage falls back to predeclared work only.
_GLOBAL_FALLBACK = (
    "all remaining time goes to PREDECLARED work only (contract §10)",
)


def fallback_for(stage: str, reason: str | None = None) -> tuple[str, ...]:
    """The frozen predeclared fallback work for a refused stage."""
    key = str(stage).upper()
    specific = FALLBACKS.get(key)
    if specific is None:
        # An unknown stage still falls back to predeclared work only; it never
        # unlocks improvisation.
        specific = ()
    items = tuple(specific) + _GLOBAL_FALLBACK
    if reason:
        items = (f"reason: {reason}",) + items
    return items
