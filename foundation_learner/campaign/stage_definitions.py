"""The declarative campaign stage table (contract §7, §10, §11).

Every stage is a :class:`StageDefinition` record carrying

* its **priority** in the frozen §11 session order (BENCH, FL0, core matched
  comparison, FL4, FL5, FL6, FL7, FL8, additional predeclared seeds, and last
  the single sealed opening);
* an **entry condition** expressed as a dotted path to a promotion predicate
  (``campaign.promotion`` is the only source of promotion truth);
* a **work function** expressed as a DOTTED PATH resolved lazily at runtime, so
  a stage that depends on a module still under construction (``mechanisms/``)
  neither imports it at package-import time nor pins its API here;
* a **projection formula** consuming the BENCH measurements;
* the **outputs** it must produce, and
* its **predeclared fallback work** (contract §10) — no objective is ever
  invented live.

The work functions themselves live at the bottom of this module.  They are the
real integration points into W1 (data), W2 (training) and W4 (evaluation); the
mechanism rungs FL4–FL8 feature-detect ``foundation_learner.mechanisms.*`` and
record ``SKIPPED_MECHANISMS`` when that package is absent.  A mechanism module
that IS importable but lacks its declared entry point is an integration ERROR,
never a silent skip.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from . import o1_isolation, promotion, sealed_gate
from .affordability import (BenchMeasurement, PEFT_MODE, ThroughputTable,
                            grid_updates, projected_arm_seconds)

__all__ = [
    "StageError",
    "StageAbortedOverrun",
    "StageWatchdog",
    "StageDefinition",
    "StageContext",
    "STAGE_TABLE",
    "STAGES_BY_ID",
    "CORE_ARM_STAGES",
    "DIAGNOSTIC_STAGES",
    "EVAL_EPISODES_PER_STAGE",
    "EVAL_SPLIT_BY_STAGE",
    "STAGE_EVAL_MULTIPLIER",
    "STAGE_TRAIN_MULTIPLIER",
    "STAGE_BUNDLE_LOADS",
    "FL4_TARGET_FORWARDS_PER_EPISODE",
    "BENCH_PROBE_UPDATES",
    "BENCH_PROBE_EPISODES",
    "resolve_dotted",
    "module_available",
    "resolve_optional",
    "split_families",
    "eval_split_for",
    "eval_plan",
    "assert_eval_family_coverage",
    "prepare_bundle_for_evaluation",
    "project_stage_seconds",
    "stages_in_priority_order",
]


class StageError(RuntimeError):
    """A stage definition, entry condition, or work function refused."""


class StageAbortedOverrun(RuntimeError):
    """A running stage exceeded its projected budget and was aborted.

    Not a scientific failure: the stage is recorded as ``STAGE_ABORTED_OVERRUN``
    with its predeclared fallback work, and the ladder continues (contract §10 —
    all remaining time goes to predeclared work only).
    """


# --------------------------------------------------------------------------
# lazy dotted-path resolution
# --------------------------------------------------------------------------

def resolve_dotted(dotted: str) -> Any:
    """Resolve ``package.module:attribute`` at call time.

    Import happens HERE, not at module import, which is what lets the stage
    table name modules that are still being built by another worker.
    """
    if ":" not in dotted:
        raise StageError(
            f"dotted path {dotted!r} must be 'package.module:attribute'")
    module_name, attribute = dotted.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise StageError(
            f"module {module_name!r} has no attribute {attribute!r}; the stage "
            "table names an entry point that the module does not provide"
        ) from exc


def module_available(module_name: str) -> bool:
    """True when ``module_name`` can be imported (feature detection)."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def resolve_optional(dotted: str) -> Any | None:
    """Resolve a dotted path, or ``None`` when its MODULE is absent.

    A missing module is a legitimate "not built yet / not shipped" condition
    and yields ``None``.  A present module that lacks the named attribute is an
    integration error and raises: silently skipping that case would hide a real
    break.
    """
    module_name = dotted.split(":", 1)[0]
    if not module_available(module_name):
        return None
    return resolve_dotted(dotted)


# --------------------------------------------------------------------------
# frozen per-stage evaluation maxima (preregistration §12 item 5)
# --------------------------------------------------------------------------

#: Maximum DEVELOPMENT episodes evaluated per stage.  Frozen here so that the
#: only B200-derived quantity left open is the eval BATCH SIZE (post
#: equivalence gate), not the amount of evidence collected.
EVAL_EPISODES_PER_STAGE: dict[str, int] = {
    "FL0": 300,
    "DEV_GRID": 150,
    "FL1": 300,
    "FL2": 300,
    "FL3": 300,
    "FL4": 200,
    "FL5": 200,
    "FL6": 200,
    "FL7": 150,
    "FL8": 150,
    "REMAP_DIAG": 150,
    "INTERFERENCE_DIAG": 120,     # A->B->A chains (20 per ordered DEV pair)
    "POISON_DIAG": 150,
    "SEALED_EVAL": 300,
}

#: Which split a stage's evaluation set is drawn from.  Everything evaluates on
#: DEVELOPMENT except the single sealed opening.
EVAL_SPLIT_BY_STAGE: dict[str, str] = {"SEALED_EVAL": "SEALED_TEST"}

#: How many ONLINE episode walks one "evaluation episode" of a stage really
#: costs (R-M8).  These are structural facts of the stage definitions, not
#: tuning knobs: FL0 runs {structured, correctness_only} x {history,
#: context_reset}; FL5 evaluates the ON/OFF/ON_S0 triad plus the context-reset
#: cell; FL6 walks 2 gate variants x the 5 frozen §9 poison conditions; FL7
#: walks its ungated and gated variants plus the matched no-adapter baseline;
#: FL8 walks 3 consolidation modes x the 3 positions of an A->B->A chain;
#: REMAP_DIAG walks the canonical control plus the 2 pre-generated remap
#: surfaces; INTERFERENCE_DIAG walks the 3 chain positions; POISON_DIAG walks
#: the 5 frozen conditions.
STAGE_EVAL_MULTIPLIER: dict[str, int] = {
    "FL0": 4,
    "FL5": 4,
    "FL6": 10,
    "FL7": 3,
    "FL8": 9,
    "REMAP_DIAG": 3,
    "INTERFERENCE_DIAG": 3,
    "POISON_DIAG": 5,
}

#: How many TRAINING arms of ``U`` updates a stage runs.
STAGE_TRAIN_MULTIPLIER: dict[str, int] = {"FL5": 2}

#: How many FRESH backbone loads a stage performs (contract §7 gives every
#: independent arm its own fresh load; a load is minutes on a 2.6 B checkpoint,
#: so it belongs in the projection).
STAGE_BUNDLE_LOADS: dict[str, int] = {
    "BENCH": 2, "FL0": 1, "DEV_GRID": 2, "FL1": 1, "FL2": 1, "FL3": 1,
    "FL4": 2, "FL5": 3, "FL6": 2, "FL7": 3, "FL8": 4,
    "CORE_MATCHING": 0,
    "REMAP_DIAG": 1, "INTERFERENCE_DIAG": 1, "POISON_DIAG": 1,
    "SECOND_SEED": 0, "SEALED_EVAL": 1,
}

#: Teacher-forced forwards per episode in the FL4 target computation: the
#: realized-learning-value target is a presence/ablation contrast over the
#: episode's feedback items, i.e. a small multiple of full-context forwards.
FL4_TARGET_FORWARDS_PER_EPISODE = 8

#: BENCH probe size: small, measured, and never a scientific result.
BENCH_PROBE_UPDATES = 12
BENCH_PROBE_EPISODES = 2


# --------------------------------------------------------------------------
# family coverage of an evaluation set (R-C3)
# --------------------------------------------------------------------------

def split_families(split: str) -> tuple[str, ...]:
    """The frozen §5 family list of one split, sorted."""
    from ..ecology.split import compute_split

    families = compute_split().get(str(split))
    if not families:
        raise StageError(f"unknown split {split!r}")
    return tuple(sorted(str(f) for f in families))


def eval_split_for(stage: StageDefinition | str) -> str:
    stage_id = getattr(stage, "stage_id", stage)
    return EVAL_SPLIT_BY_STAGE.get(str(stage_id), "DEVELOPMENT")


def eval_plan(ctx: "StageContext", stage: "StageDefinition", *,
              split: str | None = None) -> dict:
    """Per-FAMILY evaluation caps for one stage.

    A single global ``episodes[:cap]`` slice collapses the evaluation set onto
    whichever family happens to be first in shard order, which silently turns a
    macro-averaged, family-clustered metric into a one-family metric.  The cap
    is therefore divided across the split's families and applied PER SHARD, and
    the assembled set is checked against the frozen split membership.
    """
    split = split or eval_split_for(stage)
    families = split_families(split)
    cap = ctx.eval_cap(stage)
    per_family = max(1, int(cap) // len(families))
    return {
        "split": split,
        "families": list(families),
        "n_families": len(families),
        "cap": int(cap),
        "limit_per_family": per_family,
        "total": per_family * len(families),
        "rule": ("the per-stage evaluation cap is divided EQUALLY across the "
                 "split's families; no family may dominate an aggregate "
                 "(contract §9)"),
    }


#: Stages whose ``eval_episodes`` counts CHAINS, not episodes (each chain is
#: three episode walks, which the stage's eval multiplier already carries).
CHAIN_COUNTED_STAGES: tuple[str, ...] = ("INTERFERENCE_DIAG",)


def planned_eval_episodes(ctx: "StageContext", stage: "StageDefinition") -> int:
    """How many evaluation UNITS the stage will really assemble.

    The projection must use this, not the raw cap: the cap is divided across
    the split's families and then applied per family, so a cap of 1 on three
    DEVELOPMENT families really evaluates three episodes.  Projecting the cap
    would under-project every stage by the number of families.
    """
    if not stage.eval_episodes:
        return 0
    if stage.stage_id in CHAIN_COUNTED_STAGES:
        return int(ctx.eval_cap(stage))
    return int(eval_plan(ctx, stage)["total"])


def assert_eval_family_coverage(episodes: Sequence[Any], split: str, *,
                                context: str) -> dict:
    """Fail loudly unless an evaluation set covers exactly the split's families.

    Applied wherever an evaluation set is assembled.  A missing family is a
    silent metric corruption (macro averages stop being macro), so it is an
    error, not a warning.
    """
    expected = set(split_families(split))
    counts: dict[str, int] = {}
    for episode in episodes:
        family = str(getattr(episode, "family_id", None)
                     or (episode or {}).get("family_id"))
        counts[family] = counts.get(family, 0) + 1
    got = set(counts)
    if got != expected:
        raise StageError(
            f"{context}: the evaluation set covers families {sorted(got)} but "
            f"the frozen {split} split is {sorted(expected)} "
            f"(missing {sorted(expected - got)}, unexpected "
            f"{sorted(got - expected)}). A macro-averaged metric computed on a "
            "partial family set is not the metric the contract defines "
            "(§9/§14).")
    return {"split": split, "episodes_per_family": dict(sorted(counts.items())),
            "n_episodes": sum(counts.values())}


# --------------------------------------------------------------------------
# stage watchdog (R-M8)
# --------------------------------------------------------------------------

@dataclass
class StageWatchdog:
    """Aborts a stage that outruns the budget its admission was granted on.

    Two independent conditions, both checked at every phase boundary and on
    every trainer step:

    * ``elapsed > max(projected * safety_factor, minimum_seconds)`` — the
      admission rule reserved exactly ``projected * safety_factor``, so beyond
      it the stage is eating time that was projected for later work.  The floor
      exists because a projection derived from a short BENCH probe cannot bound
      a stage below its own measurement noise;
    * the remaining authorized time has fallen to the FINAL TRANSFER RESERVE —
      the hard money guarantee, with no floor and no tolerance.

    It cannot interrupt a single in-flight model call; it stops the stage at the
    next checked boundary.  That is recorded honestly rather than claimed away.
    """

    stage_id: str
    projected_seconds: float
    safety_factor: float
    minimum_seconds: float
    started: float
    clock: Any
    remaining_fn: Any = None
    reserve_seconds: float = 0.0
    checks: int = 0
    tripped: str | None = None

    @property
    def deadline_seconds(self) -> float:
        return max(float(self.projected_seconds) * float(self.safety_factor),
                   float(self.minimum_seconds))

    def elapsed(self) -> float:
        return max(0.0, float(self.clock.monotonic()) - float(self.started))

    def check(self, phase: str = "") -> None:
        self.checks += 1
        elapsed = self.elapsed()
        if elapsed > self.deadline_seconds:
            self.tripped = "PROJECTION_OVERRUN"
            raise StageAbortedOverrun(
                f"{self.stage_id}: ABORTED at phase {phase or '<unnamed>'} "
                f"after {elapsed:.1f}s; the admitted budget was "
                f"{self.projected_seconds:.1f}s x {self.safety_factor} "
                f"(floor {self.minimum_seconds:.0f}s). The stage's predeclared "
                "fallback work is recorded and the ladder continues.")
        if self.remaining_fn is not None:
            remaining = float(self.remaining_fn())
            if remaining <= float(self.reserve_seconds):
                self.tripped = "RESERVE_FLOOR"
                raise StageAbortedOverrun(
                    f"{self.stage_id}: ABORTED at phase {phase or '<unnamed>'}; "
                    f"only {remaining:.1f}s of authorized time remain and the "
                    f"final transfer reserve is {self.reserve_seconds:.1f}s. "
                    "The reserve is never consumed by a stage (contract §11).")

    def to_dict(self) -> dict:
        return {"stage_id": self.stage_id,
                "projected_seconds": float(self.projected_seconds),
                "deadline_seconds": self.deadline_seconds,
                "elapsed_seconds": round(self.elapsed(), 6),
                "checks": int(self.checks), "tripped": self.tripped}


# --------------------------------------------------------------------------
# stage definitions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    priority: int
    kind: str                      # BENCH | EVAL | TRAIN | GRID | MECHANISM | SEALED
    work: str                      # dotted path, resolved lazily
    projection: str                # projection formula key
    outputs: tuple[str, ...]
    entry: str | None = None       # dotted path to a promotion predicate
    fallback_work: tuple[str, ...] = ()
    requires_modules: tuple[str, ...] = ()
    arm_id: str | None = None
    eval_episodes: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "priority": self.priority,
            "kind": self.kind,
            "work": self.work,
            "projection": self.projection,
            "outputs": list(self.outputs),
            "entry": self.entry,
            "fallback_work": list(self.fallback_work),
            "requires_modules": list(self.requires_modules),
            "arm_id": self.arm_id,
            "eval_episodes": int(self.eval_episodes),
            "notes": self.notes,
        }


_W = "foundation_learner.campaign.stage_definitions"

STAGE_TABLE: tuple[StageDefinition, ...] = (
    StageDefinition(
        stage_id="BENCH", priority=1, kind="BENCH",
        work=f"{_W}:bench_work", projection="BENCH",
        outputs=("bench_measurement.json",),
        notes=("runtime/throughput validation; measured on a NON-EVALUATION "
               "TRAIN shard, never on DEV or SEALED data"),
        fallback_work=("refuse the whole FL ladder: without measured "
                       "throughput no projection may be made",),
    ),
    StageDefinition(
        stage_id="FL0", priority=2, kind="EVAL",
        work=f"{_W}:fl0_work", projection="EVAL",
        outputs=("fl0_base_report.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL0"],
        notes="base model, no training; not a treatment",
        fallback_work=("record the FL0 records that completed",),
    ),
    StageDefinition(
        stage_id="DEV_GRID", priority=3, kind="GRID",
        work=f"{_W}:dev_grid_work", projection="GRID",
        outputs=("DEV_DECISIONS_FROZEN.json", "dev_grid_report.json"),
        arm_id="FL3",
        eval_episodes=EVAL_EPISODES_PER_STAGE["DEV_GRID"],
        notes=("the frozen 2-LR development grid on FL3 at 25 % U; it is part "
               "of the core-comparison allocation (contract §8/§11) and must "
               "complete before any core arm starts"),
        fallback_work=("refuse the core arms: their learning rate is only "
                       "defined by this grid",),
    ),
    StageDefinition(
        stage_id="FL1", priority=4, kind="TRAIN",
        work=f"{_W}:core_arm_work", projection="TRAIN_ARM", arm_id="FL1",
        outputs=("arm_result_FL1.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL1"],
        notes="static baseline (isolated TASK -> ANSWER pairs)",
        fallback_work=("record the partial arm and its compute ledger",),
    ),
    StageDefinition(
        stage_id="FL2", priority=4, kind="TRAIN",
        work=f"{_W}:core_arm_work", projection="TRAIN_ARM", arm_id="FL2",
        outputs=("arm_result_FL2.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL2"],
        notes="successful-history imitation",
        fallback_work=("record the partial arm and its compute ledger",),
    ),
    StageDefinition(
        stage_id="FL3", priority=4, kind="TRAIN",
        work=f"{_W}:core_arm_work", projection="TRAIN_ARM", arm_id="FL3",
        outputs=("arm_result_FL3.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL3"],
        notes="ordered-feedback meta-training (CORE TREATMENT)",
        fallback_work=promotion.FALLBACKS["FL3_TO_EXTENSIONS"],
    ),
    StageDefinition(
        stage_id="CORE_MATCHING", priority=5, kind="AUDIT",
        work=f"{_W}:core_matching_work", projection="AUDIT",
        outputs=("core_matching_report.json",),
        notes=("compute matching + shared base identity across FL1/FL2/FL3, "
               "invoked in production immediately after the core comparison: "
               "training.arms.assert_core_arms_matched, "
               "training.compute_accounting.compare_core_ledgers and the base "
               "identity check must all pass before any extension consumes the "
               "comparison (contract §8/§15)"),
        fallback_work=("record which core arms completed and refuse to present "
                       "an unmatched comparison as the headline",),
    ),
    StageDefinition(
        stage_id="FL4", priority=6, kind="MECHANISM",
        work=f"{_W}:fl4_work", projection="MECHANISM",
        entry=f"{_W}:fl4_entry",
        outputs=("fl4_value_head_report.json",),
        requires_modules=("foundation_learner.mechanisms.value_head",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL4"],
        fallback_work=promotion.FALLBACKS["FL4"],
    ),
    StageDefinition(
        stage_id="FL5", priority=7, kind="MECHANISM",
        work=f"{_W}:fl5_work", projection="MECHANISM",
        entry=f"{_W}:fl5_entry",
        outputs=("fl5_fast_state_report.json",),
        requires_modules=("foundation_learner.mechanisms.fast_state",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL5"],
        fallback_work=promotion.FALLBACKS["FL5"],
    ),
    StageDefinition(
        stage_id="FL6", priority=8, kind="MECHANISM",
        work=f"{_W}:fl6_work", projection="MECHANISM",
        entry=f"{_W}:fl6_entry",
        outputs=("fl6_value_gating_report.json",),
        requires_modules=("foundation_learner.mechanisms.value_gating",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL6"],
        fallback_work=promotion.FALLBACKS["FL6"],
    ),
    StageDefinition(
        stage_id="FL7", priority=9, kind="MECHANISM",
        work=f"{_W}:fl7_work", projection="MECHANISM",
        entry=f"{_W}:fl7_entry",
        outputs=("fl7_fast_adapter_report.json",),
        requires_modules=("foundation_learner.mechanisms.fast_adapter",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL7"],
        fallback_work=promotion.FALLBACKS["FL7"],
    ),
    StageDefinition(
        stage_id="FL8", priority=10, kind="MECHANISM",
        work=f"{_W}:fl8_work", projection="MECHANISM",
        entry=f"{_W}:fl8_entry",
        outputs=("fl8_consolidation_report.json",),
        requires_modules=("foundation_learner.mechanisms.consolidation",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["FL8"],
        fallback_work=promotion.FALLBACKS["FL8"],
    ),
    StageDefinition(
        stage_id="REMAP_DIAG", priority=11, kind="EVAL",
        work=f"{_W}:remap_diag_work", projection="EVAL",
        outputs=("remap_diag_report.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["REMAP_DIAG"],
        notes=("frozen surface-remap diagnostic (metric 13). UNCONDITIONAL: it "
               "has no promotion entry condition, so it also runs under the "
               "§10 FL3-null fallback, which names exactly these diagnostics"),
        fallback_work=("record the remap records that completed",),
    ),
    StageDefinition(
        stage_id="INTERFERENCE_DIAG", priority=11, kind="EVAL",
        work=f"{_W}:interference_diag_work", projection="EVAL",
        outputs=("interference_diag_report.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["INTERFERENCE_DIAG"],
        notes=("frozen A->B->A interference diagnostic (metric 11), on the "
               "pre-generated DEVELOPMENT chains; UNCONDITIONAL"),
        fallback_work=("record the chains that completed",),
    ),
    StageDefinition(
        stage_id="POISON_DIAG", priority=11, kind="EVAL",
        work=f"{_W}:poison_diag_work", projection="EVAL",
        outputs=("poison_diag_report.json",),
        eval_episodes=EVAL_EPISODES_PER_STAGE["POISON_DIAG"],
        notes=("frozen poisoned-feedback diagnostic (metric 12) over the five "
               "§9 conditions; UNCONDITIONAL and independent of FL6, which "
               "measures GATING rather than blind incorporation"),
        fallback_work=("record the conditions that completed",),
    ),
    StageDefinition(
        stage_id="SECOND_SEED", priority=12, kind="TRAIN",
        work=f"{_W}:second_seed_work", projection="CORE_COMPARISON",
        outputs=("second_seed_report.json",),
        notes=("additional predeclared seed 20260810; predeclared work only, "
               "runs last among training work"),
        fallback_work=("skip; the single-seed result stands as reported",),
    ),
    StageDefinition(
        stage_id="SEALED_EVAL", priority=13, kind="SEALED",
        work=f"{_W}:sealed_eval_work", projection="EVAL",
        entry=f"{_W}:sealed_eval_entry",
        outputs=("SEALED_OPENING_LEDGER.jsonl", "sealed_eval_report.json"),
        eval_episodes=EVAL_EPISODES_PER_STAGE["SEALED_EVAL"],
        notes=("the single sealed opening, LAST: it can never inform a "
               "development decision because every such decision is already "
               "frozen in DEV_DECISIONS_FROZEN.json. It requires the COMPLETED "
               "core comparison and evaluates the FROZEN promoted candidate "
               "from its recorded checkpoint, never a fresh base model"),
        fallback_work=("leave the sealed set unopened",),
    ),
)

STAGES_BY_ID: dict[str, StageDefinition] = {s.stage_id: s for s in STAGE_TABLE}
CORE_ARM_STAGES: tuple[str, ...] = ("FL1", "FL2", "FL3")
#: Unconditional post-core diagnostics (contract §9 metrics 11, 12, 13).
DIAGNOSTIC_STAGES: tuple[str, ...] = ("REMAP_DIAG", "INTERFERENCE_DIAG",
                                      "POISON_DIAG")
#: The frozen core treatment: the arm the sealed set evaluates unless the
#: development decisions name another core arm.
DEFAULT_PROMOTED_ARM = "FL3"


def stages_in_priority_order() -> list[StageDefinition]:
    """Frozen §11 session priority (stable within a priority band)."""
    return sorted(STAGE_TABLE, key=lambda s: (s.priority, s.stage_id))


# --------------------------------------------------------------------------
# projections
# --------------------------------------------------------------------------

#: Declared bounded budget of a pure bookkeeping stage (CORE_MATCHING): it
#: hashes manifests and compares ledgers already on disk.
AUDIT_STAGE_SECONDS = 60.0


def _eval_seconds(bench: BenchMeasurement, stage_id: str, episodes: int,
                  *, factor: int = 1) -> float:
    if int(episodes) <= 0:
        return 0.0
    if bench.eval_seconds_per_episode is None:
        raise StageError(
            f"{stage_id}: BENCH measured no evaluation cost; evaluation "
            "seconds may not be guessed")
    return float(bench.eval_seconds_per_episode) * int(episodes) * int(factor)


def _load_seconds(bench: BenchMeasurement, stage: StageDefinition) -> float:
    """Fresh-backbone-load term (R-M8).

    A measurement is required whenever the stage loads a bundle: contract §7
    forces a fresh multi-GB load per arm, and pretending it is free is exactly
    the kind of optimistic projection the reserve exists to prevent.
    """
    loads = int(STAGE_BUNDLE_LOADS.get(stage.stage_id, 1))
    if loads <= 0:
        return 0.0
    if bench.model_load_seconds is None:
        raise StageError(
            f"{stage.stage_id}: BENCH measured no model-load time; the "
            "per-stage fresh-load cost may not be guessed (contract §7 loads "
            "the frozen checkpoint again for every arm)")
    return float(bench.model_load_seconds) * loads


def project_stage_seconds(stage: StageDefinition, *, bench: BenchMeasurement,
                          updates: int | None,
                          eval_episodes: int | None = None,
                          extra: Mapping[str, Any] | None = None) -> dict:
    """Projected seconds for one stage from the measured BENCH numbers.

    Every stage-shape multiplier comes from :data:`STAGE_EVAL_MULTIPLIER` /
    :data:`STAGE_TRAIN_MULTIPLIER` / :data:`STAGE_BUNDLE_LOADS`, i.e. from the
    number of arms, cells, variants and conditions the stage actually runs.  A
    projection that models a multi-cell stage as one cell is not conservative,
    it is wrong, and it is what the safety factor is NOT for (R-M8).
    """
    extra = dict(extra or {})
    episodes = int(stage.eval_episodes if eval_episodes is None else eval_episodes)
    eval_factor = int(STAGE_EVAL_MULTIPLIER.get(stage.stage_id, 1))
    train_factor = int(STAGE_TRAIN_MULTIPLIER.get(stage.stage_id, 1))
    detail: dict[str, Any] = {"stage": stage.stage_id,
                              "projection": stage.projection,
                              "scope": bench.scope,
                              "eval_multiplier": eval_factor,
                              "train_multiplier": train_factor,
                              "bundle_loads": int(
                                  STAGE_BUNDLE_LOADS.get(stage.stage_id, 1))}
    if stage.projection == "BENCH":
        seconds = projected_arm_seconds(bench, BENCH_PROBE_UPDATES)
        detail["probe_updates"] = BENCH_PROBE_UPDATES
        detail["projected_seconds"] = float(seconds)
        return detail
    if stage.projection == "AUDIT":
        # a pure bookkeeping stage: it reads manifests and ledgers already on
        # disk and loads no model.  Declared, bounded, and never estimated from
        # a throughput number it does not use.
        detail.update({"projected_seconds": float(AUDIT_STAGE_SECONDS),
                       "note": "declared bookkeeping budget; no model is loaded"})
        return detail

    load_seconds = _load_seconds(bench, stage)
    detail["load_seconds"] = load_seconds

    if stage.projection == "EVAL":
        eval_seconds = _eval_seconds(bench, stage.stage_id, episodes,
                                     factor=eval_factor)
        seconds = eval_seconds + load_seconds
        detail.update({"eval_episodes": episodes, "eval_seconds": eval_seconds})
    elif stage.projection == "TRAIN_ARM":
        if updates is None:
            raise StageError(
                f"{stage.stage_id}: no U selected; a training stage cannot be "
                "projected before the affordability rule chooses U")
        train = train_factor * projected_arm_seconds(bench, int(updates))
        eval_seconds = _eval_seconds(bench, stage.stage_id, episodes,
                                     factor=eval_factor)
        seconds = train + eval_seconds + load_seconds
        detail.update({"updates": int(updates), "train_seconds": train,
                       "eval_seconds": eval_seconds, "eval_episodes": episodes})
    elif stage.projection == "MECHANISM":
        if updates is None:
            raise StageError(
                f"{stage.stage_id}: no U selected; a mechanism stage cannot be "
                "projected before the affordability rule chooses U")
        train = train_factor * projected_arm_seconds(bench, int(updates))
        eval_seconds = _eval_seconds(bench, stage.stage_id, episodes,
                                     factor=eval_factor)
        forward_seconds = 0.0
        if stage.stage_id == "FL4":
            # FL4 trains a small head, but its cost is dominated by the
            # TARGET computation: presence/ablation teacher-forced forwards
            # over the training and DEV episodes (contract §7).
            n_train = int(extra.get("fl4_train_episodes") or episodes)
            n_dev = int(extra.get("fl4_dev_episodes") or episodes)
            if bench.forward_seconds_per_episode is None:
                raise StageError(
                    "FL4: BENCH measured no teacher-forced forward cost; the "
                    "value-head target computation may not be guessed")
            forward_seconds = (float(bench.forward_seconds_per_episode)
                               * FL4_TARGET_FORWARDS_PER_EPISODE
                               * (n_train + n_dev))
            train = 0.0     # the head itself is a 25 M-parameter MLP
            detail.update({"fl4_train_episodes": n_train,
                           "fl4_dev_episodes": n_dev,
                           "forwards_per_episode":
                               FL4_TARGET_FORWARDS_PER_EPISODE})
        seconds = train + eval_seconds + forward_seconds + load_seconds
        detail.update({"updates": int(updates), "train_seconds": train,
                       "eval_seconds": eval_seconds, "eval_episodes": episodes,
                       "forward_seconds": forward_seconds})
    elif stage.projection == "GRID":
        if updates is None:
            raise StageError(f"{stage.stage_id}: no U selected")
        per_config = projected_arm_seconds(bench, grid_updates(int(updates)))
        eval_seconds = _eval_seconds(bench, stage.stage_id, episodes, factor=2)
        seconds = 2.0 * per_config + eval_seconds + load_seconds
        detail.update({"configs": 2, "updates_per_config": grid_updates(int(updates)),
                       "eval_seconds": eval_seconds})
    elif stage.projection == "CORE_COMPARISON":
        if updates is None:
            raise StageError(f"{stage.stage_id}: no U selected")
        train = 3.0 * projected_arm_seconds(bench, int(updates))
        eval_seconds = _eval_seconds(bench, stage.stage_id, episodes, factor=3)
        seconds = train + eval_seconds + load_seconds
        detail.update({"arms": 3, "updates": int(updates),
                       "eval_seconds": eval_seconds})
    else:  # pragma: no cover - the table is closed
        raise StageError(f"unknown projection formula {stage.projection!r}")
    detail["projected_seconds"] = float(seconds)
    return detail


# --------------------------------------------------------------------------
# the stage context
# --------------------------------------------------------------------------

@dataclass
class StageContext:
    """Everything a work function may touch.

    ``bundle_factory`` returns a FRESH model bundle: contract §7 requires every
    independent arm to start from a fresh load of the identical frozen
    checkpoint, so no work function is given a shared, already-trained model.
    """

    out_dir: str
    pregen_root: str
    bundle_factory: Callable[[], Any]
    guard: o1_isolation.IsolationGuard
    throughput: ThroughputTable = field(default_factory=ThroughputTable)
    scope: str = PEFT_MODE
    updates: int | None = None
    learning_rate: float | None = None
    root_seed: int = 20260809
    max_tokens_per_batch: int = 2048
    eval_episode_cap: int | None = None
    train_example_cap: int | None = None
    rehearsal: bool = False
    label: str = "FL_CAMPAIGN"
    results: dict[str, Any] = field(default_factory=dict)
    scheduler: Any = None
    watchdog: StageWatchdog | None = None
    checkpoint_dir: str | None = None
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)

    # -- helpers ------------------------------------------------------------

    def stage_dir(self, stage_id: str) -> str:
        return self.guard.makedirs(os.path.join(self.out_dir, stage_id.lower()))

    def eval_cap(self, stage: StageDefinition) -> int:
        cap = stage.eval_episodes or 1
        if self.eval_episode_cap is not None:
            cap = min(cap, int(self.eval_episode_cap))
        return max(1, cap)

    def checkpoint(self, phase: str = "") -> None:
        """Watchdog checkpoint: raise :class:`StageAbortedOverrun` if overrun.

        Called at every phase boundary of a stage's work function.  It is a
        no-op when the stage runs outside a scheduler (unit tests, adapters).
        """
        if self.watchdog is not None:
            self.watchdog.check(phase)

    def write(self, stage_id: str, name: str, payload: Any) -> str:
        return self.guard.write_json(
            os.path.join(self.stage_dir(stage_id), name), payload)


# --------------------------------------------------------------------------
# data access (always through the loader guard)
# --------------------------------------------------------------------------

def _episode_shards(pregen_root: str, split: str, mode: str,
                    guard: o1_isolation.IsolationGuard) -> list[str]:
    root = guard.guard(os.path.join(pregen_root, "episodes", split),
                       o1_isolation.MODE_READ)
    if not os.path.isdir(root):
        raise StageError(f"missing pre-generated episodes under {root!r}")
    return [os.path.join(root, name) for name in sorted(os.listdir(root))
            if name.endswith(f".{mode}.jsonl")]


def load_episodes(pregen_root: str, split: str, mode: str, *,
                  guard: o1_isolation.IsolationGuard,
                  limit_per_family: int | None = None) -> list[Any]:
    """Load plain (non-sealed) episodes through the loader guard."""
    from ..data.shards import read_shard
    from ..episodes.schema import Episode

    episodes: list[Any] = []
    for path in _episode_shards(pregen_root, split, mode, guard):
        resolved = sealed_gate.loader_guard(path, unlock=None, guard=guard,
                                            context="stage_definitions")
        rows = read_shard(resolved)
        if limit_per_family is not None:
            rows = rows[:int(limit_per_family)]
        episodes.extend(Episode.from_dict(row) for row in rows)
    return episodes


def load_balanced_eval_episodes(ctx: "StageContext", stage: "StageDefinition",
                                *, mode: str = "scripted",
                                split: str | None = None,
                                context: str | None = None) -> list[Any]:
    """THE evaluation-set assembler: per-family caps + a coverage assertion.

    Every DEVELOPMENT evaluation set in the campaign comes from here, so the
    "no single-family domination" requirement of contract §9 is enforced in one
    place instead of being re-derived per stage.
    """
    plan = eval_plan(ctx, stage, split=split)
    episodes = load_episodes(ctx.pregen_root, plan["split"], mode,
                             guard=ctx.guard,
                             limit_per_family=plan["limit_per_family"])
    coverage = assert_eval_family_coverage(
        episodes, plan["split"],
        context=context or f"{stage.stage_id} evaluation set")
    ctx.results.setdefault("_eval_plans", {})[stage.stage_id] = {
        "plan": plan, "coverage": coverage}
    return episodes


def load_side_table(ctx: "StageContext", kind: str, split: str) -> list[dict]:
    """Guarded read of a pre-generated side table (remaps / poison / chains)."""
    root = ctx.guard.guard(os.path.join(ctx.pregen_root, kind, split),
                           o1_isolation.MODE_READ)
    if not os.path.isdir(root):
        raise StageError(
            f"missing pre-generated {kind!r} table for {split} under {root!r}")
    from ..data.shards import read_shard

    rows: list[dict] = []
    for name in ctx.guard.listdir(root):
        if not name.endswith(".jsonl"):
            continue
        path = sealed_gate.loader_guard(os.path.join(root, name), unlock=None,
                                        guard=ctx.guard,
                                        context=f"stage_definitions.{kind}")
        rows.extend(read_shard(path))
    return rows


def load_chain_specs(ctx: "StageContext", episodes: Sequence[Any], *,
                     split: str = "DEVELOPMENT",
                     limit: int | None = None) -> list[Any]:
    """Pre-generated A->B->A chains, resolved and BALANCED across pairs.

    Two defects this replaces (R-C3).  First, a chain names three episode ids
    and was resolved only against whatever capped subset a caller happened to
    hold, so most chains silently vanished; the pool is therefore completed
    from the split's own shards when an id is missing.  Second, the rows arrive
    grouped by ordered family pair, so a plain ``limit`` slice returned chains
    of ONE pair — the interference metric is defined as family-balanced
    (contract §9), and an unbalanced draw is not a smaller version of it.  The
    selection is round-robin over the ordered pairs.
    """
    from ..evaluation.interference import ChainSpec

    by_id = {ep.episode_id: ep for ep in episodes}
    rows = load_side_table(ctx, "chains", split)
    needed = {str(i) for row in rows for i in row.get("episode_ids", [])}
    if not needed.issubset(by_id):
        for episode in load_episodes(ctx.pregen_root, split, "scripted",
                                     guard=ctx.guard):
            by_id.setdefault(episode.episode_id, episode)

    by_pair: dict[tuple[str, str], list[Any]] = {}
    unresolved = 0
    for row in rows:
        ids = [str(i) for i in row.get("episode_ids", [])]
        if len(ids) != 3 or any(i not in by_id for i in ids):
            unresolved += 1
            continue
        pair = (str(row["family_a"]), str(row["family_b"]))
        by_pair.setdefault(pair, []).append(ChainSpec(
            chain_id=str(row["chain_id"]), family_a=pair[0], family_b=pair[1],
            episode_a1=by_id[ids[0]], episode_b=by_id[ids[1]],
            episode_a2=by_id[ids[2]]))
    if unresolved and not by_pair:
        raise StageError(
            f"none of the {len(rows)} pre-generated {split} chains resolved "
            "against the episode pool; the A->B->A evaluation would be empty")

    ordered_pairs = sorted(by_pair)
    selected: list[Any] = []
    index = 0
    while ordered_pairs and (limit is None or len(selected) < int(limit)):
        added = False
        for pair in ordered_pairs:
            chains = by_pair[pair]
            if index >= len(chains):
                continue
            selected.append(chains[index])
            added = True
            if limit is not None and len(selected) >= int(limit):
                break
        if not added:
            break
        index += 1
    return selected


def load_fl1_items(pregen_root: str, *, guard: o1_isolation.IsolationGuard,
                   limit_per_family: int | None = None) -> list[dict]:
    from ..data.shards import read_shard

    root = guard.guard(os.path.join(pregen_root, "fl1_static", "TRAIN"),
                       o1_isolation.MODE_READ)
    if not os.path.isdir(root):
        raise StageError(f"missing FL1 static pool under {root!r}")
    items: list[dict] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".jsonl"):
            continue
        rows = read_shard(os.path.join(root, name))
        if limit_per_family is not None:
            rows = rows[:int(limit_per_family)]
        items.extend(rows)
    return items


def env_factory(episode: Any):
    """Exact-verifier environment for one episode (W4 API)."""
    from ..ecology.families import get_family
    from ..evaluation.learning_curve import environment_from_episode

    return environment_from_episode(episode, get_family(episode.family_id))


def _render_fl1_pair(prompt_text: str, answer_payload: str):
    """FL1 pool rendering with the SAME surface as the episode renderer.

    Contract §8/§20 forbid hidden asymmetries between the core arms; a
    different textual surface for FL1 would be exactly such an asymmetry, so
    the instruction header and the ``TASK:``/``ATTEMPT:`` markers are
    reproduced from ``episodes/render.py`` rather than reinvented.
    """
    from ..episodes.render import INSTRUCTION_HEADER_V0, MARKERS
    from ..episodes.schema import Role

    prefix = (f"{INSTRUCTION_HEADER_V0}{MARKERS[Role.TASK]} "
              f"{prompt_text.rstrip()}\n{MARKERS[Role.MODEL_ATTEMPT]} ")
    answer_line = f"ANSWER: {answer_payload.strip()}"
    return prefix + answer_line + "\n", (len(prefix), len(prefix) + len(answer_line))


def build_arm_examples(ctx: StageContext, arm_id: str, tokenizer) -> list[dict]:
    """Tokenised training examples for one arm from the pre-generated data."""
    from ..training.tokenization import (isolated_pair_to_training_example,
                                         iter_examples)

    cap = ctx.train_example_cap
    if arm_id == "FL1":
        items = load_fl1_items(ctx.pregen_root, guard=ctx.guard,
                               limit_per_family=cap)
        examples = [
            isolated_pair_to_training_example(
                item["prompt_text"], item["answer_canonical"], tokenizer,
                render_pair_fn=_render_fl1_pair,
                meta={"item_id": item.get("item_id"),
                      "episode_id": item.get("source_episode_id"),
                      "family_id": item.get("family_id")})
            for item in items]
    else:
        mode = "successful" if arm_id == "FL2" else "scripted"
        episodes = load_episodes(ctx.pregen_root, "TRAIN", mode,
                                 guard=ctx.guard, limit_per_family=cap)
        examples = list(iter_examples(episodes, tokenizer, arm_id))
    if not examples:
        raise StageError(f"{arm_id}: no training examples were built")
    return examples


# --------------------------------------------------------------------------
# work functions
# --------------------------------------------------------------------------

def _arm_config(ctx: StageContext, arm_id: str, *, updates: int, stage: str,
                learning_rate: float):
    """Build the arm configuration, with the frozen checkpoint cadence.

    A DRESS REHEARSAL runs the miniature ladder, whose step counts are (by
    design) not on the frozen ladder; such a run is tagged ``STAGE_SMOKE``,
    which ``training/arms.py`` defines as "local mechanics only; never a
    scientific result".  A real campaign keeps the CORE/GRID stage tags and is
    therefore fully bound by the frozen ladder and learning-rate grid.
    """
    from ..training.arms import STAGE_SMOKE, make_arm_config

    stage_tag = STAGE_SMOKE if ctx.rehearsal else stage
    return make_arm_config(
        arm_id, learning_rate=float(learning_rate), updates=int(updates),
        max_tokens_per_batch=int(ctx.max_tokens_per_batch), peft_mode=ctx.scope,
        seed=int(ctx.root_seed), stage=stage_tag,
        checkpoint_every_steps=200, checkpoint_every_seconds=600.0,
        max_seq_len=min(2048, int(ctx.max_tokens_per_batch)))


def _measure_forward_seconds(ctx: StageContext, bundle: Any,
                             probe_episodes: Sequence[Any]) -> float | None:
    """Time ONE teacher-forced forward over a rendered TRAIN episode.

    This is the unit of the FL4 target computation (a presence/ablation
    contrast is a handful of full-context forwards, not optimizer updates), so
    the FL4 projection is measured rather than guessed.  Returns ``None`` only
    when there is no probe episode at all, in which case the FL4 projection
    refuses rather than estimating.
    """
    if not probe_episodes:
        return None
    import torch

    from ..training.tokenization import episode_to_training_example

    example = episode_to_training_example(probe_episodes[0], bundle.tokenizer,
                                          "FL3")
    ids = torch.tensor([list(example["input_ids"])], dtype=torch.long,
                       device=bundle.device)
    was_training = bundle.model.training
    bundle.model.eval()
    try:
        with torch.no_grad():
            t0 = time.monotonic()
            bundle.model(input_ids=ids)
            seconds = time.monotonic() - t0
    finally:
        bundle.model.train(was_training)
    return float(seconds)


def bench_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """Measure real throughput on a NON-EVALUATION TRAIN shard (contract §11).

    Two kinds of number are measured, both by really running the code the
    campaign will run:

    * seconds per optimizer update, per trainable-parameter scope (W2 trainer
      on TRAIN episodes).  Both scopes are measured when both are requested,
      because the §11 FULL-vs-PEFT rule needs a FULL projection to decide, and
      an unmeasured scope is treated as INELIGIBLE rather than estimated;
    * seconds per evaluation episode (W4 online walker on TRAIN episodes).  The
      evaluation cost does not depend on the trainable-parameter scope — the
      walker only runs greedy inference — so it is measured once and attached
      to every measurement;
    * seconds for ONE fresh backbone load (contract §7 loads the frozen
      checkpoint again for every arm, so the projections need it), and
    * seconds for ONE teacher-forced forward over a rendered episode, which is
      the unit of the FL4 target computation.

    The benchmark never touches DEVELOPMENT or SEALED_TEST data.
    """
    from ..training.arms import STAGE_SMOKE
    from ..training.trainer import run_training_arm

    out_dir = ctx.stage_dir(stage.stage_id)
    scopes = list(ctx.extra.get("bench_scopes") or [ctx.scope])
    updates = int(ctx.extra.get("bench_probe_updates", BENCH_PROBE_UPDATES))
    original_scope = ctx.scope

    # evaluation cost, measured on TRAIN episodes (never on evaluation data)
    n_probe = int(ctx.extra.get("bench_probe_episodes", BENCH_PROBE_EPISODES))
    probe_episodes = load_episodes(ctx.pregen_root, "TRAIN", "scripted",
                                   guard=ctx.guard, limit_per_family=1)[:n_probe]
    eval_seconds_per_episode = None
    ctx.checkpoint("bench:model_load")
    t_load = time.monotonic()
    eval_bundle = ctx.bundle_factory()
    model_load_seconds = time.monotonic() - t_load
    # measure the EVALUATION cost in the state evaluation really runs in
    # (eval mode, no layer-level checkpointing) - Amendment 16
    prepare_bundle_for_evaluation(eval_bundle)
    if probe_episodes:
        from ..evaluation.learning_curve import LearningCurveConfig, run_episodes

        ctx.checkpoint("bench:eval_probe")
        t0 = time.monotonic()
        records = run_episodes(eval_bundle, probe_episodes, env_factory,
                               cfg=LearningCurveConfig(
                                   arm_tag="BENCH_PROBE",
                                   generation=_generation_config(ctx)))
        elapsed = time.monotonic() - t0
        eval_seconds_per_episode = elapsed / max(1, len(records))
    forward_seconds_per_episode = _measure_forward_seconds(
        ctx, eval_bundle, probe_episodes)

    measurements: dict[str, dict] = {}
    ledgers: dict[str, dict] = {}
    try:
        for scope in scopes:
            ctx.checkpoint(f"bench:train_probe:{scope}")
            ctx.scope = scope
            bundle = ctx.bundle_factory()
            examples = build_arm_examples(ctx, "FL3", bundle.tokenizer)
            cfg = _arm_config(ctx, "FL3", updates=updates, stage=STAGE_SMOKE,
                              learning_rate=float(ctx.extra.get("bench_lr", 1e-4)))
            if ctx.scheduler is not None:
                ctx.scheduler.assert_checkpoint_cadence(cfg)
            result = run_training_arm(cfg, bundle, examples,
                                      os.path.join(out_dir, f"bench_arm_{scope}"),
                                      _trainer_hooks(ctx, stage.stage_id),
                                      allow_retry=False)
            measurement = BenchMeasurement.from_ledger(
                result.ledger, scope,
                eval_seconds_per_episode=eval_seconds_per_episode,
                model_load_seconds=model_load_seconds,
                forward_seconds_per_episode=forward_seconds_per_episode,
                notes=("measured on a non-evaluation TRAIN shard",
                       "NOT a scientific result: throughput only"))
            ctx.throughput.add(measurement)
            measurements[scope] = measurement.to_dict()
            ledgers[scope] = result.ledger
    finally:
        ctx.scope = original_scope

    payload = {
        "schema": "flb200.bench_report.v1",
        "stage": stage.stage_id,
        "scopes_measured": scopes,
        "measurements": measurements,
        "compute_ledgers": ledgers,
        "eval_seconds_per_episode": eval_seconds_per_episode,
        "model_load_seconds": model_load_seconds,
        "forward_seconds_per_episode": forward_seconds_per_episode,
        "probe_episodes": len(probe_episodes),
        "note": ("BENCH measures runtime only; it never produces a scientific "
                 "result and never reads DEVELOPMENT or SEALED_TEST data"),
    }
    ctx.write(stage.stage_id, "bench_measurement.json", payload)
    ctx.results["BENCH"] = payload
    return payload


#: Contract §6/§7 frozen evaluation decode budget.
FROZEN_MAX_NEW_TOKENS = 64


def _generation_config(ctx: StageContext):
    """Greedy decode configuration.

    ``max_new_tokens`` is FROZEN at 64 for the campaign.  A dress rehearsal may
    shorten it (``ctx.extra["max_new_tokens"]``) purely to keep the offline
    mechanics walk under its wall-clock limit; the override is only honoured
    for a run explicitly marked ``rehearsal`` and is recorded in the report.
    """
    from ..evaluation.generation import GenerationConfig

    tokens = FROZEN_MAX_NEW_TOKENS
    override = ctx.extra.get("max_new_tokens")
    if override is not None:
        if not ctx.rehearsal:
            raise StageError(
                "max_new_tokens is frozen at 64 for the campaign; only a "
                "labelled dress rehearsal may shorten it (contract §6)")
        tokens = int(override)
    return GenerationConfig(max_new_tokens=tokens)


def _dev_episodes(ctx: StageContext, stage: StageDefinition, *,
                  mode: str = "scripted") -> list[Any]:
    """The DEVELOPMENT evaluation set: per-family capped, coverage-asserted."""
    return load_balanced_eval_episodes(ctx, stage, mode=mode,
                                       split="DEVELOPMENT")


def _dev_metrics(records: Sequence[Mapping[str, Any]], stage: str,
                 stability_events: Sequence[str] = ()) -> promotion.DevMetrics:
    return promotion.DevMetrics.from_records(
        records, stage=stage, stability_events=stability_events)


def prepare_bundle_for_evaluation(bundle: Any) -> dict:
    """Put a bundle into the ONLY state in which a decode is valid (Amend. 16).

    ``model.eval()`` + layer-level gradient checkpointing OFF.  A model left in
    TRAIN mode with the checkpointing flag set decodes DIFFERENTLY: transformers'
    ``GradientCheckpointingLayer.__call__`` drops ``use_cache`` and
    ``past_key_values`` in exactly that state, so the manual greedy decode
    recomputes from scratch and yields different text.  ``run_training_arm``
    already returns an eval-mode model; this is the idempotent belt-and-braces
    at the evaluation boundary, because a trained arm is not the only thing the
    campaign evaluates.
    """
    from ..training.model_loading import set_evaluation_mode

    model = getattr(bundle, "model", None)
    if model is None:
        return {"skipped": "bundle carries no model"}
    return set_evaluation_mode(model)


def _run_dev_eval(ctx: StageContext, bundle: Any, episodes: Sequence[Any],
                  arm_tag: str) -> list[dict]:
    from ..evaluation.learning_curve import (LearningCurveConfig,
                                             annotate_records, run_episodes)

    prepare_bundle_for_evaluation(bundle)
    records = run_episodes(bundle, episodes, env_factory,
                           cfg=LearningCurveConfig(
                               arm_tag=arm_tag,
                               generation=_generation_config(ctx)))
    return annotate_records(records, split="DEVELOPMENT")


def fl0_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """FL0: base model, no training; the four frozen cells on DEVELOPMENT."""
    from ..evaluation.fl0_base import run_fl0_base

    episodes = _dev_episodes(ctx, stage)
    ctx.checkpoint("fl0:model_load")
    bundle = ctx.bundle_factory()
    prepare_bundle_for_evaluation(bundle)
    # Contract §9 reports UNSEEN-INSTANCE and UNSEEN-FAMILY separately, which
    # is only computable against the frozen TRAIN family list.  FL0 is
    # untrained, so every DEVELOPMENT family is an unseen FAMILY here; passing
    # the real TRAIN list (rather than an empty one, which would make the
    # distinction vacuous) is what makes FL0 comparable with the trained arms.
    trained_family_ids = list(split_families("TRAIN"))
    ctx.checkpoint("fl0:cells")
    outcome = run_fl0_base(bundle, episodes, env_factory,
                           out_dir=ctx.stage_dir(stage.stage_id),
                           arm_tag="FL0_BASE",
                           generation=_generation_config(ctx),
                           run_gate=bool(ctx.extra.get("run_equivalence_gate", True)),
                           trained_family_ids=trained_family_ids)
    from ..evaluation.learning_curve import annotate_records

    records = outcome["records_by_cell"].get("structured__history", [])
    records = annotate_records(records, split="DEVELOPMENT")
    ctx.results["FL0"] = {
        "report": outcome["report"],
        "dev": _dev_metrics(records, "FL0").to_dict(),
        "trained_family_ids": trained_family_ids,
        "trained_family_ids_source": (
            "the frozen §5 TRAIN split; FL0 itself is untrained, so the list "
            "only fixes the UNSEEN-INSTANCE / UNSEEN-FAMILY partition"),
        "eval_plan": ctx.results.get("_eval_plans", {}).get(stage.stage_id),
    }
    ctx.results.setdefault("_records", {})["FL0"] = records
    return ctx.results["FL0"]


def _trainer_hooks(ctx: StageContext, stage_id: str,
                   arm_out_dir: str | None = None):
    """Scheduler heartbeat + checkpoint notification (+ off-pod checkpoint
    mirroring under interruptible capacity), when a scheduler exists."""
    from ..training.checkpointing import checkpoint_paths
    from ..training.trainer import TrainerHooks

    if ctx.scheduler is None:
        return None
    mirror = ctx.extra.get("durability_mirror")

    def _on_checkpoint(tag, manifest):
        ctx.scheduler.journal(
            "CHECKPOINT_WRITTEN",
            {"stage_id": stage_id, "tag": tag,
             "manifest_hash": manifest.get("manifest_hash")})
        if mirror is not None and arm_out_dir is not None:
            payload_path, manifest_path = checkpoint_paths(arm_out_dir, tag)
            try:
                mirror.push_checkpoint(payload_path, manifest_path)
            except Exception as exc:  # noqa: BLE001 - loud, never fatal:
                # generation continues on local atomic state; the eviction
                # loss window is widened and journalled
                ctx.scheduler.journal(
                    "CHECKPOINT_MIRROR_FAILED",
                    {"stage_id": stage_id, "tag": tag,
                     "error": str(exc)[:200]})

    return TrainerHooks(
        on_step=ctx.scheduler.heartbeat_hook(stage_id, watchdog=ctx.watchdog),
        on_checkpoint=_on_checkpoint)


def _train_arm(ctx: StageContext, arm_id: str, *, updates: int, stage_name: str,
               learning_rate: float, out_dir: str) -> dict:
    from ..training.trainer import run_training_arm

    ctx.checkpoint(f"{arm_id}:model_load")
    bundle = ctx.bundle_factory()          # FRESH load per arm (contract §7)
    examples = build_arm_examples(ctx, arm_id, bundle.tokenizer)
    cfg = _arm_config(ctx, arm_id, updates=updates, stage=stage_name,
                      learning_rate=learning_rate)
    if ctx.scheduler is not None:
        # the frozen §11 cadence (600 s / 200 steps) must reach the trainer
        ctx.scheduler.assert_checkpoint_cadence(cfg)
    ctx.checkpoint(f"{arm_id}:train")
    result = run_training_arm(cfg, bundle, examples, out_dir,
                              _trainer_hooks(ctx, arm_id, out_dir))
    return {"result": result, "bundle": bundle, "config": cfg,
            "out_dir": out_dir}


def dev_grid_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """The frozen 2-LR FL3 grid at 25 % U, its selection, and the freeze."""
    from ..training.arms import STAGE_GRID
    from . import dev_selector

    if ctx.updates is None:
        raise StageError("DEV_GRID needs the affordability-selected U")
    out_root = ctx.stage_dir(stage.stage_id)
    grid_lrs = dev_selector.frozen_grid(ctx.scope)
    n_updates = dev_selector.expected_grid_updates(ctx.updates)
    episodes = _dev_episodes(ctx, stage)
    runs = []
    reports = []
    for lr in grid_lrs:
        out_dir = os.path.join(out_root, f"lr_{lr:g}")
        trained = _train_arm(ctx, "FL3", updates=n_updates, stage_name=STAGE_GRID,
                             learning_rate=lr, out_dir=out_dir)
        records = _run_dev_eval(ctx, trained["bundle"], episodes,
                                arm_tag=f"FL3_GRID_lr{lr:g}")
        dev = _dev_metrics(records, "FL3_GRID")
        runs.append(dev_selector.GridRun(
            learning_rate=lr, scope=ctx.scope, updates=n_updates, dev=dev,
            arm_config_hash=trained["result"].arm_config_hash,
            checkpoint_tags=tuple(c["tag"] for c in trained["result"].checkpoints)))
        reports.append({"learning_rate": lr,
                        "arm_result": trained["result"].to_dict(),
                        "dev": dev.to_dict()})
    selection = dev_selector.select_learning_rate(
        runs, scope=ctx.scope, updates_U=ctx.updates)
    ctx.learning_rate = selection.chosen_learning_rate

    hashes = dict(ctx.extra.get("campaign_hashes", {}))
    hashes.setdefault("grid_arm_config_hashes",
                      [r.arm_config_hash for r in runs])
    stage_states = dict(ctx.extra.get("stage_states", {}))
    stage_states.setdefault("BENCH", "COMPLETE" if "BENCH" in ctx.results else "SKIPPED")
    stage_states.setdefault("FL0", "COMPLETE" if "FL0" in ctx.results else "SKIPPED")
    stage_states["DEV_GRID"] = "COMPLETE"
    checkpoint_tags = {f"FL3_GRID_lr{r.learning_rate:g}": list(r.checkpoint_tags)
                       for r in runs}
    decisions_path = os.path.join(ctx.out_dir, sealed_gate.DEV_DECISIONS_NAME)
    decisions = dev_selector.freeze_dev_decisions(
        decisions_path, selection=selection, updates_U=ctx.updates,
        stage_states=stage_states, checkpoint_tags=checkpoint_tags,
        hashes=hashes, runs=runs, guard=ctx.guard)
    payload = {
        "schema": "flb200.dev_grid_report.v1",
        "selection": selection.to_dict(),
        "runs": reports,
        "dev_decisions_path": decisions_path,
        "dev_decisions_sha256": decisions["dev_decisions_sha256"],
    }
    ctx.write(stage.stage_id, "dev_grid_report.json", payload)
    ctx.results["DEV_GRID"] = payload
    return payload


def core_arm_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """One matched core arm (FL1 / FL2 / FL3) plus its DEVELOPMENT evaluation."""
    from ..training.arms import STAGE_CORE

    arm_id = stage.arm_id or stage.stage_id
    if ctx.updates is None:
        raise StageError(f"{arm_id}: no U selected")
    if ctx.learning_rate is None:
        raise StageError(
            f"{arm_id}: no learning rate; the DEV_GRID stage must select it "
            "before any core arm runs (contract §8)")
    out_dir = os.path.join(ctx.stage_dir(stage.stage_id), "arm")
    trained = _train_arm(ctx, arm_id, updates=int(ctx.updates),
                         stage_name=STAGE_CORE,
                         learning_rate=float(ctx.learning_rate), out_dir=out_dir)
    episodes = _dev_episodes(ctx, stage)
    ctx.checkpoint(f"{arm_id}:dev_eval")
    records = _run_dev_eval(ctx, trained["bundle"], episodes, arm_tag=arm_id)
    dev = _dev_metrics(records, arm_id,
                       stability_events=[e.get("kind", "EVENT")
                                         for e in trained["result"].stability_events])
    payload = {
        "schema": "flb200.core_arm_report.v1",
        "arm_id": arm_id,
        "arm_result": trained["result"].to_dict(),
        "dev": dev.to_dict(),
        "eval_plan": ctx.results.get("_eval_plans", {}).get(stage.stage_id),
    }
    ctx.write(stage.stage_id, f"arm_result_{arm_id}.json", payload)
    ctx.results[arm_id] = payload
    ctx.results.setdefault("_records", {})[arm_id] = records
    ctx.results.setdefault("_dev_metrics", {})[arm_id] = dev
    # kept for the CORE_MATCHING audit, which needs the real ArmConfig objects
    # and the checkpoint directory of every core arm
    ctx.results.setdefault("_arm_configs", {})[arm_id] = trained["config"]
    ctx.results.setdefault("_arm_out_dirs", {})[arm_id] = trained["out_dir"]
    return payload


def core_matching_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """Compute matching + shared base identity across the completed core arms.

    Contract §8 forbids hidden unequal compute and §1/§15 require every arm to
    start from the identical frozen checkpoint.  The three checks that enforce
    this (``training.arms.assert_core_arms_matched``,
    ``training.compute_accounting.compare_core_ledgers`` and the base-identity
    comparison) existed but were never invoked outside the test suite (V-Gap4);
    they run HERE, in the campaign, immediately after the comparison and before
    any extension consumes it.  A mismatch raises, which the scheduler records
    as a failed stage with its predeclared fallback.
    """
    from . import core_matching

    ctx.checkpoint("core_matching")
    report = core_matching.build_core_matching_report(ctx, CORE_ARM_STAGES)
    ctx.write(stage.stage_id, "core_matching_report.json", report)
    ctx.results["CORE_MATCHING"] = report
    return report


def second_seed_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """The second predeclared seed: the same core comparison, seed 20260810.

    ``PREDECLARED_NOT_RUN`` is a DECLARATION, not a result and not a silent
    skip (Amendment 12): the whole campaign — pre-generated data, split
    manifest, episode seeds, RNG substreams — is a pure function of the root
    seed, so a second seed is a second CAMPAIGN RUN launched with
    ``--seed 20260810`` against its own pre-generated root, not a stage that
    can mutate this run's frozen seed midway.  The stage exists in the table so
    that §11 priority 9 ("additional predeclared seeds") is explicit, the
    admission arithmetic covers it, and the journal records whether the time
    for it existed.
    """
    from ..training.arms import ROOT_SEED_SECONDARY

    if int(ctx.root_seed) == int(ROOT_SEED_SECONDARY):
        raise StageError("the campaign is already running the second seed")
    payload = {
        "schema": "flb200.second_seed_declaration.v1",
        "status": "PREDECLARED_NOT_RUN",
        "note": ("the second predeclared seed repeats the frozen core "
                 "comparison with root seed 20260810; it is launched as a "
                 "fresh campaign run with --seed 20260810 rather than "
                 "mutating this run's frozen seed"),
        "launch": ("python -m foundation_learner.scripts.pregenerate_all "
                   "--seed 20260810 --out artifacts_fl/pregen_seed20260810, "
                   "then a fresh session with root_seed=20260810"),
        "this_run_root_seed": int(ctx.root_seed),
        "seed": int(ROOT_SEED_SECONDARY),
    }
    ctx.write(stage.stage_id, "second_seed_report.json", payload)
    ctx.results["SECOND_SEED"] = payload
    return payload


# -- the promoted candidate ------------------------------------------------

def promoted_arm_id(ctx: StageContext) -> str:
    """Which trained arm the sealed set (and the diagnostics) evaluate.

    Mechanical and frozen BEFORE the sealed opening: ``ctx.extra`` may pin a
    core arm explicitly (recorded in ``DEV_DECISIONS_FROZEN.json`` when the
    development decisions are frozen), otherwise it is the frozen core
    treatment FL3.  It is never chosen by looking at a sealed outcome, and it
    is never the untrained base.
    """
    declared = ctx.extra.get("promoted_arm")
    if declared is None:
        return DEFAULT_PROMOTED_ARM
    if str(declared) not in CORE_ARM_STAGES:
        raise StageError(
            f"promoted_arm={declared!r} is not one of the core arms "
            f"{list(CORE_ARM_STAGES)}; the sealed set evaluates a core arm")
    return str(declared)


def promoted_arm_bundle(ctx: StageContext, *, require: bool) -> tuple[Any, dict]:
    """A fresh bundle carrying the promoted arm's FINAL trained state.

    Contract §7 hands out a FRESH BASE bundle by design, so the arm's trained
    weights must be restored explicitly — otherwise the "evaluation of the
    promoted candidate" silently evaluates the untrained base model (R-C4).
    ``require=True`` (the sealed opening) REFUSES when the checkpoint is
    absent; ``require=False`` (the diagnostics) records the fallback to the
    base checkpoint in the payload instead.
    """
    arm_id = promoted_arm_id(ctx)
    bundle = ctx.bundle_factory()
    try:
        from ..mechanisms.stage_support import arm_checkpoint_state
    except ImportError as exc:      # pragma: no cover - shipped package has it
        raise StageError(
            "restoring the promoted arm needs "
            "foundation_learner.mechanisms.stage_support.arm_checkpoint_state, "
            f"which is not importable here ({exc}); the campaign refuses to "
            "evaluate an untrained base model as the promoted candidate"
        ) from exc

    record = dict(arm_checkpoint_state(ctx, bundle, arm_id, tag="final"))
    # restoring trained weights can leave the module in whatever mode the
    # restoring path used; evaluation is only valid in eval mode with
    # layer-level checkpointing off (Amendment 16)
    record["evaluation_mode"] = prepare_bundle_for_evaluation(bundle)
    record["promoted_arm"] = arm_id
    record["rule"] = ("the frozen core treatment, or the core arm pinned in the "
                      "development decisions; never selected from a sealed "
                      "outcome")
    if not record.get("loaded"):
        if require:
            raise StageError(
                f"REFUSED: no {arm_id!r} 'final' checkpoint is available "
                f"({record.get('reason')}). The sealed set evaluates the "
                "FROZEN promoted candidate; evaluating a fresh base model and "
                "calling it the promoted arm would misdescribe the result "
                "(contract §12, Amendment 12).")
        record["note"] = ("the promoted arm's checkpoint was unavailable; this "
                          "diagnostic ran on the BASE checkpoint and says so")
    return bundle, record


# -- unconditional post-core diagnostics (metrics 11, 12, 13) --------------

def _diag_head(ctx: StageContext, stage: StageDefinition) -> tuple[Any, dict, list]:
    episodes = _dev_episodes(ctx, stage)
    ctx.checkpoint(f"{stage.stage_id}:model_load")
    bundle, model_record = promoted_arm_bundle(ctx, require=False)
    return bundle, model_record, episodes


def _diag_payload(ctx: StageContext, stage: StageDefinition, *,
                  model_record: dict, **fields: Any) -> dict:
    payload = {
        "schema": "flb200.diagnostic_report.v1",
        "stage": stage.stage_id,
        "status": "COMPLETE",
        "model": model_record,
        "split": "DEVELOPMENT",
        "eval_plan": ctx.results.get("_eval_plans", {}).get(stage.stage_id),
        "rehearsal": bool(ctx.rehearsal),
        "unconditional": ("this diagnostic has no promotion entry condition; "
                          "it also runs under the §10 FL3-null fallback"),
    }
    payload.update(fields)
    ctx.write(stage.stage_id, stage.outputs[0], payload)
    ctx.results[stage.stage_id] = payload
    return payload


def remap_diag_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """Metric 13: surface-remap robustness on DEVELOPMENT (contract §9)."""
    from ..evaluation.learning_curve import LearningCurveConfig
    from ..evaluation.remap_eval import pairs_from_remap_records, run_remap_eval

    bundle, model_record, episodes = _diag_head(ctx, stage)
    wanted = {ep.episode_id for ep in episodes}
    rows = [r for r in load_side_table(ctx, "remaps", "DEVELOPMENT")
            if str(r.get("episode_id")) in wanted]
    pairs = [p for p in pairs_from_remap_records(episodes, rows) if p.variants]
    if not pairs:
        raise StageError(
            "REMAP_DIAG: the pre-generated remap table covers none of the "
            "assembled DEVELOPMENT episodes; metric 13 cannot be computed")
    ctx.checkpoint("remap_diag:walk")
    report = run_remap_eval(
        bundle, pairs, env_factory,
        cfg=LearningCurveConfig(arm_tag="REMAP_DIAG",
                                generation=_generation_config(ctx)))
    return _diag_payload(
        ctx, stage, model_record=model_record,
        n_pairs=len(pairs), variants=report["variants"],
        robustness=report["robustness"],
        metric="13 (remap robustness: remapped vs canonical AULC)")


def interference_diag_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """Metric 11: A->B->A retention / interference (contract §9)."""
    from ..evaluation.interference import run_interference_eval
    from ..evaluation.learning_curve import LearningCurveConfig

    ctx.checkpoint("interference_diag:model_load")
    bundle, model_record = promoted_arm_bundle(ctx, require=False)
    # the chains name specific episodes, so the WHOLE DEVELOPMENT pool is the
    # resolution set here; its family coverage is asserted and recorded exactly
    # like every other evaluation set
    pool = load_episodes(ctx.pregen_root, "DEVELOPMENT", "scripted",
                         guard=ctx.guard)
    coverage = assert_eval_family_coverage(
        pool, "DEVELOPMENT", context="INTERFERENCE_DIAG episode pool")
    n_chains = ctx.eval_cap(stage)
    ctx.results.setdefault("_eval_plans", {})[stage.stage_id] = {
        "plan": {"split": "DEVELOPMENT", "unit": "A->B->A chains",
                 "chains_requested": int(n_chains),
                 "families": list(split_families("DEVELOPMENT")),
                 "n_families": len(split_families("DEVELOPMENT")),
                 "rule": ("chains are drawn ROUND-ROBIN over the ordered family "
                          "pairs; the interference metric is family-balanced "
                          "(contract §9)")},
        "coverage": coverage}
    chains = load_chain_specs(ctx, pool, limit=n_chains)
    if not chains:
        raise StageError(
            "INTERFERENCE_DIAG: no pre-generated A->B->A chain resolved against "
            "the DEVELOPMENT episode pool; metric 11 cannot be computed")
    ctx.checkpoint("interference_diag:walk")
    report = run_interference_eval(
        bundle, chains, env_factory,
        cfg=LearningCurveConfig(arm_tag="INTERFERENCE_DIAG",
                                generation=_generation_config(ctx)))
    return _diag_payload(
        ctx, stage, model_record=model_record,
        n_chains=len(chains),
        pair_counts=report.get("pair_counts"),
        summary=report.get("summary"),
        chains=report.get("chains"),
        metric="11 (A->B->A retention, interference cost, recovery)")


def poison_diag_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """Metric 12: poisoned-feedback robustness (contract §9).

    Independent of FL6: this measures BLIND incorporation of poisoned hints by
    the promoted arm, which is defined whether or not a value gate exists.
    """
    from ..evaluation.learning_curve import LearningCurveConfig
    from ..evaluation.poison_eval import (episodes_from_poison_record,
                                          run_poison_eval)

    bundle, model_record, episodes = _diag_head(ctx, stage)
    records = {str(r["episode_id"]): r
               for r in load_side_table(ctx, "poison", "DEVELOPMENT")}
    variants: list[Any] = []
    for episode in episodes:
        record = records.get(str(episode.episode_id))
        if record is None:
            continue
        variants.extend(episodes_from_poison_record(episode, record).values())
    if not variants:
        raise StageError(
            "POISON_DIAG: the pre-generated poison table covers none of the "
            "assembled DEVELOPMENT episodes; metric 12 cannot be computed")
    ctx.checkpoint("poison_diag:walk")
    report = run_poison_eval(
        bundle, variants, env_factory,
        cfg=LearningCurveConfig(arm_tag="POISON_DIAG",
                                generation=_generation_config(ctx)))
    return _diag_payload(
        ctx, stage, model_record=model_record,
        conditions_present=report["conditions_present"],
        n_episodes_by_condition=report["n_episodes_by_condition"],
        rendering_audit=report["rendering_audit"],
        robustness=report["robustness"],
        metric="12 (poison robustness: poisoned vs clean AULC)")


# -- mechanism rungs (W3); feature-detected -------------------------------

def _mechanism_stage(ctx: StageContext, stage: StageDefinition,
                     entry_point: str) -> dict:
    """Run a mechanism rung, or record SKIPPED_MECHANISMS when absent."""
    missing = [m for m in stage.requires_modules if not module_available(m)]
    if missing:
        payload = {
            "schema": "flb200.mechanism_stage.v1",
            "stage": stage.stage_id,
            "status": "SKIPPED_MECHANISMS",
            "missing_modules": missing,
            "note": ("the mechanisms package is not importable in this "
                     "runtime; the rung is recorded as skipped, never as a "
                     "null result"),
        }
        ctx.write(stage.stage_id, stage.outputs[0], payload)
        ctx.results[stage.stage_id] = payload
        return payload
    run = resolve_dotted(entry_point)     # present module without the entry
    payload = run(ctx, stage)             # point is an error, not a skip
    ctx.results[stage.stage_id] = payload
    return payload


def fl4_work(ctx: StageContext, stage: StageDefinition) -> dict:
    return _mechanism_stage(
        ctx, stage, "foundation_learner.mechanisms.value_head:run_fl4_stage")


def fl5_work(ctx: StageContext, stage: StageDefinition) -> dict:
    return _mechanism_stage(
        ctx, stage, "foundation_learner.mechanisms.fast_state:run_fl5_stage")


def fl6_work(ctx: StageContext, stage: StageDefinition) -> dict:
    return _mechanism_stage(
        ctx, stage, "foundation_learner.mechanisms.value_gating:run_fl6_stage")


def fl7_work(ctx: StageContext, stage: StageDefinition) -> dict:
    return _mechanism_stage(
        ctx, stage, "foundation_learner.mechanisms.fast_adapter:run_fl7_stage")


def fl8_work(ctx: StageContext, stage: StageDefinition) -> dict:
    return _mechanism_stage(
        ctx, stage, "foundation_learner.mechanisms.consolidation:run_fl8_stage")


# -- sealed opening --------------------------------------------------------

def _sealed_episodes(ctx: StageContext, stage: StageDefinition,
                     unlock: sealed_gate.SealedUnlock) -> list[Any]:
    """Per-SHARD sliced sealed evaluation set, family coverage asserted.

    Slicing the concatenation would evaluate the sealed set on whichever sealed
    family sorts first (R-C3); the cap is divided across the sealed families
    and applied per shard, exactly as on DEVELOPMENT.
    """
    from ..episodes.schema import Episode

    plan = eval_plan(ctx, stage, split="SEALED_TEST")
    episodes: list[Any] = []
    for path in _episode_shards(ctx.pregen_root, "SEALED_TEST", "scripted",
                                ctx.guard):
        rows = unlock.read_shard(path)[:plan["limit_per_family"]]
        episodes.extend(Episode.from_dict(row) for row in rows)
    coverage = assert_eval_family_coverage(
        episodes, "SEALED_TEST", context="SEALED_EVAL evaluation set")
    ctx.results.setdefault("_eval_plans", {})[stage.stage_id] = {
        "plan": plan, "coverage": coverage}
    return episodes


def sealed_eval_work(ctx: StageContext, stage: StageDefinition) -> dict:
    """The single sealed opening and the sealed evaluation (contract §12).

    Two changes the review made non-negotiable:

    * the evaluation runs the FROZEN PROMOTED CANDIDATE, restored from its
      recorded ``final`` checkpoint, and records which arm and which checkpoint
      manifest hash it was.  Evaluating a fresh base bundle here would have
      measured the untrained backbone and reported it as the campaign's sealed
      result (R-C4);
    * the opening is TWO-PHASE: the ledger's single ``SEALED_OPENED`` entry is
      written only once the evaluation records exist, so an exception cannot
      burn the seal without producing evidence.  A failed attempt is recorded
      as ``SEALED_OPENING_ABORTED`` and leaves exactly one retry (R-C5).
    """
    from ..evaluation.learning_curve import (LearningCurveConfig,
                                             annotate_records, run_episodes)
    from ..evaluation import metrics as _metrics
    from ..evaluation.family_holdout import build_family_holdout_report

    out_dir = ctx.stage_dir(stage.stage_id)
    ctx.checkpoint("sealed:promoted_arm")
    bundle, model_record = promoted_arm_bundle(ctx, require=True)
    # The LAST watchdog checkpoint before the seal.  Nothing inside the opening
    # may raise StageAbortedOverrun: a timing abort there would consume one of
    # the two permanent opening attempts for a reason that has nothing to do
    # with the sealed evaluation (Amendment 16, N2).
    ctx.checkpoint("sealed:before_opening")

    with sealed_gate.sealed_opening(
            ledger_path=os.path.join(ctx.out_dir, sealed_gate.LEDGER_NAME),
            dev_decisions_path=os.path.join(ctx.out_dir,
                                            sealed_gate.DEV_DECISIONS_NAME),
            split_manifest_path=os.path.join(ctx.pregen_root,
                                             "family_split_manifest.json"),
            stage_states=ctx.extra.get("stage_states"),
            guard=ctx.guard,
            opened_by=f"{ctx.label}:SEALED_EVAL") as unlock:
        episodes = _sealed_episodes(ctx, stage, unlock)
        records = run_episodes(bundle, episodes, env_factory,
                               cfg=LearningCurveConfig(
                                   arm_tag=f"SEALED_EVAL_{model_record['promoted_arm']}",
                                   generation=_generation_config(ctx)))
        # the split tag goes on through the evaluation layer's own annotator,
        # which RE-HASHES the record: a hand-mutated dict would carry a
        # record_hash certifying content the record no longer has
        records = annotate_records(records, split="SEALED_TEST")
        if not records:
            raise StageError(
                "SEALED_EVAL produced no episode records; the opening is "
                "aborted rather than committed (contract §12 + Amendment 12)")
        trained_family_ids = list(split_families("TRAIN"))
        summary = _metrics.summarize(records, trained_family_ids)
        holdout = build_family_holdout_report(
            records, trained_family_ids,
            arm_tag=f"SEALED_EVAL_{model_record['promoted_arm']}",
            split="SEALED_TEST")
        # phase two: the evidence exists, so the seal may be consumed
        unlock.commit(evaluation={
            "n_episodes": len(episodes), "n_records": len(records),
            "promoted_arm": model_record["promoted_arm"],
            "checkpoint_manifest_hash": model_record.get("manifest_hash"),
            "arm_tag": f"SEALED_EVAL_{model_record['promoted_arm']}"})
        unlock.write_result_records(
            os.path.join(out_dir, "sealed_records.jsonl"), records)
        report = {
            "schema": "flb200.sealed_eval_report.v1",
            "n_episodes": len(episodes),
            "promoted_arm": model_record["promoted_arm"],
            "promoted_arm_checkpoint": model_record,
            "eval_plan": ctx.results.get("_eval_plans", {}).get(stage.stage_id),
            "summary": summary,
            "family_holdout": holdout,
            "unlock": unlock.to_dict(),
            "policy": ("no model modification may be justified from these "
                       "outcomes; a further cycle requires a NEW sealed set"),
            "claim_scope": (
                "3 sealed family clusters: intervals are conditional on these "
                "three held-out generators and are never a population-level "
                "unseen-family claim (preregistration §13.1)"),
        }
        unlock.write_result(os.path.join(out_dir, "sealed_eval_report.json"),
                            report)
        unlock.revoke()
    ctx.results["SEALED_EVAL"] = report
    return report


# --------------------------------------------------------------------------
# entry conditions (all promotion truth comes from campaign.promotion)
# --------------------------------------------------------------------------

def _core_dev(ctx: StageContext, arm_id: str) -> promotion.DevMetrics | None:
    return ctx.results.get("_dev_metrics", {}).get(arm_id)


def fl3_extension_gate(ctx: StageContext) -> promotion.PromotionDecision:
    fl3 = _core_dev(ctx, "FL3")
    fl1 = _core_dev(ctx, "FL1")
    if fl3 is None or fl1 is None:
        return promotion.PromotionDecision(
            stage="FL3_TO_EXTENSIONS", admitted=False,
            reasons=("FAIL: the core comparison has not produced DEV metrics "
                     "for both FL3 and FL1",),
            fallback=tuple(promotion.fallback_for("FL3_TO_EXTENSIONS")))
    return promotion.promote_fl3_to_extensions(fl3, fl1)


def fl4_entry(ctx: StageContext) -> promotion.PromotionDecision:
    gate = fl3_extension_gate(ctx)
    if not gate.admitted:
        return gate
    counts = ctx.extra.get("fl4_scoreable_items_per_episode", [])
    return promotion.promote_fl4(fl3_finished="FL3" in ctx.results,
                                 scoreable_items_per_episode=counts)


def fl5_entry(ctx: StageContext) -> promotion.PromotionDecision:
    gate = fl3_extension_gate(ctx)
    if not gate.admitted:
        return gate
    complete = all(arm in ctx.results for arm in CORE_ARM_STAGES)
    return promotion.promote_fl5(core_comparison_complete=complete,
                                 scheduler_admits=True)


def fl6_entry(ctx: StageContext) -> promotion.PromotionDecision:
    fl4 = ctx.results.get("FL4") or {}
    accuracy = fl4.get("pairwise_ranking_accuracy")
    if accuracy is None:
        return promotion.PromotionDecision(
            stage="FL6", admitted=False,
            reasons=("FAIL: no FL4 DEV pairwise ranking accuracy is available",),
            fallback=tuple(promotion.fallback_for("FL6")))
    return promotion.promote_fl6(pairwise_ranking_accuracy=float(accuracy))


def fl7_entry(ctx: StageContext) -> promotion.PromotionDecision:
    fl4 = ctx.results.get("FL4") or {}
    return promotion.promote_fl7(
        scheduler_admits=True,
        fast_update_stable=bool(ctx.extra.get("fast_update_stable", False)),
        fl4_succeeded=bool(fl4.get("status") == "COMPLETE"),
        only_remaining_variant_is_fl6_gated=bool(
            ctx.extra.get("only_fl6_gated_variant_remains", False)))


def sealed_eval_entry(ctx: StageContext) -> promotion.PromotionDecision:
    """The sealed set opens only after a COMPLETE core comparison (R-C4).

    Contract §8 makes the sealed evaluation the final test of the core
    comparison, and §12 makes it a single opening.  Opening it with an
    incomplete comparison would spend the one opening on a result that cannot
    be interpreted, and (before this gate existed) on whatever model the stage
    happened to hold.  SEALED_TEST still informs NOTHING: this is an entry
    condition on the campaign's own state, not on any sealed outcome.
    """
    arm_id = promoted_arm_id(ctx)
    missing = [arm for arm in CORE_ARM_STAGES if arm not in ctx.results]
    decisions_path = os.path.join(ctx.out_dir, sealed_gate.DEV_DECISIONS_NAME)
    frozen = os.path.isfile(ctx.guard.guard(decisions_path,
                                            o1_isolation.MODE_READ))
    checkpointed = os.path.isfile(ctx.guard.guard(
        os.path.join(ctx.out_dir, arm_id.lower(), "arm",
                     "ckpt_final.manifest.json"), o1_isolation.MODE_READ))
    checks = (
        (f"the core comparison completed ({', '.join(CORE_ARM_STAGES)})",
         not missing),
        (f"{sealed_gate.DEV_DECISIONS_NAME} is frozen", frozen),
        (f"the promoted arm {arm_id} has a 'final' checkpoint", checkpointed),
    )
    reasons = tuple(f"{'PASS' if ok else 'FAIL'}: {label}"
                    for label, ok in checks)
    admitted = all(ok for _, ok in checks)
    return promotion.PromotionDecision(
        stage="SEALED_EVAL", admitted=admitted, reasons=reasons,
        evidence={"promoted_arm": arm_id, "missing_core_arms": missing,
                  "dev_decisions_frozen": frozen,
                  "promoted_arm_checkpointed": checkpointed},
        fallback=() if admitted else ("leave the sealed set unopened",
                                      "a further cycle requires a NEW sealed "
                                      "set; nothing is gained by opening it "
                                      "without the comparison it tests"))


def fl8_entry(ctx: StageContext) -> promotion.PromotionDecision:
    evidence = ctx.extra.get("persistence_evidence")
    if not evidence:
        return promotion.PromotionDecision(
            stage="FL8", admitted=False,
            reasons=("FAIL: no DEV context-reset persistence evidence",),
            fallback=tuple(promotion.fallback_for("FL8")))
    return promotion.promote_fl8(
        persistence_point=float(evidence["point"]),
        persistence_ci=evidence["ci"])
