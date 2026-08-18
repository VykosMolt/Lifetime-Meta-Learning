"""Frozen affordability arithmetic (contract §11).

Pure functions over measured numbers.  Nothing here loads a model, touches a
clock, or spends money; everything is exercised by unit tests with explicit
inputs, which is what makes the two frozen rules auditable:

**Admission rule** (every stage, including BENCH)::

    projected_stage_seconds * SAFETY_FACTOR + remaining_reserve
        <= remaining_authorized

**Scope + U rule** (mechanical, never outcome-based)::

    FULL_MODEL_MODE is eligible iff
        projected_core_comparison_seconds(FULL, U_min = 600 updates/arm,
                                          grid + 3 arms + evals)
            * SAFETY_FACTOR + FINAL_TRANSFER_RESERVE
        <= available_foundation_learner_seconds
    otherwise PEFT_MODE for all core arms.
    U = largest value of {600, 1200, 2400, 4800} satisfying the same
    inequality for the SELECTED scope.

The constants live in ``FL_BUDGET_POLICY.json``; :func:`load_policy` refuses a
policy whose frozen fields were altered, exactly as the O1 runner does for its
own (separate, non-reusable) policy.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "BudgetPolicyError",
    "AffordabilityRefusal",
    "POLICY_PATH",
    "POLICY_SCHEMA",
    "SAFETY_FACTOR",
    "FINAL_TRANSFER_RESERVE",
    "BENCH_DECLARED_BUDGET_SECONDS",
    "STAGE_WATCHDOG_MINIMUM_SECONDS",
    "UPDATE_LADDER",
    "FULL_MODEL_MIN_UPDATES",
    "PEFT_MODE",
    "FULL_MODEL_MODE",
    "SCOPES",
    "load_policy",
    "policy_sha256",
    "BenchMeasurement",
    "ThroughputTable",
    "admission_check",
    "projected_eval_seconds",
    "projected_arm_seconds",
    "projected_core_comparison_seconds",
    "full_model_eligible",
    "largest_affordable_updates",
    "CorePlan",
    "plan_core_comparison",
]

POLICY_SCHEMA = "flb200.budget_policy.v1"
POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "FL_BUDGET_POLICY.json")

#: Contract §11 frozen constants (cross-checked against the policy file).
SAFETY_FACTOR = 1.25
FINAL_TRANSFER_RESERVE = 1200.0
UPDATE_LADDER: tuple[int, ...] = (600, 1200, 2400, 4800)
FULL_MODEL_MIN_UPDATES = 600
#: BENCH is the only stage that cannot be projected from a measurement (it IS
#: the measurement); it is admitted against this DECLARED bounded budget.
BENCH_DECLARED_BUDGET_SECONDS = 900.0
#: Stage watchdog floor (Amendment 12).  A stage is aborted past
#: ``max(projected * safety_factor, this)``; the hard money guarantee is the
#: separate reserve condition, which has no floor.
STAGE_WATCHDOG_MINIMUM_SECONDS = 900.0
CORE_ARMS = 3
GRID_CONFIGURATIONS = 2
GRID_UPDATE_FRACTION = 0.25

PEFT_MODE = "PEFT_MODE"
FULL_MODEL_MODE = "FULL_MODEL_MODE"
SCOPES = (PEFT_MODE, FULL_MODEL_MODE)


class BudgetPolicyError(RuntimeError):
    """The frozen budget policy is missing, malformed, or altered."""


class AffordabilityRefusal(RuntimeError):
    """A frozen affordability rule refused the requested work."""


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------

_FROZEN_NUMERIC = {
    "safety_factor": SAFETY_FACTOR,
    "final_transfer_reserve_seconds": FINAL_TRANSFER_RESERVE,
    "bench_declared_budget_seconds": BENCH_DECLARED_BUDGET_SECONDS,
    "stage_watchdog_minimum_seconds": STAGE_WATCHDOG_MINIMUM_SECONDS,
    "checkpoint_cadence_seconds": 600.0,
    "checkpoint_cadence_steps": 200.0,
    "full_model_minimum_updates_per_arm": float(FULL_MODEL_MIN_UPDATES),
    "core_arms": float(CORE_ARMS),
    "grid_configurations": float(GRID_CONFIGURATIONS),
    "grid_update_fraction": GRID_UPDATE_FRACTION,
    "session_total_authorized_usd": 45.0,
}

_FROZEN_FALSE = ("confirmation_authorized", "automatic_extension_allowed",
                 "operator_override_allowed", "reserve_consumable_by_stages")


def load_policy(path: str | None = None, *, guard: Any = None) -> dict:
    """Load and validate ``FL_BUDGET_POLICY.json``.

    Every frozen field is re-checked; an altered policy is an error, never a
    silently accepted new setting.  The read ALWAYS goes through an O1
    isolation guard — the default guard when none is supplied — so that no
    campaign file read bypasses §13's realpath refusal (V-Obs7).
    """
    if guard is None:
        from . import o1_isolation

        guard = o1_isolation.default_guard()
    target = guard.guard(os.path.abspath(path or POLICY_PATH), "read")
    try:
        with open(target, encoding="utf-8") as fh:
            policy = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise BudgetPolicyError(f"cannot read budget policy {target!r}: {exc}") from exc
    if policy.get("schema") != POLICY_SCHEMA:
        raise BudgetPolicyError(
            f"{target}: schema {policy.get('schema')!r} != {POLICY_SCHEMA!r}")
    for key, want in _FROZEN_NUMERIC.items():
        if key not in policy:
            raise BudgetPolicyError(f"{target}: missing frozen field {key!r}")
        if float(policy[key]) != float(want):
            raise BudgetPolicyError(
                f"{target}: frozen field {key} altered from {want} to {policy[key]}")
    for key in _FROZEN_FALSE:
        if policy.get(key) is not False:
            raise BudgetPolicyError(f"{target}: {key} must be false")
    if tuple(int(u) for u in policy.get("update_ladder", ())) != UPDATE_LADDER:
        raise BudgetPolicyError(f"{target}: update ladder altered from {UPDATE_LADDER}")
    if policy.get("rental_confirmation") != "NOT_AUTHORIZED":
        raise BudgetPolicyError(f"{target}: rental confirmation is not NOT_AUTHORIZED")
    if policy.get("o1_priority") != "ABSOLUTE_FIRST":
        raise BudgetPolicyError(f"{target}: O1 priority is frozen at ABSOLUTE_FIRST")
    return policy


def policy_sha256(path: str | None = None) -> str:
    from ..ecology.base import sha256_file

    return sha256_file(os.path.abspath(path or POLICY_PATH))


# --------------------------------------------------------------------------
# measured throughput
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchMeasurement:
    """One MEASURED throughput point from the ``BENCH`` stage.

    ``seconds_per_update`` is the quantity the projections use; it is measured,
    never assumed.  ``tokens_per_second`` is reported alongside because
    contract §11 names measured throughput as the BENCH output and §8 requires
    the token budget to be recorded.
    """

    scope: str
    seconds_per_update: float
    tokens_per_second: float
    updates_measured: int
    wall_seconds: float
    forward_tokens: int
    max_tokens_per_batch: int
    device: str = "unknown"
    eval_seconds_per_episode: float | None = None
    #: seconds for ONE fresh ``bundle_factory()`` load, measured in BENCH.
    #: Contract §7 gives every arm a fresh load, so this is a real per-stage
    #: cost that the projections previously ignored (R-M8).
    model_load_seconds: float | None = None
    #: seconds for ONE teacher-forced forward over one rendered episode,
    #: measured in BENCH.  This is the unit of the FL4 target computation
    #: (presence/ablation variants), which is not an optimizer update.
    forward_seconds_per_episode: float | None = None
    source: str = "MEASURED_BENCH"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {self.scope!r}")
        for name in ("seconds_per_update", "tokens_per_second", "wall_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if int(self.updates_measured) <= 0:
            raise ValueError("updates_measured must be positive")

    def to_dict(self) -> dict:
        return {
            "schema": "flb200.bench_measurement.v1",
            "scope": self.scope,
            "seconds_per_update": float(self.seconds_per_update),
            "tokens_per_second": float(self.tokens_per_second),
            "updates_measured": int(self.updates_measured),
            "wall_seconds": float(self.wall_seconds),
            "forward_tokens": int(self.forward_tokens),
            "max_tokens_per_batch": int(self.max_tokens_per_batch),
            "device": self.device,
            "eval_seconds_per_episode": (
                None if self.eval_seconds_per_episode is None
                else float(self.eval_seconds_per_episode)),
            "model_load_seconds": (None if self.model_load_seconds is None
                                   else float(self.model_load_seconds)),
            "forward_seconds_per_episode": (
                None if self.forward_seconds_per_episode is None
                else float(self.forward_seconds_per_episode)),
            "source": self.source,
            "notes": list(self.notes),
        }

    @staticmethod
    def from_ledger(ledger: Mapping[str, Any], scope: str, *,
                    eval_seconds_per_episode: float | None = None,
                    model_load_seconds: float | None = None,
                    forward_seconds_per_episode: float | None = None,
                    notes: Sequence[str] = ()) -> "BenchMeasurement":
        """Build a measurement from a W2 ``flb200.compute_ledger.v1`` payload."""
        updates = int(ledger.get("optimizer_updates") or 0) + int(
            ledger.get("skipped_updates") or 0)
        wall = float(ledger.get("wall_time_s") or 0.0)
        tokens = int(ledger.get("forward_tokens") or 0)
        if updates <= 0 or wall <= 0.0:
            raise ValueError(
                "benchmark ledger records no completed updates or no wall time; "
                "throughput cannot be projected from it")
        return BenchMeasurement(
            scope=scope,
            seconds_per_update=wall / updates,
            tokens_per_second=(tokens / wall) if tokens else 1e-9,
            updates_measured=updates,
            wall_seconds=wall,
            forward_tokens=tokens,
            max_tokens_per_batch=int(ledger.get("max_tokens_per_batch") or 0),
            device=str(ledger.get("device") or "unknown"),
            eval_seconds_per_episode=eval_seconds_per_episode,
            model_load_seconds=model_load_seconds,
            forward_seconds_per_episode=forward_seconds_per_episode,
            notes=tuple(notes),
        )


@dataclass
class ThroughputTable:
    """Measured throughput per trainable-parameter scope."""

    measurements: dict[str, BenchMeasurement] = field(default_factory=dict)

    def add(self, measurement: BenchMeasurement) -> None:
        self.measurements[measurement.scope] = measurement

    def get(self, scope: str) -> BenchMeasurement:
        if scope not in self.measurements:
            raise AffordabilityRefusal(
                f"no measured throughput for scope {scope!r}; the BENCH stage "
                "must measure a scope before any projection may use it "
                "(projections are never guessed)")
        return self.measurements[scope]

    def has(self, scope: str) -> bool:
        return scope in self.measurements

    def to_dict(self) -> dict:
        return {"schema": "flb200.throughput_table.v1",
                "measurements": {k: v.to_dict()
                                 for k, v in sorted(self.measurements.items())}}


# --------------------------------------------------------------------------
# the frozen rules
# --------------------------------------------------------------------------

def admission_check(*, projected_stage_seconds: float,
                    remaining_reserve_seconds: float,
                    remaining_authorized_seconds: float,
                    safety_factor: float = SAFETY_FACTOR,
                    stage: str = "stage",
                    strict: bool = True) -> dict:
    """The frozen §11 admission rule.

    Returns the decision record.  With ``strict`` (the default) a refusal
    raises :class:`AffordabilityRefusal`, so no caller can proceed by ignoring
    the return value.
    """
    values = {
        "projected_stage_seconds": projected_stage_seconds,
        "remaining_reserve_seconds": remaining_reserve_seconds,
        "remaining_authorized_seconds": remaining_authorized_seconds,
        "safety_factor": safety_factor,
    }
    for name, value in values.items():
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise AffordabilityRefusal(
                f"{name} must be finite and non-negative, got {value!r}")
    required = (float(projected_stage_seconds) * float(safety_factor)
                + float(remaining_reserve_seconds))
    admitted = required <= float(remaining_authorized_seconds)
    record = {
        "schema": "flb200.admission_decision.v1",
        "stage": stage,
        "projected_stage_seconds": float(projected_stage_seconds),
        "safety_factor": float(safety_factor),
        "remaining_reserve_seconds": float(remaining_reserve_seconds),
        "remaining_authorized_seconds": float(remaining_authorized_seconds),
        "required_seconds": required,
        "admitted": bool(admitted),
        "rule": ("projected_stage_seconds * safety_factor + remaining_reserve "
                 "<= remaining_authorized"),
    }
    if not admitted and strict:
        raise AffordabilityRefusal(
            f"admission REFUSED for stage {stage!r}: requires {required:.1f}s "
            f"(= {projected_stage_seconds:.1f} x {safety_factor} + "
            f"{remaining_reserve_seconds:.1f} reserve) but only "
            f"{float(remaining_authorized_seconds):.1f}s remain")
    return record


def projected_eval_seconds(bench: BenchMeasurement, n_episodes: int) -> float:
    """Projected evaluation seconds from the measured per-episode cost."""
    if bench.eval_seconds_per_episode is None:
        raise AffordabilityRefusal(
            "the BENCH stage did not measure an evaluation cost; evaluation "
            "seconds may not be guessed")
    if int(n_episodes) < 0:
        raise ValueError("n_episodes must be non-negative")
    return float(bench.eval_seconds_per_episode) * int(n_episodes)


def projected_arm_seconds(bench: BenchMeasurement, updates: int) -> float:
    """Projected training seconds for one arm of ``updates`` optimizer steps."""
    if int(updates) < 0:
        raise ValueError("updates must be non-negative")
    return float(bench.seconds_per_update) * int(updates)


def grid_updates(updates: int) -> int:
    """The frozen 25 % shortened step count of the development grid (§8)."""
    return max(1, int(round(GRID_UPDATE_FRACTION * int(updates))))


def projected_core_comparison_seconds(bench: BenchMeasurement, updates: int, *,
                                      eval_seconds: float,
                                      core_arms: int = CORE_ARMS,
                                      grid_configurations: int = GRID_CONFIGURATIONS
                                      ) -> dict:
    """Projected seconds for ``grid + 3 arms + evals`` at ``updates``/arm."""
    if float(eval_seconds) < 0.0 or not math.isfinite(float(eval_seconds)):
        raise ValueError("eval_seconds must be finite and non-negative")
    arms = float(core_arms) * projected_arm_seconds(bench, updates)
    grid = float(grid_configurations) * projected_arm_seconds(
        bench, grid_updates(updates))
    total = arms + grid + float(eval_seconds)
    return {
        "scope": bench.scope,
        "updates_per_arm": int(updates),
        "grid_updates_per_config": grid_updates(updates),
        "core_arm_seconds": arms,
        "grid_seconds": grid,
        "eval_seconds": float(eval_seconds),
        "total_seconds": total,
    }


def _fits(total_seconds: float, available_seconds: float,
          reserve: float = FINAL_TRANSFER_RESERVE,
          safety_factor: float = SAFETY_FACTOR) -> bool:
    return (total_seconds * safety_factor + reserve) <= available_seconds


def full_model_eligible(table: ThroughputTable, available_seconds: float, *,
                        eval_seconds_full: float,
                        reserve: float = FINAL_TRANSFER_RESERVE,
                        safety_factor: float = SAFETY_FACTOR) -> dict:
    """The frozen FULL_MODEL_MODE eligibility test at U_min = 600."""
    if not table.has(FULL_MODEL_MODE):
        return {
            "eligible": False,
            "reason": ("no measured FULL_MODEL_MODE throughput; the scope is "
                       "ineligible because eligibility may never be assumed"),
            "projection": None,
        }
    bench = table.get(FULL_MODEL_MODE)
    projection = projected_core_comparison_seconds(
        bench, FULL_MODEL_MIN_UPDATES, eval_seconds=eval_seconds_full)
    eligible = _fits(projection["total_seconds"], float(available_seconds),
                     reserve, safety_factor)
    return {
        "eligible": bool(eligible),
        "reason": ("fits the frozen inequality at U_min=600"
                   if eligible else
                   "does not fit the frozen inequality at U_min=600"),
        "projection": projection,
        "required_seconds": projection["total_seconds"] * safety_factor + reserve,
        "available_seconds": float(available_seconds),
    }


def largest_affordable_updates(bench: BenchMeasurement, available_seconds: float, *,
                               eval_seconds: float,
                               reserve: float = FINAL_TRANSFER_RESERVE,
                               safety_factor: float = SAFETY_FACTOR,
                               ladder: Sequence[int] = UPDATE_LADDER
                               ) -> tuple[int | None, list[dict]]:
    """Largest ladder value satisfying the frozen inequality (mechanical)."""
    considered: list[dict] = []
    chosen: int | None = None
    for updates in sorted(int(u) for u in ladder):
        projection = projected_core_comparison_seconds(
            bench, updates, eval_seconds=eval_seconds)
        fits = _fits(projection["total_seconds"], float(available_seconds),
                     reserve, safety_factor)
        considered.append({"updates": updates, "fits": bool(fits),
                           "projection": projection,
                           "required_seconds": (projection["total_seconds"]
                                                * safety_factor + reserve)})
        if fits:
            chosen = updates
    return chosen, considered


@dataclass(frozen=True)
class CorePlan:
    """The mechanical outcome of the §11 scope + U rule."""

    scope: str
    updates: int | None
    feasible: bool
    available_seconds: float
    record: dict

    def to_dict(self) -> dict:
        return dict(self.record)


def plan_core_comparison(*, available_seconds: float, table: ThroughputTable,
                         eval_seconds_by_scope: Mapping[str, float],
                         reserve: float = FINAL_TRANSFER_RESERVE,
                         safety_factor: float = SAFETY_FACTOR,
                         ladder: Sequence[int] = UPDATE_LADDER,
                         rehearsal: bool = False) -> CorePlan:
    """Apply the frozen §11 rule: FULL-vs-PEFT first, then the U ladder.

    The decision is MECHANICAL — it consumes measured throughput and the
    authorized time only.  No outcome, no dev metric, and no sealed record can
    reach it.

    ``ladder`` exists ONLY so that the offline dress rehearsal can walk the real
    scheduler with a miniature step count (contract §18: "tiny model, tiny
    pools, seconds-scale budgets").  Any ladder other than the frozen one
    REFUSES unless ``rehearsal=True`` is passed explicitly, and the resulting
    plan is stamped ``REHEARSAL_LADDER_OVERRIDE`` so no report can present it as
    a campaign plan.
    """
    if float(available_seconds) < 0.0:
        raise ValueError("available_seconds must be non-negative")
    ladder = tuple(int(u) for u in ladder)
    override = ladder != UPDATE_LADDER
    if override and not rehearsal:
        raise AffordabilityRefusal(
            f"REFUSED: update ladder {ladder} is not the frozen ladder "
            f"{UPDATE_LADDER}; only an explicitly labelled dress rehearsal may "
            "use a miniature ladder (contract §8/§20)")
    eligibility = full_model_eligible(
        table, available_seconds,
        eval_seconds_full=float(eval_seconds_by_scope.get(FULL_MODEL_MODE, 0.0)),
        reserve=reserve, safety_factor=safety_factor)
    scope = FULL_MODEL_MODE if eligibility["eligible"] else PEFT_MODE
    if not table.has(scope):
        record = {
            "schema": "flb200.core_plan.v1",
            "scope": scope,
            "updates": None,
            "feasible": False,
            "available_seconds": float(available_seconds),
            "safety_factor": float(safety_factor),
            "final_transfer_reserve_seconds": float(reserve),
            "full_model_eligibility": eligibility,
            "ladder_considered": [],
            "refusal": (f"no measured throughput for the selected scope "
                        f"{scope!r}; the core comparison cannot be planned"),
        }
        return CorePlan(scope, None, False, float(available_seconds), record)
    bench = table.get(scope)
    eval_seconds = float(eval_seconds_by_scope.get(scope, 0.0))
    updates, considered = largest_affordable_updates(
        bench, available_seconds, eval_seconds=eval_seconds,
        reserve=reserve, safety_factor=safety_factor, ladder=ladder)
    record = {
        "schema": "flb200.core_plan.v1",
        "scope": scope,
        "updates": updates,
        "update_ladder": list(ladder),
        "ladder_status": ("REHEARSAL_LADDER_OVERRIDE" if override
                          else "FROZEN_LADDER"),
        "feasible": updates is not None,
        "available_seconds": float(available_seconds),
        "safety_factor": float(safety_factor),
        "final_transfer_reserve_seconds": float(reserve),
        "eval_seconds": eval_seconds,
        "full_model_eligibility": eligibility,
        "ladder_considered": considered,
        "bench": bench.to_dict(),
        "rule": ("FULL eligible iff core(FULL, U=600) * 1.25 + 1200 <= "
                 "available; U = largest ladder value satisfying the same "
                 "inequality for the selected scope"),
    }
    if updates is None:
        record["refusal"] = (
            "no ladder value fits the frozen inequality; the core comparison "
            "is unaffordable in the available Foundation Learner time")
    return CorePlan(scope, updates, updates is not None,
                    float(available_seconds), record)
