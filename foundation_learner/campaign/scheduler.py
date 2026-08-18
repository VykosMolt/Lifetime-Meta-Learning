"""Time-aware stage scheduler (contract §11, §15).

The scheduler:

* accepts ``available_foundation_learner_seconds`` as an INPUT and never
  assumes ownership of the whole rental (O1 has absolute priority; FL time
  begins only after the O1 process is closed — contract §13);
* admits a stage only when
  ``projected_stage_seconds * 1.25 + remaining_reserve <= remaining_authorized``;
* never consumes the 1200 s final transfer reserve;
* takes every projection from the MEASURED ``BENCH`` stage (the only exception
  is BENCH itself, which is admitted against a declared bounded budget, since
  it is the measurement);
* runs the frozen §11 priority order: BENCH, FL0, the core matched comparison
  (development grid first, because the grid defines the core learning rate),
  FL4, FL5, FL6, FL7, FL8, additional predeclared seeds, and finally the single
  sealed opening;
* wires the frozen checkpoint cadence (600 s or 200 optimizer steps, whichever
  comes first) into the W2 trainer configuration and asserts it;
* reads its clock through an injected :class:`Clock`, so every admission rule is
  testable with a deterministic manual clock and no wall-clock value ever
  enters a decision;
* journals frequently: every admission, start, completion, refusal, skip,
  failure and heartbeat is an fsync'd append-only record.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from . import o1_isolation, stage_definitions
from .affordability import (AffordabilityRefusal, CorePlan, admission_check,
                            load_policy, plan_core_comparison, policy_sha256)
from .stage_definitions import (StageAbortedOverrun, StageContext,
                                StageDefinition, StageError, StageWatchdog,
                                planned_eval_episodes, project_stage_seconds,
                                resolve_dotted, stages_in_priority_order)

__all__ = [
    "Clock",
    "MonotonicClock",
    "ManualClock",
    "SchedulerError",
    "JOURNAL_NAME",
    "JOURNAL_SCHEMA",
    "StageOutcome",
    "Scheduler",
    "STATE_ABORTED_OVERRUN",
]

JOURNAL_NAME = "FL_SCHEDULER_JOURNAL.jsonl"
JOURNAL_SCHEMA = "flb200.scheduler_journal.v1"

STATE_COMPLETE = "COMPLETE"
STATE_REFUSED = "REFUSED_UNAFFORDABLE"
STATE_SKIPPED = "SKIPPED_ENTRY_CONDITION"
STATE_FAILED = "FAILED"
STATE_BLOCKED = "BLOCKED_MISSING_PREREQUISITE"
STATE_ABORTED_OVERRUN = "STAGE_ABORTED_OVERRUN"


class SchedulerError(RuntimeError):
    """The scheduler refused to run a stage for a structural reason."""


class Clock(Protocol):
    """Injected monotonic clock (contract §20: no wall clock in decisions)."""

    def monotonic(self) -> float: ...  # pragma: no cover - protocol


class MonotonicClock:
    """The real clock: ``time.monotonic`` only."""

    def monotonic(self) -> float:
        return time.monotonic()


class ManualClock:
    """Deterministic clock for tests and rehearsals."""

    def __init__(self, start: float = 0.0):
        self._t = float(start)

    def monotonic(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        self._t += float(seconds)
        return self._t


@dataclass
class StageOutcome:
    stage_id: str
    state: str
    seconds: float = 0.0
    admission: dict | None = None
    projection: dict | None = None
    entry: dict | None = None
    result: Any = None
    error: str | None = None
    fallback: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "state": self.state,
            "seconds": float(self.seconds),
            "admission": self.admission,
            "projection": self.projection,
            "entry": self.entry,
            "error": self.error,
            "fallback": list(self.fallback),
        }


@dataclass
class Scheduler:
    """Admission, ordering, and journalling for the FL ladder."""

    available_foundation_learner_seconds: float
    out_dir: str
    guard: o1_isolation.IsolationGuard = field(
        default_factory=o1_isolation.default_guard)
    clock: Clock = field(default_factory=MonotonicClock)
    policy_path: str | None = None
    label: str = "FL_LADDER"
    policy: dict = field(init=False)
    journal_path: str = field(init=False)
    outcomes: list[StageOutcome] = field(default_factory=list, init=False)
    plan: CorePlan | None = field(default=None, init=False)
    _t0: float | None = field(default=None, init=False)
    _journal_records: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if float(self.available_foundation_learner_seconds) < 0:
            raise SchedulerError(
                "available_foundation_learner_seconds must be non-negative")
        self.policy = load_policy(self.policy_path, guard=self.guard)
        self.guard.makedirs(self.out_dir)
        self.journal_path = os.path.join(self.out_dir, JOURNAL_NAME)

    # ---------------- clock + budget ----------------

    def start(self) -> float:
        if self._t0 is None:
            self._t0 = self.clock.monotonic()
            self.journal("SCHEDULER_START", {
                "available_foundation_learner_seconds":
                    float(self.available_foundation_learner_seconds),
                "safety_factor": self.safety_factor,
                "final_transfer_reserve_seconds": self.reserve,
                "policy_sha256": policy_sha256(self.policy_path),
                "o1_priority": self.policy["o1_priority"],
            })
        return self._t0

    @property
    def safety_factor(self) -> float:
        return float(self.policy["safety_factor"])

    @property
    def reserve(self) -> float:
        return float(self.policy["final_transfer_reserve_seconds"])

    def elapsed(self) -> float:
        if self._t0 is None:
            return 0.0
        return max(0.0, self.clock.monotonic() - self._t0)

    def remaining_authorized(self) -> float:
        return max(0.0, float(self.available_foundation_learner_seconds)
                   - self.elapsed())

    @property
    def watchdog_minimum(self) -> float:
        return float(self.policy["stage_watchdog_minimum_seconds"])

    def checkpoint_cadence(self) -> tuple[float, int]:
        return (float(self.policy["checkpoint_cadence_seconds"]),
                int(self.policy["checkpoint_cadence_steps"]))

    def build_watchdog(self, stage_id: str, projected_seconds: float,
                       started: float) -> StageWatchdog:
        """The stage watchdog for one admitted stage (contract §11, R-M8)."""
        return StageWatchdog(
            stage_id=stage_id, projected_seconds=float(projected_seconds),
            safety_factor=self.safety_factor,
            minimum_seconds=self.watchdog_minimum,
            started=float(started), clock=self.clock,
            remaining_fn=self.remaining_authorized,
            reserve_seconds=self.reserve)

    def assert_checkpoint_cadence(self, cfg: Any) -> None:
        """Refuse an arm configuration whose cadence is not the frozen one."""
        seconds, steps = self.checkpoint_cadence()
        if float(getattr(cfg, "checkpoint_every_seconds", -1)) != seconds or \
                int(getattr(cfg, "checkpoint_every_steps", -1)) != steps:
            raise SchedulerError(
                f"arm {getattr(cfg, 'arm_id', '?')!r} has checkpoint cadence "
                f"({getattr(cfg, 'checkpoint_every_seconds', None)}s, "
                f"{getattr(cfg, 'checkpoint_every_steps', None)} steps); the "
                f"frozen cadence is ({seconds}s, {steps} steps)")

    # ---------------- journal ----------------

    def journal(self, event: str, payload: dict | None = None) -> dict:
        record = {
            "schema": JOURNAL_SCHEMA,
            "index": self._journal_records,
            "event": str(event),
            "label": self.label,
            "monotonic": round(self.clock.monotonic(), 6),
            "elapsed_seconds": round(self.elapsed(), 6),
            "remaining_authorized_seconds": round(self.remaining_authorized(), 6),
            **(payload or {}),
        }
        self.guard.append_line(self.journal_path,
                               json.dumps(record, sort_keys=True,
                                          separators=(",", ":"),
                                          ensure_ascii=False))
        self._journal_records += 1
        return record

    def read_journal(self) -> list[dict]:
        path = self.guard.guard(self.journal_path, o1_isolation.MODE_READ)
        if not os.path.exists(path):
            return []
        out = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
        return out

    def heartbeat_hook(self, stage_id: str, every: int = 25,
                       watchdog: StageWatchdog | None = None
                       ) -> Callable[[dict], None]:
        """A trainer ``on_step`` hook that journals frequently.

        It also checks the stage watchdog on EVERY step (not only on journalled
        steps), which is the finest granularity at which a training stage can
        be stopped without interrupting an in-flight kernel.
        """
        def _hook(state: dict) -> None:
            if watchdog is not None:
                watchdog.check(f"{stage_id}:step {state.get('step')}")
            step = int(state.get("step", 0))
            if step % max(1, int(every)) == 0:
                self.journal("STAGE_HEARTBEAT", {
                    "stage_id": stage_id, "step": step,
                    "loss": state.get("loss"),
                })
        return _hook

    # ---------------- admission ----------------

    def admit(self, stage_id: str, projected_seconds: float, *,
              strict: bool = True) -> dict:
        """The frozen §11 admission rule, journalled either way."""
        try:
            record = admission_check(
                projected_stage_seconds=float(projected_seconds),
                remaining_reserve_seconds=self.reserve,
                remaining_authorized_seconds=self.remaining_authorized(),
                safety_factor=self.safety_factor,
                stage=stage_id, strict=strict)
        except AffordabilityRefusal as exc:
            record = admission_check(
                projected_stage_seconds=float(projected_seconds),
                remaining_reserve_seconds=self.reserve,
                remaining_authorized_seconds=self.remaining_authorized(),
                safety_factor=self.safety_factor,
                stage=stage_id, strict=False)
            record["refusal"] = str(exc)
            self.journal("STAGE_REFUSED", {"stage_id": stage_id,
                                           "admission": record})
            raise
        self.journal("STAGE_ADMITTED", {"stage_id": stage_id,
                                        "admission": record})
        return record

    # ---------------- planning ----------------

    def plan_core(self, ctx: StageContext, *,
                  eval_episodes_for_core: int | None = None) -> CorePlan:
        """Apply the §11 scope + U rule to the MEASURED throughput table."""
        if not ctx.throughput.measurements:
            raise SchedulerError(
                "no measured throughput: the BENCH stage must complete before "
                "the core comparison can be planned")
        eval_seconds_by_scope: dict[str, float] = {}
        episodes = eval_episodes_for_core
        if episodes is None:
            table = stage_definitions.EVAL_EPISODES_PER_STAGE
            episodes = (table["FL0"] + table["FL1"] + table["FL2"]
                        + table["FL3"] + 2 * table["DEV_GRID"])
            if ctx.eval_episode_cap is not None:
                episodes = min(episodes, 6 * int(ctx.eval_episode_cap))
        for scope, measurement in ctx.throughput.measurements.items():
            per_episode = measurement.eval_seconds_per_episode or 0.0
            eval_seconds_by_scope[scope] = float(per_episode) * int(episodes)
        ladder = ctx.extra.get("update_ladder")
        plan = plan_core_comparison(
            available_seconds=self.remaining_authorized(),
            table=ctx.throughput,
            eval_seconds_by_scope=eval_seconds_by_scope,
            reserve=self.reserve, safety_factor=self.safety_factor,
            **({"ladder": ladder, "rehearsal": bool(ctx.rehearsal)}
               if ladder is not None else {}))
        self.plan = plan
        ctx.scope = plan.scope
        ctx.updates = plan.updates
        self.journal("CORE_PLAN", {"plan": plan.to_dict(),
                                   "eval_episodes_assumed": int(episodes)})
        return plan

    # ---------------- running ----------------

    def _bench_declared_budget(self, ctx: StageContext) -> float:
        """The DECLARED BENCH budget, pinned to the policy outside a rehearsal.

        ``ctx.extra`` may only shorten the declared budget for an explicitly
        labelled dress rehearsal; in a real campaign the one declared (rather
        than measured) projection in the whole ladder comes from the frozen
        policy file and nowhere else.
        """
        policy_value = float(self.policy["bench_declared_budget_seconds"])
        override = ctx.extra.get("bench_declared_budget_seconds")
        if override is None:
            return policy_value
        if float(override) != policy_value and not ctx.rehearsal:
            raise StageError(
                f"bench_declared_budget_seconds={float(override)} overrides the "
                f"frozen policy value {policy_value}; only an explicitly "
                "labelled dress rehearsal may do that (contract §11/§20)")
        return float(override)

    def _projection_for(self, stage: StageDefinition, ctx: StageContext) -> dict:
        if stage.projection == "BENCH":
            scopes = len(list(ctx.extra.get("bench_scopes") or [ctx.scope]))
            return {"stage": stage.stage_id, "projection": "BENCH_DECLARED",
                    "projected_seconds": self._bench_declared_budget(ctx) * scopes,
                    "scopes": scopes,
                    "note": ("BENCH is admitted against a DECLARED bounded "
                             "budget: it is the measurement every other "
                             "projection depends on")}
        bench = ctx.throughput.get(ctx.scope)
        return project_stage_seconds(
            stage, bench=bench, updates=ctx.updates,
            eval_episodes=planned_eval_episodes(ctx, stage),
            extra=ctx.extra)

    def run_stage(self, stage: StageDefinition, ctx: StageContext) -> StageOutcome:
        """Check the entry condition, admit, run, and journal a stage."""
        ctx.scheduler = self
        if stage.entry:
            decision = resolve_dotted(stage.entry)(ctx)
            entry = decision.to_dict()
            if not decision.admitted:
                outcome = StageOutcome(stage.stage_id, STATE_SKIPPED, entry=entry,
                                       fallback=tuple(decision.fallback)
                                       or stage.fallback_work)
                self.journal("STAGE_SKIPPED", {"stage_id": stage.stage_id,
                                               "entry": entry,
                                               "fallback": list(outcome.fallback)})
                self.outcomes.append(outcome)
                return outcome
        else:
            entry = None

        try:
            projection = self._projection_for(stage, ctx)
        except (StageError, AffordabilityRefusal) as exc:
            outcome = StageOutcome(stage.stage_id, STATE_BLOCKED,
                                   entry=entry, error=str(exc),
                                   fallback=stage.fallback_work)
            self.journal("STAGE_BLOCKED", {"stage_id": stage.stage_id,
                                           "error": str(exc)})
            self.outcomes.append(outcome)
            return outcome

        try:
            admission = self.admit(stage.stage_id, projection["projected_seconds"])
        except AffordabilityRefusal as exc:
            outcome = StageOutcome(stage.stage_id, STATE_REFUSED, entry=entry,
                                   projection=projection, error=str(exc),
                                   fallback=stage.fallback_work)
            self.outcomes.append(outcome)
            return outcome

        self.journal("STAGE_STARTED", {"stage_id": stage.stage_id,
                                       "work": stage.work,
                                       "projection": projection})
        started = self.clock.monotonic()
        watchdog = self.build_watchdog(
            stage.stage_id, projection["projected_seconds"], started)
        ctx.watchdog = watchdog
        work = resolve_dotted(stage.work)
        try:
            result = work(ctx, stage)
        except StageAbortedOverrun as exc:
            seconds = self.clock.monotonic() - started
            outcome = StageOutcome(stage.stage_id, STATE_ABORTED_OVERRUN,
                                   seconds=seconds, admission=admission,
                                   projection=projection, entry=entry,
                                   error=str(exc),
                                   fallback=stage.fallback_work)
            self.journal("STAGE_ABORTED_OVERRUN", {
                "stage_id": stage.stage_id, "error": str(exc),
                "watchdog": watchdog.to_dict(),
                "fallback": list(stage.fallback_work)})
            self.outcomes.append(outcome)
            return outcome
        except BaseException as exc:  # noqa: BLE001 - recorded, never swallowed
            seconds = self.clock.monotonic() - started
            outcome = StageOutcome(stage.stage_id, STATE_FAILED, seconds=seconds,
                                   admission=admission, projection=projection,
                                   entry=entry, error=repr(exc),
                                   fallback=stage.fallback_work)
            self.journal("STAGE_FAILED", {
                "stage_id": stage.stage_id, "error": repr(exc),
                "traceback": traceback.format_exc(limit=12),
                "fallback": list(stage.fallback_work)})
            self.outcomes.append(outcome)
            return outcome
        finally:
            ctx.watchdog = None
        seconds = self.clock.monotonic() - started
        outcome = StageOutcome(stage.stage_id, STATE_COMPLETE, seconds=seconds,
                               admission=admission, projection=projection,
                               entry=entry, result=result)
        self.journal("STAGE_COMPLETED", {
            "stage_id": stage.stage_id, "seconds": round(seconds, 6),
            "outputs": list(stage.outputs)})
        self.outcomes.append(outcome)
        return outcome

    def run_ladder(self, ctx: StageContext, *,
                   stage_ids: Sequence[str] | None = None,
                   stop_on_failure: bool = False) -> dict:
        """Run the frozen §11 priority order.

        A refused, skipped, failed, or watchdog-ABORTED stage never stops the
        ladder by default: the predeclared fallback work is recorded and the
        scheduler moves on (contract §10 — all remaining time goes to
        predeclared work only).
        """
        self.start()
        stages = stages_in_priority_order()
        if stage_ids is not None:
            wanted = list(stage_ids)
            stages = [s for s in stages if s.stage_id in wanted]
        planned = False
        for stage in stages:
            if stage.projection != "BENCH" and not planned:
                # the §11 scope + U rule runs exactly once, immediately after
                # BENCH and before anything is projected from it
                try:
                    self.plan_core(ctx)
                except (SchedulerError, AffordabilityRefusal) as exc:
                    self.journal("CORE_PLAN_BLOCKED", {"error": str(exc)})
                planned = True
            outcome = self.run_stage(stage, ctx)
            if stop_on_failure and outcome.state == STATE_FAILED:
                break
        summary = self.summary()
        self.guard.write_json(os.path.join(self.out_dir, "LADDER_SUMMARY.json"),
                              summary)
        self.journal("SCHEDULER_COMPLETE", {"states": summary["states"]})
        return summary

    def summary(self) -> dict:
        return {
            "schema": "flb200.ladder_summary.v1",
            "label": self.label,
            "available_foundation_learner_seconds":
                float(self.available_foundation_learner_seconds),
            "elapsed_seconds": round(self.elapsed(), 6),
            "remaining_authorized_seconds": round(self.remaining_authorized(), 6),
            "final_transfer_reserve_seconds": self.reserve,
            "safety_factor": self.safety_factor,
            "core_plan": None if self.plan is None else self.plan.to_dict(),
            "stages": [o.to_dict() for o in self.outcomes],
            "states": {o.stage_id: o.state for o in self.outcomes},
            "journal_records": self._journal_records,
            "note": ("the final transfer reserve is never consumed by a stage; "
                     "every projection above the BENCH stage comes from "
                     "measured throughput"),
        }
