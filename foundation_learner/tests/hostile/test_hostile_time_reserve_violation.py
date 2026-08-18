"""HOSTILE: admit a stage that eats the final transfer reserve.

Contract §11: ``projected * 1.25 + remaining_reserve <= remaining_authorized``,
SAFETY_FACTOR = 1.25, FINAL_TRANSFER_RESERVE = 1200 s, and the reserve is never
consumed.  The attack tries to weaken the safety factor, spend the reserve,
skip the projection, and run a stage that would overrun.
"""
from __future__ import annotations

import json

import pytest

from foundation_learner.campaign import affordability as aff
from foundation_learner.campaign import o1_isolation, scheduler as sched
from foundation_learner.campaign.stage_definitions import (StageContext,
                                                           StageDefinition)

RAN: list[str] = []


def greedy_work(ctx, stage):
    RAN.append(stage.stage_id)
    return {"ok": True}


THIS = "foundation_learner.tests.hostile.test_hostile_time_reserve_violation"


def setup(tmp_path, available, spu=1.0):
    RAN.clear()
    guard = o1_isolation.IsolationGuard(label="HOSTILE")
    ctx = StageContext(out_dir=str(tmp_path / "ladder"),
                       pregen_root=str(tmp_path), bundle_factory=lambda: None,
                       guard=guard, updates=600)
    ctx.throughput.add(aff.BenchMeasurement(
        scope=aff.PEFT_MODE, seconds_per_update=spu, tokens_per_second=10.0,
        updates_measured=4, wall_seconds=4 * spu, forward_tokens=100,
        max_tokens_per_batch=2048, eval_seconds_per_episode=0.0,
        # measured as zero so that this fixture isolates the RESERVE
        # arithmetic; the load term itself is attacked below
        model_load_seconds=0.0, forward_seconds_per_episode=0.0))
    s = sched.Scheduler(available_foundation_learner_seconds=available,
                        out_dir=str(tmp_path / "ladder"), guard=guard,
                        clock=sched.ManualClock())
    s.start()
    stage = StageDefinition(stage_id="GREEDY", priority=1, kind="TRAIN",
                            work=f"{THIS}:greedy_work", projection="TRAIN_ARM",
                            outputs=("x",), eval_episodes=0,
                            fallback_work=("record the refusal",))
    return s, ctx, stage


def test_a_stage_that_would_eat_the_reserve_never_runs(tmp_path):
    # 600 updates * 1 s = 600 s; required = 600*1.25 + 1200 = 1950
    s, ctx, stage = setup(tmp_path, available=1949.99)
    outcome = s.run_stage(stage, ctx)
    assert outcome.state == sched.STATE_REFUSED
    assert RAN == []
    assert outcome.fallback == ("record the refusal",)


def test_the_reserve_is_never_counted_as_available(tmp_path):
    s, ctx, stage = setup(tmp_path, available=1200.0)
    assert s.run_stage(stage, ctx).state == sched.STATE_REFUSED
    assert RAN == []


def test_the_safety_factor_and_reserve_cannot_be_weakened_in_the_policy(tmp_path):
    policy = aff.load_policy()
    for weakened in ({"safety_factor": 1.0},
                     {"final_transfer_reserve_seconds": 0},
                     {"reserve_consumable_by_stages": True}):
        payload = dict(policy)
        payload.update(weakened)
        path = tmp_path / "weak_policy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(aff.BudgetPolicyError):
            aff.load_policy(str(path))
        guard = o1_isolation.IsolationGuard(label="HOSTILE")
        with pytest.raises(aff.BudgetPolicyError):
            sched.Scheduler(available_foundation_learner_seconds=1e6,
                            out_dir=str(tmp_path / "l"), guard=guard,
                            policy_path=str(path))


def test_a_stage_cannot_run_without_a_measured_projection(tmp_path):
    s, ctx, stage = setup(tmp_path, available=1e6)
    ctx.throughput.measurements.clear()
    outcome = s.run_stage(stage, ctx)
    assert outcome.state == sched.STATE_BLOCKED
    assert RAN == []


def test_elapsed_time_shrinks_the_budget_for_later_stages(tmp_path):
    s, ctx, stage = setup(tmp_path, available=2100.0)
    assert s.run_stage(stage, ctx).state == sched.STATE_COMPLETE
    s.clock.advance(200.0)
    second = StageDefinition(stage_id="GREEDY2", priority=1, kind="TRAIN",
                             work=f"{THIS}:greedy_work",
                             projection="TRAIN_ARM", outputs=("x",),
                             eval_episodes=0)
    assert s.run_stage(second, ctx).state == sched.STATE_REFUSED
    assert RAN == ["GREEDY"]


def test_the_refusal_is_journalled_with_its_arithmetic(tmp_path):
    s, ctx, stage = setup(tmp_path, available=100.0)
    s.run_stage(stage, ctx)
    refusals = [r for r in s.read_journal() if r["event"] == "STAGE_REFUSED"]
    assert refusals
    admission = refusals[0]["admission"]
    assert admission["safety_factor"] == 1.25
    assert admission["remaining_reserve_seconds"] == 1200.0
    assert admission["admitted"] is False


def test_an_unmeasured_model_load_cannot_be_assumed_free(tmp_path):
    """Attack: hide the per-arm fresh checkpoint load from the projection.

    Contract §7 reloads the multi-GB frozen checkpoint for EVERY arm.  A
    projection that silently treats that as zero seconds under-projects every
    stage and spends the difference out of the transfer reserve, so a stage
    whose load cost was never measured must BLOCK rather than run (Amendment
    12 item 12).
    """
    s, ctx, stage = setup(tmp_path, available=1e6)
    ctx.throughput.measurements.clear()
    ctx.throughput.add(aff.BenchMeasurement(
        scope=aff.PEFT_MODE, seconds_per_update=1.0, tokens_per_second=10.0,
        updates_measured=4, wall_seconds=4.0, forward_tokens=100,
        max_tokens_per_batch=2048, eval_seconds_per_episode=0.0))
    outcome = s.run_stage(stage, ctx)
    assert outcome.state == sched.STATE_BLOCKED
    assert "model-load" in outcome.error
    assert RAN == []


def test_a_stage_may_not_run_past_the_reserve_once_it_has_started(tmp_path):
    """Attack: get admitted, then run long enough to eat the reserve anyway.

    Admission alone never bounded a RUNNING stage (review finding R-M8): a
    stage admitted with a projection could overrun without limit.  The watchdog
    aborts it the moment the remaining authorized time reaches the reserve.
    """
    s, ctx, stage = setup(tmp_path, available=3000.0)
    watchdog = s.build_watchdog("GREEDY", projected_seconds=1e9,
                                started=s.clock.monotonic())
    watchdog.check("start")
    s.clock.advance(1800.1)                 # 1199.9 s left, reserve is 1200 s
    with pytest.raises(sched.StageAbortedOverrun):
        watchdog.check("overrun")
    assert watchdog.tripped == "RESERVE_FLOOR"
