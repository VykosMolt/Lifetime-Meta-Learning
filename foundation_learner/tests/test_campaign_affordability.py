"""Frozen affordability arithmetic (contract §11).

Pure arithmetic with explicit inputs: every rule is checked at its boundary,
and the frozen policy file is checked for tampering in both directions.
"""
from __future__ import annotations

import json

import pytest

from foundation_learner.campaign import affordability as aff


def bench(scope=aff.PEFT_MODE, spu=1.0, eval_per_episode=0.5):
    return aff.BenchMeasurement(
        scope=scope, seconds_per_update=spu, tokens_per_second=1000.0,
        updates_measured=10, wall_seconds=10 * spu, forward_tokens=10000,
        max_tokens_per_batch=2048, device="cpu",
        eval_seconds_per_episode=eval_per_episode)


# ---------------- policy ----------------

def test_policy_loads_and_pins_the_frozen_constants():
    policy = aff.load_policy()
    assert policy["schema"] == aff.POLICY_SCHEMA
    assert policy["safety_factor"] == 1.25
    assert policy["final_transfer_reserve_seconds"] == 1200
    assert policy["update_ladder"] == [600, 1200, 2400, 4800]
    assert policy["session_total_authorized_usd"] == 45.0
    assert policy["rental_confirmation"] == "NOT_AUTHORIZED"
    assert policy["confirmation_authorized"] is False
    assert policy["o1_priority"] == "ABSOLUTE_FIRST"


@pytest.mark.parametrize("mutation", [
    {"safety_factor": 1.0},
    {"final_transfer_reserve_seconds": 0},
    {"update_ladder": [100, 200]},
    {"confirmation_authorized": True},
    {"rental_confirmation": "AUTHORIZED"},
    {"o1_priority": "SECOND"},
    {"reserve_consumable_by_stages": True},
    {"session_total_authorized_usd": 90.0},
])
def test_an_altered_policy_is_refused(tmp_path, mutation):
    policy = aff.load_policy()
    policy.update(mutation)
    path = tmp_path / "FL_BUDGET_POLICY.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(aff.BudgetPolicyError):
        aff.load_policy(str(path))


# ---------------- admission ----------------

def test_admission_rule_at_the_boundary():
    # projected 1000 * 1.25 + 1200 reserve = 2450
    ok = aff.admission_check(projected_stage_seconds=1000.0,
                             remaining_reserve_seconds=1200.0,
                             remaining_authorized_seconds=2450.0)
    assert ok["admitted"] is True and ok["required_seconds"] == 2450.0
    with pytest.raises(aff.AffordabilityRefusal):
        aff.admission_check(projected_stage_seconds=1000.0,
                            remaining_reserve_seconds=1200.0,
                            remaining_authorized_seconds=2449.999)


def test_admission_refusal_can_be_inspected_without_raising():
    record = aff.admission_check(projected_stage_seconds=1e6,
                                 remaining_reserve_seconds=1200.0,
                                 remaining_authorized_seconds=10.0,
                                 stage="FL3", strict=False)
    assert record["admitted"] is False
    assert record["stage"] == "FL3"


def test_negative_or_nonfinite_inputs_are_refused():
    with pytest.raises(aff.AffordabilityRefusal):
        aff.admission_check(projected_stage_seconds=float("nan"),
                            remaining_reserve_seconds=1200.0,
                            remaining_authorized_seconds=10.0)
    with pytest.raises(aff.AffordabilityRefusal):
        aff.admission_check(projected_stage_seconds=-1.0,
                            remaining_reserve_seconds=1200.0,
                            remaining_authorized_seconds=10.0)


# ---------------- projections ----------------

def test_core_comparison_projection_is_grid_plus_three_arms_plus_evals():
    b = bench(spu=2.0)
    projection = aff.projected_core_comparison_seconds(b, 600, eval_seconds=50.0)
    assert projection["grid_updates_per_config"] == 150
    assert projection["core_arm_seconds"] == 3 * 600 * 2.0
    assert projection["grid_seconds"] == 2 * 150 * 2.0
    assert projection["total_seconds"] == 3600.0 + 600.0 + 50.0


def test_evaluation_cost_may_not_be_guessed():
    b = aff.BenchMeasurement(scope=aff.PEFT_MODE, seconds_per_update=1.0,
                             tokens_per_second=1.0, updates_measured=1,
                             wall_seconds=1.0, forward_tokens=1,
                             max_tokens_per_batch=1)
    with pytest.raises(aff.AffordabilityRefusal):
        aff.projected_eval_seconds(b, 10)


def test_unmeasured_scope_refuses_instead_of_estimating():
    table = aff.ThroughputTable()
    with pytest.raises(aff.AffordabilityRefusal):
        table.get(aff.PEFT_MODE)


def test_bench_measurement_from_a_w2_compute_ledger():
    ledger = {"schema": "flb200.compute_ledger.v1", "optimizer_updates": 8,
              "skipped_updates": 2, "wall_time_s": 20.0,
              "forward_tokens": 40000, "max_tokens_per_batch": 2048,
              "device": "cuda"}
    m = aff.BenchMeasurement.from_ledger(ledger, aff.PEFT_MODE,
                                         eval_seconds_per_episode=1.0)
    assert m.seconds_per_update == 2.0
    assert m.tokens_per_second == 2000.0
    with pytest.raises(ValueError):
        aff.BenchMeasurement.from_ledger(dict(ledger, wall_time_s=0.0),
                                         aff.PEFT_MODE)


# ---------------- the scope + U rule ----------------

def test_full_model_is_ineligible_when_unmeasured():
    table = aff.ThroughputTable()
    table.add(bench(aff.PEFT_MODE, spu=1.0))
    verdict = aff.full_model_eligible(table, 1e9, eval_seconds_full=0.0)
    assert verdict["eligible"] is False
    assert "no measured FULL_MODEL_MODE throughput" in verdict["reason"]


def test_full_model_selected_only_when_the_frozen_inequality_holds():
    table = aff.ThroughputTable()
    table.add(bench(aff.PEFT_MODE, spu=0.1))
    table.add(bench(aff.FULL_MODEL_MODE, spu=1.0))
    # FULL at U=600: 3*600 + 2*150 = 2100 updates * 1.0 s = 2100 s
    # required = 2100 * 1.25 + 1200 = 3825
    plan = aff.plan_core_comparison(
        available_seconds=3825.0, table=table,
        eval_seconds_by_scope={aff.FULL_MODEL_MODE: 0.0, aff.PEFT_MODE: 0.0})
    assert plan.scope == aff.FULL_MODEL_MODE and plan.updates == 600
    tight = aff.plan_core_comparison(
        available_seconds=3824.9, table=table,
        eval_seconds_by_scope={aff.FULL_MODEL_MODE: 0.0, aff.PEFT_MODE: 0.0})
    assert tight.scope == aff.PEFT_MODE


def test_u_is_the_largest_ladder_value_that_fits():
    table = aff.ThroughputTable()
    table.add(bench(aff.PEFT_MODE, spu=0.1))
    # PEFT at U: 3.5*U*0.1 s -> required = 0.35*U*1.25 + 1200
    plan = aff.plan_core_comparison(
        available_seconds=0.35 * 2400 * 1.25 + 1200,
        table=table, eval_seconds_by_scope={aff.PEFT_MODE: 0.0})
    assert plan.scope == aff.PEFT_MODE and plan.updates == 2400
    assert [c["updates"] for c in plan.record["ladder_considered"]] == \
        [600, 1200, 2400, 4800]
    assert plan.record["ladder_status"] == "FROZEN_LADDER"


def test_no_ladder_value_fitting_is_recorded_not_invented():
    table = aff.ThroughputTable()
    table.add(bench(aff.PEFT_MODE, spu=100.0))
    plan = aff.plan_core_comparison(
        available_seconds=1500.0, table=table,
        eval_seconds_by_scope={aff.PEFT_MODE: 0.0})
    assert plan.feasible is False and plan.updates is None
    assert "unaffordable" in plan.record["refusal"]


def test_a_non_frozen_ladder_refuses_outside_a_rehearsal():
    table = aff.ThroughputTable()
    table.add(bench(aff.PEFT_MODE, spu=0.001))
    with pytest.raises(aff.AffordabilityRefusal):
        aff.plan_core_comparison(available_seconds=1e6, table=table,
                                 eval_seconds_by_scope={aff.PEFT_MODE: 0.0},
                                 ladder=(2,))
    plan = aff.plan_core_comparison(available_seconds=1e6, table=table,
                                    eval_seconds_by_scope={aff.PEFT_MODE: 0.0},
                                    ladder=(2,), rehearsal=True)
    assert plan.updates == 2
    assert plan.record["ladder_status"] == "REHEARSAL_LADDER_OVERRIDE"


def test_the_reserve_is_never_treated_as_available():
    table = aff.ThroughputTable()
    table.add(bench(aff.PEFT_MODE, spu=1.0))
    plan = aff.plan_core_comparison(
        available_seconds=1199.0, table=table,
        eval_seconds_by_scope={aff.PEFT_MODE: 0.0})
    assert plan.feasible is False
