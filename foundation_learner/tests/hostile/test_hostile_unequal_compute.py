"""HOSTILE: hidden unequal compute between the core arms.

Attack: give the treatment arm more optimizer updates, or a larger per-update
token budget, than the baselines — and report only the headline metric.  The
comparison would then measure compute, not the objective.

Required behaviour: both the CONFIGURATION check (before running) and the
LEDGER check (after running) refuse; the intrinsically differing quantities
(loss tokens per objective) are reported, never equalised or hidden.
Contract §8, §18, §20.
"""

from __future__ import annotations

import dataclasses

import pytest

from foundation_learner.training import arms
from foundation_learner.training import compute_accounting as ca
from foundation_learner.training.peft_modes import (
    FULL_MODEL_MODE_NAME,
    PEFT_MODE_NAME,
    TrainableModeError,
)


def _configs(**overrides):
    return arms.make_core_arm_configs(
        learning_rate=1e-4,
        updates=600,
        max_tokens_per_batch=8192,
        peft_mode=PEFT_MODE_NAME,
        **overrides,
    )


def _ledger(arm_id, *, updates=600, budget=8192, loss_tokens=1000):
    ledger = ca.ComputeLedger(
        arm_id=arm_id, max_tokens_per_batch=budget, planned_updates=updates
    )
    for step in range(updates):
        ledger.record_step(
            n_examples=1,
            forward_tokens=budget,
            forward_tokens_unpadded=budget,
            loss_tokens=loss_tokens,
            weighted_loss_mass=float(loss_tokens),
            did_update=True,
            step_seconds=0.0,
            episode_ids=[f"{arm_id}-{step}"],
        )
    return ledger.to_dict()


def test_extra_updates_for_the_treatment_arm_are_refused_at_config_time():
    configs = _configs()
    cheated = [
        configs["FL1"],
        configs["FL2"],
        dataclasses.replace(configs["FL3"], updates=1200),
    ]
    with pytest.raises(arms.ArmConfigError) as exc:
        arms.assert_core_arms_matched(cheated)
    assert "REFUSED" in str(exc.value)
    assert "updates" in str(exc.value)


def test_bigger_batch_budget_for_the_treatment_arm_is_refused_at_config_time():
    configs = _configs()
    cheated = [
        configs["FL1"],
        configs["FL2"],
        dataclasses.replace(configs["FL3"], max_tokens_per_batch=16384),
    ]
    with pytest.raises(arms.ArmConfigError) as exc:
        arms.assert_core_arms_matched(cheated)
    assert "max_tokens_per_batch" in str(exc.value)


def test_different_learning_rate_for_one_arm_is_refused():
    configs = _configs()
    cheated = [
        configs["FL1"],
        configs["FL2"],
        dataclasses.replace(configs["FL3"], learning_rate=3e-4),
    ]
    with pytest.raises(arms.ArmConfigError):
        arms.assert_core_arms_matched(cheated)


def test_mixed_trainable_parameter_modes_are_refused():
    configs = _configs()
    cheated = [
        configs["FL1"],
        configs["FL2"],
        dataclasses.replace(
            configs["FL3"], peft_mode=FULL_MODEL_MODE_NAME, learning_rate=1e-5
        ),
    ]
    with pytest.raises((arms.ArmConfigError, TrainableModeError)):
        arms.assert_core_arms_matched(cheated)


def test_mismatched_updates_are_refused_at_ledger_time():
    ledgers = [_ledger("FL1"), _ledger("FL2"), _ledger("FL3", updates=900)]
    with pytest.raises(ca.ComputeMismatchError) as exc:
        ca.compare_core_ledgers(ledgers)
    assert "REFUSED" in str(exc.value)
    assert "optimizer_updates" in str(exc.value)


def test_mismatched_batch_budget_is_refused_at_ledger_time():
    ledgers = [_ledger("FL1"), _ledger("FL2"), _ledger("FL3", budget=16384)]
    with pytest.raises(ca.ComputeMismatchError) as exc:
        ca.compare_core_ledgers(ledgers)
    assert "max_tokens_per_batch" in str(exc.value)


def test_intrinsic_loss_token_differences_are_reported_not_refused():
    ledgers = [
        _ledger("FL1", loss_tokens=200),
        _ledger("FL2", loss_tokens=1500),
        _ledger("FL3", loss_tokens=400),
    ]
    report = ca.compare_core_ledgers(ledgers)
    assert report["matched"]["optimizer_updates"] == 600
    differences = report["reported_differences"]["loss_tokens"]
    assert differences["FL1"] != differences["FL2"] != differences["FL3"]
    assert set(differences) == {"FL1", "FL2", "FL3"}


def test_a_matched_set_passes_both_gates():
    configs = _configs()
    arms.assert_core_arms_matched(list(configs.values()))
    report = ca.compare_core_ledgers(
        [_ledger("FL1"), _ledger("FL2"), _ledger("FL3")]
    )
    assert report["schema"] == "flb200.compute_comparison.v1"
