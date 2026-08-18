"""Arm registry, frozen grid, and compute matching (contract §8, §11)."""

from __future__ import annotations

import pytest

from foundation_learner.training import arms
from foundation_learner.training.peft_modes import (
    FULL_MODEL_MODE_NAME,
    PEFT_MODE_NAME,
    TrainableModeError,
)


def _core(**overrides):
    kwargs = dict(
        learning_rate=1e-4,
        updates=600,
        max_tokens_per_batch=8192,
        peft_mode=PEFT_MODE_NAME,
    )
    kwargs.update(overrides)
    return arms.make_core_arm_configs(**kwargs)


def test_frozen_registry_bindings():
    assert arms.CORE_ARMS == ("FL1", "FL2", "FL3")
    assert arms.ARM_DATA_MODES == {
        "FL1": "ISOLATED_PAIRS",
        "FL2": "SUCCESS_EPISODES",
        "FL3": "ORDERED_EPISODES",
    }
    assert arms.UPDATE_LADDER == (600, 1200, 2400, 4800)
    assert arms.LEARNING_RATE_GRID == {
        PEFT_MODE_NAME: (1e-4, 3e-4),
        FULL_MODEL_MODE_NAME: (1e-5, 3e-5),
    }
    assert arms.ADAMW_BETAS == (0.9, 0.95)
    assert arms.ADAMW_EPS == 1e-8
    assert arms.WEIGHT_DECAY == 0.01
    assert arms.GRAD_CLIP_NORM == 1.0
    assert arms.WARMUP_FRACTION == 0.03
    assert arms.LR_SCHEDULE == "cosine"
    assert arms.ROOT_SEED_PRIMARY == 20260809
    assert arms.ROOT_SEED_SECONDARY == 20260810


def test_core_arm_configs_are_frozen_dataclasses():
    configs = _core()
    assert set(configs) == set(arms.CORE_ARMS)
    with pytest.raises(Exception):
        configs["FL3"].learning_rate = 5e-4
    assert configs["FL3"].objective == "OBJ_EPISODE_MEAN"
    assert configs["FL1"].objective == "OBJ_TOKEN_MEAN"
    assert configs["FL3"].warmup_steps == 18  # 3 % of 600


def test_learning_rate_outside_the_frozen_grid_is_refused():
    with pytest.raises(arms.ArmConfigError):
        arms.make_arm_config(
            "FL3",
            learning_rate=2e-4,
            updates=600,
            max_tokens_per_batch=8192,
            peft_mode=PEFT_MODE_NAME,
        )
    with pytest.raises(arms.ArmConfigError):
        arms.make_arm_config(
            "FL3",
            learning_rate=1e-4,  # PEFT value under FULL mode
            updates=600,
            max_tokens_per_batch=8192,
            peft_mode=FULL_MODEL_MODE_NAME,
        )


def test_update_ladder_and_grid_fraction_are_enforced():
    with pytest.raises(arms.ArmConfigError):
        arms.make_arm_config(
            "FL3",
            learning_rate=1e-4,
            updates=700,
            max_tokens_per_batch=8192,
            peft_mode=PEFT_MODE_NAME,
        )
    grid = arms.make_arm_config(
        "FL3",
        learning_rate=1e-4,
        updates=150,
        max_tokens_per_batch=8192,
        peft_mode=PEFT_MODE_NAME,
        stage=arms.STAGE_GRID,
    )
    assert grid.updates == 150 == int(600 * arms.GRID_UPDATE_FRACTION)
    with pytest.raises(arms.ArmConfigError):
        arms.make_arm_config(
            "FL3",
            learning_rate=1e-4,
            updates=175,
            max_tokens_per_batch=8192,
            peft_mode=PEFT_MODE_NAME,
            stage=arms.STAGE_GRID,
        )


def test_optimizer_and_schedule_cannot_be_overridden():
    for override in (
        {"betas": (0.9, 0.999)},
        {"eps": 1e-6},
        {"weight_decay": 0.1},
        {"grad_clip_norm": 5.0},
        {"warmup_fraction": 0.1},
        {"lr_schedule": "linear"},
    ):
        with pytest.raises(arms.ArmConfigError):
            arms.make_arm_config(
                "FL3",
                learning_rate=1e-4,
                updates=600,
                max_tokens_per_batch=8192,
                peft_mode=PEFT_MODE_NAME,
                **override,
            )


def test_sequence_budget_and_batch_budget_are_consistent():
    with pytest.raises(arms.ArmConfigError):
        arms.make_arm_config(
            "FL3",
            learning_rate=1e-4,
            updates=600,
            max_tokens_per_batch=1024,  # cannot hold one 2048-token episode
            peft_mode=PEFT_MODE_NAME,
        )
    with pytest.raises(arms.ArmConfigError):
        arms.make_arm_config(
            "FL3",
            learning_rate=1e-4,
            updates=600,
            max_tokens_per_batch=8192,
            peft_mode=PEFT_MODE_NAME,
            max_seq_len=4096,
        )


def test_wrong_objective_or_data_mode_is_refused():
    cfg = arms.make_arm_config(
        "FL3",
        learning_rate=1e-4,
        updates=600,
        max_tokens_per_batch=8192,
        peft_mode=PEFT_MODE_NAME,
    )
    import dataclasses

    with pytest.raises(arms.ArmConfigError):
        arms.validate_arm_config(dataclasses.replace(cfg, objective="OBJ_TOKEN_MEAN"))
    with pytest.raises(arms.ArmConfigError):
        arms.validate_arm_config(dataclasses.replace(cfg, data_mode="ISOLATED_PAIRS"))


def test_matched_core_arms_pass_the_matching_check():
    configs = _core()
    arms.assert_core_arms_matched(list(configs.values()))
    hashes = {arm: arms.arm_config_hash(cfg) for arm, cfg in configs.items()}
    assert len(set(hashes.values())) == 3  # arms differ in id/objective/data mode
    assert arms.arm_config_hash(configs["FL3"]) == arms.arm_config_hash(
        _core()["FL3"]
    )


def test_unequal_updates_or_budget_or_mode_is_refused():
    import dataclasses

    configs = _core()
    unequal_updates = [
        configs["FL1"],
        configs["FL2"],
        dataclasses.replace(configs["FL3"], updates=1200),
    ]
    with pytest.raises(arms.ArmConfigError):
        arms.assert_core_arms_matched(unequal_updates)

    unequal_budget = [
        configs["FL1"],
        dataclasses.replace(configs["FL2"], max_tokens_per_batch=4096),
        configs["FL3"],
    ]
    with pytest.raises(arms.ArmConfigError):
        arms.assert_core_arms_matched(unequal_budget)

    mixed_mode = [
        configs["FL1"],
        configs["FL2"],
        dataclasses.replace(
            configs["FL3"], peft_mode=FULL_MODEL_MODE_NAME, learning_rate=1e-5
        ),
    ]
    with pytest.raises((arms.ArmConfigError, TrainableModeError)):
        arms.assert_core_arms_matched(mixed_mode)

    with pytest.raises(arms.ArmConfigError):
        arms.assert_core_arms_matched([configs["FL1"], configs["FL1"]])
    with pytest.raises(arms.ArmConfigError):
        arms.assert_core_arms_matched([])
