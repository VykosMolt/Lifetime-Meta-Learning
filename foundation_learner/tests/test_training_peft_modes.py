"""Trainable-parameter modes (contract §8) and FL7/FL8 adapter specs (§7)."""

from __future__ import annotations

import os

import pytest

from foundation_learner.training import model_loading as ml
from foundation_learner.training import peft_modes as pm
from foundation_learner.training.tiny_model import build_tiny_model


def test_peft_mode_is_the_predeclared_spec():
    assert pm.PEFT_MODE.name == "PEFT_MODE"
    spec = pm.PEFT_MODE.lora_spec
    assert spec.rank == 16
    assert spec.alpha == 32.0
    assert spec.dropout == 0.0
    assert set(spec.target_suffixes) == {"q_proj", "v_proj", "down_proj"}
    assert pm.PEFT_MODE.learning_rates == (1e-4, 3e-4)


def test_full_model_mode_is_the_predeclared_spec():
    assert pm.FULL_MODEL_MODE.lora_spec is None
    assert pm.FULL_MODEL_MODE.learning_rates == (1e-5, 3e-5)


def test_fl7_and_fl8_specs_are_the_contract_values():
    assert pm.FL7_FAST_ADAPTER_SPEC.rank == 8
    assert pm.FL7_FAST_ADAPTER_SPEC.alpha == 16.0
    assert pm.FL7_FAST_ADAPTER_SPEC.dropout == 0.0
    assert set(pm.FL7_FAST_ADAPTER_SPEC.target_suffixes) == {"q_proj", "v_proj"}
    assert pm.FL7_FAST_DELTA_MAX_FROBENIUS == 1.0
    assert pm.FL7_INNER_STEPS == 4
    assert pm.FL7_INNER_LR == 1e-3
    assert pm.FL8_CONSOLIDATION_SCALE == 0.5


def test_unknown_mode_is_refused():
    with pytest.raises(pm.TrainableModeError):
        pm.get_mode("LORA_MODE")


def test_same_mode_assertion():
    assert pm.assert_same_mode_across_arms(["PEFT_MODE"] * 3) == "PEFT_MODE"
    with pytest.raises(pm.TrainableModeError):
        pm.assert_same_mode_across_arms(["PEFT_MODE", "PEFT_MODE", "FULL_MODEL_MODE"])
    with pytest.raises(pm.TrainableModeError):
        pm.assert_same_mode_across_arms([])
    with pytest.raises(pm.TrainableModeError):
        pm.assert_same_mode_across_arms(["PEFT_MODE", "NOT_A_MODE"])


@pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory (code) not present",
)
def test_apply_mode_selects_the_right_parameters():
    bundle = build_tiny_model(seed=3, vocab_size=64, attach_tokenizer=False)
    peft = pm.apply_mode(bundle.model, pm.PEFT_MODE, seed=1)
    assert peft.lora_handles is not None
    assert len(peft.lora_handles) == 6  # q/v/down over 2 layers
    assert peft.num_trainable == sum(p.numel() for p in peft.parameters)
    trainable = [n for n, p in bundle.model.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable)
    assert peft.to_dict()["mode"] == "PEFT_MODE"
    assert peft.to_dict()["lora_spec"]["rank"] == 16

    other = build_tiny_model(seed=3, vocab_size=64, attach_tokenizer=False)
    full = pm.apply_mode(other.model, pm.FULL_MODEL_MODE)
    assert full.lora_handles is None
    assert full.num_trainable == sum(p.numel() for p in other.model.parameters())
    assert full.num_trainable > peft.num_trainable


@pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory (code) not present",
)
def test_peft_mode_init_is_seed_deterministic():
    import torch

    a = build_tiny_model(seed=3, vocab_size=64, attach_tokenizer=False)
    b = build_tiny_model(seed=3, vocab_size=64, attach_tokenizer=False)
    c = build_tiny_model(seed=3, vocab_size=64, attach_tokenizer=False)
    ha = pm.apply_mode(a.model, pm.PEFT_MODE, seed=7).lora_handles
    hb = pm.apply_mode(b.model, pm.PEFT_MODE, seed=7).lora_handles
    hc = pm.apply_mode(c.model, pm.PEFT_MODE, seed=8).lora_handles
    for name in ha.names():
        assert torch.equal(ha.layers[name].lora_A, hb.layers[name].lora_A)
        assert not torch.equal(ha.layers[name].lora_A, hc.layers[name].lora_A)
