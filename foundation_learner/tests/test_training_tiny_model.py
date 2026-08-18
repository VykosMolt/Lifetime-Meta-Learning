"""Tiny NONSCIENTIFIC Ouro model (contract §18, §20, §23)."""

from __future__ import annotations

import os

import pytest
import torch

from foundation_learner.training import model_loading as ml
from foundation_learner.training.tiny_model import (
    TINY_HIDDEN_SIZE,
    TINY_NUM_LAYERS,
    TINY_TOTAL_UT_STEPS,
    build_tiny_model,
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory (code + tokenizer) not present",
)


def test_tiny_model_uses_the_checkpoints_own_modeling_code():
    bundle = build_tiny_model(seed=1)
    module = type(bundle.model).__module__
    assert module.endswith("modeling_ouro"), module
    assert "ouro_rltt_local" in module
    assert type(bundle.model).__name__ == "OuroForCausalLM"
    assert type(bundle.model.config).__name__ == "OuroConfig"


def test_tiny_model_shapes_and_pinning():
    bundle = build_tiny_model(seed=1)
    cfg = bundle.model.config
    assert cfg.hidden_size == TINY_HIDDEN_SIZE
    assert cfg.num_hidden_layers == TINY_NUM_LAYERS
    assert cfg.total_ut_steps == TINY_TOTAL_UT_STEPS
    assert cfg.early_exit_threshold == 1.0
    assert cfg.vocab_size == len(bundle.tokenizer)
    assert len(bundle.model.model.layers) == TINY_NUM_LAYERS
    n_params = sum(p.numel() for p in bundle.model.parameters())
    assert n_params < 20_000_000, n_params


def test_tiny_model_is_marked_nonscientific():
    bundle = build_tiny_model(seed=1)
    assert bundle.identity["kind"] == "TINY_NONSCIENTIFIC"
    assert bundle.identity["scientific"] is False
    assert bundle.identity["NONSCIENTIFIC"] is True
    assert bundle.identity["checkpoint_tree_sha256"] is None
    assert "RANDOM_INIT" in bundle.identity["weights_source"]
    assert bundle.scientific is False
    with pytest.raises(ml.FrozenBindingError):
        bundle.assert_scientific()


def test_tiny_model_init_is_deterministic_in_the_seed():
    a = build_tiny_model(seed=4242)
    b = build_tiny_model(seed=4242)
    c = build_tiny_model(seed=4243)
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(pa, pb)
    assert not all(
        torch.equal(pa, pc) for pa, pc in zip(a.model.parameters(), c.model.parameters())
    )
    assert a.identity["identity_hash"] == b.identity["identity_hash"]
    # a different init seed is a different starting point, hence a different
    # base identity (the trainer refuses to resume across it)
    assert a.identity["identity_hash"] != c.identity["identity_hash"]


def test_tiny_model_build_does_not_disturb_global_rng():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    build_tiny_model(seed=99)
    assert torch.equal(before, torch.get_rng_state())


def test_tiny_model_forward_and_backward_are_finite():
    bundle = build_tiny_model(seed=5)
    bundle.model.train()
    ids = torch.randint(0, 128, (2, 16))
    out = bundle.model(input_ids=ids, use_cache=False)
    assert out.logits.shape == (2, 16, bundle.model.config.vocab_size)
    assert torch.isfinite(out.logits).all()
    out.logits.float().mean().backward()
    grads = [p.grad for p in bundle.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_tiny_model_can_use_a_small_stub_vocabulary():
    bundle = build_tiny_model(seed=6, vocab_size=64, attach_tokenizer=False)
    assert bundle.tokenizer is None
    assert bundle.model.config.vocab_size == 64
    out = bundle.model(input_ids=torch.randint(0, 64, (1, 8)), use_cache=False)
    assert out.logits.shape[-1] == 64
