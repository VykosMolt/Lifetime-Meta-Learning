"""HOSTILE: evaluate a model that is still in TRAIN mode.

The attack is not adversarial input — it is the most ordinary sequence of calls
in the campaign: train an arm, then evaluate it with the bundle you already
have.  Transformers 4.54's ``GradientCheckpointingLayer.__call__`` contains

    if self.gradient_checkpointing and self.training:
        ... kwargs["use_cache"] = False; kwargs["past_key_values"] = None

and ``OuroDecoderLayer`` inherits that class.  The package's manual greedy
decoder carries its OWN KV cache, so a model left in train mode with
checkpointing enabled silently recomputes from scratch and emits DIFFERENT text
than the same weights in eval mode.  Nothing failed; the numbers were just
wrong, and the only visible trace was a log line.

Required behaviour (Amendment 16):

* the decode entry point REFUSES a train-mode or checkpointing-enabled model;
* ``run_training_arm`` returns the model in eval mode with layer-level
  checkpointing disabled, so a trained arm generates exactly what a fresh
  eval-mode load of the same weights generates;
* the campaign / mechanism evaluation helpers enforce the same state
  idempotently.

Contract §9 (evaluation is real greedy generation), §22 (the decode recipe),
§20 (no silent substitution of capability).
"""
from __future__ import annotations

import os

import pytest
import torch

from foundation_learner.campaign import stage_definitions as sd
from foundation_learner.evaluation.generation import (GenerationError,
                                                      assert_decodable,
                                                      greedy_generate)
from foundation_learner.training import model_loading as ml
from foundation_learner.training.arms import make_arm_config
from foundation_learner.training.tiny_model import build_tiny_model
from foundation_learner.training.tokenization import (
    isolated_pair_to_training_example)
from foundation_learner.training.trainer import run_training_arm

PROMPTS = ["TASK: alpha\nATTEMPT: ", "TASK: beta\nATTEMPT: "]


def tiny(seed: int = 20260809):
    return build_tiny_model(seed=seed, device="cpu")


# ---------------- the hazard is real ----------------

def test_the_decoder_really_is_corrupted_by_train_mode_checkpointing():
    """Pin the defect itself, so the guard can never be 'simplified' away."""
    bundle = tiny()
    ml.set_evaluation_mode(bundle.model)
    reference = greedy_generate(bundle, PROMPTS, max_new_tokens=8)

    ml.enable_training_memory_savings(bundle.model)
    bundle.model.train()
    with pytest.raises(GenerationError):
        greedy_generate(bundle, PROMPTS, max_new_tokens=8)

    # bypass the guard to demonstrate WHY it exists: the raw decode in that
    # state produces different text
    from foundation_learner.evaluation import generation as gen

    original = gen.assert_decodable
    gen.assert_decodable = lambda model: None
    try:
        corrupted = greedy_generate(bundle, PROMPTS, max_new_tokens=8)
    finally:
        gen.assert_decodable = original
    assert corrupted != reference, (
        "the fixture's premise no longer holds: train-mode decoding matched "
        "eval-mode decoding, so this guard's justification must be rechecked")


# ---------------- the guard ----------------

def test_the_decode_entry_refuses_a_train_mode_model():
    bundle = tiny()
    ml.enable_training_memory_savings(bundle.model)
    bundle.model.train()
    with pytest.raises(GenerationError) as exc:
        greedy_generate(bundle, PROMPTS, max_new_tokens=4)
    assert "TRAIN mode" in str(exc.value)


def test_the_decode_entry_refuses_checkpointing_even_in_eval_mode():
    """eval() alone is not enough: the flag itself is refused."""
    bundle = tiny()
    ml.enable_training_memory_savings(bundle.model)
    bundle.model.eval()
    assert any(getattr(m, "gradient_checkpointing", False)
               for m in bundle.model.modules())
    with pytest.raises(GenerationError) as exc:
        assert_decodable(bundle.model)
    assert "checkpointing still enabled" in str(exc.value)


def test_set_evaluation_mode_makes_the_model_decodable():
    bundle = tiny()
    ml.enable_training_memory_savings(bundle.model)
    bundle.model.train()
    state = ml.set_evaluation_mode(bundle.model)
    assert state["was_training"] is True and state["training"] is False
    assert state["still_enabled"] == []
    assert_decodable(bundle.model)          # does not raise


# ---------------- the post-training contract ----------------

def _example(bundle):
    return isolated_pair_to_training_example("TASK: alpha", "1", bundle.tokenizer)


def test_a_trained_arm_generates_what_a_fresh_eval_load_generates(tmp_path):
    """The end-to-end property the campaign depends on."""
    bundle = tiny()
    cfg = make_arm_config("FL1", learning_rate=1e-4, updates=2,
                          max_tokens_per_batch=512, peft_mode="PEFT_MODE",
                          seed=20260809, stage="SMOKE",
                          checkpoint_every_steps=200,
                          checkpoint_every_seconds=600.0, max_seq_len=512)
    result = run_training_arm(cfg, bundle, [_example(bundle)],
                              str(tmp_path / "arm"), None, allow_retry=False)

    # POST-CONDITION: the trainer hands the model back ready to evaluate
    assert bundle.model.training is False
    assert not any(getattr(m, "gradient_checkpointing", False)
                   for m in bundle.model.modules())
    assert result.trainable["post_training_mode"]["was_training"] is True

    after_training = greedy_generate(bundle, PROMPTS, max_new_tokens=8)

    # the SAME weights loaded into a fresh eval-mode bundle must decode
    # identically; anything else means the arm was evaluated in a state its
    # weights alone do not determine
    from foundation_learner.training.peft_modes import apply_mode

    fresh = tiny()
    apply_mode(fresh.model, cfg.peft_mode, seed=cfg.seed)   # same module shape
    fresh.model.load_state_dict(bundle.model.state_dict())
    ml.set_evaluation_mode(fresh.model)
    assert greedy_generate(fresh, PROMPTS, max_new_tokens=8) == after_training


def test_the_campaign_and_mechanism_helpers_enforce_the_same_state():
    bundle = tiny()
    ml.enable_training_memory_savings(bundle.model)
    bundle.model.train()
    state = sd.prepare_bundle_for_evaluation(bundle)
    assert state["training"] is False and state["still_enabled"] == []
    assert_decodable(bundle.model)

    # idempotent: calling it again on an already-evaluating bundle is a no-op
    again = sd.prepare_bundle_for_evaluation(bundle)
    assert again["was_training"] is False and again["was_enabled"] is False


def test_the_identity_record_states_the_hazard_instead_of_denying_it():
    """Amendment 12 item 16 said this was a no-op; Amendment 16 corrects it."""
    bundle = tiny()
    record = ml.enable_training_memory_savings(bundle.model)
    assert ml.layer_checkpointing_is_consulted(bundle.model) is True
    assert record["layer_level_checkpointing_active"] is True
    assert record["layer_level_checkpointing_detection"] == "CONSULTED_BY_FORWARD"
    assert record["cache_disabled_while_training"] is True
    assert "eval" in record["evaluation_requires_eval_mode"].lower()
