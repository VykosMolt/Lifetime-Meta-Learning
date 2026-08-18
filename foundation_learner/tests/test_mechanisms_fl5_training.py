"""FL5 segmented-episode training under the FL3 objective (W3, contract §7/§8)."""
from __future__ import annotations

import os

import pytest
import torch

from foundation_learner.mechanisms.fast_state import FastStateModule
from foundation_learner.mechanisms.fl5_training import (
    ARM_FAST_STATE_OFF,
    ARM_FAST_STATE_ON,
    FL5TrainConfig,
    Fl5TrainingError,
    batch_loss,
    episode_loss,
    segment_episode,
    segment_token_nll,
    run_fast_state_arm,
)
from foundation_learner.training.objectives import per_token_nll
from foundation_learner.training.tokenization import FL3_INTERACTION_WEIGHTS
from foundation_learner.tests._w3_support import (
    embedding_matrix,
    fresh_tiny_bundle,
    real_episode,
    tiny_bundle,
)

STATE_DIM = 32


def _module(bundle, seed: int = 21) -> FastStateModule:
    return FastStateModule(bundle.model.config.hidden_size,
                           embedding_matrix=embedding_matrix(bundle),
                           state_dim=STATE_DIM, seed=seed)


def _segments(bundle=None, episode=None):
    bundle = bundle or tiny_bundle()
    episode = episode or real_episode()
    return segment_episode(episode, bundle.tokenizer)


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------
def test_segments_partition_the_token_sequence_exactly():
    segments = _segments()
    assert len(segments.segments) >= 2
    cursor = 0
    covered: list[int] = []
    for segment in segments.segments:
        assert segment.token_start == cursor
        assert segment.token_end > segment.token_start
        covered.extend(range(segment.token_start, segment.token_end))
        cursor = segment.token_end
    assert cursor == len(segments.input_ids)
    assert covered == list(range(len(segments.input_ids)))


def test_every_segment_boundary_follows_a_feedback_event():
    bundle = tiny_bundle()
    episode = real_episode()
    segments = segment_episode(episode, bundle.tokenizer)
    for segment in segments.segments[:-1]:
        last = episode.events[segment.event_indices[-1]]
        assert last.role.value in ("FEEDBACK", "HINT", "REVEAL")


def test_loss_spans_carry_the_frozen_fl3_weights():
    bundle = tiny_bundle()
    episode = real_episode()
    segments = segment_episode(episode, bundle.tokenizer)
    seen: dict[int, float] = {}
    for segment in segments.segments:
        for event_index, interaction, weight, t0, t1 in segment.loss_spans:
            assert weight == FL3_INTERACTION_WEIGHTS[interaction] > 0.0
            assert t1 > t0 >= 1        # never token 0 of a segment
            seen[interaction] = weight
    # queries and transfers carry 1.0, revisions 0.25, nothing else is weighted
    assert seen == {1: 0.25, 3: 0.25, 5: 1.0, 6: 1.0}


def test_feedback_items_are_attached_to_their_own_segment():
    bundle = tiny_bundle()
    episode = real_episode()
    segments = segment_episode(episode, bundle.tokenizer)
    total = sum(len(s.feedback_items) for s in segments.segments)
    expected = sum(1 for e in episode.events
                   if e.role.value in ("FEEDBACK", "HINT", "REVEAL"))
    assert total == expected


def test_oversized_episode_is_refused_not_truncated():
    bundle = tiny_bundle()
    with pytest.raises(Fl5TrainingError):
        segment_episode(real_episode(), bundle.tokenizer, max_seq_len=32)


# --------------------------------------------------------------------------
# per-segment NLL convention
# --------------------------------------------------------------------------
def test_segment_nll_matches_w2_convention_without_a_prefix():
    bundle = tiny_bundle()
    ids = torch.tensor([bundle.tokenizer(
        "TASK: 3 + 4\nATTEMPT: ANSWER: 7\n", add_special_tokens=False)["input_ids"]])
    with torch.inference_mode():
        logits = bundle.model(input_ids=ids, use_cache=False).logits
    mine = segment_token_nll(logits, ids, 0)
    w2 = per_token_nll(logits, ids)[0]
    assert torch.allclose(mine, w2, atol=1e-6)
    assert float(mine[0]) == 0.0


def test_segment_nll_offsets_by_the_prefix_length():
    bundle = tiny_bundle()
    hidden = bundle.model.config.hidden_size
    ids = torch.tensor([bundle.tokenizer(
        "TASK: 3 + 4\n", add_special_tokens=False)["input_ids"]])
    embed = bundle.model.get_input_embeddings()
    prefix = torch.zeros(1, 4, hidden)
    embeds = torch.cat([prefix, embed(ids)], dim=1)
    with torch.inference_mode():
        logits = bundle.model(inputs_embeds=embeds, use_cache=False).logits
    nll = segment_token_nll(logits, ids, 4)
    assert nll.shape == (ids.shape[1],)
    assert float(nll[0]) == 0.0
    with pytest.raises(Fl5TrainingError):
        segment_token_nll(logits, ids[:, :1], 4)


# --------------------------------------------------------------------------
# the episode loss
# --------------------------------------------------------------------------
def test_gradients_reach_w_in_and_the_gru_through_several_segments():
    bundle = tiny_bundle()
    module = _module(bundle)
    segments = _segments(bundle)
    assert len(segments.segments) >= 3
    loss, stats = episode_loss(bundle, module, segments, inject=True)
    loss.backward()
    assert stats["n_state_updates"] >= 2
    assert float(module.W_in.weight.grad.abs().sum()) > 0.0
    assert float(module.gru.weight_hh.grad.abs().sum()) > 0.0
    assert float(module.W_p.weight.grad.abs().sum()) > 0.0


def test_off_arm_runs_the_same_pipeline_without_injection():
    bundle = tiny_bundle()
    module = _module(bundle)
    segments = _segments(bundle)
    loss, stats = episode_loss(bundle, None, segments, inject=False)
    assert stats["inject"] is False
    assert stats["n_state_updates"] == 0
    assert stats["n_segments"] == len(segments.segments)
    on_loss, on_stats = episode_loss(bundle, module, segments, inject=True)
    assert on_stats["n_loss_tokens"] == stats["n_loss_tokens"]
    assert on_stats["forward_tokens"] > stats["forward_tokens"]  # prefix tokens
    assert float(loss.detach()) != float(on_loss.detach())
    with pytest.raises(Fl5TrainingError):
        episode_loss(bundle, None, segments, inject=True)


def test_episode_loss_is_the_weighted_token_mean():
    bundle = tiny_bundle()
    segments = _segments(bundle)
    loss, stats = episode_loss(bundle, None, segments, inject=False)
    assert stats["loss_weight"] == pytest.approx(segments.total_weight)
    assert float(loss.detach()) > 0.0


def test_batch_loss_is_the_mean_over_episodes():
    bundle = tiny_bundle()
    episodes = [_segments(bundle, real_episode(index=i)) for i in range(2)]
    total, stats = batch_loss(bundle, None, episodes, inject=False)
    singles = [float(episode_loss(bundle, None, e, inject=False)[0].detach())
               for e in episodes]
    assert float(total.detach()) == pytest.approx(sum(singles) / len(singles),
                                                  abs=1e-6)
    assert stats["objective"] == "OBJ_EPISODE_MEAN"
    assert stats["n_examples"] == 2


def test_episode_loss_is_deterministic():
    bundle = tiny_bundle()
    module = _module(bundle)
    segments = _segments(bundle)
    a = float(episode_loss(bundle, module, segments, inject=True)[0].detach())
    b = float(episode_loss(bundle, module, segments, inject=True)[0].detach())
    assert a == pytest.approx(b, abs=1e-9)


def test_detached_state_stops_cross_segment_gradient():
    bundle = tiny_bundle()
    module = _module(bundle)
    segments = _segments(bundle)
    loss, _ = episode_loss(bundle, module, segments, inject=True,
                           detach_state_between_segments=True)
    loss.backward()
    assert module.W_p.weight.grad is not None


# --------------------------------------------------------------------------
# the arm loop
# --------------------------------------------------------------------------
def test_arm_runs_records_and_checkpoints(tmp_path):
    bundle = fresh_tiny_bundle()
    module = _module(bundle)
    cfg = FL5TrainConfig(arm_id=ARM_FAST_STATE_ON, updates=2,
                         episodes_per_batch=1, learning_rate=1e-3,
                         train_backbone=False)
    episodes = [real_episode(index=i) for i in range(2)]
    result = run_fast_state_arm(cfg, bundle, module, episodes, str(tmp_path))
    assert result.steps == 2
    assert result.stopped_reason == "COMPLETED"
    assert len(result.loss_history) == 2
    assert result.ledger["optimizer_updates"] == 2
    assert result.ledger["loss_tokens"] > 0
    tags = {c["tag"] for c in result.checkpoints}
    assert {"initial", "final"} <= tags
    assert os.path.isfile(tmp_path / f"compute_ledger_{ARM_FAST_STATE_ON}.json")
    assert result.config["fast_state"]["state_dim"] == STATE_DIM


def test_off_arm_needs_backbone_parameters():
    bundle = tiny_bundle()
    module = _module(bundle)
    cfg = FL5TrainConfig(arm_id=ARM_FAST_STATE_OFF, updates=1)
    with pytest.raises(Fl5TrainingError):
        run_fast_state_arm(cfg, bundle, module, [real_episode()], "/tmp/unused_fl5")


def test_arm_config_is_validated():
    with pytest.raises(Fl5TrainingError):
        FL5TrainConfig(arm_id="FL3")
    with pytest.raises(Fl5TrainingError):
        FL5TrainConfig(updates=0)
    assert FL5TrainConfig(arm_id=ARM_FAST_STATE_ON).inject is True
    assert FL5TrainConfig(arm_id=ARM_FAST_STATE_OFF).inject is False
