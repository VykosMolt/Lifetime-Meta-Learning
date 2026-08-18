"""Objectives: weighted NLL and the FL3 per-episode reduction (contract §7)."""

from __future__ import annotations

import math

import pytest
import torch

from foundation_learner.training import objectives as ob


def _fixture(seed: int = 0, batch: int = 3, seq: int = 5, vocab: int = 7):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=gen)
    ids = torch.randint(0, vocab, (batch, seq), generator=gen)
    mask = torch.zeros(batch, seq)
    mask[0, 1] = 1.0
    mask[0, 3] = 0.25
    mask[1, 2] = 1.0
    mask[2, 4] = 0.5
    return logits, ids, mask


def _manual_nll(logits, ids):
    out = torch.zeros(ids.shape)
    logprobs = torch.log_softmax(logits.double(), dim=-1)
    for b in range(ids.shape[0]):
        for t in range(1, ids.shape[1]):
            out[b, t] = -logprobs[b, t - 1, ids[b, t]]
    return out


def test_per_token_nll_matches_a_manual_shifted_computation():
    logits, ids, _ = _fixture()
    nll = ob.per_token_nll(logits, ids)
    manual = _manual_nll(logits, ids)
    assert torch.allclose(nll.double(), manual.double(), atol=1e-6)
    assert torch.all(nll[:, 0] == 0)


def test_token_mean_is_the_global_weighted_mean():
    logits, ids, mask = _fixture()
    loss, stats = ob.weighted_nll(logits, ids, mask, ob.OBJ_TOKEN_MEAN)
    manual = _manual_nll(logits, ids)
    expected = float((manual * mask).sum() / mask.sum())
    assert float(loss) == pytest.approx(expected, rel=1e-6)
    assert stats["sum_weight"] == pytest.approx(float(mask.sum()))
    assert stats["n_weighted_tokens"] == 4
    assert stats["n_examples"] == 3


def test_episode_mean_is_per_episode_then_batch_mean():
    logits, ids, mask = _fixture()
    loss, stats = ob.weighted_nll(logits, ids, mask, ob.OBJ_EPISODE_MEAN)
    manual = _manual_nll(logits, ids)
    per_example = [
        float((manual[b] * mask[b]).sum() / mask[b].sum()) for b in range(ids.shape[0])
    ]
    assert float(loss) == pytest.approx(sum(per_example) / len(per_example), rel=1e-6)
    assert stats["per_example_loss"] == pytest.approx(per_example, rel=1e-5)


def test_the_two_reductions_differ_when_weights_are_unbalanced():
    logits, ids, mask = _fixture()
    mask[0] *= 40.0
    token_mean, _ = ob.weighted_nll(logits, ids, mask, ob.OBJ_TOKEN_MEAN)
    episode_mean, _ = ob.weighted_nll(logits, ids, mask, ob.OBJ_EPISODE_MEAN)
    assert not math.isclose(float(token_mean), float(episode_mean), rel_tol=1e-3)


def test_frozen_arm_objective_binding():
    assert ob.objective_for_arm("FL1") == ob.OBJ_TOKEN_MEAN
    assert ob.objective_for_arm("FL2") == ob.OBJ_TOKEN_MEAN
    assert ob.objective_for_arm("FL3") == ob.OBJ_EPISODE_MEAN
    with pytest.raises(ob.ObjectiveError):
        ob.objective_for_arm("FL9")


def test_zero_weight_example_is_refused():
    logits, ids, mask = _fixture()
    mask[1] = 0.0
    with pytest.raises(ob.ZeroLossWeightError):
        ob.weighted_nll(logits, ids, mask, ob.OBJ_EPISODE_MEAN)
    with pytest.raises(ob.ZeroLossWeightError):
        ob.weighted_nll(logits, ids, mask, ob.OBJ_TOKEN_MEAN)


def test_malformed_inputs_are_refused():
    logits, ids, mask = _fixture()
    with pytest.raises(ob.ObjectiveError):
        ob.weighted_nll(logits, ids, mask, "OBJ_UNKNOWN")
    with pytest.raises(ob.ObjectiveError):
        ob.weighted_nll(logits, ids, mask[:, :-1], ob.OBJ_TOKEN_MEAN)
    bad = mask.clone()
    bad[0, 0] = 1.0
    with pytest.raises(ob.ObjectiveError):
        ob.weighted_nll(logits, ids, bad, ob.OBJ_TOKEN_MEAN)
    negative = mask.clone()
    negative[0, 1] = -1.0
    with pytest.raises(ob.ObjectiveError):
        ob.weighted_nll(logits, ids, negative, ob.OBJ_TOKEN_MEAN)
    with pytest.raises(ob.ObjectiveError):
        ob.per_token_nll(logits[:, :1], ids[:, :1])


def test_gradients_flow_only_through_weighted_positions():
    logits, ids, mask = _fixture()
    logits.requires_grad_(True)
    loss, _ = ob.weighted_nll(logits, ids, mask, ob.OBJ_TOKEN_MEAN)
    loss.backward()
    grad = logits.grad
    # target position t is predicted from position t-1
    for b in range(ids.shape[0]):
        for t in range(ids.shape[1]):
            weighted = mask[b, t] > 0
            source = t - 1
            if source < 0:
                continue
            has_grad = bool(grad[b, source].abs().sum() > 0)
            assert has_grad == bool(weighted)


def test_bf16_logits_are_upcast_before_the_loss():
    logits, ids, mask = _fixture()
    loss32, _ = ob.weighted_nll(logits, ids, mask, ob.OBJ_TOKEN_MEAN)
    loss_bf16, _ = ob.weighted_nll(
        logits.to(torch.bfloat16).float(), ids, mask, ob.OBJ_TOKEN_MEAN
    )
    assert float(loss32) == pytest.approx(float(loss_bf16), rel=2e-2)
    assert loss32.dtype == torch.float32
