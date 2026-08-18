"""HOSTILE: FL3 silently reduces to static training.

Attack: FL3 is the CORE TREATMENT — a weighted NLL over post-feedback answer
spans inside ordered episodes.  Two failure modes would make it a relabelled
static baseline while still "training":

1. the mask puts weight OUTSIDE the declared spans (context, feedback, hints),
   so the arm is really next-token training on the whole episode; or
2. the mask puts NO weight on the QUERY/TRANSFER spans, so nothing about
   post-feedback competence is optimised at all.

Required behaviour: nonzero weight ONLY on the declared spans, and strictly
positive weight on the query spans.  Contract §7, §18.
"""

from __future__ import annotations

import os

import pytest

from foundation_learner.training import model_loading as ml
from foundation_learner.training import objectives as ob
from foundation_learner.training import tokenization as tk
from foundation_learner.training.tiny_model import _cached_tokenizer

from foundation_learner.tests.test_training_stubs import (
    make_ordered_episode,
    render_episode,
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen tokenizer not present",
)

DECLARED_NONZERO_INDICES = {1, 3, 5, 6}


@pytest.fixture(scope="module")
def tokenizer():
    return _cached_tokenizer(ml.FROZEN_CHECKPOINT_DIR)


@pytest.fixture(scope="module")
def example(tokenizer):
    return tk.episode_to_training_example(
        make_ordered_episode(0), tokenizer, "FL3", render_fn=render_episode
    )


def test_nonzero_weight_lives_only_inside_declared_spans(example, tokenizer):
    text = example["text"]
    _, offsets = tk.encode_with_offsets(tokenizer, text)
    declared = [
        (s["char_start"], s["char_end"])
        for s in example["spans_meta"]
        if s["interaction_index"] in DECLARED_NONZERO_INDICES
    ]
    assert len(declared) == len(DECLARED_NONZERO_INDICES)
    for idx, weight in enumerate(example["loss_mask"]):
        if weight == 0.0:
            continue
        a, b = offsets[idx]
        for pos in range(a, b):
            if text[pos].isspace():
                continue
            assert any(
                start <= pos < end for start, end in declared
            ), (idx, text[a:b])


def test_query_and_transfer_spans_carry_strictly_positive_weight(example):
    weights = {s["interaction_index"]: s["weight"] for s in example["spans_meta"]}
    assert weights[5] > 0.0, "FL3 has no weight on QUERY: it reduces to static"
    assert weights[6] > 0.0, "FL3 has no weight on TRANSFER: it reduces to static"
    assert weights[1] > 0.0 and weights[3] > 0.0
    assert weights[0] == 0.0 and weights[2] == 0.0 and weights[4] == 0.0


def test_context_feedback_and_hints_are_fully_masked(example, tokenizer):
    text = example["text"]
    _, offsets = tk.encode_with_offsets(tokenizer, text)
    for marker in ("FEEDBACK: INCORRECT", "HINT: residue distance 2", "TASK: compute"):
        pos = text.find(marker)
        assert pos >= 0, marker
        for idx, (a, b) in enumerate(offsets):
            if a >= pos and b <= pos + len(marker):
                assert example["loss_mask"][idx] == 0.0, (marker, text[a:b])


def test_fl3_mask_is_strictly_sparser_than_the_imitation_mask(tokenizer):
    ep = make_ordered_episode(0)
    fl2 = tk.episode_to_training_example(ep, tokenizer, "FL2", render_fn=render_episode)
    fl3 = tk.episode_to_training_example(ep, tokenizer, "FL3", render_fn=render_episode)
    assert fl3["n_loss_tokens"] < fl2["n_loss_tokens"]
    for w3, w2 in zip(fl3["loss_mask"], fl2["loss_mask"]):
        if w3 > 0:
            assert w2 > 0  # FL3 weight only where a model span exists


def test_a_query_free_episode_is_refused_rather_than_trained_as_static(tokenizer):
    """Zero total weight must raise, not silently train on nothing."""
    import dataclasses

    ep = make_ordered_episode(0)
    keep = []
    for event in ep.events:
        index = getattr(event, "interaction_index", None)
        if index in (1, 3, 5, 6):
            continue
        keep.append(event)
    stripped = dataclasses.replace(ep, events=keep)
    with pytest.raises(tk.TokenizationError) as exc:
        tk.episode_to_training_example(
            stripped, tokenizer, "FL3", render_fn=render_episode
        )
    assert "zero total loss weight" in str(exc.value)


def test_zero_weight_batch_is_refused_by_the_objective():
    import torch

    logits = torch.randn(2, 6, 11)
    ids = torch.randint(0, 11, (2, 6))
    mask = torch.zeros(2, 6)
    with pytest.raises(ob.ZeroLossWeightError):
        ob.weighted_nll(logits, ids, mask, ob.OBJ_EPISODE_MEAN)
