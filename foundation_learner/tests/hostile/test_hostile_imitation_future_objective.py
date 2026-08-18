"""HOSTILE: the imitation arm (FL2) is given a future-competence objective.

Attack: quietly give FL2 — pure behaviour cloning of a successful learner —
some of FL3's future-competence weighting (up-weighting the post-feedback
QUERY/TRANSFER answers, or down-weighting attempt0).  FL2 would then no longer
be the objective-only control, and any FL3-over-FL2 effect would be an artefact
of the comparison having been contaminated.

Required behaviour: FL2's mask is EXACTLY the plain model-span mask — every
MODEL_ATTEMPT answer span carries weight 1.0, regardless of interaction index,
regardless of whether the span comes after feedback.  Contract §7, §18.
"""

from __future__ import annotations

import os

import pytest

from foundation_learner.training import model_loading as ml
from foundation_learner.training import objectives as ob
from foundation_learner.training import tokenization as tk
from foundation_learner.training.tiny_model import _cached_tokenizer

from foundation_learner.tests.test_training_stubs import (
    event_role_name,
    make_ordered_episode,
    render_episode,
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen tokenizer not present",
)


@pytest.fixture(scope="module")
def tokenizer():
    return _cached_tokenizer(ml.FROZEN_CHECKPOINT_DIR)


def _plain_model_span_mask(example, text, tokenizer):
    """Reference mask: weight 1.0 on every MODEL_ATTEMPT answer span."""
    _, offsets = tk.encode_with_offsets(tokenizer, text)
    mask = [0.0] * len(offsets)
    for span in example["spans_meta"]:
        start, end = tk.char_span_to_token_span(
            offsets,
            span["char_start"],
            span["char_end"],
            on_straddle=tk.MASK_STRADDLE_POLICY,
            text=text,
        )
        for idx in range(start, end):
            mask[idx] = 1.0
    return mask


def test_fl2_mask_equals_the_plain_model_span_mask(tokenizer):
    ep = make_ordered_episode(0)
    ex = tk.episode_to_training_example(ep, tokenizer, "FL2", render_fn=render_episode)
    assert ex["loss_mask"] == _plain_model_span_mask(ex, ex["text"], tokenizer)


def test_fl2_never_treats_a_post_feedback_span_differently(tokenizer):
    ep = make_ordered_episode(0)
    ex = tk.episode_to_training_example(ep, tokenizer, "FL2", render_fn=render_episode)
    by_index = {}
    for span in ex["spans_meta"]:
        by_index.setdefault(span["interaction_index"], set()).add(span["weight"])
    # every interaction index 0..6 present, all at weight 1.0
    assert sorted(by_index) == [0, 1, 2, 3, 4, 5, 6]
    assert set().union(*by_index.values()) == {1.0}


def test_fl2_weight_table_is_a_constant_not_an_index_lookup():
    assert tk.FL2_ATTEMPT_WEIGHT == 1.0
    for index in range(0, 7):
        assert tk._attempt_weight("FL2", index) == 1.0
    # FL3's table must NOT leak into FL2
    assert tk._attempt_weight("FL3", 0) == 0.0
    assert tk._attempt_weight("FL2", 0) == 1.0


def test_fl2_uses_the_plain_weighted_nll_not_the_episode_reduction():
    assert ob.objective_for_arm("FL2") == ob.OBJ_TOKEN_MEAN
    assert ob.objective_for_arm("FL3") == ob.OBJ_EPISODE_MEAN
    assert ob.ARM_OBJECTIVES["FL2"] != ob.ARM_OBJECTIVES["FL3"]


def test_fl2_weights_every_attempt_event_present_in_the_episode(tokenizer):
    ep = make_ordered_episode(0)
    ex = tk.episode_to_training_example(ep, tokenizer, "FL2", render_fn=render_episode)
    attempts = [
        i for i, e in enumerate(ep.events) if event_role_name(e) == "MODEL_ATTEMPT"
    ]
    covered = {span["event_index"] for span in ex["spans_meta"]}
    assert covered == set(attempts)
