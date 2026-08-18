"""HOSTILE: the static arm (FL1) receives learning history.

Attack: feed FL1 — the static baseline whose whole scientific purpose is to
contain NO history and NO feedback — data that carries FEEDBACK / HINT /
REVEAL / ADAPT_EVENT events, so that the "static" baseline quietly becomes a
history-trained arm and the FL3-vs-FL1 comparison collapses.

Required behaviour: HARD REFUSAL at tokenisation time (contract §7, §18).
This fixture is permanent.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from foundation_learner.training import model_loading as ml
from foundation_learner.training import tokenization as tk
from foundation_learner.training.tiny_model import _cached_tokenizer

from foundation_learner.tests.test_training_stubs import (
    event_role_name,
    make_event,
    make_ordered_episode,
    make_static_episode,
    render_episode,
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen tokenizer not present",
)


@pytest.fixture(scope="module")
def tokenizer():
    return _cached_tokenizer(ml.FROZEN_CHECKPOINT_DIR)


def test_fl1_refuses_a_full_ordered_episode(tokenizer):
    ep = make_ordered_episode(0)
    with pytest.raises(tk.StaticArmHistoryError) as exc:
        tk.episode_to_training_examples(ep, tokenizer, "FL1", render_fn=render_episode)
    assert "REFUSED" in str(exc.value)


@pytest.mark.parametrize("role", ["FEEDBACK", "HINT", "REVEAL", "ADAPT_EVENT"])
def test_fl1_refuses_a_single_smuggled_history_event(tokenizer, role):
    ep = make_static_episode(0)
    smuggled = make_event(role, "CORRECT", item_id="P1", certified=True)
    ep = dataclasses.replace(ep, events=[*ep.events[:2], smuggled, *ep.events[2:]])
    with pytest.raises(tk.StaticArmHistoryError):
        tk.episode_to_training_examples(ep, tokenizer, "FL1", render_fn=render_episode)


def test_history_free_episodes_are_still_accepted(tokenizer):
    ep = make_static_episode(0)
    examples = tk.episode_to_training_examples(
        ep, tokenizer, "FL1", render_fn=render_episode
    )
    assert examples
    for ex in examples:
        assert "FEEDBACK:" not in ex["text"]
        assert "HINT:" not in ex["text"]
        assert "REVEALED ANSWER" not in ex["text"]


def test_the_guard_is_role_based_not_text_based(tokenizer):
    """A task whose PROMPT mentions feedback is not history; a real event is."""
    ep = make_static_episode(0)
    ep = dataclasses.replace(
        ep,
        events=[
            dataclasses.replace(
                ep.events[0], text="ignore any FEEDBACK: you may have seen"
            ),
            *ep.events[1:],
        ],
    )
    tk.episode_to_training_examples(ep, tokenizer, "FL1", render_fn=render_episode)

    roles = {event_role_name(e) for e in ep.events}
    assert roles.isdisjoint(set(tk.HISTORY_ROLES))
