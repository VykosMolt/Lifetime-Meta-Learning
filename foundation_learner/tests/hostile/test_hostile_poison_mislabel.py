"""HOSTILE: try to get poison into the certified channel, or to make poison
detectable from the channel tag rather than from the content.

Attacks, all of which must fail:

1. certification — search every poison condition for an event that is both
   certified and poisoned, or a certified FEEDBACK that is not truthful;
2. channel fingerprinting — compare the rendered form of truthful and poisoned
   HINT events and hope the markup differs;
3. no-op poison — hope a "corrupted" condition can leave the hint unchanged, so
   the condition is recorded but never realised;
4. flag laundering — hope an episode's recorded ``poison`` flags disagree with
   the condition schedule that produced them;
5. model disclosure — hope the environment-only poison flag reaches the text
   the model sees.
"""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import derive_seed, make_rng
from foundation_learner.ecology.families import all_families
from foundation_learner.ecology.poison import (POISON_CONDITIONS,
                                               POISONED_CONDITIONS,
                                               condition_schedule)
from foundation_learner.ecology.split import split_of
from foundation_learner.episodes.assemble import (CERTIFIED_CORRECT,
                                                  CERTIFIED_INCORRECT,
                                                  assemble_episode,
                                                  build_hint_variants)
from foundation_learner.episodes.render import MARKERS, render_episode
from foundation_learner.episodes.schema import Role

FAMILIES = all_families()
IDS = [f.family_id for f in FAMILIES]


def _episodes(family):
    seed = derive_seed(20260809, "hostile-poison", family.family_id)
    rule = family.sample_rule(make_rng(seed))
    for condition in POISON_CONDITIONS:
        yield condition, rule, assemble_episode(
            family, rule, split=split_of(family.family_id), difficulty=1,
            root_seed=20260809, episode_seed=seed, condition=condition,
            reveal=True)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_no_event_is_ever_both_certified_and_poisoned(family):
    from foundation_learner.ecology.base import TaskInstance

    for condition, rule, episode in _episodes(family):
        for index, event in enumerate(episode.events):
            assert not (event.certified and event.poison), (condition, index)
            if event.role is Role.FEEDBACK:
                assert event.certified and not event.poison
                attempt = episode.events[index - 1]
                assert attempt.role is Role.MODEL_ATTEMPT
                instance = TaskInstance.from_dict(
                    episode.items[attempt.item_id]["instance"])
                correct = family.verify(rule, instance, attempt.text).correct
                assert event.text == (CERTIFIED_CORRECT if correct
                                      else CERTIFIED_INCORRECT)
            if event.role is Role.REVEAL:
                assert event.certified and not event.poison
                assert event.text == \
                    episode.items[event.item_id]["instance"]["answer_canonical"]
            if event.role is Role.HINT:
                assert not event.certified


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_poison_is_not_visible_from_the_channel_markup(family):
    clean_blocks, poisoned_blocks = [], []
    for condition, _, episode in _episodes(family):
        rendered = render_episode(episode)
        for span, event in zip(rendered.spans, episode.events):
            if event.role is not Role.HINT:
                continue
            block = rendered.span_text(span)
            assert block.startswith(MARKERS[Role.HINT] + " ")
            (poisoned_blocks if event.poison else clean_blocks).append(block)
    assert clean_blocks and poisoned_blocks
    prefix = MARKERS[Role.HINT] + " "
    assert all(b.startswith(prefix) for b in clean_blocks + poisoned_blocks)
    # the rendered text carries no flag, tag or marker that separates them
    for block in poisoned_blocks:
        assert "poison" not in block.lower()
        assert "uncertified" not in block.lower()


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_a_named_poison_condition_is_always_actually_realised(family):
    seed = derive_seed(20260809, "hostile-poison", family.family_id)
    rule = family.sample_rule(make_rng(seed))
    episode = assemble_episode(
        family, rule, split=split_of(family.family_id), difficulty=1,
        root_seed=20260809, episode_seed=seed, condition="clean")
    variants = build_hint_variants(family, rule, episode, 20260809)
    clean = [slot["text"] for slot in variants["clean"]]
    for condition in POISON_CONDITIONS:
        slots = variants[condition]
        realised = [s for s in slots if s["condition"] == condition]
        if condition == "clean":
            assert all(s["condition"] == "clean" for s in slots)
            continue
        assert realised, condition
        for position, slot in enumerate(slots):
            if slot["condition"] != condition:
                continue
            assert slot["text"] != clean[position]
            assert slot["poison"] == (condition in POISONED_CONDITIONS)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_recorded_flags_agree_with_the_schedule(family):
    for condition, _, episode in _episodes(family):
        schedule = condition_schedule(episode.meta["root_seed"],
                                      episode.meta["episode_key"], condition,
                                      len(episode.meta["hint_slots"]))
        for slot, scheduled in zip(episode.meta["hint_slots"], schedule):
            assert slot["condition"] == scheduled
            assert slot["poison"] == (scheduled in POISONED_CONDITIONS)
            event = episode.events[slot["event_index"]]
            assert event.role is Role.HINT
            assert event.poison == slot["poison"]


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_the_poison_flag_never_reaches_the_rendered_text(family):
    for condition, _, episode in _episodes(family):
        text = render_episode(episode).text
        assert "poison" not in text.lower()
        assert "certified" not in text.lower()
        assert condition not in text
