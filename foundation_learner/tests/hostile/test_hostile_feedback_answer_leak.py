"""HOSTILE: try to make structured feedback disclose the pending answer.

Attacks, all of which must fail:

1. brute force — search every family, surface, difficulty, attempt type and
   poison condition for a hint that contains the pending canonical answer;
2. sabotage — replace a family's hint builder with one that names the answer
   and check the guard refuses to emit it;
3. digit smuggling — replace the hint with one that encodes the numeric answer
   as digits and check the digit-free invariant refuses it;
4. channel confusion — check that the certified FEEDBACK channel carries only
   ``CORRECT``/``INCORRECT`` and therefore cannot carry an answer either.
"""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import (AnswerLeakError, Feedback,
                                             HintField, answer_leak,
                                             derive_seed, make_rng)
from foundation_learner.ecology.families import all_families
from foundation_learner.ecology.poison import POISON_CONDITIONS, hint_for_condition
from foundation_learner.ecology.split import split_of
from foundation_learner.episodes.assemble import (CERTIFIED_CORRECT,
                                                  CERTIFIED_INCORRECT,
                                                  assemble_episode)
from foundation_learner.episodes.parse import format_answer
from foundation_learner.episodes.schema import Role

FAMILIES = all_families()
IDS = [f.family_id for f in FAMILIES]


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_brute_force_search_finds_no_leak(family):
    rng = make_rng(derive_seed(20260809, "hostile-leak", family.family_id))
    leaks = []
    for _ in range(4):
        rule = family.sample_rule(rng)
        for difficulty in family.difficulty_levels():
            for maker in (family.sample_instance, family.related_instance,
                          family.query_instance, family.transfer_instance):
                base = maker(rule, rng, difficulty)
                for surface in ("canonical", "remap-hostile-0",
                                "remap-hostile-1"):
                    instance = base if surface == "canonical" else \
                        family.surface_remap(rule, base, surface)
                    attempts = [
                        format_answer(instance.answer_canonical),
                        format_answer(family.wrong_answer(rule, instance, rng)),
                        "ANSWER: !!!", "no answer here",
                    ]
                    for attempt in attempts:
                        record = family.feedback_record(rule, instance,
                                                        attempt, rng)
                        for condition in POISON_CONDITIONS:
                            try:
                                text = hint_for_condition(
                                    record, condition, make_rng(11),
                                    answer_canonical=instance.answer_canonical)
                            except AnswerLeakError as exc:
                                # Amendment 13 installed the same invariant
                                # INSIDE hint_for_condition; a raise here is
                                # the guard firing, i.e. exactly the defect
                                # this fixture searches for.
                                leaks.append((condition, str(exc),
                                              instance.answer_canonical))
                                continue
                            if answer_leak(text, instance.answer_canonical):
                                leaks.append((condition, text,
                                              instance.answer_canonical))
    assert not leaks, leaks[:5]


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_sabotaged_hint_naming_the_answer_is_refused(family):
    rng = make_rng(derive_seed(1, "sabotage", family.family_id))
    rule = family.sample_rule(rng)
    instance = family.sample_instance(rule, rng, 1)
    original = type(family)._feedback

    def naming(self, rule_, item, plan, attempt, truth, rng_):
        payload = truth.replace(" ", "_").replace("=", "_")
        return Feedback("HINT-EVIL", (
            HintField("target", "ANS", 0, 2, (payload, "OTHER")),))

    type(family)._feedback = naming
    try:
        with pytest.raises((AnswerLeakError, ValueError)):
            family.structured_feedback(
                rule, instance, format_answer(instance.answer_canonical), rng)
    finally:
        type(family)._feedback = original


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_digit_smuggling_is_refused(family):
    rng = make_rng(derive_seed(2, "digits", family.family_id))
    rule = family.sample_rule(rng)
    instance = family.sample_instance(rule, rng, 1)
    original = type(family)._feedback

    def digits(self, rule_, item, plan, attempt, truth, rng_):
        return Feedback("HINT-EVIL", (
            HintField("code", "NUM", 0, 2, ("VALUE_7", "VALUE_8")),))

    type(family)._feedback = digits
    try:
        with pytest.raises(ValueError):
            family.structured_feedback(
                rule, instance, format_answer(instance.answer_canonical), rng)
    finally:
        type(family)._feedback = original


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_certified_channel_cannot_carry_an_answer(family):
    seed = derive_seed(3, "channel", family.family_id)
    rule = family.sample_rule(make_rng(seed))
    for condition in POISON_CONDITIONS:
        episode = assemble_episode(
            family, rule, split=split_of(family.family_id), difficulty=1,
            root_seed=20260809, episode_seed=seed, condition=condition,
            reveal=False)
        for event in episode.events:
            if event.role is Role.FEEDBACK:
                assert event.text in (CERTIFIED_CORRECT, CERTIFIED_INCORRECT)
            if event.role in (Role.FEEDBACK, Role.HINT):
                answer = episode.items[event.item_id]["instance"][
                    "answer_canonical"]
                assert not answer_leak(event.text, answer), event.text
