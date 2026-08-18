"""Structured feedback, its leak invariant and its corruption (contract §4)."""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import (AnswerLeakError, answer_leak,
                                             derive_seed, make_rng)
from foundation_learner.ecology.families import all_families
from foundation_learner.ecology.poison import (POISON_CONDITIONS,
                                               hint_for_condition)
from foundation_learner.episodes.parse import format_answer

FAMILIES = all_families()
IDS = [f.family_id for f in FAMILIES]

REMAP_IDS = ("canonical", "remap-fb-0", "remap-fb-1", "remap-fb-2")


def _cases(family, n_rules=3, n_items=4):
    """Rules x instances x surfaces x (correct, wrong, unparseable) attempts."""
    rng = make_rng(derive_seed(20260809, family.family_id, "feedback"))
    for _ in range(n_rules):
        rule = family.sample_rule(rng)
        for maker in (family.sample_instance, family.transfer_instance):
            for _ in range(n_items):
                base = maker(rule, rng, 1)
                for remap_id in REMAP_IDS:
                    instance = base if remap_id == "canonical" else \
                        family.surface_remap(rule, base, remap_id)
                    attempts = [
                        format_answer(instance.answer_canonical),
                        format_answer(family.wrong_answer(rule, instance, rng)),
                        "I do not know",
                    ]
                    for attempt in attempts:
                        yield rule, instance, attempt, rng


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_feedback_never_reveals_the_pending_answer(family):
    for rule, instance, attempt, rng in _cases(family):
        text = family.structured_feedback(rule, instance, attempt, rng)
        assert not answer_leak(text, instance.answer_canonical), text
        assert not any(ch.isdigit() for ch in text), text
        assert "ANSWER:" not in text


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_every_poison_condition_is_leak_free(family):
    for rule, instance, attempt, rng in _cases(family, n_rules=2, n_items=2):
        record = family.feedback_record(rule, instance, attempt, rng)
        for condition in POISON_CONDITIONS:
            local = make_rng(derive_seed(1, condition))
            text = hint_for_condition(
                record, condition, local,
                answer_canonical=instance.answer_canonical)
            assert not answer_leak(text, instance.answer_canonical), text
            assert not any(ch.isdigit() for ch in text), text


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_feedback_is_deterministic_for_a_fixed_item_and_attempt(family):
    for rule, instance, attempt, rng in _cases(family, n_rules=2, n_items=2):
        first = family.structured_feedback(rule, instance, attempt,
                                           make_rng(1))
        second = family.structured_feedback(rule, instance, attempt,
                                            make_rng(999))
        assert first == second


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_corrupted_feedback_always_differs_from_the_truth(family):
    for rule, instance, attempt, rng in _cases(family, n_rules=2, n_items=2):
        true_text = family.structured_feedback(rule, instance, attempt, rng)
        corrupted = family.corrupted_feedback(rule, instance, attempt, rng)
        assert corrupted != true_text
        assert corrupted == family.corrupted_feedback(rule, instance, attempt,
                                                      make_rng(5))
        assert not answer_leak(corrupted, instance.answer_canonical)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_feedback_is_informative_about_the_attempt_or_the_item(family):
    """Feedback must actually vary; a constant hint carries no signal."""
    seen = set()
    for rule, instance, attempt, rng in _cases(family, n_rules=3, n_items=3):
        seen.add(family.structured_feedback(rule, instance, attempt, rng))
    assert len(seen) > 1


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_answer_labels_cannot_collide_with_hint_letter_codes(family):
    """Structural precondition of the leak invariant.

    Hints are rendered from opaque bijective base-26 letter codes.  A label
    that is itself such a code (any single letter, or a string such as ``AB``)
    would appear as a standalone token inside a hint and count as a leak.
    """
    from foundation_learner.ecology.base import MAX_HINT_OPTIONS, letters

    codes = {letters(i) for i in range(MAX_HINT_OPTIONS)}
    rng = make_rng(derive_seed(11, family.family_id))
    rule = family.sample_rule(rng)
    item, _ = family._item_and_plan(rule, family.sample_instance(rule, rng, 1))
    vocabularies = [item.spec.labels, *family.label_variants]
    for vocabulary in vocabularies:
        for label in vocabulary:
            assert label not in codes, label
            assert label.isdigit() or len(label) >= 2, label
        assert len(set(vocabulary)) == len(vocabulary)
        if item.spec.labels:
            assert len(vocabulary) == len(item.spec.labels)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_hint_option_counts_stay_well_below_the_cap(family):
    """The cap that bounds the code alphabet must not be near-binding."""
    from foundation_learner.ecology.base import MAX_HINT_OPTIONS

    widest = 0
    for rule, instance, attempt, rng in _cases(family, n_rules=2, n_items=3):
        record = family.feedback_record(rule, instance, attempt, rng)
        widest = max(widest, max(f.n_options for f in record.fields))
    assert 2 <= widest <= MAX_HINT_OPTIONS // 2, widest


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_a_leaking_family_is_caught_by_the_guard(family):
    """Actively sabotage a family's hint so that it names the answer."""
    from foundation_learner.ecology.base import Feedback, HintField

    rng = make_rng(derive_seed(3, family.family_id))
    rule = family.sample_rule(rng)
    instance = family.sample_instance(rule, rng, 1)
    original = type(family)._feedback

    def leaking(self, rule_, item, plan, attempt, truth, rng_):
        return Feedback("HINT-LEAK", (
            HintField("payload", "ANS", 0, 2, (truth.replace(" ", "_"), "X")),))

    type(family)._feedback = leaking
    try:
        with pytest.raises((AnswerLeakError, ValueError)):
            family.structured_feedback(rule, instance,
                                       format_answer(instance.answer_canonical),
                                       rng)
    finally:
        type(family)._feedback = original
    # the family still behaves normally afterwards
    assert family.structured_feedback(
        rule, instance, format_answer(instance.answer_canonical), rng)
