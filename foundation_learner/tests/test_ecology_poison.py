"""Poison conditions, corruption guarantees and schedules (contract §4, §9)."""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import Feedback, HintField, make_rng
from foundation_learner.ecology import poison


#: a canonical answer that no hint body of ``_record`` can contain
ANSWER = "ZZTOP"


def _record():
    return Feedback("HINT-TEST", (
        HintField("first", "POS", 2, 6),
        HintField("length", "LEN", 0, 3, ("EQ", "SHORT", "LONG")),
    ))


def test_condition_names_cover_the_contract_set():
    assert poison.POISON_CONDITIONS == (
        "clean", "correct-redundant", "irrelevant", "partially-misleading",
        "corrupted")
    assert poison.CONDITION_DESCRIPTIONS["clean"] == "correct-informative"
    assert poison.POISONED_CONDITIONS == frozenset(
        {"partially-misleading", "corrupted"})


def test_full_corruption_moves_every_corruptible_field():
    record = _record()
    corrupted = poison.corrupt_feedback(record, make_rng(1), severity="full")
    assert corrupted.render() != record.render()
    for before, after in zip(record.fields, corrupted.fields):
        assert before.value != after.value
        assert 0 <= after.value < after.n_options


def test_partial_corruption_moves_exactly_one_field():
    record = _record()
    partial = poison.corrupt_feedback(record, make_rng(2), severity="partial")
    changed = [b.value != a.value for b, a in zip(record.fields, partial.fields)]
    assert sum(changed) == 1
    assert partial.render() != record.render()


def test_corruption_is_deterministic_in_its_generator():
    record = _record()
    a = poison.corrupt_feedback(record, make_rng(7)).render()
    b = poison.corrupt_feedback(record, make_rng(7)).render()
    assert a == b
    with pytest.raises(ValueError):
        poison.corrupt_feedback(record, make_rng(7), severity="nonsense")


def test_single_field_records_still_corrupt():
    record = Feedback("HINT-ONE", (HintField("count", "CNT", 0, 2),))
    for severity in ("full", "partial"):
        out = poison.corrupt_feedback(record, make_rng(3), severity=severity)
        assert out.render() != record.render()


def test_hint_for_condition_bodies():
    record = _record()
    clean = poison.hint_for_condition(record, "clean", make_rng(1),
                                      answer_canonical=ANSWER)
    assert clean == record.render()
    redundant = poison.hint_for_condition(record, "correct-redundant",
                                          make_rng(1), answer_canonical=ANSWER)
    assert redundant.startswith("HINT-ECHO")
    assert redundant != clean
    irrelevant = poison.hint_for_condition(record, "irrelevant", make_rng(1),
                                           answer_canonical=ANSWER)
    assert irrelevant in poison.IRRELEVANT_POOL
    for condition in ("partially-misleading", "corrupted"):
        assert poison.hint_for_condition(
            record, condition, make_rng(1), answer_canonical=ANSWER) != clean
    with pytest.raises(ValueError):
        poison.hint_for_condition(record, "no-such-condition", make_rng(1),
                                  answer_canonical=ANSWER)


def test_all_hint_bodies_are_digit_free():
    record = _record()
    for condition in poison.POISON_CONDITIONS:
        text = poison.hint_for_condition(record, condition, make_rng(4),
                                         answer_canonical=ANSWER)
        assert not any(ch.isdigit() for ch in text), text


def test_hint_for_condition_applies_the_answer_leak_invariant():
    """Amendment 13 minor (a): the condition renderer must apply the SAME
    ``answer_leak`` check that ``feedback_record`` / ``corrupted_feedback``
    apply, and it cannot be skipped by omitting the answer."""
    import inspect

    from foundation_learner.ecology.base import AnswerLeakError

    signature = inspect.signature(poison.hint_for_condition)
    parameter = signature.parameters["answer_canonical"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty, (
        "answer_canonical must be REQUIRED: an optional-and-omitted leak check "
        "is the failure mode being repaired")

    # A hint body that IS the answer must be refused in every condition where
    # the body can contain it.  "POS_C" is a code the clean body really renders.
    for condition in ("clean", "correct-redundant", "irrelevant",
                      "partially-misleading", "corrupted"):
        record = _record()
        rendered = poison.hint_for_condition(record, condition, make_rng(5),
                                             answer_canonical=ANSWER)
        leaking_answer = rendered.split()[-1].strip(";")
        with pytest.raises(AnswerLeakError):
            poison.hint_for_condition(record, condition, make_rng(5),
                                      answer_canonical=leaking_answer)


def test_schedule_is_deterministic_and_always_realised():
    for condition in poison.POISON_CONDITIONS:
        first = poison.condition_schedule(20260809, "episode-x", condition, 2)
        second = poison.condition_schedule(20260809, "episode-x", condition, 2)
        assert first == second
        assert len(first) == 2
        assert set(first) <= {"clean", condition}
        if condition == "clean":
            assert first == ["clean", "clean"]
        else:
            assert condition in first, "a named condition must occur somewhere"
    assert poison.condition_schedule(20260809, "a", "corrupted", 0) == []
    # the schedule genuinely depends on the episode
    variants = {tuple(poison.condition_schedule(20260809, f"ep{i}", "corrupted", 3))
                for i in range(20)}
    assert len(variants) > 1
    with pytest.raises(ValueError):
        poison.condition_schedule(1, "a", "nope", 2)
