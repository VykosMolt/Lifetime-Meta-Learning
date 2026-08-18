"""Exact verification and the strict answer grammar (contract §4).

Positive, negative and adversarial cases for every family, including the
whitespace / case / format attacks on the ``ANSWER:`` line.
"""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import derive_seed, make_rng
from foundation_learner.ecology.families import all_families
from foundation_learner.episodes.parse import (ANSWER_LINE_RE, canonicalize,
                                               format_answer, parse_answer)

FAMILIES = all_families()
IDS = [f.family_id for f in FAMILIES]


def _fixture(family, seed=20260809):
    rng = make_rng(derive_seed(seed, family.family_id, "verify"))
    rule = family.sample_rule(rng)
    instances = [family.sample_instance(rule, rng, 1) for _ in range(6)]
    instances += [family.transfer_instance(rule, rng, 1) for _ in range(2)]
    return rule, instances, rng


# -- the grammar itself -----------------------------------------------------

def test_parse_answer_takes_the_last_well_formed_line():
    assert parse_answer("ANSWER: a\nANSWER: b") == "b"
    assert parse_answer("noise\nANSWER: b\nmore noise") == "b"
    assert parse_answer("ANSWER:b") == "b"
    assert parse_answer("ANSWER:   b   ") == "b"
    assert parse_answer("ANSWER: a b  c") == "a b  c"


@pytest.mark.parametrize("text", [
    "answer: b",              # lowercase marker
    " ANSWER: b",             # leading space breaks the anchor
    "\tANSWER: b",
    "ANSWER:",                # empty payload
    "ANSWER:    ",
    "the ANSWER: b",          # not at line start
    "ANSWERS: b",
    "ANSWER b",
    "no answer at all",
    "",
])
def test_parse_answer_rejects_malformed_lines(text):
    assert parse_answer(text) is None
    assert ANSWER_LINE_RE.match(text) is None


def test_canonicalize_modes_and_rejections():
    assert canonicalize(" yes ", "label") == "YES"
    assert canonicalize("a  b", "label") == "A B"
    assert canonicalize("+007", "int") == "7"
    assert canonicalize("7.0", "int") is None
    assert canonicalize("seven", "int") is None
    assert canonicalize(" A b C ", "string") == "abc"
    assert canonicalize("a, b  c", "token_list") == "a b c"
    assert canonicalize("b a b", "set") == "a b"
    assert canonicalize("empty", "set") == "EMPTY"
    assert canonicalize("( ~A & b )", "formula") == "(~a&b)"
    assert canonicalize("r1=2 r0=-1", "assignment") == "r0=-1 r1=2"
    assert canonicalize("r0=1 r0=2", "assignment") is None
    assert canonicalize("r0", "assignment") is None
    assert canonicalize(None, "label") is None
    with pytest.raises(ValueError):
        canonicalize("x", "no_such_mode")


# -- per family -------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_exact_answer_verifies(family):
    rule, instances, _ = _fixture(family)
    for instance in instances:
        result = family.verify(rule, instance,
                               format_answer(instance.answer_canonical))
        assert result.correct
        assert result.canonical_answer == instance.answer_canonical
        assert result.parsed == instance.answer_canonical


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_wrong_answers_are_rejected(family):
    rule, instances, rng = _fixture(family)
    for instance in instances:
        wrong = family.wrong_answer(rule, instance, rng)
        assert wrong != instance.answer_canonical
        assert not family.verify(rule, instance, format_answer(wrong)).correct


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_adversarial_answer_formats(family):
    rule, instances, _ = _fixture(family)
    instance = instances[0]
    truth = instance.answer_canonical

    def ok(text):
        return family.verify(rule, instance, text).correct

    # tolerated by the declared canonicalisation
    assert ok(f"ANSWER: {truth}")
    assert ok(f"ANSWER:{truth}")
    assert ok(f"ANSWER:   {truth}   ")
    assert ok(f"reasoning line\nANSWER: {truth}\n")
    assert ok(f"ANSWER: nonsense payload\nANSWER: {truth}")

    # rejected: no substring credit, no lenient marker, no trailing garbage
    assert not ok(f"The answer is {truth}")
    assert not ok(f"answer: {truth}")
    assert not ok(f" ANSWER: {truth}")
    assert not ok(f"ANSWER: {truth} extra")
    assert not ok(f"ANSWER: the answer is {truth}")
    assert not ok(f"ANSWER: {truth}\nANSWER: definitely wrong payload")
    assert not ok("ANSWER:")
    assert not ok(truth)
    assert not ok("")


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_case_and_separator_policy_is_the_declared_one(family):
    rule, instances, _ = _fixture(family)
    instance = instances[0]
    truth = instance.answer_canonical
    mode = family.canon_mode
    if mode == "label":
        assert family.verify(rule, instance,
                             format_answer(truth.lower())).correct
    if mode in ("token_list", "set"):
        assert family.verify(rule, instance,
                             format_answer(truth.replace(" ", ", "))).correct
        assert family.verify(rule, instance,
                             format_answer(truth.upper())).correct
    if mode in ("string", "formula"):
        assert family.verify(rule, instance,
                             format_answer(" ".join(truth))).correct
    if mode == "int":
        assert family.verify(rule, instance,
                             format_answer(f" {truth} ")).correct
        assert not family.verify(rule, instance,
                                 format_answer(f"{truth}.0")).correct


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_verifier_never_uses_an_llm_or_partial_credit(family):
    """A truncated or padded payload must score exactly zero."""
    rule, instances, _ = _fixture(family)
    for instance in instances:
        truth = instance.answer_canonical
        if len(truth) > 1:
            assert not family.verify(rule, instance,
                                     format_answer(truth[:-1])).correct
        assert not family.verify(rule, instance,
                                 format_answer(truth + " zzz")).correct
