"""Poisoned and degraded feedback variants (contract §4, §9, §16).

Corruption operates on the STRUCTURED record (:class:`~.base.Feedback`), never
on the rendered string, so a corrupted hint is guaranteed to differ from the
true hint in at least one opaque field while remaining well formed, digit-free
and non-revealing.

Conditions (contract §9 names; ``clean`` is §16's name for
``correct-informative``):

===========================  ===========================================
condition                    HINT content                       poison
===========================  ===========================================
``clean``                    true structured feedback            False
``correct-redundant``        truthful but content-free restatement  False
``irrelevant``               a fixed-pool irrelevant statement   False
``partially-misleading``     one field corrupted by a near miss  True
``corrupted``                every corruptible field corrupted   True
===========================  ===========================================

The certified FEEDBACK channel is NEVER produced here: certified feedback is
verifier-computed ``CORRECT``/``INCORRECT`` and is always truthful.  Poison
exists only in the uncertified HINT channel, and truthful and poisoned hints
are rendered identically so the channel tag cannot disclose poison.  The
``poison`` flag is recorded by the environment and never shown to the model.
"""
from __future__ import annotations

import numpy as np

from .base import (AnswerLeakError, Feedback, HintField, answer_leak,
                   derive_seed, letters, make_rng)

__all__ = ["POISON_CONDITIONS", "CONDITION_DESCRIPTIONS", "POISONED_CONDITIONS",
           "corrupt_feedback", "redundant_hint", "irrelevant_hint",
           "hint_for_condition", "condition_schedule", "IRRELEVANT_POOL"]

POISON_CONDITIONS: tuple[str, ...] = (
    "clean", "correct-redundant", "irrelevant", "partially-misleading",
    "corrupted")

CONDITION_DESCRIPTIONS = {
    "clean": "correct-informative",
    "correct-redundant": "correct but adds nothing new",
    "irrelevant": "truthful about something unrelated",
    "partially-misleading": "one field corrupted to a near miss",
    "corrupted": "every corruptible field corrupted",
}

#: Conditions whose HINT content is untrustworthy (env-only flag).
POISONED_CONDITIONS: frozenset[str] = frozenset(
    {"partially-misleading", "corrupted"})

#: Truthful-but-useless hint bodies (digit-free, answer-free).
IRRELEVANT_POOL: tuple[str, ...] = (
    "HINT-NOTE: channel=CHN_OPEN",
    "HINT-NOTE: transcript=TRN_STORED",
    "HINT-NOTE: layout=LYT_STANDARD",
    "HINT-NOTE: encoding=ENC_PLAIN",
)


def _shift(field: HintField, rng: np.random.Generator, near: bool) -> HintField:
    """Move a field to a different value; ``near`` prefers an adjacent one."""
    if field.n_options < 2:
        return field
    if near:
        step = 1 if int(rng.integers(0, 2)) == 0 else -1
        value = field.value + step
        if not 0 <= value < field.n_options:
            value = field.value - step
        if not 0 <= value < field.n_options:
            value = (field.value + 1) % field.n_options
    else:
        offset = 1 + int(rng.integers(0, field.n_options - 1))
        value = (field.value + offset) % field.n_options
    return field.with_value(value)


def corrupt_feedback(record: Feedback, rng: np.random.Generator,
                     severity: str = "full") -> Feedback:
    """Deterministic corruption of a structured-feedback record.

    ``severity='full'`` corrupts every field with >= 2 options;
    ``severity='partial'`` corrupts exactly one such field with a near miss and
    leaves the remaining fields truthful.  Both are guaranteed to differ from
    the input record.
    """
    if severity not in ("full", "partial"):
        raise ValueError(f"unknown corruption severity {severity!r}")
    corruptible = [i for i, f in enumerate(record.fields) if f.n_options >= 2]
    if not corruptible:
        raise ValueError("record has no corruptible field")
    fields = list(record.fields)
    if severity == "full":
        for i in corruptible:
            fields[i] = _shift(fields[i], rng, near=False)
    else:
        i = corruptible[int(rng.integers(0, len(corruptible)))]
        fields[i] = _shift(fields[i], rng, near=True)
    out = record.with_fields(fields)
    if out.render() == record.render():
        raise AssertionError("corruption produced an identical record")
    return out


def redundant_hint(record: Feedback) -> str:
    """A truthful restatement that carries no new information."""
    return f"HINT-ECHO: fields=FLD_{letters(len(record.fields))}"


def irrelevant_hint(rng: np.random.Generator) -> str:
    return IRRELEVANT_POOL[int(rng.integers(0, len(IRRELEVANT_POOL)))]


def hint_for_condition(record: Feedback, condition: str,
                       rng: np.random.Generator, *,
                       answer_canonical: str) -> str:
    """Render the HINT body for one poison condition.

    ``answer_canonical`` is REQUIRED and is the pending item's canonical answer:
    every rendered hint body is checked with the SAME
    :func:`~.base.answer_leak` invariant that ``TaskFamily.feedback_record`` and
    ``TaskFamily.corrupted_feedback`` apply.  Before Amendment 13 this function
    was the one hint-rendering path with no leak check, so a condition body
    (a corruption landing on the truth's token, an irrelevant-pool entry, a
    redundant restatement) could have carried the answer without any check
    firing.  The parameter is required rather than optional because an
    optional-and-omitted check is exactly the failure mode being repaired.
    """
    if condition not in POISON_CONDITIONS:
        raise ValueError(f"unknown poison condition {condition!r}")
    if condition == "clean":
        text = record.render()
    elif condition == "correct-redundant":
        text = redundant_hint(record)
    elif condition == "irrelevant":
        text = irrelevant_hint(rng)
    else:
        severity = "partial" if condition == "partially-misleading" else "full"
        text = corrupt_feedback(record, rng, severity=severity).render()
    if answer_leak(text, answer_canonical):
        raise AnswerLeakError(
            f"hint body for condition {condition!r} would reveal the pending "
            f"answer: {text!r} vs {answer_canonical!r}")
    return text


def condition_schedule(root_seed: int, episode_id: str, condition: str,
                       n_hint_slots: int) -> list[str]:
    """Per-hint-slot conditions for one episode under a named condition.

    ``clean`` leaves every slot clean.  Every other condition applies itself to
    a deterministic, non-empty subset of the hint slots (at least one slot, so
    the condition is always realised) and leaves the rest clean.
    """
    if condition not in POISON_CONDITIONS:
        raise ValueError(f"unknown poison condition {condition!r}")
    if n_hint_slots <= 0:
        return []
    if condition == "clean":
        return ["clean"] * n_hint_slots
    rng = make_rng(derive_seed(root_seed, "poison_schedule", episode_id, condition))
    flags = rng.integers(0, 2, size=n_hint_slots)
    if not flags.any():
        flags[int(rng.integers(0, n_hint_slots))] = 1
    return [condition if int(f) else "clean" for f in flags]
