"""Frozen scripted attempt policies for pre-generated training histories.

Contract §6: training histories are OFF-POLICY and pre-generated.  Model
attempts in training data come from these frozen scripted policies, with the
frozen error model

    attempt0 wrong with p = 0.70
    revision wrong with p = 0.25
    related  wrong with p = 0.30

Wrong answers are drawn from the family's plausible-error sampler (contract §4
interface extension ``plausible_error``, wrapped by
``TaskFamily.wrong_answer`` which guarantees the drawn payload is never
accidentally correct).  QUERY and TRANSFER slots always carry the TRUE answer in
the target slot (used or masked per arm).

Modes:

``scripted``
    the frozen error model above (FL3's ordered-feedback data).
``successful``
    FL2's successful-history imitation data: the final revisions, the related
    attempt, and every query/transfer are correct.  The FIRST attempt on each
    support item keeps its 0.70 error rate, because an episode in which nothing
    was ever wrong contains no learning to imitate.

Every draw is a deterministic function of ``(root_seed, episode_id, slot_id)``;
no global RNG and no wall clock.
"""
from __future__ import annotations

from ..ecology.base import Rule, TaskFamily, TaskInstance, derive_seed, make_rng
from .parse import format_answer

__all__ = ["ERROR_RATES", "MODES", "SLOT_KINDS", "slot_error_rate",
           "attempt_for_slot"]

ERROR_RATES: dict[str, float] = {
    "attempt0": 0.70,
    "revision": 0.25,
    "related": 0.30,
    "query": 0.0,
    "transfer": 0.0,
}

SLOT_KINDS = tuple(ERROR_RATES)
MODES = ("scripted", "successful")

#: slots forced correct in the FL2 'successful' mode
_SUCCESSFUL_CORRECT = ("revision", "related", "query", "transfer")


def slot_error_rate(slot_kind: str, mode: str) -> float:
    if slot_kind not in ERROR_RATES:
        raise ValueError(f"unknown attempt slot kind {slot_kind!r}")
    if mode not in MODES:
        raise ValueError(f"unknown attempt policy mode {mode!r}")
    if mode == "successful" and slot_kind in _SUCCESSFUL_CORRECT:
        return 0.0
    return ERROR_RATES[slot_kind]


def attempt_for_slot(family: TaskFamily, rule: Rule, instance: TaskInstance,
                     slot_kind: str, mode: str, root_seed: int,
                     episode_key: str, slot_id: str) -> tuple[str, bool]:
    """Return ``(attempt_text, is_wrong)`` for one attempt slot.

    Deterministic in ``(root_seed, episode_key, slot_id)``.  The key is the
    MODE-INDEPENDENT episode identity, so the uniform draw of a slot is shared
    between the ``scripted`` and ``successful`` variants of one episode and the
    two differ only where the mode changes the error rate.
    """
    rate = slot_error_rate(slot_kind, mode)
    rng = make_rng(derive_seed(root_seed, "FL_V0_ATTEMPT", episode_key, slot_id))
    wrong = bool(rng.random() < rate)
    if wrong:
        payload = family.wrong_answer(rule, instance, rng)
    else:
        payload = instance.answer_canonical
    return format_answer(payload), wrong
