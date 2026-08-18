"""Assembly of ``EPISODE_STRUCTURE_V0`` episodes (contract §6).

One latent rule per episode.  Interaction indices are frozen::

     -   TASK P1
     0   MODEL_ATTEMPT P1                -> R_0
     -   FEEDBACK(P1 attempt0) [+ HINT in structured conditions]
     1   MODEL_ATTEMPT P1 (revision)     -> R_1
     -   FEEDBACK(P1 attempt1)
     -   TASK P2
     2   MODEL_ATTEMPT P2                -> R_2
     -   FEEDBACK(P2 attempt0) [+ HINT]
     3   MODEL_ATTEMPT P2 (revision)     -> R_3
     -   FEEDBACK(P2 attempt1) [+ REVEAL(P1,P2) in reveal conditions]
     -   RELATED_TASK P3
     4   MODEL_ATTEMPT P3                -> R_4
     -   FEEDBACK(P3)
     5   QUERY_TASK Q1..Q3               -> R_5
     6   TRANSFER_TASK T1..T2            -> R_6

The certified FEEDBACK channel carries correctness only (``CORRECT`` /
``INCORRECT``) and is always truthful.  Structured content lives in the
uncertified HINT channel, which is where poison can appear.  Hidden QUERY and
TRANSFER labels are never revealed to the model.
"""
from __future__ import annotations

from ..ecology.base import (Rule, TaskFamily, TaskInstance, derive_seed,
                            make_rng)
from ..ecology.poison import (POISON_CONDITIONS, POISONED_CONDITIONS,
                              condition_schedule, hint_for_condition)
from .attempt_policy import attempt_for_slot
from .render import render_events
from .schema import (EPISODE_STRUCTURE_V0, Episode, Event, EpisodeStructure,
                     Role, make_episode_id, make_episode_key)

__all__ = ["MAX_EPISODE_CHARS", "CERTIFIED_CORRECT", "CERTIFIED_INCORRECT",
           "assemble_episode", "build_hint_variants", "EpisodeTooLong"]

CERTIFIED_CORRECT = "CORRECT"
CERTIFIED_INCORRECT = "INCORRECT"

#: cheap assembly-time guard; the real 2048-token assertion runs with the Ouro
#: tokenizer in the pre-generation path (``data/generate_shards.py``).
MAX_EPISODE_CHARS = 6000


class EpisodeTooLong(ValueError):
    """Raised when an assembled episode exceeds the character guard."""


def _instance_record(instance: TaskInstance, slot: str, role: str) -> dict:
    return {"instance": instance.to_dict(), "slot": slot, "role": role}


def assemble_episode(family: TaskFamily, rule: Rule, *, split: str,
                     difficulty: int, root_seed: int, episode_seed: int,
                     mode: str = "scripted", condition: str = "clean",
                     structured: bool = True, reveal: bool = False,
                     arm_tag: str = "",
                     structure: EpisodeStructure = EPISODE_STRUCTURE_V0) -> Episode:
    """Build one complete episode with scripted attempts and real feedback."""
    if condition not in POISON_CONDITIONS:
        raise ValueError(f"unknown poison condition {condition!r}")
    rule_id = family.rule_id(rule)
    episode_id = make_episode_id(family.family_id, split, rule_id, episode_seed,
                                 difficulty, mode, structure.name, structured,
                                 reveal, arm_tag)
    episode_key = make_episode_key(family.family_id, split, rule_id,
                                   episode_seed, difficulty, structure.name,
                                   structured, reveal, arm_tag)
    item_rng = make_rng(derive_seed(root_seed, "FL_V0_EPISODE_ITEMS", episode_key))

    supports = [family.sample_instance(rule, item_rng, difficulty)
                for _ in range(structure.n_support)]
    related = [family.related_instance(rule, item_rng, difficulty)
               for _ in range(structure.n_related)]
    queries = [family.query_instance(rule, item_rng, difficulty)
               for _ in range(structure.n_queries)]
    transfers = [family.transfer_instance(rule, item_rng, difficulty)
                 for _ in range(structure.n_transfers)]

    events: list[Event] = []
    items: dict[str, dict] = {}
    hint_slots: list[dict] = []
    attempts: list[dict] = []

    hint_conditions = condition_schedule(root_seed, episode_key, condition,
                                         structure.n_support if structured else 0)
    hint_rng = make_rng(derive_seed(root_seed, "FL_V0_HINT", episode_key, condition))

    def add(role: Role, text: str, item_id: str, index, certified: bool,
            poison: bool, reveal_flag: bool, slot: str,
            meta: dict | None = None) -> int:
        events.append(Event(role=role, text=text, item_id=item_id,
                            interaction_index=index, certified=certified,
                            poison=poison, reveal=reveal_flag, slot=slot,
                            meta=dict(meta or {})))
        return len(events) - 1

    def attempt(instance: TaskInstance, slot_kind: str, index: int,
                slot: str) -> tuple[str, bool]:
        text, wrong = attempt_for_slot(family, rule, instance, slot_kind, mode,
                                       root_seed, episode_key, slot)
        add(Role.MODEL_ATTEMPT, text, instance.instance_id, index, False,
            False, False, slot, {"slot_kind": slot_kind})
        attempts.append({"slot": slot, "slot_kind": slot_kind,
                         "interaction_index": index,
                         "item_id": instance.instance_id, "wrong": wrong,
                         "event_index": len(events) - 1})
        add(Role.FEEDBACK,
            CERTIFIED_INCORRECT if wrong else CERTIFIED_CORRECT,
            instance.instance_id, None, True, False, False, slot,
            {"slot_kind": slot_kind})
        return text, wrong

    # -- support items --------------------------------------------------
    for si, support in enumerate(supports):
        slot = f"P{si + 1}"
        items[support.instance_id] = _instance_record(support, slot, "support")
        add(Role.TASK, support.prompt_text, support.instance_id, None, False,
            False, False, slot, {"item_role": "support"})
        base = si * structure.attempts_per_support
        text, _ = attempt(support, "attempt0", base, f"{slot}a0")
        if structured:
            record = family.feedback_record(rule, support, text, hint_rng)
            slot_condition = hint_conditions[si]
            hint_text = hint_for_condition(
                record, slot_condition, hint_rng,
                answer_canonical=support.answer_canonical)
            poisoned = slot_condition in POISONED_CONDITIONS
            index = add(Role.HINT, hint_text, support.instance_id, None, False,
                        poisoned, False, f"{slot}a0",
                        {"slot_kind": "attempt0", "condition": slot_condition})
            hint_slots.append({"event_index": index, "slot": f"{slot}a0",
                               "item_id": support.instance_id,
                               "attempt_text": text,
                               "condition": slot_condition,
                               "poison": poisoned})
        attempt(support, "revision", base + 1, f"{slot}a1")

    if reveal:
        for si, support in enumerate(supports):
            add(Role.REVEAL, support.answer_canonical, support.instance_id,
                None, True, False, True, f"P{si + 1}",
                {"slot_kind": "reveal"})

    # -- related item ---------------------------------------------------
    for ri, item in enumerate(related):
        slot = f"P{structure.n_support + ri + 1}"
        items[item.instance_id] = _instance_record(item, slot, "related")
        add(Role.RELATED_TASK, item.prompt_text, item.instance_id, None, False,
            False, False, slot, {"item_role": "related"})
        attempt(item, "related", structure.related_index, f"{slot}a0")

    # -- queries (no feedback) ------------------------------------------
    for qi, item in enumerate(queries):
        slot = f"Q{qi + 1}"
        items[item.instance_id] = _instance_record(item, slot, "query")
        add(Role.QUERY_TASK, item.prompt_text, item.instance_id, None, False,
            False, False, slot, {"item_role": "query"})
        text, wrong = attempt_for_slot(family, rule, item, "query", mode,
                                       root_seed, episode_key, slot)
        add(Role.MODEL_ATTEMPT, text, item.instance_id, structure.query_index,
            False, False, False, slot, {"slot_kind": "query"})
        attempts.append({"slot": slot, "slot_kind": "query",
                         "interaction_index": structure.query_index,
                         "item_id": item.instance_id, "wrong": wrong,
                         "event_index": len(events) - 1})

    # -- transfers (no feedback) ----------------------------------------
    for ti, item in enumerate(transfers):
        slot = f"T{ti + 1}"
        items[item.instance_id] = _instance_record(item, slot, "transfer")
        add(Role.TRANSFER_TASK, item.prompt_text, item.instance_id, None, False,
            False, False, slot, {"item_role": "transfer"})
        text, wrong = attempt_for_slot(family, rule, item, "transfer", mode,
                                       root_seed, episode_key, slot)
        add(Role.MODEL_ATTEMPT, text, item.instance_id,
            structure.transfer_index, False, False, False, slot,
            {"slot_kind": "transfer"})
        attempts.append({"slot": slot, "slot_kind": "transfer",
                         "interaction_index": structure.transfer_index,
                         "item_id": item.instance_id, "wrong": wrong,
                         "event_index": len(events) - 1})

    episode = Episode(
        episode_id=episode_id, family_id=family.family_id, split=split,
        rule_id=rule_id, rule_spec=family.rule_spec(rule), seed=episode_seed,
        difficulty=difficulty, condition=condition, mode=mode,
        structure=structure.name, structured=structured, reveal=reveal,
        events=events, items=items,
        meta={"hint_slots": hint_slots, "attempts": attempts,
              "arm_tag": arm_tag, "root_seed": root_seed,
              "episode_key": episode_key,
              "generator_version": family.generator_version},
    )
    rendered = render_events(episode.events)
    episode.meta["render_chars"] = len(rendered.text)
    if len(rendered.text) > MAX_EPISODE_CHARS:
        raise EpisodeTooLong(
            f"{family.family_id}: episode renders to {len(rendered.text)} chars "
            f"(> {MAX_EPISODE_CHARS}); instances must be sized to fit")
    return episode


def build_hint_variants(family: TaskFamily, rule: Rule, episode: Episode,
                        root_seed: int) -> dict[str, list[dict]]:
    """HINT texts for every poison condition over ONE fixed episode.

    Returns ``condition -> [{event_index, slot, condition, poison, text}]``.
    The episode's items, scripted attempts and certified feedback are identical
    across conditions; only the uncertified HINT bodies change.
    """
    out: dict[str, list[dict]] = {}
    for condition in POISON_CONDITIONS:
        episode_key = episode.meta["episode_key"]
        rng = make_rng(derive_seed(root_seed, "FL_V0_HINT",
                                   episode_key, condition))
        slot_conditions = condition_schedule(root_seed, episode_key, condition,
                                             len(episode.meta["hint_slots"]))
        variant = []
        for slot_index, slot in enumerate(episode.meta["hint_slots"]):
            instance = TaskInstance.from_dict(
                episode.items[slot["item_id"]]["instance"])
            record = family.feedback_record(rule, instance,
                                            slot["attempt_text"], rng)
            slot_condition = slot_conditions[slot_index]
            variant.append({
                "event_index": slot["event_index"],
                "slot": slot["slot"],
                "condition": slot_condition,
                "poison": slot_condition in POISONED_CONDITIONS,
                "text": hint_for_condition(
                    record, slot_condition, rng,
                    answer_canonical=instance.answer_canonical),
            })
        out[condition] = variant
    return out
