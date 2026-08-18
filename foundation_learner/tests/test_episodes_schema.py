"""Event / episode schema: roles, flags, identity and serialisation (§6)."""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import domain_sha256
from foundation_learner.episodes.schema import (EPISODE_STRUCTURE_V0, Episode,
                                                Event, EpisodeStructure,
                                                FEEDBACK_ROLES, Role,
                                                TASK_ROLES, make_episode_id,
                                                make_episode_key)


def test_roles_are_exactly_the_contract_set():
    assert {r.value for r in Role} == {
        "TASK", "SUPPORT_OBS", "MODEL_ATTEMPT", "FEEDBACK", "HINT", "REVEAL",
        "ADAPT_EVENT", "RELATED_TASK", "QUERY_TASK", "TRANSFER_TASK"}
    assert TASK_ROLES == (Role.TASK, Role.RELATED_TASK, Role.QUERY_TASK,
                          Role.TRANSFER_TASK)
    assert FEEDBACK_ROLES == (Role.FEEDBACK, Role.HINT, Role.REVEAL)


def test_event_carries_every_required_field_and_round_trips():
    event = Event(role=Role.HINT, text="HINT-X: a=TAG_B", item_id="i" * 64,
                  interaction_index=None, certified=False, poison=True,
                  reveal=False, slot="P1a0")
    payload = event.to_dict()
    assert set(payload) == {"role", "text", "item_id", "interaction_index",
                            "certified", "poison", "reveal", "slot", "meta"}
    assert payload["meta"] == {}
    # contract 23 side table; env-only, never rendered
    tagged = Event(role=Role.MODEL_ATTEMPT, text="ANSWER: x", item_id="a" * 64,
                   interaction_index=0, certified=False, poison=False,
                   reveal=False, slot="P1a0", meta={"slot_kind": "attempt0"})
    assert Event.from_dict(tagged.to_dict()) == tagged
    legacy = {k: v for k, v in tagged.to_dict().items() if k != "meta"}
    assert Event.from_dict(legacy).meta == {}
    assert payload["role"] == "HINT"
    assert Event.from_dict(payload) == event


def test_episode_structure_is_frozen_but_configurable_in_code():
    assert isinstance(EPISODE_STRUCTURE_V0, EpisodeStructure)
    other = EpisodeStructure(name="OTHER", n_support=1, attempts_per_support=1,
                             n_related=1, n_queries=1, n_transfers=1, k_max=3,
                             related_index=1, query_index=2, transfer_index=3)
    assert other.attempt_indices() == [0, 1, 2, 3]
    assert other.revision_indices() == [1]
    with pytest.raises(Exception):
        EPISODE_STRUCTURE_V0.k_max = 7  # frozen dataclass


def test_episode_identity_ignores_the_poison_condition_but_not_the_mode():
    args = ("boolean_rule", "DEVELOPMENT", "r" * 64, 11, 1)
    scripted = make_episode_id(*args, "scripted", "EPISODE_STRUCTURE_V0",
                               True, False)
    successful = make_episode_id(*args, "successful", "EPISODE_STRUCTURE_V0",
                                 True, False)
    revealed = make_episode_id(*args, "scripted", "EPISODE_STRUCTURE_V0",
                               True, True)
    unstructured = make_episode_id(*args, "scripted", "EPISODE_STRUCTURE_V0",
                                   False, False)
    assert len({scripted, successful, revealed, unstructured}) == 4
    assert scripted == domain_sha256("FL_V0_EPISODE", {
        "family_id": "boolean_rule", "split": "DEVELOPMENT",
        "rule_id": "r" * 64, "seed": 11, "difficulty": 1, "mode": "scripted",
        "structure": "EPISODE_STRUCTURE_V0", "structured": True,
        "reveal": False, "arm_tag": ""})

    key_scripted = make_episode_key(*args, "EPISODE_STRUCTURE_V0", True, False)
    key_successful = make_episode_key(*args, "EPISODE_STRUCTURE_V0", True, False)
    assert key_scripted == key_successful
    assert key_scripted != make_episode_key(*args, "EPISODE_STRUCTURE_V0",
                                            True, True)


def test_episode_round_trips_and_exposes_its_indices():
    events = [
        Event(Role.TASK, "task", "a" * 64, None, False, False, False, "P1"),
        Event(Role.MODEL_ATTEMPT, "ANSWER: x", "a" * 64, 0, False, False,
              False, "P1a0"),
        Event(Role.FEEDBACK, "CORRECT", "a" * 64, None, True, False, False,
              "P1a0"),
    ]
    episode = Episode(episode_id="e" * 64, family_id="boolean_rule",
                      split="DEVELOPMENT", rule_id="r" * 64,
                      rule_spec={"params": {}}, seed=3, difficulty=1,
                      condition="clean", mode="scripted",
                      structure="EPISODE_STRUCTURE_V0", structured=True,
                      reveal=False, events=events,
                      items={"a" * 64: {"instance": {}, "slot": "P1",
                                        "role": "support"}},
                      meta={})
    payload = episode.to_dict()
    assert Episode.from_dict(payload).to_dict() == payload
    assert episode.event_indices(Role.FEEDBACK) == [2]
    assert episode.event_indices(Role.TASK, Role.MODEL_ATTEMPT) == [0, 1]
    assert episode.instance_ids() == ["a" * 64]
