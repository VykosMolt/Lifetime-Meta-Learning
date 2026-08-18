"""Episode assembly, ordering, attempt policy and feedback channels (§6)."""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import canonical_json, derive_seed, make_rng
from foundation_learner.ecology.families import all_families
from foundation_learner.ecology.poison import POISON_CONDITIONS
from foundation_learner.ecology.split import split_of
from foundation_learner.episodes.assemble import (CERTIFIED_CORRECT,
                                                  CERTIFIED_INCORRECT,
                                                  MAX_EPISODE_CHARS,
                                                  assemble_episode,
                                                  build_hint_variants)
from foundation_learner.episodes.attempt_policy import (ERROR_RATES, MODES,
                                                        attempt_for_slot,
                                                        slot_error_rate)
from foundation_learner.episodes.schema import (EPISODE_STRUCTURE_V0, Episode,
                                                Role)

FAMILIES = all_families()
IDS = [f.family_id for f in FAMILIES]
S = EPISODE_STRUCTURE_V0


def _episode(family, index=0, **kwargs):
    seed = derive_seed(20260809, "episode-test", family.family_id, index)
    rule = family.sample_rule(make_rng(seed))
    kwargs.setdefault("split", split_of(family.family_id))
    kwargs.setdefault("difficulty", 1)
    kwargs.setdefault("root_seed", 20260809)
    kwargs.setdefault("episode_seed", seed)
    return rule, assemble_episode(family, rule, **kwargs)


def test_frozen_structure_constants():
    assert S.name == "EPISODE_STRUCTURE_V0"
    assert (S.n_support, S.attempts_per_support, S.n_related, S.n_queries,
            S.n_transfers, S.k_max) == (2, 2, 1, 3, 2, 6)
    assert S.attempt_indices() == [0, 1, 2, 3, 4, 5, 6]
    assert S.revision_indices() == [1, 3]
    assert S.attempt0_indices() == [0, 2]
    assert (S.related_index, S.query_index, S.transfer_index) == (4, 5, 6)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_event_sequence_is_exactly_episode_structure_v0(family):
    _, episode = _episode(family)
    roles = [e.role for e in episode.events]
    expected = [
        Role.TASK, Role.MODEL_ATTEMPT, Role.FEEDBACK, Role.HINT,
        Role.MODEL_ATTEMPT, Role.FEEDBACK,
        Role.TASK, Role.MODEL_ATTEMPT, Role.FEEDBACK, Role.HINT,
        Role.MODEL_ATTEMPT, Role.FEEDBACK,
        Role.RELATED_TASK, Role.MODEL_ATTEMPT, Role.FEEDBACK,
        Role.QUERY_TASK, Role.MODEL_ATTEMPT,
        Role.QUERY_TASK, Role.MODEL_ATTEMPT,
        Role.QUERY_TASK, Role.MODEL_ATTEMPT,
        Role.TRANSFER_TASK, Role.MODEL_ATTEMPT,
        Role.TRANSFER_TASK, Role.MODEL_ATTEMPT,
    ]
    assert roles == expected
    indices = [e.interaction_index for e in episode.events
               if e.role is Role.MODEL_ATTEMPT]
    assert indices == [0, 1, 2, 3, 4, 5, 5, 5, 6, 6]
    assert episode.structure == S.name
    assert len(episode.items) == 8


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_one_latent_rule_per_episode_and_item_roles(family):
    rule, episode = _episode(family)
    assert episode.rule_id == family.rule_id(rule)
    roles = sorted(r["role"] for r in episode.items.values())
    assert roles == ["query"] * 3 + ["related"] + ["support"] * 2 + \
        ["transfer"] * 2
    for item_id, record in episode.items.items():
        assert record["instance"]["rule_id"] == episode.rule_id
        assert record["instance"]["instance_id"] == item_id


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_certified_feedback_is_truthful_and_never_poisoned(family):
    rule, episode = _episode(family)
    attempts = {a["event_index"]: a for a in episode.meta["attempts"]}
    for index, event in enumerate(episode.events):
        if event.role is not Role.FEEDBACK:
            continue
        assert event.certified and not event.poison
        assert event.text in (CERTIFIED_CORRECT, CERTIFIED_INCORRECT)
        attempt = attempts[index - 1]
        instance = episode.items[attempt["item_id"]]["instance"]
        from foundation_learner.ecology.base import TaskInstance
        verdict = family.verify(rule, TaskInstance.from_dict(instance),
                                episode.events[index - 1].text)
        assert event.text == (CERTIFIED_CORRECT if verdict.correct
                              else CERTIFIED_INCORRECT)
        assert verdict.correct != attempt["wrong"]


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_hints_are_uncertified_and_queries_get_no_feedback(family):
    _, episode = _episode(family)
    for event in episode.events:
        if event.role is Role.HINT:
            assert not event.certified
        if event.role is Role.MODEL_ATTEMPT and \
                event.interaction_index in (S.query_index, S.transfer_index):
            following = episode.events[episode.events.index(event) + 1:]
            for nxt in following:
                if nxt.role is Role.MODEL_ATTEMPT:
                    break
                assert nxt.role not in (Role.FEEDBACK, Role.HINT, Role.REVEAL)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_query_and_transfer_slots_carry_true_answers(family):
    from foundation_learner.ecology.base import TaskInstance

    rule, episode = _episode(family)
    for attempt in episode.meta["attempts"]:
        if attempt["slot_kind"] not in ("query", "transfer"):
            continue
        instance = TaskInstance.from_dict(
            episode.items[attempt["item_id"]]["instance"])
        text = episode.events[attempt["event_index"]].text
        assert family.verify(rule, instance, text).correct
        assert not attempt["wrong"]


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_successful_mode_fixes_revisions_related_and_queries(family):
    from foundation_learner.ecology.base import TaskInstance

    rule, episode = _episode(family, mode="successful")
    assert episode.mode == "successful"
    for attempt in episode.meta["attempts"]:
        instance = TaskInstance.from_dict(
            episode.items[attempt["item_id"]]["instance"])
        correct = family.verify(rule, instance,
                                episode.events[attempt["event_index"]].text).correct
        if attempt["slot_kind"] == "attempt0":
            continue
        assert correct, attempt


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_scripted_and_successful_share_items_and_first_attempts(family):
    _, scripted = _episode(family, mode="scripted")
    _, successful = _episode(family, mode="successful")
    assert scripted.meta["episode_key"] == successful.meta["episode_key"]
    assert set(scripted.items) == set(successful.items)
    assert scripted.episode_id != successful.episode_id
    first_scripted = [a for a in scripted.meta["attempts"]
                      if a["slot_kind"] == "attempt0"]
    first_successful = [a for a in successful.meta["attempts"]
                        if a["slot_kind"] == "attempt0"]
    assert [a["wrong"] for a in first_scripted] == \
        [a["wrong"] for a in first_successful]


def test_attempt_policy_rates_are_frozen():
    assert ERROR_RATES["attempt0"] == 0.70
    assert ERROR_RATES["revision"] == 0.25
    assert ERROR_RATES["related"] == 0.30
    assert ERROR_RATES["query"] == 0.0 and ERROR_RATES["transfer"] == 0.0
    assert MODES == ("scripted", "successful")
    for kind in ("revision", "related", "query", "transfer"):
        assert slot_error_rate(kind, "successful") == 0.0
    assert slot_error_rate("attempt0", "successful") == 0.70
    with pytest.raises(ValueError):
        slot_error_rate("nope", "scripted")
    with pytest.raises(ValueError):
        slot_error_rate("attempt0", "nope")


def test_attempt_policy_realises_the_declared_error_rates():
    family = FAMILIES[0]
    rng = make_rng(derive_seed(1, "rates"))
    rule = family.sample_rule(rng)
    instance = family.sample_instance(rule, rng, 1)
    counts = {"attempt0": 0, "revision": 0, "related": 0}
    trials = 600
    for kind in counts:
        for i in range(trials):
            _, wrong = attempt_for_slot(family, rule, instance, kind,
                                        "scripted", 20260809, f"key{i}", kind)
            counts[kind] += int(wrong)
    for kind, rate in (("attempt0", 0.70), ("revision", 0.25), ("related", 0.30)):
        observed = counts[kind] / trials
        assert abs(observed - rate) < 0.06, (kind, observed)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_assembly_is_deterministic(family):
    _, first = _episode(family)
    _, second = _episode(family)
    assert canonical_json(first.to_dict()) == canonical_json(second.to_dict())
    assert Episode.from_dict(first.to_dict()).to_dict() == first.to_dict()


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_reveal_events_only_appear_in_reveal_conditions(family):
    _, plain = _episode(family)
    assert not any(e.role is Role.REVEAL for e in plain.events)
    rule, revealed = _episode(family, reveal=True)
    reveals = [e for e in revealed.events if e.role is Role.REVEAL]
    assert len(reveals) == S.n_support
    for event in reveals:
        assert event.certified and event.reveal and not event.poison
        assert event.text == \
            revealed.items[event.item_id]["instance"]["answer_canonical"]
        assert revealed.items[event.item_id]["role"] == "support"
    # hidden query/transfer labels are never revealed
    revealed_items = {e.item_id for e in reveals}
    for item_id, record in revealed.items.items():
        if record["role"] in ("query", "transfer"):
            assert item_id not in revealed_items


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_hint_variants_cover_every_condition_over_one_fixed_episode(family):
    rule, episode = _episode(family)
    variants = build_hint_variants(family, rule, episode, 20260809)
    assert set(variants) == set(POISON_CONDITIONS)
    clean = [slot["text"] for slot in variants["clean"]]
    for condition, slots in variants.items():
        assert len(slots) == len(episode.meta["hint_slots"])
        for slot in slots:
            assert episode.events[slot["event_index"]].role is Role.HINT
            assert slot["poison"] == (slot["condition"] in
                                      ("partially-misleading", "corrupted"))
        if condition != "clean":
            assert [s["text"] for s in slots] != clean
    assert variants == build_hint_variants(family, rule, episode, 20260809)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_episodes_fit_the_character_guard(family):
    for index in range(4):
        _, episode = _episode(family, index=index)
        assert episode.meta["render_chars"] <= MAX_EPISODE_CHARS


def test_unknown_condition_is_refused():
    family = FAMILIES[0]
    with pytest.raises(ValueError):
        _episode(family, condition="not-a-condition")
