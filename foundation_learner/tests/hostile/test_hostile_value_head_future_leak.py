"""HOSTILE: the FL4 value head must not see future answers (contract §18).

Attacks:

1. Hand the head-input builder a feedback event that has been MOVED AFTER the
   query block, so its context would contain the query answers.
2. Ask the target pipeline to execute an ablation plan that removes a QUERY
   block / a query attempt (which would silently redefine the estimand).
3. Ask for targets on DEVELOPMENT and SEALED_TEST families.
4. Read the actual rendered head-input context and look for query answer text.

Each attack must be REFUSED or must demonstrably contain nothing of the future.
"""
from __future__ import annotations

import dataclasses

import pytest

from foundation_learner.episodes.render import render_subset
from foundation_learner.episodes.schema import Role
from foundation_learner.mechanisms.value_head import (
    AblationPlanError,
    SplitRefusedError,
    ValueHeadLeakError,
    ablation_plan_for_episode,
    build_value_dataset,
    episode_targets,
    head_context_indices,
    head_input_hidden,
    validate_ablation_plan,
)
from foundation_learner.tests._w3_support import (
    DEV_FAMILY,
    SEALED_FAMILY,
    feedback_event_indices,
    real_episode,
    tiny_bundle,
)


def _episode_with_feedback_after_the_queries(episode):
    """Move one FEEDBACK event to the END, i.e. behind the query answers."""
    events = list(episode.events)
    victim = events.pop(feedback_event_indices(episode)[0])
    return dataclasses.replace(episode, events=events + [victim])


def test_head_input_after_a_query_block_is_refused():
    episode = _episode_with_feedback_after_the_queries(real_episode())
    poisoned_index = len(episode.events) - 1
    assert episode.events[poisoned_index].role is Role.FEEDBACK
    with pytest.raises(ValueHeadLeakError):
        head_context_indices(episode, poisoned_index)
    with pytest.raises(ValueHeadLeakError):
        head_input_hidden(tiny_bundle(), episode, poisoned_index)


def test_head_input_context_contains_no_query_material():
    episode = real_episode()
    query_texts = [e.text for e in episode.events
                   if e.role in (Role.QUERY_TASK, Role.TRANSFER_TASK)]
    answers = [str(rec["instance"]["answer_canonical"])
               for rec in episode.items.values()
               if rec["role"] in ("query", "transfer")]
    assert query_texts and answers
    for index in feedback_event_indices(episode):
        keep = head_context_indices(episode, index)
        text = render_subset(episode, keep).text
        for prompt in query_texts:
            assert prompt not in text
        # the context must also end exactly at item j
        assert keep[-1] == index
        assert max(keep) < min(i for i, e in enumerate(episode.events)
                               if e.role is Role.QUERY_TASK)


def test_plan_that_ablates_a_query_event_is_refused():
    episode = real_episode()
    query_block = next(i for i, e in enumerate(episode.events)
                       if e.role is Role.QUERY_TASK)
    query_attempt = query_block + 1
    assert episode.events[query_attempt].role is Role.MODEL_ATTEMPT
    base = ablation_plan_for_episode(episode)
    for victim, label in ((query_block, "QUERY_BLOCK"),
                          (query_attempt, "QUERY_ATTEMPT")):
        plan = dict(base)
        plan["plans"] = list(base["plans"]) + [{
            "ablation_id": f"ABLATE_{label}",
            "ablated_event_indices": [victim],
            "ablated_item_id": episode.events[victim].item_id,
            "ablated_role": episode.events[victim].role.value}]
        with pytest.raises(AblationPlanError):
            validate_ablation_plan(episode, plan)
        with pytest.raises(AblationPlanError):
            episode_targets(tiny_bundle(), episode, plan)


def test_plan_from_another_episode_is_refused():
    episode = real_episode()
    other = ablation_plan_for_episode(real_episode(index=1))
    with pytest.raises(AblationPlanError):
        validate_ablation_plan(episode, other)


def test_targets_refuse_development_and_sealed_families():
    bundle = tiny_bundle()
    for family in (DEV_FAMILY, SEALED_FAMILY):
        episode = real_episode(family)
        with pytest.raises(SplitRefusedError):
            episode_targets(bundle, episode)
        with pytest.raises(SplitRefusedError):
            build_value_dataset(bundle, [episode])


def test_a_relabelled_sealed_episode_is_still_refused():
    """Smuggling attempt: claim TRAIN in the record, keep the sealed family."""
    episode = dataclasses.replace(real_episode(SEALED_FAMILY), split="TRAIN")
    with pytest.raises(SplitRefusedError):
        episode_targets(tiny_bundle(), episode)
