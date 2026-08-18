"""Rendering, the exact span map and subset rendering (contract §6)."""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import derive_seed, make_rng
from foundation_learner.ecology.families import all_families
from foundation_learner.ecology.split import split_of
from foundation_learner.episodes.assemble import assemble_episode
from foundation_learner.episodes.render import (INSTRUCTION_HEADER_V0, MARKERS,
                                                render_episode, render_events,
                                                render_subset)
from foundation_learner.episodes.schema import Role

FAMILIES = all_families()
IDS = [f.family_id for f in FAMILIES]


def _episode(family, reveal=False):
    seed = derive_seed(20260809, "render-test", family.family_id)
    rule = family.sample_rule(make_rng(seed))
    return assemble_episode(family, rule, split=split_of(family.family_id),
                            difficulty=1, root_seed=20260809,
                            episode_seed=seed, reveal=reveal)


def test_instruction_header_is_frozen():
    assert INSTRUCTION_HEADER_V0 == (
        "You are working through a series of problems that share one hidden "
        "rule.\n"
        "Each ATTEMPT must end with a line of the form: ANSWER: <payload>\n"
        "FEEDBACK is verified. HINT is not verified and can be wrong.\n")
    assert INSTRUCTION_HEADER_V0.endswith("\n")


def test_markers_cover_every_role_and_match_the_contract():
    assert set(MARKERS) == set(Role)
    assert MARKERS[Role.TASK] == "TASK:"
    assert MARKERS[Role.MODEL_ATTEMPT] == "ATTEMPT:"
    assert MARKERS[Role.FEEDBACK] == "FEEDBACK:"
    assert MARKERS[Role.HINT] == "HINT:"
    assert MARKERS[Role.REVEAL] == "REVEALED ANSWER"
    assert MARKERS[Role.RELATED_TASK] == "RELATED TASK:"
    assert MARKERS[Role.QUERY_TASK] == "QUERY:"
    assert MARKERS[Role.TRANSFER_TASK] == "TRANSFER:"


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_span_map_is_exact_and_contiguous(family):
    episode = _episode(family, reveal=True)
    rendered = render_episode(episode)
    assert rendered.text.startswith(INSTRUCTION_HEADER_V0)
    assert rendered.header_end == len(INSTRUCTION_HEADER_V0)
    assert len(rendered.spans) == len(episode.events)
    cursor = rendered.header_end
    for span, event in zip(rendered.spans, episode.events):
        assert span.start == cursor
        assert span.end > span.start
        cursor = span.end
        block = rendered.span_text(span)
        assert block.endswith("\n")
        assert event.text in block
        assert block.startswith(MARKERS[event.role])
        assert span.role == event.role.value
        assert span.item_id == event.item_id
        assert span.interaction_index == event.interaction_index
    assert cursor == len(rendered.text)
    assert rendered.text == INSTRUCTION_HEADER_V0 + "".join(
        rendered.span_text(s) for s in rendered.spans)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_reveal_marker_carries_the_support_slot(family):
    episode = _episode(family, reveal=True)
    rendered = render_episode(episode)
    for span, event in zip(rendered.spans, episode.events):
        if event.role is Role.REVEAL:
            assert rendered.span_text(span).startswith(
                f"REVEALED ANSWER ({event.slot}): ")


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_header_can_be_omitted(family):
    episode = _episode(family)
    rendered = render_events(episode.events, include_header=False)
    assert rendered.header_end == 0
    assert not rendered.text.startswith(INSTRUCTION_HEADER_V0)
    assert rendered.spans[0].start == 0


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_subset_rendering_keeps_original_indices(family):
    episode = _episode(family)
    keep = [i for i, e in enumerate(episode.events)
            if e.role is not Role.HINT]
    rendered = render_subset(episode, keep)
    assert [s.event_index for s in rendered.spans] == keep
    assert "HINT:" not in rendered.text
    full = render_episode(episode)
    assert len(rendered.text) < len(full.text)
    cursor = rendered.header_end
    for span in rendered.spans:
        assert span.start == cursor
        cursor = span.end
        assert rendered.span_text(span) == \
            full.span_text(full.spans[span.event_index])
    assert cursor == len(rendered.text)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_ablating_one_event_removes_exactly_that_block(family):
    episode = _episode(family)
    full = render_episode(episode)
    target = next(i for i, e in enumerate(episode.events)
                  if e.role is Role.FEEDBACK)
    keep = [i for i in range(len(episode.events)) if i != target]
    ablated = render_subset(episode, keep)
    removed = full.span_text(full.spans[target])
    assert len(ablated.text) == len(full.text) - len(removed)


def test_subset_rejects_out_of_range_indices():
    episode = _episode(FAMILIES[0])
    with pytest.raises(IndexError):
        render_subset(episode, [len(episode.events)])
    with pytest.raises(IndexError):
        render_subset(episode, [-1])


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_contract_23_selectors_and_char_spans(family):
    episode = _episode(family)
    full = render_episode(episode)
    assert full.char_spans == tuple((s.start, s.end) for s in full.spans)

    keep = [0, 1, 2]
    subset = render_episode(episode, include_event_indices=keep)
    assert [s.event_index for s in subset.spans] == keep

    prefix = render_episode(episode, upto_event=4)
    assert [s.event_index for s in prefix.spans] == [0, 1, 2, 3, 4]
    assert prefix.text == full.text[:prefix.spans[-1].end]

    both = render_episode(episode, include_event_indices=[0, 2, 6],
                          upto_event=2)
    assert [s.event_index for s in both.spans] == [0, 2]


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_render_reset_query_refuses_history_bearing_events(family):
    from foundation_learner.episodes.render import render_reset_query

    episode = _episode(family)
    query_index = next(i for i, e in enumerate(episode.events)
                       if e.role is Role.QUERY_TASK)
    rendered = render_reset_query(episode, query_index)
    body = rendered.text[rendered.header_end:]
    assert body.startswith("QUERY: ")
    assert len(rendered.spans) == 1
    assert rendered.spans[0].event_index == query_index
    for marker in ("TASK:", "ATTEMPT:", "FEEDBACK:", "HINT:", "RELATED TASK:",
                   "REVEALED ANSWER"):
        assert marker not in body
    attempt_index = next(i for i, e in enumerate(episode.events)
                         if e.role is Role.MODEL_ATTEMPT)
    feedback_index = next(i for i, e in enumerate(episode.events)
                          if e.role is Role.FEEDBACK)
    for bad in (attempt_index, feedback_index):
        with pytest.raises(ValueError):
            render_reset_query(episode, bad)
    with pytest.raises(IndexError):
        render_reset_query(episode, len(episode.events))


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_context_reset_rendering_leaves_only_the_query(family):
    """The machinery a context-reset evaluation needs: a query-only prompt."""
    episode = _episode(family)
    query_events = [i for i, e in enumerate(episode.events)
                    if e.role is Role.QUERY_TASK]
    rendered = render_subset(episode, query_events[:1], include_header=True)
    body = rendered.text[rendered.header_end:]
    assert body.startswith("QUERY: ")
    for marker in ("TASK:", "ATTEMPT:", "FEEDBACK:", "HINT:", "RELATED TASK:"):
        assert marker not in body
