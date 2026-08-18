"""Shared episode toolkit for the W2 (training core) tests.

The training core is written against the contract §23 episode interface, not
against a particular implementation.  This module binds the REAL
``foundation_learner.episodes`` modules when they are importable and otherwise
falls back to a minimal stand-in built to the §23 dataclass shape.

The fallback is IMPORT-LEVEL ONLY: if the real modules import successfully they
are used unconditionally, so an incompatibility between the real renderer and
the training core surfaces as a loud test failure instead of silently reverting
to the stand-in.  ``EPISODE_TOOLKIT_SOURCE`` records which binding is active
and is reported by :func:`test_reports_episode_toolkit_source`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

try:  # real W1 schema
    from foundation_learner.episodes.schema import Episode as _Episode  # type: ignore
    from foundation_learner.episodes.schema import Event as _Event  # type: ignore
    from foundation_learner.episodes.schema import Role as _Role  # type: ignore

    SCHEMA_SOURCE = "foundation_learner.episodes.schema"
except Exception:  # pragma: no cover - exercised while W1 is in flight
    _Episode = _Event = _Role = None
    SCHEMA_SOURCE = "stand-in (contract §23 shape)"

try:  # real W1 renderer
    from foundation_learner.episodes.render import (  # type: ignore
        render_episode as _render_episode,
    )

    RENDER_SOURCE = "foundation_learner.episodes.render"
except Exception:  # pragma: no cover - exercised while W1 is in flight
    _render_episode = None
    RENDER_SOURCE = "stand-in (contract §6 markers)"

EPISODE_TOOLKIT_SOURCE = {"schema": SCHEMA_SOURCE, "render": RENDER_SOURCE}


# ---------------------------------------------------------------------------
# Stand-in schema (contract §23)
# ---------------------------------------------------------------------------


@dataclass
class StubEvent:
    role: str
    text: str
    item_id: str = ""
    interaction_index: int | None = None
    certified: bool = True
    poison: bool = False
    reveal: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class StubEpisode:
    episode_id: str
    family_id: str
    split: str
    rule_id: str
    seed: int
    condition: str
    events: list
    structure: str = "EPISODE_STRUCTURE_V0"


#: Contract §6 frozen markers, used by the stand-in renderer only.
STUB_MARKERS = {
    "TASK": "TASK:",
    "SUPPORT_OBS": "OBSERVATION:",
    "MODEL_ATTEMPT": "ATTEMPT:",
    "FEEDBACK": "FEEDBACK:",
    "HINT": "HINT:",
    "REVEAL": "REVEALED ANSWER:",
    "ADAPT_EVENT": "ADAPT:",
    "RELATED_TASK": "RELATED TASK:",
    "QUERY_TASK": "QUERY:",
    "TRANSFER_TASK": "TRANSFER:",
}


@dataclass
class StubRendered:
    text: str
    spans: list


def stub_render_episode(ep, include_event_indices=None, upto_event=None) -> StubRendered:
    """Minimal §6 rendering: ``MARKER\\n<event text>\\n`` per event.

    ``spans[i]`` is the (char_start, char_end) of the whole block for the i-th
    INCLUDED event, excluding the block's terminating newline.
    """
    events = list(getattr(ep, "events", []))
    indices = list(range(len(events))) if include_event_indices is None else list(
        include_event_indices
    )
    if upto_event is not None:
        indices = [i for i in indices if i <= upto_event]
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for i in indices:
        event = events[i]
        role = getattr(event, "role", None)
        role = getattr(role, "value", role)
        role = str(role).rsplit(".", 1)[-1]
        block = f"{STUB_MARKERS[role]}\n{event.text}\n"
        parts.append(block)
        spans.append((pos, pos + len(block) - 1))
        pos += len(block)
    return StubRendered("".join(parts), spans)


Event = _Event or StubEvent
Episode = _Episode or StubEpisode
render_episode = _render_episode or stub_render_episode


def role(name: str):
    """Role value in whatever representation the active schema uses."""
    if _Role is not None:
        return getattr(_Role, name)
    return name


def _field_names(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def make_event(role_name: str, text: str, **kwargs):
    """Build an Event, keeping only the fields the ACTIVE schema declares.

    The real ``episodes/schema.py`` Event is frozen and carries ``slot`` but no
    ``meta``; the §23 stand-in carries ``meta``.  Callers pass the union and
    this function projects onto whichever schema is bound.
    """
    payload = {
        "role": role(role_name),
        "text": text,
        "item_id": "",
        "interaction_index": None,
        "certified": False,
        "poison": False,
        "reveal": False,
        "slot": "",
        "meta": {},
    }
    payload.update(kwargs)
    allowed = _field_names(Event)
    return Event(**{k: v for k, v in payload.items() if k in allowed})


def make_episode(**kwargs):
    """Build an Episode, keeping only the fields the ACTIVE schema declares."""
    payload = {
        "episode_id": "ep",
        "family_id": "modular_arithmetic",
        "split": "train",
        "rule_id": "rule0",
        "rule_spec": {},
        "seed": 0,
        "difficulty": 1,
        "condition": "clean",
        "mode": "scripted",
        "structure": "EPISODE_STRUCTURE_V0",
        "structured": True,
        "reveal": False,
        "events": [],
        "items": {},
        "meta": {},
    }
    payload.update(kwargs)
    allowed = _field_names(Episode)
    return Episode(**{k: v for k, v in payload.items() if k in allowed})


def replace_event(episode, index: int, **changes):
    """Return a copy of ``episode`` with one event replaced (events may be frozen)."""
    events = list(episode.events)
    events[index] = dataclasses.replace(events[index], **changes)
    return dataclasses.replace(episode, events=events)


def event_role_name(event) -> str:
    return str(getattr(event.role, "value", event.role)).rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Canonical test episodes
# ---------------------------------------------------------------------------


def make_ordered_episode(k: int = 0, *, history: bool = True, split: str = "train"):
    """An EPISODE_STRUCTURE_V0-shaped episode with interaction indices 0..6.

    ``history=False`` drops every FEEDBACK/HINT/REVEAL event, which is the
    shape FL1 (static baseline) data must have.  Canonical answers are carried
    BOTH in ``episode.items`` (the assembler's shape) and in per-event ``meta``
    when the active schema has that field.
    """
    events = []
    items: dict[str, dict] = {}

    def attempt(idx: int, item: str, answer: str, correct: bool):
        return make_event(
            "MODEL_ATTEMPT",
            f"ANSWER: {answer}",
            item_id=item,
            interaction_index=idx,
            slot=item,
            meta={"answer_canonical": answer, "correct": correct},
        )

    def task(role_name: str, item: str, prompt: str, answer: str, kind: str):
        items[item] = {
            "instance": {"answer_canonical": answer, "prompt_text": prompt},
            "slot": item,
            "role": kind,
        }
        return make_event(
            role_name,
            prompt,
            item_id=item,
            interaction_index=None,
            slot=item,
            meta={"answer_canonical": answer},
        )

    events.append(task("TASK", "P1", f"compute f({k})", str(k + 1), "support"))
    events.append(attempt(0, "P1", str(k + 9), False))
    if history:
        events.append(make_event("FEEDBACK", "INCORRECT", item_id="P1", certified=True))
        events.append(
            make_event("HINT", "residue distance 2", item_id="P1", certified=False)
        )
    events.append(attempt(1, "P1", str(k + 1), True))
    events.append(task("TASK", "P2", f"compute f({k + 1})", str(k + 2), "support"))
    events.append(attempt(2, "P2", str(k + 7), False))
    if history:
        events.append(make_event("FEEDBACK", "INCORRECT", item_id="P2", certified=True))
    events.append(attempt(3, "P2", str(k + 2), True))
    events.append(
        task("RELATED_TASK", "P3", f"compute f({k + 2})", str(k + 3), "related")
    )
    events.append(attempt(4, "P3", str(k + 3), True))
    events.append(task("QUERY_TASK", "Q1", f"compute f({k + 3})", str(k + 4), "query"))
    events.append(attempt(5, "Q1", str(k + 4), True))
    events.append(
        task("TRANSFER_TASK", "T1", f"compute f({k + 4})", str(k + 5), "transfer")
    )
    events.append(attempt(6, "T1", str(k + 5), True))

    return make_episode(
        episode_id=f"ep{k}",
        family_id="modular_arithmetic",
        split=split,
        rule_id="rule0",
        seed=k,
        condition="structured" if history else "isolated",
        events=events,
        items=items,
    )


def make_static_episode(k: int = 0, **kwargs):
    return make_ordered_episode(k, history=False, **kwargs)


# ---------------------------------------------------------------------------
# Tests for the toolkit itself
# ---------------------------------------------------------------------------

_CONTRACT_EVENT_FIELDS = {
    "role",
    "text",
    "item_id",
    "interaction_index",
    "certified",
    "poison",
    "reveal",
}
#: §23 lists ``meta`` on Event; ``episodes/schema.py`` carries ``slot`` instead
#: and keeps per-item data in ``Episode.items`` (Amendment 5).  The training
#: core reads both shapes, so ``meta`` is optional here.
_CONTRACT_EVENT_FIELDS_OPTIONAL = {"meta", "slot"}
_CONTRACT_EPISODE_FIELDS = {
    "episode_id",
    "family_id",
    "split",
    "rule_id",
    "seed",
    "condition",
    "events",
    "structure",
}


def test_active_schema_matches_contract_section_23():
    event_fields = {f.name for f in dataclasses.fields(Event)}
    episode_fields = {f.name for f in dataclasses.fields(Episode)}
    assert _CONTRACT_EVENT_FIELDS <= event_fields, sorted(
        _CONTRACT_EVENT_FIELDS - event_fields
    )
    assert event_fields & _CONTRACT_EVENT_FIELDS_OPTIONAL
    assert _CONTRACT_EPISODE_FIELDS <= episode_fields, sorted(
        _CONTRACT_EPISODE_FIELDS - episode_fields
    )


def test_ordered_episode_has_frozen_interaction_indices():
    ep = make_ordered_episode(3)
    indices = [
        e.interaction_index
        for e in ep.events
        if event_role_name(e) == "MODEL_ATTEMPT"
    ]
    assert indices == [0, 1, 2, 3, 4, 5, 6]


def test_static_episode_has_no_history_events():
    ep = make_static_episode(1)
    roles = {event_role_name(e) for e in ep.events}
    assert roles.isdisjoint({"FEEDBACK", "HINT", "REVEAL", "ADAPT_EVENT"})


def test_replace_event_produces_an_independent_episode():
    ep = make_ordered_episode(0)
    changed = replace_event(ep, 1, text="ANSWER: 999")
    assert changed.events[1].text == "ANSWER: 999"
    assert ep.events[1].text != "ANSWER: 999"


def test_reports_episode_toolkit_source(capsys):
    print(f"EPISODE_TOOLKIT_SOURCE = {EPISODE_TOOLKIT_SOURCE}")
    assert set(EPISODE_TOOLKIT_SOURCE) == {"schema", "render"}
