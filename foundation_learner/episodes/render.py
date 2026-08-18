"""Episode rendering with an exact event -> character span map (contract §6).

Plain text with fixed markers and no chat template.  ``render_events`` returns
the rendered text plus, for every event, the exact half-open character span its
block occupies, so loss masks and tap points are derived from the same code
path that produced the text.  A frozen terse instruction header precedes each
episode.

Rendering an event SUBSET is a first-class operation: FL4 ablation plans and
the context-reset evaluation both need "the same episode minus these events",
and both must produce a span map for what remains.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Episode, Event, Role

__all__ = ["INSTRUCTION_HEADER_V0", "MARKERS", "EventSpan", "RenderedEpisode",
           "render_events", "render_episode", "render_subset",
           "render_reset_query"]

INSTRUCTION_HEADER_V0 = (
    "You are working through a series of problems that share one hidden rule.\n"
    "Each ATTEMPT must end with a line of the form: ANSWER: <payload>\n"
    "FEEDBACK is verified. HINT is not verified and can be wrong.\n"
)

MARKERS: dict[Role, str] = {
    Role.TASK: "TASK:",
    Role.SUPPORT_OBS: "OBSERVATION:",
    Role.MODEL_ATTEMPT: "ATTEMPT:",
    Role.FEEDBACK: "FEEDBACK:",
    Role.HINT: "HINT:",
    Role.REVEAL: "REVEALED ANSWER",
    Role.ADAPT_EVENT: "ADAPT:",
    Role.RELATED_TASK: "RELATED TASK:",
    Role.QUERY_TASK: "QUERY:",
    Role.TRANSFER_TASK: "TRANSFER:",
}


@dataclass(frozen=True)
class EventSpan:
    """Half-open ``[start, end)`` character span of one event's block."""

    event_index: int
    role: str
    item_id: str
    interaction_index: int | None
    start: int
    end: int


@dataclass(frozen=True)
class RenderedEpisode:
    text: str
    spans: tuple[EventSpan, ...]
    header_end: int

    def span_text(self, span: EventSpan) -> str:
        return self.text[span.start:span.end]

    @property
    def char_spans(self) -> tuple[tuple[int, int], ...]:
        """Contract §23 view of the span map: ``(char_start, char_end)`` pairs."""
        return tuple((s.start, s.end) for s in self.spans)


def _block(event: Event) -> str:
    marker = MARKERS[event.role]
    if event.role is Role.REVEAL:
        label = f" ({event.slot})" if event.slot else ""
        return f"{marker}{label}: {event.text}\n"
    return f"{marker} {event.text}\n"


def render_events(events, include_header: bool = True) -> RenderedEpisode:
    """Render an ordered event sequence (possibly a subset) with span map."""
    parts: list[str] = []
    spans: list[EventSpan] = []
    cursor = 0
    if include_header:
        parts.append(INSTRUCTION_HEADER_V0)
        cursor = len(INSTRUCTION_HEADER_V0)
    header_end = cursor
    for index, event in enumerate(events):
        block = _block(event)
        spans.append(EventSpan(
            event_index=index, role=event.role.value, item_id=event.item_id,
            interaction_index=event.interaction_index,
            start=cursor, end=cursor + len(block)))
        parts.append(block)
        cursor += len(block)
    text = "".join(parts)
    if cursor != len(text):  # pragma: no cover - arithmetic invariant
        raise AssertionError("span cursor disagrees with rendered length")
    return RenderedEpisode(text=text, spans=tuple(spans), header_end=header_end)


def render_episode(episode: Episode, include_header: bool = True,
                   include_event_indices=None,
                   upto_event: int | None = None) -> RenderedEpisode:
    """Render an episode, optionally restricted to a subset or a prefix.

    ``include_event_indices`` and ``upto_event`` are the contract §23 selectors;
    they compose (the subset is intersected with the prefix).  Spans always
    carry the ORIGINAL episode event indices.
    """
    keep = list(range(len(episode.events))) if include_event_indices is None \
        else sorted(int(i) for i in include_event_indices)
    if upto_event is not None:
        keep = [i for i in keep if i <= int(upto_event)]
    if include_event_indices is None and upto_event is None:
        return render_events(episode.events, include_header=include_header)
    return render_subset(episode, keep, include_header=include_header)


def render_subset(episode: Episode, keep_indices,
                  include_header: bool = True) -> RenderedEpisode:
    """Render only ``keep_indices`` (ascending) of an episode's events.

    The returned spans carry the ORIGINAL episode event indices, so an ablation
    plan can be traced back to the events it removed.
    """
    keep = sorted(int(i) for i in keep_indices)
    if any(i < 0 or i >= len(episode.events) for i in keep):
        raise IndexError("keep_indices outside the episode's event range")
    rendered = render_events([episode.events[i] for i in keep],
                             include_header=include_header)
    spans = tuple(
        EventSpan(event_index=keep[s.event_index], role=s.role,
                  item_id=s.item_id, interaction_index=s.interaction_index,
                  start=s.start, end=s.end)
        for s in rendered.spans)
    return RenderedEpisode(text=rendered.text, spans=spans,
                           header_end=rendered.header_end)


def render_reset_query(episode: Episode, query_event_index: int,
                       include_header: bool = True) -> RenderedEpisode:
    """Context-reset prompt: the instruction header plus ONE query, nothing else.

    Contract §9 makes context reset load-bearing: the textual learning history
    must be gone.  This function refuses any event that is not a query-style
    task, so a caller cannot accidentally leave history in the reset prompt.
    """
    index = int(query_event_index)
    if not 0 <= index < len(episode.events):
        raise IndexError("query_event_index outside the episode")
    event = episode.events[index]
    if event.role not in (Role.QUERY_TASK, Role.TRANSFER_TASK,
                          Role.RELATED_TASK):
        raise ValueError(
            f"render_reset_query needs a query-style task event, got "
            f"{event.role.value}")
    return render_subset(episode, [index], include_header=include_header)
