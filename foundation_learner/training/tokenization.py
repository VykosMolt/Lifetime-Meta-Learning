"""Episode -> training example, per arm (contract §7, §23).

Contract §23 entry point::

    episode_to_training_example(ep, tokenizer, arm) -> {input_ids, loss_mask, spans_meta}

Loss-mask convention (frozen here, documented once)
---------------------------------------------------
``loss_mask[i]`` is the weight of PREDICTING ``input_ids[i]`` given
``input_ids[:i]`` (i.e. target-aligned, not input-aligned).  The trainer shifts
logits by one position accordingly.  ``loss_mask[0]`` is always 0.0 because the
first token has no context; a weighted span that begins at position 0 is a hard
error rather than a silent loss of weight.

Loss-bearing span (frozen here, documented once)
------------------------------------------------
For every ``MODEL_ATTEMPT`` event the renderer's span map gives the event's
block ``[char_start, char_end)``.  The event's OWN text is located exactly
inside that block (it must occur exactly once), and the §4 answer grammar
``^ANSWER:\\s*(\\S.*?)\\s*$`` is applied LINE-WISE to that text — the same
string the verifier parses.  Each matching line contributes a loss-bearing
span covering the FULL ``ANSWER: <payload>`` line: from the ``A`` of ``ANSWER``
through the last non-whitespace character of the payload; the terminating
newline is excluded.

Rationale for "full line" rather than "payload only": byte-level BPE tokens
routinely straddle the ``": "`` / payload boundary (e.g. ``ĠYES``), so a
payload-only definition would require splitting tokens.  Locating the event
text rather than scanning rendered lines keeps the rule exact for BOTH block
layouts a renderer may choose (``MARKER\\n<text>\\n`` and ``MARKER <text>\\n``).

Straddling tokens are handled by an explicit policy, never silently:
``on_straddle="expand_whitespace"`` (default) includes a straddling token when
everything it contributes outside the span is WHITESPACE — which is exactly
the ``ĠANSWER``-style token produced when a renderer puts the marker and the
answer on one line, and which the model must emit anyway.  Any other straddle
is a hard error.

The standard episode structure has exactly one ``ANSWER:`` line per attempt
event, in which case "every matching line" equals the §4 last-line grammar; the
rule also stays exact if an attempt event carries several sub-items (Q1..Q3).

Arm weights (contract §7)
-------------------------
* ``FL1`` — isolated (TASK -> ANSWER) pairs, no history, no feedback; the
  answer span has weight 1.0.
* ``FL2`` — complete successful episodes; EVERY ``MODEL_ATTEMPT`` answer span
  has weight 1.0 (attempts, revisions, related, queries, transfers).  No
  future-competence weighting whatsoever.
* ``FL3`` — ordered episodes; answer spans are weighted by interaction index:
  QUERY (5) 1.0, TRANSFER (6) 1.0, revisions (1, 3) 0.25, attempt0 (0, 2) 0.0,
  everything else 0.0.  All ``TASK``/``FEEDBACK``/``HINT``/``REVEAL`` context
  is masked (weight 0.0).

Contract §7 does not name the related-task attempt (interaction index 4); the
frozen instruction "everything else 0.0" resolves it to 0.0.  Recorded in
``docs/CONTRACT_AMENDMENTS.md`` (Amendment 5).
"""

from __future__ import annotations

import copy
import dataclasses
import re
from typing import Any, Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Roles / arms (contract §6, §23)
# ---------------------------------------------------------------------------

ROLE_TASK = "TASK"
ROLE_SUPPORT_OBS = "SUPPORT_OBS"
ROLE_MODEL_ATTEMPT = "MODEL_ATTEMPT"
ROLE_FEEDBACK = "FEEDBACK"
ROLE_HINT = "HINT"
ROLE_REVEAL = "REVEAL"
ROLE_ADAPT_EVENT = "ADAPT_EVENT"
ROLE_RELATED_TASK = "RELATED_TASK"
ROLE_QUERY_TASK = "QUERY_TASK"
ROLE_TRANSFER_TASK = "TRANSFER_TASK"

ALL_ROLES = (
    ROLE_TASK,
    ROLE_SUPPORT_OBS,
    ROLE_MODEL_ATTEMPT,
    ROLE_FEEDBACK,
    ROLE_HINT,
    ROLE_REVEAL,
    ROLE_ADAPT_EVENT,
    ROLE_RELATED_TASK,
    ROLE_QUERY_TASK,
    ROLE_TRANSFER_TASK,
)

#: Roles that state a problem (used to build FL1 isolated pairs).
TASK_ROLES = (ROLE_TASK, ROLE_RELATED_TASK, ROLE_QUERY_TASK, ROLE_TRANSFER_TASK)

#: Roles that FL1 (static baseline) must never see (contract §7, hostile-tested).
HISTORY_ROLES = (ROLE_FEEDBACK, ROLE_HINT, ROLE_REVEAL, ROLE_ADAPT_EVENT)

ARM_FL1 = "FL1"
ARM_FL2 = "FL2"
ARM_FL3 = "FL3"
SUPPORTED_ARMS = (ARM_FL1, ARM_FL2, ARM_FL3)

#: Contract §7 FL3 weights, keyed by MODEL_ATTEMPT interaction index.
FL3_INTERACTION_WEIGHTS: dict[int, float] = {
    0: 0.0,  # attempt0 on P1
    1: 0.25,  # revision on P1
    2: 0.0,  # attempt0 on P2
    3: 0.25,  # revision on P2
    4: 0.0,  # related-task attempt (Amendment 5: "everything else 0.0")
    5: 1.0,  # QUERY
    6: 1.0,  # TRANSFER
}

FL2_ATTEMPT_WEIGHT = 1.0
FL1_ANSWER_WEIGHT = 1.0

#: §4 answer grammar, applied line-wise inside an event span.
ANSWER_LINE_RE = re.compile(r"^ANSWER:\s*(\S.*?)\s*$")

#: §6 frozen markers used by the minimal isolated-pair renderer.
TASK_MARKER = "TASK:"
ANSWER_MARKER = "ANSWER:"

MAX_SEQ_LEN = 2048  # contract §7 sequence budget

#: Straddle policy used by the mask builders (see the module docstring).
MASK_STRADDLE_POLICY = "expand_whitespace"


class TokenizationError(RuntimeError):
    """Base class for tokenisation / masking failures."""


class SequenceTooLongError(TokenizationError):
    """Rendered example exceeds the frozen sequence budget (no truncation)."""


class SpanAlignmentError(TokenizationError):
    """A loss span does not align with tokenizer boundaries."""


class MissingCanonicalAnswerError(TokenizationError):
    """FL1 item has no recoverable canonical answer."""


class StaticArmHistoryError(TokenizationError):
    """FL1 (static baseline) data contains history/feedback events."""


# ---------------------------------------------------------------------------
# Duck-typed access to W1's episode objects (contract §23 shapes)
# ---------------------------------------------------------------------------


def role_name(event: Any) -> str:
    role = getattr(event, "role", None)
    if role is None and isinstance(event, dict):
        role = event.get("role")
    value = getattr(role, "value", role)
    name = str(value)
    if name not in ALL_ROLES:
        # Enum members may stringify as "Role.TASK".
        tail = name.rsplit(".", 1)[-1]
        if tail in ALL_ROLES:
            return tail
        raise TokenizationError(f"unknown event role {role!r}")
    return name


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def event_text(event: Any) -> str:
    return str(_attr(event, "text", ""))


def event_meta(event: Any) -> dict:
    meta = _attr(event, "meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def _replace_event_text(event: Any, text: str) -> Any:
    if dataclasses.is_dataclass(event) and not isinstance(event, type):
        return dataclasses.replace(event, text=text)
    if isinstance(event, dict):
        new = dict(event)
        new["text"] = text
        return new
    new = copy.copy(event)
    setattr(new, "text", text)
    return new


def _replace_events(episode: Any, events: Sequence[Any]) -> Any:
    if dataclasses.is_dataclass(episode) and not isinstance(episode, type):
        return dataclasses.replace(episode, events=list(events))
    if isinstance(episode, dict):
        new = dict(episode)
        new["events"] = list(events)
        return new
    new = copy.copy(episode)
    setattr(new, "events", list(events))
    return new


def _default_render_fn():
    """Lazily bind W1's renderer (contract §23 ``episodes/render.py``)."""
    from foundation_learner.episodes.render import render_episode  # noqa: PLC0415

    return render_episode


def _normalise_spans(
    rendered: Any, included: Sequence[int]
) -> dict[int, tuple[int, int]]:
    """Accept the plausible shapes of ``Rendered.spans`` and normalise them."""
    spans = _attr(rendered, "spans", None)
    if spans is None:
        raise TokenizationError("renderer returned no span map")
    out: dict[int, tuple[int, int]] = {}
    if isinstance(spans, dict):
        for key, value in spans.items():
            out[int(key)] = _span_pair(value)
        return out
    seq = list(spans)
    if len(seq) != len(included):
        raise TokenizationError(
            f"renderer returned {len(seq)} spans for {len(included)} included events"
        )
    for event_index, value in zip(included, seq):
        idx = event_index
        if isinstance(value, dict) and "event_index" in value:
            idx = int(value["event_index"])
        elif not isinstance(value, (tuple, list)) and hasattr(value, "event_index"):
            idx = int(getattr(value, "event_index"))
        out[idx] = _span_pair(value)
    return out


def _span_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, dict):
        for a, b in (("char_start", "char_end"), ("start", "end")):
            if a in value and b in value:
                return int(value[a]), int(value[b])
        raise TokenizationError(f"unsupported span mapping {value!r}")
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            return int(value[0]), int(value[1])
        raise TokenizationError(f"span must be a (start, end) pair, got {value!r}")
    for a, b in (("char_start", "char_end"), ("start", "end")):
        start = getattr(value, a, None)
        end = getattr(value, b, None)
        if start is not None and end is not None:
            return int(start), int(end)
    raise TokenizationError(f"unsupported span object {value!r}")


# ---------------------------------------------------------------------------
# Character span -> token span (exact)
# ---------------------------------------------------------------------------


def encode_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Encode without special tokens and return exact character offsets.

    The frozen Ouro tokenizer is a fast (Rust) tokenizer, so offsets come from
    the tokenizer itself.  A slow tokenizer without offsets is refused rather
    than silently approximated.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise TokenizationError(
            "a fast tokenizer with offset mapping is required; the frozen Ouro "
            "tokenizer is GPT2TokenizerFast. Refusing to approximate offsets."
        )
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    return list(enc["input_ids"]), offsets


STRADDLE_POLICIES = ("error", "expand_whitespace", "expand")


def char_span_to_token_span(
    offsets: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
    *,
    on_straddle: str = "error",
    what: str = "span",
    text: str | None = None,
) -> tuple[int, int]:
    """Map ``[char_start, char_end)`` to a half-open token range.

    A token is selected iff its own character range is fully contained in the
    requested span.  A token that overlaps the span without being contained
    (a "straddle") is handled by ``on_straddle``:

    ``"error"``               hard failure (the default for raw span queries);
    ``"expand_whitespace"``   include it iff every character it contributes
                              outside the span is whitespace (requires
                              ``text``); any other straddle is a hard failure;
    ``"expand"``              always include it.

    No policy ever drops a straddling token silently.
    """
    if on_straddle not in STRADDLE_POLICIES:
        raise TokenizationError(f"unknown on_straddle policy {on_straddle!r}")
    if char_end <= char_start:
        raise TokenizationError(f"{what}: empty character span [{char_start},{char_end})")
    if on_straddle == "expand_whitespace" and text is None:
        raise TokenizationError("expand_whitespace requires the rendered text")
    selected: list[int] = []
    straddles: list[tuple[int, tuple[int, int]]] = []
    for idx, (start, end) in enumerate(offsets):
        if end <= start:  # zero-width token (e.g. added special tokens)
            continue
        if start >= char_end or end <= char_start:
            continue
        if start >= char_start and end <= char_end:
            selected.append(idx)
        else:
            straddles.append((idx, (start, end)))
    if straddles and on_straddle != "expand":
        offending = []
        for idx, (start, end) in straddles:
            if on_straddle == "error":
                offending.append((idx, (start, end)))
                continue
            outside = (
                text[start : min(end, char_start)] + text[max(start, char_end) : end]
            )
            if outside.strip():
                offending.append((idx, (start, end), outside))
        if offending:
            raise SpanAlignmentError(
                f"{what}: character span [{char_start},{char_end}) does not align "
                f"with token boundaries; offending tokens {offending[:4]}"
            )
    if straddles:
        selected.extend(idx for idx, _ in straddles)
    if not selected:
        raise SpanAlignmentError(
            f"{what}: character span [{char_start},{char_end}) selected no tokens"
        )
    selected.sort()
    first, last = selected[0], selected[-1]
    if selected != list(range(first, last + 1)):
        raise SpanAlignmentError(f"{what}: non-contiguous token selection {selected}")
    return first, last + 1


def find_answer_line_spans(text: str, char_start: int, char_end: int) -> list[tuple[int, int]]:
    """All ``ANSWER: <payload>`` line spans inside ``[char_start, char_end)``.

    Lines are the TRUE lines of ``text`` (the §4 grammar is line-anchored on the
    full rendering, so a mid-line ``ANSWER:`` is not an answer).  A line counts
    when it BEGINS inside the event span.  Each returned span starts at the
    ``A`` of ``ANSWER`` and ends just past the last non-whitespace character of
    the payload; the terminating newline is excluded.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while pos <= len(text):
        nl = text.find("\n", pos)
        line_end = len(text) if nl < 0 else nl
        if char_start <= pos < char_end:
            line = text[pos:line_end]
            if ANSWER_LINE_RE.match(line) is not None:
                stripped_end = pos + len(line.rstrip())
                if stripped_end > char_end:
                    raise SpanAlignmentError(
                        f"ANSWER line at char {pos} runs past the end of its "
                        f"event span [{char_start},{char_end}); the renderer's "
                        "span map and its text disagree"
                    )
                spans.append((pos, stripped_end))
        if nl < 0:
            break
        pos = nl + 1
    return spans


def locate_event_text(
    rendered_text: str, char_start: int, char_end: int, event_text: str
) -> int:
    """Absolute offset of ``event_text`` inside its rendered block.

    Renderer-agnostic: the block may be ``MARKER\\n<text>\\n`` or
    ``MARKER <text>\\n``.  The event text must occur EXACTLY once inside the
    block; anything else means the renderer transformed the event text and the
    mask can no longer be derived exactly, which is a hard error.
    """
    if not event_text:
        raise TokenizationError("event has empty text; cannot locate its answer")
    block = rendered_text[char_start:char_end]
    count = block.count(event_text)
    if count != 1:
        raise SpanAlignmentError(
            f"event text occurs {count} times in its rendered block "
            f"[{char_start},{char_end}); expected exactly once"
        )
    return char_start + block.index(event_text)


def answer_spans_for_event(
    rendered_text: str, char_start: int, char_end: int, event_text: str
) -> list[tuple[int, int]]:
    """Absolute ``ANSWER:`` line spans of one event, located via its own text."""
    base = locate_event_text(rendered_text, char_start, char_end, event_text)
    spans = []
    for start, end in find_answer_line_spans(event_text, 0, len(event_text)):
        spans.append((base + start, base + end))
    return spans


# ---------------------------------------------------------------------------
# Weighted-span extraction per arm
# ---------------------------------------------------------------------------


def _attempt_weight(arm: str, interaction_index: int | None) -> float:
    if arm == ARM_FL2:
        return FL2_ATTEMPT_WEIGHT
    if arm == ARM_FL3:
        if interaction_index is None:
            raise TokenizationError(
                "FL3 requires an interaction_index on every MODEL_ATTEMPT event"
            )
        try:
            return FL3_INTERACTION_WEIGHTS[int(interaction_index)]
        except KeyError:
            raise TokenizationError(
                f"FL3: no frozen weight for interaction index {interaction_index}; "
                f"EPISODE_STRUCTURE_V0 defines 0..6"
            ) from None
    if arm == ARM_FL1:
        return FL1_ANSWER_WEIGHT
    raise TokenizationError(f"unknown arm {arm!r}; supported: {list(SUPPORTED_ARMS)}")


def _build_example(
    *,
    arm: str,
    text: str,
    weighted_spans: list[dict],
    tokenizer,
    max_seq_len: int,
    on_straddle: str,
    episode: Any = None,
    item_index: int | None = None,
) -> dict:
    input_ids, offsets = encode_with_offsets(tokenizer, text)
    if len(input_ids) > max_seq_len:
        raise SequenceTooLongError(
            f"rendered example is {len(input_ids)} tokens > max_seq_len {max_seq_len} "
            "(episodes are sized to fit; truncation is forbidden by contract §7)"
        )
    loss_mask = [0.0] * len(input_ids)
    spans_meta: list[dict] = []
    for span in weighted_spans:
        start, end = char_span_to_token_span(
            offsets,
            span["char_start"],
            span["char_end"],
            on_straddle=on_straddle,
            what=f"{arm} answer span (event {span.get('event_index')})",
            text=text,
        )
        if start == 0 and span["weight"] > 0.0:
            raise TokenizationError(
                "a weighted answer span starts at token 0 (no context to predict "
                "from); the rendering is malformed"
            )
        for idx in range(start, end):
            loss_mask[idx] = float(span["weight"])
        spans_meta.append(
            {
                **span,
                "token_start": start,
                "token_end": end,
                "n_tokens": end - start,
            }
        )
    total_weight = float(sum(loss_mask))
    return {
        "arm": arm,
        "episode_id": _attr(episode, "episode_id"),
        "family_id": _attr(episode, "family_id"),
        "split": _attr(episode, "split"),
        "condition": _attr(episode, "condition"),
        "item_index": item_index,
        "text": text,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "spans_meta": spans_meta,
        "n_tokens": len(input_ids),
        "n_loss_tokens": sum(1 for w in loss_mask if w > 0.0),
        "total_weight": total_weight,
    }


def _history_episode_example(
    episode: Any,
    tokenizer,
    arm: str,
    *,
    render_fn: Callable | None,
    max_seq_len: int,
    on_straddle: str,
) -> dict:
    """FL2/FL3: one example per complete rendered episode."""
    render = render_fn or _default_render_fn()
    events = list(_attr(episode, "events", []) or [])
    included = list(range(len(events)))
    rendered = render(episode)
    text = str(_attr(rendered, "text"))
    spans = _normalise_spans(rendered, included)

    weighted: list[dict] = []
    for event_index, event in enumerate(events):
        if role_name(event) != ROLE_MODEL_ATTEMPT:
            continue
        if event_index not in spans:
            raise TokenizationError(
                f"renderer produced no span for MODEL_ATTEMPT event {event_index}"
            )
        char_start, char_end = spans[event_index]
        interaction_index = _attr(event, "interaction_index")
        weight = _attempt_weight(arm, interaction_index)
        answer_spans = answer_spans_for_event(
            text, char_start, char_end, event_text(event)
        )
        if not answer_spans:
            raise TokenizationError(
                f"MODEL_ATTEMPT event {event_index} contains no "
                f"'ANSWER: <payload>' line in its rendered span"
            )
        for line_start, line_end in answer_spans:
            weighted.append(
                {
                    "event_index": event_index,
                    "role": ROLE_MODEL_ATTEMPT,
                    "interaction_index": (
                        int(interaction_index) if interaction_index is not None else None
                    ),
                    "item_id": _attr(event, "item_id"),
                    "char_start": line_start,
                    "char_end": line_end,
                    "weight": float(weight),
                }
            )
    if not weighted:
        raise TokenizationError(
            f"{arm}: episode produced no MODEL_ATTEMPT answer spans"
        )
    example = _build_example(
        arm=arm,
        text=text,
        weighted_spans=weighted,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        on_straddle=on_straddle,
        episode=episode,
    )
    if example["total_weight"] <= 0.0:
        raise TokenizationError(
            f"{arm}: episode has zero total loss weight; it would contribute "
            "nothing and must not enter a batch"
        )
    return example


def _item_record_answer(episode: Any, item_id: Any) -> str | None:
    """Canonical answer from the episode's item table, if it has one.

    Accepts the shapes actually produced by ``episodes/assemble.py``
    (``episode.items[item_id]["instance"]["answer_canonical"]``) as well as a
    flat ``{item_id: {"answer_canonical": ...}}`` mapping and objects carrying
    an ``answer_canonical`` attribute.
    """
    items = _attr(episode, "items", None)
    if not isinstance(items, dict) or item_id is None:
        return None
    record = items.get(item_id)
    for candidate in (record, (record or {}).get("instance") if isinstance(record, dict) else None):
        if candidate is None:
            continue
        if isinstance(candidate, dict) and candidate.get("answer_canonical"):
            return str(candidate["answer_canonical"])
        value = getattr(candidate, "answer_canonical", None)
        if value:
            return str(value)
    return None


def _canonical_answer_for_item(
    episode: Any, task_index: int, events: Sequence[Any]
) -> str:
    """Resolve the canonical answer of an FL1 item (explicit precedence).

    1. ``episode.items[item_id]`` (the assembler's item table)
    2. ``task_event.meta["answer_canonical"]``
    3. ``attempt_event.meta["answer_canonical"]`` of a matching MODEL_ATTEMPT
    4. the ``ANSWER:`` payload of a matching MODEL_ATTEMPT explicitly marked
       ``meta["correct"] is True``

    Anything else is a hard error: FL1 must be trained on correct answers, and
    guessing one would silently corrupt the static baseline.
    """
    task_event = events[task_index]
    item_id = _attr(task_event, "item_id")
    from_items = _item_record_answer(episode, item_id)
    if from_items:
        return from_items
    meta = event_meta(task_event)
    if meta.get("answer_canonical"):
        return str(meta["answer_canonical"])
    candidates: list[Any] = []
    for idx in range(task_index + 1, len(events)):
        event = events[idx]
        if role_name(event) in TASK_ROLES:
            break
        if role_name(event) != ROLE_MODEL_ATTEMPT:
            continue
        if item_id is not None and _attr(event, "item_id") not in (None, item_id):
            continue
        candidates.append(event)
    for event in candidates:
        emeta = event_meta(event)
        if emeta.get("answer_canonical"):
            return str(emeta["answer_canonical"])
    for event in candidates:
        emeta = event_meta(event)
        if emeta.get("correct") is True:
            lines = find_answer_line_spans(event_text(event), 0, len(event_text(event)))
            if lines:
                start, end = lines[-1]
                match = ANSWER_LINE_RE.match(event_text(event)[start:end])
                if match:
                    return match.group(1)
    raise MissingCanonicalAnswerError(
        f"FL1: no canonical answer for task event {task_index} "
        f"(item_id={item_id!r}); provide it through episode.items[item_id], "
        "meta['answer_canonical'] on the task event, or a MODEL_ATTEMPT with "
        "meta['correct'] is True"
    )


def assert_no_history(episode: Any) -> None:
    """Contract §7/§18: the static arm must never receive history/feedback."""
    offending = [
        (idx, role_name(event))
        for idx, event in enumerate(_attr(episode, "events", []) or [])
        if role_name(event) in HISTORY_ROLES
    ]
    if offending:
        raise StaticArmHistoryError(
            "REFUSED: FL1 is the STATIC baseline and must contain no history "
            f"events; found {offending} in episode "
            f"{_attr(episode, 'episode_id')!r}"
        )


def _isolated_pair_examples(
    episode: Any,
    tokenizer,
    *,
    render_fn: Callable | None,
    max_seq_len: int,
    on_straddle: str,
) -> list[dict]:
    """FL1: one example per task-bearing item, rendered without history."""
    render = render_fn or _default_render_fn()
    events = list(_attr(episode, "events", []) or [])
    examples: list[dict] = []
    item_index = 0
    for task_index, task_event in enumerate(events):
        if role_name(task_event) not in TASK_ROLES:
            continue
        answer = _canonical_answer_for_item(episode, task_index, events)
        attempt_event = None
        for idx in range(task_index + 1, len(events)):
            if role_name(events[idx]) == ROLE_MODEL_ATTEMPT:
                attempt_event = events[idx]
                break
            if role_name(events[idx]) in TASK_ROLES:
                break
        if attempt_event is None:
            raise TokenizationError(
                f"FL1: task event {task_index} has no following MODEL_ATTEMPT "
                "event to use as the answer slot"
            )
        answer_event = _replace_event_text(
            attempt_event, f"{ANSWER_MARKER} {answer}"
        )
        pair_episode = _replace_events(episode, [task_event, answer_event])
        rendered = render(pair_episode)
        text = str(_attr(rendered, "text"))
        spans = _normalise_spans(rendered, [0, 1])
        char_start, char_end = spans[1]
        answer_spans = answer_spans_for_event(
            text, char_start, char_end, event_text(answer_event)
        )
        if len(answer_spans) != 1:
            raise TokenizationError(
                f"FL1: expected exactly one ANSWER line in the isolated pair, "
                f"got {len(answer_spans)}"
            )
        line_start, line_end = answer_spans[0]
        example = _build_example(
            arm=ARM_FL1,
            text=text,
            weighted_spans=[
                {
                    "event_index": 1,
                    "role": ROLE_MODEL_ATTEMPT,
                    "interaction_index": _attr(attempt_event, "interaction_index"),
                    "item_id": _attr(task_event, "item_id"),
                    "char_start": line_start,
                    "char_end": line_end,
                    "weight": FL1_ANSWER_WEIGHT,
                }
            ],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            on_straddle=on_straddle,
            episode=episode,
            item_index=item_index,
        )
        examples.append(example)
        item_index += 1
    if not examples:
        raise TokenizationError("FL1: episode contains no task-bearing events")
    return examples


def episode_to_training_examples(
    episode: Any,
    tokenizer,
    arm: str,
    *,
    render_fn: Callable | None = None,
    max_seq_len: int = MAX_SEQ_LEN,
    on_straddle: str = MASK_STRADDLE_POLICY,
) -> list[dict]:
    """All training examples produced by one episode under ``arm``.

    FL1 yields one example per isolated item (that is what "isolated pairs"
    means); FL2/FL3 yield exactly one whole-episode example.
    """
    if arm not in SUPPORTED_ARMS:
        raise TokenizationError(f"unknown arm {arm!r}; supported: {list(SUPPORTED_ARMS)}")
    if arm == ARM_FL1:
        assert_no_history(episode)
        return _isolated_pair_examples(
            episode,
            tokenizer,
            render_fn=render_fn,
            max_seq_len=max_seq_len,
            on_straddle=on_straddle,
        )
    return [
        _history_episode_example(
            episode,
            tokenizer,
            arm,
            render_fn=render_fn,
            max_seq_len=max_seq_len,
            on_straddle=on_straddle,
        )
    ]


def episode_to_training_example(
    episode: Any,
    tokenizer,
    arm: str,
    *,
    item_index: int = 0,
    render_fn: Callable | None = None,
    max_seq_len: int = MAX_SEQ_LEN,
    on_straddle: str = MASK_STRADDLE_POLICY,
) -> dict:
    """Contract §23 interface (single example).

    For FL2/FL3 this is the whole-episode example.  For FL1 an episode maps to
    several isolated pairs; ``item_index`` selects one (default: the first).
    Use :func:`episode_to_training_examples` to obtain all of them.
    """
    examples = episode_to_training_examples(
        episode,
        tokenizer,
        arm,
        render_fn=render_fn,
        max_seq_len=max_seq_len,
        on_straddle=on_straddle,
    )
    if arm != ARM_FL1 and len(examples) != 1:
        raise TokenizationError(
            f"{arm}: expected exactly one example per episode, got {len(examples)}"
        )
    if not 0 <= item_index < len(examples):
        raise TokenizationError(
            f"item_index {item_index} out of range (episode yields "
            f"{len(examples)} examples under {arm})"
        )
    return examples[item_index]


def render_isolated_pair(prompt_text: str, answer_payload: str) -> tuple[str, tuple[int, int]]:
    """Minimal §6-marker rendering of a pool item; returns (text, answer span)."""
    prefix = f"{TASK_MARKER} {prompt_text.rstrip()}\n"
    answer_line = f"{ANSWER_MARKER} {answer_payload.strip()}"
    text = prefix + answer_line + "\n"
    return text, (len(prefix), len(prefix) + len(answer_line))


def isolated_pair_to_training_example(
    prompt_text: str,
    answer_payload: str,
    tokenizer,
    *,
    render_pair_fn: Callable[[str, str], tuple[str, tuple[int, int]]] | None = None,
    max_seq_len: int = MAX_SEQ_LEN,
    on_straddle: str = MASK_STRADDLE_POLICY,
    meta: dict | None = None,
) -> dict:
    """FL1 example built directly from a pool item (contract §7 data source).

    IMPORTANT for the core comparison: FL1 must present the SAME textual
    surface as FL2/FL3, otherwise the baseline is confounded by formatting.
    The episode-derived path (:func:`episode_to_training_examples`) gets this
    for free because it renders through ``episodes/render.py`` (instruction
    header included).  This pool primitive uses a minimal two-line rendering
    with the §6 markers and NO instruction header, so a caller that feeds FL1
    from ``data/pools.py`` must pass ``render_pair_fn`` producing the same
    surface as the episode renderer.
    """
    render = render_pair_fn or render_isolated_pair
    text, (char_start, char_end) = render(prompt_text, answer_payload)
    example = _build_example(
        arm=ARM_FL1,
        text=text,
        weighted_spans=[
            {
                "event_index": None,
                "role": ROLE_MODEL_ATTEMPT,
                "interaction_index": None,
                "item_id": (meta or {}).get("item_id"),
                "char_start": char_start,
                "char_end": char_end,
                "weight": FL1_ANSWER_WEIGHT,
            }
        ],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        on_straddle=on_straddle,
    )
    if meta:
        example.update(
            {k: v for k, v in meta.items() if k not in ("input_ids", "loss_mask")}
        )
    return example


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def examples_to_batch(
    examples: Sequence[dict],
    pad_token_id: int,
    device: str = "cpu",
    dtype=None,
):
    """Right-pad a list of examples into tensors.

    Padding positions get ``attention_mask=0`` and ``loss_mask=0.0``.  Right
    padding is correct here because training never generates.
    """
    import torch

    if not examples:
        raise TokenizationError("cannot batch an empty example list")
    max_len = max(len(ex["input_ids"]) for ex in examples)
    batch_ids = []
    batch_attn = []
    batch_mask = []
    for ex in examples:
        ids = list(ex["input_ids"])
        mask = list(ex["loss_mask"])
        if len(ids) != len(mask):
            raise TokenizationError("input_ids and loss_mask length mismatch")
        pad = max_len - len(ids)
        batch_ids.append(ids + [int(pad_token_id)] * pad)
        batch_attn.append([1] * len(ids) + [0] * pad)
        batch_mask.append(mask + [0.0] * pad)
    return {
        "input_ids": torch.tensor(batch_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(batch_attn, dtype=torch.long, device=device),
        "loss_mask": torch.tensor(
            batch_mask, dtype=dtype or torch.float32, device=device
        ),
        "n_examples": len(examples),
        "n_padded_tokens": len(examples) * max_len,
        "n_real_tokens": sum(len(ex["input_ids"]) for ex in examples),
        "n_loss_tokens": sum(ex["n_loss_tokens"] for ex in examples),
    }


def iter_examples(
    episodes: Iterable[Any],
    tokenizer,
    arm: str,
    **kwargs,
):
    """Convenience generator over many episodes (deterministic order preserved)."""
    for episode in episodes:
        yield from episode_to_training_examples(episode, tokenizer, arm, **kwargs)
