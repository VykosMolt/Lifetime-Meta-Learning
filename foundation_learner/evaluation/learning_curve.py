"""ONLINE learning-curve harness over ``EPISODE_STRUCTURE_V0`` (contract 6/9/22).

What this module does
=====================
It walks a pre-generated episode as a PLAN, not as a transcript.  Every
``MODEL_ATTEMPT`` slot is filled by a REAL greedy generation from the model
under evaluation, every ``FEEDBACK`` / ``HINT`` / ``REVEAL`` slot is filled by
feedback that the exact family verifier really computes FROM THAT GENERATION,
and the ongoing context is re-rendered so that the model always reads its own
history.  The scripted attempt/feedback texts that the training-data generator
stored in the episode are never shown to the model here; they only supply the
plan (which items, in which order, with which certified/poison flags).

Result: per-episode ``R_0 .. R_6`` where ``R_5`` is the mean over the three
QUERY items and ``R_6`` the mean over the two TRANSFER items (the frozen
structure of contract section 6 puts three attempts at interaction index 5 and
two at index 6, and ``R_k`` is defined as the mean verifier success over the
attempts carried by index ``k``), plus a full transcript record
(``flb200.episode_record.v1``: prompts' hashes, generations, token ids,
finish reasons, verifier outcomes, feedback texts).

Eval-time sequence allowance and per-episode isolation (Amendment 13)
====================================================================
The ONLINE context is longer than the pre-generated rendering it walks: every
MODEL_ATTEMPT slot carries up to ``max_new_tokens`` REAL generated tokens
instead of the scripted answer line.  ``EVAL_ONLINE_MAX_SEQ_LEN`` (4096) is the
frozen EVAL-TIME allowance; the contract's 2048 remains the training-side
rendering budget and is unchanged.  An episode whose online context would still
exceed the allowance is NEVER truncated: it is recorded with
``status = "ONLINE_BUDGET_EXCEEDED"``, its ``R`` is marked missing (so no metric
consumes a partial curve), and every other episode in the batch continues.

Feedback channel policy (frozen for the ONLINE evaluator)
=========================================================
* ``FEEDBACK`` is the CERTIFIED channel: it is always verifier-truthful and,
  under both the ``structured`` and ``correctness_only`` conditions, carries
  the correctness verdict computed from the model's real attempt.  It is never
  corrupted, in any condition.
* ``HINT`` is the UNCERTIFIED channel: it carries the family's structured
  feedback content, replaced by the family's corrupted variant exactly when
  the episode plan marks that event ``poison=True``.  Truthful and poisoned
  hints are rendered identically; the poison flag is environment-side only and
  never enters the text, the prompt, or any callback payload.
* Under ``correctness_only`` the walker drops HINT events entirely (contract
  section 6: hints exist "in structured conditions").
* ``REVEAL`` carries the canonical answer of a SUPPORT item only, and only
  where the episode plan contains such an event.

Adaptation callback protocol (implemented by the mechanisms worker)
===================================================================
The walker calls, in this order, per episode::

    cb.on_episode_start(ctx: EpisodeContext) -> None

    # ... for every MODEL_ATTEMPT slot, immediately before generation:
    cb.prefix_provider() -> torch.Tensor | None
        # None, or a [k, hidden_size] tensor prepended to THIS episode's row of
        # the batched decode through inputs_embeds (contract section 7 FL5 uses
        # k = 8, hidden 2048).  Episodes advance in lockstep, so every row of a
        # batch carries the prefix its own mechanism returned; rows with None
        # carry no prefix.  Decoding runs under torch.inference_mode, so the
        # tensor is read, never differentiated here.

    # ... after each generation is verified:
    cb.on_attempt(attempt: AttemptEvent) -> None
        # AttemptEvent(role, item_id, interaction_index, text, parsed)
        # It deliberately does NOT carry the verifier verdict: QUERY and
        # TRANSFER labels are hidden from the model and therefore from every
        # mechanism.  Correctness reaches a mechanism only through the
        # certified FEEDBACK text of SUPPORT items, exactly as the model sees
        # it.

    # ... for every FEEDBACK / HINT / REVEAL event, after it is rendered:
    cb.on_feedback(event: FeedbackEvent, hidden_summary_fn) -> None
        # FeedbackEvent(role, item_id, interaction_index, text, certified,
        #               event_index, char_span, context_sha256)
        # It deliberately does NOT carry the poison flag (environment-only)
        # nor any hidden label.
        # hidden_summary_fn(reduce="mean", include_prefix=False) -> Tensor
        #   reduce="mean"   -> [hidden]   mean of the FINAL-loop, final-layer
        #                                 hidden states over item-j tokens
        #                                 (the FL5 u_j input)
        #   reduce="last"   -> [hidden]   hidden state at the LAST token of
        #                                 item j (the FL4 head input)
        #   reduce="tokens" -> [n_j, hidden]
        # The forward is executed lazily on first call and cached per event;
        # a mechanism that ignores the hidden state costs nothing.
        # include_prefix=True re-runs the context with the mechanism's current
        # prefix prepended; the default False keeps the summary a pure
        # function of the textual context (no s -> prefix -> s recursion).

    cb.on_episode_end(ctx: EpisodeContext) -> None

:class:`NullCallbacks` implements all five as no-ops; mechanisms subclass it
and override what they need.  Episode-level state resetting is the
mechanism's responsibility inside ``on_episode_start``.
"""
from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import torch

from . import domain_sha256, text_sha256
from .generation import (
    GenerationConfig,
    GenerationRecord,
    complete_answer_line_end,
    greedy_generate_detailed,
    resolve_answer_parser,
)
from .scoring import align_char_span_covering, encode_with_offsets

EPISODE_RECORD_SCHEMA = "flb200.episode_record.v1"
EPISODE_RECORD_DOMAIN = "FL_V0_EPISODE_RECORD"
STRUCTURE_V0 = "EPISODE_STRUCTURE_V0"

#: Contract section 6: interaction indices 0..6, K = 6.
K_INTERACTIONS = 6
INTERACTION_INDICES: tuple[int, ...] = tuple(range(K_INTERACTIONS + 1))

TASK_ROLES = frozenset({"TASK", "RELATED_TASK", "QUERY_TASK", "TRANSFER_TASK"})
FEEDBACK_ROLES = frozenset({"FEEDBACK", "HINT", "REVEAL"})
HIDDEN_LABEL_ROLES = frozenset({"QUERY_TASK", "TRANSFER_TASK"})

ATTEMPT_SENTINEL = "FL_V0_ATTEMPT_SLOT"
DEFAULT_ATTEMPT_CUE = "ATTEMPT:"
#: Contract section 7: the budget every PRE-GENERATED episode RENDERING is
#: sized to fit.  It bounds the training-side sequence length and is unchanged.
MAX_SEQ_LEN = 2048

#: Frozen EVAL-TIME online context allowance (Amendment 13, recorded before any
#: run).  An ONLINE episode is not a rendering of the pre-generated transcript:
#: every one of the ten MODEL_ATTEMPT slots is replaced by up to
#: ``max_new_tokens`` REAL generated tokens, so the last prompt of the longest
#: episodes projects past 2048 even though the scripted rendering fits.  2048
#: was a DATA-GENERATION constraint, not an evaluation constraint: the frozen
#: backbone's ``max_position_embeddings`` is 65536 (contract section 1).  4096
#: covers the worst case present in the pre-generated data: measured with the
#: real Ouro tokenizer, the longest scripted rendering is 1740 tokens and the
#: longest projected online context 1740 + 11 x 64 = 2444.  Truncation
#: remains forbidden; an episode that would still exceed this allowance is
#: recorded, excluded and counted, never silently shortened.
EVAL_ONLINE_MAX_SEQ_LEN = 4096

#: episode-record ``status`` values
STATUS_COMPLETE = "COMPLETE"
STATUS_ONLINE_BUDGET_EXCEEDED = "ONLINE_BUDGET_EXCEEDED"

FEEDBACK_CONDITIONS = ("structured", "correctness_only")
CONTEXT_MODES = ("history", "context_reset")


class RenderIntegrationError(RuntimeError):
    """The renderer does not support the online walker's required operation."""


class EpisodeEnvironmentError(RuntimeError):
    """The episode environment cannot serve an item the plan references."""


class SequenceBudgetError(RuntimeError):
    """An online prompt exceeds the frozen eval-time allowance (never truncated).

    Since Amendment 13 the ONLINE driver does not let this abort a whole batch:
    :meth:`EpisodeWalker.abort_online_budget` isolates the offending episode and
    stores this exception's message in that episode's record.  The class is kept
    because it is the named, catchable type of that condition and because
    callers that drive a single episode may still want to raise it.
    """


class FeedbackLeakError(RuntimeError):
    """Feedback text revealed the canonical answer of the pending item."""


class CertifiedChannelViolation(RuntimeError):
    """A certified event carried a poison flag (contract section 6 forbids it)."""


# --------------------------------------------------------------------------
# episode / event access helpers (tolerant of the producer's exact classes)
# --------------------------------------------------------------------------
def role_name(event: Any) -> str:
    role = getattr(event, "role", None)
    if role is None and isinstance(event, dict):
        role = event.get("role")
    value = getattr(role, "value", role)
    return str(value).rsplit(".", 1)[-1].upper()


def event_attr(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def replace_fields(obj: Any, **changes: Any) -> Any:
    """``dataclasses.replace`` when possible, else a shallow copy with setattr."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **changes)
    if isinstance(obj, dict):
        out = dict(obj)
        out.update(changes)
        return out
    clone = copy.copy(obj)
    for k, v in changes.items():
        setattr(clone, k, v)
    return clone


def episode_events(ep: Any) -> list[Any]:
    events = event_attr(ep, "events")
    if events is None:
        raise EpisodeEnvironmentError("episode has no .events (contract section 23)")
    return list(events)


def _jsonable(value: Any) -> Any:
    """Coerce producer-side enums / ids into canonical-JSON-safe scalars."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    inner = getattr(value, "value", None)
    if isinstance(inner, (str, int, float, bool)):
        return inner
    return str(value)


def condition_get(ep: Any, key: str) -> Any:
    """Best-effort read of a sub-field of ``Episode.condition``."""
    cond = event_attr(ep, "condition")
    if isinstance(cond, dict):
        return cond.get(key)
    meta = event_attr(ep, "meta")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return None


def condition_label(ep: Any) -> str:
    cond = event_attr(ep, "condition")
    if isinstance(cond, dict):
        from . import canonical_json
        return canonical_json(cond)
    return "" if cond is None else str(cond)


# --------------------------------------------------------------------------
# environment protocol
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VerifierOutcome:
    correct: bool
    canonical_answer: str
    parsed: str | None

    @classmethod
    def coerce(cls, value: Any) -> "VerifierOutcome":
        if isinstance(value, VerifierOutcome):
            return value
        if isinstance(value, dict):
            return cls(bool(value.get("correct")),
                       str(value.get("canonical_answer", "")),
                       value.get("parsed"))
        return cls(bool(getattr(value, "correct")),
                   str(getattr(value, "canonical_answer", "")),
                   getattr(value, "parsed", None))


class EpisodeEnvironment(Protocol):
    """Exact-verifier environment for ONE episode (one latent rule)."""

    def verify(self, item_id: str, answer_text: str) -> Any:
        """Exact family verification of the model's attempt text."""

    def feedback_text(
        self,
        item_id: str,
        answer_text: str,
        outcome: VerifierOutcome,
        *,
        role: str,
        certified: bool,
        poison: bool,
        condition: str,
        attempt_number: int,
        event: Any = None,
    ) -> str:
        """Real feedback content for a FEEDBACK / HINT / REVEAL slot."""


#: certified verdict strings (identical to ``episodes.assemble``'s constants,
#: which are imported when available so the online text matches the shards)
CERTIFIED_CORRECT = "CORRECT"
CERTIFIED_INCORRECT = "INCORRECT"


def _certified_verdict(correct: bool) -> str:
    try:
        from foundation_learner.episodes import assemble as _assemble  # type: ignore
        return (_assemble.CERTIFIED_CORRECT if correct
                else _assemble.CERTIFIED_INCORRECT)
    except Exception:
        return CERTIFIED_CORRECT if correct else CERTIFIED_INCORRECT


class FamilyEnvironment:
    """:class:`EpisodeEnvironment` built from a real ``TaskFamily`` (section 4).

    ``instances`` maps ``item_id -> TaskInstance``.  Feedback randomness uses a
    seeded ``numpy`` generator derived through ``ecology.base.derive_seed``;
    no global RNG and no wall clock.

    HINT content honours the frozen poison conditions: when a per-slot
    condition is known (``hint_conditions``, keyed by event slot or item id, or
    the episode-level ``poison_condition``) the hint body is produced by
    ``ecology.poison.hint_for_condition`` from the family's own feedback
    record, exactly as the pre-generated shards do — but recomputed from the
    model's REAL attempt.  Without a declared condition the event's poison flag
    selects ``corrupted_feedback`` vs ``structured_feedback``.
    """

    def __init__(self, family: Any, rule: Any, instances: dict[str, Any],
                 *, root_seed: int = 0, episode_id: str = "",
                 hint_conditions: dict[str, str] | None = None,
                 poison_condition: str | None = None):
        self.family = family
        self.rule = rule
        self.instances = dict(instances)
        self.root_seed = int(root_seed)
        self.episode_id = str(episode_id)
        self.hint_conditions = dict(hint_conditions or {})
        self.poison_condition = poison_condition

    def _instance(self, item_id: str) -> Any:
        try:
            return self.instances[item_id]
        except KeyError as exc:
            raise EpisodeEnvironmentError(
                f"no TaskInstance registered for item_id {item_id!r}; "
                f"known items: {sorted(self.instances)}") from exc

    def _rng(self, item_id: str, tag: str, attempt_number: int):
        import numpy as np
        from . import derive_seed
        seed = derive_seed(self.root_seed, "FEEDBACK", self.episode_id, item_id,
                           tag, int(attempt_number))
        return np.random.Generator(np.random.PCG64(int(seed)))

    def canonical_answer(self, item_id: str) -> str:
        inst = self._instance(item_id)
        value = event_attr(inst, "answer_canonical", None)
        return "" if value is None else str(value)

    def verify(self, item_id: str, answer_text: str) -> Any:
        return self.family.verify(self.rule, self._instance(item_id), answer_text)

    def hint_condition(self, item_id: str, event: Any) -> str | None:
        slot = str(event_attr(event, "slot", "") or "") if event is not None else ""
        for key in (slot, item_id):
            if key and key in self.hint_conditions:
                return self.hint_conditions[key]
        return self.poison_condition

    def feedback_text(self, item_id, answer_text, outcome, *, role, certified,
                      poison, condition, attempt_number, event=None) -> str:
        role = role.upper()
        if role == "FEEDBACK":
            return _certified_verdict(bool(outcome.correct))
        if role == "REVEAL":
            return self.canonical_answer(item_id)
        if role != "HINT":
            raise EpisodeEnvironmentError(f"unsupported feedback role {role!r}")
        inst = self._instance(item_id)
        declared = self.hint_condition(item_id, event)
        if declared is not None:
            try:
                from foundation_learner.ecology.poison import hint_for_condition
            except Exception as exc:  # pragma: no cover - integration guard
                raise EpisodeEnvironmentError(
                    "a per-slot poison condition was declared but "
                    f"ecology.poison.hint_for_condition is unavailable: {exc!r}") from exc
            rng = self._rng(item_id, f"hint::{declared}", attempt_number)
            record = self.family.feedback_record(self.rule, inst, answer_text, rng)
            return str(hint_for_condition(
                record, declared, rng,
                answer_canonical=self.canonical_answer(item_id)))
        if poison:
            return str(self.family.corrupted_feedback(
                self.rule, inst, answer_text,
                self._rng(item_id, "corrupted", attempt_number)))
        return str(self.family.structured_feedback(
            self.rule, inst, answer_text,
            self._rng(item_id, "structured", attempt_number)))


def instances_from_episode(ep: Any) -> dict[str, Any]:
    """``item_id -> TaskInstance`` from the episode's own item table."""
    items = event_attr(ep, "items")
    if not items:
        raise EpisodeEnvironmentError(
            "episode carries no .items table; the online evaluator needs the "
            "TaskInstance of every item it must verify (contract section 4)")
    from foundation_learner.ecology.base import TaskInstance  # type: ignore

    out: dict[str, Any] = {}
    for item_id, record in items.items():
        payload = record.get("instance") if isinstance(record, dict) else record
        out[str(item_id)] = (TaskInstance.from_dict(payload)
                             if isinstance(payload, dict) else payload)
    return out


def rule_from_episode(ep: Any) -> Any:
    """Rebuild the latent ``Rule`` from the episode's serialized rule spec."""
    spec = event_attr(ep, "rule_spec")
    if not isinstance(spec, dict):
        raise EpisodeEnvironmentError("episode carries no rule_spec dict")
    from foundation_learner.ecology.base import Rule  # type: ignore

    return Rule(family_id=str(spec.get("family_id", event_attr(ep, "family_id", ""))),
                params=dict(spec.get("params", {})))


def hint_conditions_from_episode(ep: Any) -> dict[str, str]:
    """``slot -> poison condition`` from the episode's hint schedule."""
    meta = event_attr(ep, "meta") or {}
    slots = meta.get("hint_slots") if isinstance(meta, dict) else None
    if not slots:
        return {}
    out: dict[str, str] = {}
    for entry in slots:
        condition = entry.get("condition")
        if condition is None:
            continue
        if entry.get("slot"):
            out[str(entry["slot"])] = str(condition)
    return out


def environment_from_episode(ep: Any, family: Any, *, rule: Any = None,
                             poison_condition: str | None = None
                             ) -> "FamilyEnvironment":
    """Build the exact-verifier environment for one pre-generated episode."""
    meta = event_attr(ep, "meta") or {}
    root_seed = int(meta.get("root_seed", 0)) if isinstance(meta, dict) else 0
    return FamilyEnvironment(
        family, rule if rule is not None else rule_from_episode(ep),
        instances_from_episode(ep),
        root_seed=root_seed,
        episode_id=str(event_attr(ep, "episode_id", "")),
        hint_conditions=hint_conditions_from_episode(ep),
        poison_condition=poison_condition)


# --------------------------------------------------------------------------
# callback protocol
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeContext:
    episode_id: str
    family_id: str
    split: str | None
    condition: str
    context_mode: str
    feedback_condition: str
    structure: str


@dataclass(frozen=True)
class AttemptEvent:
    """What a mechanism may know about an attempt (never the hidden verdict)."""

    role: str
    item_id: str
    interaction_index: int | None
    text: str
    parsed: str | None


@dataclass(frozen=True)
class FeedbackEvent:
    """What a mechanism may know about a feedback item (never the poison flag)."""

    role: str
    item_id: str
    interaction_index: int | None
    text: str
    certified: bool
    event_index: int
    char_span: tuple[int, int]
    context_sha256: str


HiddenSummaryFn = Callable[..., torch.Tensor]


class AdaptationCallbacks(Protocol):
    def on_episode_start(self, ctx: EpisodeContext) -> None: ...
    def prefix_provider(self) -> torch.Tensor | None: ...
    def on_attempt(self, attempt: AttemptEvent) -> None: ...
    def on_feedback(self, event: FeedbackEvent,
                    hidden_summary_fn: HiddenSummaryFn) -> None: ...
    def on_episode_end(self, ctx: EpisodeContext) -> None: ...


class NullCallbacks:
    """No-op implementation; the FL0/FL1/FL2/FL3 arms use it unchanged."""

    def on_episode_start(self, ctx: EpisodeContext) -> None:
        return None

    def prefix_provider(self) -> torch.Tensor | None:
        return None

    def on_attempt(self, attempt: AttemptEvent) -> None:
        return None

    def on_feedback(self, event: FeedbackEvent,
                    hidden_summary_fn: HiddenSummaryFn) -> None:
        return None

    def on_episode_end(self, ctx: EpisodeContext) -> None:
        return None


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LearningCurveConfig:
    feedback_condition: str = "structured"
    context_mode: str = "history"
    #: attempts at interaction indices >= this are asked under a reset context
    reset_from_index: int = 4
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt_mode: str = "sentinel"          # "sentinel" | "upto_event_with_cue"
    attempt_cue: str = DEFAULT_ATTEMPT_CUE
    #: EVAL-TIME allowance, not the section 7 rendering budget (see
    #: ``EVAL_ONLINE_MAX_SEQ_LEN``); training-side rendering still uses 2048.
    max_seq_len: int = EVAL_ONLINE_MAX_SEQ_LEN
    record_generations: bool = True
    record_prompts: bool = False           # full prompt text (large); hashes always
    arm_tag: str = "unspecified"
    strict_feedback_leak: bool = True

    def __post_init__(self) -> None:
        if self.feedback_condition not in FEEDBACK_CONDITIONS:
            raise ValueError(f"feedback_condition must be one of {FEEDBACK_CONDITIONS}")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}")
        if self.prompt_mode not in ("sentinel", "upto_event_with_cue"):
            raise ValueError("prompt_mode must be 'sentinel' or 'upto_event_with_cue'")


# --------------------------------------------------------------------------
# renderer access
# --------------------------------------------------------------------------
def resolve_renderer(render_module: Any = None) -> Any:
    if render_module is not None:
        return render_module
    from foundation_learner.episodes import render as _render  # type: ignore
    return _render


def rendered_text(rendered: Any) -> str:
    text = event_attr(rendered, "text")
    if text is None:
        raise RenderIntegrationError("Rendered object has no .text")
    return str(text)


def rendered_spans(rendered: Any) -> list[tuple[int, int]]:
    """Per-event ``(char_start, char_end)`` from the producer's span map.

    Accepts the three shapes the renderer may legitimately use: objects with
    ``.start`` / ``.end`` (``episodes.render.EventSpan``), dicts with
    ``char_start`` / ``char_end``, and bare pairs.
    """
    spans = event_attr(rendered, "spans")
    if spans is None:
        raise RenderIntegrationError("Rendered object has no .spans")
    out: list[tuple[int, int]] = []
    for s in spans:
        start = getattr(s, "start", None)
        end = getattr(s, "end", None)
        if start is not None and end is not None:
            out.append((int(start), int(end)))
        elif isinstance(s, dict):
            out.append((int(s.get("char_start", s.get("start"))),
                        int(s.get("char_end", s.get("end")))))
        else:
            out.append((int(s[0]), int(s[1])))
    return out


def resolve_reset_renderer(render_module: Any) -> tuple[Callable[[Any, int], Any], str]:
    """Resolve "instruction header + this query only" from the real renderer.

    Contract section 23 names ``render_reset_query(ep, query_event_index)``.
    The renderer that actually landed exposes the same capability as
    ``render_subset(ep, keep_indices)`` (its docstring names the context-reset
    evaluation as a motivating caller).  Consumers adapt to the producer, so
    both spellings are accepted, in that order, and the resolved spelling is
    recorded in the episode record.
    """
    fn = getattr(render_module, "render_reset_query", None)
    if callable(fn):
        return (lambda ep, idx: fn(ep, idx)), "render_reset_query"
    subset = getattr(render_module, "render_subset", None)
    if callable(subset):
        return (lambda ep, idx: subset(ep, [idx])), "render_subset"
    raise RenderIntegrationError(
        "context-reset evaluation requires episodes.render.render_reset_query "
        "or episodes.render.render_subset")


# --------------------------------------------------------------------------
# the walker
# --------------------------------------------------------------------------
def _final_loop_hidden_states(model: Any, model_inputs: dict[str, Any],
                              attention_mask: torch.Tensor) -> torch.Tensor:
    """Final-layer, FINAL-LOOP hidden states ``[1, T, hidden]``.

    For ``OuroForCausalLM`` the inner ``OuroModel`` returns
    ``(BaseModelOutputWithPast, hidden_states_list, gate_list)`` whose
    ``last_hidden_state`` IS the final recurrent loop's normalized output; that
    is the state the contract binds for the FL4 head input and the FL5 update.
    The causal-LM ``output_hidden_states=True`` path (whose ``hidden_states``
    tuple is ``(inputs_embeds, *per_loop_states)``) is the documented fallback
    for any wrapper that hides the inner module.
    """
    inner = getattr(model, "model", None)
    if inner is not None and callable(inner):
        with torch.inference_mode():
            result = inner(attention_mask=attention_mask, use_cache=False, **model_inputs)
        base = result[0] if isinstance(result, tuple) else result
        hidden = getattr(base, "last_hidden_state", None)
        if hidden is not None:
            return hidden
    with torch.inference_mode():
        out = model(attention_mask=attention_mask, use_cache=False,
                    output_hidden_states=True, **model_inputs)
    hs = getattr(out, "hidden_states", None)
    if hs:
        return hs[-1]
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        raise RenderIntegrationError(
            "model exposes no final-loop hidden states (needed by hidden_summary_fn)")
    return hidden


@dataclass
class PromptRequest:
    walker: "EpisodeWalker"
    prompt: str
    prefix: torch.Tensor | None


class EpisodeWalker:
    """Drives ONE episode; the driver batches generations across walkers."""

    def __init__(self, bundle: Any, ep: Any, env: EpisodeEnvironment,
                 cfg: LearningCurveConfig, callbacks: Any,
                 render_module: Any, parser: Callable[[str], str | None],
                 parser_source: str):
        self.bundle = bundle
        self.ep = ep
        self.env = env
        self.cfg = cfg
        self.cb = callbacks if callbacks is not None else NullCallbacks()
        self.render = render_module
        self.parser = parser
        self.parser_source = parser_source

        self.plan = episode_events(ep)
        self.live: list[Any] = []
        self.live_source_index: list[int] = []
        self.pos = 0
        self.finished = False
        self.status = STATUS_COMPLETE
        self.budget_event: dict[str, Any] | None = None
        self.attempt_cue: str | None = None
        self.attempts: list[dict[str, Any]] = []
        self.outcomes: dict[str, VerifierOutcome] = {}
        self.last_answer: dict[str, str] = {}
        self.attempt_counts: dict[str, int] = {}
        self.last_task_item: str | None = None
        self.advisory_flags: list[dict[str, Any]] = []
        self.prefix_used = False
        self.n_feedback_events = 0
        self.batch_sizes: list[int] = []
        self.reset_prompts_used = 0
        self._pending: dict[str, Any] | None = None
        self._reset_renderer: Callable[[Any, int], Any] | None = None
        self.reset_render_source: str | None = None

        self.ctx = EpisodeContext(
            episode_id=str(event_attr(ep, "episode_id", "")),
            family_id=str(event_attr(ep, "family_id", "")),
            split=event_attr(ep, "split"),
            condition=condition_label(ep),
            context_mode=cfg.context_mode,
            feedback_condition=cfg.feedback_condition,
            structure=str(event_attr(ep, "structure", STRUCTURE_V0)),
        )
        self.cb.on_episode_start(self.ctx)

    # -- rendering -------------------------------------------------------
    def _live_episode(self, events: list[Any]) -> Any:
        return replace_fields(self.ep, events=list(events))

    def _render_live(self, events: list[Any]) -> Any:
        return self.render.render_episode(self._live_episode(events))

    def _history_text(self) -> str:
        if not self.live:
            return ""
        return rendered_text(self._render_live(self.live))

    def _attempt_prompt(self, attempt_event: Any) -> str:
        """Prompt that ends exactly where the model must start writing."""
        if self.cfg.prompt_mode == "upto_event_with_cue":
            history = self._history_text()
            cue = self.cfg.attempt_cue
            if self.attempt_cue is None:
                self.attempt_cue = cue
            return history + cue
        slot = replace_fields(attempt_event, text=ATTEMPT_SENTINEL)
        rendered = self._render_live(self.live + [slot])
        text = rendered_text(rendered)
        hits = text.count(ATTEMPT_SENTINEL)
        if hits != 1:
            raise RenderIntegrationError(
                f"attempt-slot sentinel appeared {hits} times in the rendered "
                "episode; the renderer must reproduce MODEL_ATTEMPT text verbatim")
        prompt = text[: text.index(ATTEMPT_SENTINEL)]
        history = self._history_text()
        if prompt.startswith(history):
            cue = prompt[len(history):]
            if self.attempt_cue is None:
                self.attempt_cue = cue
        elif self.attempt_cue is None:
            self.attempt_cue = self.cfg.attempt_cue
        return prompt

    def _reset_prompt(self, source_index: int) -> str:
        if self._reset_renderer is None:
            self._reset_renderer, self.reset_render_source = resolve_reset_renderer(
                self.render)
        base = rendered_text(self._reset_renderer(self.ep, source_index))
        cue = self.attempt_cue if self.attempt_cue is not None else self.cfg.attempt_cue
        return base + cue

    # -- plan walking ----------------------------------------------------
    def next_request(self) -> PromptRequest | None:
        """Advance the plan until the next attempt slot; None when finished."""
        if self.finished or self._pending is not None:
            return None
        while self.pos < len(self.plan):
            ev = self.plan[self.pos]
            role = role_name(ev)
            if role == "MODEL_ATTEMPT":
                return self._open_attempt(ev)
            self.pos += 1
            if role in FEEDBACK_ROLES:
                self._append_feedback(ev)
            else:
                if role in TASK_ROLES:
                    self.last_task_item = str(event_attr(ev, "item_id", "") or "")
                self.live.append(ev)
                self.live_source_index.append(self.pos - 1)
        self.finished = True
        self.cb.on_episode_end(self.ctx)
        return None

    def _resolve_item(self, ev: Any) -> str:
        item_id = event_attr(ev, "item_id", None)
        item_id = "" if item_id is None else str(item_id)
        if item_id:
            return item_id
        if self.last_task_item:
            return self.last_task_item
        raise EpisodeEnvironmentError(
            f"event at plan index {self.pos} has no item_id and no preceding task")

    def _task_source_index(self, item_id: str) -> int:
        for i in range(len(self.live) - 1, -1, -1):
            ev = self.live[i]
            if role_name(ev) in TASK_ROLES and str(event_attr(ev, "item_id", "")) == item_id:
                return self.live_source_index[i]
        raise EpisodeEnvironmentError(f"no rendered task event found for item {item_id!r}")

    def _open_attempt(self, ev: Any) -> PromptRequest:
        item_id = self._resolve_item(ev)
        idx = event_attr(ev, "interaction_index", None)
        use_reset = (self.cfg.context_mode == "context_reset"
                     and idx is not None and int(idx) >= int(self.cfg.reset_from_index))
        if use_reset:
            if self.attempt_cue is None:
                # derive the renderer's attempt cue from a throw-away probe so
                # the reset prompt uses the identical protocol marker
                self._attempt_prompt(ev)
            prompt = self._reset_prompt(self._task_source_index(item_id))
            self.reset_prompts_used += 1
        else:
            prompt = self._attempt_prompt(ev)
        prefix = self.cb.prefix_provider()
        if prefix is not None:
            self.prefix_used = True
        self._pending = {
            "event": ev,
            "item_id": item_id,
            "interaction_index": None if idx is None else int(idx),
            "prompt": prompt,
            "reset": use_reset,
        }
        return PromptRequest(walker=self, prompt=prompt, prefix=prefix)

    # -- per-episode budget isolation ------------------------------------
    def abort_online_budget(self, *, prompt_tokens: int, max_new_tokens: int,
                            limit: int) -> dict[str, Any]:
        """Stop THIS episode because its online context exceeds the allowance.

        Contract section 7 forbids generation-side truncation, so an episode
        whose online context would not fit cannot be shortened.  Before
        Amendment 13 the whole ``run_episodes`` batch raised
        :class:`SequenceBudgetError` on the first such episode, i.e. one long
        episode destroyed an entire evaluation.  The episode is now recorded
        with ``status = ONLINE_BUDGET_EXCEEDED``, its partial ``R_k`` is marked
        MISSING (so no metric can consume a truncated curve), it is counted in
        the record set, and every other episode proceeds.  Nothing is dropped
        silently: the record is returned like any other and
        ``metrics.excluded_records`` reports the count.
        """
        pending = self._pending or {}
        self._pending = None
        self.status = STATUS_ONLINE_BUDGET_EXCEEDED
        self.budget_event = {
            "reason": STATUS_ONLINE_BUDGET_EXCEEDED,
            "interaction_index": pending.get("interaction_index"),
            "item_id": pending.get("item_id"),
            "attempts_completed": len(self.attempts),
            "prompt_tokens": int(prompt_tokens),
            "max_new_tokens": int(max_new_tokens),
            "projected_tokens": int(prompt_tokens) + int(max_new_tokens),
            "allowance": int(limit),
            "context_mode": ("context_reset" if pending.get("reset")
                             else "history"),
            "truncated": False,
            "detail": str(SequenceBudgetError(
                f"prompt {int(prompt_tokens)} tokens + {int(max_new_tokens)} new "
                f"exceeds the frozen eval-time allowance {int(limit)} "
                f"(episode {self.ctx.episode_id}); generation-side truncation is "
                "forbidden, so this episode is recorded and excluded")),
        }
        self.finished = True
        self.cb.on_episode_end(self.ctx)
        return self.budget_event

    # -- generation feedback --------------------------------------------
    def submit(self, gen: GenerationRecord) -> None:
        assert self._pending is not None, "submit() without a pending request"
        pending = self._pending
        self._pending = None
        ev = pending["event"]
        item_id = pending["item_id"]

        raw = gen.text
        end = complete_answer_line_end(raw)
        attempt_text = (raw[:end] if end is not None else raw).rstrip()
        outcome = VerifierOutcome.coerce(self.env.verify(item_id, attempt_text))
        parsed = self.parser(attempt_text)

        live_ev = replace_fields(ev, text=attempt_text, item_id=item_id)
        self.live.append(live_ev)
        self.live_source_index.append(self.pos)
        self.pos += 1
        self.outcomes[item_id] = outcome
        self.last_answer[item_id] = attempt_text
        self.attempt_counts[item_id] = self.attempt_counts.get(item_id, 0) + 1
        self.batch_sizes.append(gen.batch_size)

        record: dict[str, Any] = {
            "interaction_index": pending["interaction_index"],
            "item_id": item_id,
            "role": role_name(ev),
            "attempt_number": self.attempt_counts[item_id] - 1,
            "context_mode": "context_reset" if pending["reset"] else "history",
            "prompt_sha256": text_sha256(pending["prompt"]),
            "prompt_chars": len(pending["prompt"]),
            "prompt_tokens": gen.prompt_token_count,
            "prefix_len": gen.prefix_len,
            "finish_reason": gen.finish_reason,
            "parsed": parsed,
            "correct": bool(outcome.correct),
            "canonical_answer": outcome.canonical_answer,
            "batch_size": gen.batch_size,
            "logits_finite": gen.logits_finite,
        }
        if self.cfg.record_generations:
            record["generated_text"] = raw
            record["attempt_text"] = attempt_text
            record["generated_token_ids"] = list(gen.token_ids)
            record["generated_text_sha256"] = text_sha256(raw)
        if self.cfg.record_prompts:
            record["prompt_text"] = pending["prompt"]
        self.attempts.append(record)

        self.cb.on_attempt(AttemptEvent(
            role=role_name(ev), item_id=item_id,
            interaction_index=pending["interaction_index"],
            text=attempt_text, parsed=parsed))

    # -- feedback --------------------------------------------------------
    def _append_feedback(self, ev: Any) -> None:
        role = role_name(ev)
        if role == "HINT" and self.cfg.feedback_condition == "correctness_only":
            return  # hints exist only in structured conditions (section 6)
        item_id = self._resolve_item(ev)
        outcome = self.outcomes.get(item_id)
        if outcome is None:
            raise EpisodeEnvironmentError(
                f"{role} for item {item_id!r} precedes any model attempt on it")
        answer_text = self.last_answer.get(item_id, "")
        poison = bool(event_attr(ev, "poison", False))
        certified = bool(event_attr(ev, "certified", role != "HINT"))
        if certified and poison:
            raise CertifiedChannelViolation(
                f"{role} event for item {item_id!r} is certified AND flagged poison; "
                "the certified channel is never corrupted (contract section 6)")
        text = str(self.env.feedback_text(
            item_id, answer_text, outcome,
            role=role, certified=certified, poison=poison,
            condition=self.cfg.feedback_condition,
            attempt_number=max(0, self.attempt_counts.get(item_id, 1) - 1),
            event=ev))
        self._check_feedback_leak(role, item_id, text, outcome)
        live_ev = replace_fields(ev, text=text, item_id=item_id)
        self.live.append(live_ev)
        self.live_source_index.append(self.pos - 1)
        self.n_feedback_events += 1

        rendered = self._render_live(self.live)
        full_text = rendered_text(rendered)
        try:
            span = rendered_spans(rendered)[len(self.live) - 1]
        except Exception as exc:  # renderer must expose the span map
            raise RenderIntegrationError(
                f"renderer did not expose a span for event {len(self.live) - 1}: {exc!r}"
            ) from exc
        fb_event = FeedbackEvent(
            role=role, item_id=item_id,
            interaction_index=(None if event_attr(ev, "interaction_index") is None
                               else int(event_attr(ev, "interaction_index"))),
            text=text, certified=certified,
            event_index=len(self.live) - 1,
            char_span=(int(span[0]), int(span[1])),
            context_sha256=text_sha256(full_text),
        )
        self.cb.on_feedback(fb_event, self._make_hidden_summary_fn(full_text, span))

    def _check_feedback_leak(self, role: str, item_id: str, text: str,
                             outcome: VerifierOutcome) -> None:
        """Frozen guard: no feedback channel may state the pending answer.

        Exact and non-fragile: the failure conditions are (a) the feedback text
        IS the canonical answer, or (b) it contains a literal ``ANSWER: <canon>``
        commitment line.  REVEAL is exempt by definition (contract section 6:
        it exposes the answer of a SUPPORT item only).  A weaker "contains the
        canonical string anywhere" test would fire on legitimate numeric hints
        (e.g. a residue-distance band), so it is RECORDED as an advisory flag
        instead of raising.
        """
        canon = (outcome.canonical_answer or "").strip()
        if not canon or role == "REVEAL":
            return
        stripped = text.strip()
        hard = stripped == canon or f"ANSWER: {canon}" in text
        if hard and self.cfg.strict_feedback_leak:
            raise FeedbackLeakError(
                f"{role} for item {item_id!r} states the canonical answer")
        if hard or re.search(r"(?<![A-Za-z0-9_])" + re.escape(canon) + r"(?![A-Za-z0-9_])", text):
            self.advisory_flags.append({
                "kind": "feedback_contains_canonical_token",
                "role": role, "item_id": item_id, "hard": bool(hard)})

    def _make_hidden_summary_fn(self, context_text: str,
                                span: tuple[int, int]) -> HiddenSummaryFn:
        cache: dict[tuple[str, bool], torch.Tensor] = {}

        def hidden_summary_fn(reduce: str = "mean", include_prefix: bool = False):
            key = (reduce, bool(include_prefix))
            if key in cache:
                return cache[key]
            tokens_key = ("tokens", bool(include_prefix))
            if tokens_key not in cache:
                cache[tokens_key] = self._item_hidden_states(
                    context_text, span, include_prefix=bool(include_prefix))
            item_h = cache[tokens_key]
            if reduce == "tokens":
                value = item_h
            elif reduce == "mean":
                value = item_h.mean(dim=0)
            elif reduce == "last":
                value = item_h[-1]
            else:
                raise ValueError(f"unknown reduce {reduce!r}")
            cache[key] = value
            return value

        return hidden_summary_fn

    def _item_hidden_states(self, context_text: str, span: tuple[int, int],
                            *, include_prefix: bool) -> torch.Tensor:
        model = getattr(self.bundle, "model")
        tokenizer = getattr(self.bundle, "tokenizer")
        device = getattr(self.bundle, "device", None)
        if device is None:
            device = next(model.parameters()).device
        ids, offsets = encode_with_offsets(tokenizer, context_text)
        t0, t1 = align_char_span_covering(offsets, span)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        embed = model.get_input_embeddings()
        prefix = self.cb.prefix_provider() if include_prefix else None
        if prefix is None:
            model_inputs: dict[str, Any] = {"input_ids": input_ids}
            mask = torch.ones_like(input_ids)
        else:
            pfx = prefix.to(device=device, dtype=embed.weight.dtype)
            embeds = torch.cat([pfx.unsqueeze(0), embed(input_ids)], dim=1)
            model_inputs = {"inputs_embeds": embeds}
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
        hidden = _final_loop_hidden_states(model, model_inputs, mask)
        shift = 0 if prefix is None else int(prefix.shape[0])
        return hidden[0, shift + t0: shift + t1].detach().float()

    # -- record ----------------------------------------------------------
    def episode_record(self) -> dict[str, Any]:
        by_index: dict[int, list[bool]] = {}
        for a in self.attempts:
            idx = a["interaction_index"]
            if idx is None:
                continue
            by_index.setdefault(int(idx), []).append(bool(a["correct"]))
        R = {str(k): float(sum(v) / len(v)) for k, v in sorted(by_index.items())}
        aulc = float(sum(R.values()) / len(R)) if R else None
        partial_indices: list[int] = []
        if self.status != STATUS_COMPLETE:
            # The curve of an aborted episode is INCOMPLETE, so it is marked
            # missing rather than reported as a shorter curve: a partial R
            # would silently change what macro-AULC averages over.
            partial_indices = sorted(int(k) for k in R)
            R = {}
            aulc = None
        record = {
            "schema": EPISODE_RECORD_SCHEMA,
            "episode_id": self.ctx.episode_id,
            "family_id": self.ctx.family_id,
            "split": _jsonable(self.ctx.split),
            "rule_id": _jsonable(event_attr(self.ep, "rule_id")),
            "seed": _jsonable(event_attr(self.ep, "seed")),
            "condition": self.ctx.condition,
            "structure": self.ctx.structure,
            "arm_tag": self.cfg.arm_tag,
            "feedback_condition": self.cfg.feedback_condition,
            "context_mode": self.cfg.context_mode,
            "reset_from_index": int(self.cfg.reset_from_index),
            "poison_condition": _jsonable(condition_get(self.ep, "poison_condition")),
            "remap_id": _jsonable(condition_get(self.ep, "remap_id")),
            "generalization_scope": None,
            "status": self.status,
            "R": R,
            "interaction_indices": sorted(int(k) for k in R),
            "aulc": aulc,
            "R_missing_reason": (None if self.status == STATUS_COMPLETE
                                 else self.status),
            "partial_interaction_indices": partial_indices,
            "online_budget_event": self.budget_event,
            "eval_online_allowance": int(self.cfg.max_seq_len),
            "n_attempts": len(self.attempts),
            "n_feedback_events": self.n_feedback_events,
            "reset_prompts_used": self.reset_prompts_used,
            "prefix_used": bool(self.prefix_used),
            "prompt_mode": self.cfg.prompt_mode,
            "reset_render_source": self.reset_render_source,
            "parser_source": self.parser_source,
            "max_new_tokens": int(self.cfg.generation.max_new_tokens),
            "batch_sizes": list(self.batch_sizes),
            "callbacks_class": type(self.cb).__name__,
            "attempts": self.attempts,
            "advisory_flags": self.advisory_flags,
        }
        record["record_hash"] = domain_sha256(EPISODE_RECORD_DOMAIN, record)
        return record


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def _prompt_token_count(bundle: Any, prompt: str) -> int:
    tokenizer = getattr(bundle, "tokenizer")
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def run_episodes(
    bundle: Any,
    episodes: Sequence[Any],
    env_factory: Callable[[Any], EpisodeEnvironment],
    *,
    cfg: LearningCurveConfig | None = None,
    callbacks_factory: Callable[[Any], Any] | None = None,
    render_module: Any = None,
    parser: Callable[[str], str | None] | None = None,
    walker_class: type = None,  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Run ONLINE learning-curve episodes; returns ``episode_record.v1`` dicts.

    This is the public ONLINE evaluation boundary (Amendment 16): direct
    walker callers need the same eval-mode guarantee as campaign/mechanism
    callers before any callback or decode work begins.

    Episodes advance in lockstep so that the attempt slots pending at any
    moment are decoded in ONE batched greedy call (subject to
    ``cfg.generation``).  Batch composition never changes a row's numerics
    (left padding + per-row position ids + non-compacting slots), and the
    equivalence gate in ``generation.py`` is what licenses batching at all.
    """
    try:
        model = getattr(bundle, "model")
    except AttributeError as exc:
        raise AttributeError(
            "run_episodes requires an evaluation bundle with a .model attribute"
        ) from exc
    if model is None:
        raise AttributeError(
            "run_episodes requires an evaluation bundle with a non-None .model"
        )
    # Keep the import lazy: evaluation helpers are also imported by training and
    # mechanism modules, while this boundary is entered only for ONLINE runs.
    from foundation_learner.training.model_loading import set_evaluation_mode

    set_evaluation_mode(model)
    cfg = cfg or LearningCurveConfig()
    render_module = resolve_renderer(render_module)
    if parser is None:
        parser, parser_source = resolve_answer_parser()
    else:
        parser_source = getattr(parser, "__qualname__", "caller-supplied")

    callback_objs = [callbacks_factory(ep) if callbacks_factory else NullCallbacks()
                     for ep in episodes]
    # NOTE: mechanisms SUBCLASS NullCallbacks, so the exemption must be by
    # exact type - a subclass carries state and may not be shared.
    stateful = [cb for cb in callback_objs
                if cb is not None and type(cb) is not NullCallbacks]
    if len(stateful) > 1 and len({id(cb) for cb in stateful}) != len(stateful):
        raise ValueError(
            "callbacks_factory returned the SAME callbacks object for more than one "
            "episode; episodes advance in lockstep, so shared adaptation state would "
            "leak across episodes (contract section 7: fast state resets per episode)")
    wcls = walker_class or EpisodeWalker
    walkers = [
        wcls(bundle, ep, env_factory(ep), cfg, cb,
             render_module, parser, parser_source)
        for ep, cb in zip(episodes, callback_objs)
    ]
    active = list(walkers)
    while active:
        requests: list[PromptRequest] = []
        still: list[EpisodeWalker] = []
        for w in active:
            req = w.next_request()
            if req is not None:
                requests.append(req)
                still.append(w)
        if not requests:
            break
        admitted: list[PromptRequest] = []
        for req in requests:
            n_tok = _prompt_token_count(bundle, req.prompt)
            projected = n_tok + int(cfg.generation.max_new_tokens)
            if projected > int(cfg.max_seq_len):
                # PER-EPISODE ISOLATION (Amendment 13): the offending episode is
                # recorded with status ONLINE_BUDGET_EXCEEDED and its R marked
                # missing; the rest of the batch continues.  Truncating instead
                # is forbidden (contract section 7), and aborting the batch made
                # one long episode destroy every other episode's evidence.
                req.walker.abort_online_budget(
                    prompt_tokens=n_tok,
                    max_new_tokens=int(cfg.generation.max_new_tokens),
                    limit=int(cfg.max_seq_len))
                continue
            admitted.append(req)
        if admitted:
            prefixes = [r.prefix for r in admitted]
            gens = greedy_generate_detailed(
                bundle, [r.prompt for r in admitted],
                prefix_embeds=(prefixes if any(p is not None for p in prefixes)
                               else None),
                config=cfg.generation, parser=parser)
            for req, gen in zip(admitted, gens):
                req.walker.submit(gen)
        active = [w for w in still if not w.finished]
    return [w.episode_record() for w in walkers]


def run_episode(
    bundle: Any,
    ep: Any,
    env: EpisodeEnvironment,
    *,
    cfg: LearningCurveConfig | None = None,
    callbacks: Any = None,
    render_module: Any = None,
    parser: Callable[[str], str | None] | None = None,
    walker_class: type = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Single-episode convenience wrapper around :func:`run_episodes`."""
    return run_episodes(
        bundle, [ep], lambda _ep: env, cfg=cfg,
        callbacks_factory=(lambda _ep: callbacks) if callbacks is not None else None,
        render_module=render_module, parser=parser, walker_class=walker_class)[0]


def episode_R(record: dict[str, Any]) -> dict[int, float]:
    return {int(k): float(v) for k, v in (record.get("R") or {}).items()}


def annotate_records(records: Sequence[dict[str, Any]], **fields: Any) -> list[dict[str, Any]]:
    """Set post-hoc record fields (e.g. ``remap_id``) and re-hash the records.

    The hash is recomputed over the annotated content so a record's
    ``record_hash`` always certifies exactly what the record says.
    """
    out = []
    for r in records:
        new = dict(r)
        new.update({k: _jsonable(v) for k, v in fields.items()})
        new.pop("record_hash", None)
        new["record_hash"] = domain_sha256(EPISODE_RECORD_DOMAIN, new)
        out.append(new)
    return out


__all__ = [
    "EPISODE_RECORD_SCHEMA",
    "STRUCTURE_V0",
    "K_INTERACTIONS",
    "INTERACTION_INDICES",
    "TASK_ROLES",
    "FEEDBACK_ROLES",
    "ATTEMPT_SENTINEL",
    "MAX_SEQ_LEN",
    "EVAL_ONLINE_MAX_SEQ_LEN",
    "STATUS_COMPLETE",
    "STATUS_ONLINE_BUDGET_EXCEEDED",
    "FEEDBACK_CONDITIONS",
    "CONTEXT_MODES",
    "RenderIntegrationError",
    "EpisodeEnvironmentError",
    "SequenceBudgetError",
    "DEFAULT_ATTEMPT_CUE",
    "FeedbackLeakError",
    "CertifiedChannelViolation",
    "annotate_records",
    "EPISODE_RECORD_DOMAIN",
    "VerifierOutcome",
    "EpisodeEnvironment",
    "FamilyEnvironment",
    "environment_from_episode",
    "instances_from_episode",
    "rule_from_episode",
    "hint_conditions_from_episode",
    "resolve_reset_renderer",
    "CERTIFIED_CORRECT",
    "CERTIFIED_INCORRECT",
    "EpisodeContext",
    "AttemptEvent",
    "FeedbackEvent",
    "AdaptationCallbacks",
    "NullCallbacks",
    "LearningCurveConfig",
    "EpisodeWalker",
    "PromptRequest",
    "run_episode",
    "run_episodes",
    "episode_R",
    "role_name",
    "event_attr",
    "replace_fields",
    "episode_events",
    "rendered_text",
    "rendered_spans",
    "resolve_renderer",
    "condition_get",
]
