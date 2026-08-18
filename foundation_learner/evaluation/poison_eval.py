"""Poison-condition evaluation (contract sections 4, 6, 9).

The five frozen conditions are

    correct_informative (a.k.a. "clean"), correct_redundant, irrelevant,
    partially_misleading, corrupted

and each episode carries its condition and its per-event poison flags from the
DATA layer's pre-generated schedule; this module never decides which items are
poisoned.

Two guarantees are ENFORCED here rather than assumed:

1. The certified channel is never corrupted.  ``EpisodeWalker`` raises
   ``CertifiedChannelViolation`` if a certified event is flagged poison, and
   certified FEEDBACK text is always the verifier's own verdict.
2. Poison is not detectable from the rendering.  ``PoisonAuditWalker``
   (a) scans every prompt against a frozen deny-list of marking words, and
   (b) records, for every HINT, the rendered DECORATION SIGNATURE
   (the line prefix before the event plus the event's rendered wrapper with
   its content masked).  ``run_poison_eval`` then requires the signature sets
   of poisoned and truthful hints to be identical: a rendering that marks
   poison necessarily produces a signature that occurs in one group only.

The environment records the poison flag; the model is never told, and the
flag never reaches a mechanism (see the callback protocol in
``learning_curve``).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .learning_curve import (
    EpisodeWalker,
    LearningCurveConfig,
    annotate_records,
    condition_get,
    event_attr,
    rendered_spans,
    rendered_text,
    role_name,
    run_episodes,
)
from . import metrics as _metrics
from .metrics import CLEAN_CONDITION, POISON_CONDITIONS, canonical_poison_condition

POISON_REPORT_SCHEMA = "flb200.poison_report.v1"

#: Words whose presence in a rendered prompt would tell the model (or a reader)
#: that an item is poisoned.  Case-insensitive substring scan.
POISON_MARKER_DENYLIST: tuple[str, ...] = (
    "poison", "corrupt", "uncertified", "untrusted", "unreliable",
    "may be false", "possibly wrong", "do not trust",
)


class PoisonMarkingError(AssertionError):
    """The rendering reveals which items are poisoned."""


class PoisonAuditWalker(EpisodeWalker):
    """:class:`EpisodeWalker` that audits every prompt and every hint rendering."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hint_signatures: list[dict[str, Any]] = []
        self.marker_hits: list[dict[str, Any]] = []

    # -- prompt scan -----------------------------------------------------
    def _scan_prompt(self, prompt: str) -> None:
        low = prompt.lower()
        for marker in POISON_MARKER_DENYLIST:
            if marker in low:
                self.marker_hits.append({"marker": marker,
                                         "episode_id": self.ctx.episode_id})
                raise PoisonMarkingError(
                    f"rendered prompt contains the poison-marking token {marker!r}; "
                    "truthful and poisoned hints must render identically "
                    "(contract section 6)")

    def _attempt_prompt(self, attempt_event: Any) -> str:
        prompt = super()._attempt_prompt(attempt_event)
        self._scan_prompt(prompt)
        return prompt

    def _reset_prompt(self, source_index: int) -> str:
        prompt = super()._reset_prompt(source_index)
        self._scan_prompt(prompt)
        return prompt

    # -- hint decoration signature ---------------------------------------
    def _append_feedback(self, ev: Any) -> None:
        n_before = len(self.live)
        super()._append_feedback(ev)
        if len(self.live) == n_before:
            return  # dropped (correctness_only condition)
        if role_name(ev) != "HINT":
            return
        live_ev = self.live[-1]
        rendered = self._render_live(self.live)
        text = rendered_text(rendered)
        start, end = rendered_spans(rendered)[len(self.live) - 1]
        content = str(event_attr(live_ev, "text", "") or "")
        span_text = text[start:end]
        line_start = text.rfind("\n", 0, start) + 1
        signature = (text[line_start:start],
                     span_text.replace(content, "<CONTENT>") if content in span_text
                     else "<UNRESOLVED>")
        self.hint_signatures.append({
            "episode_id": self.ctx.episode_id,
            "item_id": str(event_attr(live_ev, "item_id", "")),
            "poison": bool(event_attr(ev, "poison", False)),
            "signature": list(signature),
        })

    def episode_record(self) -> dict[str, Any]:
        record = super().episode_record()
        record["hint_signatures"] = self.hint_signatures
        from . import domain_sha256
        from .learning_curve import EPISODE_RECORD_DOMAIN
        record.pop("record_hash", None)
        record["record_hash"] = domain_sha256(EPISODE_RECORD_DOMAIN, record)
        return record


def apply_hint_variant(ep: Any, variant_rows: Sequence[Mapping[str, Any]],
                       condition: str) -> Any:
    """Materialize the poison-condition variant of ONE pre-generated episode.

    The data layer stores, per episode, one event plan plus a table of
    alternative HINT bodies per condition (``event_index``, ``condition``,
    ``poison``, ``text``).  Only the uncertified HINT events differ across
    conditions, which is what makes the poison comparison controlled.  The
    ONLINE evaluator still recomputes the hint body from the model's real
    attempt; what this function transfers is the SCHEDULE (which slot is in
    which condition, and its poison flag), not the pre-generated text.
    """
    from .learning_curve import episode_events, replace_fields

    events = list(episode_events(ep))
    slots: list[dict[str, Any]] = []
    for row in variant_rows:
        index = int(row["event_index"])
        if role_name(events[index]) != "HINT":
            raise ValueError(
                f"hint variant targets event {index} whose role is "
                f"{role_name(events[index])}, not HINT")
        events[index] = replace_fields(events[index], poison=bool(row.get("poison")))
        slots.append({"event_index": index, "slot": row.get("slot"),
                      "item_id": row.get("item_id"),
                      "condition": row.get("condition"),
                      "poison": bool(row.get("poison"))})
    new_ep = replace_fields(ep, events=events, condition=str(condition))
    meta = event_attr(new_ep, "meta")
    if isinstance(meta, dict):
        merged = dict(meta)
        existing = {str(s.get("event_index")): s for s in merged.get("hint_slots", [])}
        for slot in slots:
            base = dict(existing.get(str(slot["event_index"]), {}))
            base.update({k: v for k, v in slot.items() if v is not None})
            existing[str(slot["event_index"])] = base
        merged["hint_slots"] = [existing[k] for k in sorted(existing, key=int)]
        merged["poison_condition"] = str(condition)
        new_ep = replace_fields(new_ep, meta=merged)
    return new_ep


def episodes_from_poison_record(ep: Any, poison_record: Mapping[str, Any]) -> dict[str, Any]:
    """``condition -> episode`` from a data-layer poison record."""
    variants = poison_record.get("variants") or {}
    return {str(condition): apply_hint_variant(ep, rows, str(condition))
            for condition, rows in variants.items()}


def episode_poison_condition(ep: Any) -> str:
    """Resolve the frozen poison-condition id carried by an episode."""
    value = condition_get(ep, "poison_condition")
    if value is None:
        cond = event_attr(ep, "condition")
        value = cond if isinstance(cond, str) else None
    if value is None:
        raise ValueError(
            "episode carries no poison condition; the data layer supplies the "
            "per-episode schedule (contract section 16)")
    return canonical_poison_condition(str(value))


def audit_hint_signatures(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require identical rendered signatures for poisoned and truthful hints."""
    poisoned: set[tuple[str, str]] = set()
    truthful: set[tuple[str, str]] = set()
    unresolved = 0
    for rec in records:
        for sig in rec.get("hint_signatures", []) or []:
            pair = (str(sig["signature"][0]), str(sig["signature"][1]))
            if pair[1] == "<UNRESOLVED>":
                unresolved += 1
            (poisoned if sig.get("poison") else truthful).add(pair)
    only_poisoned = sorted(poisoned - truthful)
    only_truthful = sorted(truthful - poisoned)
    if poisoned and truthful and only_poisoned:
        raise PoisonMarkingError(
            f"rendering signature(s) {only_poisoned} occur only for poisoned hints")
    return {
        "n_poisoned_signatures": len(poisoned),
        "n_truthful_signatures": len(truthful),
        "signatures_only_poisoned": [list(s) for s in only_poisoned],
        "signatures_only_truthful": [list(s) for s in only_truthful],
        "unresolved_signatures": unresolved,
    }


def run_poison_eval(
    bundle: Any,
    episodes: Sequence[Any],
    env_factory: Callable[[Any], Any],
    *,
    cfg: LearningCurveConfig | None = None,
    condition_of: Callable[[Any], str] | None = None,
    callbacks_factory: Callable[[Any], Any] | None = None,
    render_module: Any = None,
    parser: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Run every poison condition present and report robustness vs clean."""
    cfg = cfg or LearningCurveConfig()
    condition_of = condition_of or episode_poison_condition

    grouped: dict[str, list[Any]] = {}
    for ep in episodes:
        grouped.setdefault(condition_of(ep), []).append(ep)
    unknown = set(grouped) - set(POISON_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown poison conditions {sorted(unknown)}")

    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition, eps in sorted(grouped.items()):
        recs = run_episodes(
            bundle, eps, env_factory, cfg=cfg,
            callbacks_factory=callbacks_factory, render_module=render_module,
            parser=parser, walker_class=PoisonAuditWalker)
        records_by_condition[condition] = annotate_records(
            recs, poison_condition=condition)

    all_records = [r for rs in records_by_condition.values() for r in rs]
    audit = audit_hint_signatures(all_records)
    robustness = _metrics.poison_robustness(records_by_condition)
    return {
        "schema": POISON_REPORT_SCHEMA,
        "clean_condition": CLEAN_CONDITION,
        "conditions_present": sorted(records_by_condition),
        "n_episodes_by_condition": {k: len(v) for k, v in records_by_condition.items()},
        "rendering_audit": audit,
        "robustness": robustness,
        "records_by_condition": records_by_condition,
    }


__all__ = [
    "POISON_REPORT_SCHEMA",
    "POISON_MARKER_DENYLIST",
    "PoisonMarkingError",
    "PoisonAuditWalker",
    "apply_hint_variant",
    "episodes_from_poison_record",
    "episode_poison_condition",
    "audit_hint_signatures",
    "run_poison_eval",
]
