"""Context-reset evaluation — LOAD BEARING (contract section 9).

Procedure, per episode:

1. run the learning interactions normally (support items, real attempts, real
   verifier feedback) — the textual history exists during this phase;
2. let the mechanism update through the callbacks (``on_feedback``);
3. REMOVE the textual learning history: from ``reset_from_index`` onwards the
   prompt is re-rendered as instruction header + the current query ONLY, via
   ``episodes.render.render_reset_query``, plus the renderer's own attempt cue
   and any mechanism-supplied prefix embeddings;
4. answer the related / query / transfer items under that reset context;
5. measure retained improvement against MATCHED history-present episodes (same
   episodes, same seeds, same callbacks factory, same generation config).

The removal in step 3 is verified, not asserted by construction: every reset
prompt is checked against the text of EVERY prior live event of that episode
(support statements, the model's own attempts, certified feedback, hints,
reveals, earlier queries).  Any occurrence raises
:class:`ContextLeakError`.  The hostile fixture plants a marker string inside
the support text and requires it to be absent from the reset prompt.

FL0/FL3 history-only behaviour is EXPECTED to collapse here; that collapse is
the control against which FL5/FL7 persistence is measured.  No result of this
module is evidence of persistence by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from . import text_sha256
from .learning_curve import (
    EpisodeWalker,
    LearningCurveConfig,
    event_attr,
    role_name,
    run_episodes,
)

CONTEXT_RESET_REPORT_SCHEMA = "flb200.context_reset_report.v1"

#: Prior-event texts shorter than this are not searched for as substrings: a
#: 1-3 character fragment collides with ordinary prompt content and would make
#: the check meaningless rather than strict.  Every skipped text is REPORTED.
MIN_LEAK_PROBE_LEN = 4


class ContextLeakError(AssertionError):
    """Textual learning history survived into a context-reset prompt."""


@dataclass(frozen=True)
class LeakReport:
    prompt_sha256: str
    n_probes: int
    n_skipped_short: int
    leaks: tuple[tuple[str, str], ...]  # (role, offending text)

    @property
    def clean(self) -> bool:
        return not self.leaks

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_sha256": self.prompt_sha256,
            "n_probes": self.n_probes,
            "n_skipped_short": self.n_skipped_short,
            "leaks": [{"role": r, "text": t} for r, t in self.leaks],
            "clean": self.clean,
        }


def check_no_history_leak(
    prompt: str,
    prior_events: Sequence[Any],
    *,
    current_text: str | None = None,
    min_len: int = MIN_LEAK_PROBE_LEN,
) -> LeakReport:
    """Report every prior-event text that survived into ``prompt``.

    ``current_text`` is the body of the item the reset prompt is legitimately
    allowed to state (the current query).  Exactly ONE occurrence of it is
    removed before probing, because two items of the same family can carry
    byte-identical surfaces (a Boolean assignment repeats after 2^k draws) and
    a naive substring probe would then flag the query itself as a leak.
    Removing one occurrence keeps the check strict: a renderer that really
    kept the history leaves the support block behind in the residual, whether
    or not it duplicates the query text.
    """
    residual = prompt
    if current_text:
        body = str(current_text).strip()
        if body and body in residual:
            residual = residual.replace(body, "\x00", 1)
    leaks: list[tuple[str, str]] = []
    probes = 0
    skipped = 0
    for ev in prior_events:
        text = str(event_attr(ev, "text", "") or "").strip()
        if len(text) < int(min_len):
            skipped += 1
            continue
        probes += 1
        if text in residual:
            leaks.append((role_name(ev), text))
    return LeakReport(text_sha256(prompt), probes, skipped, tuple(leaks))


def assert_no_history_leak(prompt: str, prior_events: Sequence[Any],
                           *, current_text: str | None = None,
                           min_len: int = MIN_LEAK_PROBE_LEN,
                           context: str = "") -> LeakReport:
    report = check_no_history_leak(prompt, prior_events,
                                   current_text=current_text, min_len=min_len)
    if not report.clean:
        roles = ", ".join(sorted({r for r, _ in report.leaks}))
        raise ContextLeakError(
            f"context-reset prompt {context} still contains history text from: {roles}")
    return report


class LeakCheckingWalker(EpisodeWalker):
    """:class:`EpisodeWalker` that verifies every reset prompt it builds."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.leak_reports: list[dict[str, Any]] = []

    def _reset_prompt(self, source_index: int) -> str:
        prompt = super()._reset_prompt(source_index)
        prior = [ev for ev, src in zip(self.live, self.live_source_index)
                 if src != source_index]
        current = next((str(event_attr(ev, "text", "") or "")
                        for ev, src in zip(self.live, self.live_source_index)
                        if src == source_index), None)
        report = assert_no_history_leak(
            prompt, prior, current_text=current,
            context=f"(episode {self.ctx.episode_id})")
        self.leak_reports.append(report.to_dict())
        return prompt

    def episode_record(self) -> dict[str, Any]:
        record = super().episode_record()
        record["context_reset_leak_reports"] = self.leak_reports
        from . import domain_sha256
        from .learning_curve import EPISODE_RECORD_DOMAIN
        record.pop("record_hash", None)
        record["record_hash"] = domain_sha256(EPISODE_RECORD_DOMAIN, record)
        return record


def run_context_reset_eval(
    bundle: Any,
    episodes: Sequence[Any],
    env_factory: Callable[[Any], Any],
    *,
    cfg: LearningCurveConfig | None = None,
    callbacks_factory: Callable[[Any], Any] | None = None,
    render_module: Any = None,
    parser: Callable[[str], str | None] | None = None,
    run_history_control: bool = True,
) -> dict[str, Any]:
    """Run the reset arm and (by default) the matched history control.

    Both arms use identical episodes, identical generation config and an
    identical ``callbacks_factory``; only ``context_mode`` differs, which is
    exactly the comparison contract metric 10 (context-reset persistence)
    requires.
    """
    base = cfg or LearningCurveConfig()
    reset_cfg = replace(base, context_mode="context_reset")
    reset_records = run_episodes(
        bundle, episodes, env_factory, cfg=reset_cfg,
        callbacks_factory=callbacks_factory, render_module=render_module,
        parser=parser, walker_class=LeakCheckingWalker)

    history_records: list[dict[str, Any]] = []
    if run_history_control:
        history_cfg = replace(base, context_mode="history")
        history_records = run_episodes(
            bundle, episodes, env_factory, cfg=history_cfg,
            callbacks_factory=callbacks_factory, render_module=render_module,
            parser=parser)

    from .metrics import context_reset_persistence, macro_aulc

    summary: dict[str, Any] = {
        "schema": CONTEXT_RESET_REPORT_SCHEMA,
        "arm_tag": base.arm_tag,
        "feedback_condition": base.feedback_condition,
        "reset_from_index": int(base.reset_from_index),
        "n_episodes": len(reset_records),
        "reset_macro_aulc": macro_aulc(reset_records),
        "history_macro_aulc": macro_aulc(history_records) if history_records else None,
        "n_leak_reports": sum(len(r.get("context_reset_leak_reports", []))
                              for r in reset_records),
        "leaks_detected": 0,  # a leak raises; a completed run has none
    }
    if history_records:
        summary["persistence"] = context_reset_persistence(
            reset_records, history_records, reset_from_index=int(base.reset_from_index))
    return {
        "summary": summary,
        "reset_records": reset_records,
        "history_records": history_records,
    }


__all__ = [
    "CONTEXT_RESET_REPORT_SCHEMA",
    "MIN_LEAK_PROBE_LEN",
    "ContextLeakError",
    "LeakReport",
    "check_no_history_leak",
    "assert_no_history_leak",
    "LeakCheckingWalker",
    "run_context_reset_eval",
]
