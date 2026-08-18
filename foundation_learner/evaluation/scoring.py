"""Teacher-forced answer-span scoring (contract section 23; FL4 target maths).

``answer_logprob(bundle, text, answer_char_span, prefix_embeds=None)`` returns
the MEAN PER-TOKEN log-probability of the tokens that cover
``text[answer_char_span[0]:answer_char_span[1]]``, conditioned on everything
that precedes them (plus optional prefix embeddings).

Character -> token alignment is EXACT.  Spans handed to this function come
from the renderer's span map and are therefore expected to fall on token
boundaries; a span that does not is a hard failure
(:class:`SpanAlignmentError`), never a silently snapped approximation, because
FL4 targets are differences of two such numbers and a silent re-alignment
would quietly change the estimand.

The FL4 target is
``y_j = mean-logprob(correct QUERY answers | item j present)
      - mean-logprob(correct QUERY answers | item j ablated)``
so both halves must be computed with this same function and the same span
convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .generation import GenerationError, _bundle_parts, _pad_id, normalize_prefix_embeds

SCORE_RECORD_SCHEMA = "flb200.answer_score.v1"


class SpanAlignmentError(ValueError):
    """A character span does not coincide with tokenizer boundaries."""


@dataclass(frozen=True)
class ScoreRequest:
    """One scoring item: full ``text`` plus the answer character span."""

    text: str
    answer_char_span: tuple[int, int]


@dataclass(frozen=True)
class ScoreResult:
    mean_logprob: float
    total_logprob: float
    n_tokens: int
    token_start: int
    token_end: int
    token_logprobs: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCORE_RECORD_SCHEMA,
            "mean_logprob": self.mean_logprob,
            "total_logprob": self.total_logprob,
            "n_tokens": self.n_tokens,
            "token_start": self.token_start,
            "token_end": self.token_end,
        }


def encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize with char offsets (requires a fast tokenizer)."""
    if not getattr(tokenizer, "is_fast", False):
        raise GenerationError(
            "exact char->token alignment requires a fast tokenizer "
            "(tokenizer.json); the frozen checkpoint provides one")
    enc = tokenizer(str(text), add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(t) for t in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    return ids, offsets


def align_char_span_to_tokens(
    offsets: Sequence[tuple[int, int]],
    span: tuple[int, int],
) -> tuple[int, int]:
    """Return ``[token_start, token_end)`` covering ``span`` exactly.

    Raises :class:`SpanAlignmentError` unless ``span`` starts exactly at a
    token start and ends exactly at a token end, with no partially covered
    token in between.
    """
    lo, hi = int(span[0]), int(span[1])
    if hi <= lo:
        raise SpanAlignmentError(f"empty or inverted answer span {span!r}")
    start = None
    end = None
    for i, (a, b) in enumerate(offsets):
        if b <= a:  # zero-width tokens carry no characters
            continue
        if a == lo and start is None:
            start = i
        if b == hi:
            end = i + 1
        if a < lo < b:
            raise SpanAlignmentError(
                f"answer span start {lo} falls inside token {i} covering ({a}, {b})")
        if a < hi < b:
            raise SpanAlignmentError(
                f"answer span end {hi} falls inside token {i} covering ({a}, {b})")
    if start is None:
        raise SpanAlignmentError(f"no token starts exactly at char {lo}")
    if end is None:
        raise SpanAlignmentError(f"no token ends exactly at char {hi}")
    if end <= start:
        raise SpanAlignmentError(f"degenerate token span [{start}, {end}) for chars {span!r}")
    return start, end


def align_char_span_covering(
    offsets: Sequence[tuple[int, int]],
    span: tuple[int, int],
) -> tuple[int, int]:
    """Return ``[token_start, token_end)`` of every token INTERSECTING ``span``.

    This deliberately differs from :func:`align_char_span_to_tokens`: hidden-state
    summaries (FL5 ``u_j``, FL4 head input) are taken over "the tokens of item j"
    where the item boundary is an event boundary that a byte-level BPE token may
    straddle.  Log-probability estimands use the STRICT function; hidden-state
    summaries use this covering function.  The two are never interchanged.
    """
    lo, hi = int(span[0]), int(span[1])
    if hi <= lo:
        raise SpanAlignmentError(f"empty or inverted span {span!r}")
    idx = [i for i, (a, b) in enumerate(offsets) if b > a and a < hi and b > lo]
    if not idx:
        raise SpanAlignmentError(f"no tokens intersect chars {span!r}")
    return idx[0], idx[-1] + 1


def _positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask.cumsum(-1) - 1).clamp(min=0)


def answer_logprob_batch(
    bundle: Any,
    items: Sequence[ScoreRequest | tuple[str, tuple[int, int]]],
    prefix_embeds: Any = None,
    *,
    detailed: bool = False,
) -> list[float] | list[ScoreResult]:
    """Batched teacher-forced mean per-token log-probability of answer spans.

    Rows are LEFT padded exactly as in ``generation.py`` (same mask and
    ``position_ids`` conventions), so a batched score equals the score the row
    would receive alone up to floating-point noise.
    """
    reqs = [it if isinstance(it, ScoreRequest) else ScoreRequest(it[0], tuple(it[1]))
            for it in items]
    if not reqs:
        return []
    model, tokenizer, device = _bundle_parts(bundle)
    pad_id = _pad_id(tokenizer)
    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    prefixes = normalize_prefix_embeds(prefix_embeds, len(reqs), hidden_size)

    encoded = []
    for r in reqs:
        ids, offsets = encode_with_offsets(tokenizer, r.text)
        t0, t1 = align_char_span_to_tokens(offsets, r.answer_char_span)
        encoded.append((ids, t0, t1))

    prefix_lens = [0 if p is None else int(p.shape[0]) for p in prefixes]
    total_lens = [prefix_lens[i] + len(encoded[i][0]) for i in range(len(reqs))]
    width = max(total_lens)
    n = len(reqs)

    attention_mask = torch.zeros((n, width), dtype=torch.long, device=device)
    for i in range(n):
        attention_mask[i, width - total_lens[i]:] = 1

    embed = model.get_input_embeddings()
    emb_dtype = embed.weight.dtype
    rows = []
    for i in range(n):
        ids = torch.tensor(encoded[i][0], dtype=torch.long, device=device)
        parts = []
        pad_n = width - total_lens[i]
        if pad_n:
            parts.append(embed(torch.full((pad_n,), pad_id, dtype=torch.long, device=device)))
        if prefixes[i] is not None:
            parts.append(prefixes[i].to(device=device, dtype=emb_dtype))
        parts.append(embed(ids))
        rows.append(torch.cat(parts, dim=0))
    inputs_embeds = torch.stack(rows, dim=0)

    with torch.inference_mode():
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=_positions_from_mask(attention_mask),
            use_cache=False,
        )
    logits = out.logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)

    results: list[ScoreResult] = []
    for i in range(n):
        ids, t0, t1 = encoded[i]
        base = width - len(ids)  # absolute position of token 0 of this row
        row_start = width - total_lens[i]  # first non-padding position of the row
        abs_start = base + t0
        if abs_start - 1 < row_start:
            raise SpanAlignmentError(
                "answer span starts at the first real position of the row: the "
                "conditioning position would be padding, so the score would be "
                "undefined (give the span a prompt, or supply prefix_embeds)")
        per_token = []
        for j in range(t0, t1):
            pos = base + j
            lp = logprobs[i, pos - 1, int(ids[j])]
            per_token.append(float(lp.item()))
        if not per_token:
            raise SpanAlignmentError("answer span covers zero tokens")
        total = float(sum(per_token))
        results.append(ScoreResult(
            mean_logprob=total / len(per_token),
            total_logprob=total,
            n_tokens=len(per_token),
            token_start=t0,
            token_end=t1,
            token_logprobs=tuple(per_token),
        ))
    if detailed:
        return results
    return [r.mean_logprob for r in results]


def answer_logprob(
    bundle: Any,
    text: str,
    answer_char_span: tuple[int, int],
    prefix_embeds: Any = None,
) -> float:
    """Contract section 23 signature: teacher-forced mean per-token logprob."""
    vals = answer_logprob_batch(bundle, [ScoreRequest(text, tuple(answer_char_span))],
                                prefix_embeds=prefix_embeds)
    return float(vals[0])


def answer_logprob_detailed(
    bundle: Any,
    text: str,
    answer_char_span: tuple[int, int],
    prefix_embeds: Any = None,
) -> ScoreResult:
    res = answer_logprob_batch(bundle, [ScoreRequest(text, tuple(answer_char_span))],
                               prefix_embeds=prefix_embeds, detailed=True)
    return res[0]  # type: ignore[return-value]


__all__ = [
    "SCORE_RECORD_SCHEMA",
    "SpanAlignmentError",
    "ScoreRequest",
    "ScoreResult",
    "encode_with_offsets",
    "align_char_span_to_tokens",
    "align_char_span_covering",
    "answer_logprob",
    "answer_logprob_batch",
    "answer_logprob_detailed",
]
