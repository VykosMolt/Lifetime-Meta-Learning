"""Teacher-forced answer-span scoring (contract §23; FL4 target machinery)."""

from __future__ import annotations

import pytest
import torch

from foundation_learner.evaluation import scoring as SC
from foundation_learner.tests import _w4_support as S

pytestmark = pytest.mark.skipif(
    not S.checkpoint_present(),
    reason="frozen checkpoint directory (code + tokenizer) not present")

TEXT = ("TASK: a hidden rule maps inputs to outputs\n"
        "ATTEMPT: reasoning about the rule\nANSWER: 1 0 1\n")


def _manual_mean_logprob(bundle, text: str, t0: int, t1: int) -> float:
    """Independent recomputation: full forward, log-softmax, gather, mean."""
    ids, _offsets = SC.encode_with_offsets(bundle.tokenizer, text)
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        out = bundle.model(input_ids=input_ids,
                           attention_mask=torch.ones_like(input_ids),
                           use_cache=False)
    logprobs = torch.log_softmax(out.logits.float(), dim=-1)
    values = [float(logprobs[0, j - 1, ids[j]].item()) for j in range(t0, t1)]
    return sum(values) / len(values)


def _aligned_span(bundle, text: str, t0: int, t1: int) -> tuple[int, int]:
    _ids, offsets = SC.encode_with_offsets(bundle.tokenizer, text)
    return offsets[t0][0], offsets[t1 - 1][1]


# -- alignment -----------------------------------------------------------
def test_alignment_accepts_exact_token_boundaries():
    bundle = S.tiny_bundle()
    _ids, offsets = SC.encode_with_offsets(bundle.tokenizer, TEXT)
    span = (offsets[4][0], offsets[8][1])
    assert SC.align_char_span_to_tokens(offsets, span) == (4, 9)


def test_alignment_rejects_spans_that_split_a_token():
    bundle = S.tiny_bundle()
    _ids, offsets = SC.encode_with_offsets(bundle.tokenizer, TEXT)
    inner = next((a, b) for a, b in offsets if b - a > 1)
    with pytest.raises(SC.SpanAlignmentError):
        SC.align_char_span_to_tokens(offsets, (inner[0] + 1, inner[1]))
    with pytest.raises(SC.SpanAlignmentError):
        SC.align_char_span_to_tokens(offsets, (inner[0], inner[1] - 1))
    with pytest.raises(SC.SpanAlignmentError):
        SC.align_char_span_to_tokens(offsets, (5, 5))


def test_covering_alignment_is_a_separate_deliberate_relaxation():
    bundle = S.tiny_bundle()
    _ids, offsets = SC.encode_with_offsets(bundle.tokenizer, TEXT)
    inner = next((a, b) for a, b in offsets if b - a > 1)
    lo, hi = SC.align_char_span_covering(offsets, (inner[0] + 1, inner[1]))
    assert lo < hi
    with pytest.raises(SC.SpanAlignmentError):
        SC.align_char_span_covering(offsets, (3, 3))


@pytest.mark.skipif(not S.integration_available(), reason="episodes package absent")
def test_rendered_event_spans_are_token_aligned():
    """The renderer's event spans land on token boundaries, so FL4 targets
    computed over event blocks are exactly scoreable."""
    from foundation_learner.episodes.render import render_events

    bundle = S.tiny_bundle()
    ep, _family, _rule = S.real_episode()
    rendered = render_events(ep.events)
    _ids, offsets = SC.encode_with_offsets(bundle.tokenizer, rendered.text)
    for span in rendered.spans:
        SC.align_char_span_to_tokens(offsets, (span.start, span.end))


# -- values --------------------------------------------------------------
def test_answer_logprob_matches_an_independent_recomputation():
    bundle = S.tiny_bundle()
    span = _aligned_span(bundle, TEXT, 6, 11)
    got = SC.answer_logprob(bundle, TEXT, span)
    expected = _manual_mean_logprob(bundle, TEXT, 6, 11)
    assert got == pytest.approx(expected, abs=1e-5)
    assert got < 0.0


def test_detailed_result_reports_tokens_and_totals():
    bundle = S.tiny_bundle()
    span = _aligned_span(bundle, TEXT, 6, 11)
    result = SC.answer_logprob_detailed(bundle, TEXT, span)
    assert result.n_tokens == 5
    assert (result.token_start, result.token_end) == (6, 11)
    assert result.total_logprob == pytest.approx(sum(result.token_logprobs), abs=1e-6)
    assert result.mean_logprob == pytest.approx(
        result.total_logprob / result.n_tokens, abs=1e-9)
    assert result.to_dict()["schema"] == SC.SCORE_RECORD_SCHEMA


def test_batched_scoring_matches_single_row_scoring():
    bundle = S.tiny_bundle()
    long_text = TEXT
    short_text = "TASK: x\nANSWER: 1 0 1\n"
    items = [SC.ScoreRequest(long_text, _aligned_span(bundle, long_text, 6, 11)),
             SC.ScoreRequest(short_text, _aligned_span(bundle, short_text, 3, 7))]
    batched = SC.answer_logprob_batch(bundle, items)
    singles = [SC.answer_logprob(bundle, it.text, it.answer_char_span) for it in items]
    for a, b in zip(batched, singles):
        assert a == pytest.approx(b, abs=1e-4)


def test_tuple_items_are_accepted_and_empty_input_is_empty():
    bundle = S.tiny_bundle()
    span = _aligned_span(bundle, TEXT, 6, 11)
    assert SC.answer_logprob_batch(bundle, []) == []
    assert SC.answer_logprob_batch(bundle, [(TEXT, span)])[0] == pytest.approx(
        SC.answer_logprob(bundle, TEXT, span), abs=1e-6)


def test_span_without_any_conditioning_context_is_rejected():
    bundle = S.tiny_bundle()
    span = _aligned_span(bundle, TEXT, 0, 3)
    with pytest.raises(SC.SpanAlignmentError):
        SC.answer_logprob(bundle, TEXT, span)


def test_prefix_embeds_change_the_score_and_enable_position_zero():
    bundle = S.tiny_bundle()
    hidden = bundle.model.get_input_embeddings().weight.shape[1]
    gen = torch.Generator().manual_seed(5)
    prefix = torch.empty(8, hidden).normal_(0.0, 1.0, generator=gen)
    span = _aligned_span(bundle, TEXT, 6, 11)
    plain = SC.answer_logprob(bundle, TEXT, span)
    prefixed = SC.answer_logprob(bundle, TEXT, span, prefix_embeds=prefix)
    assert plain != prefixed
    first = _aligned_span(bundle, TEXT, 0, 3)
    assert SC.answer_logprob(bundle, TEXT, first, prefix_embeds=prefix) < 0.0


def test_fl4_style_presence_minus_ablation_difference_is_finite():
    """The FL4 target is a DIFFERENCE of two of these numbers."""
    bundle = S.tiny_bundle()
    with_item = TEXT
    without_item = TEXT.replace("TASK: a hidden rule maps inputs to outputs\n", "")
    y = (SC.answer_logprob(bundle, with_item, _aligned_span(bundle, with_item, 6, 11))
         - SC.answer_logprob(bundle, without_item,
                             _aligned_span(bundle, without_item, 2, 7)))
    assert abs(y) < 1e3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
