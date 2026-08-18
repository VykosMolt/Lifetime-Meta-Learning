"""Manual greedy decode loop + equivalence gate (contract §22, §23)."""

from __future__ import annotations

import pytest
import torch

from foundation_learner.evaluation import generation as G
from foundation_learner.tests import _w4_support as S

pytestmark = pytest.mark.skipif(
    not S.checkpoint_present(),
    reason="frozen checkpoint directory (code + tokenizer) not present")

MIXED_PROMPTS = [
    "TASK: short\nATTEMPT:",
    "TASK: a considerably longer support problem statement whose token count "
    "differs from the others so that left padding is genuinely exercised\n"
    "FEEDBACK: INCORRECT\nATTEMPT:",
    "TASK: middling length statement\nATTEMPT:",
]


# -- answer grammar ------------------------------------------------------
def test_local_parser_takes_the_last_answer_line():
    text = "ANSWER: 3\nsome noise\nANSWER: 7\n"
    assert G.parse_answer_local(text) == "7"
    assert G.parse_answer_local("no commitment here") is None
    assert G.parse_answer_local(" ANSWER: 3") is None  # leading space rejected
    assert G.parse_answer_local("answer: 3") is None   # case sensitive


def test_local_parser_matches_the_producer_grammar():
    from foundation_learner.episodes import parse as real_parse

    for text in ("ANSWER: 1", "ANSWER: a b c\n", "junk\nANSWER: YES\ntail",
                 "ANSWER:\n", "nothing"):
        assert G.parse_answer_local(text) == real_parse.parse_answer(text), text


def test_streaming_stop_requires_a_terminated_answer_line():
    # an unterminated line must NOT stop generation: multi-token payloads
    # would be truncated after their first token
    assert G.complete_answer_line_end("ANSWER: a b") is None
    end = G.complete_answer_line_end("ANSWER: a b c\ntail")
    assert end is not None and "ANSWER: a b c\n" == "ANSWER: a b c\ntail"[:end]
    assert G.complete_answer_line_end("no answer\n") is None


def test_resolved_parser_prefers_the_real_episodes_parse():
    parser, source = G.resolve_answer_parser()
    assert source == "episodes.parse.parse_answer"
    assert parser("ANSWER: 4") == "4"


# -- the decode loop -----------------------------------------------------
def test_greedy_generate_returns_texts_and_respects_max_new_tokens():
    bundle = S.tiny_bundle()
    records = G.greedy_generate_detailed(bundle, MIXED_PROMPTS, max_new_tokens=5)
    assert len(records) == len(MIXED_PROMPTS)
    for rec in records:
        assert len(rec.token_ids) <= 5
        assert rec.finish_reason in ("answer_line", "eos", "max_new_tokens",
                                     "nonfinite_logits")
        assert rec.logits_finite
        assert rec.prompt_token_count > 0
    texts = G.greedy_generate(bundle, MIXED_PROMPTS, max_new_tokens=5)
    assert texts == [r.text for r in records]


def test_model_generate_is_never_used():
    """Contract §22: the evaluation path never calls ``model.generate()``."""
    import ast

    for module in (G, __import__("foundation_learner.evaluation.learning_curve",
                                 fromlist=["x"])):
        tree = ast.parse(open(module.__file__, encoding="utf-8").read())
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in ("generate", "sample", "beam_search")]
        assert not calls, f"{module.__name__}: {[c.func.attr for c in calls]}"


def test_empty_prompt_is_rejected():
    bundle = S.tiny_bundle()
    with pytest.raises(G.GenerationError):
        G.greedy_generate(bundle, [""], max_new_tokens=2)


# -- EQUIVALENCE GATE ----------------------------------------------------
def test_batched_greedy_equals_unbatched_token_for_token():
    bundle = S.tiny_bundle()
    batched = G.greedy_generate_detailed(
        bundle, MIXED_PROMPTS, config=G.GenerationConfig(max_new_tokens=8))
    single = G.greedy_generate_detailed(
        bundle, MIXED_PROMPTS,
        config=G.GenerationConfig(max_new_tokens=8, force_unbatched=True))
    assert [r.token_ids for r in batched] == [r.token_ids for r in single]
    assert [r.batch_size for r in batched] == [3, 3, 3]
    assert [r.batch_size for r in single] == [1, 1, 1]


def test_batched_greedy_equals_unbatched_with_prefix_embeds():
    bundle = S.tiny_bundle()
    prefix = G.build_gate_prefix(bundle, k=4)
    batched = G.greedy_generate_detailed(
        bundle, MIXED_PROMPTS, prefix_embeds=prefix,
        config=G.GenerationConfig(max_new_tokens=8))
    single = G.greedy_generate_detailed(
        bundle, MIXED_PROMPTS, prefix_embeds=prefix,
        config=G.GenerationConfig(max_new_tokens=8, force_unbatched=True))
    assert [r.token_ids for r in batched] == [r.token_ids for r in single]
    assert all(r.prefix_len == 4 for r in batched)


def test_equivalence_gate_passes_and_reports_every_variant():
    bundle = S.tiny_bundle()
    gate = G.run_equivalence_gate(bundle, max_new_tokens=6)
    assert gate["schema"] == G.EQUIVALENCE_GATE_SCHEMA
    assert gate["passed"] is True
    assert {c["variant"] for c in gate["cases"]} == {
        "no_prefix", "shared_prefix", "per_row_prefix"}
    for case in gate["cases"]:
        assert case["passed"]
        assert all(row["first_divergence_step"] is None for row in case["rows"])


def test_gate_failure_forces_batch_size_one():
    base = G.GenerationConfig(max_new_tokens=32, batch_size=16)
    assert G.generation_policy_from_gate({"passed": True}, base) is base
    fallback = G.generation_policy_from_gate({"passed": False}, base)
    assert fallback.force_unbatched is True
    assert fallback.batch_size == 1
    assert fallback.effective_batch_size(64) == 1
    assert fallback.max_new_tokens == 32


def test_unbatched_fallback_path_produces_the_same_generations():
    bundle = S.tiny_bundle()
    fallback = G.generation_policy_from_gate(
        {"passed": False}, G.GenerationConfig(max_new_tokens=6))
    fell_back = G.greedy_generate_detailed(bundle, MIXED_PROMPTS, config=fallback)
    normal = G.greedy_generate_detailed(
        bundle, MIXED_PROMPTS, config=G.GenerationConfig(max_new_tokens=6))
    assert [r.text for r in fell_back] == [r.text for r in normal]
    assert all(r.batch_size == 1 for r in fell_back)


def test_chunked_batch_size_preserves_order():
    bundle = S.tiny_bundle()
    prompts = MIXED_PROMPTS * 2
    chunked = G.greedy_generate_detailed(
        bundle, prompts, config=G.GenerationConfig(max_new_tokens=4, batch_size=2))
    single = G.greedy_generate_detailed(
        bundle, prompts, config=G.GenerationConfig(max_new_tokens=4,
                                                   force_unbatched=True))
    assert [r.text for r in chunked] == [r.text for r in single]
    assert [r.batch_size for r in chunked] == [2, 2, 2, 2, 2, 2]


# -- prefix embedding shapes --------------------------------------------
def test_prefix_embeds_accepts_shared_batched_and_per_row_forms():
    bundle = S.tiny_bundle()
    hidden = bundle.model.get_input_embeddings().weight.shape[1]
    shared = torch.zeros(3, hidden)
    assert [p.shape for p in G.normalize_prefix_embeds(shared, 2, hidden)] == \
        [shared.shape, shared.shape]
    batched = torch.zeros(2, 3, hidden)
    assert len(G.normalize_prefix_embeds(batched, 2, hidden)) == 2
    per_row = [torch.zeros(1, hidden), None]
    out = G.normalize_prefix_embeds(per_row, 2, hidden)
    assert out[1] is None and out[0].shape == (1, hidden)
    assert G.normalize_prefix_embeds(None, 3, hidden) == [None, None, None]


def test_prefix_embeds_shape_errors_are_loud():
    bundle = S.tiny_bundle()
    hidden = bundle.model.get_input_embeddings().weight.shape[1]
    with pytest.raises(G.GenerationError):
        G.normalize_prefix_embeds(torch.zeros(3, hidden + 1), 2, hidden)
    with pytest.raises(G.GenerationError):
        G.normalize_prefix_embeds(torch.zeros(5, 3, hidden), 2, hidden)
    with pytest.raises(G.GenerationError):
        G.normalize_prefix_embeds([torch.zeros(3, hidden)], 2, hidden)
    with pytest.raises(G.GenerationError):
        G.normalize_prefix_embeds(torch.zeros(2, 2, 2, hidden), 2, hidden)


def test_prefix_embeds_change_the_generation():
    bundle = S.tiny_bundle()
    hidden = bundle.model.get_input_embeddings().weight.shape[1]
    gen = torch.Generator().manual_seed(3)
    prefix = torch.empty(8, hidden).normal_(0.0, 1.0, generator=gen)
    plain = G.greedy_generate(bundle, MIXED_PROMPTS, max_new_tokens=5)
    with_prefix = G.greedy_generate(bundle, MIXED_PROMPTS, max_new_tokens=5,
                                    prefix_embeds=prefix)
    assert plain != with_prefix


def test_generation_records_serialize():
    bundle = S.tiny_bundle()
    rec = G.greedy_generate_detailed(bundle, MIXED_PROMPTS[:1], max_new_tokens=3)[0]
    payload = rec.to_dict()
    assert payload["schema"] == G.GENERATION_RECORD_SCHEMA
    assert payload["token_ids"] == list(rec.token_ids)
    assert len(payload["generated_text_sha256"]) == 64


if __name__ == "__main__":  # pragma: no cover - convenience runner
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
