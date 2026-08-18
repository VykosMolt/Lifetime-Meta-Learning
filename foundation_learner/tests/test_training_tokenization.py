"""Loss-mask exactness (contract §7, §23).

The masks are checked against HAND-COMPUTED expectations on a constructed
episode: the exact character span of each ``ANSWER:`` line is recomputed
independently here (by string search on the rendered text), converted to token
indices through the tokenizer's offsets, and compared element-wise with the
mask produced by ``training/tokenization.py``.
"""

from __future__ import annotations

import os

import pytest

from foundation_learner.training import model_loading as ml
from foundation_learner.training import tokenization as tk
from foundation_learner.training.tiny_model import _cached_tokenizer

from foundation_learner.tests.test_training_stubs import (
    event_role_name,
    make_ordered_episode,
    make_static_episode,
    render_episode,
    replace_event,
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen tokenizer not present",
)


@pytest.fixture(scope="module")
def tokenizer():
    return _cached_tokenizer(ml.FROZEN_CHECKPOINT_DIR)


def _expected_mask(text, tokenizer, wanted):
    """Independent mask computation from ``(answer line text, weight)`` pairs.

    Implements the documented rule directly: tokens fully inside the answer
    line, plus any straddling token whose characters outside the line are pure
    whitespace (``expand_whitespace``).  Anything else fails the test here,
    which is exactly the condition the implementation raises on.
    """
    _, offsets = tk.encode_with_offsets(tokenizer, text)
    mask = [0.0] * len(offsets)
    for line, weight in wanted:
        # the line must be unique for the hand-computed check to be meaningful
        assert text.count(line) == 1, line
        start = text.index(line)
        end = start + len(line)
        hits = []
        for i, (a, b) in enumerate(offsets):
            if b <= a or a >= end or b <= start:
                continue
            if a >= start and b <= end:
                hits.append(i)
                continue
            outside = text[a : min(b, start)] + text[max(a, end) : b]
            assert not outside.strip(), (line, text[a:b])
            hits.append(i)
        assert hits, line
        for i in hits:
            mask[i] = weight
    return mask


def test_fl3_mask_is_exactly_the_frozen_weight_table(tokenizer):
    ep = make_ordered_episode(0)
    ex = tk.episode_to_training_example(ep, tokenizer, "FL3", render_fn=render_episode)
    text = ex["text"]
    expected = _expected_mask(
        text,
        tokenizer,
        [
            ("ANSWER: 9", 0.0),  # attempt0 on P1  (index 0)
            ("ANSWER: 1", 0.25),  # revision on P1  (index 1)
            ("ANSWER: 7", 0.0),  # attempt0 on P2  (index 2)
            ("ANSWER: 2", 0.25),  # revision on P2  (index 3)
            ("ANSWER: 3", 0.0),  # related attempt (index 4)
            ("ANSWER: 4", 1.0),  # QUERY           (index 5)
            ("ANSWER: 5", 1.0),  # TRANSFER        (index 6)
        ],
    )
    assert ex["loss_mask"] == expected
    assert ex["total_weight"] == pytest.approx(
        sum(expected)
    )
    # exact per-span accounting
    weights = {s["interaction_index"]: s["weight"] for s in ex["spans_meta"]}
    assert weights == dict(tk.FL3_INTERACTION_WEIGHTS)


def test_fl3_weights_are_the_frozen_table():
    assert tk.FL3_INTERACTION_WEIGHTS == {
        0: 0.0,
        1: 0.25,
        2: 0.0,
        3: 0.25,
        4: 0.0,
        5: 1.0,
        6: 1.0,
    }


def test_fl2_weights_every_model_attempt_answer_equally(tokenizer):
    ep = make_ordered_episode(0)
    ex = tk.episode_to_training_example(ep, tokenizer, "FL2", render_fn=render_episode)
    text = ex["text"]
    expected = _expected_mask(
        text,
        tokenizer,
        [(f"ANSWER: {v}", 1.0) for v in (9, 1, 7, 2, 3, 4, 5)],
    )
    assert ex["loss_mask"] == expected
    assert {s["weight"] for s in ex["spans_meta"]} == {1.0}
    assert ex["n_loss_tokens"] == sum(1 for w in expected if w > 0)


def test_fl2_and_fl3_share_the_same_token_stream(tokenizer):
    ep = make_ordered_episode(2)
    fl2 = tk.episode_to_training_example(ep, tokenizer, "FL2", render_fn=render_episode)
    fl3 = tk.episode_to_training_example(ep, tokenizer, "FL3", render_fn=render_episode)
    assert fl2["input_ids"] == fl3["input_ids"]
    assert fl2["text"] == fl3["text"]
    assert fl2["loss_mask"] != fl3["loss_mask"]


def test_context_tokens_are_never_weighted(tokenizer):
    ep = make_ordered_episode(0)
    for arm in ("FL2", "FL3"):
        ex = tk.episode_to_training_example(ep, tokenizer, arm, render_fn=render_episode)
        _, offsets = tk.encode_with_offsets(tokenizer, ex["text"])
        answer_ranges = [
            (s["char_start"], s["char_end"]) for s in ex["spans_meta"]
        ]
        for idx, weight in enumerate(ex["loss_mask"]):
            if weight <= 0:
                continue
            a, b = offsets[idx]
            # every NON-WHITESPACE character of a weighted token must lie
            # inside one declared answer span
            for pos in range(a, b):
                if ex["text"][pos].isspace():
                    continue
                assert any(
                    start <= pos < end for start, end in answer_ranges
                ), (idx, ex["text"][a:b], ex["text"][pos])
        # every FEEDBACK / HINT / TASK character has zero weight
        for marker in ("FEEDBACK:", "HINT:", "TASK:", "INCORRECT", "residue"):
            pos = ex["text"].find(marker)
            if pos < 0:
                continue
            for idx, (a, b) in enumerate(offsets):
                if a >= pos and b <= pos + len(marker):
                    assert ex["loss_mask"][idx] == 0.0


def test_fl1_produces_one_isolated_pair_per_item(tokenizer):
    ep = make_static_episode(0)
    examples = tk.episode_to_training_examples(
        ep, tokenizer, "FL1", render_fn=render_episode
    )
    assert len(examples) == 5  # TASK P1, TASK P2, RELATED, QUERY, TRANSFER
    for ex in examples:
        assert ex["arm"] == "FL1"
        assert len(ex["spans_meta"]) == 1
        assert ex["spans_meta"][0]["weight"] == 1.0
        assert "FEEDBACK:" not in ex["text"] and "HINT:" not in ex["text"]
        span = ex["spans_meta"][0]
        span_text = ex["text"][span["char_start"] : span["char_end"]]
        assert span_text.startswith("ANSWER: ")
        assert ex["text"].count(span_text) == 1
    # FL1 always trains on the CANONICAL answer, never on the scripted error
    assert "ANSWER: 1" in examples[0]["text"]
    assert "ANSWER: 9" not in examples[0]["text"]


def test_fl1_single_example_interface_selects_by_item_index(tokenizer):
    ep = make_static_episode(0)
    first = tk.episode_to_training_example(
        ep, tokenizer, "FL1", render_fn=render_episode
    )
    third = tk.episode_to_training_example(
        ep, tokenizer, "FL1", item_index=2, render_fn=render_episode
    )
    assert first["item_index"] == 0
    assert third["item_index"] == 2
    assert first["input_ids"] != third["input_ids"]
    with pytest.raises(tk.TokenizationError):
        tk.episode_to_training_example(
            ep, tokenizer, "FL1", item_index=99, render_fn=render_episode
        )


def test_fl1_refuses_history_bearing_episodes(tokenizer):
    ep = make_ordered_episode(0)
    with pytest.raises(tk.StaticArmHistoryError):
        tk.episode_to_training_examples(ep, tokenizer, "FL1", render_fn=render_episode)


def test_fl1_refuses_items_without_a_canonical_answer(tokenizer):
    import dataclasses

    ep = make_static_episode(0)
    ep = dataclasses.replace(ep, items={})
    for index, event in enumerate(list(ep.events)):
        if "meta" in {f.name for f in dataclasses.fields(type(event))}:
            ep = replace_event(ep, index, meta={})
    with pytest.raises(tk.MissingCanonicalAnswerError):
        tk.episode_to_training_examples(ep, tokenizer, "FL1", render_fn=render_episode)


def test_isolated_pair_helper_masks_only_the_answer_line(tokenizer):
    ex = tk.isolated_pair_to_training_example("compute f(3)", "7", tokenizer)
    assert ex["text"] == "TASK: compute f(3)\nANSWER: 7\n"
    expected = _expected_mask(ex["text"], tokenizer, [("ANSWER: 7", 1.0)])
    assert ex["loss_mask"] == expected
    assert ex["loss_mask"][0] == 0.0


def test_answer_span_covers_the_full_line_including_the_marker(tokenizer):
    ex = tk.isolated_pair_to_training_example("compute f(3)", "YES", tokenizer)
    span = ex["spans_meta"][0]
    assert ex["text"][span["char_start"] : span["char_end"]] == "ANSWER: YES"
    _, offsets = tk.encode_with_offsets(tokenizer, ex["text"])
    covered = "".join(
        ex["text"][offsets[i][0] : offsets[i][1]]
        for i, w in enumerate(ex["loss_mask"])
        if w > 0
    )
    assert covered == "ANSWER: YES"


def test_answer_grammar_matches_the_shared_verifier_grammar():
    parse = pytest.importorskip("foundation_learner.episodes.parse")
    assert tk.ANSWER_LINE_RE.pattern == parse.ANSWER_LINE_RE.pattern


def test_answer_line_detection_follows_the_strict_grammar():
    text = "TASK:\nanswer: 5\n ANSWER: 6\nANSWER: 7\nANSWER:\n"
    spans = tk.find_answer_line_spans(text, 0, len(text))
    assert [text[a:b] for a, b in spans] == ["ANSWER: 7"]


def test_multiple_answer_lines_in_one_event_are_all_weighted(tokenizer):
    ep = make_ordered_episode(0)
    for index, event in enumerate(list(ep.events)):
        if event_role_name(event) == "MODEL_ATTEMPT" and event.interaction_index == 5:
            ep = replace_event(ep, index, text="ANSWER: 4\nANSWER: 44\nANSWER: 444")
    ex = tk.episode_to_training_example(ep, tokenizer, "FL3", render_fn=render_episode)
    query_spans = [s for s in ex["spans_meta"] if s["interaction_index"] == 5]
    assert len(query_spans) == 3
    assert all(s["weight"] == 1.0 for s in query_spans)


def test_straddling_span_is_a_hard_error_not_a_silent_approximation(tokenizer):
    _, offsets = tk.encode_with_offsets(tokenizer, "TASK: hello world\n")
    # A span that starts in the middle of the 'Ġworld' token.
    start = offsets[-2][0] + 1
    end = offsets[-2][1]
    with pytest.raises(tk.SpanAlignmentError):
        tk.char_span_to_token_span(offsets, start, end)
    first, last = tk.char_span_to_token_span(offsets, start, end, on_straddle="expand")
    assert last - first == 1


def test_sequence_budget_is_enforced_without_truncation(tokenizer):
    ep = make_ordered_episode(0)
    with pytest.raises(tk.SequenceTooLongError):
        tk.episode_to_training_example(
            ep, tokenizer, "FL3", render_fn=render_episode, max_seq_len=16
        )


def test_unknown_arm_and_unknown_interaction_index_are_refused(tokenizer):
    ep = make_ordered_episode(0)
    with pytest.raises(tk.TokenizationError):
        tk.episode_to_training_example(ep, tokenizer, "FL9", render_fn=render_episode)
    for index, event in enumerate(list(ep.events)):
        if event_role_name(event) == "MODEL_ATTEMPT" and event.interaction_index == 6:
            ep = replace_event(ep, index, interaction_index=7)
    with pytest.raises(tk.TokenizationError):
        tk.episode_to_training_example(ep, tokenizer, "FL3", render_fn=render_episode)


def test_batching_pads_and_zeroes_the_padded_weights(tokenizer):
    ep_a = make_ordered_episode(0)
    ep_b = make_ordered_episode(1)
    examples = [
        tk.episode_to_training_example(e, tokenizer, "FL3", render_fn=render_episode)
        for e in (ep_a, ep_b)
    ]
    examples[1]["input_ids"] = examples[1]["input_ids"][:-3]
    examples[1]["loss_mask"] = examples[1]["loss_mask"][:-3]
    batch = tk.examples_to_batch(examples, pad_token_id=0)
    assert batch["input_ids"].shape == batch["loss_mask"].shape
    assert batch["n_padded_tokens"] == 2 * batch["input_ids"].shape[1]
    padded = batch["attention_mask"] == 0
    assert float(batch["loss_mask"][padded].abs().sum()) == 0.0
