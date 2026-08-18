"""FL7 bounded resettable fast adapter (W3, contract §7)."""
from __future__ import annotations

import pytest
import torch

from foundation_learner.episodes.render import render_reset_query
from foundation_learner.episodes.schema import Role
from foundation_learner.evaluation.scoring import answer_logprob, encode_with_offsets
from foundation_learner.mechanisms.value_head import token_aligned_answer_span
from foundation_learner.mechanisms.fast_adapter import (
    FastAdapter,
    FastAdapterCallbacks,
    FastAdapterError,
    revealed_support_items,
    supervised_example,
)
from foundation_learner.mechanisms.value_gating import GateDecision
from foundation_learner.training.lora import lora_state_dict
from foundation_learner.training.peft_modes import (
    FL7_FAST_ADAPTER_SPEC,
    FL7_FAST_DELTA_MAX_FROBENIUS,
    FL7_INNER_LR,
    FL7_INNER_STEPS,
)
from foundation_learner.tests._w3_support import fresh_tiny_bundle, real_episode


def _adapter(seed: int = 31, **kwargs) -> FastAdapter:
    return FastAdapter(fresh_tiny_bundle(), seed=seed, **kwargs)


def test_spec_is_w2s_frozen_fl7_spec():
    adapter = _adapter()
    assert adapter.spec.rank == FL7_FAST_ADAPTER_SPEC.rank == 8
    assert adapter.spec.alpha == FL7_FAST_ADAPTER_SPEC.alpha == 16.0
    assert adapter.spec.dropout == 0.0
    assert set(adapter.spec.target_suffixes) == {"q_proj", "v_proj"}
    assert adapter.inner_steps == FL7_INNER_STEPS == 4
    assert adapter.inner_lr == FL7_INNER_LR == 1e-3
    assert adapter.max_frobenius == FL7_FAST_DELTA_MAX_FROBENIUS == 1.0
    assert all(name.endswith(("q_proj", "v_proj"))
               for name in adapter.handles.names())


def test_inner_loop_reduces_the_revealed_answer_nll():
    adapter = _adapter()
    episode = real_episode(reveal=True)
    item_id = next(iter(revealed_support_items(episode)))
    record = adapter.adapt_item(episode, item_id)
    assert record.applied and record.steps == FL7_INNER_STEPS
    assert len(record.losses) == FL7_INNER_STEPS
    assert record.losses[-1] < record.losses[0]
    assert all(b <= a for a, b in zip(record.losses, record.losses[1:]))


def test_delta_starts_at_zero_and_resets_to_zero():
    adapter = _adapter()
    episode = real_episode(reveal=True)
    assert adapter.delta_norm() == 0.0
    adapter.adapt_episode(episode)
    assert adapter.delta_norm() > 0.0
    adapter.reset_episode()
    assert adapter.delta_norm() == 0.0
    assert adapter.episodes_seen == 1


def test_frobenius_bound_is_enforced_after_every_item():
    adapter = _adapter(inner_lr=5.0, max_frobenius=1e-3)
    episode = real_episode(reveal=True)
    records = adapter.adapt_episode(episode)
    assert any(r.pre_clip_norm > 1e-3 for r in records), "clip never bit"
    for record in records:
        assert record.post_clip_norm <= 1e-3 + 1e-9
    assert adapter.delta_norm() <= 1e-3 + 1e-9


def test_base_parameters_are_bitwise_unchanged():
    adapter = _adapter()
    episode = real_episode(reveal=True)
    before = adapter.base_fingerprint()
    adapter.adapt_episode(episode)
    assert adapter.base_fingerprint() == before
    adapter.reset_episode()
    assert adapter.base_fingerprint() == before


def test_supervised_example_uses_the_revealed_answer_only():
    adapter = _adapter()
    episode = real_episode(reveal=True)
    item_id, reveal_index = next(iter(revealed_support_items(episode).items()))
    example = supervised_example(episode, item_id, adapter.tokenizer)
    revealed = episode.events[reveal_index].text.strip()
    assert f"ANSWER: {revealed}" in example["text"]
    # exactly ONE loss-bearing span, and it is the revealed answer line
    assert len(example["spans_meta"]) == 1
    span = example["spans_meta"][0]
    assert example["text"][span["char_start"]:span["char_end"]] == \
        f"ANSWER: {revealed}"
    assert example["n_loss_tokens"] > 0
    assert sum(example["loss_mask"]) > 0
    # no feedback, hint or query material may appear in the inner-loop example
    for event in episode.events:
        if event.role in (Role.QUERY_TASK, Role.TRANSFER_TASK):
            assert event.text not in example["text"]


def test_query_and_unrevealed_items_cannot_enter_the_inner_loop():
    adapter = _adapter()
    episode = real_episode(reveal=True)
    query_item = next(e.item_id for e in episode.events
                      if e.role is Role.QUERY_TASK)
    with pytest.raises(FastAdapterError):
        supervised_example(episode, query_item, adapter.tokenizer)
    with pytest.raises(FastAdapterError):
        adapter.adapt_item(episode, query_item)
    no_reveal = real_episode(reveal=False)
    support = next(iter(k for k, v in no_reveal.items.items()
                        if v["role"] == "support"))
    with pytest.raises(FastAdapterError):
        supervised_example(no_reveal, support, adapter.tokenizer)


def test_checkpoint_is_deterministic():
    a = _adapter(seed=41)
    b = _adapter(seed=41)
    episode = real_episode(reveal=True)
    a.adapt_episode(episode)
    b.adapt_episode(episode)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa["config_hash"] == sb["config_hash"]
    assert set(sa["lora"]) == set(sb["lora"])
    for key in sa["lora"]:
        assert torch.equal(sa["lora"][key], sb["lora"][key]), key
    assert _adapter(seed=42).config_hash() != sa["config_hash"]


def test_delta_survives_a_context_reset_inside_the_episode():
    adapter = _adapter()
    episode = real_episode(reveal=True)
    adapter.adapt_episode(episode)
    before = {k: v.clone() for k, v in lora_state_dict(adapter.handles).items()}
    query_index = next(i for i, e in enumerate(episode.events)
                       if e.role is Role.QUERY_TASK)
    rendered = render_reset_query(episode, query_index)
    text = rendered.text + "ATTEMPT: ANSWER: 1\n"
    line = (text.rindex("ANSWER:"), len(text.rstrip()))
    _ids, offsets = encode_with_offsets(adapter.tokenizer, text)
    value = answer_logprob(adapter.bundle, text,
                           token_aligned_answer_span(offsets, text, line))
    assert value == value  # finite, i.e. the adapted model really ran
    after = lora_state_dict(adapter.handles)
    for key in before:
        assert torch.equal(before[key], after[key]), key


class _StubGate:
    def __init__(self, incorporate: bool) -> None:
        self.incorporate = incorporate

    def decide(self, hidden_summary):
        assert isinstance(hidden_summary, torch.Tensor)
        return GateDecision(incorporate=self.incorporate, value=0.1,
                            threshold=0.0, gate="stub")


def test_gated_variant_skips_declined_items():
    adapter = _adapter(gate=_StubGate(False))
    episode = real_episode(reveal=True)
    item_id = next(iter(revealed_support_items(episode)))
    record = adapter.adapt_item(episode, item_id,
                               hidden_summary=torch.zeros(64))
    assert record.applied is False and record.steps == 0
    assert adapter.delta_norm() == 0.0
    with pytest.raises(FastAdapterError):
        adapter.adapt_item(episode, item_id)          # gate needs a summary
    with pytest.raises(FastAdapterError):
        adapter.adapt_episode(episode)                # ungated entry point


def test_callbacks_only_adapt_on_reveal_events():
    adapter = _adapter()
    episode = real_episode(reveal=True)
    cb = FastAdapterCallbacks(adapter, episode)
    cb.on_episode_start(object())

    class _Event:
        def __init__(self, role, item_id):
            self.role = role
            self.item_id = item_id

    def summary_fn(reduce="mean", include_prefix=False):
        return torch.zeros(64)

    item_id, reveal_index = next(iter(revealed_support_items(episode).items()))
    cb.on_feedback(_Event("FEEDBACK", item_id), summary_fn)
    assert adapter.delta_norm() == 0.0
    cb.on_feedback(_Event("HINT", item_id), summary_fn)
    assert adapter.delta_norm() == 0.0
    cb.on_feedback(_Event("REVEAL", item_id), summary_fn)
    assert adapter.delta_norm() > 0.0
    assert cb.prefix_provider() is None
    assert len(cb.summary()["records"]) == 1
