"""The shared final-loop hidden-state tap (W3).

Pins the recorded choice: ``OuroForCausalLM(return_per_loop_hidden_states=True)
.per_loop_hidden_states[-1]`` IS ``OuroModel(...)[0].last_hidden_state``, i.e.
the same tensor ``evaluation/learning_curve.py`` taps online.
"""
from __future__ import annotations

import pytest
import torch

from foundation_learner.evaluation.learning_curve import _final_loop_hidden_states
from foundation_learner.mechanisms.hidden_states import (
    FINAL_LOOP_SOURCE_FALLBACK,
    FINAL_LOOP_SOURCE_PRIMARY,
    HiddenStateAccessError,
    forward_final_loop,
    item_hidden_summary,
)
from foundation_learner.tests._w3_support import tiny_bundle


def _ids(bundle, text="TASK: 1 + 1\nATTEMPT: ANSWER: 2\n"):
    ids = bundle.tokenizer(text, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids], dtype=torch.long)


def test_primary_tap_equals_inner_model_last_hidden_state():
    bundle = tiny_bundle()
    ids = _ids(bundle)
    with torch.inference_mode():
        tap = forward_final_loop(bundle.model, input_ids=ids, need_logits=True)
        inner = bundle.model.model(input_ids=ids, use_cache=False)
    assert tap.source == FINAL_LOOP_SOURCE_PRIMARY
    assert torch.equal(tap.hidden, inner[0].last_hidden_state)
    assert tap.logits is not None
    assert tap.logits.shape[:2] == ids.shape


def test_primary_tap_equals_the_w4_walker_tap():
    """W3 and W4 must summarise the SAME tensor."""
    bundle = tiny_bundle()
    ids = _ids(bundle)
    mask = torch.ones_like(ids)
    w4 = _final_loop_hidden_states(bundle.model, {"input_ids": ids}, mask)
    with torch.inference_mode():
        w3 = forward_final_loop(bundle.model, input_ids=ids, attention_mask=mask)
    assert torch.equal(w3.hidden, w4)


def test_tap_is_differentiable():
    bundle = tiny_bundle()
    ids = _ids(bundle)
    embed = bundle.model.get_input_embeddings()
    embeds = embed(ids).clone().detach().requires_grad_(True)
    tap = forward_final_loop(bundle.model, inputs_embeds=embeds, need_logits=True)
    tap.hidden.sum().backward()
    assert embeds.grad is not None
    assert float(embeds.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("reduce,expected_dims", [("mean", 1), ("last", 1),
                                                  ("tokens", 2)])
def test_item_hidden_summary_shapes_and_determinism(reduce, expected_dims):
    bundle = tiny_bundle()
    text = "TASK: 1 + 1\nFEEDBACK: CORRECT\n"
    span = (text.index("FEEDBACK:"), len(text))
    a = item_hidden_summary(bundle, text, span, reduce=reduce)
    b = item_hidden_summary(bundle, text, span, reduce=reduce)
    assert a.dim() == expected_dims
    assert a.shape[-1] == bundle.model.config.hidden_size
    assert a.dtype is torch.float32
    assert torch.equal(a, b)


def test_mean_and_last_agree_with_the_token_reduction():
    bundle = tiny_bundle()
    text = "TASK: 1 + 1\nFEEDBACK: CORRECT\n"
    span = (text.index("FEEDBACK:"), len(text))
    tokens = item_hidden_summary(bundle, text, span, reduce="tokens")
    assert torch.allclose(item_hidden_summary(bundle, text, span, reduce="mean"),
                          tokens.mean(dim=0))
    assert torch.equal(item_hidden_summary(bundle, text, span, reduce="last"),
                       tokens[-1])


def test_missing_tap_is_a_hard_error_not_a_substitution():
    class NoTap(torch.nn.Module):
        def forward(self, **kwargs):
            return type("Out", (), {"logits": None,
                                    "per_loop_hidden_states": None})()

    with pytest.raises(HiddenStateAccessError):
        forward_final_loop(NoTap(), input_ids=torch.zeros(1, 3, dtype=torch.long))


def test_fallback_refuses_to_invent_logits():
    class InnerOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = self._Inner()

        class _Inner(torch.nn.Module):
            def forward(self, **kwargs):
                hidden = torch.zeros(1, 3, 8)
                return (type("Base", (), {"last_hidden_state": hidden})(), [], [])

        def forward(self, **kwargs):
            return type("Out", (), {"logits": None,
                                    "per_loop_hidden_states": None})()

    model = InnerOnly()
    ids = torch.zeros(1, 3, dtype=torch.long)
    tap = forward_final_loop(model, input_ids=ids)
    assert tap.source == FINAL_LOOP_SOURCE_FALLBACK
    with pytest.raises(HiddenStateAccessError):
        forward_final_loop(model, input_ids=ids, need_logits=True)


def test_exactly_one_input_is_required():
    bundle = tiny_bundle()
    with pytest.raises(HiddenStateAccessError):
        forward_final_loop(bundle.model)
