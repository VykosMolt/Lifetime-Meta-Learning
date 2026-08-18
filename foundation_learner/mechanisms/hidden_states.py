"""Final-layer, FINAL-LOOP hidden states of the frozen Ouro backbone (W3).

Every FL4-FL8 mechanism taps the SAME quantity the contract binds in §7:

    "input = final-layer, final-loop hidden state at the last token of
     feedback/hint item j"   (FL4)
    "u_j = W_in . mean(final-layer, final-loop hidden states over item-j
     tokens)"                (FL5)

How that tensor is obtained from ``OuroForCausalLM`` (recorded choice)
----------------------------------------------------------------------
``modeling_ouro.OuroModel.forward`` runs ``active_total_ut_steps`` recurrent
loops; after each loop it appends the POST-``self.norm`` hidden state to
``hidden_states_list``.  It returns
``(BaseModelOutputWithPast, hidden_states_list, gate_list)`` where
``last_hidden_state is hidden_states_list[-1]``.  ``OuroForCausalLM.forward``
forwards that list as ``OuroRLTTCausalLMOutput.per_loop_hidden_states`` when
``return_per_loop_hidden_states=True``.

This module therefore uses, as its PRIMARY path,

    out = model(..., return_per_loop_hidden_states=True, use_cache=False)
    hidden = out.per_loop_hidden_states[-1]          # [B, T, hidden]
    logits = out.logits                              # same forward

which is the only way to obtain the final-loop hidden states AND the logits
from ONE differentiable forward (FL5 training needs both).  The equality
``per_loop_hidden_states[-1] == model.model(...)[0].last_hidden_state`` — i.e.
this tap is identical to the one ``evaluation/learning_curve.py`` uses for its
``hidden_summary_fn`` — is pinned by ``tests/test_mechanisms_hidden_states.py``.

FALLBACK (documented, never silent): if a wrapper hides the RLTT output fields,
the inner ``model.model(...)`` call is used for the hidden states.  That path
cannot produce logits, so a caller that asked for logits gets a hard error
instead of a substituted quantity.

Nothing here uses the global RNG or the wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from foundation_learner.evaluation.scoring import (
    align_char_span_covering,
    encode_with_offsets,
)

__all__ = [
    "HiddenStateAccessError",
    "FINAL_LOOP_SOURCE_PRIMARY",
    "FINAL_LOOP_SOURCE_FALLBACK",
    "ForwardTap",
    "forward_final_loop",
    "item_token_range",
    "item_hidden_states",
    "item_hidden_summary",
    "REDUCTIONS",
]


class HiddenStateAccessError(RuntimeError):
    """The model does not expose the final-loop hidden states."""


FINAL_LOOP_SOURCE_PRIMARY = "OuroForCausalLM.per_loop_hidden_states[-1]"
FINAL_LOOP_SOURCE_FALLBACK = "OuroModel(...)[0].last_hidden_state"

#: reductions supported by :func:`item_hidden_summary`; identical spelling and
#: semantics to ``evaluation.learning_curve``'s ``hidden_summary_fn``.
REDUCTIONS = ("mean", "last", "tokens")


@dataclass(frozen=True)
class ForwardTap:
    """One forward pass: final-loop hidden states (+ logits when requested)."""

    hidden: torch.Tensor                 # [B, T, hidden]
    logits: torch.Tensor | None          # [B, T, vocab] or None
    source: str

    @property
    def hidden_size(self) -> int:
        return int(self.hidden.shape[-1])


def forward_final_loop(
    model: Any,
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    need_logits: bool = False,
) -> ForwardTap:
    """One forward returning the FINAL-LOOP, final-layer hidden states.

    Exactly one of ``input_ids`` / ``inputs_embeds`` must be given (the model
    itself enforces this too).  The returned tensors keep their autograd graph:
    callers that want a detached summary detach explicitly, so a mechanism can
    never lose gradient silently.
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise HiddenStateAccessError(
            "exactly one of input_ids / inputs_embeds must be supplied")
    kwargs: dict[str, Any] = {"use_cache": False}
    if input_ids is not None:
        kwargs["input_ids"] = input_ids
    else:
        kwargs["inputs_embeds"] = inputs_embeds
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids

    out = model(return_per_loop_hidden_states=True, **kwargs)
    per_loop = getattr(out, "per_loop_hidden_states", None)
    if per_loop:
        return ForwardTap(hidden=per_loop[-1],
                          logits=getattr(out, "logits", None),
                          source=FINAL_LOOP_SOURCE_PRIMARY)

    if need_logits:
        raise HiddenStateAccessError(
            "model did not return per_loop_hidden_states; the fallback tap "
            "cannot produce logits from the same forward, and substituting a "
            "different quantity is forbidden")
    inner = getattr(model, "model", None)
    if inner is None or not callable(inner):
        raise HiddenStateAccessError(
            "model exposes neither per_loop_hidden_states nor an inner "
            "OuroModel; the final-loop hidden state is unavailable")
    result = inner(**kwargs)
    base = result[0] if isinstance(result, tuple) else result
    hidden = getattr(base, "last_hidden_state", None)
    if hidden is None:
        raise HiddenStateAccessError(
            "inner model returned no last_hidden_state")
    return ForwardTap(hidden=hidden, logits=None,
                      source=FINAL_LOOP_SOURCE_FALLBACK)


def item_token_range(tokenizer: Any, text: str,
                     char_span: tuple[int, int]) -> tuple[list[int], int, int]:
    """``(token_ids, t0, t1)`` for the tokens INTERSECTING ``char_span``.

    Uses ``evaluation.scoring.align_char_span_covering`` — the covering rule
    W4 froze for hidden-state summaries (an event boundary may be straddled by
    one byte-level BPE token).  Log-probability estimands use the STRICT
    alignment instead; the two are never interchanged.
    """
    ids, offsets = encode_with_offsets(tokenizer, text)
    t0, t1 = align_char_span_covering(offsets, char_span)
    return ids, t0, t1


def item_hidden_states(
    bundle: Any,
    text: str,
    char_span: tuple[int, int],
    *,
    prefix_embeds: torch.Tensor | None = None,
    grad: bool = False,
) -> torch.Tensor:
    """``[n_item_tokens, hidden]`` final-loop states over ``char_span``.

    ``grad=False`` (default) runs under ``torch.inference_mode`` and returns a
    detached float32 tensor — the offline/eval path.  ``grad=True`` keeps the
    graph, which the FL5 training path needs.
    """
    model = getattr(bundle, "model")
    tokenizer = getattr(bundle, "tokenizer")
    device = getattr(bundle, "device", None)
    if device is None:
        device = next(model.parameters()).device
    ids, t0, t1 = item_token_range(tokenizer, text, char_span)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    shift = 0
    if prefix_embeds is None:
        kwargs = {"input_ids": input_ids}
        mask = torch.ones_like(input_ids)
    else:
        embed = model.get_input_embeddings()
        pfx = prefix_embeds.to(device=device, dtype=embed.weight.dtype)
        embeds = torch.cat([pfx.unsqueeze(0), embed(input_ids)], dim=1)
        kwargs = {"inputs_embeds": embeds}
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
        shift = int(pfx.shape[0])
    if grad:
        tap = forward_final_loop(model, attention_mask=mask, **kwargs)
        return tap.hidden[0, shift + t0: shift + t1]
    with torch.inference_mode():
        tap = forward_final_loop(model, attention_mask=mask, **kwargs)
    return tap.hidden[0, shift + t0: shift + t1].detach().float()


def item_hidden_summary(
    bundle: Any,
    text: str,
    char_span: tuple[int, int],
    *,
    reduce: str = "mean",
    prefix_embeds: torch.Tensor | None = None,
    grad: bool = False,
) -> torch.Tensor:
    """Reduce :func:`item_hidden_states` exactly as W4's ``hidden_summary_fn``."""
    if reduce not in REDUCTIONS:
        raise ValueError(f"unknown reduce {reduce!r}; frozen: {list(REDUCTIONS)}")
    states = item_hidden_states(bundle, text, char_span,
                                prefix_embeds=prefix_embeds, grad=grad)
    if reduce == "tokens":
        return states
    if reduce == "mean":
        return states.mean(dim=0)
    return states[-1]
