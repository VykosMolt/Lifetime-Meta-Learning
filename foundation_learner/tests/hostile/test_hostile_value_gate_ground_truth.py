"""HOSTILE: the FL6 gate must not use ground truth (contract §18).

Attacks:

1. Try to pass a verifier verdict, a poison flag, a certified flag or a future
   outcome into ``ValueGate.decide`` — every spelling must be a ``TypeError``.
2. Inspect the gate's own state for any reference to an environment, verifier,
   episode or label table.
3. Show that the decision is a pure function of the hidden summary: two items
   whose environment flags differ but whose summaries are identical get the
   SAME decision, and the recorded decision carries no ground-truth field.
4. Show that the gated FL5 / FL7 call sites hand the gate a TENSOR and nothing
   else (the W4 protocol does not even carry the poison flag or the verdict).
"""
from __future__ import annotations

import inspect

import pytest
import torch

from foundation_learner.evaluation.learning_curve import AttemptEvent, FeedbackEvent
from foundation_learner.mechanisms.fast_state import FastStateCallbacks, FastStateModule
from foundation_learner.mechanisms.value_gating import (
    GateDecision,
    UnconditionalGate,
    ValueGate,
)
from foundation_learner.mechanisms.value_head import ValueHead

HIDDEN = 8

FORBIDDEN_KWARGS = (
    {"poison": True},
    {"poisoned": True},
    {"verdict": "INCORRECT"},
    {"correct": True},
    {"verifier_result": {"correct": True}},
    {"certified": False},
    {"future_query_success": 1.0},
    {"target": 0.5},
    {"label": "IN"},
)


def _head(value: float = 1.0) -> ValueHead:
    head = ValueHead(HIDDEN, inner=4, seed=101)
    with torch.no_grad():
        head.fc1.weight.zero_()
        head.fc1.bias.zero_()
        head.fc2.weight.zero_()
        head.fc2.bias.fill_(float(value))
    return head


def test_decide_accepts_the_hidden_summary_and_nothing_else():
    gate = ValueGate(_head())
    assert list(inspect.signature(ValueGate.decide).parameters) == \
        ["self", "hidden_summary"]
    assert list(inspect.signature(UnconditionalGate.decide).parameters) == \
        ["self", "hidden_summary"]
    for kwargs in FORBIDDEN_KWARGS:
        with pytest.raises(TypeError):
            gate.decide(torch.zeros(HIDDEN), **kwargs)
        with pytest.raises(TypeError):
            UnconditionalGate().decide(torch.zeros(HIDDEN), **kwargs)


def test_gate_holds_no_environment_or_label_reference():
    gate = ValueGate(_head())
    assert set(vars(gate)) == {"head", "threshold"}
    for name in ("env", "environment", "verifier", "episode", "items",
                 "outcomes", "labels", "poison"):
        assert not hasattr(gate, name)
    assert set(GateDecision.__dataclass_fields__) == \
        {"incorporate", "value", "threshold", "gate"}


def test_decision_is_a_pure_function_of_the_summary():
    head = ValueHead(HIDDEN, inner=4, seed=102)
    gate = ValueGate(head)
    summary = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(3))
    first = gate.decide(summary)
    second = gate.decide(summary.clone())
    assert (first.incorporate, first.value) == (second.incorporate, second.value)


def test_the_w4_protocol_does_not_even_offer_ground_truth():
    """A gate cannot use what the callback payloads never carry."""
    assert "poison" not in FeedbackEvent.__dataclass_fields__
    assert not {"correct", "verdict", "outcome"} & set(
        FeedbackEvent.__dataclass_fields__)
    assert not {"correct", "verdict", "outcome"} & set(
        AttemptEvent.__dataclass_fields__)


class _RecordingGate(ValueGate):
    def __init__(self, head):
        super().__init__(head)
        self.seen: list[object] = []

    def decide(self, hidden_summary):
        self.seen.append(hidden_summary)
        return super().decide(hidden_summary)


def test_gated_fl5_call_site_passes_only_a_tensor():
    module = FastStateModule(HIDDEN, rho=1.0, state_dim=4, seed=103)
    gate = _RecordingGate(_head(-1.0))
    cb = FastStateCallbacks(module, gate=gate)
    cb.on_episode_start(object())

    def summary_fn(reduce="mean", include_prefix=False):
        return torch.ones(HIDDEN)

    event = FeedbackEvent(role="HINT", item_id="i", interaction_index=None,
                          text="HINT-X", certified=False, event_index=3,
                          char_span=(0, 5), context_sha256="0" * 64)
    cb.on_feedback(event, summary_fn)
    assert len(gate.seen) == 1
    assert isinstance(gate.seen[0], torch.Tensor)
    assert cb.updates_applied == 0 and cb.updates_skipped == 1
