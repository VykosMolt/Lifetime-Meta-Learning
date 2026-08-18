"""FL6 value gate (W3, contract §7): threshold behaviour and input isolation."""
from __future__ import annotations

import inspect

import pytest
import torch

from foundation_learner.evaluation import canonical_json
from foundation_learner.mechanisms.value_gating import (
    FL6_GATE_THRESHOLD,
    GateDecision,
    GateError,
    UnconditionalGate,
    ValueGate,
)
from foundation_learner.mechanisms.value_head import ValueHead

HIDDEN = 8


def _constant_head(value: float) -> ValueHead:
    """A head whose prediction is EXACTLY ``value`` for any input."""
    head = ValueHead(HIDDEN, inner=4, seed=1)
    with torch.no_grad():
        head.fc1.weight.zero_()
        head.fc1.bias.zero_()
        head.fc2.weight.zero_()
        head.fc2.bias.fill_(float(value))
    return head


def test_threshold_is_frozen_at_zero():
    assert FL6_GATE_THRESHOLD == 0.0


@pytest.mark.parametrize("value,expected", [
    (1e-3, True),      # +eps
    (-1e-3, False),    # -eps
    (0.0, False),      # strict ">" (contract: incorporate iff v > 0)
])
def test_gate_threshold_behaviour_at_plus_minus_epsilon(value, expected):
    gate = ValueGate(_constant_head(value))
    decision = gate.decide(torch.zeros(HIDDEN))
    assert decision.incorporate is expected
    assert decision.value == pytest.approx(value, abs=1e-6)
    assert decision.threshold == 0.0


def test_gate_consumes_only_the_hidden_summary():
    params = list(inspect.signature(ValueGate.decide).parameters)
    assert params == ["self", "hidden_summary"], params
    gate = ValueGate(_constant_head(1.0))
    for kwargs in ({"poison": True}, {"verdict": "CORRECT"},
                   {"correct": False}, {"certified": True},
                   {"future_outcome": 1.0}):
        with pytest.raises(TypeError):
            gate.decide(torch.zeros(HIDDEN), **kwargs)
    assert set(vars(gate)) == {"head", "threshold"}


def test_gate_freezes_the_head_and_never_trains_it():
    head = _constant_head(0.5)
    assert any(p.requires_grad for p in head.parameters())
    gate = ValueGate(head)
    assert not any(p.requires_grad for p in gate.head.parameters())
    before = [p.detach().clone() for p in gate.head.parameters()]
    summary = torch.ones(HIDDEN, requires_grad=True)
    decision = gate.decide(summary)
    assert decision.incorporate is True
    assert summary.grad is None
    for p, b in zip(gate.head.parameters(), before):
        assert torch.equal(p, b)


def test_gate_requires_a_real_value_head():
    with pytest.raises(GateError):
        ValueGate(torch.nn.Linear(4, 1))


def test_unconditional_arm_always_incorporates():
    gate = UnconditionalGate()
    decision = gate.decide(torch.randn(HIDDEN))
    assert decision.incorporate is True
    assert decision.value is None
    assert decision.gate == "unconditional"


def test_decisions_are_canonical_json_safe():
    for gate in (ValueGate(_constant_head(1.0)), UnconditionalGate()):
        payload = gate.decide(torch.zeros(HIDDEN)).to_dict()
        assert canonical_json(payload)
        assert canonical_json(gate.config())


def test_gate_decision_carries_no_ground_truth_field():
    fields = set(GateDecision.__dataclass_fields__)
    assert fields == {"incorporate", "value", "threshold", "gate"}
