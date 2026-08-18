"""FL5 fast-state module + eval-side callbacks (W3, contract §7/§23)."""
from __future__ import annotations

import pytest
import torch

from foundation_learner.mechanisms.fast_state import (
    ARM_FAST_STATE_ON_S0,
    FL5_PREFIX_K,
    FL5_STATE_DIM,
    FastStateCallbacks,
    FastStateError,
    FastStateModule,
    FastStateOffCallbacks,
    FastStateS0Callbacks,
    rho_from_embeddings,
)
from foundation_learner.mechanisms.value_gating import GateDecision
from foundation_learner.tests._w3_support import embedding_matrix, tiny_bundle

HIDDEN = 64


def _module(state_dim: int = 32, seed: int = 11, rho: float | None = None,
            hidden: int = HIDDEN) -> FastStateModule:
    if rho is not None:
        return FastStateModule(hidden, rho=rho, state_dim=state_dim, seed=seed)
    return FastStateModule(hidden, embedding_matrix=embedding_matrix(tiny_bundle()),
                           state_dim=state_dim, seed=seed)


def test_contract_defaults_and_shapes():
    module = FastStateModule(2048, rho=1.0)
    assert module.state_dim == FL5_STATE_DIM == 1024
    assert module.prefix_k == FL5_PREFIX_K == 8
    assert module.W_in.weight.shape == (1024, 2048)
    assert module.W_p.weight.shape == (8 * 2048, 1024)
    assert module.W_in.bias is None and module.W_p.bias is None
    assert isinstance(module.gru, torch.nn.GRUCell)
    assert isinstance(module.ln, torch.nn.LayerNorm)
    # contract §7's ~25 M estimate (biases excluded there)
    assert 24e6 < module.num_parameters() < 27e6


def test_reset_update_prefix_shapes_and_dtypes():
    module = _module()
    s0 = module.reset_state(3)
    assert s0.shape == (3, 32)
    assert torch.count_nonzero(s0) == 0
    h = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(1))
    s1 = module.update(s0, h)
    assert s1.shape == (3, 32) and s1.dtype is torch.float32
    prefix = module.prefix_embeds(s1)
    assert prefix.shape == (3, 8, HIDDEN)
    assert module.prefix_embeds(s1[0]).shape == (8, HIDDEN)


def test_rho_is_two_times_the_median_row_norm():
    matrix = torch.tensor([[3.0, 4.0], [0.0, 1.0], [6.0, 8.0]])
    # row norms 5, 1, 10 -> median 5 -> rho 10
    assert rho_from_embeddings(matrix) == pytest.approx(10.0)
    module = FastStateModule(2, embedding_matrix=matrix, state_dim=4)
    assert float(module.rho.item()) == pytest.approx(10.0)
    assert module.rho_source == "2x_median_input_embedding_row_norm"
    assert module.config()["rho"] == pytest.approx(10.0)


def test_rho_must_be_supplied_exactly_once():
    with pytest.raises(FastStateError):
        FastStateModule(8, state_dim=4)
    with pytest.raises(FastStateError):
        FastStateModule(8, state_dim=4, rho=1.0,
                        embedding_matrix=torch.ones(3, 8))


def test_prefix_norm_clamp_is_exact_at_the_boundary():
    module = _module(rho=0.25)
    with torch.no_grad():
        module.W_p.weight.mul_(1000.0)  # force every prefix vector over rho
    s = torch.randn(32, generator=torch.Generator().manual_seed(3))
    prefix = module.prefix_embeds(s)
    norms = torch.linalg.vector_norm(prefix, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 0.25), atol=1e-5)


def test_prefix_below_rho_is_left_untouched():
    module = _module(rho=1e6)
    s = torch.randn(32, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        raw = module.W_p(module.ln(s.unsqueeze(0))).view(8, HIDDEN)
    assert torch.allclose(module.prefix_embeds(s), raw, atol=1e-6)


def test_gradients_flow_through_state_and_prefix():
    module = _module()
    s = module.reset_state(1)
    h = torch.randn(1, HIDDEN, generator=torch.Generator().manual_seed(5))
    s1 = module.update(s, h)
    s2 = module.update(s1, h * 0.5)
    module.prefix_embeds(s2).sum().backward()
    for name, param in module.named_parameters():
        assert param.grad is not None, name
    assert float(module.W_in.weight.grad.abs().sum()) > 0.0
    assert float(module.gru.weight_hh.grad.abs().sum()) > 0.0
    assert float(module.W_p.weight.grad.abs().sum()) > 0.0


def test_detach_controls_are_explicit():
    module = _module()
    h = torch.randn(1, HIDDEN, generator=torch.Generator().manual_seed(6))
    base = module.reset_state(1).clone().requires_grad_(True)
    module.update(base, h).sum().backward()
    assert base.grad is not None and float(base.grad.abs().sum()) > 0.0

    base2 = module.reset_state(1).clone().requires_grad_(True)
    module.update(base2, h, detach_state=True).sum().backward()
    assert base2.grad is None or float(base2.grad.abs().sum()) == 0.0

    item = h.clone().requires_grad_(True)
    module.update(module.reset_state(1), item, detach_input=True).sum().backward()
    assert item.grad is None or float(item.grad.abs().sum()) == 0.0


def test_initialisation_is_deterministic_and_seed_dependent():
    a = _module(seed=7, rho=1.0)
    b = _module(seed=7, rho=1.0)
    c = _module(seed=8, rho=1.0)
    for (name, pa), (_n, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), name
    assert not torch.equal(a.W_in.weight, c.W_in.weight)
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()


def test_malformed_inputs_are_refused():
    module = _module()
    with pytest.raises(FastStateError):
        module.update(torch.zeros(1, 5), torch.zeros(1, HIDDEN))
    with pytest.raises(FastStateError):
        module.update(module.reset_state(1), torch.zeros(1, 5))
    with pytest.raises(FastStateError):
        module.reset_state(0)


# --------------------------------------------------------------------------
# eval-side callbacks
# --------------------------------------------------------------------------
class _Ctx:
    episode_id = "ep"


def _summary_fn(value: float = 1.0, hidden: int = HIDDEN):
    def fn(reduce="mean", include_prefix=False):
        assert include_prefix is False, "the FL5 update must not re-enter itself"
        return torch.full((hidden,), float(value))
    return fn


def test_callbacks_update_and_prefix():
    module = _module()
    cb = FastStateCallbacks(module)
    cb.on_episode_start(_Ctx())
    assert torch.count_nonzero(cb.s) == 0
    prefix = cb.prefix_provider()
    assert prefix.shape == (8, HIDDEN) and not prefix.requires_grad
    cb.on_feedback(object(), _summary_fn())
    assert cb.updates_applied == 1
    assert float(torch.linalg.vector_norm(cb.s)) > 0.0
    assert cb.summary()["updates_applied"] == 1


def test_callbacks_reset_between_episodes_by_default():
    module = _module()
    cb = FastStateCallbacks(module)
    cb.on_episode_start(_Ctx())
    cb.on_feedback(object(), _summary_fn())
    assert float(torch.linalg.vector_norm(cb.s)) > 0.0
    cb.on_episode_start(_Ctx())
    assert torch.count_nonzero(cb.s) == 0


def test_chain_carry_is_opt_in_only():
    module = _module()
    cb = FastStateCallbacks(module, carry_across_episodes=True)
    cb.on_episode_start(_Ctx())
    cb.on_feedback(object(), _summary_fn())
    carried = cb.s.clone()
    cb.on_episode_start(_Ctx())
    assert torch.equal(cb.s, carried)


class _StubGate:
    def __init__(self, incorporate: bool):
        self.incorporate = incorporate
        self.seen = 0

    def decide(self, hidden_summary):
        self.seen += 1
        return GateDecision(incorporate=self.incorporate, value=0.5,
                            threshold=0.0, gate="stub")


def test_gate_can_veto_a_state_update():
    module = _module()
    gate = _StubGate(False)
    cb = FastStateCallbacks(module, gate=gate)
    cb.on_episode_start(_Ctx())
    cb.on_feedback(object(), _summary_fn())
    assert gate.seen == 1
    assert cb.updates_applied == 0 and cb.updates_skipped == 1
    assert torch.count_nonzero(cb.s) == 0

    cb2 = FastStateCallbacks(module, gate=_StubGate(True))
    cb2.on_episode_start(_Ctx())
    cb2.on_feedback(object(), _summary_fn())
    assert cb2.updates_applied == 1


def test_off_arm_callbacks_inject_nothing():
    cb = FastStateOffCallbacks()
    cb.on_episode_start(_Ctx())
    assert cb.prefix_provider() is None
    cb.on_feedback(object(), _summary_fn())
    assert cb.summary()["injection"] == "DISABLED"


# -- Amendment 13: the s = 0 control arm ---------------------------------
def test_s0_control_arm_never_updates_the_state():
    module = _module()
    cb = FastStateS0Callbacks(module)
    cb.on_episode_start(_Ctx())
    zero_prefix = cb.prefix_provider().clone()
    cb.on_feedback(object(), _summary_fn())
    cb.on_feedback(object(), _summary_fn(3.0))
    assert torch.count_nonzero(cb.s) == 0
    assert cb.updates_applied == 0 and cb.updates_skipped == 2
    # every prompt of every episode sees the SAME learned static prefix
    assert torch.equal(cb.prefix_provider(), zero_prefix)
    cb.on_episode_start(_Ctx())
    assert torch.equal(cb.prefix_provider(), zero_prefix)
    assert cb.summary()["state_pinned_to_zero"] is True
    assert cb.summary()["eval_only_control"] is True
    assert cb.summary()["arm_id"] == ARM_FAST_STATE_ON_S0


def test_s0_control_uses_the_trained_module_and_differs_from_on():
    """The control must isolate the static prefix, not disable the mechanism:
    it injects ON's own trained prefix, and ON diverges from it as soon as the
    state is updated."""
    module = _module()
    on = FastStateCallbacks(module)
    s0 = FastStateS0Callbacks(module)
    for cb in (on, s0):
        cb.on_episode_start(_Ctx())
    assert torch.equal(on.prefix_provider(), s0.prefix_provider())
    on.on_feedback(object(), _summary_fn())
    s0.on_feedback(object(), _summary_fn())
    assert not torch.equal(on.prefix_provider(), s0.prefix_provider())
    assert s0.module is module and on.module is module


def test_s0_control_refuses_to_carry_state_or_be_gated():
    module = _module()
    cb = FastStateS0Callbacks(module, carry_across_episodes=True,
                              gate=_StubGate(True))
    assert cb.carry_across_episodes is False
    assert cb.gate is None


def test_fl5_records_disclaim_fl3_comparability():
    from foundation_learner.mechanisms.fl5_training import (
        FL5_COMPARABLE_TO_FL3, FL5_NOT_COMPARABLE_REASON, FL5ArmResult)

    assert FL5_COMPARABLE_TO_FL3 is False
    for phrase in ("impossible-task control", "parameter-matched",
                   "LEARNED STATIC PREFIX", "FAST_STATE_ON_S0"):
        assert phrase in FL5_NOT_COMPARABLE_REASON
    result = FL5ArmResult(arm_id="FAST_STATE_ON", out_dir="", steps=0,
                          stopped_reason="COMPLETED", final_loss=None).to_dict()
    assert result["comparable_to_fl3"] is False
    assert result["comparable_to_fl3_reason"] == FL5_NOT_COMPARABLE_REASON


def test_the_fl5_docstring_no_longer_claims_the_state_carried_the_learning():
    """The overclaim is a scientific defect, so its absence is a fixture."""
    from foundation_learner.mechanisms import fl5_training

    doc = fl5_training.__doc__ or ""
    # the old sentence survives ONLY as the quoted claim being retracted
    assert doc.count("and nothing else") == 1
    assert 'and nothing else" OVERCLAIMED' in doc
    for phrase in ("impossible-task control", "LEARNED STATIC PREFIX",
                   "not parameter-matched", "FAST_STATE_ON_S0"):
        assert phrase in doc
