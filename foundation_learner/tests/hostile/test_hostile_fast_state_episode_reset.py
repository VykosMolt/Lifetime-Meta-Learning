"""HOSTILE: the FL5 fast state must NOT survive an episode reset (contract §18).

The attack is a callbacks subclass that "forgets" to reset at episode start.
The fixture proves (a) the real mechanism resets to EXACT zero, and (b) the
assertion used to check that has teeth — it fails on the leaking variant.
"""
from __future__ import annotations

import pytest
import torch

from foundation_learner.mechanisms.fast_state import (
    FastStateCallbacks,
    FastStateModule,
)


class _Ctx:
    episode_id = "ep"


def _module(seed: int = 71) -> FastStateModule:
    return FastStateModule(16, rho=1.0, state_dim=8, seed=seed)


def _summary_fn(value: float = 2.0):
    def fn(reduce="mean", include_prefix=False):
        return torch.full((16,), float(value))
    return fn


def assert_state_is_reset(cb) -> None:
    """The invariant under attack: s == 0 exactly at episode start."""
    assert torch.count_nonzero(cb.s) == 0, "fast state survived the episode reset"


class LeakingCallbacks(FastStateCallbacks):
    """DELIBERATELY BROKEN: never resets."""

    def on_episode_start(self, ctx):
        self.episodes_seen += 1


def _drive(cb, updates: int = 3) -> None:
    cb.on_episode_start(_Ctx())
    for _ in range(updates):
        cb.on_feedback(object(), _summary_fn())


def test_reset_state_returns_exact_zeros_and_a_fresh_tensor():
    module = _module()
    s = module.reset_state(2)
    assert torch.count_nonzero(s) == 0
    s2 = module.update(s, torch.ones(2, 16))
    assert torch.count_nonzero(s2) > 0
    assert torch.count_nonzero(module.reset_state(2)) == 0
    assert module.reset_state(2).data_ptr() != s.data_ptr()


def test_real_mechanism_resets_between_episodes():
    cb = FastStateCallbacks(_module())
    _drive(cb)
    assert float(torch.linalg.vector_norm(cb.s)) > 0.0
    cb.on_episode_start(_Ctx())
    assert_state_is_reset(cb)


def test_many_updates_still_reset_to_exact_zero():
    cb = FastStateCallbacks(_module())
    _drive(cb, updates=25)
    cb.on_episode_start(_Ctx())
    assert_state_is_reset(cb)
    assert cb.episodes_seen == 2


def test_the_assertion_has_teeth():
    cb = LeakingCallbacks(_module())
    _drive(cb)
    cb.on_episode_start(_Ctx())
    with pytest.raises(AssertionError):
        assert_state_is_reset(cb)


def test_prefix_after_a_reset_is_the_zero_state_prefix():
    module = _module()
    cb = FastStateCallbacks(module)
    cb.on_episode_start(_Ctx())
    fresh = cb.prefix_provider()
    _drive(cb)
    adapted = cb.prefix_provider()
    assert not torch.equal(fresh, adapted)
    cb.on_episode_start(_Ctx())
    assert torch.equal(cb.prefix_provider(), fresh)
