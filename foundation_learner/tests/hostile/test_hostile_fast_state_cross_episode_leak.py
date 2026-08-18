"""HOSTILE: FL5 state must not cross episodes (contract §18).

Attacks:

1. Reuse ONE default callbacks object for two episodes -> the second episode
   must start from zero.
2. Hand the real walker a callbacks factory that returns the SAME object for
   two episodes -> W4's guard must refuse the run outright.
3. Two independent episodes with independent callbacks must not share state.
4. Chain carry (the interference-evaluator mode) must be OPT-IN: the default
   never carries, and the carrying variant is a visible constructor argument.
"""
from __future__ import annotations

import pytest
import torch

from foundation_learner.evaluation.generation import GenerationConfig
from foundation_learner.evaluation.learning_curve import (
    LearningCurveConfig,
    run_episodes,
)
from foundation_learner.mechanisms.fast_state import (
    FastStateCallbacks,
    FastStateModule,
)
from foundation_learner.tests._w3_support import (
    DEV_FAMILY,
    embedding_matrix,
    real_environment,
    real_episode,
    tiny_bundle,
)


class _Ctx:
    def __init__(self, episode_id="ep"):
        self.episode_id = episode_id


def _module(seed: int = 81) -> FastStateModule:
    return FastStateModule(16, rho=1.0, state_dim=8, seed=seed)


def _summary_fn(value: float = 1.5):
    def fn(reduce="mean", include_prefix=False):
        return torch.full((16,), float(value))
    return fn


def test_default_callbacks_never_carry_state_into_the_next_episode():
    cb = FastStateCallbacks(_module())
    cb.on_episode_start(_Ctx("A"))
    cb.on_feedback(object(), _summary_fn())
    leaked = cb.s.clone()
    assert float(torch.linalg.vector_norm(leaked)) > 0.0
    cb.on_episode_start(_Ctx("B"))
    assert torch.count_nonzero(cb.s) == 0
    assert not torch.equal(cb.s, leaked)


def test_independent_episodes_do_not_share_state():
    module = _module()
    a, b = FastStateCallbacks(module), FastStateCallbacks(module)
    a.on_episode_start(_Ctx("A"))
    b.on_episode_start(_Ctx("B"))
    a.on_feedback(object(), _summary_fn(3.0))
    assert torch.count_nonzero(b.s) == 0
    b.on_feedback(object(), _summary_fn(-3.0))
    assert not torch.equal(a.s, b.s)


def test_carry_is_opt_in_and_visible():
    import inspect

    signature = inspect.signature(FastStateCallbacks.__init__)
    assert signature.parameters["carry_across_episodes"].default is False
    cb = FastStateCallbacks(_module(), carry_across_episodes=True)
    cb.on_episode_start(_Ctx("A"))
    cb.on_feedback(object(), _summary_fn())
    carried = cb.s.clone()
    cb.on_episode_start(_Ctx("B"))
    assert torch.equal(cb.s, carried)
    assert cb.summary()["carry_across_episodes"] is True


def test_the_walker_refuses_a_shared_callbacks_object():
    bundle = tiny_bundle()
    episodes = [real_episode(DEV_FAMILY, index=0),
                real_episode(DEV_FAMILY, index=1)]
    shared = FastStateCallbacks(
        FastStateModule(bundle.model.config.hidden_size,
                        embedding_matrix=embedding_matrix(bundle),
                        state_dim=8, seed=82))
    with pytest.raises(ValueError):
        run_episodes(bundle, episodes, real_environment,
                     cfg=LearningCurveConfig(
                         generation=GenerationConfig(max_new_tokens=2)),
                     callbacks_factory=lambda ep: shared)
