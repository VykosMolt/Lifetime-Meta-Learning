"""FL5/FL7 mechanisms through W4's REAL online evaluators (W3).

These are integration tests of the MECHANISM plumbing, not of competence: the
tiny model is random, so ``R_k`` is expected to be 0.  What must hold is that
the callbacks run end to end through ``evaluation.learning_curve``'s real
walker, that the episode record reports the prefix, and that the state survives
the removal of the textual history in the context-reset condition.
"""
from __future__ import annotations

import torch

from foundation_learner.evaluation.generation import GenerationConfig
from foundation_learner.evaluation.learning_curve import (
    LearningCurveConfig,
    run_episode,
    run_episodes,
)
from foundation_learner.mechanisms.fast_state import (
    FastStateCallbacks,
    FastStateModule,
    FastStateOffCallbacks,
)
from foundation_learner.tests._w3_support import (
    DEV_FAMILY,
    embedding_matrix,
    real_environment,
    real_episode,
    tiny_bundle,
)

FAST_CFG = LearningCurveConfig(generation=GenerationConfig(max_new_tokens=4),
                               arm_tag="FL5_INTEGRATION")


def _module(bundle, seed: int = 61) -> FastStateModule:
    return FastStateModule(bundle.model.config.hidden_size,
                           embedding_matrix=embedding_matrix(bundle),
                           state_dim=32, seed=seed)


def test_fl5_callbacks_run_through_the_real_walker():
    bundle = tiny_bundle()
    episode = real_episode(DEV_FAMILY)
    module = _module(bundle)
    cb = FastStateCallbacks(module)
    record = run_episode(bundle, episode, real_environment(episode),
                         cfg=FAST_CFG, callbacks=cb)
    assert record["prefix_used"] is True
    assert record["callbacks_class"] == "FastStateCallbacks"
    assert sorted(int(k) for k in record["R"]) == list(range(7))
    assert record["n_feedback_events"] > 0
    assert cb.updates_applied == record["n_feedback_events"]
    assert float(torch.linalg.vector_norm(cb.s)) > 0.0
    assert cb.summary()["module_config_hash"] == module.config_hash()


def test_off_arm_reports_no_prefix_on_the_same_episode():
    bundle = tiny_bundle()
    episode = real_episode(DEV_FAMILY)
    record = run_episode(bundle, episode, real_environment(episode),
                         cfg=FAST_CFG, callbacks=FastStateOffCallbacks())
    assert record["prefix_used"] is False
    assert record["callbacks_class"] == "FastStateOffCallbacks"


def test_state_survives_the_context_reset_condition():
    bundle = tiny_bundle()
    episode = real_episode(DEV_FAMILY)
    module = _module(bundle)
    cb = FastStateCallbacks(module)
    cfg = LearningCurveConfig(generation=GenerationConfig(max_new_tokens=4),
                             context_mode="context_reset", reset_from_index=4,
                             arm_tag="FL5_CONTEXT_RESET")
    record = run_episode(bundle, episode, real_environment(episode),
                         cfg=cfg, callbacks=cb)
    assert record["context_mode"] == "context_reset"
    assert record["reset_prompts_used"] > 0
    # the textual history is gone for those attempts, but the state is not
    assert record["prefix_used"] is True
    assert cb.updates_applied > 0
    assert float(torch.linalg.vector_norm(cb.s)) > 0.0


def test_one_callbacks_instance_per_episode_is_enforced_by_the_walker():
    bundle = tiny_bundle()
    episodes = [real_episode(DEV_FAMILY, index=0), real_episode(DEV_FAMILY, index=1)]
    module = _module(bundle)
    per_episode = {id(ep): FastStateCallbacks(module) for ep in episodes}
    records = run_episodes(bundle, episodes, real_environment,
                           cfg=FAST_CFG,
                           callbacks_factory=lambda ep: per_episode[id(ep)])
    assert len(records) == 2
    assert all(r["prefix_used"] for r in records)
    states = [cb.s for cb in per_episode.values()]
    assert not torch.equal(states[0], states[1])
