"""FL8 consolidation into a separate slow bank (W3, contract §7)."""
from __future__ import annotations

import pytest
import torch

from foundation_learner.mechanisms.consolidation import (
    CONSOLIDATION_MODES,
    FL8_CONSOLIDATION_SCALE,
    ConsolidationError,
    Consolidator,
    SealedInputRefused,
    capture_fast_delta,
    consolidation_eval_plan,
    slow_bank_report,
)
from foundation_learner.mechanisms.fast_adapter import FastAdapter
from foundation_learner.tests._w3_support import (
    DEV_FAMILY,
    SEALED_FAMILY,
    fresh_tiny_bundle,
    real_episode,
)


def _adapted_adapter(seed: int = 51):
    """A real FL7 adapter carrying a real (non-zero) fast delta.

    ``inner_steps=1``: FL8 consolidates whatever delta FL7 produced, so a
    single real inner step is enough here and keeps the suite fast.  The
    4-step contract value is exercised in ``test_mechanisms_fast_adapter``.
    """
    adapter = FastAdapter(fresh_tiny_bundle(), seed=seed, inner_steps=1)
    adapter.adapt_episode(real_episode(reveal=True))
    return adapter


def _delta(adapter, *, episode_id="ep", n_admitted=1, family=None):
    episode = real_episode()
    return capture_fast_delta(
        adapter.handles, episode_id=episode_id,
        family_id=family or episode.family_id, split="TRAIN",
        n_admitted=n_admitted)


def test_frozen_modes_and_scale():
    assert CONSOLIDATION_MODES == ("none", "indiscriminate", "value_gated")
    assert FL8_CONSOLIDATION_SCALE == 0.5


def test_merge_math_matches_a_hand_computed_dense_delta():
    adapter = _adapted_adapter()
    name = adapter.handles.names()[0]
    layer = adapter.handles.layers[name]
    d1 = _delta(adapter, episode_id="e1")
    dense1 = layer.scaling * (layer.lora_B.detach() @ layer.lora_A.detach())

    # a second, different delta
    adapter.reset_episode()
    adapter.adapt_episode(real_episode(index=1, reveal=True))
    d2 = _delta(adapter, episode_id="e2")
    dense2 = layer.scaling * (layer.lora_B.detach() @ layer.lora_A.detach())
    assert not torch.allclose(dense1, dense2)

    consolidator = Consolidator(adapter.handles, mode="indiscriminate")
    consolidator.record_many([d1, d2])
    report = consolidator.consolidate()
    assert report["n_selected"] == 2
    expected = 0.5 * (dense1 + dense2)
    assert torch.allclose(layer.effective_delta().detach(), expected, atol=1e-6)


def test_fast_state_is_cleared_and_the_bank_survives():
    adapter = _adapted_adapter()
    consolidator = Consolidator(adapter.handles, mode="indiscriminate")
    consolidator.record(_delta(adapter))
    report = consolidator.consolidate()
    assert report["fast_delta_norm_before_clear"] > 0.0
    assert report["fast_delta_norm_after_clear"] == 0.0
    assert adapter.delta_norm() == 0.0
    bank = slow_bank_report(adapter.handles)
    assert bank["n_components"] == len(adapter.handles)
    assert bank["frobenius_norm"] > 0.0


def test_mode_none_merges_nothing():
    adapter = _adapted_adapter()
    consolidator = Consolidator(adapter.handles, mode="none")
    consolidator.record(_delta(adapter))
    report = consolidator.consolidate()
    assert report["n_selected"] == 0
    assert slow_bank_report(adapter.handles)["n_components"] == 0


def test_value_gated_mode_selects_only_admitted_episodes():
    adapter = _adapted_adapter()
    consolidator = Consolidator(adapter.handles, mode="value_gated")
    consolidator.record(_delta(adapter, episode_id="admitted", n_admitted=2))
    consolidator.record(_delta(adapter, episode_id="declined", n_admitted=0))
    report = consolidator.consolidate()
    assert report["selected_episode_ids"] == ["admitted"]
    assert report["declined_episode_ids"] == ["declined"]
    assert slow_bank_report(adapter.handles)["n_components"] == len(adapter.handles)


def test_consolidation_is_deterministic():
    reports = []
    for _ in range(2):
        adapter = _adapted_adapter()
        consolidator = Consolidator(adapter.handles, mode="indiscriminate")
        consolidator.record(_delta(adapter))
        reports.append(consolidator.consolidate())
    assert reports[0]["report_hash"] == reports[1]["report_hash"]


def test_recorded_delta_must_target_the_same_modules():
    adapter = _adapted_adapter()
    delta = _delta(adapter)
    delta.layers.pop(next(iter(delta.layers)))
    with pytest.raises(ConsolidationError):
        Consolidator(adapter.handles).record(delta)


def test_unknown_mode_is_refused():
    adapter = _adapted_adapter()
    with pytest.raises(ConsolidationError):
        Consolidator(adapter.handles, mode="everything")


def test_sealed_families_are_refused_everywhere():
    adapter = _adapted_adapter()
    with pytest.raises(SealedInputRefused):
        capture_fast_delta(adapter.handles, episode_id="x",
                           family_id=SEALED_FAMILY, split="SEALED_TEST")
    with pytest.raises(SealedInputRefused):
        capture_fast_delta(adapter.handles, episode_id="x",
                           family_id=SEALED_FAMILY, split="TRAIN")
    delta = _delta(adapter)
    consolidator = Consolidator(adapter.handles)
    consolidator.record(delta)
    delta.family_id = SEALED_FAMILY          # smuggled in after recording
    with pytest.raises(SealedInputRefused):
        consolidator.consolidate()


def test_eval_plan_builds_chains_for_w4():
    a1 = [real_episode(index=0), real_episode(index=1)]
    b = [real_episode(DEV_FAMILY, index=0), real_episode(DEV_FAMILY, index=1)]
    a2 = [real_episode(index=2), real_episode(index=3)]
    unrelated = [real_episode("dsl_execution", index=0)]
    out = consolidation_eval_plan(
        family_a=a1[0].family_id, family_b=DEV_FAMILY, episodes_a1=a1,
        episodes_b=b, episodes_a2=a2, unrelated_episodes=unrelated)
    assert out["plan"]["n_chains"] == 2
    assert len(out["chains"]) == 2
    chain = out["chains"][0]
    assert set(chain.episodes()) == {"a1", "b", "a2"}
    assert chain.family_a == a1[0].family_id and chain.family_b == DEV_FAMILY
    assert out["plan"]["unrelated_families"] == ["dsl_execution"]
    assert out["plan"]["plan_hash"]

    from foundation_learner.evaluation.interference import check_pair_balance
    assert check_pair_balance(out["chains"])  # consumable by W4 unchanged


def test_eval_plan_refuses_sealed_and_degenerate_inputs():
    a1 = [real_episode()]
    b = [real_episode(DEV_FAMILY)]
    a2 = [real_episode(index=1)]
    with pytest.raises(SealedInputRefused):
        consolidation_eval_plan(family_a=SEALED_FAMILY, family_b=DEV_FAMILY,
                                episodes_a1=a1, episodes_b=b, episodes_a2=a2)
    with pytest.raises(ConsolidationError):
        consolidation_eval_plan(family_a=a1[0].family_id,
                                family_b=a1[0].family_id, episodes_a1=a1,
                                episodes_b=a1, episodes_a2=a2)
    with pytest.raises(ConsolidationError):
        consolidation_eval_plan(family_a=a1[0].family_id, family_b=DEV_FAMILY,
                                episodes_a1=a1, episodes_b=b, episodes_a2=a2,
                                unrelated_episodes=[real_episode(DEV_FAMILY,
                                                                 index=9)])
    with pytest.raises(ConsolidationError):
        consolidation_eval_plan(family_a=a1[0].family_id, family_b=DEV_FAMILY,
                                episodes_a1=[], episodes_b=b, episodes_a2=a2)
