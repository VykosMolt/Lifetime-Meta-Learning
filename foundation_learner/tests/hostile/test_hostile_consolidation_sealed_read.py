"""HOSTILE: FL8 must refuse SEALED_TEST material anywhere (contract §12, §18).

"Consolidation reading sealed performance" is one of the named attacks.  A
sealed family id must be refused at EVERY entry point of the consolidation
path, whether it arrives honestly (declared split) or smuggled (relabelled
split, or mutated after recording).  A consolidator must never merge a delta
that came from a sealed family, because that would be a model modification
justified by sealed material.
"""
from __future__ import annotations

import pytest

from foundation_learner.ecology.split import families_in, split_of
from foundation_learner.mechanisms.consolidation import (
    ConsolidationError,
    Consolidator,
    SealedInputRefused,
    assert_not_sealed,
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

SEALED_FAMILIES = tuple(families_in("SEALED_TEST"))


def _adapter():
    adapter = FastAdapter(fresh_tiny_bundle(), seed=111, inner_steps=1)
    adapter.adapt_episode(real_episode(reveal=True))
    return adapter


def test_every_sealed_family_is_refused_by_the_guard():
    assert SEALED_FAMILY in SEALED_FAMILIES
    for family in SEALED_FAMILIES:
        assert split_of(family) == "SEALED_TEST"
        with pytest.raises(SealedInputRefused):
            assert_not_sealed(family)
        with pytest.raises(SealedInputRefused):
            assert_not_sealed(family, "TRAIN")          # relabelled
        with pytest.raises(SealedInputRefused):
            assert_not_sealed("unknown_family", "SEALED_TEST")


def test_unknown_family_without_a_split_is_refused_not_guessed():
    with pytest.raises(ConsolidationError):
        assert_not_sealed("not_a_family")


def test_capture_refuses_sealed_deltas():
    adapter = _adapter()
    for family in SEALED_FAMILIES:
        for split in ("SEALED_TEST", "TRAIN", "DEVELOPMENT"):
            with pytest.raises(SealedInputRefused):
                capture_fast_delta(adapter.handles, episode_id="x",
                                   family_id=family, split=split)


def test_recording_and_consolidating_refuse_smuggled_sealed_material():
    adapter = _adapter()
    episode = real_episode()
    delta = capture_fast_delta(adapter.handles, episode_id="e1",
                               family_id=episode.family_id, split="TRAIN",
                               n_admitted=1)
    consolidator = Consolidator(adapter.handles, mode="indiscriminate")
    consolidator.record(delta)

    # smuggle: mutate the record AFTER it was accepted
    delta.family_id = SEALED_FAMILY
    with pytest.raises(SealedInputRefused):
        consolidator.consolidate()
    # nothing was merged and the fast delta was not cleared
    assert slow_bank_report(adapter.handles)["n_components"] == 0
    assert adapter.delta_norm() > 0.0

    delta.family_id = episode.family_id
    delta.split = "SEALED_TEST"
    with pytest.raises(SealedInputRefused):
        consolidator.consolidate()
    assert slow_bank_report(adapter.handles)["n_components"] == 0


def test_eval_plan_refuses_sealed_families_in_every_position():
    a1 = [real_episode(index=0)]
    b = [real_episode(DEV_FAMILY, index=0)]
    a2 = [real_episode(index=1)]
    sealed = [real_episode(SEALED_FAMILY, index=0)]
    family_a = a1[0].family_id

    with pytest.raises(SealedInputRefused):
        consolidation_eval_plan(family_a=SEALED_FAMILY, family_b=DEV_FAMILY,
                                episodes_a1=sealed, episodes_b=b,
                                episodes_a2=sealed)
    with pytest.raises(SealedInputRefused):
        consolidation_eval_plan(family_a=family_a, family_b=SEALED_FAMILY,
                                episodes_a1=a1, episodes_b=sealed,
                                episodes_a2=a2)
    with pytest.raises(SealedInputRefused):
        consolidation_eval_plan(family_a=family_a, family_b=DEV_FAMILY,
                                episodes_a1=a1, episodes_b=b, episodes_a2=a2,
                                unrelated_episodes=sealed)


def test_a_clean_plan_still_works_so_the_guard_is_not_blanket_refusal():
    a1 = [real_episode(index=0)]
    b = [real_episode(DEV_FAMILY, index=0)]
    a2 = [real_episode(index=1)]
    out = consolidation_eval_plan(family_a=a1[0].family_id,
                                  family_b=DEV_FAMILY, episodes_a1=a1,
                                  episodes_b=b, episodes_a2=a2)
    assert out["plan"]["n_chains"] == 1
