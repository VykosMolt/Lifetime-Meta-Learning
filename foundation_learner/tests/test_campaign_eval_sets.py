"""Evaluation-set assembly against the REAL pre-generated shards (R-C3).

The campaign's metrics are macro-averaged over families and family-clustered
(contract §9/§14).  A global ``episodes[:cap]`` slice over shard-ordered
episodes silently returns one family's episodes, which turns every one of those
metrics into a single-family metric without any error.  These tests use the
tiny pre-generation root, so they exercise the real loaders, the real shard
layout and the real split — not a stand-in.
"""
from __future__ import annotations

import os

import pytest

from foundation_learner.campaign import o1_isolation
from foundation_learner.campaign import stage_definitions as sd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TINY = os.path.join(REPO, "artifacts_fl", "pregen_tiny")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(TINY, "PREGEN_MANIFEST.json")),
    reason="no tiny pregeneration present")

DEV_FAMILIES = sd.split_families("DEVELOPMENT")


def ctx(tmp_path, **kw):
    guard = o1_isolation.IsolationGuard(label="TEST_EVAL_SETS")
    return sd.StageContext(out_dir=str(tmp_path), pregen_root=TINY,
                           bundle_factory=lambda: None, guard=guard, **kw)


def families_of(episodes):
    return sorted({ep.family_id for ep in episodes})


def test_a_capped_dev_eval_set_still_covers_every_development_family(tmp_path):
    c = ctx(tmp_path, eval_episode_cap=3)
    episodes = sd._dev_episodes(c, sd.STAGES_BY_ID["FL0"])
    assert families_of(episodes) == list(DEV_FAMILIES)
    assert len(episodes) == 3                      # one per family, not 3 of one


def test_the_uncapped_dev_eval_set_is_balanced(tmp_path):
    c = ctx(tmp_path)
    episodes = sd._dev_episodes(c, sd.STAGES_BY_ID["FL3"])
    counts = {f: sum(1 for ep in episodes if ep.family_id == f)
              for f in DEV_FAMILIES}
    assert set(counts.values()) == {3}, counts       # the whole tiny DEV pool


def test_the_old_global_slice_would_have_collapsed_to_one_family(tmp_path):
    """The defect this replaces, demonstrated on the real shards."""
    c = ctx(tmp_path)
    everything = sd.load_episodes(TINY, "DEVELOPMENT", "scripted", guard=c.guard)
    collapsed = everything[:3]                       # the previous behaviour
    assert len(families_of(collapsed)) == 1
    with pytest.raises(sd.StageError):
        sd.assert_eval_family_coverage(collapsed, "DEVELOPMENT",
                                       context="the previous global slice")


def test_the_mechanism_helper_uses_the_same_balanced_assembler(tmp_path):
    from foundation_learner.mechanisms import stage_support as ss

    c = ctx(tmp_path, eval_episode_cap=3)
    stage = sd.STAGES_BY_ID["FL5"]
    assert families_of(ss.dev_episodes(c, stage)) == list(DEV_FAMILIES)
    assert families_of(ss.dev_episodes(c, stage, count=3)) == list(DEV_FAMILIES)


def test_chains_resolve_against_the_full_pool_and_stay_pair_balanced(tmp_path):
    c = ctx(tmp_path, eval_episode_cap=3)
    # a caller holding only the capped subset still gets resolved chains: the
    # resolver completes the pool from the split's own shards
    subset = sd._dev_episodes(c, sd.STAGES_BY_ID["FL8"])
    chains = sd.load_chain_specs(c, subset)
    assert chains, "no chain resolved against the capped subset"
    pairs = {(ch.family_a, ch.family_b) for ch in chains}
    assert len(pairs) == 6, pairs                  # every ordered DEV pair

    # a LIMIT draws round-robin over the ordered pairs, never one pair only
    limited = sd.load_chain_specs(c, subset, limit=3)
    assert len(limited) == 3
    assert len({(ch.family_a, ch.family_b) for ch in limited}) == 3


def test_the_mechanism_chain_helper_delegates_to_the_campaign(tmp_path):
    from foundation_learner.evaluation.interference import check_pair_balance
    from foundation_learner.mechanisms import stage_support as ss

    c = ctx(tmp_path)
    pool = sd.load_episodes(TINY, "DEVELOPMENT", "scripted", guard=c.guard)
    chains = ss.load_chain_specs(c, pool)
    assert len(chains) == 12                        # 2 per ordered pair
    check_pair_balance(chains, require_balanced=True)


def test_the_sealed_eval_set_is_sliced_per_shard(tmp_path):
    """The sealed opening's own set is balanced too (it is the headline test)."""
    c = ctx(tmp_path, eval_episode_cap=3)
    plan = sd.eval_plan(c, sd.STAGES_BY_ID["SEALED_EVAL"])
    assert plan["split"] == "SEALED_TEST"
    assert plan["limit_per_family"] == 1
    assert sorted(plan["families"]) == list(sd.split_families("SEALED_TEST"))
