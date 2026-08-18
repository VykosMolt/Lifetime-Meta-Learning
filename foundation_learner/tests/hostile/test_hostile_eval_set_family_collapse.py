"""HOSTILE: an evaluation set that quietly drops a split family.

Attack: assemble an evaluation set that contains only some of the split's
families — by a global ``episodes[:cap]`` slice over shard-ordered episodes, by
a shard that failed to load, or by a caller passing a hand-picked list — and
see whether the campaign reports a "macro-AULC" computed over one family as
though it were the contract's family-macro metric.

Required behaviour: every place an evaluation set is assembled asserts that the
assembled families are EXACTLY the frozen split's families and refuses
otherwise.  A one-family macro average is not a smaller version of the metric,
it is a different metric (contract §9 "no single-family domination", §14
family-clustered uncertainty).
"""
from __future__ import annotations

import os

import pytest

from foundation_learner.campaign import o1_isolation
from foundation_learner.campaign import stage_definitions as sd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
TINY = os.path.join(REPO, "artifacts_fl", "pregen_tiny")

DEV = sd.split_families("DEVELOPMENT")


class _Ep:
    def __init__(self, family_id, episode_id):
        self.family_id = family_id
        self.episode_id = episode_id


def ctx(tmp_path, pregen=TINY, **kw):
    guard = o1_isolation.IsolationGuard(label="HOSTILE_EVAL_SET")
    return sd.StageContext(out_dir=str(tmp_path), pregen_root=pregen,
                           bundle_factory=lambda: None, guard=guard, **kw)


def test_a_single_family_eval_set_is_refused():
    episodes = [_Ep(DEV[0], f"e{i}") for i in range(50)]
    with pytest.raises(sd.StageError) as exc:
        sd.assert_eval_family_coverage(episodes, "DEVELOPMENT",
                                       context="single-family attack")
    assert "macro-averaged metric" in str(exc.value)


def test_dropping_one_family_is_refused():
    episodes = [_Ep(f, f"e{i}") for i, f in enumerate(DEV[:-1])]
    with pytest.raises(sd.StageError) as exc:
        sd.assert_eval_family_coverage(episodes, "DEVELOPMENT",
                                       context="dropped-family attack")
    assert DEV[-1] in str(exc.value)


def test_an_empty_eval_set_is_refused():
    with pytest.raises(sd.StageError):
        sd.assert_eval_family_coverage([], "DEVELOPMENT", context="empty")


def test_padding_with_duplicates_of_one_family_does_not_hide_the_gap():
    """A set of the right SIZE but the wrong composition still refuses."""
    episodes = ([_Ep(DEV[0], f"a{i}") for i in range(10)]
                + [_Ep(DEV[1], f"b{i}") for i in range(10)])
    with pytest.raises(sd.StageError):
        sd.assert_eval_family_coverage(episodes, "DEVELOPMENT",
                                       context="padded attack")


@pytest.mark.skipif(not os.path.isfile(os.path.join(TINY,
                                                    "PREGEN_MANIFEST.json")),
                    reason="no tiny pregeneration present")
def test_a_missing_development_shard_makes_the_stage_refuse(tmp_path):
    """A shard that is not there must break the stage, not the metric."""
    import shutil

    pregen = tmp_path / "pregen"
    shutil.copytree(TINY, pregen)
    victim = DEV[0]
    os.remove(pregen / "episodes" / "DEVELOPMENT" / f"{victim}.scripted.jsonl")
    c = ctx(tmp_path, pregen=str(pregen))
    with pytest.raises(sd.StageError) as exc:
        sd._dev_episodes(c, sd.STAGES_BY_ID["FL0"])
    assert victim in str(exc.value)


@pytest.mark.skipif(not os.path.isfile(os.path.join(TINY,
                                                    "PREGEN_MANIFEST.json")),
                    reason="no tiny pregeneration present")
def test_the_real_loader_keeps_every_family_at_every_cap(tmp_path):
    for cap in (1, 2, 3, 9, None):
        c = ctx(tmp_path, eval_episode_cap=cap)
        episodes = sd._dev_episodes(c, sd.STAGES_BY_ID["FL3"])
        assert sorted({ep.family_id for ep in episodes}) == list(DEV), cap
