"""HOSTILE: one over-long online episode must not destroy an evaluation.

Attack (Amendment 13, finding R-M1).  The ONLINE context is longer than the
pre-generated rendering it walks: every MODEL_ATTEMPT slot carries up to
``max_new_tokens`` REAL generated tokens instead of the scripted answer line.
Before the repair, ``run_episodes`` raised ``SequenceBudgetError`` on the FIRST
prompt that exceeded the budget, which aborted the whole batch — so a single
long episode deleted every other episode's evidence, and the loss was invisible
because no records came back at all.

What this fixture pins:

1. per-episode isolation — the offending episode is stopped, every other
   episode in the same batch runs to completion and keeps its full curve;
2. the offending episode is RECORDED (not dropped), with
   ``status = ONLINE_BUDGET_EXCEEDED``;
3. its partial ``R`` is MARKED MISSING, so no metric can average a truncated
   curve, and the generations it did complete are still in the transcript;
4. nothing is truncated (the record says so explicitly);
5. ``metrics`` excludes it from every aggregate AND reports the count, so a
   silently shrinking denominator is impossible.

The overflow is produced by injecting a SMALL allowance into the config, which
is the same code path a genuinely over-long episode takes; the frozen 4096
allowance is asserted separately (it must not be reachable by the real data).
"""
from __future__ import annotations

import pytest

from foundation_learner.evaluation import learning_curve as LC
from foundation_learner.evaluation import metrics as EM
from foundation_learner.evaluation.generation import GenerationConfig
from foundation_learner.tests import _w4_support as S

pytestmark = [
    pytest.mark.skipif(not S.checkpoint_present(),
                       reason="frozen checkpoint directory not present"),
    pytest.mark.skipif(not S.integration_available(),
                       reason="ecology/episodes packages absent"),
]

FAST_GEN = GenerationConfig(max_new_tokens=2)


def _cfg(**kwargs) -> LC.LearningCurveConfig:
    kwargs.setdefault("generation", FAST_GEN)
    return LC.LearningCurveConfig(**kwargs)


def _prompt_tokens(bundle, text: str) -> int:
    return len(bundle.tokenizer(text, add_special_tokens=False)["input_ids"])


def _allowance_that_only_the_long_episode_exceeds(bundle, episodes) -> int:
    """A budget that the SHORT episodes fit and the LONG one does not.

    Measured from the real rendered episodes, so the fixture cannot pass by
    picking a number that nothing exceeds.
    """
    from foundation_learner.episodes.render import render_episode

    lengths = [_prompt_tokens(bundle, render_episode(ep).text) for ep in episodes]
    short, long = min(lengths), max(lengths)
    assert long > short, "the fixture needs episodes of different lengths"
    return int((short + long) // 2)


def _long_episode(ep):
    """The same real episode with a long (but legal) task statement.

    Only the TASK text grows; the plan, the items, the verifier and the answers
    are untouched, so the episode remains a real episode that the walker would
    otherwise complete.
    """
    from dataclasses import replace

    from foundation_learner.episodes.schema import Role

    filler = " ".join(["irrelevant background sentence."] * 400)
    events = [replace(e, text=f"{e.text}\n{filler}") if e.role is Role.TASK else e
              for e in ep.events]
    return replace(ep, events=events)


def test_one_overflowing_episode_does_not_abort_the_batch():
    bundle = S.tiny_bundle()
    episodes, family, env_factory = S.episode_batch(3)
    episodes = [episodes[0], _long_episode(episodes[1]), episodes[2]]
    allowance = _allowance_that_only_the_long_episode_exceeds(bundle, episodes)

    records = LC.run_episodes(bundle, episodes, env_factory,
                             cfg=_cfg(max_seq_len=allowance))

    # (1) nothing is dropped: one record per episode, in order
    assert len(records) == len(episodes)
    assert [r["episode_id"] for r in records] == [
        ep.episode_id for ep in episodes]

    by_status: dict[str, list[dict]] = {}
    for r in records:
        by_status.setdefault(r["status"], []).append(r)
    assert set(by_status) == {LC.STATUS_COMPLETE,
                              LC.STATUS_ONLINE_BUDGET_EXCEEDED}
    assert len(by_status[LC.STATUS_ONLINE_BUDGET_EXCEEDED]) == 1

    # (2) the survivors are complete, not degraded by their neighbour
    for ok in by_status[LC.STATUS_COMPLETE]:
        assert ok["interaction_indices"] == list(range(7))
        assert ok["aulc"] is not None
        assert ok["n_attempts"] == 10

    # (3) the offending episode is recorded, marked, and NOT truncated
    bad = by_status[LC.STATUS_ONLINE_BUDGET_EXCEEDED][0]
    assert bad["R"] == {} and bad["aulc"] is None
    assert bad["interaction_indices"] == []
    assert bad["R_missing_reason"] == LC.STATUS_ONLINE_BUDGET_EXCEEDED
    event = bad["online_budget_event"]
    assert event["truncated"] is False
    assert event["allowance"] == allowance
    assert event["projected_tokens"] > allowance
    assert event["prompt_tokens"] > 0
    assert "truncation is forbidden" in event["detail"]


def test_the_excluded_episode_is_counted_and_never_averaged():
    bundle = S.tiny_bundle()
    episodes, family, env_factory = S.episode_batch(3)
    episodes = [episodes[0], _long_episode(episodes[1]), episodes[2]]
    allowance = _allowance_that_only_the_long_episode_exceeds(bundle, episodes)
    records = LC.run_episodes(bundle, episodes, env_factory,
                              cfg=_cfg(max_seq_len=allowance))

    kept = [r for r in records if r["status"] == LC.STATUS_COMPLETE]
    counts = EM.excluded_records(records)
    assert counts["n_records"] == 3
    assert counts["n_scoreable"] == len(kept) == 2
    assert counts["n_excluded"] == 1
    assert counts["by_status"] == {LC.STATUS_ONLINE_BUDGET_EXCEEDED: 1}
    assert counts["excluded_episode_ids"] == [
        r["episode_id"] for r in records
        if r["status"] == LC.STATUS_ONLINE_BUDGET_EXCEEDED]

    # the aggregate is EXACTLY the aggregate over the scoreable records: the
    # excluded episode neither contributes a partial curve nor a zero
    assert EM.macro_aulc(records) == EM.macro_aulc(kept)
    assert EM.summarize(records)["excluded_records"]["n_excluded"] == 1
    assert EM.record_is_scoreable(
        next(r for r in records
             if r["status"] == LC.STATUS_ONLINE_BUDGET_EXCEEDED)) is False


def test_the_batch_does_not_silently_drop_the_overflowing_episode():
    """The failure mode this fixture exists for: a driver that quietly skips a
    too-long episode would leave a SHORTER record list and a smaller
    denominator with nothing recording the loss."""
    bundle = S.tiny_bundle()
    episodes, family, env_factory = S.episode_batch(2)
    episodes = [_long_episode(episodes[0]), episodes[1]]
    allowance = _allowance_that_only_the_long_episode_exceeds(bundle, episodes)
    records = LC.run_episodes(bundle, episodes, env_factory,
                              cfg=_cfg(max_seq_len=allowance))

    assert len(records) == 2, "the overflowing episode must still be reported"
    excluded = [r for r in records
                if r["status"] == LC.STATUS_ONLINE_BUDGET_EXCEEDED]
    assert len(excluded) == 1
    assert excluded[0]["episode_id"] == episodes[0].episode_id
    # its record still certifies its own content
    payload = dict(excluded[0])
    claimed = payload.pop("record_hash")
    assert claimed == LC.domain_sha256(LC.EPISODE_RECORD_DOMAIN, payload)


def test_the_frozen_allowance_covers_the_real_pregenerated_data():
    """The 4096 allowance is not decorative: the longest real episode plus a
    full complement of generated tokens must fit inside it."""
    from foundation_learner.data.pools import (EVAL_MAX_NEW_TOKENS,
                                               EVAL_ONLINE_MAX_SEQ_LEN)
    from foundation_learner.episodes.render import render_episode
    from foundation_learner.episodes.schema import Role

    assert LC.EVAL_ONLINE_MAX_SEQ_LEN == EVAL_ONLINE_MAX_SEQ_LEN == 4096
    bundle = S.tiny_bundle()
    episodes, _family, _env = S.episode_batch(4)
    for ep in episodes:
        scripted = _prompt_tokens(bundle, render_episode(ep).text)
        slots = sum(1 for e in ep.events if e.role is Role.MODEL_ATTEMPT)
        projected = scripted + (slots + 1) * EVAL_MAX_NEW_TOKENS
        assert projected <= EVAL_ONLINE_MAX_SEQ_LEN


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
