"""HOSTILE: choose the best-DEV checkpoint from SEALED_TEST performance.

Contract §15 fixes the retained checkpoints (initial, best-DEV, final,
pre-promotion) and the best-DEV rule (max DEV macro-AULC at scheduled evals).
Selecting on the sealed set would be sealed-test peeking with extra steps, so
the rule consumes :class:`DevMetrics` — which cannot hold a sealed record — and
refuses anything else.
"""
from __future__ import annotations

import pytest

from foundation_learner.campaign.promotion import (DevMetrics, PromotionError,
                                                   SealedEvidenceRefused,
                                                   select_best_dev_checkpoint)
from foundation_learner.ecology.split import compute_split

DEV = tuple(compute_split()["DEVELOPMENT"])
SEALED = tuple(compute_split()["SEALED_TEST"])


def dev(aulc, families=DEV):
    return DevMetrics(stage="FL3", macro_aulc=aulc, slope=0.1,
                      n_episodes=100, family_ids=tuple(families))


def test_a_sealed_scored_checkpoint_cannot_even_be_described():
    with pytest.raises(SealedEvidenceRefused):
        DevMetrics(stage="FL3", macro_aulc=0.99, split="SEALED_TEST")
    with pytest.raises(SealedEvidenceRefused):
        dev(0.99, families=DEV + SEALED[:1])


def test_raw_sealed_scores_are_refused_by_the_selector():
    candidates = [("step200", dev(0.4)),
                  ("step400", {"macro_aulc": 0.99, "split": "SEALED_TEST"})]
    with pytest.raises(PromotionError) as exc:
        select_best_dev_checkpoint(candidates)
    assert "DevMetrics" in str(exc.value)


def test_the_dev_winner_is_kept_even_when_a_sealed_score_would_differ():
    """The sealed set cannot change the choice because it cannot enter it."""
    chosen = select_best_dev_checkpoint([("step200", dev(0.61)),
                                         ("step400", dev(0.60))])
    assert chosen["tag"] == "step200"
    assert chosen["sealed_consulted"] is False
    assert [c["tag"] for c in chosen["candidates"]] == ["step200", "step400"]


def test_a_checkpoint_without_dev_evidence_is_refused_not_guessed():
    with pytest.raises(PromotionError):
        select_best_dev_checkpoint([
            ("step200", DevMetrics(stage="FL3", macro_aulc=None))])


def test_the_retained_tags_are_the_frozen_set():
    from foundation_learner.training.checkpointing import RETAINED_TAGS

    assert set(RETAINED_TAGS) == {"initial", "best_dev", "final",
                                  "pre_promotion"}


def test_the_dev_decisions_record_carries_the_checkpoint_tags(tmp_path):
    """§12 requires the frozen decisions to name the checkpoints they used."""
    from foundation_learner.campaign import sealed_gate

    payload = sealed_gate.build_dev_decisions(
        chosen_learning_rate=1e-4, chosen_scope="PEFT_MODE", updates_U=600,
        stage_states={"FL3": "COMPLETE"},
        checkpoint_tags={"FL3": ["initial", "best_dev", "final"]},
        hashes={"split_manifest_sha256": "0" * 64})
    sealed_gate.validate_dev_decisions(payload)
    assert payload["checkpoint_tags"]["FL3"] == ["initial", "best_dev", "final"]
