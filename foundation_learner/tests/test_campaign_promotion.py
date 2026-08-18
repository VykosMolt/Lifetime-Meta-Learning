"""Promotion ladder rules (contract §10) and the §15 best-DEV rule.

The rules are checked at their exact thresholds, and the structural sealed
exclusion is checked from both directions: a sealed split tag and a sealed
family id must each make the evidence object unconstructable.
"""
from __future__ import annotations

import pytest

from foundation_learner.campaign import promotion as promo
from foundation_learner.ecology.split import compute_split

DEV_FAMILIES = tuple(compute_split()["DEVELOPMENT"])
SEALED_FAMILIES = tuple(compute_split()["SEALED_TEST"])


def dev(stage, aulc, slope=0.1, events=(), families=DEV_FAMILIES):
    return promo.DevMetrics(stage=stage, macro_aulc=aulc, slope=slope,
                            n_episodes=300, family_ids=tuple(families),
                            stability_events=tuple(events))


# ---------------- structural sealed exclusion ----------------

def test_dev_metrics_refuses_a_sealed_split_tag():
    with pytest.raises(promo.SealedEvidenceRefused):
        promo.DevMetrics(stage="FL3", macro_aulc=0.9, split="SEALED_TEST")


def test_dev_metrics_refuses_sealed_families():
    with pytest.raises(promo.SealedEvidenceRefused) as exc:
        dev("FL3", 0.9, families=DEV_FAMILIES + SEALED_FAMILIES[:1])
    assert SEALED_FAMILIES[0] in str(exc.value)


def test_from_records_refuses_sealed_records():
    records = [{"split": "SEALED_TEST", "family_id": SEALED_FAMILIES[0],
                "R": {"0": 1.0}}]
    with pytest.raises(promo.SealedEvidenceRefused):
        promo.DevMetrics.from_records(records, stage="FL3")


def test_from_records_refuses_untagged_records():
    with pytest.raises(promo.PromotionError):
        promo.DevMetrics.from_records([{"family_id": DEV_FAMILIES[0]}],
                                      stage="FL3")


def test_from_records_builds_from_real_development_records():
    records = [{"split": "DEVELOPMENT", "family_id": DEV_FAMILIES[0],
                "R": {"0": 0.0, "1": 0.5, "2": 1.0}},
               {"split": "DEVELOPMENT", "family_id": DEV_FAMILIES[1],
                "R": {"0": 0.0, "1": 0.5, "2": 1.0}}]
    metrics = promo.DevMetrics.from_records(records, stage="FL3")
    assert metrics.split == "DEVELOPMENT"
    assert metrics.macro_aulc == pytest.approx(0.5)
    assert metrics.slope > 0
    assert set(metrics.family_ids) == {DEV_FAMILIES[0], DEV_FAMILIES[1]}


# ---------------- FL3 -> extensions ----------------

def test_fl3_gate_needs_margin_stability_and_positive_slope():
    fl1 = dev("FL1", 0.50)
    assert promo.promote_fl3_to_extensions(dev("FL3", 0.52), fl1).admitted
    assert not promo.promote_fl3_to_extensions(dev("FL3", 0.5199), fl1).admitted
    assert not promo.promote_fl3_to_extensions(
        dev("FL3", 0.60, slope=0.0), fl1).admitted
    assert not promo.promote_fl3_to_extensions(
        dev("FL3", 0.60, events=("GRAD_EXPLOSION",)), fl1).admitted


def test_fl3_gate_refuses_missing_slope_evidence():
    with pytest.raises(promo.PromotionError):
        promo.promote_fl3_to_extensions(
            promo.DevMetrics(stage="FL3", macro_aulc=0.9, slope=None),
            dev("FL1", 0.1))


def test_fl3_gate_refuses_swapped_arms():
    with pytest.raises(promo.PromotionError):
        promo.promote_fl3_to_extensions(dev("FL1", 0.9), dev("FL3", 0.1))


def test_a_refused_gate_carries_the_predeclared_fallback():
    decision = promo.promote_fl3_to_extensions(dev("FL3", 0.5), dev("FL1", 0.5))
    assert decision.admitted is False
    assert any("second predeclared seed" in f for f in decision.fallback)
    assert any("PREDECLARED work only" in f for f in decision.fallback)


# ---------------- FL4 .. FL8 ----------------

def test_fl4_data_sufficiency_boundary():
    assert promo.promote_fl4(fl3_finished=True,
                             scoreable_items_per_episode=[2] * 200).admitted
    assert not promo.promote_fl4(fl3_finished=True,
                                 scoreable_items_per_episode=[2] * 199).admitted
    assert not promo.promote_fl4(fl3_finished=True,
                                 scoreable_items_per_episode=[1] * 500).admitted
    assert not promo.promote_fl4(fl3_finished=False,
                                 scoreable_items_per_episode=[2] * 500).admitted


def test_fl5_needs_a_complete_core_comparison_and_admission():
    assert promo.promote_fl5(core_comparison_complete=True,
                             scheduler_admits=True).admitted
    assert not promo.promote_fl5(core_comparison_complete=False,
                                 scheduler_admits=True).admitted
    assert not promo.promote_fl5(core_comparison_complete=True,
                                 scheduler_admits=False).admitted


def test_fl6_threshold_is_055():
    assert promo.promote_fl6(pairwise_ranking_accuracy=0.55).admitted
    assert not promo.promote_fl6(pairwise_ranking_accuracy=0.5499).admitted
    with pytest.raises(promo.PromotionError):
        promo.promote_fl6()


def test_fl7_is_independently_predeclared_but_skips_the_orphan_gated_variant():
    assert promo.promote_fl7(scheduler_admits=True, fast_update_stable=True,
                             fl4_succeeded=False).admitted
    assert not promo.promote_fl7(scheduler_admits=True, fast_update_stable=False,
                                 fl4_succeeded=True).admitted
    assert not promo.promote_fl7(
        scheduler_admits=True, fast_update_stable=True, fl4_succeeded=False,
        only_remaining_variant_is_fl6_gated=True).admitted


def test_fl8_needs_persistence_with_a_ci_excluding_zero():
    assert promo.promote_fl8(persistence_point=0.2,
                             persistence_ci=[0.05, 0.4]).admitted
    assert not promo.promote_fl8(persistence_point=0.2,
                                 persistence_ci=[-0.01, 0.4]).admitted
    assert not promo.promote_fl8(persistence_point=0.0,
                                 persistence_ci=[0.0, 0.4]).admitted
    with pytest.raises(promo.SealedEvidenceRefused):
        promo.promote_fl8(persistence_point=1.0, persistence_ci=[0.5, 1.5],
                          split="SEALED_TEST")


# ---------------- §15 best-DEV checkpoint ----------------

def test_best_dev_checkpoint_picks_max_macro_aulc_with_earlier_tie_break():
    choice = promo.select_best_dev_checkpoint([
        ("step200", dev("FL3", 0.4)),
        ("step400", dev("FL3", 0.7)),
        ("step600", dev("FL3", 0.7)),
    ])
    assert choice["tag"] == "step400"
    assert choice["sealed_consulted"] is False


def test_best_dev_checkpoint_refuses_non_devmetrics_evidence():
    with pytest.raises(promo.PromotionError):
        promo.select_best_dev_checkpoint([("step200", {"macro_aulc": 0.9})])
