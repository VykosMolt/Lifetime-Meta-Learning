"""Clustered bootstrap and rank statistics (contract §14). numpy only."""

from __future__ import annotations

import numpy as np
import pytest

from foundation_learner.analysis import stats as ST
from foundation_learner.tests import _w4_support as S


def _sample(sizes: dict[str, int], rng, *, family_effect: float = 0.0,
            noise: float = 0.1) -> ST.ClusteredSample:
    families = tuple(sorted(sizes))
    values = []
    ids = []
    for i, fid in enumerate(families):
        centre = 0.5 + family_effect * (i - (len(families) - 1) / 2)
        values.append(np.clip(centre + noise * rng.standard_normal(sizes[fid]), 0, 1))
        ids.append(tuple(f"{fid}-{j}" for j in range(sizes[fid])))
    return ST.ClusteredSample(families, tuple(values), tuple(ids))


def test_no_scipy_anywhere_in_the_analysis_layer():
    import ast

    from foundation_learner.analysis import report as RP

    for module in (ST, RP):
        tree = ast.parse(open(module.__file__, encoding="utf-8").read())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        assert not any(n.startswith("scipy") for n in names), names


# -- sample construction -------------------------------------------------
def test_clustered_sample_from_records_groups_by_family():
    records = (S.constant_records("famA", 3, 0.5)
               + S.constant_records("famB", 2, 1.0))
    sample = ST.clustered_sample_from_records(records)
    assert sample.family_ids == ("famA", "famB")
    assert sample.sizes == (3, 2)
    assert sample.n_episodes == 5
    assert sample.family_means() == {"famA": 0.5, "famB": 1.0}
    assert sample.macro_mean() == pytest.approx(0.75)


# -- determinism ---------------------------------------------------------
def test_bootstrap_is_deterministic_given_a_seed():
    rng = np.random.Generator(np.random.PCG64(1))
    sample = _sample({"a": 20, "b": 30, "c": 25}, rng)
    first = ST.clustered_bootstrap_ci(sample, n_replicates=500, seed=7)
    second = ST.clustered_bootstrap_ci(sample, n_replicates=500, seed=7)
    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]
    assert first["plan_digest"] == second["plan_digest"]
    other = ST.clustered_bootstrap_ci(sample, n_replicates=500, seed=8)
    assert other["plan_digest"] != first["plan_digest"]


def test_plan_indices_are_drawn_once_and_shared_across_arms():
    rng = np.random.Generator(np.random.PCG64(2))
    sample_a = _sample({"a": 15, "b": 15}, rng)
    sample_b = ST.ClusteredSample(sample_a.family_ids,
                                  tuple(v + 0.1 for v in sample_a.values),
                                  sample_a.episode_ids)
    plan = ST.make_resample_plan(sample_a, n_replicates=400, seed=3)
    diff = plan.macro_replicates(sample_a) - plan.macro_replicates(sample_b)
    # a constant shift must produce a degenerate (zero-variance) difference
    assert float(diff.std()) == pytest.approx(0.0, abs=1e-12)
    paired = ST.paired_clustered_bootstrap(sample_a, sample_b, plan=plan)
    assert paired["estimate"] == pytest.approx(-0.1, abs=1e-9)
    assert paired["ci_low"] == pytest.approx(-0.1, abs=1e-9)
    assert paired["excludes_zero"] is True


def test_paired_bootstrap_requires_matched_structure():
    rng = np.random.Generator(np.random.PCG64(4))
    a = _sample({"a": 10, "b": 10}, rng)
    b = _sample({"a": 10, "c": 10}, rng)
    with pytest.raises(ST.ClusteredSampleError):
        ST.paired_clustered_bootstrap(a, b, n_replicates=50)
    c = _sample({"a": 10, "b": 11}, rng)
    with pytest.raises(ST.ClusteredSampleError):
        ST.paired_clustered_bootstrap(a, c, n_replicates=50)
    shuffled = ST.ClusteredSample(a.family_ids, a.values,
                                  tuple(tuple(reversed(ids)) for ids in a.episode_ids))
    with pytest.raises(ST.ClusteredSampleError):
        ST.paired_clustered_bootstrap(a, shuffled, n_replicates=50)


def test_plan_refuses_incompatible_samples_and_empty_input():
    rng = np.random.Generator(np.random.PCG64(5))
    sample = _sample({"a": 5, "b": 5}, rng)
    plan = ST.make_resample_plan(sample, n_replicates=20, seed=1)
    other = _sample({"a": 5, "b": 6}, rng)
    with pytest.raises(ST.ClusteredSampleError):
        plan.macro_replicates(other)
    with pytest.raises(ST.ClusteredSampleError):
        ST.make_resample_plan(ST.ClusteredSample((), (), ()), n_replicates=10)
    with pytest.raises(ST.ClusteredSampleError):
        ST.make_resample_plan(sample, n_replicates=10, max_plan_entries=5)


# -- statistical behaviour ----------------------------------------------
def test_clustered_interval_covers_the_truth_on_synthetic_data():
    """Coverage sanity: the interval must contain the generating macro mean
    in the large majority of independent trials."""
    rng = np.random.Generator(np.random.PCG64(20260809))
    truth = 0.5
    covered = 0
    trials = 120
    for t in range(trials):
        sample = _sample({f"f{i}": 40 for i in range(8)}, rng, noise=0.15)
        out = ST.clustered_bootstrap_ci(sample, n_replicates=400, seed=t)
        if out["ci_low"] <= truth <= out["ci_high"]:
            covered += 1
    assert covered / trials >= 0.85, covered / trials


def test_clustered_intervals_are_wider_than_naive_ones_under_family_effects():
    """With family-correlated data the naive i.i.d. bootstrap understates
    uncertainty; the clustered one must be wider."""
    rng = np.random.Generator(np.random.PCG64(99))
    sample = _sample({f"f{i}": 60 for i in range(6)}, rng,
                     family_effect=0.12, noise=0.02)
    clustered = ST.clustered_bootstrap_ci(sample, n_replicates=2000, seed=1)
    naive = ST.naive_bootstrap_ci(sample, n_replicates=2000, seed=1)
    width_c = clustered["ci_high"] - clustered["ci_low"]
    width_n = naive["ci_high"] - naive["ci_low"]
    assert width_c > 2.0 * width_n, (width_c, width_n)
    assert clustered["clustered"] is True and naive["clustered"] is False


# -- curves, retention, robustness --------------------------------------
def test_learning_curve_with_ci_reports_every_index():
    records = (S.constant_records("famA", 4, 0.5)
               + S.constant_records("famB", 4, 0.25))
    out = ST.learning_curve_with_ci(records, n_replicates=200, seed=2)
    assert out["indices"] == list(range(7))
    point = out["points"]["3"]
    assert point["estimate"] == pytest.approx(0.375)
    assert point["ci_low"] <= point["estimate"] <= point["ci_high"]


def test_arm_comparison_reports_a_paired_delta():
    treated = S.constant_records("famA", 5, 0.8) + S.constant_records("famB", 5, 0.6)
    control = S.constant_records("famA", 5, 0.4) + S.constant_records("famB", 5, 0.2)
    out = ST.arm_comparison(treated, control, label_a="FL3", label_b="FL1",
                            n_replicates=200, seed=1)
    assert out["schema"] == ST.ARM_COMPARISON_SCHEMA
    assert out["estimate"] == pytest.approx(0.4)
    assert out["label_a"] == "FL3" and out["label_b"] == "FL1"


def test_retention_and_robustness_summaries():
    chains = [{"family_a": "famA", "chain_id": f"c{i}", "retention_ratio": 0.5}
              for i in range(6)]
    chains += [{"family_a": "famB", "chain_id": f"d{i}", "retention_ratio": 0.9}
               for i in range(6)]
    out = ST.retention_summary(chains, n_replicates=200, seed=1)
    assert out["estimate"] == pytest.approx(0.7)

    by_condition = {
        "clean": S.constant_records("famA", 4, 0.8),
        "corrupted": S.constant_records("famA", 4, 0.3, prefix="c"),
    }
    rob = ST.robustness_summary(by_condition, "clean", n_replicates=200, seed=1)
    assert rob["conditions"]["corrupted"]["paired"] is True
    assert rob["conditions"]["corrupted"]["paired_gap"]["estimate"] == \
        pytest.approx(0.5)
    with pytest.raises(ST.ClusteredSampleError):
        ST.robustness_summary(by_condition, "absent-condition")


# -- rank statistics -----------------------------------------------------
def test_rankdata_average_handles_ties():
    ranks = ST.rankdata_average([10.0, 20.0, 20.0, 40.0])
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_matches_hand_computation():
    assert ST.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert ST.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # ranks 1,2,3,4 vs 1,3,2,4 -> rho = 1 - 6*2/(4*15) = 0.8
    assert ST.spearman([1, 2, 3, 4], [10, 30, 20, 40]) == pytest.approx(0.8)
    assert ST.spearman([1], [1]) is None
    assert ST.spearman([1, 1, 1], [1, 2, 3]) is None


def test_pearson_and_calibration_fit():
    assert ST.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    fit = ST.calibration_fit([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["intercept"] == pytest.approx(1.0)
    assert ST.pearson([1, 2], [1]) is None


def test_pairwise_ranking_accuracy_uses_within_group_pairs():
    pred = [0.9, 0.1, 0.1, 0.9]
    targ = [1.0, 0.0, 1.0, 0.0]
    groups = ["e1", "e1", "e2", "e2"]
    out = ST.pairwise_ranking_accuracy(pred, targ, groups)
    assert out["n_pairs"] == 2 and out["n_groups"] == 2
    assert out["accuracy"] == pytest.approx(0.5)
    flat = ST.pairwise_ranking_accuracy([1.0, 1.0], [0.0, 1.0])
    assert flat["accuracy"] == pytest.approx(0.5)  # prediction tie scores 0.5
    assert ST.pairwise_ranking_accuracy([1.0], [1.0])["accuracy"] is None
    with pytest.raises(ValueError):
        ST.pairwise_ranking_accuracy([1.0, 2.0], [1.0])


def test_top1_regret_against_oracle_and_random():
    pred = [0.1, 0.9, 0.5, 0.4]
    targ = [1.0, 0.0, 0.2, 0.8]
    groups = ["e1", "e1", "e2", "e2"]
    out = ST.top1_regret(pred, targ, groups)
    # e1: picks index 1 (target 0.0) vs best 1.0 -> regret 1.0
    # e2: picks index 2 (target 0.2) vs best 0.8 -> regret 0.6
    assert out["top1_regret"] == pytest.approx(0.8)
    assert out["top1_regret_random"] == pytest.approx((1.0 - 0.5 + 0.8 - 0.5) / 2)
    assert out["top1_regret_oracle"] == 0.0
    assert out["n_groups"] == 2


def test_value_head_metrics_bundle():
    out = ST.value_head_metrics([0.1, 0.9, 0.4, 0.6], [0.0, 1.0, 0.3, 0.7],
                                ["e1", "e1", "e2", "e2"])
    assert out["schema"] == ST.VALUE_METRICS_SCHEMA
    assert out["spearman"] == pytest.approx(1.0)
    assert out["pairwise"]["accuracy"] == pytest.approx(1.0)
    assert out["regret"]["top1_regret"] == pytest.approx(0.0)
    assert out["calibration"]["slope"] is not None


def test_summarize_arm_bundles_metrics_and_intervals():
    records = S.constant_records("famA", 3, 0.5) + S.constant_records("famB", 3, 0.9)
    out = ST.summarize_arm(records, label="FL0", n_replicates=200, seed=1)
    assert out["label"] == "FL0"
    assert out["metrics"]["macro_aulc"] == pytest.approx(0.7)
    assert out["macro_aulc_ci"]["estimate"] == pytest.approx(0.7)
    assert out["learning_curve"]["indices"] == list(range(7))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
