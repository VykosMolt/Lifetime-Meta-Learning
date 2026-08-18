"""HOSTILE: no single family may dominate an aggregate.

Attack: inflate one family's episode count (a cheap family, an over-sampled
shard, a family whose episodes are shorter and therefore faster to run) so
that the macro numbers are really that family's numbers.  Contract §9 requires
macro (family-level) reporting everywhere and explicitly forbids single-family
domination of aggregates.

Every aggregate in ``evaluation.metrics`` and ``analysis.stats`` must be
invariant to replicating one family's episodes.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundation_learner.analysis import stats as ST
from foundation_learner.evaluation import metrics as M
from foundation_learner.tests import _w4_support as S

CURVE_A = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0, 6: 1.0}
CURVE_B = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.2, 6: 0.2}


def _records(n_a: int, n_b: int):
    out = []
    for i in range(n_a):
        out.append(S.make_record(f"a{i}", "famA", CURVE_A))
    for i in range(n_b):
        out.append(S.make_record(f"b{i}", "famB", CURVE_B))
    return out


BALANCED = _records(3, 3)
DOMINATED = _records(30, 3)      # famA over-represented 10x
DOMINATED_B = _records(3, 30)    # famB over-represented 10x


def test_macro_aulc_is_invariant_to_a_ten_fold_family():
    base = M.macro_aulc(BALANCED)
    assert M.macro_aulc(DOMINATED) == pytest.approx(base)
    assert M.macro_aulc(DOMINATED_B) == pytest.approx(base)
    # the naive pooled mean WOULD move, which is what the macro rule prevents
    pooled_balanced = np.mean([M.record_aulc(r) for r in BALANCED])
    pooled_dominated = np.mean([M.record_aulc(r) for r in DOMINATED])
    assert abs(pooled_dominated - pooled_balanced) > 0.1


def test_every_curve_metric_is_invariant():
    for fn in (M.r_0, M.r_K, M.improvement_slope, M.related_task_transfer):
        base = fn(BALANCED)
        assert fn(DOMINATED) == pytest.approx(base), fn.__name__
        assert fn(DOMINATED_B) == pytest.approx(base), fn.__name__
    assert M.macro_curve(DOMINATED) == pytest.approx(M.macro_curve(BALANCED))


def test_interactions_to_threshold_is_invariant():
    base = M.interactions_to_threshold(BALANCED)
    other = M.interactions_to_threshold(DOMINATED)
    assert base["per_family"] == other["per_family"]
    assert base["macro_mean"] == pytest.approx(other["macro_mean"])


def test_delta_and_robustness_aggregates_are_invariant():
    baseline = _records(3, 3)
    for r in baseline:
        r["R"] = {k: 0.0 for k in r["R"]}
    base = M.delta_aulc(BALANCED, baseline)
    assert M.delta_aulc(DOMINATED, baseline) == pytest.approx(base)

    clean_balanced = {"clean": BALANCED, "corrupted": baseline}
    clean_dominated = {"clean": DOMINATED, "corrupted": baseline}
    assert (M.poison_robustness(clean_dominated)["corrupted_gap"]
            == pytest.approx(M.poison_robustness(clean_balanced)["corrupted_gap"]))


def test_context_reset_persistence_is_invariant():
    history = [S.make_record(f"h{i}", fam, curve)
               for fam, curve in (("famA", CURVE_A), ("famB", CURVE_B))
               for i in range(2)]
    reset = [S.make_record(r["episode_id"], r["family_id"],
                           {int(k): float(v) / 2 for k, v in r["R"].items()})
             for r in history]
    base = M.context_reset_persistence(reset, history)["macro_retained_fraction"]
    dominated_history = history + [S.make_record(f"x{i}", "famA", CURVE_A)
                                   for i in range(20)]
    dominated_reset = reset + [S.make_record(f"x{i}", "famA",
                                             {k: v / 2 for k, v in CURVE_A.items()})
                               for i in range(20)]
    other = M.context_reset_persistence(dominated_reset, dominated_history)
    assert other["macro_retained_fraction"] == pytest.approx(base)


def test_retention_interference_is_invariant_to_chain_counts():
    chains = ([{"family_a": "famA", "aulc_a1": 1.0, "aulc_a2": 0.5}] * 2
              + [{"family_a": "famB", "aulc_a1": 1.0, "aulc_a2": 1.0}] * 2)
    dominated = ([{"family_a": "famA", "aulc_a1": 1.0, "aulc_a2": 0.5}] * 40
                 + [{"family_a": "famB", "aulc_a1": 1.0, "aulc_a2": 1.0}] * 2)
    base = M.retention_interference(chains)["macro_retention_ratio"]
    assert M.retention_interference(dominated)["macro_retention_ratio"] == \
        pytest.approx(base)


def test_the_clustered_bootstrap_point_estimate_is_invariant():
    balanced = ST.clustered_sample_from_records(BALANCED)
    dominated = ST.clustered_sample_from_records(DOMINATED)
    assert dominated.macro_mean() == pytest.approx(balanced.macro_mean())
    ci = ST.clustered_bootstrap_ci(dominated, n_replicates=500, seed=1)
    assert ci["estimate"] == pytest.approx(balanced.macro_mean())
    # and the family means themselves are unchanged
    assert dominated.family_means()["famA"] == pytest.approx(
        balanced.family_means()["famA"])


def test_summarize_and_arm_summaries_are_invariant():
    base = M.summarize(BALANCED)
    other = M.summarize(DOMINATED)
    for key in ("macro_aulc", "r_0", "r_K", "improvement_slope",
                "related_task_transfer"):
        assert other[key] == pytest.approx(base[key]), key
    assert other["n_episodes"] != base["n_episodes"]  # the data really differs


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
