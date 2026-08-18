"""The 14 frozen metrics, with hand-computed values (contract §9)."""

from __future__ import annotations

import pytest

from foundation_learner.evaluation import metrics as M
from foundation_learner.tests import _w4_support as S


def _records():
    """Two families, hand-computable curves.

    family A: one episode  R = [0, 0, 1, 1, 1, 1, 1]        AULC = 5/7
    family B: two episodes R = [0, 0, 0, 0, 0, 0, 0] and
                               [0, 1, 1, 1, 1, 1, 1]        AULC = (0 + 6/7)/2
    """
    a = [S.make_record("a0", "famA", {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})]
    b = [S.make_record("b0", "famB", {k: 0 for k in range(7)}),
         S.make_record("b1", "famB", {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})]
    return a + b


# -- accessors -----------------------------------------------------------
def test_record_accessors_and_episode_aulc():
    rec = S.make_record("x", "famA", {0: 0.0, 1: 1.0})
    assert M.record_family(rec) == "famA"
    assert M.record_R(rec) == {0: 0.0, 1: 1.0}
    assert M.record_aulc(rec) == pytest.approx(0.5)
    assert M.record_aulc(rec, indices=[1]) == pytest.approx(1.0)
    assert M.record_aulc(S.make_record("y", "f", {})) is None
    with pytest.raises(KeyError):
        M.record_family({"R": {}})


# -- 1-3 -----------------------------------------------------------------
def test_macro_aulc_is_family_level_first():
    records = _records()
    fam = M.per_family_aulc(records)
    assert fam["famA"] == pytest.approx(5 / 7)
    assert fam["famB"] == pytest.approx((0.0 + 6 / 7) / 2)
    assert M.macro_aulc(records) == pytest.approx((5 / 7 + 3 / 7) / 2)
    assert M.macro_aulc([]) is None


def test_delta_aulc_against_a_baseline():
    treatment = _records()
    baseline = S.constant_records("famA", 1, 0.0) + S.constant_records("famB", 2, 0.0)
    assert M.delta_aulc(treatment, baseline) == pytest.approx(M.macro_aulc(treatment))
    assert M.delta_aulc(treatment, []) is None


# -- 4-6 -----------------------------------------------------------------
def test_curves_r0_rK_and_slope():
    records = _records()
    curve = M.macro_curve(records)
    assert curve[0] == pytest.approx(0.0)
    assert curve[1] == pytest.approx((0.0 + 0.5) / 2)
    assert curve[2] == pytest.approx((1.0 + 0.5) / 2)
    assert M.r_0(records) == pytest.approx(0.0)
    assert M.r_K(records) == pytest.approx(0.75)
    assert M.r_at(records, 4) == pytest.approx(0.75)
    slope = M.improvement_slope(records)
    assert slope is not None and slope > 0


def test_ols_slope_is_exact_on_a_known_line():
    assert M.ols_slope([0, 1, 2, 3], [1, 3, 5, 7]) == pytest.approx(2.0)
    assert M.ols_fit([0, 1, 2, 3], [1, 3, 5, 7]) == pytest.approx((2.0, 1.0))
    assert M.ols_slope([1], [1]) is None
    assert M.ols_slope([2, 2, 2], [1, 2, 3]) is None


def test_the_two_slope_conventions_coincide_on_a_shared_index_grid():
    records = _records()
    per_family = M.per_family_slopes(records)
    assert set(per_family) == {"famA", "famB"}
    assert M.slope_conventions_agree(records)
    assert M.improvement_slope(records) == pytest.approx(
        sum(per_family.values()) / 2)


# -- 7 -------------------------------------------------------------------
def test_interactions_to_threshold_reports_unidentifiable_families():
    records = _records()
    out = M.interactions_to_threshold(records, threshold=0.5)
    assert out["per_family"]["famA"] == 2      # first index with 1.0 >= 0.5
    assert out["per_family"]["famB"] == 1      # 0.5 >= 0.5
    assert out["n_identifiable"] == 2
    assert out["macro_mean"] == pytest.approx(1.5)
    hard = M.interactions_to_threshold(S.constant_records("famC", 2, 0.1))
    assert hard["per_family"]["famC"] is None
    assert hard["macro_mean"] is None and hard["n_identifiable"] == 0


# -- 8-9 -----------------------------------------------------------------
def test_related_task_transfer_and_whole_family_transfer():
    records = _records()
    assert M.related_task_transfer(records) == pytest.approx(0.75)
    assert M.whole_family_transfer(records, trained_family_ids=["famA"]) == \
        pytest.approx(3 / 7)
    assert M.whole_family_transfer(records, ["famA", "famB"]) is None


# -- 10 ------------------------------------------------------------------
def test_context_reset_persistence_is_a_retained_fraction():
    history = [S.make_record("e0", "famA", {0: 0.0, 4: 1.0, 5: 1.0, 6: 1.0})]
    reset = [S.make_record("e0", "famA", {0: 0.0, 4: 0.5, 5: 0.5, 6: 0.5})]
    out = M.context_reset_persistence(reset, history, reset_from_index=4)
    fam = out["per_family"]["famA"]
    assert fam["gain_history"] == pytest.approx(1.0)
    assert fam["gain_reset"] == pytest.approx(0.5)
    assert fam["retained_fraction"] == pytest.approx(0.5)
    assert out["macro_retained_fraction"] == pytest.approx(0.5)


def test_context_reset_persistence_is_undefined_without_an_in_context_gain():
    flat = [S.make_record("e0", "famA", {0: 0.4, 4: 0.4, 5: 0.4, 6: 0.4})]
    out = M.context_reset_persistence(flat, flat, reset_from_index=4)
    assert out["per_family"]["famA"]["retained_fraction"] is None
    assert out["macro_retained_fraction"] is None
    assert out["n_families_defined"] == 0 and out["n_families"] == 1


# -- 11 ------------------------------------------------------------------
def test_retention_interference_is_family_balanced():
    chains = [
        {"family_a": "famA", "aulc_a1": 1.0, "aulc_a2": 0.5, "aulc_b": 0.2,
         "recovery_interactions": 2},
        {"family_a": "famA", "aulc_a1": 1.0, "aulc_a2": 0.5, "aulc_b": 0.2,
         "recovery_interactions": 4},
        {"family_a": "famB", "aulc_a1": 0.8, "aulc_a2": 0.8, "aulc_b": 0.1,
         "recovery_interactions": 0},
    ]
    out = M.retention_interference(chains)
    assert out["per_family"]["famA"]["retention_ratio"] == pytest.approx(0.5)
    assert out["per_family"]["famA"]["interference_cost"] == pytest.approx(0.5)
    assert out["per_family"]["famA"]["recovery_interactions"] == pytest.approx(3.0)
    assert out["per_family"]["famB"]["retention_ratio"] == pytest.approx(1.0)
    # two chains for famA and one for famB must not weight famA twice
    assert out["macro_retention_ratio"] == pytest.approx(0.75)
    assert out["macro_interference_cost"] == pytest.approx(0.25)


# -- 12-13 ---------------------------------------------------------------
def test_poison_condition_ids_follow_the_data_layer():
    assert M.CLEAN_CONDITION == "clean"
    assert M.canonical_poison_condition("correct-informative") == "clean"
    assert M.canonical_poison_condition("partially_misleading") == "partially-misleading"
    assert M.canonical_poison_condition("corrupted") == "corrupted"
    with pytest.raises(ValueError):
        M.canonical_poison_condition("mildly_confusing")
    if S.ecology_available():
        from foundation_learner.ecology.poison import POISON_CONDITIONS
        assert M.POISON_CONDITIONS == tuple(POISON_CONDITIONS)


def test_poison_robustness_gap_against_clean():
    by_condition = {
        "clean": S.constant_records("famA", 2, 0.8) + S.constant_records("famB", 2, 0.6),
        "corrupted": S.constant_records("famA", 2, 0.4) + S.constant_records("famB", 2, 0.2),
    }
    out = M.poison_robustness(by_condition)
    assert out["macro_aulc"]["clean"] == pytest.approx(0.7)
    assert out["corrupted_gap"] == pytest.approx(0.4)
    assert out["clean_condition"] == "clean"


def test_remap_robustness_gap_against_the_canonical_surface():
    canonical = S.constant_records("famA", 2, 0.9)
    variants = {"remap-1": S.constant_records("famA", 2, 0.5, prefix="r1"),
                "remap-2": S.constant_records("famA", 2, 0.7, prefix="r2")}
    out = M.remap_robustness(canonical, variants)
    assert out["canonical_macro_aulc"] == pytest.approx(0.9)
    assert out["gap_vs_canonical"]["remap-1"] == pytest.approx(0.4)
    assert out["macro_gap"] == pytest.approx(0.3)


# -- 14 + registry -------------------------------------------------------
def test_value_ranking_metrics_delegate_to_the_analysis_layer():
    out = M.value_ranking_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert out["spearman"] == pytest.approx(1.0)
    assert out["pairwise"]["accuracy"] == pytest.approx(1.0)


def test_the_frozen_registry_lists_the_fourteen_metrics_then_the_additions():
    """§9's fourteen metrics are unchanged and keep ids 1-14; Amendment 13's
    three ADDITIONS occupy 15-17 and Amendment 15's occupies 18; none of them
    replaces anything."""
    assert [e["id"] for e in M.METRICS_V0] == [str(i) for i in range(1, 19)]
    assert [e["name"] for e in M.METRICS_V0[:14]] == [
        "macro_aulc", "delta_aulc_vs_fl1", "delta_aulc_vs_fl2", "r_0", "r_K",
        "improvement_slope", "interactions_to_threshold",
        "related_task_transfer", "whole_family_transfer",
        "context_reset_persistence", "retention_interference",
        "poison_robustness", "remap_robustness", "value_ranking_metrics"]
    assert [e["name"] for e in M.METRICS_V0[14:]] == [
        "answer_line_rate", "aulc_post_feedback_fresh",
        "flip_attributable_success_rate", "constant_answer_baseline"]
    for entry in M.METRICS_V0:
        assert callable(M.metric_function(entry["name"]))
    with pytest.raises(KeyError):
        M.metric_function("not_a_metric")


# -- 15-17 + exclusion accounting (Amendment 13) -------------------------
def _attempt(index, item, number, parsed, correct,
             finish_reason="answer_line", role="MODEL_ATTEMPT",
             canonical_answer=None):
    row = {"role": role, "interaction_index": index, "item_id": item,
           "attempt_number": number, "parsed": parsed,
           "correct": bool(correct), "finish_reason": finish_reason}
    if canonical_answer is not None:
        row["canonical_answer"] = canonical_answer
    return row


def _episode_with_attempts(episode_id, family_id, attempts, **extra):
    R = {}
    for a in attempts:
        R.setdefault(a["interaction_index"], []).append(float(a["correct"]))
    curve = {k: sum(v) / len(v) for k, v in R.items()}
    return S.make_record(episode_id, family_id, curve,
                         attempts=attempts, **extra)


def test_answer_line_rate_is_macro_and_index_resolved():
    # famA: 2 of 4 attempts parse; famB: 4 of 4 parse.
    a = _episode_with_attempts("a0", "famA", [
        _attempt(0, "i1", 0, None, False, "max_new_tokens"),
        _attempt(1, "i1", 1, "YES", True),
        _attempt(2, "i2", 0, None, False, "max_new_tokens"),
        _attempt(3, "i2", 1, "NO", False)])
    b = _episode_with_attempts("b0", "famB", [
        _attempt(0, "j1", 0, "YES", False),
        _attempt(1, "j1", 1, "NO", True),
        _attempt(2, "j2", 0, "YES", False),
        _attempt(3, "j2", 1, "NO", True)])
    records = [a, b]
    assert M.record_answer_line_rate(a) == pytest.approx(0.5)
    assert M.answer_line_rate(records) == pytest.approx(0.75)
    by_index = M.answer_line_rate_by_index(records)
    assert by_index[0] == pytest.approx(0.5)   # famA 0, famB 1
    assert by_index[1] == pytest.approx(1.0)
    assert M.per_family_answer_line_rate(records)["famA"] == pytest.approx(0.5)
    assert M.finish_reason_counts(records)["max_new_tokens"] == 2


def test_format_flag_annotates_and_never_replaces_the_number():
    floored = [_episode_with_attempts("z0", "famZ", [
        _attempt(k, "i1", k, None, False, "max_new_tokens") for k in range(4)])]
    compliance = M.format_compliance(floored)
    assert compliance["macro_answer_line_rate"] == 0.0
    assert compliance["flag"] == M.FORMAT_NONCOMPLIANT_FLAG
    assert compliance["flagged_families"] == ["famZ"]
    assert compliance["flagged_indices"] == [0, 1, 2, 3]
    # the flag is additional information: the AULC itself is still computed
    assert M.macro_aulc(floored) == pytest.approx(0.0)
    assert M.summarize(floored)["macro_aulc"] == pytest.approx(0.0)
    assert M.summarize(floored)["format_flag"] == M.FORMAT_NONCOMPLIANT_FLAG
    assert M.format_flag(0.5) is None and M.format_flag(0.49) is not None
    assert M.format_flag(None) is None


def test_attempt_without_a_parse_field_is_refused_not_guessed():
    with pytest.raises(KeyError):
        M.attempt_has_answer_line({"role": "MODEL_ATTEMPT",
                                   "interaction_index": 0})


def test_aulc_post_feedback_fresh_uses_only_indices_4_5_6():
    assert M.POST_FEEDBACK_FRESH_INDICES == (4, 5, 6)
    records = _records()
    assert M.aulc_post_feedback_fresh(records) == pytest.approx(
        M.macro_aulc(records, (4, 5, 6)))
    # famA has R_4..R_6 = 1; famB has (0,0,0) and (1,1,1) -> 0.5
    assert M.aulc_post_feedback_fresh(records) == pytest.approx(0.75)
    fresh = M.per_family_aulc_post_feedback_fresh(records)
    assert fresh["famA"] == pytest.approx(1.0)
    assert fresh["famB"] == pytest.approx(0.5)


def test_flip_attributable_success_rate_isolates_the_flip_heuristic():
    # Two revisions: the first flips the label after INCORRECT (attributable),
    # the second was already correct at attempt0 and stayed (not attributable).
    record = _episode_with_attempts("g0", "graph_edge_semantics", [
        _attempt(0, "i1", 0, "NO", False),
        _attempt(1, "i1", 1, "YES", True),
        _attempt(2, "i2", 0, "YES", True),
        _attempt(3, "i2", 1, "YES", True)])
    out = M.flip_attributable_success_rate([record])
    fam = out["per_family"]["graph_edge_semantics"]
    assert fam["n_successes"] == 2
    assert fam["n_flip_attributable"] == 1
    assert fam["rate"] == pytest.approx(0.5)
    assert out["pooled_rate"] == pytest.approx(0.5)
    assert out["indices"] == [1, 3]
    # a non-binary family is never counted
    other = _episode_with_attempts("s0", "sequence_transform", [
        _attempt(0, "i1", 0, "a b", False),
        _attempt(1, "i1", 1, "b a", True)])
    assert M.flip_attributable_success_rate([other])["per_family"] == {}


def test_binary_answer_families_match_the_real_generators():
    """The pinned list may not drift from the ecology it describes."""
    from foundation_learner.ecology.families import all_families

    measured = set()
    for family in all_families():
        variants = getattr(family, "label_variants", ())
        if (getattr(family, "canon_mode", None) == "label" and variants
                and all(len(v) == 2 for v in variants)):
            measured.add(family.family_id)
    assert set(M.BINARY_ANSWER_FAMILIES) == measured


def test_constant_answer_floor_is_the_best_always_answer_x_policy():
    """Three of four items are UNSAT, so 'always answer UNSAT' scores 0.75 —
    and a family AULC of 0.75 is therefore worth nothing."""
    record = _episode_with_attempts("c0", "constraint_rules", [
        _attempt(0, "i1", 0, "SAT", False, canonical_answer="UNSAT"),
        _attempt(1, "i2", 0, "SAT", True, canonical_answer="SAT"),
        _attempt(2, "i3", 0, "UNSAT", True, canonical_answer="UNSAT"),
        _attempt(3, "i4", 0, "SAT", False, canonical_answer="UNSAT")])
    out = M.constant_answer_baseline([record])
    cell = out["per_family"]["constraint_rules"]
    assert cell["floor"] == pytest.approx(0.75)
    assert cell["best_constant_answer"] == "UNSAT"
    assert cell["n_distinct_answers"] == 2
    assert out["macro_floor_label_families"] == pytest.approx(0.75)

    above = M.aulc_above_constant_answer_floor([record])["constraint_rules"]
    assert above["constant_answer_floor"] == pytest.approx(0.75)
    assert above["above_floor"] == pytest.approx(
        above["macro_aulc"] - 0.75)


def test_open_answer_families_report_no_floor_and_say_why():
    record = _episode_with_attempts("s0", "sequence_transform", [
        _attempt(k, f"i{k}", 0, "x", False, canonical_answer=f"answer-{k}")
        for k in range(6)])
    cell = M.constant_answer_baseline([record])["per_family"]["sequence_transform"]
    assert cell["floor"] is None
    assert cell["n_distinct_answers"] == 6
    assert "open-answer" in cell["reason"]
    assert M.constant_answer_baseline([record])["n_families_defined"] == 0
    # and the delta column is undefined rather than silently zero
    assert M.aulc_above_constant_answer_floor(
        [record])["sequence_transform"]["above_floor"] is None


def test_records_without_canonical_answers_report_no_floor():
    plain = S.constant_records("famA", 2, 0.5)
    cell = M.constant_answer_baseline(plain)["per_family"]["famA"]
    assert cell["floor"] is None and cell["n_distinct_answers"] == 0
    assert M.summarize(plain)["constant_answer_baseline"]["n_families"] == 1


def test_excluded_records_are_counted_and_never_averaged():
    good = S.make_record("ok", "famA", {k: 1.0 for k in range(7)})
    dropped = S.make_record(
        "budget", "famA", {},
        status=M.STATUS_ONLINE_BUDGET_EXCEEDED,
        R_missing_reason=M.STATUS_ONLINE_BUDGET_EXCEEDED)
    records = [good, dropped]
    assert M.record_is_scoreable(good) is True
    assert M.record_is_scoreable(dropped) is False
    counts = M.excluded_records(records)
    assert counts["n_records"] == 2 and counts["n_excluded"] == 1
    assert counts["by_status"][M.STATUS_ONLINE_BUDGET_EXCEEDED] == 1
    assert counts["by_family"]["famA"] == 1
    assert counts["excluded_episode_ids"] == ["budget"]
    # the excluded episode changes no aggregate
    assert M.macro_aulc(records) == pytest.approx(M.macro_aulc([good]))
    assert M.summarize(records)["excluded_records"]["n_excluded"] == 1


def test_summarize_bundles_the_record_only_metrics():
    out = M.summarize(_records(), trained_family_ids=["famA"])
    assert out["schema"] == M.METRICS_SCHEMA
    assert out["n_families"] == 2 and out["n_episodes"] == 3
    assert out["macro_aulc"] == pytest.approx(M.macro_aulc(_records()))
    assert out["whole_family_transfer"] == pytest.approx(3 / 7)
    assert set(out["macro_curve"]) == {str(k) for k in range(7)}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
