"""Plain-markdown + JSON reporting over records (contract §14)."""

from __future__ import annotations

import json

import pytest

from foundation_learner.analysis import report as RP
from foundation_learner.tests import _w4_support as S


def _arms():
    return {
        "FL3": S.constant_records("famA", 3, 0.8) + S.constant_records("famB", 3, 0.6),
        "FL1": S.constant_records("famA", 3, 0.4)
               + S.constant_records("famB", 3, 0.2),
    }


def _report(**kwargs):
    return RP.build_report(_arms(), baselines=["FL1"], n_replicates=200, seed=1,
                           **kwargs)


def test_report_contains_every_arm_and_paired_comparison():
    report = _report()
    assert report["schema"] == RP.ANALYSIS_REPORT_SCHEMA
    assert set(report["arms"]) == {"FL3", "FL1"}
    assert report["arms"]["FL3"]["metrics"]["macro_aulc"] == pytest.approx(0.7)
    comparison = report["comparisons"]["FL3_vs_FL1"]
    assert comparison["estimate"] == pytest.approx(0.4)
    assert "FL1_vs_FL1" not in report["comparisons"]


def test_unpairable_arms_are_reported_as_unavailable_not_dropped():
    arms = {"FL3": S.constant_records("famA", 3, 0.8),
            "FL1": S.constant_records("famB", 2, 0.4)}
    report = RP.build_report(arms, baselines=["FL1"], n_replicates=50, seed=1)
    entry = report["comparisons"]["FL3_vs_FL1"]
    assert "unavailable_reason" in entry
    assert entry["label_a"] == "FL3"


def test_markdown_is_plain_and_states_missing_values_as_na():
    report = _report(
        context_reset={"summary": {"reset_macro_aulc": 0.2,
                                   "history_macro_aulc": 0.7,
                                   "persistence": {"macro_retained_fraction": None,
                                                   "n_families_defined": 0,
                                                   "n_families": 2}}},
        interference={"summary": {"macro_retention_ratio": 0.5,
                                  "macro_interference_cost": 0.25,
                                  "n_chains": 4, "n_families": 2}},
        poison={"robustness": {"macro_aulc": {"clean": 0.7, "corrupted": 0.3},
                               "gap_vs_clean": {"corrupted": 0.4}}},
        remap={"robustness": {"canonical_macro_aulc": 0.7,
                              "remapped_macro_aulc": {"remap-a": 0.5},
                              "gap_vs_canonical": {"remap-a": 0.2}}},
        value_head={"spearman": 0.42,
                    "pairwise": {"accuracy": 0.6, "n_pairs": 100},
                    "calibration": {"slope": 0.9, "intercept": 0.01},
                    "regret": {"top1_regret": 0.2, "top1_regret_random": 0.5}},
        family_holdout={"unseen_instance": {"n_families": 1, "n_episodes": 3,
                                            "macro_aulc": 0.8, "r_0": 0.8,
                                            "r_K": 0.8},
                        "unseen_family": {"n_families": 1, "n_episodes": 3,
                                          "macro_aulc": 0.6, "r_0": 0.6,
                                          "r_K": 0.6}},
        notes=["a null result is a result"])
    text = RP.render_markdown(report)
    assert text.startswith("# ")
    assert "| FL3 |" in text and "| FL1 |" in text
    assert "macro retained fraction of in-context gain: n/a" in text
    assert "A -> B -> A retention and interference" in text
    assert "unseen_family" in text
    assert "not transferable learning" in text
    assert "a null result is a result" in text
    assert "top-1 regret: 0.2000 (random 0.5000, oracle 0)" in text
    assert all(ord(ch) < 128 for ch in text)  # plain ASCII markdown


# -- Amendment 13 report additions ---------------------------------------
def _attempt(index, item, number, parsed, correct,
             finish_reason="answer_line"):
    return {"role": "MODEL_ATTEMPT", "interaction_index": index,
            "item_id": item, "attempt_number": number, "parsed": parsed,
            "correct": bool(correct), "finish_reason": finish_reason}


def _flip_records(family_id, n, *, parses=True, prefix=""):
    """Episodes whose revision successes are pure answer flips."""
    out = []
    for i in range(n):
        attempts = [
            _attempt(0, "i1", 0, "NO" if parses else None, False,
                     "answer_line" if parses else "max_new_tokens"),
            _attempt(1, "i1", 1, "YES" if parses else None, parses),
            _attempt(2, "i2", 0, "NO" if parses else None, False,
                     "answer_line" if parses else "max_new_tokens"),
            _attempt(3, "i2", 1, "YES" if parses else None, parses),
        ]
        curve = {a["interaction_index"]: float(a["correct"]) for a in attempts}
        curve.update({4: 0.0, 5: 0.0, 6: 0.0})
        out.append(S.make_record(f"{prefix}{family_id}-{i}", family_id, curve,
                                 attempts=attempts))
    return out


def test_comparisons_carry_the_two_frozen_restricted_companions():
    report = _report()
    entry = report["comparisons"]["FL3_vs_FL1"]
    restricted = entry["restricted"]
    assert set(restricted) == {"post_feedback_fresh", "excluding_index_0"}
    assert restricted["post_feedback_fresh"]["indices"] == [4, 5, 6]
    assert restricted["excluding_index_0"]["indices"] == [1, 2, 3, 4, 5, 6]
    # constant records: every restriction gives the same delta
    assert restricted["post_feedback_fresh"]["estimate"] == pytest.approx(0.4)
    assert restricted["excluding_index_0"]["estimate"] == pytest.approx(0.4)
    # the headline number is untouched
    assert entry["estimate"] == pytest.approx(0.4)


def test_restricted_companions_separate_flip_credit_from_learning():
    arms = {"FL3": _flip_records("graph_edge_semantics", 3)
                   + _flip_records("boolean_rule", 3),
            "FL1": _flip_records("graph_edge_semantics", 3, parses=False)
                   + _flip_records("boolean_rule", 3, parses=False)}
    report = RP.build_report(arms, baselines=["FL1"], n_replicates=100, seed=1)
    entry = report["comparisons"]["FL3_vs_FL1"]
    assert entry["estimate"] > 0.0                      # headline gain
    assert entry["restricted"]["post_feedback_fresh"]["estimate"] == \
        pytest.approx(0.0)                              # nothing on fresh items
    flip = report["arms"]["FL3"]["metrics"]["flip_attributable_success_rate"]
    assert flip["pooled_rate"] == pytest.approx(1.0)
    assert flip["n_successes"] == 12


def test_format_noncompliance_is_flagged_next_to_the_number():
    arms = {"FL0": _flip_records("boolean_rule", 3, parses=False)}
    report = RP.build_report(arms, n_replicates=50, seed=1)
    summary = report["arms"]["FL0"]
    assert summary["answer_line_rate"] == pytest.approx(0.0)
    assert summary["format_flag"] == "FORMAT_NONCOMPLIANT"
    text = RP.render_markdown(report)
    assert "FORMAT_NONCOMPLIANT" in text
    assert "Format compliance (answer-line rate)" in text
    assert "max_new_tokens=6" in text
    # the flag never replaces the macro-AULC number
    assert summary["macro_aulc_ci"]["estimate"] is not None
    assert "n/a **FORMAT_NONCOMPLIANT**" not in text


def test_every_family_result_is_printed_against_its_constant_answer_floor():
    """Amendment 15: a family AULC is unreadable without the score an
    'always answer X' policy would get on the same items."""
    attempts = [
        _attempt(0, "i1", 0, "UNSAT", True) | {"canonical_answer": "UNSAT"},
        _attempt(1, "i2", 0, "UNSAT", True) | {"canonical_answer": "UNSAT"},
        _attempt(2, "i3", 0, "UNSAT", False) | {"canonical_answer": "SAT"},
        _attempt(3, "i4", 0, "UNSAT", True) | {"canonical_answer": "UNSAT"},
    ]
    curve = {a["interaction_index"]: float(a["correct"]) for a in attempts}
    records = [S.make_record("c0", "constraint_rules", curve,
                             attempts=attempts)]
    report = RP.build_report({"FL0": records}, n_replicates=50, seed=1)
    cell = report["arms"]["FL0"]["metrics"][
        "aulc_above_constant_answer_floor"]["constraint_rules"]
    assert cell["constant_answer_floor"] == pytest.approx(0.75)
    assert cell["macro_aulc"] == pytest.approx(0.75)
    assert cell["above_floor"] == pytest.approx(0.0)
    text = RP.render_markdown(report)
    assert "Per-family results against the constant-answer floor" in text
    assert "AT-OR-BELOW-FLOOR" in text
    assert "always answer X" in text


def test_excluded_episodes_are_reported_in_the_markdown():
    from foundation_learner.evaluation import metrics as EM

    records = S.constant_records("famA", 2, 0.5) + [
        S.make_record("dropped", "famA", {},
                      status=EM.STATUS_ONLINE_BUDGET_EXCEEDED)]
    report = RP.build_report({"FL0": records}, n_replicates=50, seed=1)
    text = RP.render_markdown(report)
    assert "Episodes recorded but EXCLUDED from the metrics" in text
    assert "ONLINE_BUDGET_EXCEEDED=1" in text
    assert report["arms"]["FL0"]["excluded_records"]["n_excluded"] == 1


def test_fresh_index_aulc_is_reported_with_its_own_interval():
    report = _report()
    summary = report["arms"]["FL3"]
    assert summary["aulc_post_feedback_fresh_ci"]["estimate"] == \
        pytest.approx(0.7)
    text = RP.render_markdown(report)
    assert "macro-AULC on fresh {4,5,6}" in text
    assert "an answer-flip heuristic" in text


def test_write_report_emits_json_and_markdown(tmp_path):
    report = _report()
    paths = RP.write_report(str(tmp_path), report, stem="fl0")
    payload = json.loads(open(paths["json"], encoding="utf-8").read())
    assert payload["schema"] == RP.ANALYSIS_REPORT_SCHEMA
    assert open(paths["json"], encoding="utf-8").read().endswith("\n")
    assert open(paths["markdown"], encoding="utf-8").read().startswith("# ")


def test_report_from_record_files_round_trips(tmp_path):
    from foundation_learner.evaluation import write_jsonl_records

    path = str(tmp_path / "fl3.jsonl")
    write_jsonl_records(path, S.constant_records("famA", 2, 0.5))
    report = RP.report_from_record_files({"FL3": path}, n_replicates=50, seed=1)
    assert report["arms"]["FL3"]["metrics"]["n_episodes"] == 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
