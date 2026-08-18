"""UNSEEN-INSTANCE vs UNSEEN-FAMILY reporting (contract §9)."""

from __future__ import annotations

import pytest

from foundation_learner.evaluation import family_holdout as FH
from foundation_learner.tests import _w4_support as S

TRAINED = ["grammar_classification", "set_operations"]


def _records():
    return (S.constant_records("grammar_classification", 2, 0.8)
            + S.constant_records("boolean_rule", 2, 0.3))


def test_scope_is_derived_from_the_family_split():
    assert FH.classify_scope("grammar_classification", TRAINED) == FH.UNSEEN_INSTANCE
    assert FH.classify_scope("boolean_rule", TRAINED) == FH.UNSEEN_FAMILY


def test_assign_scopes_stamps_every_record():
    scoped = FH.assign_scopes(_records(), TRAINED)
    seen = {r["family_id"]: r["generalization_scope"] for r in scoped}
    assert seen == {"grammar_classification": FH.UNSEEN_INSTANCE,
                    "boolean_rule": FH.UNSEEN_FAMILY}


def test_a_record_declaring_the_wrong_scope_is_refused():
    records = S.constant_records("boolean_rule", 1, 0.5)
    records[0]["generalization_scope"] = FH.UNSEEN_INSTANCE
    with pytest.raises(FH.ScopeConflictError):
        FH.assign_scopes(records, TRAINED)


def test_report_always_contains_both_sections():
    report = FH.build_family_holdout_report(_records(), TRAINED, arm_tag="FL3")
    assert report["schema"] == FH.FAMILY_HOLDOUT_REPORT_SCHEMA
    for key in FH.REQUIRED_SECTIONS:
        assert key in report
        assert report[key]["scope"] == key
    assert report[FH.UNSEEN_INSTANCE]["macro_aulc"] == pytest.approx(0.8)
    assert report[FH.UNSEEN_FAMILY]["macro_aulc"] == pytest.approx(0.3)
    assert report[FH.UNSEEN_INSTANCE]["family_ids"] == ["grammar_classification"]
    assert report[FH.UNSEEN_FAMILY]["family_ids"] == ["boolean_rule"]
    assert report["trained_family_ids"] == sorted(TRAINED)


def test_an_empty_section_is_reported_as_empty_not_omitted():
    only_trained = S.constant_records("set_operations", 3, 0.6)
    report = FH.build_family_holdout_report(only_trained, TRAINED)
    assert report[FH.UNSEEN_FAMILY]["n_episodes"] == 0
    assert report[FH.UNSEEN_FAMILY]["macro_aulc"] is None
    assert report[FH.UNSEEN_FAMILY]["family_ids"] == []
    FH.validate_family_holdout_report(report, TRAINED)


def test_validation_rejects_a_missing_section():
    report = FH.build_family_holdout_report(_records(), TRAINED)
    broken = {k: v for k, v in report.items() if k != FH.UNSEEN_FAMILY}
    with pytest.raises(FH.ReportSchemaError, match="unseen_family"):
        FH.validate_family_holdout_report(broken, TRAINED)


def test_validation_rejects_instance_results_placed_in_family_fields():
    report = dict(FH.build_family_holdout_report(_records(), TRAINED))
    forged = dict(report[FH.UNSEEN_FAMILY])
    forged["family_ids"] = ["grammar_classification"]   # a TRAINED generator
    forged["macro_aulc"] = 0.8
    report[FH.UNSEEN_FAMILY] = forged
    with pytest.raises(FH.ReportSchemaError, match="instance-level"):
        FH.validate_family_holdout_report(report, TRAINED)


def test_validation_rejects_a_family_appearing_in_both_sections():
    report = dict(FH.build_family_holdout_report(_records(), TRAINED))
    forged = dict(report[FH.UNSEEN_INSTANCE])
    forged["family_ids"] = ["grammar_classification", "boolean_rule"]
    report[FH.UNSEEN_INSTANCE] = forged
    with pytest.raises(FH.ReportSchemaError):
        FH.validate_family_holdout_report(report, TRAINED)


def test_validation_recomputes_the_numbers_from_the_records():
    records = FH.assign_scopes(_records(), TRAINED)
    report = dict(FH.build_family_holdout_report(_records(), TRAINED))
    forged = dict(report[FH.UNSEEN_FAMILY])
    forged["macro_aulc"] = 0.99
    report[FH.UNSEEN_FAMILY] = forged
    with pytest.raises(FH.ReportSchemaError, match="macro_aulc"):
        FH.validate_family_holdout_report(report, TRAINED, records=records)
    forged["macro_aulc"] = 0.3
    forged["n_episodes"] = 99
    with pytest.raises(FH.ReportSchemaError, match="episodes"):
        FH.validate_family_holdout_report(report, TRAINED, records=records)


def test_the_note_states_the_interpretation_boundary():
    report = FH.build_family_holdout_report(_records(), TRAINED)
    assert "not transferable learning" in report["note"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
