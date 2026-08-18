"""HOSTILE: instance-level generalization must never be reported as family-level.

Attack: describe UNSEEN-INSTANCE results (fresh instances of a generator the
model trained on) as whole-family transfer.  Contract §9 requires the two
scopes to be reported SEPARATELY and instance-level generalization never to be
described as transferable learning.

The fixture attacks the report schema three ways: dropping the family-level
section, writing instance-level families into the family-level section, and
mislabelling a record's scope.  All three must fail loudly.
"""

from __future__ import annotations

import pytest

from foundation_learner.evaluation import family_holdout as FH
from foundation_learner.evaluation import metrics as M
from foundation_learner.tests import _w4_support as S

TRAINED = ["grammar_classification", "set_operations", "dsl_execution"]
HELD_OUT = ["boolean_rule", "sequence_transform"]


def _mixed_records():
    return (S.constant_records("grammar_classification", 4, 0.9)
            + S.constant_records("set_operations", 4, 0.85)
            + S.constant_records("boolean_rule", 4, 0.2)
            + S.constant_records("sequence_transform", 4, 0.1))


def test_the_report_cannot_omit_either_scope():
    report = FH.build_family_holdout_report(_mixed_records(), TRAINED)
    for section in FH.REQUIRED_SECTIONS:
        broken = {k: v for k, v in report.items() if k != section}
        with pytest.raises(FH.ReportSchemaError):
            FH.validate_family_holdout_report(broken, TRAINED)


def test_instance_level_results_cannot_be_written_into_family_level_fields():
    report = dict(FH.build_family_holdout_report(_mixed_records(), TRAINED))
    instance_section = report[FH.UNSEEN_INSTANCE]
    # the attack: copy the (much better) instance-level numbers into the
    # family-level section and relabel them
    forged = dict(instance_section)
    forged["scope"] = FH.UNSEEN_FAMILY
    report[FH.UNSEEN_FAMILY] = forged
    with pytest.raises(FH.ReportSchemaError):
        FH.validate_family_holdout_report(report, TRAINED)


def test_a_trained_family_cannot_be_declared_unseen():
    records = S.constant_records("grammar_classification", 2, 0.9)
    for record in records:
        record["generalization_scope"] = FH.UNSEEN_FAMILY
    with pytest.raises(FH.ScopeConflictError):
        FH.assign_scopes(records, TRAINED)


def test_a_held_out_family_cannot_be_declared_instance_level():
    records = S.constant_records("boolean_rule", 2, 0.2)
    for record in records:
        record["generalization_scope"] = FH.UNSEEN_INSTANCE
    with pytest.raises(FH.ScopeConflictError):
        FH.assign_scopes(records, TRAINED)


def test_the_two_scopes_really_carry_different_numbers():
    """A conflated report would show the same macro-AULC in both sections."""
    report = FH.build_family_holdout_report(_mixed_records(), TRAINED)
    instance = report[FH.UNSEEN_INSTANCE]["macro_aulc"]
    family = report[FH.UNSEEN_FAMILY]["macro_aulc"]
    assert instance == pytest.approx(0.875)
    assert family == pytest.approx(0.15)
    assert instance != family
    assert set(report[FH.UNSEEN_INSTANCE]["family_ids"]) <= set(TRAINED)
    assert set(report[FH.UNSEEN_FAMILY]["family_ids"]) == set(HELD_OUT)


def test_whole_family_transfer_metric_excludes_trained_generators():
    records = _mixed_records()
    assert M.whole_family_transfer(records, TRAINED) == pytest.approx(0.15)
    # pooling everything would give a much rosier number; the metric must not
    assert M.macro_aulc(records) == pytest.approx((0.9 + 0.85 + 0.2 + 0.1) / 4)


def test_a_forged_episode_count_is_caught_against_the_records():
    records = FH.assign_scopes(_mixed_records(), TRAINED)
    report = dict(FH.build_family_holdout_report(_mixed_records(), TRAINED))
    forged = dict(report[FH.UNSEEN_FAMILY])
    forged["n_episodes"] = report[FH.UNSEEN_INSTANCE]["n_episodes"] * 10
    report[FH.UNSEEN_FAMILY] = forged
    with pytest.raises(FH.ReportSchemaError):
        FH.validate_family_holdout_report(report, TRAINED, records=records)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
