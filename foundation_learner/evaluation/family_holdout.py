"""UNSEEN-INSTANCE vs UNSEEN-FAMILY reporting (contract section 9).

Contract, verbatim: whole-family generalization "reports UNSEEN-INSTANCE
(known generator) and UNSEEN-FAMILY (held-out generator) SEPARATELY;
instance-level generalization is never described as transferable learning".

This module makes that separation structural rather than editorial:

* the scope of a record is DERIVED here from the family-split fact
  ``trained_family_ids``, never read from a caller-supplied label; a record
  that carries a contradicting ``generalization_scope`` raises
  :class:`ScopeConflictError`;
* the report ALWAYS contains both ``unseen_instance`` and ``unseen_family``
  sections, each with its own family list, episode count and metrics — an
  empty section is reported as empty, never omitted;
* :func:`validate_family_holdout_report` recomputes the sections from the
  embedded evidence, so a report whose family-level fields were filled with
  instance-level results fails validation.  The hostile fixture
  ``test_hostile_instance_vs_family_conflation.py`` attacks exactly this.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .learning_curve import annotate_records
from . import metrics as _metrics

FAMILY_HOLDOUT_REPORT_SCHEMA = "flb200.family_holdout_report.v1"

UNSEEN_INSTANCE = "unseen_instance"
UNSEEN_FAMILY = "unseen_family"
REQUIRED_SECTIONS: tuple[str, str] = (UNSEEN_INSTANCE, UNSEEN_FAMILY)


class ScopeConflictError(AssertionError):
    """A record's declared generalization scope contradicts the family split."""


class ReportSchemaError(AssertionError):
    """A family-holdout report is missing a required section or is inconsistent."""


def classify_scope(family_id: str, trained_family_ids: Iterable[str]) -> str:
    """``unseen_instance`` when the GENERATOR was trained on, else ``unseen_family``."""
    return UNSEEN_INSTANCE if str(family_id) in {str(f) for f in trained_family_ids} \
        else UNSEEN_FAMILY


def assign_scopes(records: Sequence[Mapping[str, Any]],
                  trained_family_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Derive and stamp ``generalization_scope`` on every record."""
    trained = {str(f) for f in trained_family_ids}
    out: list[dict[str, Any]] = []
    for rec in records:
        derived = classify_scope(_metrics.record_family(rec), trained)
        declared = rec.get("generalization_scope")
        if declared is not None and str(declared) != derived:
            raise ScopeConflictError(
                f"record {rec.get('episode_id')!r} declares scope {declared!r} but the "
                f"family split makes it {derived!r}")
        out.extend(annotate_records([rec], generalization_scope=derived))
    return out


def _section(records: Sequence[Mapping[str, Any]], scope: str,
             trained_family_ids: Iterable[str]) -> dict[str, Any]:
    subset = [r for r in records if r.get("generalization_scope") == scope]
    families = sorted({_metrics.record_family(r) for r in subset})
    section: dict[str, Any] = {
        "scope": scope,
        "family_ids": families,
        "n_families": len(families),
        "n_episodes": len(subset),
        "episode_ids": sorted(str(r.get("episode_id")) for r in subset),
        "macro_aulc": _metrics.macro_aulc(subset),
        "per_family_aulc": _metrics.per_family_aulc(subset),
        "macro_curve": {str(k): v for k, v in _metrics.macro_curve(subset).items()},
        "r_0": _metrics.r_0(subset),
        "r_K": _metrics.r_K(subset),
        "improvement_slope": _metrics.improvement_slope(subset),
        "related_task_transfer": _metrics.related_task_transfer(subset),
        "interactions_to_threshold": _metrics.interactions_to_threshold(subset),
    }
    return section


def build_family_holdout_report(
    records: Sequence[Mapping[str, Any]],
    trained_family_ids: Iterable[str],
    *,
    arm_tag: str = "unspecified",
    split: str | None = None,
) -> dict[str, Any]:
    """Build the two-section report; both sections are ALWAYS present."""
    trained = sorted({str(f) for f in trained_family_ids})
    scoped = assign_scopes(records, trained)
    report = {
        "schema": FAMILY_HOLDOUT_REPORT_SCHEMA,
        "arm_tag": arm_tag,
        "split": split,
        "trained_family_ids": trained,
        "n_episodes": len(scoped),
        UNSEEN_INSTANCE: _section(scoped, UNSEEN_INSTANCE, trained),
        UNSEEN_FAMILY: _section(scoped, UNSEEN_FAMILY, trained),
        "note": ("UNSEEN-INSTANCE is generalization to fresh instances of a "
                 "generator the model was trained on; it is not transferable "
                 "learning and is never reported as such."),
    }
    validate_family_holdout_report(report, trained, records=scoped)
    return report


def validate_family_holdout_report(
    report: Mapping[str, Any],
    trained_family_ids: Iterable[str],
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Fail loudly on a missing section or on cross-contaminated sections."""
    for key in REQUIRED_SECTIONS:
        if key not in report:
            raise ReportSchemaError(f"family-holdout report lacks the {key!r} section")
        section = report[key]
        for field in ("scope", "family_ids", "n_episodes", "macro_aulc"):
            if field not in section:
                raise ReportSchemaError(f"section {key!r} lacks {field!r}")
        if section.get("scope") != key:
            raise ReportSchemaError(
                f"section {key!r} declares scope {section.get('scope')!r}")
    trained = {str(f) for f in trained_family_ids}
    instance_families = set(report[UNSEEN_INSTANCE].get("family_ids", []))
    family_families = set(report[UNSEEN_FAMILY].get("family_ids", []))
    if instance_families & family_families:
        raise ReportSchemaError(
            f"families {sorted(instance_families & family_families)} appear in BOTH "
            "the instance-level and family-level sections")
    stray = instance_families - trained
    if stray:
        raise ReportSchemaError(
            f"families {sorted(stray)} were reported as UNSEEN-INSTANCE but their "
            "generator is not in the trained set")
    stray = family_families & trained
    if stray:
        raise ReportSchemaError(
            f"families {sorted(stray)} were reported as UNSEEN-FAMILY but their "
            "generator WAS trained on; instance-level results may not be written "
            "into family-level fields")
    if records is not None:
        for key in REQUIRED_SECTIONS:
            expected = [r for r in records if r.get("generalization_scope") == key]
            if int(report[key].get("n_episodes", -1)) != len(expected):
                raise ReportSchemaError(
                    f"section {key!r} claims {report[key].get('n_episodes')} episodes "
                    f"but {len(expected)} records carry that scope")
            recomputed = _metrics.macro_aulc(expected)
            claimed = report[key].get("macro_aulc")
            if recomputed is None or claimed is None:
                if recomputed is not claimed:
                    raise ReportSchemaError(
                        f"section {key!r} macro_aulc {claimed!r} does not match the "
                        f"recomputed {recomputed!r}")
            elif abs(float(claimed) - float(recomputed)) > 1e-12:
                raise ReportSchemaError(
                    f"section {key!r} macro_aulc {claimed} does not match the "
                    f"recomputed {recomputed}")


__all__ = [
    "FAMILY_HOLDOUT_REPORT_SCHEMA",
    "UNSEEN_INSTANCE",
    "UNSEEN_FAMILY",
    "REQUIRED_SECTIONS",
    "ScopeConflictError",
    "ReportSchemaError",
    "classify_scope",
    "assign_scopes",
    "build_family_holdout_report",
    "validate_family_holdout_report",
]
