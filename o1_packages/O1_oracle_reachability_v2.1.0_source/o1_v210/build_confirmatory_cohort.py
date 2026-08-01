#!/usr/bin/env python3
"""Materialize the confirmatory cohort deterministically after calibration.

v2.0 left this step as prose plus manual FREEZE_SLOTs — post-calibration
operator discretion over the stratum mix.  v2.1 removes the discretion: the
allocation is derived inside ``compute_calibration`` (sealed before
calibration), stored in CALIBRATION_RESULTS.json, and this command only
re-derives it independently via ``cohort_allocation`` and materializes the
task rows.  Any disagreement between the stored and re-derived allocation
aborts.

Inputs are all hash-bound: the sealed manifest binds the calibration results
and the candidate pool; the results bind the selected settings, G values and
allocation.  The output cohort keeps full task content so gate G22 can
recompute scoring, and is emitted in a canonical deterministic order
(hard_medium by sealed stratum rank, then medium).
"""
from __future__ import annotations

import argparse
import json
import os

from cohort_allocation import AllocationError, allocate, materialize
from o1_analysis import IntegrityError, load_jsonl, sha256_file

PROCEED = ("PROCEED_POWERED", "PROCEED_BUDGET_CONSTRAINED")


def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _mget(manifest: dict, dotted: str):
    node = manifest
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise IntegrityError("B0_COHORT_BUILD", f"manifest missing {dotted!r}")
        node = node[part]
    return node


def build_confirmatory_cohort(
    manifest_path: str,
    results_path: str,
    candidate_pool_path: str,
    output_path: str,
    report_path: str,
) -> dict:
    for p in (output_path, report_path):
        if os.path.exists(p):
            raise IntegrityError("B0_COHORT_BUILD", f"refusing to overwrite {p}")
    manifest = _load_json(manifest_path)
    if sha256_file(results_path) != _mget(manifest, "calibration.results_sha256"):
        raise IntegrityError(
            "B0_COHORT_BUILD",
            "CALIBRATION_RESULTS.json does not match the sealed manifest hash")
    if sha256_file(candidate_pool_path) != _mget(
            manifest, "cohorts.confirmatory_candidate_pool_sha256"):
        raise IntegrityError(
            "B0_COHORT_BUILD",
            "candidate pool does not match the sealed manifest hash")
    results = _load_json(results_path)
    verdict = results.get("verdict")
    if verdict not in PROCEED:
        raise IntegrityError(
            "B0_COHORT_BUILD",
            f"calibration verdict {verdict!r} does not authorize a confirmatory "
            f"cohort")
    stored = results.get("confirmatory_allocation")
    if not isinstance(stored, dict):
        raise IntegrityError(
            "B0_COHORT_BUILD", "calibration results carry no confirmatory_allocation")

    selected = results["difficulty"]["selected_settings_per_stratum"]
    eligible = {s for vals in selected.values() for s in vals}
    pool = load_jsonl(candidate_pool_path)
    pool_counts: dict[str, int] = {}
    eligible_pool = []
    for row in pool:
        s = row.get("generator_setting_id")
        if s in eligible:
            pool_counts[s] = pool_counts.get(s, 0) + 1
            eligible_pool.append(row)

    G_confirm = int(results["power"]["G_confirm"])
    try:
        rederived = allocate(selected, pool_counts, G_confirm)
    except AllocationError as exc:
        raise IntegrityError(
            "B1_ALLOCATION_MISMATCH",
            f"stored calibration allocation cannot be re-derived: {exc}")
    # Diagnostics-consistency: the copied status fields must agree with the
    # stored constrained_by (red-team finding: previously copied unchecked).
    stored_cb = results["power"].get("constrained_by")
    want_status = ("POOL_CAPACITY_CONSTRAINED"
                   if isinstance(stored_cb, list) and "pool" in stored_cb
                   else "POOL_CAPACITY_OK")
    if stored.get("pool_capacity_status") != want_status:
        raise IntegrityError(
            "B1_ALLOCATION_MISMATCH",
            f"stored pool_capacity_status {stored.get('pool_capacity_status')!r} "
            f"disagrees with constrained_by {stored_cb!r}")
    for key in ("expected_counts", "per_setting_quota", "settings_per_stratum",
                "G_confirm"):
        if rederived[key] != stored.get(key):
            raise IntegrityError(
                "B1_ALLOCATION_MISMATCH",
                f"re-derived allocation {key} disagrees with the stored "
                f"calibration allocation",
                detail={"key": key, "stored": stored.get(key),
                        "rederived": rederived[key]})
    if int(stored.get("G_pool", -1)) != sum(pool_counts.values()):
        raise IntegrityError(
            "B1_ALLOCATION_MISMATCH",
            f"stored G_pool {stored.get('G_pool')} != recomputed eligible pool "
            f"{sum(pool_counts.values())}")

    try:
        rows = materialize(eligible_pool, rederived)
    except AllocationError as exc:
        raise IntegrityError(
            "B1_ALLOCATION_MISMATCH",
            f"cohort materialization failed against the sealed pool: {exc}")
    with open(output_path, "x", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    cohort_sha = sha256_file(output_path)
    report = {
        "schema_version": "o1.cohort_derivation.v2.1",
        "calibration_results_sha256": sha256_file(results_path),
        "candidate_pool_sha256": sha256_file(candidate_pool_path),
        "manifest_sha256": sha256_file(manifest_path),
        "verdict": verdict,
        "G_power": results["power"]["G_power"],
        "G_max": results["power"]["G_max"],
        "G_pool": results["power"]["G_pool"],
        "G_confirm": G_confirm,
        "constrained_by": results["power"]["constrained_by"],
        "pool_capacity_status": stored.get("pool_capacity_status"),
        "expected_counts": rederived["expected_counts"],
        "per_setting_quota": rederived["per_setting_quota"],
        "settings_per_stratum": rederived["settings_per_stratum"],
        "confirmatory_task_manifest": os.path.basename(output_path),
        "confirmatory_task_manifest_sha256": cohort_sha,
        "n_rows": len(rows),
        "selection_uses_model_outcomes": False,
    }
    with open(report_path, "x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="sealed post-calibration FREEZE_MANIFEST.json")
    p.add_argument("--calibration-results", required=True)
    p.add_argument("--candidate-pool", required=True)
    p.add_argument("--output", required=True,
                   help="confirmatory_tasks.jsonl (refuses overwrite)")
    p.add_argument("--report", required=True,
                   help="COHORT_DERIVATION_REPORT.json (refuses overwrite)")
    a = p.parse_args()
    print(json.dumps(build_confirmatory_cohort(
        a.manifest, a.calibration_results, a.candidate_pool, a.output, a.report,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
