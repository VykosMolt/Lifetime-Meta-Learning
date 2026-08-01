"""Validate the complete O1 run package and create PREGENERATION_SEAL.json.

This command must run before the confirmatory JSONL exists. It performs the
same load-bearing checks that run_primary will repeat later, then writes the
external commitment consumed by the analysis.
"""
from __future__ import annotations

import argparse
import json
import os

from calibration_analysis import verify_calibration_binding
from o1_analysis import (
    Manifest,
    create_pregeneration_seal,
    gate_G14_provenance,
    gate_G17_manifest_consistency,
    load_task_manifest,
    sha256_file,
    validate_seed_matrix,
)
from verify_axis_artifact import verify as verify_axis_artifact


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--confirmatory-task-manifest", required=True)
    p.add_argument("--calibration-task-manifest", required=True)
    p.add_argument("--calibration-precommit", required=True)
    p.add_argument("--seed-matrix", required=True)
    p.add_argument("--calibration-records", required=True)
    p.add_argument("--calibration-metadata", required=True)
    p.add_argument("--calibration-results", required=True)
    p.add_argument("--axis-package", required=True)
    p.add_argument("--records", required=True,
                   help="planned confirmatory JSONL path; must not exist or be empty")
    a = p.parse_args()

    man = Manifest(a.manifest)
    gate_G17_manifest_consistency(man)
    gate_G14_provenance(man, a.provenance)
    axis = verify_axis_artifact(a.axis_package)
    if axis.get("verdict") != "SEALABLE":
        raise SystemExit("axis package did not verify as SEALABLE")
    verify_calibration_binding(
        a.calibration_records, a.calibration_metadata, a.calibration_results,
        a.manifest, a.calibration_task_manifest, a.calibration_precommit,
    )
    confirmatory = load_task_manifest(a.confirmatory_task_manifest)
    calibration = load_task_manifest(a.calibration_task_manifest)
    # Explicit content disjointness before confirmation starts.
    shared = ({t["task_content_sha256"] for t in confirmatory}
              & {t["task_content_sha256"] for t in calibration})
    if shared:
        raise SystemExit(f"calibration/confirmatory content overlap: {len(shared)}")
    with open(a.seed_matrix) as fh:
        seed_matrix = json.load(fh)
    validate_seed_matrix(seed_matrix, confirmatory, man)

    seal = create_pregeneration_seal(
        a.output, a.manifest, a.provenance,
        a.confirmatory_task_manifest, a.calibration_task_manifest,
        a.calibration_precommit, a.seed_matrix, a.calibration_records,
        a.calibration_metadata, a.calibration_results, a.axis_package,
        records_path=a.records,
    )
    print(json.dumps({
        "seal": a.output,
        "sha256": sha256_file(a.output),
        "created_at_utc": seal["created_at_utc"],
        "axis_verdict": axis["verdict"],
        "preflight": "PASS",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
