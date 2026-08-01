"""Command-line entry point for the sealed O1 primary analysis."""
from __future__ import annotations

import argparse
import json

from o1_analysis import run_primary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--confirmatory-task-manifest", required=True)
    p.add_argument("--calibration-task-manifest", required=True)
    p.add_argument("--calibration-precommit", required=True)
    p.add_argument("--seed-matrix", required=True)
    p.add_argument("--result-seal", required=True)
    p.add_argument("--run-provenance", required=True)
    p.add_argument("--pregeneration-seal", required=True)
    p.add_argument("--calibration-records", required=True)
    p.add_argument("--calibration-metadata", required=True)
    p.add_argument("--calibration-results", required=True)
    p.add_argument("--axis-package", required=True)
    p.add_argument("--candidate-pool", required=True)
    a = p.parse_args()
    out = run_primary(
        a.records, a.manifest, a.confirmatory_task_manifest,
        a.calibration_task_manifest, a.calibration_precommit, a.seed_matrix,
        a.result_seal, a.run_provenance, a.pregeneration_seal,
        a.calibration_records, a.calibration_metadata, a.calibration_results,
        a.axis_package, a.candidate_pool,
    )
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
