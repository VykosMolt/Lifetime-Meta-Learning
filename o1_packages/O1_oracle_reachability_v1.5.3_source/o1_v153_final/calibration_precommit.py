"""Create CALIBRATION_PRECOMMIT.json before any calibration continuation exists."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from calibration_analysis import (
    CALIBRATION_PRECOMMIT_SCHEMA_VERSION,
    CalibrationError,
    calibration_generation_config,
)
from o1_analysis import sha256_file, sha256_tree
from build_run_provenance import REQUIRED_ARTIFACT_KEYS


def _load(path: str) -> dict:
    with open(path) as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise CalibrationError("C0_PRECOMMIT", f"{path} must contain a JSON object")
    return obj


def _dget(raw: dict, dotted: str):
    node = raw
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise CalibrationError("C0_PRECOMMIT", f"missing manifest key {dotted}")
        node = node[part]
    return node


def create_calibration_precommit(
    output_path: str,
    manifest_design_path: str,
    artifact_paths_path: str,
    calibration_task_manifest_path: str,
    repository_diff_path: str | None = None,
    records_path: str | None = None,
) -> dict:
    if os.path.exists(output_path):
        raise CalibrationError("C0_PRECOMMIT", f"refusing to overwrite {output_path}")
    if records_path and os.path.exists(records_path) and os.path.getsize(records_path) > 0:
        raise CalibrationError(
            "C0_PRECOMMIT",
            "calibration records already exist; precommit must be created first")
    manifest = _load(manifest_design_path)
    paths = _load(artifact_paths_path)
    missing = [k for k in REQUIRED_ARTIFACT_KEYS if k not in paths]
    extra = sorted(set(paths) - set(REQUIRED_ARTIFACT_KEYS))
    if missing or extra:
        raise CalibrationError("C0_PRECOMMIT",
                               f"artifact map mismatch: missing={missing}, extra={extra}")
    for key in REQUIRED_ARTIFACT_KEYS:
        path = paths[key]
        if not isinstance(path, str) or not os.path.exists(path):
            raise CalibrationError("C0_PRECOMMIT", f"{key}: missing artifact {path!r}")
        got = sha256_tree(path)
        want = _dget(manifest, key)
        if got != want:
            raise CalibrationError(
                "C0_PRECOMMIT",
                f"{key}: actual artifact hash {got[:16]}... != sealed {str(want)[:16]}...")

    if repository_diff_path is None:
        diff = b""
    else:
        with open(repository_diff_path, "rb") as fh:
            diff = fh.read()
    diff_hash = hashlib.sha256(diff).hexdigest()
    want_diff = _dget(manifest, "code.repository_diff_sha256")
    if diff_hash != want_diff:
        raise CalibrationError(
            "C0_PRECOMMIT",
            f"repository diff hash {diff_hash[:16]}... != sealed {str(want_diff)[:16]}...")
    clean = bool(_dget(manifest, "code.repository_clean_required"))
    if clean and diff:
        raise CalibrationError("C0_PRECOMMIT",
                               "manifest requires a clean repository but diff is non-empty")

    want_task = _dget(manifest, "cohorts.calibration_task_manifest_sha256")
    got_task = sha256_file(calibration_task_manifest_path)
    if got_task != want_task:
        raise CalibrationError("C0_PRECOMMIT",
                               "calibration task-manifest hash disagrees")

    payload = {
        "schema_version": CALIBRATION_PRECOMMIT_SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resolved_calibration_config": calibration_generation_config(manifest),
        "runtime_attestation": {
            "generation_module_sha256": _dget(manifest, "code.generation_module_sha256"),
            "repository_clean": clean,
            "repository_diff_sha256": diff_hash,
        },
        "calibration_task_manifest_sha256": got_task,
    }
    tmp = output_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, output_path)
    return {"output": output_path, "sha256": sha256_file(output_path),
            "created_at_utc": payload["created_at_utc"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--manifest-design", required=True)
    p.add_argument("--artifact-paths", required=True)
    p.add_argument("--calibration-task-manifest", required=True)
    p.add_argument("--repository-diff")
    p.add_argument("--records",
                   help="planned calibration JSONL path; must not exist or must be empty")
    a = p.parse_args()
    print(json.dumps(create_calibration_precommit(
        a.output, a.manifest_design, a.artifact_paths,
        a.calibration_task_manifest, a.repository_diff, a.records), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
