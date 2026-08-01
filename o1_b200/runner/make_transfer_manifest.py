#!/usr/bin/env python3
"""Generate deploy/TRANSFER_MANIFEST.json — provider-neutral artifact list.

The model checkpoint is a 'reference' entry: hashed, transferred out of band,
never in Git and never in the archive.  No credentials of any kind exist in
the listed trees; the transfer tooling additionally enforces its deny-list.
"""
from __future__ import annotations

import json
import os

from .identity import sha256_file
from .transfer_package import build_manifest
from .runbuild import AXIS_PACKAGE_TREE_SHA256, WORKTREE_ROOT
from .sealed_import import BASE_PACKAGE_ZIP_SHA256

CHECKPOINT_PATH = "/home/moloch/ouro_project/models/ouro_rltt_local"
CHECKPOINT_TREE_SHA256 = (
    "a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889")


def main() -> int:
    root = WORKTREE_ROOT
    run_dir = os.path.join(root, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN")
    entries = {
        "o1_package_zip": {
            "path": os.path.join(root, "o1_packages",
                                 "O1_oracle_reachability_v2.1.0_verified.zip"),
            "kind": "file", "expected_sha256": BASE_PACKAGE_ZIP_SHA256},
        "o1_package_source": {
            "path": os.path.join(root, "o1_packages",
                                 "O1_oracle_reachability_v2.1.0_source"),
            "kind": "tree"},
        "ouro_rltt_checkpoint": {
            "path": CHECKPOINT_PATH, "kind": "reference",
            "expected_sha256": CHECKPOINT_TREE_SHA256},
        "tokenizer_binding": {
            "path": os.path.join(run_dir, "TOKENIZER_BINDING.json"),
            "kind": "file"},
        "axis_package": {
            "path": os.path.join(run_dir, "AXIS_PACKAGE_V2"),
            "kind": "tree", "expected_sha256": AXIS_PACKAGE_TREE_SHA256},
        "calibration_task_manifest": {
            "path": os.path.join(run_dir, "COHORTS", "calibration_tasks.jsonl"),
            "kind": "file"},
        "confirmatory_candidate_pool": {
            "path": os.path.join(run_dir, "COHORTS",
                                 "confirmatory_candidate_pool.jsonl"),
            "kind": "file"},
        "calibration_seed_matrix": {
            "path": os.path.join(run_dir, "calibration_seed_matrix.json"),
            "kind": "file"},
        "calibration_precommit": {
            "path": os.path.join(run_dir, "CALIBRATION_PRECOMMIT.json"),
            "kind": "file",
            "expected_sha256": "e819aeebc642fefde769f398b385f23f98dc3980412e"
                               "eb67274d0f5891c5457d"},
        "freeze_manifest": {
            "path": os.path.join(run_dir, "FREEZE_MANIFEST.precalibration.json"),
            "kind": "file"},
        "validation_corpus": {
            "path": os.path.join(root, "o1_b200", "corpus"), "kind": "tree"},
        "runner_source": {
            "path": os.path.join(root, "o1_b200", "runner"), "kind": "tree"},
        "policies": {
            "path": os.path.join(root, "o1_b200", "policies"), "kind": "tree"},
        "deploy_scripts": {
            "path": os.path.join(root, "o1_b200", "deploy"), "kind": "tree"},
        "container_reference": {
            "path": os.path.join(root, "o1_b200", "deploy", "Dockerfile"),
            "kind": "file"},
    }
    out_path = os.path.join(root, "o1_b200", "deploy", "TRANSFER_MANIFEST.json")
    manifest = build_manifest(entries, out_path)
    manifest["output_destination_layout"] = {
        "instance_outputs": "/outputs (records, reports, logs)",
        "packaging": "deploy/package_outputs.sh -> single checksummed archive",
        "return_path": "hash-verified download, then local analysis",
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {out_path}: {len(manifest['artifacts'])} artifacts, "
          f"manifest sha256 {sha256_file(out_path)[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
