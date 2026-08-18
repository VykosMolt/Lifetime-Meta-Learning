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
    manifest["path_domain"] = (
        "HOST paths on the build machine — build-time provenance only. The "
        "POD verifies POD_TRANSFER_MANIFEST.json, whose paths are the "
        "container's.")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {out_path}: {len(manifest['artifacts'])} artifacts, "
          f"manifest sha256 {sha256_file(out_path)[:16]}...")

    pod_path = write_pod_manifest(manifest, root)
    print(f"wrote {pod_path}: container-path manifest, sha256 "
          f"{sha256_file(pod_path)[:16]}...")
    return 0


#: Where the image places each host tree (Dockerfile.b300 COPY targets).
CONTAINER_ROOT = "/opt/o1_b200"
CONTAINER_ARTIFACTS = "/artifacts"

#: Entries the POD manifest deliberately omits, because hashing them on the
#: pod is either self-referential or non-deterministic:
#:
#:   deploy_scripts    — this manifest is WRITTEN INTO that tree, so its
#:                       hash can never describe the tree that contains it.
#:   runner_source     — importing the runner materialises __pycache__ in
#:                       the same tree, so the digest changes as a side
#:                       effect of reading it.
#:   container_reference — a Dockerfile inside deploy/, same problem.
#:
#: This is not a coverage gap: the pod's CODE integrity is established by
#: the immutable image digest, which the rental authorization binds through
#: image_digest_ref, plus the sealed package's own byte-hash bridge.  What
#: this manifest exists to verify is the DATA that arrives separately —
#: above all the checkpoint, which is never in the image.
POD_MANIFEST_EXCLUDED = ("deploy_scripts", "runner_source",
                         "container_reference")


def to_container_path(host_path: str, root: str) -> str:
    """Map a build-host path to where the artifact actually lives on the pod.

    The checkpoint is never baked into the image: it is fetched to
    /artifacts by runner/fetch_artifacts.py before verification runs.
    """
    if host_path.rstrip("/") == CHECKPOINT_PATH.rstrip("/"):
        return os.path.join(CONTAINER_ARTIFACTS, "ouro_rltt_local")
    if host_path.startswith(root.rstrip("/") + "/"):
        return os.path.join(CONTAINER_ROOT,
                            os.path.relpath(host_path, root))
    raise ValueError(f"no container mapping for host path {host_path!r}")


def write_pod_manifest(host_manifest: dict, root: str) -> str:
    """The manifest the POD verifies, with container paths.

    verify_artifacts.py uses absolute paths verbatim, so a manifest full of
    build-host paths makes on-pod verification fail every artifact — the
    container has no /home/moloch.  Hashes are identical; only the paths
    are rewritten.
    """
    artifacts = {}
    for name, spec in host_manifest["artifacts"].items():
        if name in POD_MANIFEST_EXCLUDED:
            continue
        entry = dict(spec)
        entry["host_path"] = spec["path"]
        entry["path"] = to_container_path(spec["path"], root)
        artifacts[name] = entry
    pod = {
        "schema": host_manifest["schema"],
        "artifacts": artifacts,
        "path_domain": (
            "CONTAINER paths. Baked trees live under /opt/o1_b200; the "
            "checkpoint is fetched to /artifacts by "
            "runner/fetch_artifacts.py before this manifest is verified."),
        "generated_from": "deploy/TRANSFER_MANIFEST.json",
    }
    out = os.path.join(root, "o1_b200", "deploy", "POD_TRANSFER_MANIFEST.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(pod, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
