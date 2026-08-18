#!/usr/bin/env python3
"""Generate RUNPOD_ARTIFACT_TRANSFER_MANIFEST.json.

Per artifact: logical name, local path, byte size, SHA-256/tree hash,
destination path on the Pod, confidentiality classification, transfer
mechanism, transfer order, and the post-transfer verification command.  The
checkpoint is out-of-band (never Git, never a public image).  The future Pod
refuses model loading until deploy/verify_artifacts.py passes on this
manifest's hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)

from o1_b200.runner.identity import sha256_file, sha256_tree  # noqa: E402

RUN_DIR = os.path.join(_ROOT, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN")
CHECKPOINT = "/home/moloch/ouro_project/models/ouro_rltt_local"


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def build_manifest() -> dict:
    verify = ("PYTHONPATH=/opt/o1_b200 python -m "
              "o1_b200.deploy.verify_artifacts --manifest "
              "/artifacts/RUNPOD_ARTIFACT_TRANSFER_MANIFEST.json "
              "--root /artifacts")
    entries = [
        # order 1: big out-of-band artifact first (staged before GPU billing)
        ("ouro_rltt_checkpoint", CHECKPOINT, "/artifacts/ouro_rltt_local",
         "RESTRICTED (model weights; never Git, never a public image)",
         "object-store adapter (LocalArtifactStore now; S3-compatible store "
         "chosen in the rental session) — resumable chunked fetch", 1),
        ("tokenizer_binding",
         os.path.join(RUN_DIR, "TOKENIZER_BINDING.json"),
         "/artifacts/TOKENIZER_BINDING.json", "public", "with runner image", 2),
        ("o1_v210_package_zip",
         os.path.join(_ROOT, "o1_packages",
                      "O1_oracle_reachability_v2.1.0_verified.zip"),
         "/artifacts/O1_oracle_reachability_v2.1.0_verified.zip",
         "public", "baked into image + re-verified", 3),
        ("axis_package", os.path.join(RUN_DIR, "AXIS_PACKAGE_V2"),
         "/artifacts/AXIS_PACKAGE_V2", "public", "baked into image", 4),
        ("calibration_task_manifest",
         os.path.join(RUN_DIR, "COHORTS", "calibration_tasks.jsonl"),
         "/artifacts/COHORTS/calibration_tasks.jsonl", "public",
         "baked into image", 5),
        ("calibration_seed_matrix",
         os.path.join(RUN_DIR, "calibration_seed_matrix.json"),
         "/artifacts/calibration_seed_matrix.json", "public",
         "baked into image", 6),
        ("validation_corpus", os.path.join(_ROOT, "o1_b200", "corpus"),
         "/artifacts/validation_corpus", "public", "baked into image", 7),
        ("environment_lock",
         os.path.join(_ROOT, "o1_b200", "deploy", "requirements.b300.lock"),
         "/artifacts/requirements.b300.lock", "public", "baked into image", 8),
        ("backend_selection_policy",
         os.path.join(_ROOT, "o1_b200", "policies",
                      "B200_BACKEND_SELECTION_POLICY.json"),
         "/artifacts/policies/B200_BACKEND_SELECTION_POLICY.json", "public",
         "baked into image", 9),
        ("budget_policy",
         os.path.join(_ROOT, "o1_b200", "policies",
                      "B200_BUDGET_POLICY.template.json"),
         "/artifacts/policies/B200_BUDGET_POLICY.template.json", "public",
         "baked into image", 10),
        ("precommit_template",
         os.path.join(_ROOT, "o1_b200", "policies",
                      "CALIBRATION_PRECOMMIT_B200.template.json"),
         "/artifacts/policies/CALIBRATION_PRECOMMIT_B200.template.json",
         "public", "baked into image", 11),
        ("b200_runner_package_zip",
         os.path.join(_ROOT, "o1_packages", "O1_B200_RUNNER_v0.1.0.zip"),
         "/artifacts/O1_B200_RUNNER_v0.1.0.zip", "public",
         "baked into image", 12),
    ]
    artifacts = {}
    for name, path, dest, conf, mech, order in entries:
        digest = sha256_tree(path) if os.path.isdir(path) else sha256_file(path)
        artifacts[name] = {
            "local_path": path,
            "byte_size": _size(path),
            "sha256": digest,
            "hash_kind": "tree" if os.path.isdir(path) else "file",
            "destination_path": dest,
            "confidentiality": conf,
            "transfer_mechanism": mech,
            "transfer_order": order,
            "post_transfer_verification": verify,
        }
    manifest = {
        "schema": "o1b200.runpod_artifact_transfer_manifest.v1",
        "note": ("model loading on the Pod is REFUSED until every hash "
                 "passes; large artifacts are staged before the GPU billing "
                 "clock where the chosen store allows"),
        "artifacts": artifacts,
    }
    return manifest


def main() -> int:
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "RUNPOD_ARTIFACT_TRANSFER_MANIFEST.json")
    manifest = build_manifest()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {out} ({len(manifest['artifacts'])} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
