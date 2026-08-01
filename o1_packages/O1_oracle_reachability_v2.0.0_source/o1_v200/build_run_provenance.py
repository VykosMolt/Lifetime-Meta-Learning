"""Build O1 RUN_PROVENANCE.json from the sealed manifest and real artifacts.

The output is not a second configuration surface. It contains an exact recursive
copy of the manifest's generation-relevant sections plus a runtime attestation.
Every mapped artifact is hashed before the file is written; a mismatch aborts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from o1_analysis import (
    IntegrityError,
    Manifest,
    PROVENANCE_SCHEMA_VERSION,
    canonical_json_bytes,
    resolved_generation_config,
    sha256_file,
    sha256_tree,
)

REQUIRED_ARTIFACT_KEYS = (
    "code.generation_module_sha256",
    "model.checkpoint_sha256",
    "artifact_hashes.tokenizer",
    "artifact_hashes.prompt_template",
    "artifact_hashes.parser",
    "artifact_hashes.verifier_implementation",
    "artifact_hashes.structured_axis_tensor",
    "artifact_hashes.random_axis_tensor",
)


def _load_json(path: str) -> dict:
    with open(path) as fh:
        out = json.load(fh)
    if not isinstance(out, dict):
        raise IntegrityError("P1_ARTIFACT_MAP", f"{path} must contain a JSON object")
    return out


def build_run_provenance(
    manifest_path: str,
    artifact_paths_path: str,
    output_path: str,
    repository_diff_path: str | None = None,
) -> dict:
    if os.path.exists(output_path):
        raise IntegrityError("P0_IMMUTABLE", f"refusing to overwrite {output_path}")
    man = Manifest(manifest_path)
    paths = _load_json(artifact_paths_path)
    missing = [k for k in REQUIRED_ARTIFACT_KEYS if k not in paths]
    extra = sorted(set(paths) - set(REQUIRED_ARTIFACT_KEYS))
    if missing or extra:
        raise IntegrityError(
            "P1_ARTIFACT_MAP",
            f"artifact map mismatch: missing={missing}, extra={extra}",
            detail={"missing": missing, "extra": extra},
        )

    checked: dict[str, dict[str, str]] = {}
    for key in REQUIRED_ARTIFACT_KEYS:
        path = paths[key]
        if not isinstance(path, str) or not os.path.exists(path):
            raise IntegrityError("P1_ARTIFACT_MAP", f"{key}: path does not exist: {path!r}")
        got = sha256_tree(path)
        want = man.get(key)
        if got != want:
            raise IntegrityError(
                "P2_ARTIFACT_HASH",
                f"{key}: sealed {want[:16]}... but {path} hashes {got[:16]}...",
                detail={"key": key, "path": path, "sealed": want, "actual": got},
            )
        checked[key] = {"path_type": "directory" if os.path.isdir(path) else "file",
                        "sha256": got}

    clean_required = bool(man.get("code.repository_clean_required"))
    if repository_diff_path is None:
        diff_bytes = b""
    else:
        with open(repository_diff_path, "rb") as fh:
            diff_bytes = fh.read()
    diff_hash = __import__("hashlib").sha256(diff_bytes).hexdigest()
    want_diff = man.get("code.repository_diff_sha256")
    if diff_hash != want_diff:
        raise IntegrityError(
            "P3_WORKTREE",
            f"repository diff hashes {diff_hash[:16]}..., sealed value is {want_diff[:16]}...",
            detail={"sealed": want_diff, "actual": diff_hash},
        )
    if clean_required and diff_bytes:
        raise IntegrityError("P3_WORKTREE", "manifest requires a clean repository but diff is non-empty")

    out: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resolved_generation_config": resolved_generation_config(man),
        "runtime_attestation": {
            "generation_module_sha256": man.get("code.generation_module_sha256"),
            "repository_clean": clean_required,
            "repository_diff_sha256": diff_hash,
        },
    }
    # Validate serializability and NaN rejection before the atomic write.
    canonical_json_bytes(out)
    tmp = output_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, output_path)
    return {"output": output_path, "sha256": sha256_file(output_path),
            "checked_artifacts": checked}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--artifact-paths", required=True,
                   help="JSON mapping the eight required manifest hash keys to real files/directories")
    p.add_argument("--output", required=True)
    p.add_argument("--repository-diff",
                   help="raw git diff file; omit for a clean tree")
    a = p.parse_args()
    print(json.dumps(build_run_provenance(
        a.manifest, a.artifact_paths, a.output, a.repository_diff), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
