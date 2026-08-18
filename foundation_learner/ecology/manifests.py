"""Ecology manifests (contract §5).

``family_split_manifest.json`` records the per-family split hashes, the induced
ordering, the assignment, and the SHA-256 of every generator source file, so an
auditor can recompute the split and confirm that the generators that produced
the data are the generators that were hashed.

``split_manifest_sha256 = domain_sha256("FL_V0_SPLIT_MANIFEST", manifest)`` is
computed over the manifest WITHOUT that field and then attached; it is also the
input to the sealed-shard key derivation (Amendment 1).
"""
from __future__ import annotations

import hashlib
import os

from .base import canonical_json, domain_sha256, sha256_file
from .split import (EXPECTED_SPLIT, FAMILY_IDS, SPLIT_DOMAIN, compute_split,
                    ordered_families)

__all__ = ["SPLIT_MANIFEST_NAME", "build_split_manifest",
           "split_manifest_sha256", "write_split_manifest",
           "read_split_manifest", "verify_split_manifest"]

SPLIT_MANIFEST_NAME = "family_split_manifest.json"


def build_split_manifest() -> dict:
    """Recompute the split manifest from the frozen rule and the sources."""
    from .families import family_module_paths

    order = ordered_families()
    assignment = compute_split()
    sources = family_module_paths()
    manifest = {
        "schema": "fl.family_split_manifest.v1",
        "split_domain": SPLIT_DOMAIN,
        "family_ids": list(FAMILY_IDS),
        "family_hashes": {fid: digest for fid, digest in order},
        "hash_order": [fid for fid, _ in order],
        "assignment": {name: list(fids) for name, fids in assignment.items()},
        "expected_assignment": {name: list(fids)
                                for name, fids in EXPECTED_SPLIT.items()},
        "generator_sources": {
            fid: {"basename": os.path.basename(path),
                  "sha256": sha256_file(path)}
            for fid, path in sorted(sources.items())},
        "rule": ("sha256(domain + NUL + family_id), ascending hex; "
                 "positions 0-5 TRAIN, 6-8 DEVELOPMENT, 9-11 SEALED_TEST"),
    }
    manifest["split_manifest_sha256"] = split_manifest_sha256(manifest)
    return manifest


def split_manifest_sha256(manifest: dict) -> str:
    """Digest of a manifest, excluding the digest field itself."""
    payload = {k: v for k, v in manifest.items()
               if k != "split_manifest_sha256"}
    return domain_sha256("FL_V0_SPLIT_MANIFEST", payload)


def write_split_manifest(path: str) -> dict:
    manifest = build_split_manifest()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(manifest))
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return manifest


def read_split_manifest(path: str) -> dict:
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_split_manifest(manifest: dict, check_sources: bool = True) -> None:
    """Hard-fail if a manifest disagrees with the frozen rule, its digest or
    the generator sources actually present on disk.

    Re-hashing the sources matters because an attacker who edits a manifest can
    also recompute its digest; only the files themselves are authoritative.
    """
    from .families import family_module_paths

    if manifest.get("split_manifest_sha256") != split_manifest_sha256(manifest):
        raise ValueError("family split manifest digest mismatch")
    if manifest["assignment"] != {k: list(v) for k, v in compute_split().items()}:
        raise ValueError("family split manifest disagrees with the frozen rule")
    if manifest["assignment"] != {k: list(v) for k, v in EXPECTED_SPLIT.items()}:
        raise ValueError("family split manifest disagrees with Amendment 2")
    for fid, digest in manifest["family_hashes"].items():
        if digest != hashlib.sha256(
                SPLIT_DOMAIN.encode("utf-8") + b"\0" + fid.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"family split hash mismatch for {fid}")
    if check_sources:
        paths = family_module_paths()
        recorded = manifest["generator_sources"]
        if set(recorded) != set(paths):
            raise ValueError("generator source set disagrees with the package")
        for fid, path in paths.items():
            if recorded[fid]["sha256"] != sha256_file(path):
                raise ValueError(
                    f"generator source for {fid} does not match the manifest")
