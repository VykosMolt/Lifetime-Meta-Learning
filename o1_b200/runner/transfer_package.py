"""Provider-neutral artifact-transfer manifest and deterministic archiving.

Builds TRANSFER_MANIFEST.json listing every artifact the future B200 session
needs, each with its SHA-256 (files) or tree hash (directories).  The full
model checkpoint is referenced by tree hash, never placed in Git or in the
archive.  A deny-list refuses obvious secret material.  Every transferred
artifact must be hash-verified on the instance before any scientific process
starts (deploy/verify_artifacts.py consumes this manifest).
"""
from __future__ import annotations

import json
import os
import zipfile

from .identity import sha256_file, sha256_tree
from .persistence import atomic_write_text

SECRET_PATTERNS = (
    ".pem", ".key", "id_rsa", "id_ed25519", ".aws", "credentials",
    ".netrc", "token", "apikey", "api_key", "password", ".env",
)


class TransferError(RuntimeError):
    pass


def _check_no_secrets(path: str) -> None:
    base = os.path.basename(path).lower()
    # "tokenizer" artifacts are model assets, not credentials; every other
    # occurrence of "token" in a filename is treated as secret material.
    scrubbed = base.replace("tokenizer", "")
    for pat in SECRET_PATTERNS:
        if pat in scrubbed:
            raise TransferError(
                f"refusing to package {path}: name matches secret pattern "
                f"{pat!r}")


def build_manifest(entries: dict[str, dict], out_path: str) -> dict:
    """entries: {name: {"path": ..., "kind": "file"|"tree"|"reference",
    "expected_sha256": optional}}.

    'reference' entries (e.g. the model checkpoint) are hashed but marked
    transfer_out_of_band=True — they are NOT archived and NOT in Git.
    """
    manifest = {"schema": "o1b200.transfer_manifest.v1", "artifacts": {}}
    for name, spec in sorted(entries.items()):
        path = spec["path"]
        kind = spec["kind"]
        _check_no_secrets(path)
        if not os.path.exists(path):
            raise TransferError(f"{name}: missing artifact {path}")
        digest = sha256_tree(path) if os.path.isdir(path) else sha256_file(path)
        want = spec.get("expected_sha256")
        if want is not None and digest != want:
            raise TransferError(
                f"{name}: hashes {digest[:16]}... but expected {want[:16]}...")
        manifest["artifacts"][name] = {
            "path": os.path.relpath(path) if not os.path.isabs(path) else path,
            "kind": kind,
            "sha256": digest,
            "transfer_out_of_band": kind == "reference",
        }
    atomic_write_text(out_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def deterministic_zip(file_paths: list[str], root: str, out_zip: str) -> str:
    """Reproducible zip: sorted entries, fixed timestamps, no extra attrs."""
    entries = []
    for p in sorted(file_paths):
        _check_no_secrets(p)
        rel = os.path.relpath(p, root)
        if rel.startswith(".."):
            raise TransferError(f"{p} escapes archive root {root}")
        entries.append((rel, p))
    tmp = out_zip + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for rel, p in entries:
            with open(p, "rb") as fh:
                data = fh.read()
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    os.replace(tmp, out_zip)
    return sha256_file(out_zip)


def write_sha256sums(file_paths: list[str], root: str, out_path: str) -> str:
    lines = []
    for p in sorted(file_paths):
        rel = os.path.relpath(p, root)
        lines.append(f"{sha256_file(p)}  {rel}")
    atomic_write_text(out_path, "\n".join(lines) + "\n")
    return sha256_file(out_path)


def verify_sha256sums(sums_path: str, root: str) -> dict:
    checked = 0
    with open(sums_path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            digest, rel = ln.split("  ", 1)
            full = os.path.join(root, rel)
            if not os.path.exists(full):
                raise TransferError(f"missing file listed in sums: {rel}")
            got = sha256_file(full)
            if got != digest:
                raise TransferError(f"hash mismatch for {rel}")
            checked += 1
    return {"checked": checked}
