"""Run-output verification and the deterministic transfer archive.

Two jobs, both mirrored from the O1 runner's conventions
(``o1_b200/runner/transfer_package.py``), re-implemented here rather than
imported because the FL package may never import the sealed O1 modules
(contract §22):

* :func:`build_result_manifest` / :func:`verify_result_tree` — hash every file
  of a run-output tree and refuse on any mismatch, missing file, or unexpected
  extra file.  Sealed results (``chmod 0444``) hash normally; hashing bytes is
  not reading outcomes.
* :func:`build_transfer_archive` — a byte-deterministic zip (sorted entries,
  ``date_time=(1980,1,1,0,0,0)``, ``external_attr = 0o644 << 16``,
  ``ZIP_DEFLATED`` level 9) with the secret-pattern deny-list applied to every
  member name.

Every path goes through the O1 isolation guard, so an artefact tree that
reaches into an O1 root cannot be hashed, archived, or transferred.
"""
from __future__ import annotations

import os
import re
import zipfile
from typing import Iterable, Mapping, Sequence

from ..ecology.base import sha256_file
from . import o1_isolation

__all__ = [
    "TransferError",
    "SECRET_PATTERNS",
    "RESULT_MANIFEST_SCHEMA",
    "DEFAULT_EXCLUDES",
    "check_no_secrets",
    "iter_tree_files",
    "build_result_manifest",
    "write_result_manifest",
    "verify_result_tree",
    "deterministic_zip",
    "write_sha256sums",
    "verify_sha256sums",
    "build_transfer_archive",
]

RESULT_MANIFEST_SCHEMA = "flb200.result_manifest.v1"

#: identical to the O1 deny-list (``tokenizer`` is a model asset, not a secret)
SECRET_PATTERNS = (
    ".pem", ".key", "id_rsa", "id_ed25519", ".aws", "credentials",
    ".netrc", "token", "apikey", "api_key", "password", ".env",
)

DEFAULT_EXCLUDES = ("__pycache__", ".pyc", ".tmp")


class TransferError(RuntimeError):
    """A verification or packaging rule refused."""


#: The O1 deny-list scrubs the word "tokenizer" before matching, because a
#: tokenizer is a model asset and not a credential.  This package additionally
#: contains ``tokenization.py`` and its test, so the scrub covers the whole
#: word family.  A bare ``token`` in a file name (``hf_token``, ``api_token``,
#: ``token.txt``) still trips the deny-list.
_TOKENIZER_WORDS = re.compile(r"tokeniz(er|ers|ation|ations|e|es|ing|ed)")


def check_no_secrets(path: str) -> None:
    base = os.path.basename(str(path)).lower()
    scrubbed = _TOKENIZER_WORDS.sub("", base)
    for pattern in SECRET_PATTERNS:
        if pattern in scrubbed:
            raise TransferError(
                f"refusing to package {path}: name matches secret pattern "
                f"{pattern!r}")


def _excluded(relpath: str, excludes: Sequence[str]) -> bool:
    parts = relpath.replace(os.sep, "/").split("/")
    for token in excludes:
        if token.startswith("."):
            if relpath.endswith(token):
                return True
        elif token in parts:
            return True
    return False


def iter_tree_files(root: str, *, excludes: Sequence[str] = DEFAULT_EXCLUDES,
                    guard: o1_isolation.IsolationGuard | None = None
                    ) -> list[str]:
    """Sorted relative paths of every non-excluded file under ``root``."""
    guard = guard or o1_isolation.default_guard()
    resolved = guard.guard(root, o1_isolation.MODE_READ)
    out: list[str] = []
    for dirpath, dirnames, filenames in guard.walk(resolved):
        dirnames[:] = [d for d in dirnames if d not in excludes]
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, resolved).replace(os.sep, "/")
            if _excluded(rel, excludes):
                continue
            out.append(rel)
    return sorted(out)


def build_result_manifest(root: str, *, label: str = "FL_RUN",
                          excludes: Sequence[str] = DEFAULT_EXCLUDES,
                          guard: o1_isolation.IsolationGuard | None = None) -> dict:
    """Hash every file of a run-output tree."""
    guard = guard or o1_isolation.default_guard()
    resolved = guard.guard(root, o1_isolation.MODE_READ)
    files = {}
    for rel in iter_tree_files(resolved, excludes=excludes, guard=guard):
        full = os.path.join(resolved, rel)
        files[rel] = {"sha256": sha256_file(full),
                      "bytes": os.path.getsize(full),
                      "mode": oct(os.stat(full).st_mode & 0o777)}
    return {
        "schema": RESULT_MANIFEST_SCHEMA,
        "label": str(label),
        "root_basename": os.path.basename(resolved.rstrip(os.sep)),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "files": files,
        "excludes": list(excludes),
        "note": ("file-level hashing only: the verifier never parses the "
                 "scientific content of what it hashes"),
    }


def write_result_manifest(root: str, out_path: str, *, label: str = "FL_RUN",
                          excludes: Sequence[str] = DEFAULT_EXCLUDES,
                          guard: o1_isolation.IsolationGuard | None = None) -> dict:
    guard = guard or o1_isolation.default_guard()
    manifest = build_result_manifest(root, label=label, excludes=excludes,
                                     guard=guard)
    guard.write_json(out_path, manifest)
    return manifest


def verify_result_tree(root: str, manifest: Mapping | str, *,
                       allow_extra: bool = False,
                       guard: o1_isolation.IsolationGuard | None = None) -> dict:
    """Re-hash a tree against a recorded manifest; refuse on any difference."""
    guard = guard or o1_isolation.default_guard()
    if isinstance(manifest, str):
        manifest = guard.read_json(manifest)
    if manifest.get("schema") != RESULT_MANIFEST_SCHEMA:
        raise TransferError(
            f"unexpected result-manifest schema {manifest.get('schema')!r}")
    resolved = guard.guard(root, o1_isolation.MODE_READ)
    recorded = dict(manifest["files"])
    present = set(iter_tree_files(resolved,
                                  excludes=manifest.get("excludes", DEFAULT_EXCLUDES),
                                  guard=guard))
    missing = sorted(set(recorded) - present)
    extra = sorted(present - set(recorded))
    mismatched = []
    for rel, entry in sorted(recorded.items()):
        full = os.path.join(resolved, rel)
        if not os.path.isfile(full):
            continue
        if sha256_file(full) != entry["sha256"]:
            mismatched.append(rel)
    report = {
        "schema": "flb200.result_verification.v1",
        "checked": len(recorded),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "ok": not missing and not mismatched and (allow_extra or not extra),
    }
    if not report["ok"]:
        raise TransferError(
            f"run-output verification FAILED: {len(missing)} missing, "
            f"{len(mismatched)} hash mismatches, {len(extra)} unexpected files "
            f"(first missing {missing[:1]}, first mismatch {mismatched[:1]}, "
            f"first extra {extra[:1]})")
    return report


def deterministic_zip(file_paths: Sequence[str], root: str, out_zip: str, *,
                      guard: o1_isolation.IsolationGuard | None = None) -> str:
    """Reproducible zip (O1 algorithm): sorted, 1980 dates, 0o644, deflate 9."""
    guard = guard or o1_isolation.default_guard()
    root_resolved = guard.guard(root, o1_isolation.MODE_READ)
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in sorted(str(p) for p in file_paths):
        full = guard.guard(path, o1_isolation.MODE_READ)
        check_no_secrets(full)
        rel = os.path.relpath(full, root_resolved).replace(os.sep, "/")
        if rel.startswith(".."):
            raise TransferError(f"{path} escapes archive root {root}")
        if rel in seen:
            raise TransferError(f"duplicate archive entry {rel!r}")
        seen.add(rel)
        entries.append((rel, full))
    entries.sort(key=lambda pair: pair[0])
    out_resolved = guard.guard(out_zip, o1_isolation.MODE_WRITE)
    os.makedirs(os.path.dirname(out_resolved) or ".", exist_ok=True)
    tmp = out_resolved + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for rel, full in entries:
            with open(full, "rb") as fh:
                data = fh.read()
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    os.replace(tmp, out_resolved)
    return sha256_file(out_resolved)


def write_sha256sums(file_paths: Sequence[str], root: str, out_path: str, *,
                     guard: o1_isolation.IsolationGuard | None = None) -> str:
    """O1 two-space ``SHA256SUMS`` format over relative paths."""
    guard = guard or o1_isolation.default_guard()
    root_resolved = guard.guard(root, o1_isolation.MODE_READ)
    lines = []
    for path in sorted(str(p) for p in file_paths):
        full = guard.guard(path, o1_isolation.MODE_READ)
        rel = os.path.relpath(full, root_resolved).replace(os.sep, "/")
        lines.append(f"{sha256_file(full)}  {rel}")
    guard.write_text(out_path, "\n".join(lines) + "\n")
    return sha256_file(guard.guard(out_path, o1_isolation.MODE_READ))


def verify_sha256sums(sums_path: str, root: str, *,
                      guard: o1_isolation.IsolationGuard | None = None) -> dict:
    guard = guard or o1_isolation.default_guard()
    root_resolved = guard.guard(root, o1_isolation.MODE_READ)
    checked = 0
    bad: list[str] = []
    missing: list[str] = []
    for line in guard.read_text(sums_path).splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        full = os.path.join(root_resolved, rel)
        if not os.path.exists(full):
            missing.append(rel)
            continue
        if sha256_file(full) != digest:
            bad.append(rel)
        checked += 1
    if bad or missing:
        raise TransferError(
            f"SHA256SUMS verification failed: {len(bad)} mismatches, "
            f"{len(missing)} missing (first bad {bad[:1]}, first missing "
            f"{missing[:1]})")
    return {"checked": checked, "ok": True}


def build_transfer_archive(root: str, out_zip: str, *,
                           excludes: Sequence[str] = DEFAULT_EXCLUDES,
                           extra_files: Iterable[str] = (),
                           label: str = "FL_TRANSFER",
                           guard: o1_isolation.IsolationGuard | None = None) -> dict:
    """Verify a run tree, then archive it deterministically."""
    guard = guard or o1_isolation.default_guard()
    resolved = guard.guard(root, o1_isolation.MODE_READ)
    rels = iter_tree_files(resolved, excludes=excludes, guard=guard)
    paths = [os.path.join(resolved, rel) for rel in rels]
    paths.extend(guard.guard(p, o1_isolation.MODE_READ) for p in extra_files)
    manifest = build_result_manifest(resolved, label=label, excludes=excludes,
                                     guard=guard)
    digest = deterministic_zip(paths, resolved, out_zip, guard=guard)
    sidecar = guard.guard(out_zip + ".sha256", o1_isolation.MODE_WRITE)
    guard.write_text(sidecar, f"{digest}  {os.path.basename(out_zip)}\n")
    return {
        "schema": "flb200.transfer_archive.v1",
        "label": str(label),
        "zip_path": guard.guard(out_zip, o1_isolation.MODE_READ),
        "zip_sha256": digest,
        "zip_bytes": os.path.getsize(guard.guard(out_zip, o1_isolation.MODE_READ)),
        "n_entries": len(paths),
        "manifest": manifest,
        "sidecar": sidecar,
    }
