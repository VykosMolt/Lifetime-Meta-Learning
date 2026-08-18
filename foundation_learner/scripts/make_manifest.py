#!/usr/bin/env python3
"""Fill the campaign manifest template (contract §19).

``FOUNDATION_LEARNER_V0_MANIFEST.template.json`` ->
``FOUNDATION_LEARNER_V0_MANIFEST.json``, resolving exactly the fields that are
knowable BEFORE the accelerator session:

* ``ecology.split_manifest_sha256`` — from the pregenerated split manifest;
* ``ecology.generator_source_hashes`` — the per-family generator source digests
  recorded in that manifest (re-verified against the files on disk);
* ``ecology.shard_sums_sha256`` — the digest of ``SHARD_SUMS.json``;
* ``package.zip_sha256`` — from the release zip (or its ``.sha256`` sidecar);
* ``package.source_commit`` — ``git rev-parse HEAD`` of the worktree.

Everything under ``b200_derived_unresolved`` is left UNRESOLVED, deliberately:
measured throughput, the affordable ``U``, the FULL-vs-PEFT outcome, the
available Foundation Learner seconds, the post-gate evaluation batch size, the
operator-bound ``o1_entry_command`` and the operator-bound
``container_registry_digest_ref`` (the B200 image is pushed by the operator;
its registry digest does not exist until then — verification finding V-Obs6)
are all decided on (or before) the accelerator session, never here.

ORDER (contract §19, repaired): validate -> package_release -> make_manifest.
The manifest binds the zip hash, so it is written AFTER the zip and is never
covered by it; ``scripts/package_release.py`` excludes it from the bundle for
exactly that reason.

Usage::

    python -m foundation_learner.scripts.make_manifest --check
    python -m foundation_learner.scripts.make_manifest --zip <path>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from foundation_learner.ecology.base import sha256_file  # noqa: E402
from foundation_learner.ecology.manifests import (  # noqa: E402
    read_split_manifest, verify_split_manifest)
from foundation_learner.scripts.package_release import (  # noqa: E402
    MANIFEST_DIR_NAME, ZIP_NAME)

TEMPLATE_NAME = "FOUNDATION_LEARNER_V0_MANIFEST.template.json"
OUTPUT_NAME = "FOUNDATION_LEARNER_V0_MANIFEST.json"

FILLED_BY_PREGENERATION = "UNRESOLVED_FILLED_BY_PREGENERATION"
FILLED_BY_PACKAGING = "UNRESOLVED_FILLED_BY_PACKAGING"


class ManifestError(RuntimeError):
    """The manifest could not be filled from real artefacts."""


def source_commit(repo_root: str) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestError(f"cannot read the source commit: {exc}") from exc


def _pregen_file(pregen_root: str, name: str) -> str:
    """Prefer the committed ``MANIFESTS/`` mirror, fall back to the raw root."""
    mirrored = os.path.join(pregen_root, MANIFEST_DIR_NAME, name)
    if os.path.isfile(mirrored):
        return mirrored
    direct = os.path.join(pregen_root, name)
    if os.path.isfile(direct):
        return direct
    raise ManifestError(
        f"{name} not found under {pregen_root} (neither in {MANIFEST_DIR_NAME}/ "
        "nor at the root); run scripts/pregenerate_all.py first")


def fill(template: dict, *, pregen_root: str, repo_root: str,
         zip_path: str | None = None,
         verify_sources: bool = True) -> dict:
    manifest = json.loads(json.dumps(template))   # deep copy

    split_path = _pregen_file(pregen_root, "family_split_manifest.json")
    split = read_split_manifest(split_path)
    verify_split_manifest(split, check_sources=verify_sources)
    manifest["ecology"]["split_manifest_sha256"] = split["split_manifest_sha256"]
    manifest["ecology"]["generator_source_hashes"] = {
        fid: entry["sha256"] for fid, entry in
        sorted(split["generator_sources"].items())}

    shard_sums = _pregen_file(pregen_root, "SHARD_SUMS.json")
    manifest["ecology"]["shard_sums_sha256"] = sha256_file(shard_sums)

    pregen_manifest_path = _pregen_file(pregen_root, "PREGEN_MANIFEST.json")
    with open(pregen_manifest_path, encoding="utf-8") as fh:
        pregen = json.load(fh)
    manifest["ecology"]["pregen_manifest_sha256"] = sha256_file(pregen_manifest_path)
    manifest["ecology"]["pregen_root_basename"] = os.path.basename(
        os.path.abspath(pregen_root))
    manifest["ecology"]["pregen_root_seed"] = pregen.get("root_seed")
    manifest["ecology"]["pregen_files"] = len(pregen.get("files", []))
    if pregen.get("split_manifest_sha256") != split["split_manifest_sha256"]:
        raise ManifestError(
            "the pregenerated data was produced under a DIFFERENT family split "
            "manifest than the one on disk; refusing to write a manifest that "
            "would misdescribe it")

    if zip_path:
        sidecar = zip_path + ".sha256"
        if os.path.isfile(zip_path):
            manifest["package"]["zip_sha256"] = sha256_file(zip_path)
        elif os.path.isfile(sidecar):
            manifest["package"]["zip_sha256"] = open(
                sidecar, encoding="utf-8").read().split("  ", 1)[0]
        else:
            raise ManifestError(f"no zip and no sidecar at {zip_path!r}")
        manifest["package"]["zip_name"] = os.path.basename(zip_path)
    manifest["package"]["source_commit"] = source_commit(repo_root)

    manifest["provenance"] = {
        "filled_by": "foundation_learner/scripts/make_manifest.py",
        "split_manifest_path": os.path.relpath(split_path, repo_root),
        "shard_sums_path": os.path.relpath(shard_sums, repo_root),
        "note": ("B200-derived fields stay UNRESOLVED by design: they are "
                 "measured or decided mechanically on the accelerator"),
    }
    return manifest


def unresolved_fields(manifest: dict) -> list[str]:
    found: list[str] = []

    def walk(node, path=""):
        if isinstance(node, str) and node.startswith("UNRESOLVED"):
            found.append(path)
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(manifest)
    return sorted(found)


def check_only_b200_fields_remain(manifest: dict) -> dict:
    """Contract §21b: only B200-derived fields may still be open."""
    open_fields = unresolved_fields(manifest)
    leaked = [f for f in open_fields
              if not f.startswith("b200_derived_unresolved")]
    return {"unresolved": open_fields, "non_b200_unresolved": leaked,
            "ok": not leaked}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill the FL campaign manifest")
    parser.add_argument("--repo-root", default=_PKG_PARENT)
    parser.add_argument("--pregen",
                        default=os.path.join(_PKG_PARENT, "artifacts_fl", "pregen"))
    parser.add_argument("--zip", dest="zip_path",
                        default=os.path.join(_PKG_PARENT, ZIP_NAME))
    parser.add_argument("--out", default=None,
                        help="output path (default: alongside the template)")
    parser.add_argument("--check", action="store_true",
                        help="do not write; report which fields would remain open")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    package = os.path.join(args.repo_root, "foundation_learner")
    with open(os.path.join(package, TEMPLATE_NAME), encoding="utf-8") as fh:
        template = json.load(fh)
    zip_path = args.zip_path if os.path.exists(args.zip_path) or \
        os.path.exists(args.zip_path + ".sha256") else None
    manifest = fill(template, pregen_root=args.pregen,
                    repo_root=args.repo_root, zip_path=zip_path)
    status = check_only_b200_fields_remain(manifest)
    out = args.out or os.path.join(package, OUTPUT_NAME)
    if not args.check:
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
        print(f"wrote {out}")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
