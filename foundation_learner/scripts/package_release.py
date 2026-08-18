#!/usr/bin/env python3
"""Deterministic release packaging (contract §19).

Builds ``FOUNDATION_LEARNER_B200_V0.2.0.zip`` with the O1 algorithm — sorted
entries, ``date_time=(1980,1,1,0,0,0)``, ``external_attr = 0o644 << 16``,
``ZIP_DEFLATED`` level 9, secret-pattern deny-list.

CONTENT POLICY (repaired after the review + the integrator's fresh-clone
finding).  The zip is the ACCELERATOR BUNDLE — what the pod needs in order to
run the campaign — and nothing else:

* every GIT-TRACKED file under ``foundation_learner/`` (``git ls-files``).
  Tracked-ness is the definition, which is what makes a fresh clone of the
  pushed branch reproduce the same zip: a local scratch file that was never
  committed can no longer sneak into a release and make its hash unverifiable;
* every file under ``artifacts_fl/pregen/**`` (git-ignored because it is large,
  but BITWISE reproducible from ``scripts/pregenerate_all.py``, which contains
  no wall-clock value);

and it EXCLUDES, deliberately:

* ``foundation_learner/reports/**`` — the evidence JSONs (test report, smoke
  report, rehearsal reports) stay in GIT, where they are reviewable and
  history-tracked, but they are not part of what the accelerator runs.  Keeping
  them out also removes the packaging/validation ordering trap in which a
  validation run rewrites a file the zip has already hashed;
* ``FOUNDATION_LEARNER_V0_MANIFEST.json`` — it BINDS the zip hash, so it cannot
  be inside the thing it describes.  It is written AFTER packaging
  (``scripts/make_manifest.py``) and is tracked in git;
* ``SHA256SUMS`` as CONTENT — it covers everything else and is the LAST file
  written and added, so it cannot cover itself.

Also written:

* ``foundation_learner/SHA256SUMS`` — O1 two-space format over the whole
  content manifest, EXACT coverage (a bijection between the listed files and
  the files actually present: neither a missing nor an extra file passes);
* ``<zip>.sha256`` — the sidecar digest.

Usage::

    python -m foundation_learner.scripts.package_release --dry-run
    python -m foundation_learner.scripts.package_release
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

from foundation_learner.campaign import o1_isolation  # noqa: E402
from foundation_learner.campaign import result_verifier as rv  # noqa: E402
from foundation_learner.ecology.base import sha256_file  # noqa: E402

PACKAGE_NAME = "FOUNDATION_LEARNER_B200_V0"
ZIP_NAME = "FOUNDATION_LEARNER_B200_V0.2.0.zip"
SUMS_NAME = "SHA256SUMS"
MANIFEST_DIR_NAME = "MANIFESTS"
CAMPAIGN_MANIFEST_NAME = "FOUNDATION_LEARNER_V0_MANIFEST.json"

#: package-relative directories whose git-tracked contents stay OUT of the zip
EXCLUDED_PACKAGE_DIRS = ("reports",)
#: package-relative files that are tracked but must not be packaged
EXCLUDED_PACKAGE_FILES = (CAMPAIGN_MANIFEST_NAME, SUMS_NAME, ZIP_NAME,
                          ZIP_NAME + ".sha256")

#: mirrored into ``artifacts_fl/pregen*/MANIFESTS/`` so that packaging (and
#: review) can find them while the raw shard directories stay git-ignored.
PREGEN_MANIFEST_FILES = ("PREGEN_MANIFEST.json", "SHARD_SUMS.json",
                         "family_split_manifest.json")

EXCLUDED_DIRS = ("__pycache__", "local_runs", ".pytest_cache", ".git")
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".tmp", ".orig", ".rej")
EXCLUDED_NAMES = (SUMS_NAME, ZIP_NAME, ZIP_NAME + ".sha256")


class PackagingError(RuntimeError):
    """A packaging invariant refused."""


def _excluded(rel: str) -> bool:
    parts = rel.replace(os.sep, "/").split("/")
    if any(part in EXCLUDED_DIRS for part in parts):
        return True
    if parts[-1] in EXCLUDED_NAMES:
        return True
    return any(rel.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def git_tracked_files(repo_root: str, prefix: str) -> list[str]:
    """``git ls-files`` under ``prefix`` (absolute paths, existing files only).

    Refuses when ``repo_root`` is not a git work tree: a release built from an
    unversioned directory cannot be reproduced from the pushed branch, which is
    exactly the fresh-clone defect this policy repairs.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "ls-files", "-z", "--", prefix],
            capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackagingError(
            f"cannot list git-tracked files under {prefix!r} in {repo_root!r}: "
            f"{exc}. The release content policy is 'git-tracked', so the "
            "package must be built from a git work tree.") from exc
    names = [n for n in proc.stdout.decode("utf-8").split("\0") if n]
    out = []
    for rel in names:
        full = os.path.join(repo_root, rel)
        if os.path.isfile(full):
            out.append(full)
    if not out:
        raise PackagingError(
            f"no git-tracked files under {prefix!r}; refusing to build an "
            "empty release")
    return sorted(out)


def assert_clean_work_tree(repo_root: str, *, paths=("foundation_learner",)
                           ) -> dict:
    """A RELEASE may only be built from a committed tree.

    The content policy is "git-tracked", so an uncommitted new module is
    silently ABSENT from the bundle and a modified tracked file makes the zip
    differ from the branch it claims to come from.  Both break the property the
    policy exists for: a fresh clone of the pushed ref must reproduce this zip
    byte for byte.  A dry run may proceed (it is labelled a throwaway); a
    release refuses.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "status", "--porcelain", "--", *paths],
            capture_output=True, check=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackagingError(
            f"cannot check the work tree state of {repo_root!r}: {exc}") from exc
    package = os.path.join(repo_root, "foundation_learner")

    def relevant(path: str) -> bool:
        """Would this path have been PACKAGED if it were tracked and clean?"""
        full = os.path.join(repo_root, path.strip().strip('"'))
        rel = os.path.relpath(full, package).replace(os.sep, "/").rstrip("/")
        if rel.startswith(".."):
            return True                     # outside the package: report it
        return not _package_excluded(rel)

    entries = [line for line in proc.stdout.splitlines() if line.strip()]
    untracked = [e[3:] for e in entries if e.startswith("??") and relevant(e[3:])]
    modified = [e[3:] for e in entries
                if not e.startswith("??") and relevant(e[3:])]
    return {"clean": not (untracked or modified), "untracked": untracked,
            "modified": modified,
            "ignored_for_packaging": len(entries) - len(untracked) - len(modified)}


def _package_excluded(rel: str) -> bool:
    """Is a package-relative path excluded from the ACCELERATOR bundle?"""
    parts = rel.replace(os.sep, "/").split("/")
    if parts and parts[0] in EXCLUDED_PACKAGE_DIRS:
        return True
    if len(parts) == 1 and parts[0] in EXCLUDED_PACKAGE_FILES:
        return True
    return _excluded(rel)


def _tree(root: str, repo_root: str) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, repo_root).replace(os.sep, "/")
            if _excluded(rel):
                continue
            out.append(full)
    return sorted(out)


def mirror_pregen_manifests(pregen_root: str) -> dict:
    """Copy the pregen manifests into ``<pregen_root>/MANIFESTS/``.

    The raw shard directories are git-ignored (they are large and bitwise
    reproducible), but their manifests must stay findable — for packaging, for
    the campaign manifest, and for review.  The mirror is verified by hash on
    every run, so a stale copy is an error rather than a silent divergence.
    """
    pregen_root = os.path.abspath(pregen_root)
    target = os.path.join(pregen_root, MANIFEST_DIR_NAME)
    os.makedirs(target, exist_ok=True)
    mirrored = {}
    for name in PREGEN_MANIFEST_FILES:
        source = os.path.join(pregen_root, name)
        if not os.path.isfile(source):
            raise PackagingError(
                f"pregeneration is incomplete: {name} missing under "
                f"{pregen_root}")
        destination = os.path.join(target, name)
        payload = open(source, "rb").read()
        if not os.path.isfile(destination) or \
                open(destination, "rb").read() != payload:
            tmp = destination + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, destination)
        if sha256_file(source) != sha256_file(destination):  # pragma: no cover
            raise PackagingError(f"manifest mirror mismatch for {name}")
        mirrored[name] = sha256_file(destination)
    return mirrored


def content_manifest(repo_root: str, *, pregen_root: str | None,
                     include_pregen: bool = True) -> list[str]:
    """The explicit list of files the release contains (see the module docstring).

    GIT-TRACKED ``foundation_learner/**`` minus ``reports/**``, the campaign
    manifest and the release artefacts, plus ``artifacts_fl/pregen/**`` from
    disk.
    """
    repo_root = os.path.abspath(repo_root)
    package = os.path.join(repo_root, "foundation_learner")
    if not os.path.isdir(package):
        raise PackagingError(f"package directory not found: {package}")
    files = []
    for full in git_tracked_files(repo_root, "foundation_learner"):
        rel_to_package = os.path.relpath(full, package).replace(os.sep, "/")
        if _package_excluded(rel_to_package):
            continue
        files.append(full)
    if not files:
        raise PackagingError(
            "the content manifest is empty after applying the exclusions")
    if include_pregen:
        if not pregen_root or not os.path.isdir(pregen_root):
            raise PackagingError(
                f"pregenerated data not found at {pregen_root!r}; run "
                "scripts/pregenerate_all.py, or pass --no-pregen for a "
                "code-only dry run (which is NOT a release)")
        files += _tree(os.path.abspath(pregen_root), repo_root)
    return sorted(set(files))


def write_sha256sums(files: list[str], repo_root: str, out_path: str) -> str:
    lines = []
    for path in files:
        rel = os.path.relpath(path, repo_root).replace(os.sep, "/")
        lines.append(f"{sha256_file(path)}  {rel}")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    return sha256_file(out_path)


def verify_exact_coverage(sums_path: str, repo_root: str, *,
                          pregen_root: str | None,
                          include_pregen: bool = True) -> dict:
    """Bijection check: every listed file exists and every file is listed."""
    listed: dict[str, str] = {}
    with open(sums_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            digest, rel = line.rstrip("\n").split("  ", 1)
            if rel in listed:
                raise PackagingError(f"{sums_path}:{lineno}: duplicate entry {rel}")
            listed[rel] = digest
    present = {os.path.relpath(p, repo_root).replace(os.sep, "/")
               for p in content_manifest(repo_root, pregen_root=pregen_root,
                                         include_pregen=include_pregen)}
    missing = sorted(set(listed) - present)
    unlisted = sorted(present - set(listed))
    bad = []
    for rel, digest in sorted(listed.items()):
        full = os.path.join(repo_root, rel)
        if os.path.isfile(full) and sha256_file(full) != digest:
            bad.append(rel)
    report = {"listed": len(listed), "present": len(present),
              "missing": missing, "unlisted": unlisted, "mismatched": bad,
              "ok": not missing and not unlisted and not bad}
    if not report["ok"]:
        raise PackagingError(
            f"SHA256SUMS coverage is not exact: {len(missing)} listed-but-absent, "
            f"{len(unlisted)} present-but-unlisted, {len(bad)} hash mismatches "
            f"(first: {missing[:1]} {unlisted[:1]} {bad[:1]})")
    return report


def build(repo_root: str, *, pregen_root: str | None, out_dir: str,
          include_pregen: bool = True, label: str = "RELEASE",
          guard: o1_isolation.IsolationGuard | None = None) -> dict:
    """Write SHA256SUMS, the deterministic zip, and its sidecar."""
    guard = guard or o1_isolation.default_guard()
    repo_root = os.path.abspath(repo_root)
    work_tree = assert_clean_work_tree(repo_root)
    if label == "RELEASE" and not work_tree["clean"]:
        raise PackagingError(
            "REFUSED: the work tree is not clean, so this zip could not be "
            "reproduced from the pushed branch. The content policy packages "
            "GIT-TRACKED files, which means an uncommitted module is silently "
            "MISSING from the release. Commit first.\n"
            f"  untracked: {work_tree['untracked'][:10]}\n"
            f"  modified:  {work_tree['modified'][:10]}")
    os.makedirs(out_dir, exist_ok=True)
    mirrored = None
    if include_pregen and pregen_root:
        mirrored = mirror_pregen_manifests(pregen_root)
    files = content_manifest(repo_root, pregen_root=pregen_root,
                             include_pregen=include_pregen)
    for path in files:
        rv.check_no_secrets(path)
    sums_path = os.path.join(repo_root, "foundation_learner", SUMS_NAME)
    sums_sha256 = write_sha256sums(files, repo_root, sums_path)
    coverage = verify_exact_coverage(sums_path, repo_root,
                                     pregen_root=pregen_root,
                                     include_pregen=include_pregen)
    # SHA256SUMS is written LAST and added LAST: it covers every other file in
    # the bundle and therefore cannot cover itself.
    zip_path = os.path.join(out_dir, ZIP_NAME)
    zip_sha256 = rv.deterministic_zip(files + [sums_path], repo_root, zip_path,
                                      guard=guard)
    sidecar = zip_path + ".sha256"
    with open(sidecar, "w", encoding="utf-8") as fh:
        fh.write(f"{zip_sha256}  {ZIP_NAME}\n")
    return {
        "schema": "flb200.package_release.v1",
        "label": label,
        "repo_root": repo_root,
        "zip_path": zip_path,
        "zip_name": ZIP_NAME,
        "zip_sha256": zip_sha256,
        "zip_bytes": os.path.getsize(zip_path),
        "sha256sums_path": sums_path,
        "sha256sums_sha256": sums_sha256,
        "entries": len(files) + 1,
        "coverage": coverage,
        "include_pregen": bool(include_pregen),
        "pregen_root": pregen_root if include_pregen else None,
        "pregen_manifests_mirrored": mirrored,
        "work_tree": work_tree,
        "content_policy": {
            "package_files": "git ls-files -- foundation_learner",
            "pregen_files": "artifacts_fl/pregen/** (bitwise reproducible)",
            "excluded_package_dirs": list(EXCLUDED_PACKAGE_DIRS),
            "excluded_package_files": list(EXCLUDED_PACKAGE_FILES),
            "why_reports_excluded": (
                "the zip is the ACCELERATOR bundle, not the evidence archive; "
                "reports/ stays in git, where it is reviewable and tracked"),
            "why_manifest_excluded": (
                "FOUNDATION_LEARNER_V0_MANIFEST.json binds the zip hash, so it "
                "cannot be inside the zip; it is written AFTER packaging"),
            "why_sums_not_content": (
                "SHA256SUMS covers every other file and is written and added "
                "last, so it never covers itself"),
        },
        "excluded": {"dirs": list(EXCLUDED_DIRS),
                     "suffixes": list(EXCLUDED_SUFFIXES),
                     "names": list(EXCLUDED_NAMES)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic FL release zip")
    parser.add_argument("--repo-root", default=_PKG_PARENT)
    parser.add_argument("--pregen",
                        default=os.path.join(_PKG_PARENT, "artifacts_fl", "pregen"),
                        help="pre-generated data root")
    parser.add_argument("--out", default=_PKG_PARENT,
                        help="where to write the zip (default: repo root)")
    parser.add_argument("--no-pregen", action="store_true",
                        help="code-only build; NOT a release (dry runs only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build into reports/local_runs/ and label it a "
                             "THROWAWAY dry run")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out
    label = "RELEASE"
    if args.dry_run:
        out_dir = os.path.join(args.repo_root, "foundation_learner", "reports",
                               "local_runs", "package_dry_run")
        label = "THROWAWAY_DRY_RUN"
    report = build(args.repo_root, pregen_root=args.pregen, out_dir=out_dir,
                   include_pregen=not args.no_pregen, label=label)
    if not report["work_tree"]["clean"]:
        print("WARNING: the work tree is NOT clean; this throwaway build does "
              "NOT match the branch. Untracked files are absent from the zip "
              f"({report['work_tree']['untracked'][:5]}).", file=sys.stderr)
    print(json.dumps({k: v for k, v in report.items() if k != "coverage"},
                     indent=2, sort_keys=True))
    print(f"coverage: {report['coverage']['listed']} files, exact="
          f"{report['coverage']['ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
