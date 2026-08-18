#!/usr/bin/env python3
"""Pod-side pregen ingestion: bring the episode corpus onto the pod.

The container bakes the FL source, but NOT the ~480 MB pregenerated episode
corpus, which is neither in Git nor in the image.  ``campaign/entry.py``
refuses a session whose ``pregen_root`` does not exist ("the B200 never
generates episodes"), so without this step a combined O1 -> FL rental
acquires an accelerator and then refuses at the FL handover — paying for
capacity it structurally cannot use.

Source of truth is ``FL_PREGEN_SOURCE`` (an ``hf://<ns>/<repo>[/<prefix>]``
staging URI).  Network access goes through the isolated ``hf_transfer``
subprocess helper, so the supervisor stays HF_HUB_OFFLINE-locked.

Nothing fetched here is trusted: every shard is re-hashed against
``MANIFESTS/SHARD_SUMS.json`` before the tree is moved into place, and the
sealed shards are verified by their on-disk digest so SEALED_TEST is never
opened or decrypted to check it.  A tree that fails verification is left in
the staging directory and the session refuses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

from .hf_transfer import child_env
from .o1_isolation import MODE_READ, MODE_WRITE, guard_path

DEFAULT_PREGEN_ROOT = "/workspace/foundation_learner/artifacts_fl/pregen"
DEFAULT_REMOTE_PREFIX = "artifacts_fl/pregen"
SHARD_SUMS_REL = os.path.join("MANIFESTS", "SHARD_SUMS.json")
PREGEN_MANIFEST_REL = os.path.join("MANIFESTS", "PREGEN_MANIFEST.json")


class PregenFetchError(RuntimeError):
    pass


def parse_source(uri: str) -> tuple[str, str]:
    """``hf://ns/repo[/prefix]`` -> (``ns/repo``, prefix)."""
    if not uri.startswith("hf://"):
        raise PregenFetchError(
            f"pregen source {uri!r} is not an hf:// staging URI; the pod has "
            f"no other ingestion path")
    parts = uri[len("hf://"):].strip("/").split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise PregenFetchError(f"malformed pregen staging URI {uri!r}")
    prefix = "/".join(parts[2:]) if len(parts) > 2 else DEFAULT_REMOTE_PREFIX
    return "/".join(parts[:2]), prefix


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(guard_path(path, MODE_READ), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_tree(root: str) -> dict:
    """Re-hash every shard the manifest declares.  Raises on any defect."""
    sums_path = os.path.join(root, SHARD_SUMS_REL)
    if not os.path.isfile(sums_path):
        raise PregenFetchError(
            f"{SHARD_SUMS_REL} is absent from {root!r}; an unverifiable "
            f"episode corpus is not usable")
    with open(guard_path(sums_path, MODE_READ), encoding="utf-8") as fh:
        sums = json.load(fh)
    if sums.get("schema") != "fl.shard_sums.v1":
        raise PregenFetchError(
            f"{SHARD_SUMS_REL} schema {sums.get('schema')!r} is not "
            f"fl.shard_sums.v1")
    shards = sums.get("shards") or []
    if not shards:
        raise PregenFetchError(f"{SHARD_SUMS_REL} declares no shards")
    missing, mismatched, sealed = [], [], 0
    for shard in shards:
        path = os.path.join(root, shard["path"])
        if not os.path.isfile(path):
            missing.append(shard["path"])
            continue
        # The on-disk digest covers sealed and unsealed shards alike, so
        # SEALED_TEST is verified without ever being read as episodes.
        if (os.path.getsize(path) != shard["bytes"]
                or sha256_file(path) != shard["sha256"]):
            mismatched.append(shard["path"])
        if shard.get("sealed"):
            sealed += 1
    if missing or mismatched:
        raise PregenFetchError(
            f"pregen verification FAILED in {root!r}: "
            f"{len(missing)} missing {missing[:4]}, "
            f"{len(mismatched)} corrupt {mismatched[:4]}")
    if not os.path.isfile(os.path.join(root, PREGEN_MANIFEST_REL)):
        raise PregenFetchError(f"{PREGEN_MANIFEST_REL} is absent from {root!r}")
    return {"shards_verified": len(shards), "sealed_shards": sealed}


def _helper_error(text: str, limit: int = 400) -> str:
    """The last meaningful line, not a raw tail.

    A Python traceback ends with the exception line — the one thing the
    operator needs ("Invalid user token", "401 Unauthorized").  Slicing the
    last N characters instead lands mid-frame and prints source fragments.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1][:limit] if lines else "(helper produced no output)"


def _run_helper(args: list[str], timeout: float) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "foundation_learner.campaign.hf_transfer",
         *args],
        capture_output=True, text=True, timeout=timeout,
        env=child_env(os.environ.get("HF_TOKEN")))
    if proc.returncode != 0:
        raise PregenFetchError(
            f"pregen fetch failed: {_helper_error(proc.stdout + proc.stderr)}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise PregenFetchError("pregen fetch produced no result") from None


def fetch(source_uri: str, pregen_root: str = DEFAULT_PREGEN_ROOT,
          timeout: float = 7200.0, runner=None) -> dict:
    """Materialise and verify the pregen corpus at ``pregen_root``."""
    if os.path.isdir(pregen_root):
        # Already mounted, or a resumed pod: verification still has the last
        # word on whether the bytes are the right ones.
        return {"repo": None, "action": "already_present",
                "pregen_root": pregen_root, **verify_tree(pregen_root)}
    repo, prefix = parse_source(source_uri)
    staging = pregen_root.rstrip("/") + ".incoming"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    run = runner or (lambda args: _run_helper(args, timeout))
    run(["snapshot", "--repo", repo, "--prefix", prefix, "--local", staging])
    fetched = os.path.join(staging, prefix)
    if not os.path.isdir(fetched):
        raise PregenFetchError(
            f"prefix {prefix!r} is absent from the staging repo {repo!r} "
            f"after fetch; the FL ladder cannot proceed")
    # Verify BEFORE publishing: a tree that fails stays in .incoming, so the
    # session never sees a half-fetched pregen root it would treat as valid.
    stats = verify_tree(fetched)
    os.makedirs(os.path.dirname(os.path.abspath(pregen_root)) or "/",
                exist_ok=True)
    os.replace(fetched, pregen_root)
    shutil.rmtree(staging, ignore_errors=True)
    return {"repo": repo, "prefix": prefix, "action": "fetched",
            "pregen_root": pregen_root, **stats}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=os.environ.get("FL_PREGEN_SOURCE", ""))
    p.add_argument("--pregen-root", default=None)
    p.add_argument("--config", default=None,
                   help="session config; supplies pregen_root when given")
    p.add_argument("--out")
    a = p.parse_args()

    pregen_root, source = a.pregen_root, a.source
    if a.config and (pregen_root is None or not source):
        try:
            with open(guard_path(a.config, MODE_READ), encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"REFUSED: cannot read {a.config}: {exc}", file=sys.stderr)
            return 2
        pregen_root = pregen_root or cfg.get("pregen_root")
        # the session config is the operator's single binding surface; the
        # env var stays supported for a standalone/rehearsal invocation
        source = source or cfg.get("fl_pregen_source") or ""
    pregen_root = pregen_root or DEFAULT_PREGEN_ROOT

    if not os.path.isdir(pregen_root) and (
            not source or source.startswith("UNRESOLVED")):
        print(f"REFUSED: {pregen_root} is absent and FL_PREGEN_SOURCE is "
              f"unset; the pod cannot obtain the episode corpus and the FL "
              f"ladder cannot run", file=sys.stderr)
        return 2
    try:
        report = fetch(source, pregen_root)
    except PregenFetchError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        with open(guard_path(a.out, MODE_WRITE), "w", encoding="utf-8") as fh:
            fh.write(payload)
    print(payload.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
