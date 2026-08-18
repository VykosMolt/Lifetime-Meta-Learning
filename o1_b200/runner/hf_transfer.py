#!/usr/bin/env python3
"""Isolated Hugging Face transfer helper for the durable store.

Why this exists as a SUBPROCESS rather than an in-process call:

The pod sets ``HF_HUB_OFFLINE=1`` so that model/tokenizer/artifact loading
can never reach the network — the checkpoint must come from the verified
mount and nothing else.  huggingface_hub honours that flag by mounting an
``OfflineAdapter`` that raises for EVERY request, uploads included, and it
reads the flag into a module constant at import time.  So the same flag that
protects the science also disables the durability, external-commitment and
result-transfer paths that preemptible execution depends on.

Rather than weaken the invariant globally, the hub is reachable ONLY here:
the parent process stays offline-locked (so a model load cannot silently
fetch), and this helper is spawned with the offline flags stripped from its
environment.  It never imports transformers and never loads a model — its
whole job is byte transfer with digests.

Every operation prints one JSON object on stdout.  Credentials arrive via
the environment and are never echoed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

#: Flags that must be cleared for the hub to be reachable at all.
OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "TRANSFERS_OFFLINE", "TRANSFORMERS_OFFLINE")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def child_env(token: str | None = None) -> dict:
    """Environment for the helper: offline flags stripped, telemetry off."""
    env = {k: v for k, v in os.environ.items() if k not in OFFLINE_FLAGS}
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if token:
        env["HF_TOKEN"] = token
    return env


def _assert_online_capable() -> None:
    """Fail loudly if the offline flag survived into this process."""
    from huggingface_hub import constants
    if getattr(constants, "HF_HUB_OFFLINE", False) or \
            os.environ.get("HF_HUB_OFFLINE"):
        raise SystemExit(
            "hf_transfer: HF_HUB_OFFLINE is still set in the helper "
            "process; the durable store cannot operate (the parent must "
            "spawn this helper with child_env())")


def do_push(repo_id: str, local: str, remote_rel: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import HfApi
    digest = sha256_file(local)
    HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
        path_or_fileobj=local, path_in_repo=remote_rel,
        repo_id=repo_id, repo_type="model")
    return {"remote": remote_rel, "sha256": digest,
            "bytes": os.path.getsize(local)}


def do_fetch(repo_id: str, remote_rel: str, local: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import hf_hub_download
    got = hf_hub_download(repo_id=repo_id, filename=remote_rel,
                          token=os.environ.get("HF_TOKEN"))
    os.makedirs(os.path.dirname(os.path.abspath(local)) or ".", exist_ok=True)
    tmp = local + ".tmp"
    shutil.copyfile(got, tmp)
    digest = sha256_file(tmp)
    if digest != sha256_file(got):
        os.remove(tmp)
        raise SystemExit("hf_transfer: local copy corrupted during fetch")
    os.replace(tmp, local)
    return {"local": local, "sha256": digest}


def do_list(repo_id: str, prefix: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import HfApi
    files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo_id)
    return {"files": sorted(f for f in files if f.startswith(prefix))}


def do_snapshot(repo_id: str, prefix: str, local: str) -> dict:
    """Materialise a whole staged directory (e.g. the model checkpoint).

    Used by the pod-side artifact fetch: the checkpoint is many files and
    is deliberately never baked into the image, so it is pulled here and
    then tree-hash verified by deploy/verify_artifacts.py before any model
    load.  Returns the fetched file list for the caller's record.
    """
    _assert_online_capable()
    from huggingface_hub import snapshot_download
    patterns = [f"{prefix}/**", prefix] if prefix else None
    got = snapshot_download(repo_id=repo_id, local_dir=local,
                            allow_patterns=patterns,
                            token=os.environ.get("HF_TOKEN"))
    files = []
    for dirpath, _dirs, names in os.walk(got):
        for name in sorted(names):
            files.append(os.path.relpath(os.path.join(dirpath, name), got))
    return {"local": got, "files": sorted(files), "count": len(files)}


#: Where the write probe writes.  A dedicated prefix so the probe can never
#: collide with a result key, and so an interrupted probe leaves an obviously
#: non-scientific artefact behind.
PREFLIGHT_PREFIX = ".preflight"


def do_scope(repo_id: str, mode: str) -> dict:
    """Prove the token can do what the session is about to depend on.

    A single HF_TOKEN has to cover BOTH directions: read, to pull the staged
    checkpoint, and write, to push committed rows and checkpoints to the
    durable mirror.  A read-only token passes artifact ingestion and every
    gate, then silently loses every durability push — which under
    interruptible capacity means eviction destroys the run.  Read is proven
    by auth_check; write can only be proven by writing, so the probe uploads
    a few bytes under PREFLIGHT_PREFIX and deletes them again.
    """
    _assert_online_capable()
    from uuid import uuid4
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    who = api.whoami()
    auth = (who.get("auth") or {}).get("accessToken") or {}
    out = {"repo": repo_id, "mode": mode,
           "identity": who.get("name") or "unknown",
           "declared_role": auth.get("role") or "unreported",
           "read": False, "write": False}
    api.auth_check(repo_id, repo_type="model")
    out["read"] = True
    if mode == "write":
        rel = f"{PREFLIGHT_PREFIX}/scope_{uuid4().hex}"
        api.upload_file(path_or_fileobj=b"o1-b300 write-scope probe\n",
                        path_in_repo=rel, repo_id=repo_id, repo_type="model")
        api.delete_file(path_in_repo=rel, repo_id=repo_id, repo_type="model")
        out["write"] = True
        out["probe_path"] = rel
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command",
                   choices=["push", "fetch", "list", "snapshot", "scope"])
    p.add_argument("--repo", required=True)
    p.add_argument("--local")
    p.add_argument("--remote-rel")
    p.add_argument("--prefix", default="")
    p.add_argument("--mode", choices=["read", "write"], default="read")
    a = p.parse_args()
    if a.command == "push":
        out = do_push(a.repo, a.local, a.remote_rel)
    elif a.command == "fetch":
        out = do_fetch(a.repo, a.remote_rel, a.local)
    elif a.command == "snapshot":
        out = do_snapshot(a.repo, a.prefix, a.local)
    elif a.command == "scope":
        out = do_scope(a.repo, a.mode)
    else:
        out = do_list(a.repo, a.prefix)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
