#!/usr/bin/env python3
"""Isolated Hugging Face transfer helper for the FL durable mirror.

Deliberately self-contained (no O1 import — isolation contract) and
deliberately a SUBPROCESS.  The pod sets ``HF_HUB_OFFLINE=1`` so that model
and tokenizer loading can never reach the network; huggingface_hub honours
that by mounting an adapter that raises for every request, uploads
included, and it reads the flag into a module constant at import time.  The
same flag that protects the science would therefore disable the durability
mirror that preemptible execution depends on.

So the hub is reachable ONLY here: the supervisor process stays
offline-locked, and this helper is spawned with the offline flags stripped.
It never imports torch or transformers and never loads a model.

Each operation prints one JSON object on stdout.  Filtering happens INSIDE
this helper, so the FL process never even sees non-FL filenames from a
shared repository.  Credentials arrive via the environment, never echoed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "TRANSFERS_OFFLINE", "TRANSFORMERS_OFFLINE")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def child_env(token: str | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in OFFLINE_FLAGS}
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if token:
        env["HF_TOKEN"] = token
    return env


def _assert_online_capable() -> None:
    from huggingface_hub import constants
    if getattr(constants, "HF_HUB_OFFLINE", False) or \
            os.environ.get("HF_HUB_OFFLINE"):
        raise SystemExit(
            "hf_transfer: HF_HUB_OFFLINE is still set in the helper "
            "process; the FL durable mirror cannot operate")


def do_push(repo_id: str, local: str, remote_rel: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import HfApi
    digest = sha256_file(local)
    HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
        path_or_fileobj=local, path_in_repo=remote_rel,
        repo_id=repo_id, repo_type="model")
    return {"remote": remote_rel, "sha256": digest}


def do_fetch(repo_id: str, remote_rel: str, local: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import hf_hub_download
    got = hf_hub_download(repo_id=repo_id, filename=remote_rel,
                          token=os.environ.get("HF_TOKEN"))
    os.makedirs(os.path.dirname(os.path.abspath(local)) or ".", exist_ok=True)
    tmp = local + ".tmp"
    shutil.copyfile(got, tmp)
    digest = sha256_file(tmp)
    os.replace(tmp, local)
    return {"local": local, "sha256": digest}


#: Dedicated prefix for the write probe, so it can never collide with a
#: journal or checkpoint key.
PREFLIGHT_PREFIX = ".preflight"


def do_scope(repo_id: str, mode: str) -> dict:
    """Prove the token can do what the session is about to depend on.

    One HF_TOKEN covers both the pregen read and the durable-mirror write.
    A read-only token pulls the corpus, trains for hours, and loses every
    journal and checkpoint push — under interruptible capacity that means
    eviction destroys the arm.  Read is proven by auth_check; write can only
    be proven by writing, so the probe uploads a few bytes and deletes them.
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
        api.upload_file(path_or_fileobj=b"fl-b200 write-scope probe\n",
                        path_in_repo=rel, repo_id=repo_id, repo_type="model")
        api.delete_file(path_in_repo=rel, repo_id=repo_id, repo_type="model")
        out["write"] = True
        out["probe_path"] = rel
    return out


def do_snapshot(repo_id: str, prefix: str, local: str) -> dict:
    """Materialise a whole staged directory (the pregenerated episode tree).

    The ~480 MB pregen corpus is neither in Git nor in the container image,
    so a fresh pod has to pull it before the ladder can start.  Nothing here
    is trusted: fetch_pregen.py re-hashes every shard against SHARD_SUMS.json
    afterwards.
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


def do_list(repo_id: str, prefix: str) -> dict:
    _assert_online_capable()
    from huggingface_hub import HfApi
    files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo_id)
    return {"files": sorted(f for f in files if f.startswith(prefix))}


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
