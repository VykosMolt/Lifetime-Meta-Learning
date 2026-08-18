#!/usr/bin/env python3
"""Pod-side credential-scope preflight: prove the token before spending time.

One HF_TOKEN has to serve both directions of an interruptible session:

  * READ   ``O1_B200_ARTIFACT_SOURCE``   — the staged checkpoint and cohorts;
  * WRITE  ``O1_B200_RESULT_DESTINATION`` — committed rows and checkpoints,
    pushed continuously so an eviction is recoverable.

A read-only token passes artifact ingestion, passes every hardware and
scientific gate, runs for hours, and then loses every durability push.  Under
interruptible capacity that is the worst available failure: the run looks
healthy right up to the eviction that destroys it.  Nothing else in the stack
detects it, because nothing else writes until there are results to write.

So this runs FIRST, before the multi-gigabyte fetch, and it fails closed.
Read is proven by auth_check; write is proven by writing — the probe uploads
a few bytes under a dedicated ``.preflight/`` prefix and deletes them again.
Non-``hf://`` destinations (a mounted path) are checked directly instead.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .hf_transfer import child_env

READ_ENV = "O1_B200_ARTIFACT_SOURCE"
WRITE_ENV = "O1_B200_RESULT_DESTINATION"


class ScopeError(RuntimeError):
    pass


def parse_repo(uri: str) -> str | None:
    """``hf://ns/repo[/prefix]`` -> ``ns/repo``; None for anything else."""
    if not uri or not uri.startswith("hf://"):
        return None
    parts = uri[len("hf://"):].strip("/").split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ScopeError(f"malformed hf:// URI {uri!r}")
    return "/".join(parts[:2])


def _helper_error(text: str, limit: int = 400) -> str:
    """The last meaningful line, not a raw tail.

    A Python traceback ends with the exception line — the one thing the
    operator needs ("Invalid user token", "401 Unauthorized").  Slicing the
    last N characters instead lands mid-frame and prints source fragments.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1][:limit] if lines else "(helper produced no output)"


def _run_helper(repo: str, mode: str, timeout: float) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "o1_b200.runner.hf_transfer", "scope",
         "--repo", repo, "--mode", mode],
        capture_output=True, text=True, timeout=timeout,
        env=child_env(os.environ.get("HF_TOKEN")))
    if proc.returncode != 0:
        from ..provider.runpod.redaction import redact
        raise ScopeError(redact(
            f"{mode.upper()} scope check failed for {repo}: "
            f"{_helper_error(proc.stdout + proc.stderr)}"))
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise ScopeError(
            f"{mode} scope check for {repo} produced no result") from None


def _check_local_write(path: str) -> dict:
    probe = os.path.join(path, ".preflight_write_probe")
    try:
        os.makedirs(path, exist_ok=True)
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("probe\n")
        os.remove(probe)
    except OSError as exc:
        raise ScopeError(
            f"the durable destination {path!r} is not writable: {exc}") from None
    return {"repo": path, "mode": "write", "read": True, "write": True,
            "identity": "local-filesystem"}


def check(read_uri: str, write_uri: str, timeout: float = 300.0,
          runner=None) -> dict:
    run = runner or (lambda repo, mode: _run_helper(repo, mode, timeout))
    checks = []
    read_repo = parse_repo(read_uri)
    write_repo = parse_repo(write_uri)
    if read_repo:
        checks.append(run(read_repo, "read"))
    if write_repo:
        # Same repo in both directions still needs the write probe: read
        # access says nothing about whether pushes will be accepted.
        checks.append(run(write_repo, "write"))
    elif write_uri and not write_uri.startswith("UNRESOLVED"):
        checks.append(_check_local_write(write_uri))
    else:
        # production_entry requires this anyway; refusing here means the
        # session dies in seconds rather than after the gates.  A verified
        # read with no verified write is the exact shape of the failure
        # this preflight exists to prevent.
        raise ScopeError(
            f"{WRITE_ENV} is unset or UNRESOLVED; an interruptible session "
            f"with no durable destination loses every committed row on "
            f"eviction")
    if not checks:
        raise ScopeError(
            f"neither {READ_ENV} nor {WRITE_ENV} names a destination this "
            f"preflight can verify; refusing to start a session whose "
            f"durability is unproven")
    for res in checks:
        if res["mode"] == "write" and not res.get("write"):
            raise ScopeError(
                f"the token can read {res['repo']} but cannot write to it; "
                f"an interruptible session with no durable push loses "
                f"everything on eviction")
    return {"schema": "o1b300.hf_scope_report.v1", "checks": checks}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--read-source", default=os.environ.get(READ_ENV, ""))
    p.add_argument("--write-destination", default=os.environ.get(WRITE_ENV, ""))
    p.add_argument("--out")
    a = p.parse_args()
    # A token is only required when a hub repo is actually involved: a pod
    # with mounted artifacts and a mounted durable path needs none, and
    # refusing that configuration would be an invented requirement.
    needs_token = any(str(uri).startswith("hf://")
                      for uri in (a.read_source, a.write_destination))
    if needs_token and not os.environ.get("HF_TOKEN"):
        print("REFUSED: HF_TOKEN is unset but the session names an hf:// "
              "source or destination; the pod can neither fetch the staged "
              "checkpoint nor push results", file=sys.stderr)
        return 2
    try:
        report = check(a.read_source, a.write_destination)
    except ScopeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
    print(payload.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
