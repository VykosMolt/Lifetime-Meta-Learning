"""Continuous off-pod durability for the FL session under preemption.

INTERRUPTIBLE accelerator capacity can be evicted with ~5s notice, and the
session pod carries container disk only — anything not already off-pod is
lost.  This module mirrors the FL session's durable state continuously:

  * the supervisor journal (the resume source of truth) after every append;
  * every atomic training checkpoint (payload + manifest, manifest LAST so
    a half-pushed checkpoint is never referenced);
  * O1-close receipts and other session receipts as they are written.

On a fresh pod after eviction, ``restore_all`` repopulates the FL output
tree from the mirror BEFORE the supervisor reads its journal, after which
the EXISTING resume machinery proceeds exactly as it would after a
same-pod restart: completed states skip, training resumes from the last
valid atomic checkpoint, and work performed after the last durable
checkpoint is simply rerun.  No hyperparameter may change on resume — the
checkpoint manifest binds the arm configuration hash and the trainer
refuses a mismatch.

Restore NEVER overwrites an existing local file (local atomic writes win)
and hash-verifies every fetched object; a corrupt mirrored file is
quarantined, never restored.  This module deliberately imports nothing
from the O1 package (isolation contract): the store pattern is
re-implemented, not shared.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

_PREFIX = "fl_durable"


class DurabilityError(RuntimeError):
    pass


def _sha256_file(path: str, guard=None) -> str:
    if guard is not None:
        path = guard(path, "read")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _LocalStore:
    def __init__(self, root: str, guard=None):
        self.guard = guard or (lambda path, mode="read": path)
        self.root = os.path.abspath(self.guard(root, "write"))
        os.makedirs(self.root, exist_ok=True)

    def _p(self, rel: str) -> str:
        p = os.path.abspath(os.path.join(self.root, rel))
        if not p.startswith(self.root + os.sep):
            raise DurabilityError(f"path escape refused: {rel!r}")
        return p

    def push(self, local: str, rel: str) -> None:
        local = self.guard(local, "read")
        dest = self.guard(self._p(rel), "write")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".tmp"
        shutil.copyfile(local, tmp)
        if _sha256_file(tmp, self.guard) != _sha256_file(local, self.guard):
            os.remove(tmp)
            raise DurabilityError(f"push corruption for {rel}")
        os.replace(tmp, dest)

    def fetch(self, rel: str, local: str) -> None:
        src = self.guard(self._p(rel), "read")
        local = self.guard(local, "write")
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        tmp = local + ".tmp"
        shutil.copyfile(src, tmp)
        if _sha256_file(tmp, self.guard) != _sha256_file(src, self.guard):
            os.remove(tmp)
            raise DurabilityError(f"fetch corruption for {rel}")
        os.replace(tmp, local)

    def list_all(self) -> list[str]:
        out = []
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                out.append(os.path.relpath(os.path.join(dirpath, name),
                                           self.root))
        return sorted(out)


class _HfStore:
    """Private Hugging Face repo store (token via HF_TOKEN).

    Every hub call runs in the isolated ``campaign.hf_transfer`` subprocess
    so the supervisor process itself stays offline-locked (see that module),
    and every transfer is digest-verified end to end — an unverified copy
    would let a silently-corrupted mirror be restored as if it were sound.
    Prefix filtering happens inside the helper, so this process never sees
    non-FL filenames from a shared repository.
    """

    def __init__(self, repo_id: str, guard=None, runner=None,
                 timeout: float = 3600.0):
        self.guard = guard or (lambda path, mode="read": path)
        self.repo_id = repo_id
        self.token = os.environ.get("HF_TOKEN")
        self.timeout = timeout
        self._runner = runner or self._spawn

    def _spawn(self, args: list[str]) -> dict:
        import subprocess
        import sys

        from .hf_transfer import child_env
        proc = subprocess.run(
            [sys.executable, "-m", "foundation_learner.campaign.hf_transfer",
             *args],
            capture_output=True, text=True, timeout=self.timeout,
            env=child_env(self.token))
        if proc.returncode != 0:
            detail = (proc.stdout + proc.stderr)[-400:]
            if self.token:
                detail = detail.replace(self.token, "[REDACTED]")
            raise DurabilityError(f"hf {args[0]} failed: {detail}")
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            raise DurabilityError(
                f"hf {args[0]} produced no result object") from None

    def push(self, local: str, rel: str) -> None:
        local = self.guard(local, "read")
        out = self._runner(["push", "--repo", self.repo_id,
                            "--local", local, "--remote-rel", rel])
        if out.get("sha256") != _sha256_file(local):
            raise DurabilityError(f"push digest disagreement for {rel}")

    def fetch(self, rel: str, local: str) -> None:
        local = self.guard(local, "write")
        out = self._runner(["fetch", "--repo", self.repo_id,
                            "--remote-rel", rel, "--local", local])
        if out.get("sha256") != _sha256_file(local):
            raise DurabilityError(f"fetch digest disagreement for {rel}")

    def list_all(self) -> list[str]:
        out = self._runner(["list", "--repo", self.repo_id,
                            "--prefix", _PREFIX])
        return list(out.get("files") or [])


def _store_for(destination: str, guard=None):
    if destination.startswith("hf://"):
        parts = destination[len("hf://"):].split("/")
        if len(parts) < 2 or not all(parts[:2]):
            raise DurabilityError(f"malformed destination {destination!r}")
        return _HfStore("/".join(parts[:2]), guard=guard)
    return _LocalStore(destination, guard=guard)


class FlDurableMirror:
    """Mirrors files under ``run_root`` to the durable destination."""

    def __init__(self, destination: str, run_root: str, on_event=None,
                 guard=None):
        # guard: the campaign isolation guard (o1_isolation.IsolationGuard
        # .guard); every local path this mirror touches is routed through it
        self.guard = guard or (lambda path, mode="read": path)
        self.store = _store_for(destination, guard=self.guard)
        self.run_root = os.path.abspath(run_root)
        self.on_event = on_event or (lambda *a, **k: None)

    def _rel(self, path: str) -> str:
        ap = os.path.abspath(path)
        if not ap.startswith(self.run_root + os.sep):
            raise DurabilityError(
                f"{path!r} is outside the mirrored run root")
        return f"{_PREFIX}/" + os.path.relpath(ap, self.run_root)

    def push_file(self, path: str) -> None:
        """Best-effort single-file mirror; failure is reported loudly but
        never destroys local state (it widens the eviction-loss window)."""
        try:
            self.store.push(path, self._rel(path))
        except Exception as exc:  # noqa: BLE001
            self.on_event("FL_DURABILITY_PUSH_FAILED",
                          path=os.path.basename(path), error=str(exc)[:200])

    def push_checkpoint(self, payload_path: str, manifest_path: str) -> None:
        """Checkpoint mirror: payload FIRST, manifest LAST — a checkpoint
        whose manifest is durable is guaranteed completely durable."""
        self.store.push(payload_path, self._rel(payload_path))
        self.store.push(manifest_path, self._rel(manifest_path))
        self.on_event("FL_CHECKPOINT_MIRRORED",
                      manifest=os.path.basename(manifest_path))

    def restore_all(self) -> dict:
        """Repopulate the run root from the mirror.  Existing local files
        are never overwritten; a corrupt fetch is quarantined."""
        restored, skipped, corrupt = 0, 0, 0
        quarantine = os.path.join(self.run_root, "durability_quarantine")
        for rel in self.store.list_all():
            if not rel.startswith(_PREFIX + "/"):
                continue
            local = os.path.join(self.run_root,
                                 rel[len(_PREFIX) + 1:])
            if os.path.exists(local):
                skipped += 1
                continue
            tmp = local + ".restoring"
            try:
                self.store.fetch(rel, tmp)
                os.replace(tmp, local)
                restored += 1
            except Exception as exc:  # noqa: BLE001
                corrupt += 1
                os.makedirs(quarantine, exist_ok=True)
                if os.path.exists(tmp):
                    os.replace(tmp, os.path.join(
                        quarantine, os.path.basename(local) + ".corrupt"))
                self.on_event("FL_DURABILITY_RESTORE_CORRUPT",
                              path=os.path.basename(local),
                              error=str(exc)[:200])
        report = {"restored": restored, "skipped_existing": skipped,
                  "corrupt_quarantined": corrupt}
        self.on_event("FL_DURABILITY_RESTORED", **report)
        return report


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["restore"])
    p.add_argument("--destination", required=True)
    p.add_argument("--run-root", required=True)
    a = p.parse_args()
    mirror = FlDurableMirror(a.destination, a.run_root)
    print(json.dumps(mirror.restore_all()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
