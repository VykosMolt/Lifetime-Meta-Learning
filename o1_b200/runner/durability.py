"""Continuous off-pod durability for preemptible execution.

INTERRUPTIBLE capacity can be evicted at any moment with ~5s notice, and
the session's pods carry container disk only (no persistent volume), so
anything not already off-pod is lost.  This module makes committed state
durable continuously:

  * O1: every atomically committed row file (persistence.py rows/<id>.json)
    is mirrored to the durable store; on a fresh pod, restore() pulls the
    mirrored rows back into the run directory BEFORE resume, after which
    the existing resume logic applies unchanged (identity check, corrupt/
    stale quarantine, next-missing-row continuation).  A committed row that
    was not yet mirrored at eviction is simply regenerated — sealed CRN
    stream seeding makes the rerun canonical.
  * Foundation Learner: the latest atomic training checkpoint archive is
    mirrored after each durable checkpoint write (sync_file), and restored
    before training continuation.

Stores:

  * LocalDurableStore — directory-backed (tests, dress rehearsal, and any
    mounted network path).
  * HfDurableStore    — private Hugging Face repo (the session's existing
    staging/result approach).  Constructed lazily so hermetic tests never
    import network machinery.

Every transfer is verified by SHA-256 on both push and restore; a corrupt
mirrored object is quarantined locally and NEVER silently dropped or
trusted.  Sync failures are retried and, if persistent, surface loudly —
generation may continue (rows stay locally committed) but the eviction-loss
window is then explicitly widened, so the state machine records every sync
failure in the attempt log.
"""
from __future__ import annotations

import json
import os
import shutil

from .identity import sha256_file

SYNC_EVERY_ROWS_DEFAULT = 5


class DurabilityError(RuntimeError):
    pass


class DurableStore:
    def push_file(self, local_path: str, remote_rel: str) -> dict:
        raise NotImplementedError

    def fetch_file(self, remote_rel: str, local_path: str) -> dict:
        raise NotImplementedError

    def list_prefix(self, remote_prefix: str) -> list[str]:
        raise NotImplementedError


class LocalDurableStore(DurableStore):
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _p(self, rel: str) -> str:
        p = os.path.normpath(os.path.join(self.root, rel))
        if not p.startswith(os.path.abspath(self.root) + os.sep) \
                and p != os.path.abspath(self.root):
            p2 = os.path.abspath(p)
            if not p2.startswith(os.path.abspath(self.root) + os.sep):
                raise DurabilityError(f"path escape refused: {rel!r}")
        return p

    def push_file(self, local_path: str, remote_rel: str) -> dict:
        dest = self._p(remote_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".tmp"
        shutil.copyfile(local_path, tmp)
        want = sha256_file(local_path)
        got = sha256_file(tmp)
        if got != want:
            os.remove(tmp)
            raise DurabilityError(f"push corruption for {remote_rel}")
        os.replace(tmp, dest)
        return {"remote": remote_rel, "sha256": got}

    def fetch_file(self, remote_rel: str, local_path: str) -> dict:
        src = self._p(remote_rel)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        tmp = local_path + ".tmp"
        shutil.copyfile(src, tmp)
        want = sha256_file(src)
        got = sha256_file(tmp)
        if got != want:
            os.remove(tmp)
            raise DurabilityError(f"fetch corruption for {remote_rel}")
        os.replace(tmp, local_path)
        return {"local": local_path, "sha256": got}

    def list_prefix(self, remote_prefix: str) -> list[str]:
        base = self._p(remote_prefix)
        if not os.path.isdir(base):
            return []
        out = []
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                full = os.path.join(dirpath, name)
                out.append(os.path.relpath(full, self.root))
        return sorted(out)


class HfDurableStore(DurableStore):
    """Private HF repo store (the session's existing artifact approach).

    Every hub call runs in the isolated ``runner.hf_transfer`` subprocess:
    the pod keeps ``HF_HUB_OFFLINE=1`` so a model load can never reach the
    network, and only that helper is spawned with the offline flags
    stripped.  Doing the calls in-process would either fail closed (the
    OfflineAdapter raises for uploads too) or require weakening the
    offline guarantee for the whole session.
    """

    def __init__(self, repo_id: str, token: str | None = None,
                 runner=None, timeout: float = 3600.0):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.timeout = timeout
        self._runner = runner or self._spawn

    def _spawn(self, args: list[str]) -> dict:
        import subprocess
        import sys

        from .hf_transfer import child_env
        from ..provider.runpod.redaction import redact
        proc = subprocess.run(
            [sys.executable, "-m", "o1_b200.runner.hf_transfer", *args],
            capture_output=True, text=True, timeout=self.timeout,
            env=child_env(self.token))
        if proc.returncode != 0:
            raise DurabilityError(
                redact(f"hf transfer {args[0]} failed: "
                       f"{(proc.stdout + proc.stderr)[-500:]}"))
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            raise DurabilityError(
                f"hf transfer {args[0]} produced no result object") from None

    def push_file(self, local_path: str, remote_rel: str) -> dict:
        out = self._runner(["push", "--repo", self.repo_id,
                            "--local", local_path, "--remote-rel", remote_rel])
        want = sha256_file(local_path)
        if out.get("sha256") != want:
            raise DurabilityError(
                f"push digest disagreement for {remote_rel}")
        return {"remote": remote_rel, "sha256": want}

    def fetch_file(self, remote_rel: str, local_path: str) -> dict:
        out = self._runner(["fetch", "--repo", self.repo_id,
                            "--remote-rel", remote_rel, "--local", local_path])
        got = sha256_file(local_path)
        if out.get("sha256") != got:
            raise DurabilityError(
                f"fetch digest disagreement for {remote_rel}")
        return {"local": local_path, "sha256": got}

    def list_prefix(self, remote_prefix: str) -> list[str]:
        out = self._runner(["list", "--repo", self.repo_id,
                            "--prefix", remote_prefix])
        return list(out.get("files") or [])


class _PrefixedStore(DurableStore):
    """Scopes every key of an inner store under a session prefix.

    Without this, ``hf://ns/repo/<session>`` silently collapsed to
    ``ns/repo`` and every key became repo-root-fixed — so a run intended as
    fresh would reuse the previous session's one-time precommit and restore
    its rows, with no way to start clean short of purging the repository.
    """

    def __init__(self, inner: DurableStore, prefix: str):
        self.inner = inner
        self.prefix = prefix.strip("/")

    def _k(self, rel: str) -> str:
        return f"{self.prefix}/{rel.lstrip('/')}" if self.prefix else rel

    def push_file(self, local_path: str, remote_rel: str) -> dict:
        out = dict(self.inner.push_file(local_path, self._k(remote_rel)))
        out["remote"] = remote_rel
        return out

    def fetch_file(self, remote_rel: str, local_path: str) -> dict:
        return self.inner.fetch_file(self._k(remote_rel), local_path)

    def list_prefix(self, remote_prefix: str) -> list[str]:
        scoped = self._k(remote_prefix)
        cut = len(self.prefix) + 1 if self.prefix else 0
        return [rel[cut:] for rel in self.inner.list_prefix(scoped)]


def store_for_destination(destination: str) -> DurableStore:
    """hf://<ns>/<repo>[/<session-prefix>] -> HfDurableStore; else local dir.

    A trailing path is a SESSION SCOPE and is honoured, not discarded.
    """
    if destination.startswith("hf://"):
        body = destination[len("hf://"):]
        parts = [p for p in body.split("/") if p]
        if len(parts) < 2:
            raise DurabilityError(f"malformed hf destination {destination!r}")
        inner = HfDurableStore("/".join(parts[:2]),
                               token=os.environ.get("HF_TOKEN"))
        prefix = "/".join(parts[2:])
        return _PrefixedStore(inner, prefix) if prefix else inner
    return LocalDurableStore(destination)


class RowDurability:
    """Mirrors committed O1 row files; restores them before resume."""

    def __init__(self, run_dir: str, store: DurableStore, *,
                 remote_prefix: str = "durable_rows",
                 sync_every_rows: int = SYNC_EVERY_ROWS_DEFAULT,
                 on_event=None):
        self.run_dir = run_dir
        self.rows_dir = os.path.join(run_dir, "rows")
        self.store = store
        self.remote_prefix = remote_prefix.strip("/")
        self.sync_every_rows = max(1, int(sync_every_rows))
        self.on_event = on_event or (lambda *a, **k: None)
        self._synced: set[str] = set()
        self._since_sync = 0

    def _remote(self, name: str) -> str:
        return f"{self.remote_prefix}/{name}"

    def notify_commit(self) -> None:
        """Call after each committed row; syncs every N commits."""
        self._since_sync += 1
        if self._since_sync >= self.sync_every_rows:
            self.sync_now()

    def sync_now(self) -> dict:
        """Push every not-yet-mirrored committed row file.  Failures are
        reported loudly (widened loss window) but do not destroy local
        committed state."""
        self._since_sync = 0
        pushed, failed = [], []
        if not os.path.isdir(self.rows_dir):
            return {"pushed": 0, "failed": 0}
        for name in sorted(os.listdir(self.rows_dir)):
            if not name.endswith(".json") or name in self._synced:
                continue
            local = os.path.join(self.rows_dir, name)
            try:
                self.store.push_file(local, self._remote(name))
                self._synced.add(name)
                pushed.append(name)
            except Exception as exc:  # noqa: BLE001
                failed.append(name)
                self.on_event("DURABILITY_SYNC_FAILED", row=name,
                              error=str(exc)[:200])
        if pushed:
            self.on_event("DURABILITY_SYNCED", rows=len(pushed))
        if failed:
            self.on_event("DURABILITY_SYNC_INCOMPLETE", failed=len(failed))
        return {"pushed": len(pushed), "failed": len(failed)}

    def synced_row_count(self) -> int:
        """Durable progress marker (the session's zero-progress guard)."""
        try:
            return len(self.store.list_prefix(self.remote_prefix))
        except Exception:  # noqa: BLE001
            return len(self._synced)

    def restore(self) -> dict:
        """Pull mirrored rows into the run dir BEFORE resume.  Existing
        local rows are never overwritten (local atomic commits win); a
        corrupt mirrored row goes to quarantine, never into rows/."""
        os.makedirs(self.rows_dir, exist_ok=True)
        quarantine = os.path.join(self.run_dir, "quarantine")
        restored, skipped, corrupt = 0, 0, 0
        for rel in self.store.list_prefix(self.remote_prefix):
            name = os.path.basename(rel)
            local = os.path.join(self.rows_dir, name)
            if os.path.exists(local):
                skipped += 1
                self._synced.add(name)
                continue
            tmp = local + ".restoring"
            try:
                self.store.fetch_file(rel, tmp)
                with open(tmp, encoding="utf-8") as fh:
                    json.load(fh)      # structural sanity; full record
                                       # verification happens in resume
                os.replace(tmp, local)
                self._synced.add(name)
                restored += 1
            except Exception as exc:  # noqa: BLE001
                corrupt += 1
                os.makedirs(quarantine, exist_ok=True)
                if os.path.exists(tmp):
                    os.replace(tmp, os.path.join(
                        quarantine, name + ".restore_corrupt"))
                self.on_event("DURABILITY_RESTORE_CORRUPT", row=name,
                              error=str(exc)[:200])
        self.on_event("DURABILITY_RESTORED", restored=restored,
                      skipped=skipped, corrupt=corrupt)
        return {"restored": restored, "skipped_existing": skipped,
                "corrupt_quarantined": corrupt}


class CheckpointDurability:
    """Mirrors the latest atomic FL checkpoint archive + its manifest."""

    def __init__(self, store: DurableStore, *,
                 remote_prefix: str = "durable_checkpoint", on_event=None):
        self.store = store
        self.remote_prefix = remote_prefix.strip("/")
        self.on_event = on_event or (lambda *a, **k: None)

    def sync_checkpoint(self, archive_path: str, meta: dict) -> dict:
        """Push checkpoint + manifest; the manifest is pushed LAST so a
        half-pushed checkpoint is never referenced.

        The source is SNAPSHOTTED before hashing, and the snapshot — not the
        live file — is what gets hashed and pushed.  Without this, a file
        still being appended to (the sealed orchestrator's records JSONL is
        mirrored while it grows) is hashed at one moment and read at
        another, so the manifest records a digest for bytes that were never
        stored and every later restore refuses with a hash mismatch.
        """
        name = os.path.basename(archive_path)
        snapshot = archive_path + ".mirror_snapshot"
        shutil.copyfile(archive_path, snapshot)
        try:
            digest = sha256_file(snapshot)
            rows = None
            if snapshot.endswith(".jsonl.mirror_snapshot") or \
                    archive_path.endswith(".jsonl"):
                with open(snapshot, encoding="utf-8") as fh:
                    rows = sum(1 for ln in fh if ln.strip())
            self.store.push_file(snapshot, f"{self.remote_prefix}/{name}")
            manifest = {"archive": name, "sha256": digest, **meta}
            if rows is not None:
                manifest["rows"] = rows
            tmp = archive_path + ".manifest.json"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            try:
                self.store.push_file(tmp, f"{self.remote_prefix}/LATEST.json")
            finally:
                os.remove(tmp)
        finally:
            try:
                os.remove(snapshot)
            except FileNotFoundError:
                pass
        self.on_event("CHECKPOINT_SYNCED", archive=name, sha256=digest,
                      rows=manifest.get("rows"))
        return manifest

    def latest_row_count(self) -> int | None:
        """Durable committed-row count from the manifest, or None if absent.

        This is the externally visible progress marker the off-pod session
        driver uses for its zero-progress guard: None means "unknown", which
        is deliberately NOT the same as zero.
        """
        try:
            listing = self.store.list_prefix(self.remote_prefix)
            if f"{self.remote_prefix}/LATEST.json" not in listing:
                return None
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                local = os.path.join(tmp, "LATEST.json")
                self.store.fetch_file(
                    f"{self.remote_prefix}/LATEST.json", local)
                with open(local, encoding="utf-8") as fh:
                    return json.load(fh).get("rows")
        except Exception:  # noqa: BLE001 - advisory marker
            return None

    def restore_latest(self, dest_dir: str) -> dict | None:
        """Fetch the latest durable checkpoint; hash-verified against its
        manifest.  Returns None when no durable checkpoint exists (fresh
        start from the pristine bound model)."""
        os.makedirs(dest_dir, exist_ok=True)
        listing = self.store.list_prefix(self.remote_prefix)
        if f"{self.remote_prefix}/LATEST.json" not in listing:
            return None
        man_path = os.path.join(dest_dir, "LATEST.json")
        self.store.fetch_file(f"{self.remote_prefix}/LATEST.json", man_path)
        with open(man_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        archive = os.path.join(dest_dir, manifest["archive"])
        self.store.fetch_file(f"{self.remote_prefix}/{manifest['archive']}",
                              archive)
        got = sha256_file(archive)
        if got != manifest["sha256"]:
            raise DurabilityError(
                f"durable checkpoint hash mismatch: {got[:16]} != "
                f"{manifest['sha256'][:16]}; refusing to resume from it")
        self.on_event("CHECKPOINT_RESTORED", archive=manifest["archive"])
        # the resolved LOCAL path must win: spreading the manifest last put
        # its bare basename back into "archive", so callers received a
        # relative name that only resolved by accident of the cwd
        return {**manifest, "archive": archive}
