"""Crash-safe persistence: atomic per-row commits, resume identity, quarantine.

Layout of a run directory:

  run_dir/
    RESUME_IDENTITY.json        exact identity of the run configuration
    attempt_log.jsonl           append-only event log (fsync per line)
    rows/<row_id>.json          one atomic file per committed row
    quarantine/<name>           corrupt / stale / duplicate material (never deleted)
    records.jsonl               canonical merge (finalize only)
    records.jsonl.sha256        digest of the canonical merge

Guarantees:

  * a partially written row can never appear valid (tmp + fsync + rename);
  * a duplicate row_id with different content is quarantined and refused;
  * a corrupt row file is quarantined, never silently dropped;
  * a stale record (wrong binding or wrong record_hash) is quarantined;
  * resume refuses when ANY identity field differs;
  * completed, verified rows are never regenerated after resume;
  * the canonical merge is deterministic (exec_index order) and refuses gaps.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Callable, Iterable

# A .tmp older than this on resume scan is a dead run's partial write; a
# younger one may be a LIVE concurrent worker's in-flight atomic commit.
STALE_TMP_SECONDS = 60.0

from .identity import canonical_json, sha256_file
from .records import RecordError, record_hash
from .row_specs import RowSpec


class PersistenceError(RuntimeError):
    pass


class ResumeIdentityError(PersistenceError):
    """The run directory belongs to a different sealed configuration."""


# Every field that must be EXACTLY equal for resume to proceed.
RESUME_IDENTITY_FIELDS = (
    "source_commit", "package_zip_sha256", "model_artifact_sha256",
    "tokenizer_sha256", "axis_package_sha256", "backend_id",
    "backend_config_sha256", "batch_size", "worker_count",
    "runtime_environment_sha256", "binding_key", "binding_sha256",
    "seal_sha256", "task_manifest_sha256", "seed_matrix_sha256",
    "row_spec_manifest_sha256",
)


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(path))


class RowStore:
    def __init__(self, run_dir: str, resume_identity: dict,
                 verify_row: Callable[[dict], None]):
        """verify_row: full recomputation callback; raises on invalid rows."""
        missing = [k for k in RESUME_IDENTITY_FIELDS if k not in resume_identity]
        extra = sorted(set(resume_identity) - set(RESUME_IDENTITY_FIELDS))
        if missing or extra:
            raise PersistenceError(
                f"resume identity malformed: missing={missing} extra={extra}")
        self.run_dir = run_dir
        self.rows_dir = os.path.join(run_dir, "rows")
        self.quarantine_dir = os.path.join(run_dir, "quarantine")
        self.identity = dict(resume_identity)
        self._verify_row = verify_row
        os.makedirs(self.rows_dir, exist_ok=True)
        os.makedirs(self.quarantine_dir, exist_ok=True)
        self._identity_path = os.path.join(run_dir, "RESUME_IDENTITY.json")
        if os.path.exists(self._identity_path):
            with open(self._identity_path, encoding="utf-8") as fh:
                stored = json.load(fh)
            diffs = {k for k in RESUME_IDENTITY_FIELDS
                     if stored.get(k) != self.identity.get(k)}
            if diffs:
                raise ResumeIdentityError(
                    "resume refused; identity differs on: " + ", ".join(sorted(diffs)))
        else:
            atomic_write_text(self._identity_path,
                              canonical_json(self.identity) + "\n")
        self._log_path = os.path.join(run_dir, "attempt_log.jsonl")
        # optional continuous off-pod mirroring (durability.RowDurability);
        # attach_durability restores mirrored rows BEFORE resume reads them
        self.durability = None

    # ---------------- attempt log ----------------

    def log(self, event: str, **fields) -> None:
        from .records import utc_now
        line = canonical_json({"event": event, "utc": utc_now(), **fields})
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ---------------- row commits ----------------

    def _row_path(self, row_id: str) -> str:
        return os.path.join(self.rows_dir, f"{row_id}.json")

    def commit_row(self, record: dict) -> None:
        """Atomic per-row commit with duplicate rejection."""
        rid = record["row_id"]
        if record.get("binding_sha256") != self.identity["binding_sha256"]:
            raise PersistenceError(
                f"row {rid[:16]}: binding_sha256 does not match this run")
        if record_hash(record) != record["record_hash"]:
            raise PersistenceError(f"row {rid[:16]}: record_hash invalid at commit")
        path = self._row_path(rid)
        payload = canonical_json(record) + "\n"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                existing = fh.read()
            if existing == payload:
                # idempotent re-commit of the identical record is a no-op
                return
            self._quarantine_bytes(f"duplicate_conflict_{rid}.json", payload)
            raise PersistenceError(
                f"duplicate row {rid[:16]} with DIFFERENT content refused; "
                f"conflicting copy quarantined")
        atomic_write_text(path, payload)
        if self.durability is not None:
            self.durability.notify_commit()

    def attach_durability(self, durability) -> None:
        """Attach off-pod mirroring: restore mirrored rows first (existing
        local commits always win), then mirror every future commit."""
        durability.restore()
        self.durability = durability

    def _quarantine_bytes(self, name: str, text: str) -> None:
        atomic_write_text(os.path.join(self.quarantine_dir, name), text)

    def quarantine_file(self, path: str, reason: str) -> None:
        base = os.path.basename(path)
        dest = os.path.join(self.quarantine_dir, base)
        n = 0
        while os.path.exists(dest):
            n += 1
            dest = os.path.join(self.quarantine_dir, f"{base}.{n}")
        try:
            shutil.move(path, dest)
        except FileNotFoundError:
            # a concurrent worker completed (or already quarantined) it
            # between our scan and this move; nothing left to quarantine
            self.log("ROW_QUARANTINE_RACED", file=base, reason=reason)
            return
        self.log("ROW_QUARANTINED", file=base, reason=reason)

    # ---------------- resume scan ----------------

    def load_completed(self) -> dict[str, dict]:
        """Scan committed rows; verify each fully; quarantine invalid ones.

        Returns {row_id: record} for rows that pass FULL recomputation.
        A row that fails any check is moved to quarantine and NOT returned,
        so it is deterministically rescheduled.
        """
        done: dict[str, dict] = {}
        for name in sorted(os.listdir(self.rows_dir)):
            path = os.path.join(self.rows_dir, name)
            if name.endswith(".tmp"):
                # a STALE tmp is a partial write from a dead run and is
                # quarantined; a FRESH tmp may belong to a LIVE concurrent
                # worker mid-atomic-commit (replica backend: every worker
                # scans the shared run dir at startup) and must be left
                # alone — stealing it would break the writer's os.replace
                # and lose a committed row
                try:
                    age = time.time() - os.path.getmtime(path)
                except OSError:
                    continue        # already renamed/cleaned by its writer
                if age >= STALE_TMP_SECONDS:
                    self.quarantine_file(
                        path, "stale partial write (tmp) found on resume")
                continue
            if not name.endswith(".json"):
                self.quarantine_file(path, "unexpected file in rows dir")
                continue
            rid = name[:-len(".json")]
            try:
                with open(path, encoding="utf-8") as fh:
                    record = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.quarantine_file(path, "corrupt JSON")
                continue
            try:
                if record.get("row_id") != rid:
                    raise RecordError("filename row_id disagrees with content")
                if record_hash(record) != record.get("record_hash"):
                    raise RecordError("stale record: record_hash mismatch")
                if record.get("binding_sha256") != self.identity["binding_sha256"]:
                    raise RecordError("stale record: foreign binding_sha256")
                self._verify_row(record)
            except (RecordError, KeyError, TypeError, ValueError) as exc:
                self.quarantine_file(path, f"failed recomputation: {exc}")
                continue
            done[rid] = record
        return done

    # ---------------- canonical merge ----------------

    def finalize(self, specs: list[RowSpec]) -> dict:
        """Deterministic canonical merge in exec_index order.

        Refuses on any missing row.  Detects rows that do not belong to the
        sealed spec list (foreign rows are quarantined, then refused).
        """
        done = self.load_completed()
        by_id = {s.row_id: s for s in specs}
        foreign = sorted(set(done) - set(by_id))
        for rid in foreign:
            self.quarantine_file(self._row_path(rid),
                                 "row not in the sealed row-spec manifest")
        if foreign:
            raise PersistenceError(
                f"{len(foreign)} foreign rows quarantined; refusing to finalize")
        missing = [s.row_id for s in specs if s.row_id not in done]
        if missing:
            raise PersistenceError(
                f"missing {len(missing)} rows (first: {missing[0][:16]}); "
                f"resume until complete")
        out = os.path.join(self.run_dir, "records.jsonl")
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for spec in sorted(specs, key=lambda s: s.exec_index):
                fh.write(canonical_json(done[spec.row_id]) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
        digest = sha256_file(out)
        atomic_write_text(out + ".sha256", f"{digest}  records.jsonl\n")
        self.log("FINALIZED", n_rows=len(specs), records_sha256=digest)
        return {"records": out, "records_sha256": digest, "n_rows": len(specs)}
