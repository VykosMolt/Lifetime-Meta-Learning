"""Combined O1 + Foundation Learner session supervisor (contract §13, §22).

The O1 calibration owns the accelerator first and absolutely.  This state
machine runs the O1 pod-side entry as an OPAQUE configured subprocess, waits
for its declared completion artefacts, verifies and transfers them, closes the
O1 process, and only then reloads a pristine Ouro checkpoint and runs the FL
ladder::

    START_SESSION -> RUN_O1_CALIBRATION -> O1_HALT_OR_COMPLETE ->
    VERIFY_O1_RECORDS -> TRANSFER_O1_RECORDS -> CLOSE_O1_PROCESS ->
    RELOAD_PRISTINE_OURO -> COMPUTE_REMAINING_AUTHORIZED_TIME ->
    RUN_FL_LADDER -> CHECKPOINT_AND_VERIFY -> TRANSFER_FL_ARTIFACTS ->
    TERMINATE_ACCELERATOR

It NEVER modifies the sealed O1 runner: the O1 entry, the O1 transfer, and the
termination are configured commands, and no O1 module is imported.

**Custody of O1 records vs FL isolation.**  ``VERIFY_O1_RECORDS`` must hash
files that live under the very roots the FL isolation guard refuses.  That is
not a loophole in the guard: the FL guard still refuses those paths for all FL
work, and the verification runs through a separate, deliberately crippled
:class:`O1RecordCustodian` that

* only accepts paths under the DECLARED O1 roots,
* only ever returns SHA-256 digests and path/size metadata, and
* has no method that returns file CONTENT to any caller.

Hashing bytes is not reading outcomes: a digest carries no calibration result,
no difficulty selection, and no analysis.  Every path the custodian touched is
recorded in the journal, so the claim is auditable rather than asserted.

**UNRESOLVED refusal.**  Any configuration value beginning with ``UNRESOLVED``
refuses a real run, exactly like the O1 session config.  A DRESS REHEARSAL may
proceed with unresolved values, and then every artefact it writes is loudly
labelled ``DRESS_REHEARSAL``.

**Closing out costs money.**  :meth:`SessionSupervisor.run` wraps the whole
state machine in ``try/finally``: whatever happens — a failed state, an
unexpected exception, a keyboard interrupt — the supervisor still attempts the
FL transfer (best effort) and ALWAYS attempts ``TERMINATE_ACCELERATOR``, and
always writes ``SESSION_FINAL_STATUS.json``.  A rented accelerator that keeps
billing after a crash is a real failure mode, not a hypothetical one (review
finding R-C2).

**Resume rebuilds state.**  On resume the supervisor reconstructs
``state_results`` from the journal's ``STATE_COMPLETED`` payloads, so a session
that crashed after ``COMPUTE_REMAINING_AUTHORIZED_TIME`` still knows its FL
allowance instead of refusing ``RUN_FL_LADDER``.  The resumed allowance is
additionally REDUCED by the wall-clock gap since that record was written: the
pod kept billing while the process was gone, and an over-estimated budget is
the one error that can consume the transfer reserve.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..ecology.base import sha256_file, sha256_tree
from . import o1_isolation, result_verifier
from .scheduler import MonotonicClock, Scheduler
from .stage_definitions import StageContext

__all__ = [
    "SupervisorError",
    "SessionConfigError",
    "STATES",
    "SESSION_CONFIG_SCHEMA",
    "JOURNAL_NAME",
    "JOURNAL_SCHEMA",
    "REHEARSAL_LABEL",
    "SessionConfig",
    "O1RecordCustodian",
    "SessionSupervisor",
    "main",
]

SESSION_CONFIG_SCHEMA = "flb200.session_config.v1"
JOURNAL_NAME = "FL_SESSION_JOURNAL.jsonl"
JOURNAL_SCHEMA = "flb200.session_journal.v1"
RECEIPT_SCHEMA = "flb200.o1_close_receipt.v1"
FINAL_STATUS_SCHEMA = "flb200.session_final_status.v1"
REHEARSAL_LABEL = "DRESS_REHEARSAL"

STATES: tuple[str, ...] = (
    "START_SESSION",
    "RUN_O1_CALIBRATION",
    "O1_HALT_OR_COMPLETE",
    "VERIFY_O1_RECORDS",
    "TRANSFER_O1_RECORDS",
    "CLOSE_O1_PROCESS",
    "RELOAD_PRISTINE_OURO",
    "COMPUTE_REMAINING_AUTHORIZED_TIME",
    "RUN_FL_LADDER",
    "CHECKPOINT_AND_VERIFY",
    "TRANSFER_FL_ARTIFACTS",
    "TERMINATE_ACCELERATOR",
)

#: FL may not start before this state has completed AND left its receipt.
FL_PREREQUISITE_STATE = "TRANSFER_O1_RECORDS"
O1_TRANSFER_RECEIPT = "O1_TRANSFER_RECEIPT.json"
O1_CLOSE_RECEIPT = "O1_CLOSE_RECEIPT.json"


class SupervisorError(RuntimeError):
    """A supervisor invariant refused."""


class SessionConfigError(SupervisorError):
    """The session configuration is missing, malformed, or UNRESOLVED."""


# --------------------------------------------------------------------------
# session configuration
# --------------------------------------------------------------------------

_REQUIRED_CONFIG_FIELDS = (
    "session_id",
    "o1_entry_command",
    "o1_completion_markers",
    "o1_hash_manifests",
    "o1_transfer_command",
    "o1_roots",
    "checkpoint_dir",
    "checkpoint_tree_sha256",
    "pregen_root",
    "fl_out_dir",
    "session_authorized_seconds",
)


def _unresolved_values(obj: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, str):
        if obj.startswith("UNRESOLVED"):
            found.append(path or "<root>")
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            found.extend(_unresolved_values(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(_unresolved_values(value, f"{path}[{index}]"))
    return found


@dataclass
class SessionConfig:
    payload: dict
    path: str | None = None

    @staticmethod
    def load(path: str, *, guard: o1_isolation.IsolationGuard | None = None
             ) -> "SessionConfig":
        guard = guard or o1_isolation.default_guard()
        resolved = guard.guard(path, o1_isolation.MODE_READ)
        if not os.path.isfile(resolved):
            raise SessionConfigError(f"session config not found: {path!r}")
        with open(resolved, encoding="utf-8") as fh:
            payload = json.load(fh)
        return SessionConfig(payload=payload, path=resolved)

    def validate(self) -> dict:
        payload = self.payload
        if payload.get("schema") != SESSION_CONFIG_SCHEMA:
            raise SessionConfigError(
                f"schema {payload.get('schema')!r} != {SESSION_CONFIG_SCHEMA!r}")
        missing = [k for k in _REQUIRED_CONFIG_FIELDS if k not in payload]
        if missing:
            raise SessionConfigError(f"missing required fields {missing}")
        rehearsal = bool(payload.get("rehearsal", False))
        if not rehearsal:
            # Without these a real session would strand its results on a pod
            # that is still burning money; a rehearsal has neither concern.
            for key in ("fl_transfer_command", "terminate_command"):
                if not payload.get(key):
                    raise SessionConfigError(
                        f"a real session requires {key!r}: without it the FL "
                        "artefacts never leave the accelerator / the "
                        "accelerator is never terminated")
        unresolved = _unresolved_values(payload)
        if unresolved and not rehearsal:
            raise SessionConfigError(
                f"session config carries unresolved template fields "
                f"{unresolved}; a real session refuses to start. (The O1 "
                "pod entrypoint is a recorded, operator-bound gap: "
                "'o1_entry_command' must be resolved by the operator.)")
        if unresolved and rehearsal:
            payload.setdefault("_unresolved_fields", unresolved)
        return payload

    @property
    def rehearsal(self) -> bool:
        return bool(self.payload.get("rehearsal", False))

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


# --------------------------------------------------------------------------
# O1 record custody (hashes only, never content)
# --------------------------------------------------------------------------

class O1RecordCustodian:
    """Hash-only access to O1 artefacts, restricted to the declared O1 roots.

    This class exists so that ``VERIFY_O1_RECORDS`` can do its job without
    weakening the FL isolation guard.  It is deliberately crippled: there is no
    method that returns file content, and every accepted path must lie under a
    declared O1 root (paths outside are refused, so it cannot become a general
    file reader).  The only structured data it parses is the O1 manifest's own
    path/digest listing.
    """

    def __init__(self, o1_roots: Sequence[str]):
        if not o1_roots:
            raise SupervisorError(
                "the O1 record custodian requires the declared O1 roots")
        self.roots = sorted({os.path.realpath(os.path.abspath(str(r)))
                             for r in o1_roots})
        self.touched: list[str] = []

    def _accept(self, path: str) -> str:
        resolved = os.path.realpath(os.path.abspath(str(path)))
        for root in self.roots:
            if resolved == root or resolved.startswith(root.rstrip(os.sep) + os.sep):
                self.touched.append(resolved)
                return resolved
        raise SupervisorError(
            f"O1 custodian refuses {path!r}: it is not under a declared O1 "
            f"root {self.roots}; the custodian is not a general file reader")

    def digest(self, path: str) -> str:
        """SHA-256 of one O1 file.  A digest is not an outcome."""
        return sha256_file(self._accept(path))

    def exists(self, path: str) -> bool:
        return os.path.exists(self._accept(path))

    def manifest_entries(self, manifest_path: str) -> list[dict]:
        """Extract ``(path, sha256)`` pairs from an O1 hash manifest.

        Tolerates the O1 ``o1b200.transfer_manifest.v1`` shape, a plain
        ``{"files": {rel: {"sha256": ...}}}`` mapping, and the two-space
        ``SHA256SUMS`` text format.  Nothing but paths and digests is read.
        """
        resolved = self._accept(manifest_path)
        base = os.path.dirname(resolved)
        entries: list[dict] = []
        with open(resolved, "rb") as fh:
            raw = fh.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            for line in raw.decode("utf-8", errors="strict").splitlines():
                if not line.strip():
                    continue
                digest, rel = line.split("  ", 1)
                entries.append({"path": os.path.join(base, rel),
                                "sha256": digest})
            return entries
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        if isinstance(artifacts, dict):
            for name, spec in sorted(artifacts.items()):
                if not isinstance(spec, dict):
                    continue
                path = spec.get("path")
                digest = spec.get("sha256")
                if path and digest:
                    entries.append({
                        "name": name,
                        "path": path if os.path.isabs(path) else os.path.join(base, path),
                        "sha256": digest,
                        "kind": spec.get("kind", "file"),
                    })
        files = payload.get("files") if isinstance(payload, dict) else None
        if isinstance(files, dict):
            for rel, spec in sorted(files.items()):
                digest = spec.get("sha256") if isinstance(spec, dict) else spec
                if digest:
                    entries.append({"path": os.path.join(base, rel),
                                    "sha256": digest})
        elif isinstance(files, list):
            for spec in files:
                if isinstance(spec, dict) and spec.get("path") and spec.get("sha256"):
                    rel = spec["path"]
                    entries.append({
                        "path": rel if os.path.isabs(rel) else os.path.join(base, rel),
                        "sha256": spec["sha256"]})
        if not entries:
            raise SupervisorError(
                f"{manifest_path}: no path/digest pairs found; the supervisor "
                "cannot verify O1 records without them")
        return entries

    def verify_manifest(self, manifest_path: str) -> dict:
        """Recompute every listed digest.  File-level hashing only."""
        entries = self.manifest_entries(manifest_path)
        checked = 0
        missing: list[str] = []
        mismatched: list[str] = []
        for entry in entries:
            path = entry["path"]
            if entry.get("kind") == "reference":
                continue
            try:
                resolved = self._accept(path)
            except SupervisorError:
                missing.append(path)
                continue
            if not os.path.exists(resolved):
                missing.append(path)
                continue
            digest = (sha256_tree(resolved) if os.path.isdir(resolved)
                      else sha256_file(resolved))
            if digest != entry["sha256"]:
                mismatched.append(path)
            checked += 1
        return {
            "manifest": os.path.basename(manifest_path),
            "entries": len(entries),
            "checked": checked,
            "missing": missing,
            "mismatched": mismatched,
            "ok": not missing and not mismatched,
            "method": ("recomputed SHA-256 over file bytes / directory trees; "
                       "no scientific content was parsed"),
        }


# --------------------------------------------------------------------------
# the supervisor
# --------------------------------------------------------------------------

@dataclass
class SessionSupervisor:
    config: SessionConfig
    out_dir: str
    guard: o1_isolation.IsolationGuard = field(
        default_factory=o1_isolation.default_guard)
    clock: Any = field(default_factory=MonotonicClock)
    runner: Callable[..., Any] = subprocess.run
    ladder_runner: Callable[[Scheduler, StageContext], dict] | None = None
    context_factory: Callable[["SessionSupervisor"], StageContext] | None = None
    journal_path: str = field(init=False)
    payload: dict = field(init=False)
    state_results: dict[str, Any] = field(default_factory=dict, init=False)
    completed: list[str] = field(default_factory=list, init=False)
    determinism: dict | None = field(default=None, init=False)
    close_out: dict = field(default_factory=dict, init=False)
    _t0: float | None = field(default=None, init=False)
    _records: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.payload = self.config.validate()
        self.guard.makedirs(self.out_dir)
        self.journal_path = os.path.join(self.out_dir, JOURNAL_NAME)
        extra_roots = list(self.payload.get("o1_roots", [])) + list(
            self.payload.get("o1_forbidden_roots_extra", []))
        self.guard.add_forbidden_roots(extra_roots)
        self.custodian = O1RecordCustodian(self.payload["o1_roots"])
        # Off-pod durability under INTERRUPTIBLE capacity: restore the
        # mirrored journal/checkpoints BEFORE reading the journal, so a
        # fresh pod after eviction resumes exactly like a same-pod restart
        # (completed states skip; training resumes from the last valid
        # atomic checkpoint; post-checkpoint work reruns).  Local files are
        # never overwritten by the restore.
        self.durability = None
        self.durability_events: list[dict] = []
        destination = self.payload.get("fl_durable_destination")
        if destination and not str(destination).startswith("UNRESOLVED"):
            from .durability import FlDurableMirror

            def _mirror_event(event, **fields):
                # buffered here because the journal does not exist yet during
                # construction; drained into it by _drain_durability_events
                # so a silently failing mirror can never look durable
                self.durability_events.append({"event": str(event), **fields})

            self.durability = FlDurableMirror(
                str(destination), self.out_dir, on_event=_mirror_event,
                guard=self.guard.guard)
            try:
                self.durability.restore_all()
            except Exception as exc:  # noqa: BLE001
                # a store outage must DEGRADE (start fresh; restore never
                # overwrites local files anyway), not refuse the session
                self.durability_events.append(
                    {"event": "FL_DURABILITY_RESTORE_FAILED",
                     "error": str(exc)[:300]})
        self.completed = [r["state"] for r in self.read_journal()
                          if r.get("event") == "STATE_COMPLETED"]

    # ---------------- journal ----------------

    @property
    def rehearsal(self) -> bool:
        return self.config.rehearsal

    def label(self) -> str:
        return REHEARSAL_LABEL if self.rehearsal else str(
            self.payload.get("label", "FL_B200_SESSION"))

    def journal(self, event: str, state: str, payload: dict | None = None) -> dict:
        record = {
            "schema": JOURNAL_SCHEMA,
            "index": self._records,
            "event": str(event),
            "state": str(state),
            "session_id": self.payload.get("session_id"),
            "label": self.label(),
            "rehearsal": self.rehearsal,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "monotonic": round(self.clock.monotonic(), 6),
            **(payload or {}),
        }
        self.guard.append_line(
            self.journal_path,
            json.dumps(record, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False))
        self._records += 1
        if self.durability is not None:
            self.durability.push_file(self.journal_path)
            self._drain_durability_events()
        return record

    def _drain_durability_events(self) -> None:
        """Journal buffered mirror events (failures included).

        Without this, a journal mirror that fails for the whole session
        leaves no record, and after eviction the next pod resumes from a
        stale journal believing it was durable.
        """
        pending, self.durability_events = self.durability_events, []
        for entry in pending:
            event = entry.pop("event", "FL_DURABILITY_EVENT")
            record = {
                "schema": JOURNAL_SCHEMA, "index": self._records,
                "event": event, "state": "DURABILITY",
                "session_id": self.payload.get("session_id"),
                "label": self.label(), "rehearsal": self.rehearsal,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "monotonic": round(self.clock.monotonic(), 6), **entry,
            }
            self.guard.append_line(
                self.journal_path,
                json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False))
            self._records += 1

    def read_journal(self) -> list[dict]:
        path = self.guard.guard(self.journal_path, o1_isolation.MODE_READ)
        if not os.path.exists(path):
            return []
        out = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
        return out

    # ---------------- resume ----------------

    @staticmethod
    def _utc_seconds(stamp: str | None) -> float | None:
        if not stamp:
            return None
        try:
            return float(calendar.timegm(
                time.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")))
        except (ValueError, TypeError):
            return None

    def rebuild_state_results(self) -> dict:
        """Reconstruct ``state_results`` from the journal (review R-C2).

        Without this a resumed session skips ``COMPUTE_REMAINING_AUTHORIZED_TIME``
        as "already completed" and then refuses ``RUN_FL_LADDER`` for want of an
        FL allowance — i.e. a crash anywhere in the middle silently cost the
        whole FL half of the session.

        The restored allowance is REDUCED by the wall-clock gap between the
        journalled record and now.  The pod billed for that gap; using the
        stale value would be the one arithmetic error that can eat the final
        transfer reserve.  This is an operational budget decision, not a
        scientific path (contract §20).
        """
        restored: dict[str, Any] = {}
        stamps: dict[str, Any] = {}
        for record in self.read_journal():
            if record.get("event") != "STATE_COMPLETED":
                continue
            state = str(record.get("state"))
            if "result" in record:
                restored[state] = record["result"]
                stamps[state] = record.get("utc")
        self.state_results.update(restored)
        report = {"restored_states": sorted(restored),
                  "available_foundation_learner_seconds": None}
        compute = restored.get("COMPUTE_REMAINING_AUTHORIZED_TIME")
        if isinstance(compute, Mapping):
            available = float(
                compute.get("available_foundation_learner_seconds") or 0.0)
            recorded_at = self._utc_seconds(
                stamps.get("COMPUTE_REMAINING_AUTHORIZED_TIME"))
            gap = 0.0
            if recorded_at is not None:
                gap = max(0.0, time.time() - recorded_at)
            adjusted = max(0.0, available - gap)
            self.state_results["available_foundation_learner_seconds"] = adjusted
            report.update({
                "available_foundation_learner_seconds": adjusted,
                "journalled_available_seconds": available,
                "resume_wall_clock_gap_seconds": round(gap, 3),
                "rule": ("a resumed session re-derives its FL allowance from "
                         "the journal and subtracts the wall-clock gap; the "
                         "accelerator billed for it"),
            })
        if restored:
            self.journal("STATE_RESULTS_REBUILT", "START_SESSION", report)
        return report

    # ---------------- prerequisites ----------------

    def require_completed(self, state: str, *, for_state: str) -> None:
        if state not in self.completed:
            raise SupervisorError(
                f"REFUSED: {for_state} cannot run before {state} has completed "
                f"(completed so far: {self.completed})")

    def require_receipt(self, name: str, *, for_state: str) -> str:
        path = self.guard.guard(os.path.join(self.out_dir, name),
                                o1_isolation.MODE_READ)
        if not os.path.isfile(path):
            raise SupervisorError(
                f"REFUSED: {for_state} requires the receipt {name!r}; it does "
                "not exist. FL never starts before the O1 artefacts are "
                "transferred and verified (contract §13).")
        return path

    # ---------------- helpers ----------------

    def _child_env(self) -> dict:
        """Environment for a configured subprocess (the O1 phase included).

        O1's pod entrypoint dispatches to THIS supervisor whenever it sees
        O1_FL_SESSION_CONFIG.  Leaving that variable in the child environment
        would make the O1 phase start a second combined session, which would
        run the O1 phase again: unbounded recursion on a paid accelerator.
        O1's entry carries its own depth marker as an independent guard;
        either alone is sufficient, and neither depends on the other being
        correct.
        """
        env = {k: v for k, v in os.environ.items()
               if k != "O1_FL_SESSION_CONFIG"}
        env["O1_B300_ENTRY_ACTIVE"] = "1"
        return env

    def _run_command(self, command: Sequence[str] | str, *, state: str,
                     timeout: float | None = None) -> dict:
        if not command:
            raise SupervisorError(f"{state}: no command configured")
        shell = isinstance(command, str)
        started = self.clock.monotonic()
        proc = self.runner(command, shell=shell, capture_output=True, text=True,
                           timeout=timeout, env=self._child_env(),
                           cwd=self.payload.get("o1_workdir") or None)
        seconds = self.clock.monotonic() - started
        record = {
            "command": command if shell else list(command),
            "returncode": int(getattr(proc, "returncode", -1)),
            "seconds": round(seconds, 6),
            "stdout_tail": (getattr(proc, "stdout", "") or "")[-2000:],
            "stderr_tail": (getattr(proc, "stderr", "") or "")[-2000:],
        }
        return record

    # ---------------- states ----------------

    def state_START_SESSION(self) -> dict:
        self._t0 = self.clock.monotonic()
        return {
            "config_path": self.config.path,
            "rehearsal": self.rehearsal,
            "unresolved_fields": self.payload.get("_unresolved_fields", []),
            "o1_roots": list(self.payload["o1_roots"]),
            "isolation": self.guard.to_dict(),
            "session_authorized_seconds": float(
                self.payload["session_authorized_seconds"]),
        }

    def state_RUN_O1_CALIBRATION(self) -> dict:
        """Run the O1 pod-side entry as an OPAQUE configured subprocess."""
        record = self._run_command(self.payload["o1_entry_command"],
                                   state="RUN_O1_CALIBRATION",
                                   timeout=self.payload.get("o1_timeout_seconds"))
        record["note"] = ("the O1 entry is opaque to FL: FL neither imports O1 "
                          "modules nor interprets O1 output")
        return record

    def state_O1_HALT_OR_COMPLETE(self) -> dict:
        """Did the O1 phase COMPLETE, or did it HALT?

        Only presence is checked; no marker content is read.  A HALT is a
        legitimate O1 outcome, but it is NOT a licence for FL to proceed:
        contract §13 makes FL conditional on the O1 records being transferred
        AND verified, and a halted O1 phase has produced no such records.  The
        session therefore aborts here, with the halt recorded, rather than
        silently running FL on an accelerator whose first workload failed.
        """
        markers = list(self.payload["o1_completion_markers"])
        present = {marker: self.custodian.exists(marker) for marker in markers}
        missing = sorted(m for m, ok in present.items() if not ok)
        entry = self.state_results.get("RUN_O1_CALIBRATION") or {}
        if missing:
            raise SupervisorError(
                f"O1_HALT: the O1 phase did not produce its declared "
                f"completion artefacts {missing} (entry command returncode "
                f"{entry.get('returncode')}). FL does not start: there are no "
                "O1 records to verify or transfer (contract §13).")
        return {"o1_outcome": "COMPLETE", "markers": present,
                "entry_returncode": entry.get("returncode"),
                "note": "presence only; no marker content is read"}

    def state_VERIFY_O1_RECORDS(self) -> dict:
        reports = [self.custodian.verify_manifest(m)
                   for m in self.payload["o1_hash_manifests"]]
        bad = [r for r in reports if not r["ok"]]
        if bad:
            raise SupervisorError(
                f"O1 record verification FAILED for {[r['manifest'] for r in bad]}")
        return {
            "reports": reports,
            "paths_touched": len(self.custodian.touched),
            "method": ("recomputed SHA-256 over the files O1's own manifests "
                       "list; hashing bytes is not reading outcomes"),
        }

    def state_TRANSFER_O1_RECORDS(self) -> dict:
        record = self._run_command(self.payload["o1_transfer_command"],
                                   state="TRANSFER_O1_RECORDS",
                                   timeout=self.payload.get("o1_transfer_timeout_seconds"))
        if record["returncode"] != 0:
            raise SupervisorError(
                f"O1 transfer command failed with rc={record['returncode']}")
        receipt = {
            "schema": "flb200.o1_transfer_receipt.v1",
            "session_id": self.payload.get("session_id"),
            "label": self.label(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "transfer": record,
            "verification": self.state_results.get("VERIFY_O1_RECORDS"),
        }
        self.guard.write_json(os.path.join(self.out_dir, O1_TRANSFER_RECEIPT),
                              receipt)
        return receipt

    def state_CLOSE_O1_PROCESS(self) -> dict:
        self.require_completed("TRANSFER_O1_RECORDS", for_state="CLOSE_O1_PROCESS")
        close_command = self.payload.get("o1_close_command")
        record = None
        if close_command:
            record = self._run_command(close_command, state="CLOSE_O1_PROCESS")
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "session_id": self.payload.get("session_id"),
            "label": self.label(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "close_command": record,
            "o1_roots": list(self.payload["o1_roots"]),
            "statement": ("the O1 process is closed; FL starts from a FRESH "
                          "reload of the pristine checkpoint and never from an "
                          "O1-mutated process state"),
        }
        self.guard.write_json(os.path.join(self.out_dir, O1_CLOSE_RECEIPT), receipt)
        return receipt

    def state_RELOAD_PRISTINE_OURO(self) -> dict:
        """Verify the checkpoint tree hash before any FL work touches it."""
        self.require_completed("CLOSE_O1_PROCESS", for_state="RELOAD_PRISTINE_OURO")
        checkpoint_dir = self.payload["checkpoint_dir"]
        expected = self.payload["checkpoint_tree_sha256"]
        from ..training import model_loading

        if not self.rehearsal:
            if expected != model_loading.FROZEN_CHECKPOINT_TREE_SHA256:
                raise SupervisorError(
                    "session config binds a checkpoint tree hash that is not "
                    "the frozen §1 backbone hash; a real session refuses")
            got = model_loading.verify_checkpoint_tree(checkpoint_dir)
        else:
            got = sha256_tree(self.guard.guard(checkpoint_dir,
                                               o1_isolation.MODE_READ))
            if got != expected:
                raise SupervisorError(
                    f"rehearsal checkpoint tree hash mismatch: expected "
                    f"{expected}, got {got}")
        return {"checkpoint_dir": checkpoint_dir, "tree_sha256": got,
                "verified_against": ("frozen §1 backbone hash" if not self.rehearsal
                                     else "rehearsal stand-in tree hash"),
                "rehearsal": self.rehearsal}

    def state_COMPUTE_REMAINING_AUTHORIZED_TIME(self) -> dict:
        authorized = float(self.payload["session_authorized_seconds"])
        elapsed = 0.0 if self._t0 is None else max(
            0.0, self.clock.monotonic() - self._t0)
        available = max(0.0, authorized - elapsed)
        record = {
            "session_authorized_seconds": authorized,
            "elapsed_seconds": round(elapsed, 6),
            "available_foundation_learner_seconds": round(available, 6),
            "rule": ("FL receives what remains after O1 closes; it never "
                     "assumes ownership of the whole rental"),
        }
        self.state_results["available_foundation_learner_seconds"] = available
        return record

    def state_RUN_FL_LADDER(self) -> dict:
        self.require_completed(FL_PREREQUISITE_STATE, for_state="RUN_FL_LADDER")
        self.require_receipt(O1_TRANSFER_RECEIPT, for_state="RUN_FL_LADDER")
        self.require_completed("CLOSE_O1_PROCESS", for_state="RUN_FL_LADDER")
        self.require_completed("RELOAD_PRISTINE_OURO", for_state="RUN_FL_LADDER")
        available = self.state_results.get("available_foundation_learner_seconds")
        if available is None:
            raise SupervisorError(
                "RUN_FL_LADDER requires COMPUTE_REMAINING_AUTHORIZED_TIME")
        if self.context_factory is None:
            raise SupervisorError(
                "no StageContext factory supplied; the supervisor does not "
                "invent one (the campaign entry point wires it)")
        ctx = self.context_factory(self)
        scheduler = Scheduler(
            available_foundation_learner_seconds=float(available),
            out_dir=os.path.join(self.out_dir, "ladder"),
            guard=self.guard, clock=self.clock, label=self.label())
        runner = self.ladder_runner or (lambda sch, c: sch.run_ladder(c))
        summary = runner(scheduler, ctx)
        self.state_results["_ctx"] = ctx
        self.state_results["_scheduler"] = scheduler
        return summary

    def state_CHECKPOINT_AND_VERIFY(self) -> dict:
        root = os.path.join(self.out_dir, "ladder")
        manifest = result_verifier.write_result_manifest(
            root, os.path.join(self.out_dir, "FL_RESULT_MANIFEST.json"),
            label=self.label(), guard=self.guard)
        report = result_verifier.verify_result_tree(
            root, manifest, guard=self.guard)
        return {"manifest_files": manifest["n_files"],
                "total_bytes": manifest["total_bytes"], "verification": report}

    def state_TRANSFER_FL_ARTIFACTS(self) -> dict:
        archive = result_verifier.build_transfer_archive(
            os.path.join(self.out_dir, "ladder"),
            os.path.join(self.out_dir, "FL_ARTIFACTS.zip"),
            label=self.label(), guard=self.guard)
        command = self.payload.get("fl_transfer_command")
        transfer = None
        if command:
            transfer = self._run_command(command, state="TRANSFER_FL_ARTIFACTS")
            if transfer["returncode"] != 0:
                raise SupervisorError(
                    f"FL transfer command failed with rc={transfer['returncode']}")
        return {"archive": {k: v for k, v in archive.items() if k != "manifest"},
                "transfer": transfer}

    def state_TERMINATE_ACCELERATOR(self) -> dict:
        command = self.payload.get("terminate_command")
        record = None
        if command:
            record = self._run_command(command, state="TERMINATE_ACCELERATOR")
        return {"terminate_command": record,
                "note": ("termination is a configured command; this package "
                         "never contacts a provider by itself")}

    # ---------------- the machine ----------------

    def _emergency_close(self, failed_state: str | None) -> dict:
        """Best-effort transfer, then ALWAYS attempt termination (R-C2).

        Runs from ``run``'s ``finally`` block.  Nothing here may raise: a
        failure inside the close-out would strand the pod, so every step is
        recorded (in the journal and in the returned record) and the next one
        still runs.  Termination is attempted even when the transfer failed —
        losing artefacts is bad, paying for an abandoned accelerator is worse,
        and the artefacts also live in the session directory.
        """
        record: dict[str, Any] = {"triggered_by": failed_state,
                                  "transfer": None, "terminate": None}
        if failed_state is None and "TERMINATE_ACCELERATOR" in self.completed:
            record["note"] = "the session closed normally; nothing to force"
            return record
        for state, key in (("TRANSFER_FL_ARTIFACTS", "transfer"),
                           ("TERMINATE_ACCELERATOR", "terminate")):
            if state in self.completed:
                record[key] = {"status": "ALREADY_COMPLETED"}
                continue
            if key == "transfer" and not os.path.isdir(
                    os.path.join(self.out_dir, "ladder")):
                # nothing ran, so there is nothing to transfer; an empty
                # archive would misdescribe the session
                record[key] = {"status": "SKIPPED_NO_LADDER_OUTPUT"}
                continue
            handler = getattr(self, f"state_{state}")
            try:
                self.journal("EMERGENCY_STATE_STARTED", state,
                             {"reason": f"session aborted at {failed_state!r}"})
                result = handler()
            except BaseException as exc:  # noqa: BLE001 - recorded, not raised
                record[key] = {"status": "FAILED", "error": repr(exc)}
                try:
                    self.journal("EMERGENCY_STATE_FAILED", state,
                                 {"error": repr(exc)})
                except BaseException:  # noqa: BLE001 - the journal is gone
                    pass
                continue
            record[key] = {"status": "COMPLETED", "result": _jsonable(result)}
            self.state_results.setdefault(state, result)
            if state not in self.completed:
                self.completed.append(state)
            try:
                self.journal("EMERGENCY_STATE_COMPLETED", state,
                             {"result": _jsonable(result)})
            except BaseException:  # noqa: BLE001
                pass
        record["policy"] = (
            "a failed session still transfers what it has (best effort) and "
            "ALWAYS attempts termination: a rented accelerator keeps billing "
            "until it is terminated, not until the process exits")
        return record

    def run(self, *, resume: bool = True) -> dict:
        started = self.clock.monotonic()
        self._t0 = self._t0 or started
        failed_state = None
        failure = None
        if resume and self.completed:
            self.rebuild_state_results()
        try:
            for state in STATES:
                if resume and state in self.completed:
                    self.journal("STATE_SKIPPED_RESUMED", state, {})
                    continue
                handler = getattr(self, f"state_{state}", None)
                if handler is None:  # pragma: no cover - STATES is closed
                    raise SupervisorError(f"no handler for state {state!r}")
                self.journal("STATE_STARTED", state, {})
                try:
                    result = handler()
                except BaseException as exc:  # noqa: BLE001 - recorded, never swallowed
                    failed_state = state
                    failure = repr(exc)
                    self.journal("STATE_FAILED", state, {"error": failure})
                    break
                self.state_results[state] = result
                self.completed.append(state)
                self.journal("STATE_COMPLETED", state,
                             {"result": _jsonable(result)})
        except BaseException as exc:  # noqa: BLE001 - supervisor-level failure
            failed_state = failed_state or "SUPERVISOR"
            failure = failure or repr(exc)
        finally:
            self.close_out = self._emergency_close(failed_state)

        status = {
            "schema": FINAL_STATUS_SCHEMA,
            "session_id": self.payload.get("session_id"),
            "label": self.label(),
            "rehearsal": self.rehearsal,
            "outcome": "COMPLETE" if failed_state is None
                       else f"ABORTED_AT_{failed_state}",
            "failed_state": failed_state,
            "failure": failure,
            "states_completed": list(self.completed),
            "states_never_started": [s for s in STATES
                                     if s not in self.completed
                                     and s != failed_state],
            "elapsed_seconds": round(self.clock.monotonic() - started, 6),
            "isolation": self.guard.to_dict(),
            "o1_paths_touched_for_hashing": len(self.custodian.touched),
            "determinism": self.determinism,
            "close_out": self.close_out,
        }
        self.guard.write_json(os.path.join(self.out_dir, "SESSION_FINAL_STATUS.json"),
                              status)
        self.journal("SESSION_FINISHED", failed_state or "TERMINATE_ACCELERATOR",
                     {"outcome": status["outcome"],
                      "close_out": self.close_out})
        return status


def _jsonable(value: Any) -> Any:
    """Journal-safe projection (live objects are recorded by type name)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()
                if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return f"<{type(value).__name__}>"


# --------------------------------------------------------------------------
# CLI (the pod-side entry point)
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Foundation Learner B200 combined-session supervisor")
    parser.add_argument("--config", required=True,
                        help="flb200.session_config.v1 JSON file")
    parser.add_argument("--out", required=True, help="session output directory")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing journal and start over")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The pod-side entry point (``deploy/fl_b200_entry.sh`` execs this).

    It builds the supervisor and then wires the PRODUCTION campaign entry onto
    it (``campaign.entry.attach_to_supervisor``), which applies the §22
    determinism configuration and supplies the real ``StageContext`` factory
    and ladder runner.  Before this existed, a pod-side run reached
    ``RUN_FL_LADDER`` and refused for want of a context factory (R-C1).
    """
    args = build_parser().parse_args(argv)
    config = SessionConfig.load(args.config)
    supervisor = SessionSupervisor(config=config, out_dir=args.out)
    if supervisor.rehearsal:
        print(f"*** {REHEARSAL_LABEL}: this session is NOT a scientific run ***",
              file=sys.stderr)
    from . import entry as campaign_entry

    campaign_entry.attach_to_supervisor(supervisor)
    status = supervisor.run(resume=not args.no_resume)
    print(json.dumps({"outcome": status["outcome"],
                      "states_completed": status["states_completed"]},
                     indent=2, sort_keys=True))
    return 0 if status["outcome"] == "COMPLETE" else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
