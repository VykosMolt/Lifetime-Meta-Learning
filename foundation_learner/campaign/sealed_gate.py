"""Sealed-test access control (contract §12 + Amendment 1).

This module is the ONLY decipher path for sealed shards in the whole package.
It is the only place that derives ``K_seal`` and the only place that calls
``ecology.base.unseal_bytes`` on shard bytes.

The gate:

(a) **loader-side refusal** — :func:`loader_guard` refuses any SEALED family
    path presented without a valid unlock record, so an evaluator, the dev
    selector, or a stray script cannot read sealed data by accident;
(b) **prerequisite** — :func:`open_sealed` refuses to derive or apply
    ``K_seal`` unless ``DEV_DECISIONS_FROZEN.json`` exists and validates
    against the ``flb200.dev_decisions.v1`` schema;
(c) **single opening, in two phases** — :func:`open_sealed` grants PROVISIONAL
    access and writes NOTHING; the ``SEALED_OPENED`` entry is appended only
    when :meth:`SealedUnlock.commit` is called, i.e. after the evaluation has
    actually produced records.  A failure before the commit appends a
    ``SEALED_OPENING_ABORTED`` entry instead, which permits EXACTLY ONE retry;
    both attempts stay permanently in the append-only, hash-chained ledger, and
    a third attempt (or any attempt after a committed opening) refuses;
(d) **immutable results** — sealed results are written once, then ``chmod
    0444`` and hash-ledgered; overwriting an existing sealed result is refused,
    and a result may only be written by a COMMITTED unlock.

The two-phase rule exists because a single-use seal that is burned by the FIRST
line of an evaluation is not robust: an infrastructure error (an OOM, a missing
shard, a crashed decode) would consume the one opening without producing any
evidence.  Nothing is weakened by it — the *evidence* is what consumes the
seal, every attempt is recorded forever, and the retry budget is one.

HONEST STATEMENT OF STRENGTH (Amendment 1, restated where it bites).  The key
is derived from a PUBLIC digest (the split-manifest SHA-256), so the cipher is
not a cryptographic seal — anyone holding the package can reconstruct it.  The
protection is PROCEDURAL: the gate, the append-only ledger, the hostile
fixtures, and the fact that no other module in the package contains a decipher
path.  The cipher prevents accidental and casual plaintext access; nothing more
is claimed.  Likewise the 0444 mode and the hash ledger make a silent
modification of a sealed result detectable, not impossible.

The ledger timestamp is a wall-clock UTC string.  That is an AUDIT record, not
a scientific decision input: no code path branches on it (contract §20 forbids
wall-clock in scientific paths, and this is not one).
"""
from __future__ import annotations

import contextlib
import json
import os
import secrets
import stat
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..ecology.base import (canonical_json, domain_sha256, sealed_key,
                            sha256_bytes, sha256_file, unseal_bytes)
from ..ecology.split import compute_split
from . import o1_isolation

__all__ = [
    "SealedGateRefusal",
    "DevDecisionsError",
    "LedgerError",
    "DEV_DECISIONS_NAME",
    "DEV_DECISIONS_SCHEMA",
    "LEDGER_NAME",
    "LEDGER_SCHEMA",
    "SEALED_RESULT_SCHEMA",
    "EVENT_OPENED",
    "EVENT_ABORTED",
    "EVENT_RESULT",
    "MAX_OPENING_ATTEMPTS",
    "sealed_families",
    "is_sealed_family",
    "is_sealed_path",
    "loader_guard",
    "build_dev_decisions",
    "validate_dev_decisions",
    "read_dev_decisions",
    "read_ledger",
    "SealedUnlock",
    "open_sealed",
    "sealed_opening",
]

DEV_DECISIONS_NAME = "DEV_DECISIONS_FROZEN.json"
DEV_DECISIONS_SCHEMA = "flb200.dev_decisions.v1"
LEDGER_NAME = "SEALED_OPENING_LEDGER.jsonl"
LEDGER_SCHEMA = "flb200.sealed_opening_ledger.v1"
SEALED_RESULT_SCHEMA = "flb200.sealed_result.v1"

EVENT_OPENED = "SEALED_OPENED"
EVENT_ABORTED = "SEALED_OPENING_ABORTED"
EVENT_RESULT = "SEALED_RESULT_WRITTEN"

#: Contract §12c + Amendment 12: one opening, and at most one retry after a
#: recorded abort.  Both attempts stay in the ledger forever.
MAX_OPENING_ATTEMPTS = 2

#: Fields ``DEV_DECISIONS_FROZEN.json`` must carry (contract §12 + §8).
DEV_DECISIONS_REQUIRED = (
    "chosen_learning_rate",
    "chosen_scope",
    "updates_U",
    "stage_states",
    "checkpoint_tags",
    "hashes",
)

_GENESIS = "0" * 64


class SealedGateRefusal(RuntimeError):
    """The sealed gate refused an access, an unlock, or a result write."""


class DevDecisionsError(SealedGateRefusal):
    """``DEV_DECISIONS_FROZEN.json`` is missing or malformed."""


class LedgerError(SealedGateRefusal):
    """The opening ledger is missing, broken, or already used."""


# --------------------------------------------------------------------------
# sealed-path recognition + loader guard
# --------------------------------------------------------------------------

def sealed_families() -> frozenset[str]:
    return frozenset(compute_split()["SEALED_TEST"])


def is_sealed_family(family_id: str) -> bool:
    return str(family_id) in sealed_families()


def is_sealed_path(path: str) -> bool:
    """Conservative recognition of a sealed shard path.

    A path counts as sealed when any path component is ``SEALED_TEST`` or when
    the file name begins with a sealed family id.  Conservative means it may
    say "sealed" for a path that merely looks sealed; that direction is safe,
    because the consequence is a refusal.
    """
    resolved = os.path.realpath(os.path.abspath(str(path)))
    parts = resolved.replace(os.sep, "/").split("/")
    if "SEALED_TEST" in parts:
        return True
    base = parts[-1] if parts else ""
    for family in sealed_families():
        if base == family or base.startswith(family + "."):
            return True
    return False


def loader_guard(path: str, *, unlock: "SealedUnlock | None" = None,
                 guard: o1_isolation.IsolationGuard | None = None,
                 context: str = "loader") -> str:
    """Refuse SEALED families without a valid unlock record (contract §12a).

    Every campaign read goes through here, so the O1 isolation guard is applied
    on the same call: one entry point, two invariants.
    """
    resolved = (guard or o1_isolation.default_guard()).guard(
        path, o1_isolation.MODE_READ)
    if is_sealed_path(resolved):
        if unlock is None:
            raise SealedGateRefusal(
                f"[{context}] REFUSED: {path!r} is a SEALED_TEST artefact and "
                "no unlock record exists. Sealed data may only be read through "
                "campaign.sealed_gate.open_sealed() after the development "
                "decisions are frozen and the single-use opening-ledger entry "
                "is written (contract §12).")
        unlock.assert_valid()
    return resolved


# --------------------------------------------------------------------------
# DEV_DECISIONS_FROZEN.json
# --------------------------------------------------------------------------

def build_dev_decisions(*, chosen_learning_rate: float, chosen_scope: str,
                        updates_U: int | None,
                        stage_states: Mapping[str, Any],
                        checkpoint_tags: Mapping[str, Any],
                        hashes: Mapping[str, Any],
                        grid: Sequence[Mapping[str, Any]] = (),
                        selection_rule: str = (
                            "max DEV macro-AULC on the frozen FL3 25 %-step "
                            "grid; tie -> lower learning rate"),
                        notes: Sequence[str] = ()) -> dict:
    """Assemble the frozen development-decision record (contract §12)."""
    payload = {
        "schema": DEV_DECISIONS_SCHEMA,
        "chosen_learning_rate": float(chosen_learning_rate),
        "chosen_scope": str(chosen_scope),
        "updates_U": None if updates_U is None else int(updates_U),
        "stage_states": {str(k): v for k, v in sorted(dict(stage_states).items())},
        "checkpoint_tags": {str(k): v for k, v in sorted(dict(checkpoint_tags).items())},
        "hashes": {str(k): v for k, v in sorted(dict(hashes).items())},
        "grid": [dict(entry) for entry in grid],
        "selection_rule": str(selection_rule),
        "sealed_policy": (
            "SEALED_TEST was not consulted for any development decision; it is "
            "opened once, after this record exists (contract §12)."),
        "notes": list(notes),
    }
    payload["dev_decisions_sha256"] = domain_sha256(
        "FL_V0_DEV_DECISIONS",
        {k: v for k, v in payload.items() if k != "dev_decisions_sha256"})
    return payload


def validate_dev_decisions(payload: Mapping[str, Any], *,
                           source: str = "DEV_DECISIONS_FROZEN.json") -> dict:
    """Hard-fail on a malformed or self-inconsistent decisions record."""
    if payload.get("schema") != DEV_DECISIONS_SCHEMA:
        raise DevDecisionsError(
            f"{source}: schema {payload.get('schema')!r} != {DEV_DECISIONS_SCHEMA!r}")
    missing = [k for k in DEV_DECISIONS_REQUIRED if k not in payload]
    if missing:
        raise DevDecisionsError(f"{source}: missing required fields {missing}")
    if not payload["stage_states"]:
        raise DevDecisionsError(f"{source}: stage_states is empty")
    if not payload["hashes"]:
        raise DevDecisionsError(f"{source}: hashes is empty")
    recorded = payload.get("dev_decisions_sha256")
    recomputed = domain_sha256(
        "FL_V0_DEV_DECISIONS",
        {k: v for k, v in payload.items() if k != "dev_decisions_sha256"})
    if recorded != recomputed:
        raise DevDecisionsError(
            f"{source}: dev_decisions_sha256 mismatch (recorded {recorded!r}, "
            f"recomputed {recomputed!r}); the record was edited after freezing")
    return dict(payload)


def read_dev_decisions(path: str, *,
                       guard: o1_isolation.IsolationGuard | None = None) -> dict:
    guard = guard or o1_isolation.default_guard()
    resolved = guard.guard(path, o1_isolation.MODE_READ)
    if not os.path.isfile(resolved):
        raise DevDecisionsError(
            f"{DEV_DECISIONS_NAME} not found at {path!r}: the sealed set may "
            "not be opened before the development decisions are frozen "
            "(contract §12)")
    try:
        payload = json.loads(guard.read_text(resolved))
    except json.JSONDecodeError as exc:
        raise DevDecisionsError(f"{path}: not valid JSON ({exc})") from exc
    return validate_dev_decisions(payload, source=path)


# --------------------------------------------------------------------------
# the append-only, hash-chained opening ledger
# --------------------------------------------------------------------------

def _entry_hash(entry: Mapping[str, Any]) -> str:
    return domain_sha256(
        "FL_V0_SEALED_LEDGER",
        {k: v for k, v in entry.items() if k != "entry_sha256"})


def read_ledger(path: str, *, guard: o1_isolation.IsolationGuard | None = None
                ) -> list[dict]:
    """Read and VERIFY the hash chain; a broken chain is an error."""
    guard = guard or o1_isolation.default_guard()
    resolved = guard.guard(path, o1_isolation.MODE_READ)
    if not os.path.exists(resolved):
        return []
    entries: list[dict] = []
    prev = _GENESIS
    with open(resolved, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(
                    f"{path}:{lineno + 1}: corrupt ledger line ({exc}); the "
                    "opening ledger is append-only and must never be edited"
                ) from exc
            if entry.get("schema") != LEDGER_SCHEMA:
                raise LedgerError(
                    f"{path}:{lineno + 1}: unexpected schema {entry.get('schema')!r}")
            if entry.get("prev_sha256") != prev:
                raise LedgerError(
                    f"{path}:{lineno + 1}: hash chain broken (expected prev "
                    f"{prev[:16]}..., found {str(entry.get('prev_sha256'))[:16]}...); "
                    "the ledger was truncated or edited")
            recomputed = _entry_hash(entry)
            if entry.get("entry_sha256") != recomputed:
                raise LedgerError(
                    f"{path}:{lineno + 1}: entry hash mismatch; the ledger "
                    "entry was edited after it was written")
            prev = recomputed
            entries.append(entry)
    return entries


def _append_ledger(path: str, event: str, payload: Mapping[str, Any], *,
                   guard: o1_isolation.IsolationGuard,
                   utc: str) -> dict:
    existing = read_ledger(path, guard=guard)
    prev = existing[-1]["entry_sha256"] if existing else _GENESIS
    entry = {
        "schema": LEDGER_SCHEMA,
        "event": str(event),
        "entry_index": len(existing),
        "utc": utc,
        "prev_sha256": prev,
        **{str(k): v for k, v in sorted(dict(payload).items())},
    }
    entry["entry_sha256"] = _entry_hash(entry)
    guard.append_line(path, canonical_json(entry))
    return entry


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# the unlock
# --------------------------------------------------------------------------

@dataclass
class SealedUnlock:
    """A single-use, ledgered permission to read the sealed set.

    The unlock is PROVISIONAL when :func:`open_sealed` returns it: it may read
    sealed shards, but it has written nothing to the ledger and may not write a
    result.  :meth:`commit` appends the single ``SEALED_OPENED`` entry (the
    seal is consumed by evidence, not by intent); :meth:`abort` appends a
    ``SEALED_OPENING_ABORTED`` entry instead and leaves exactly one retry.
    """

    ledger_path: str
    dev_decisions_sha256: str
    split_manifest_sha256: str
    guard: o1_isolation.IsolationGuard
    entry: dict | None = None
    attempt_index: int = 0
    prior_aborted: int = 0
    opening_payload: dict = field(default_factory=dict)
    _key: bytes = field(repr=False, default=b"")
    reads: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    revoked: bool = False
    aborted: bool = False

    # -- validity ----------------------------------------------------------

    @property
    def committed(self) -> bool:
        return self.entry is not None

    def assert_valid(self) -> None:
        if self.revoked:
            raise SealedGateRefusal(
                "this sealed unlock was revoked; a new sealed set is required "
                "for a further cycle (contract §12)")
        if self.aborted:
            raise SealedGateRefusal(
                "this sealed opening attempt was ABORTED and permanently "
                "recorded in the ledger; a retry needs a fresh open_sealed() "
                "call (contract §12c + Amendment 12)")
        if not self._key:
            raise SealedGateRefusal("sealed unlock carries no key")

    def assert_committed(self, *, action: str) -> None:
        self.assert_valid()
        if not self.committed:
            raise SealedGateRefusal(
                f"REFUSED: {action} requires a COMMITTED sealed opening. The "
                "opening is two-phase: read the sealed shards, produce the "
                "evaluation records, then commit() writes the single "
                "SEALED_OPENED ledger entry; only then may an immutable "
                "sealed result be written (contract §12 + Amendment 12).")

    def revoke(self) -> None:
        self.revoked = True
        self._key = b""

    # -- phase two ---------------------------------------------------------

    def commit(self, *, evaluation: Mapping[str, Any] | None = None) -> dict:
        """Write the single ``SEALED_OPENED`` entry.  Idempotent per unlock."""
        self.assert_valid()
        if self.committed:
            raise SealedGateRefusal(
                "this sealed opening is already committed; commit() writes the "
                "single opening entry exactly once")
        entries = read_ledger(self.ledger_path, guard=self.guard)
        if any(e.get("event") == EVENT_OPENED for e in entries):
            raise LedgerError(
                "REFUSED: the ledger already records a committed opening; a "
                "second opening is refused (contract §12c)")
        payload = dict(self.opening_payload)
        payload.update({
            "attempt_index": int(self.attempt_index),
            "prior_aborted_attempts": int(self.prior_aborted),
            "shards_read": len(self.reads),
            "records_read": sum(int(r["records"]) for r in self.reads),
            "evaluation": dict(evaluation or {}),
            "phase": ("COMMITTED_AFTER_EVALUATION: the sealed evidence existed "
                      "before this entry was written"),
        })
        self.entry = _append_ledger(self.ledger_path, EVENT_OPENED, payload,
                                    guard=self.guard, utc=_utc_now())
        return self.entry

    def abort(self, reason: str) -> dict:
        """Record a failed opening attempt; leaves at most one retry."""
        if self.aborted:
            raise SealedGateRefusal("this opening attempt is already aborted")
        if self.committed:
            raise SealedGateRefusal(
                "a committed opening cannot be aborted; the seal is consumed")
        payload = dict(self.opening_payload)
        payload.update({
            "attempt_index": int(self.attempt_index),
            "prior_aborted_attempts": int(self.prior_aborted),
            "shards_read": len(self.reads),
            "records_read": sum(int(r["records"]) for r in self.reads),
            "reason": str(reason)[:2000],
            "retries_remaining": max(
                0, MAX_OPENING_ATTEMPTS - (self.prior_aborted + 1)),
            "phase": ("ABORTED_BEFORE_EVALUATION_RECORDS: no sealed result was "
                      "written; this attempt is permanently recorded"),
        })
        entry = _append_ledger(self.ledger_path, EVENT_ABORTED, payload,
                               guard=self.guard, utc=_utc_now())
        self.aborted = True
        self.entry = None
        self._key = b""
        return entry

    @property
    def key_fingerprint(self) -> str:
        """Digest of the key, so records can bind it without printing it."""
        return sha256_bytes(self._key) if self._key else ""

    # -- reading -----------------------------------------------------------

    def read_shard(self, path: str) -> list[dict]:
        """THE sealed read.  Every sealed record in the campaign comes here."""
        self.assert_valid()
        resolved = loader_guard(path, unlock=self, guard=self.guard,
                                context="sealed_gate.read_shard")
        with open(resolved, "rb") as fh:
            raw = fh.read()
        plaintext = unseal_bytes(raw, self._key)
        try:
            text = plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SealedGateRefusal(
                f"{path}: deciphering did not yield UTF-8 text; either the "
                "shard is not sealed with this split manifest's key or the "
                "file is corrupt") from exc
        records = []
        for line in text.splitlines():
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SealedGateRefusal(
                    f"{path}: deciphered content is not canonical JSONL") from exc
        self.reads.append({"path": resolved, "records": len(records),
                           "sha256": sha256_bytes(raw)})
        return records

    def read_shards(self, paths: Iterable[str]) -> list[dict]:
        out: list[dict] = []
        for path in paths:
            out.extend(self.read_shard(path))
        return out

    # -- immutable results --------------------------------------------------

    def _write_immutable(self, path: str, payload: bytes, kind: str) -> dict:
        self.assert_committed(action="writing a sealed result")
        resolved = self.guard.guard(path, o1_isolation.MODE_WRITE)
        if os.path.exists(resolved):
            raise SealedGateRefusal(
                f"{path!r} already exists; sealed results are written ONCE and "
                "are immutable (contract §12d)")
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
        tmp = resolved + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, resolved)
        os.chmod(resolved, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0444
        digest = sha256_file(resolved)
        entry = _append_ledger(
            self.ledger_path, EVENT_RESULT,
            {"result_path": os.path.basename(resolved),
             "result_full_path": resolved,
             "result_kind": kind,
             "result_sha256": digest,
             "result_bytes": len(payload),
             "mode": "0444",
             "dev_decisions_sha256": self.dev_decisions_sha256},
            guard=self.guard, utc=_utc_now())
        record = {"path": resolved, "sha256": digest, "bytes": len(payload),
                  "kind": kind, "ledger_entry_sha256": entry["entry_sha256"]}
        self.results.append(record)
        return record

    def write_result(self, path: str, obj: Any) -> dict:
        """Write one immutable, hash-ledgered sealed result document."""
        self.assert_committed(action="writing a sealed result")
        document = dict(obj)
        document.setdefault("schema", SEALED_RESULT_SCHEMA)
        document["sealed_opening_entry_sha256"] = self.entry["entry_sha256"]
        document["dev_decisions_sha256"] = self.dev_decisions_sha256
        payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        return self._write_immutable(path, payload, "json")

    def write_result_records(self, path: str, records: Sequence[Mapping[str, Any]]
                             ) -> dict:
        payload = "".join(canonical_json(r) + "\n" for r in records).encode("utf-8")
        return self._write_immutable(path, payload, "jsonl")

    def to_dict(self) -> dict:
        return {
            "schema": "flb200.sealed_unlock.v1",
            "ledger_path": self.ledger_path,
            "dev_decisions_sha256": self.dev_decisions_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "opening_entry_sha256": (None if self.entry is None
                                     else self.entry["entry_sha256"]),
            "committed": bool(self.committed),
            "aborted": bool(self.aborted),
            "attempt_index": int(self.attempt_index),
            "prior_aborted_attempts": int(self.prior_aborted),
            "key_fingerprint": self.key_fingerprint,
            "reads": list(self.reads),
            "results": list(self.results),
            "revoked": bool(self.revoked),
        }


def open_sealed(*, ledger_path: str, dev_decisions_path: str,
                split_manifest_path: str,
                stage_states: Mapping[str, Any] | None = None,
                opened_by: str = "campaign.session_supervisor",
                guard: o1_isolation.IsolationGuard | None = None,
                utc_fn=_utc_now,
                verify_split_sources: bool = True) -> SealedUnlock:
    """Phase ONE of the single, ledgered sealed opening (contract §12).

    Refuses unless ``DEV_DECISIONS_FROZEN.json`` exists and validates, refuses
    outright when the ledger already records a COMMITTED opening, and refuses
    once the recorded aborted attempts have exhausted the retry budget
    (:data:`MAX_OPENING_ATTEMPTS`).

    The returned unlock is PROVISIONAL: nothing has been appended to the
    ledger.  The caller must call :meth:`SealedUnlock.commit` after the
    evaluation has produced its records, or :meth:`SealedUnlock.abort` — which
    :func:`sealed_opening` does automatically.
    """
    guard = guard or o1_isolation.default_guard()
    decisions = read_dev_decisions(dev_decisions_path, guard=guard)
    decisions_file_sha256 = sha256_file(
        guard.guard(dev_decisions_path, o1_isolation.MODE_READ))

    entries = read_ledger(ledger_path, guard=guard)
    prior = [e for e in entries if e.get("event") == EVENT_OPENED]
    if prior:
        raise LedgerError(
            f"REFUSED: the sealed set has already been opened "
            f"(ledger entry {prior[0].get('entry_index')} at "
            f"{prior[0].get('utc')}). A second opening is refused; a further "
            "cycle requires a NEW sealed set (contract §12c).")
    aborted = [e for e in entries if e.get("event") == EVENT_ABORTED]
    if len(aborted) >= MAX_OPENING_ATTEMPTS:
        raise LedgerError(
            f"REFUSED: {len(aborted)} sealed opening attempts are already "
            f"recorded in the ledger and the budget is {MAX_OPENING_ATTEMPTS} "
            "(one attempt plus one retry). Every attempt is permanent; a "
            "further cycle requires a NEW sealed set (contract §12c + "
            "Amendment 12).")

    from ..ecology.manifests import read_split_manifest, verify_split_manifest

    manifest_path = guard.guard(split_manifest_path, o1_isolation.MODE_READ)
    if not os.path.isfile(manifest_path):
        raise SealedGateRefusal(
            f"family split manifest not found at {split_manifest_path!r}; the "
            "sealed key is derived from its digest (Amendment 1)")
    manifest = read_split_manifest(manifest_path)
    verify_split_manifest(manifest, check_sources=verify_split_sources)
    split_hex = manifest["split_manifest_sha256"]

    opening_payload = {
        "dev_decisions_sha256": decisions["dev_decisions_sha256"],
        "dev_decisions_file_sha256": decisions_file_sha256,
        "dev_decisions_path": os.path.basename(dev_decisions_path),
        "split_manifest_sha256": split_hex,
        "stage_states": dict(stage_states or decisions["stage_states"]),
        "chosen_scope": decisions["chosen_scope"],
        "chosen_learning_rate": decisions["chosen_learning_rate"],
        "updates_U": decisions["updates_U"],
        "opened_by": str(opened_by),
        # Audit-only uniqueness tag.  Two openings of the same campaign in the
        # same wall-clock second would otherwise produce byte-identical
        # entries, and a deleted-then-recreated ledger would be
        # indistinguishable from the original.  With the nonce, the entry hash
        # recorded INSIDE every (read-only) sealed result is a cross-check the
        # ledger cannot forge by being deleted.  It never enters a scientific
        # decision (contract §20).
        "opening_nonce": secrets.token_hex(16),
        "utc_provisional": utc_fn(),
        "policy": ("single opening; no model modification may be justified "
                   "from sealed outcomes; a future cycle requires a NEW "
                   "sealed set"),
    }

    return SealedUnlock(
        ledger_path=guard.guard(ledger_path, o1_isolation.MODE_WRITE),
        dev_decisions_sha256=decisions["dev_decisions_sha256"],
        split_manifest_sha256=split_hex,
        guard=guard,
        attempt_index=len(aborted),
        prior_aborted=len(aborted),
        opening_payload=opening_payload,
        _key=sealed_key(split_hex),
    )


@contextlib.contextmanager
def sealed_opening(**kwargs):
    """Two-phase opening as a context manager (contract §12c + Amendment 12).

    Yields a PROVISIONAL :class:`SealedUnlock`.  The body must call
    ``unlock.commit(...)`` once its evaluation records exist; anything that
    escapes the body before that appends a ``SEALED_OPENING_ABORTED`` entry —
    leaving exactly one retry — and re-raises.  The exception is never
    swallowed: the abort is a RECORD, not a recovery.
    """
    unlock = open_sealed(**kwargs)
    try:
        yield unlock
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        if not unlock.committed and not unlock.aborted:
            unlock.abort(f"{type(exc).__name__}: {exc}")
        raise
    else:
        if not unlock.committed:
            unlock.abort(
                "the opening context exited without committing: no evaluation "
                "records were produced")
