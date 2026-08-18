"""Shard writing and reading (contract §16, Amendment 1).

Plain JSONL (one canonical-JSON record per line, no compression, no new
dependencies) plus ``SHARD_SUMS.json`` holding the SHA-256 of every shard file
as it lies on disk.

SEALED_TEST shards are ENCIPHERED at write time: the whole file's bytes are
XORed with the SHA-256 counter keystream of
``K_seal = sha256(b"FL_V0_SEALED_KEY\\0" + split_manifest_sha256_hex)``.

``read_shard`` takes the contract §23 ``sealed_key`` parameter, but it never
derives one for reading: without an explicit key it REFUSES enciphered bytes.
The only code permitted to supply that key is ``campaign/sealed_gate.py``, after
the development decisions are frozen and the single-use opening-ledger entry is
written.  As Amendment 1 already records, the protection is procedural (gate +
ledger + hostile fixtures) with the cipher preventing accidental or casual
plaintext access; no cryptographic unopenability is claimed, since the key is
derived from a public digest.  The hostile fixture asserts that no module under
``ecology/``, ``episodes/``, ``data/`` or ``scripts/`` ever calls a sealed read.

Writes are atomic (temp file + fsync + rename) and byte-deterministic: the same
records always produce the same file bytes, which is what makes re-running the
pre-generation idempotent.
"""
from __future__ import annotations

import json
import os

from ..ecology.base import (canonical_json, seal_bytes, sealed_key,
                            sha256_bytes, unseal_bytes)

__all__ = ["SHARD_SUMS_NAME", "shard_bytes", "write_shard", "read_shard",
           "episode_from_json", "write_json", "read_json", "ShardSums"]

SHARD_SUMS_NAME = "SHARD_SUMS.json"


def shard_bytes(records) -> bytes:
    """Canonical JSONL bytes for a record sequence."""
    return "".join(canonical_json(r) + "\n" for r in records).encode("utf-8")


def _atomic_write(path: str, payload: bytes, skip_if_identical: bool = True) -> bool:
    """Write ``payload`` atomically; return True when the file was rewritten.

    An existing file whose bytes already equal ``payload`` is left untouched,
    which is what makes re-running the pre-generation a genuine no-op.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if skip_if_identical and os.path.exists(path):
        with open(path, "rb") as fh:
            if fh.read() == payload:
                return False
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return True


def write_shard(path: str, records, sealed: bool = False,
                split_manifest_sha256_hex: str | None = None) -> dict:
    """Write one shard; returns its summary record.

    ``sealed=True`` requires the split-manifest digest and enciphers the file.
    """
    payload = shard_bytes(records)
    plaintext_sha = sha256_bytes(payload)
    if sealed:
        if not split_manifest_sha256_hex:
            raise ValueError("sealed shards require the split manifest digest")
        payload = seal_bytes(payload, sealed_key(split_manifest_sha256_hex))
    written = _atomic_write(path, payload)
    return {"path": os.path.basename(path), "records": len(records),
            "bytes": len(payload), "sealed": bool(sealed),
            "sha256": sha256_bytes(payload),
            "plaintext_sha256": None if sealed else plaintext_sha,
            "rewritten": written}


def read_shard(path: str, sealed_key: bytes | None = None) -> list[dict]:
    """Read a shard.

    Without ``sealed_key`` this reads a PLAIN shard and REFUSES enciphered
    bytes.  ``sealed_key`` is the contract §23 parameter: the caller must
    already hold the key, and the only code permitted to DERIVE it is
    ``campaign/sealed_gate.py`` after the development decisions are frozen and
    the single-use opening-ledger entry is written (Amendment 1).  No module in
    ``ecology/``, ``episodes/``, ``data/`` or ``scripts/`` derives it, which the
    hostile fixture ``test_hostile_sealed_plaintext.py`` enforces.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    if sealed_key is not None:
        raw = unseal_bytes(raw, sealed_key)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{os.path.basename(path)} is not a plain shard; sealed shards may "
            f"only be opened through campaign/sealed_gate.py") from exc
    records = []
    for line in text.splitlines():
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{os.path.basename(path)} is not a plain shard; sealed shards "
                f"may only be opened through campaign/sealed_gate.py") from exc
    return records


def episode_from_json(record: dict):
    """Contract §23 constructor: one shard record -> ``episodes.schema.Episode``."""
    from ..episodes.schema import Episode
    return Episode.from_dict(record)


def write_json(path: str, obj) -> dict:
    payload = (canonical_json(obj) + "\n").encode("utf-8")
    written = _atomic_write(path, payload)
    return {"path": os.path.basename(path), "bytes": len(payload),
            "sha256": sha256_bytes(payload), "rewritten": written}


def read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class ShardSums:
    """Accumulator for ``SHARD_SUMS.json``."""

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}

    def add(self, relpath: str, summary: dict) -> None:
        entry = dict(summary)
        entry["path"] = relpath
        if relpath in self.entries:
            raise ValueError(f"duplicate shard path {relpath!r}")
        self.entries[relpath] = entry

    def to_dict(self) -> dict:
        return {"schema": "fl.shard_sums.v1",
                "shards": [self.entries[k] for k in sorted(self.entries)]}
