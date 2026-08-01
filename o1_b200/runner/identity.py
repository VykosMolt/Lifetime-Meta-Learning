"""Canonical hashing and identity helpers for the B200 runner.

All digests are SHA-256 over domain-tagged canonical JSON so that no two
identity spaces can collide.  The sealed package's own digests
(text_sha256, token_ids_sha256, sha256_file, sha256_tree, derive_stream_seed)
are imported from the verified modules, never re-implemented.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from . import sealed_import

sealed_import.ensure_sealed_path()

from o1_analysis import (  # noqa: E402  (sealed, hash-verified)
    BANK_SIZE,
    N_DIRECTIONS,
    action_index,
    derive_stream_seed,
    sha256_file,
    sha256_tree,
    text_sha256,
    token_ids_sha256,
)

__all__ = [
    "BANK_SIZE", "N_DIRECTIONS", "action_index", "derive_stream_seed",
    "sha256_file", "sha256_tree", "text_sha256", "token_ids_sha256",
    "canonical_json", "sha256_canonical", "domain_sha256", "sha256_bytes",
]


def canonical_json(obj: Any) -> str:
    """Canonical JSON text: sorted keys, minimal separators, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_canonical(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def domain_sha256(domain: str, obj: Any) -> str:
    """Domain-tagged canonical digest; distinct domains cannot collide."""
    payload = domain.encode("utf-8") + b"\0" + canonical_json(obj).encode("utf-8")
    return sha256_bytes(payload)
