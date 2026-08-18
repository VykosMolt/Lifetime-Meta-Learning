"""Foundation Learner V0 — evaluation package (contract sections 6, 9, 22, 23).

This package contains every ONLINE evaluation path of the campaign:

* ``generation``      — manual left-padded greedy decode loop (never
                        ``model.generate``) + the batched/unbatched
                        equivalence gate required by contract section 22.
* ``scoring``         — teacher-forced mean per-token log-probability of an
                        exact character span (FL4 target machinery).
* ``learning_curve``  — the online episode walker over ``EPISODE_STRUCTURE_V0``
                        producing ``R_0..R_6`` and full transcript records.
* ``context_reset``   — load-bearing context-reset evaluation (section 9).
* ``fl0_base``        — FL0 base-model learning-curve sweep on DEVELOPMENT.
* ``interference``    — A->B->A chains (retention / interference / recovery).
* ``poison_eval``     — the five frozen poison conditions.
* ``remap_eval``      — surface-remap robustness.
* ``family_holdout``  — UNSEEN-INSTANCE vs UNSEEN-FAMILY separation.
* ``metrics``         — the 14 frozen metrics as pure functions over records.

Hash / seed conventions
-----------------------
Contract section 3 requires the ``flhash`` helpers to be implemented ONCE in
``foundation_learner/ecology/base.py`` and imported everywhere.  This package
therefore never re-implements ``canonical_json`` / ``domain_sha256`` /
``derive_seed``; it resolves them lazily from that module and fails loudly if
they are unavailable.  ``text_sha256`` is a plain ``hashlib`` digest of UTF-8
bytes (no domain convention involved) and is defined here.

Nothing in this package imports its own submodules at import time, so the
submodules may safely import these helpers from the package.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

VERSION = "0.1.0"

_FLHASH_NAMES = ("canonical_json", "domain_sha256", "derive_seed")

_flhash_cache: dict[str, Callable[..., Any]] = {}


class FlHashUnavailable(ImportError):
    """Raised when ``ecology/base.py`` (the single flhash home) is missing."""


def _resolve_flhash() -> dict[str, Callable[..., Any]]:
    """Resolve the shared hash helpers from ``foundation_learner.ecology.base``.

    Contract section 3 fixes both the home of these functions and their exact
    semantics.  We accept either module-level functions or functions grouped
    under a ``flhash`` attribute (namespace/class), because the producing
    worker owns that file; we never substitute our own implementation.
    """
    if _flhash_cache:
        return _flhash_cache
    try:
        from foundation_learner.ecology import base as _base  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised via error path test
        raise FlHashUnavailable(
            "foundation_learner.ecology.base is required for the frozen hash / "
            "seed conventions (contract section 3); it could not be imported: "
            f"{exc!r}"
        ) from exc
    holder = getattr(_base, "flhash", _base)
    resolved: dict[str, Callable[..., Any]] = {}
    for name in _FLHASH_NAMES:
        fn = getattr(holder, name, None)
        if fn is None:
            fn = getattr(_base, name, None)
        if not callable(fn):
            raise FlHashUnavailable(
                f"foundation_learner.ecology.base does not expose a callable "
                f"{name!r} (contract section 3)"
            )
        resolved[name] = fn
    _flhash_cache.update(resolved)
    return _flhash_cache


def canonical_json(obj: Any) -> str:
    """``ecology.base.canonical_json`` (contract section 3)."""
    return _resolve_flhash()["canonical_json"](obj)


def domain_sha256(domain: str, obj: Any) -> str:
    """``ecology.base.domain_sha256`` (contract section 3)."""
    return _resolve_flhash()["domain_sha256"](domain, obj)


def derive_seed(root_seed: int, *tags: Any) -> int:
    """``ecology.base.derive_seed`` (contract section 3)."""
    return _resolve_flhash()["derive_seed"](root_seed, *tags)


def text_sha256(text: str) -> str:
    """SHA-256 of UTF-8 text bytes (plain digest, no domain separation)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# small atomic writers (contract section 22 packaging conventions)
# --------------------------------------------------------------------------
def _atomic_write_text(path: str, text: str) -> str:
    import os

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.abspath(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, os.path.abspath(path))
    return os.path.abspath(path)


def write_json_document(path: str, obj: Any) -> str:
    """JSON with ``indent=2, sort_keys=True`` + trailing newline, written atomically."""
    import json

    return _atomic_write_text(
        path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False) + "\n")


def write_jsonl_records(path: str, records: Any) -> str:
    """One canonical-JSON record per line (the shard/record convention)."""
    lines = [canonical_json(r) for r in records]
    return _atomic_write_text(path, "".join(line + "\n" for line in lines))


def read_jsonl_records(path: str) -> list[Any]:
    import json

    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


__all__ = [
    "VERSION",
    "FlHashUnavailable",
    "canonical_json",
    "domain_sha256",
    "derive_seed",
    "text_sha256",
    "write_json_document",
    "write_jsonl_records",
    "read_jsonl_records",
]
