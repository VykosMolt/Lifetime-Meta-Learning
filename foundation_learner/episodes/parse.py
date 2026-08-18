"""The strict answer grammar shared by every verifier (contract §4).

An attempt is accepted only through the LAST line that matches

    ^ANSWER:\\s*(\\S.*?)\\s*$

The match is anchored, case-sensitive, and requires the payload to start with a
non-space character.  There is NO substring credit: ``the answer is YES`` on an
ANSWER line canonicalises to ``THE ANSWER IS YES`` and fails, ``answer: YES``
(lowercase marker) does not match at all, and a leading space before ``ANSWER``
does not match.  This mirrors the O1 lesson about lenient parsers.

Payload canonicalisation is declared per family through ``canon_mode``:

===============  ======================================================
mode             canonical form
===============  ======================================================
``label``        uppercase, internal whitespace collapsed
``int``          strict optionally-signed integer, re-rendered
``string``       all whitespace removed, lowercased
``token_list``   split on whitespace/commas, lowercased, space-joined
``set``          token_list, de-duplicated and sorted; ``EMPTY`` for none
``formula``      all whitespace removed, lowercased
``assignment``   ``name=value`` pairs, lowercased names, sorted by name
===============  ======================================================

``canonicalize`` returns ``None`` for a payload that does not parse in the
declared mode; the verifier then scores the attempt incorrect.
"""
from __future__ import annotations

import re

__all__ = ["ANSWER_LINE_RE", "CANON_MODES", "parse_answer", "canonicalize",
           "format_answer"]

ANSWER_LINE_RE = re.compile(r"^ANSWER:\s*(\S.*?)\s*$")

CANON_MODES = ("label", "int", "string", "token_list", "set", "formula",
               "assignment")

_WS = re.compile(r"\s+")
_SPLIT = re.compile(r"[,\s]+")
_INT = re.compile(r"[+-]?\d+")
_ASSIGN = re.compile(r"([A-Za-z][A-Za-z0-9_]*)=([+-]?\d+)")


def parse_answer(text: str) -> str | None:
    """Return the payload of the LAST well-formed ANSWER line, else ``None``."""
    payload = None
    for line in text.splitlines():
        m = ANSWER_LINE_RE.match(line)
        if m is not None:
            payload = m.group(1)
    return payload


def format_answer(payload: str) -> str:
    """Render a canonical payload as the protocol's answer line."""
    return f"ANSWER: {payload}"


def canonicalize(payload: str | None, mode: str) -> str | None:
    """Canonicalise ``payload`` under the family-declared ``mode``."""
    if payload is None:
        return None
    if mode not in CANON_MODES:
        raise ValueError(f"unknown canonicalisation mode {mode!r}")
    text = payload.strip()
    if not text:
        return None

    if mode == "label":
        return _WS.sub(" ", text).upper()

    if mode == "int":
        if not _INT.fullmatch(text):
            return None
        return str(int(text))

    if mode in ("string", "formula"):
        return _WS.sub("", text).lower()

    if mode == "token_list":
        toks = [t for t in _SPLIT.split(text) if t]
        return " ".join(t.lower() for t in toks) if toks else None

    if mode == "set":
        if _WS.sub("", text).upper() == "EMPTY":
            return "EMPTY"
        toks = [t.lower() for t in _SPLIT.split(text) if t]
        if not toks:
            return None
        return " ".join(sorted(set(toks)))

    # assignment
    parts = [p for p in _SPLIT.split(text) if p]
    pairs: list[tuple[str, int]] = []
    for part in parts:
        m = _ASSIGN.fullmatch(part)
        if m is None:
            return None
        pairs.append((m.group(1).lower(), int(m.group(2))))
    if not pairs:
        return None
    names = [n for n, _ in pairs]
    if len(set(names)) != len(names):
        return None
    return " ".join(f"{n}={v}" for n, v in sorted(pairs))
