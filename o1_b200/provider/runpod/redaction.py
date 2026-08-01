"""Secret redaction layer for every RunPod-facing surface.

Covers Authorization headers, bearer tokens, RunPod API keys, registry
credentials, pre-signed URLs, Git credentials, and SSH material.  Every
transport log line, exception, report, and archive passes through here.
The API key itself is additionally registered at load time so even an
accidental interpolation of the raw value is caught.
"""
from __future__ import annotations

import os
import re
import stat

REDACTED = "[REDACTED]"

_PATTERNS = [
    re.compile(r"(?i)(authorization\s*[:=]\s*)((bearer|basic)\s+)?[A-Za-z0-9+/._~-]{8,}={0,2}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}={0,2}"),
    re.compile(r"\brpa_[A-Za-z0-9]{8,}\b"),                      # RunPod API key shape
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"(?i)(api[_-]?key|apikey|secret|token|password|passwd|credentials?)"
               r"(\"?\s*[:=]\s*\"?)[^\s\"',;&]{6,}"),
    re.compile(r"(?i)[?&](X-Amz-Signature|X-Amz-Credential|X-Amz-Security-Token|"
               r"sig|signature|sv|se|sp|sr|spr|st|sig\d*)=[^&\s\"']+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bssh-(rsa|ed25519|dss|ecdsa)\s+[A-Za-z0-9+/=]{40,}"),
    re.compile(r"(?i)https?://[^/\s:]+:[^@\s]+@"),               # user:pass@ URLs
]

_EXPLICIT_SECRETS: set[str] = set()


class SecretHandlingError(RuntimeError):
    pass


def register_secret(value: str | None) -> None:
    """Register a literal secret so redact() removes it wherever it appears."""
    if value and len(value) >= 6:
        _EXPLICIT_SECRETS.add(value)


def clear_registered_secrets() -> None:
    _EXPLICIT_SECRETS.clear()


def redact(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for secret in _EXPLICIT_SECRETS:
        if secret in text:
            text = text.replace(secret, REDACTED)
    for pat in _PATTERNS:
        text = pat.sub(lambda m: _keep_label(m), text)
    return text


def _keep_label(m: re.Match) -> str:
    g = m.group(0)
    # keep a stable label prefix where the pattern captured one
    try:
        label = m.group(1)
    except IndexError:
        label = None
    if label and g.lower().startswith(label.lower()):
        return f"{label}{REDACTED}"
    return REDACTED


def redact_mapping(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            if re.search(r"(?i)auth|key|token|secret|password|credential", str(k)):
                out[k] = REDACTED
            else:
                out[k] = redact(v)
        elif isinstance(v, dict):
            out[k] = redact_mapping(v)
        elif isinstance(v, list):
            out[k] = [redact_mapping(x) if isinstance(x, dict)
                      else redact(x) if isinstance(x, str) else x for x in v]
        else:
            out[k] = v
    return out


def load_api_key() -> str | None:
    """Load the RunPod API key from RUNPOD_API_KEY or a secret-file descriptor.

    RUNPOD_API_KEY_FILE, when set, must reference an owner-only (0600 or
    stricter) file.  The loaded key is immediately registered for redaction.
    Returns None when no credential is configured (never raises for absence).
    """
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        path = os.environ.get("RUNPOD_API_KEY_FILE")
        if path:
            st = os.stat(path)
            if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise SecretHandlingError(
                    "RUNPOD_API_KEY_FILE must be owner-only (chmod 600)")
            with open(path, encoding="utf-8") as fh:
                key = fh.read().strip()
    if key:
        register_secret(key)
        return key
    return None
