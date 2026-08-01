"""Small identity helpers local to the RunPod provider package."""
from __future__ import annotations

import hashlib
import json
import time


def canonical_sha256(domain: str, obj) -> str:
    body = {k: v for k, v in obj.items()
            if k not in ("quote_sha256", "request_sha256", "report_sha256")}
    payload = domain.encode() + b"\0" + json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def utcnow_iso(now=time.time) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(now())))
