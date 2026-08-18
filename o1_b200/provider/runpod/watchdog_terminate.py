#!/usr/bin/env python3
"""Independent watchdog termination path.

A SEPARATE PROCESS with a minimal raw HTTPS client (http.client only — no
shared transport code with the primary adapter), so a primary-client defect
cannot take both termination paths down.  It targets the exact recorded Pod
ID, uses the same authorization identity (env flag + API key), requests
PERMANENT termination (action=terminate then DELETE — never merely stop),
polls for confirmed termination, records redacted responses, and fails
loudly when termination cannot be confirmed.

Run as:  python -m o1_b200.provider.runpod.watchdog_terminate \
             --pod-id <id> --deadline-epoch <t> --out <dir>
The parent arms it at Pod creation; it fires at the hard budget deadline or
immediately upon receiving SIGUSR1 (terminate_now).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import time


PRODUCTION_HOST = "api.runpod.io"


def _redact(text: str) -> str:
    for var in ("RUNPOD_API_KEY", "RUNPOD_MOCK_API_KEY"):
        key = os.environ.get(var, "")
        if key:
            text = text.replace(key, "[REDACTED]")
    return text


def _load_key(host: str, scheme: str = "https") -> str | None:
    # Same isolation invariant as transport.py: the operator credential is
    # only ever sent to the production host, and only over TLS.  Any other
    # host or scheme (mock servers in tests, a cleartext misconfiguration)
    # requires the separate synthetic RUNPOD_MOCK_API_KEY.
    if (host, scheme) != (PRODUCTION_HOST, "https"):
        return os.environ.get("RUNPOD_MOCK_API_KEY")
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    # RUNPOD_API_KEY_FILE is a documented, supported credential form; a
    # watchdog that ignored it would arm silently and then be unable to
    # terminate anything — the independent path dead exactly when needed.
    path = os.environ.get("RUNPOD_API_KEY_FILE")
    if path:
        try:
            st = os.stat(path)
            if st.st_mode & 0o077:
                return None      # not owner-only: refuse, do not weaken
            with open(path, encoding="utf-8") as fh:
                return fh.read().strip() or None
        except OSError:
            return None
    return None


def _request(method: str, path: str, body: dict | None = None,
             host: str = "api.runpod.io", port: int | None = None,
             scheme: str = "https") -> tuple[int, str]:
    key = _load_key(host, scheme)
    if not key:
        return -1, "no credential"
    if scheme == "http":
        conn = http.client.HTTPConnection(host, port or 80, timeout=30)
    else:
        conn = http.client.HTTPSConnection(host, port or 443, timeout=30)
    try:
        headers = {"Authorization": f"Bearer {key}",
                   "Accept": "application/json"}
        payload = None
        if body is not None:
            payload = json.dumps(body)
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        return resp.status, _redact(resp.read().decode("utf-8", "replace"))
    finally:
        conn.close()


def terminate_pod(pod_id: str, log, host="api.runpod.io", port=None,
                  scheme="https", sleep=time.sleep,
                  confirm_timeout: float = 300.0) -> bool:
    if os.environ.get("RUNPOD_ALLOW_BILLABLE_MUTATIONS") != \
            "YES_I_AUTHORIZE_THIS_RUN":
        log({"error": "LIVE_MUTATION_NOT_AUTHORIZED",
             "note": "watchdog refuses without the authorization env flag"})
        return False
    for method, path, body in (
            ("POST", f"/v2/pods/{pod_id}/action", {"action": "terminate"}),
            ("DELETE", f"/v2/pods/{pod_id}", None)):
        try:
            status, text = _request(method, path, body, host, port, scheme)
            log({"request": f"{method} {path}", "status": status,
                 "body": text[:300]})
        except OSError as exc:
            log({"request": f"{method} {path}", "error": str(exc)})
    deadline = time.monotonic() + confirm_timeout
    while time.monotonic() < deadline:
        try:
            status, text = _request("GET", f"/v2/pods/{pod_id}", None,
                                    host, port, scheme)
        except OSError as exc:
            log({"poll_error": str(exc)})
            sleep(5)
            continue
        if status == 404:
            log({"confirmed": True, "via": "404"})
            return True
        if status == 200 and '"TERMINATED"' in text:
            log({"confirmed": True, "via": "status"})
            return True
        sleep(5)
    log({"confirmed": False, "LOUD_FAILURE":
         "watchdog could not confirm termination; operator/billing-side "
         "stop is required"})
    return False


class WatchdogHandle:
    """Parent-side handle to the armed watchdog process."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc

    def terminate_now(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGUSR1)

    def alive(self) -> bool:
        return self.proc.poll() is None


def spawn_watchdog(*, pod_id: str, hard_limit_seconds: int,
                   out_dir: str, host: str = "api.runpod.io",
                   port: int | None = None,
                   scheme: str = "https") -> WatchdogHandle:
    deadline = time.time() + hard_limit_seconds
    cmd = [sys.executable, "-m",
           "o1_b200.provider.runpod.watchdog_terminate",
           "--pod-id", pod_id, "--deadline-epoch", str(deadline),
           "--out", out_dir, "--host", host, "--scheme", scheme]
    if port:
        cmd += ["--port", str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    return WatchdogHandle(proc)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pod-id", required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="api.runpod.io")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--scheme", default="https")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "watchdog_termination.jsonl")

    def log(entry: dict) -> None:
        entry = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 **entry}
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(_redact(json.dumps(entry)) + "\n")

    fired = {"now": False}

    def on_usr1(_sig, _frm):
        fired["now"] = True

    signal.signal(signal.SIGUSR1, on_usr1)
    log({"armed": True, "pod_id": a.pod_id,
         "deadline_epoch": a.deadline_epoch})
    while time.time() < a.deadline_epoch and not fired["now"]:
        time.sleep(1)
    log({"firing": "signal" if fired["now"] else "deadline"})
    ok = terminate_pod(a.pod_id, log, a.host, a.port, a.scheme)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
