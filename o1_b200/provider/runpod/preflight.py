#!/usr/bin/env python3
"""Read-only live RunPod preflight (runpod_readonly_preflight).

GET-only by construction: the adapter is built WITHOUT authorization, so
its transport rejects every mutating verb before the network.  Without
RUNPOD_API_KEY the live portion reports SKIPPED_NO_CREDENTIAL and local
construction still succeeds.  All output is fully redacted and the report
is hash-covered.
"""
from __future__ import annotations

import argparse
import json
import os

from .adapter import RunpodV2Adapter
from .identityutil import canonical_sha256, utcnow_iso
from .redaction import load_api_key, redact
from .schema_check import verify_pinned_schema
from .transport import ApiHttpError, TransportError


def run_preflight(*, base_url: str = "https://api.runpod.io",
                  opener=None) -> dict:
    report = {
        "schema": "o1b200.runpod_readonly_preflight.v1",
        "utc": utcnow_iso(),
        "pinned_schema": verify_pinned_schema(),
        "mutations_possible": False,
        "checks": {},
    }
    key = load_api_key()
    if not key:
        report["verdict"] = "SKIPPED_NO_CREDENTIAL"
        report["note"] = ("READONLY_LIVE_PREFLIGHT: SKIPPED_NO_CREDENTIAL — "
                          "set RUNPOD_API_KEY (or RUNPOD_API_KEY_FILE, "
                          "owner-only) and rerun; do not paste the key into "
                          "chat")
        report["report_sha256"] = canonical_sha256(
            "o1b200.runpod_preflight.v1", report)
        return report
    adapter = RunpodV2Adapter(base_url=base_url, opener=opener)
    checks = report["checks"]
    failures = []

    def step(name, fn):
        try:
            checks[name] = json.loads(redact(json.dumps(fn(), default=str)))
        except (ApiHttpError, TransportError, Exception) as exc:  # noqa: BLE001
            checks[name] = {"error": redact(str(exc))}
            failures.append(name)

    step("authentication", adapter.authenticate_readonly)
    step("schema_identity", adapter.get_api_schema_identity)
    step("b200_availability_and_pricing", adapter.get_b200_availability)
    step("compatible_datacenters",
         lambda: {"datacenters": adapter.list_compatible_datacenters()})

    def owned():
        pods = adapter.list_owned_instances()
        active = [p for p in pods if p.status not in ("TERMINATED",)]
        return {
            "owned_pods": len(pods),
            "active_pods": [{"id": p.id, "name": p.name, "status": p.status}
                            for p in active],
            "unexpected_active_billable_pod": bool(active),
        }
    step("owned_pods", owned)
    step("billing", lambda: adapter.get_billing_usage())

    qualifying = (
        "b200_availability_and_pricing" not in failures
        and not checks.get("owned_pods", {}).get(
            "unexpected_active_billable_pod", True))
    report["qualifying_single_b200_secure_offer_now"] = (
        "b200_availability_and_pricing" not in failures)
    report["availability_note"] = (
        "informational only; availability and price MUST be re-checked "
        "immediately before the later authorized launch")
    report["verdict"] = "PASS" if not failures else "FAIL"
    report["failed_checks"] = failures
    report["report_sha256"] = canonical_sha256(
        "o1b200.runpod_preflight.v1", report)
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="https://api.runpod.io")
    a = p.parse_args()
    report = run_preflight(base_url=a.base_url)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"READONLY_LIVE_PREFLIGHT: {report['verdict']}")
    return 0 if report["verdict"] in ("PASS", "SKIPPED_NO_CREDENTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
