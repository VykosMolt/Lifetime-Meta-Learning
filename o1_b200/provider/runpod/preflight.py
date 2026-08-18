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
from .redaction import redact
from .schema_check import verify_pinned_schema
from .transport import ApiHttpError, TransportError, resolve_credential


def run_preflight(*, base_url: str = "https://api.runpod.io",
                  api_key: str | None = None, opener=None) -> dict:
    report = {
        "schema": "o1b200.runpod_readonly_preflight.v1",
        "utc": utcnow_iso(),
        "pinned_schema": verify_pinned_schema(),
        "mutations_possible": False,
        "checks": {},
    }
    # resolve_credential (not load_api_key) so a non-production base_url can
    # never receive the ambient operator credential
    key = resolve_credential(base_url, api_key)
    if not key:
        report["verdict"] = "SKIPPED_NO_CREDENTIAL"
        report["note"] = ("READONLY_LIVE_PREFLIGHT: SKIPPED_NO_CREDENTIAL — "
                          "set RUNPOD_API_KEY (or RUNPOD_API_KEY_FILE, "
                          "owner-only) and rerun; do not paste the key into "
                          "chat")
        report["report_sha256"] = canonical_sha256(
            "o1b200.runpod_preflight.v1", report)
        return report
    adapter = RunpodV2Adapter(base_url=base_url, api_key=key, opener=opener)
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
    step("graphql_spot_contract", adapter.graphql.verify_contract)
    step("profile_availability_and_spot_pricing", adapter.get_availability)

    def all_profile_spot():
        # informational: BOTH profiles' live spot state, so the report shows
        # the fallback landscape even when the primary qualifies
        from .policy import PROFILE_PREFERENCE
        out = {}
        for p in PROFILE_PREFERENCE:
            try:
                out[p.key] = adapter.graphql.spot_pricing(p.gpu_type_id)
            except Exception as exc:  # noqa: BLE001
                out[p.key] = {"error": redact(str(exc))}
        return out
    step("spot_pricing_all_profiles", all_profile_spot)
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

    report["qualifying_offer_now"] = (
        "profile_availability_and_spot_pricing" not in failures)
    report["selected_profile"] = checks.get(
        "profile_availability_and_spot_pricing", {}).get("profile")
    report["fallback_reason"] = checks.get(
        "profile_availability_and_spot_pricing", {}).get("fallback_reason")
    report["availability_note"] = (
        "informational only; availability, spot price and profile selection "
        "MUST be re-derived immediately before the later authorized launch "
        "and before every reacquisition")
    # live availability being NONE right now is a market state, not a
    # software failure: it must not block the software-readiness verdict
    if "profile_availability_and_spot_pricing" in failures:
        detail = str(checks["profile_availability_and_spot_pricing"])
        if "qualifies right now" in detail or "NONE" in detail:
            failures = [f for f in failures
                        if f != "profile_availability_and_spot_pricing"]
            report["capacity_unavailable_now"] = True
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
