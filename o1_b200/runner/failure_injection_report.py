#!/usr/bin/env python3
"""Generate reports/FAILURE_INJECTION_REPORT.json.

Reruns the mocked zero-touch state machine with a fault injected at every
injectable state, plus provider-operation faults and a termination fault, and
records the machine's behavior for each.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from .dress_rehearsal import run_rehearsal
from .persistence import atomic_write_text
from .provider_adapter import MockProviderAdapter, ProviderError
from .state_machine import STATES

SUBSET = ["b200val-000-commit_a", "b200val-004-malformed_eos",
          "b200val-008-stoch_commit"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    injectable = [s for s in STATES if s not in ("TERMINATE", "COMPLETE")]
    entries = []
    for state in injectable:
        d = os.path.join(a.work, f"state_{state.lower()}")
        shutil.rmtree(d, ignore_errors=True)
        status = run_rehearsal(a.corpus, d, subset=SUBSET, fail_at=state)
        entries.append({
            "injection": f"handler exception at {state}",
            "outcome": status["outcome"],
            "later_states_never_started": status.get("states_never_started"),
            "termination_requested":
                status["termination"]["termination_requested"],
            "termination_confirmed":
                status["termination"]["termination_confirmed"],
            "human_interaction_required":
                status.get("human_interaction_required", False),
        })
    for op in ("quote_instance", "validate_single_gpu", "start_instance",
               "get_billing_rate", "upload_artifacts", "launch_job",
               "download_results", "terminate_instance", "confirm_terminated"):
        d = os.path.join(a.work, f"provider_{op}")
        shutil.rmtree(d, ignore_errors=True)
        provider = MockProviderAdapter(
            fail_on={op: ProviderError(f"injected {op} failure")})
        status = run_rehearsal(a.corpus, d, subset=SUBSET, provider=provider)
        entries.append({
            "injection": f"provider failure at {op}",
            "outcome": status["outcome"],
            "termination_requested":
                status["termination"].get("termination_requested", False),
            "termination_confirmed":
                status["termination"].get("termination_confirmed", False),
            "human_interaction_required":
                status.get("human_interaction_required", False),
        })
    ok = all(
        (e["outcome"].startswith("ABORTED_AT_") or e["outcome"] == "COMPLETE")
        for e in entries)
    report = {
        "schema": "o1b200.failure_injection_report.v1",
        "mode": "LOCAL_MOCKED",
        "n_injections": len(entries),
        "all_handled_without_human_interaction_except_termination_failures":
            all(e["human_interaction_required"] is False
                or "terminate" in e["injection"] for e in entries),
        "every_failure_reached_a_machine_readable_terminal_status": ok,
        "entries": entries,
    }
    atomic_write_text(a.out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("n_injections",
                       "every_failure_reached_a_machine_readable_terminal_status")},
                     indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
