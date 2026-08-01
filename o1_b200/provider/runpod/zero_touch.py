#!/usr/bin/env python3
"""RunPod zero-touch session driver.

Layering: the SCIENTIFIC zero-touch state machine (hardware validation,
non-O1 equivalence, bounded benchmark, backend selection, replacement
precommit, external commit verification, affordability gate, calibration,
record verify) runs ON THE POD via the container start command.  This driver
owns the PROVIDER side with no human interaction:

  1. live-mutation interlock (or immediate refusal);
  2. read-only preflight + FRESH quote (stale quotes refused);
  3. canonical Pod-request render + authorization hash check;
  4. reconciled, duplicate-safe Pod creation; watchdogs armed;
  5. monitor to RUNNING; poll logs/status; budget soft/hard stops;
  6. wait for the pod's machine-readable FINAL_STATUS artifact;
  7. verified result download;
  8. TERMINATE (never merely stop) + independent confirmation;
  9. machine-readable local diagnostics package.

Every failure path terminates the pod and returns a complete local
diagnostic package; no path asks the user to SSH, read logs, choose a
datacenter/backend, monitor spend, press Terminate, or repair anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .adapter import RunpodV2Adapter
from .authorization import (
    AuthorizationError, CLI_FLAG, LiveMutationAuthorization,
)
from .identityutil import utcnow_iso
from .lifecycle import LifecycleError, PodLifecycleController
from .pod_request import build_pod_request, render_canonical_pod_request
from .preflight import run_preflight
from .redaction import redact


def load_session_config(root: str) -> dict:
    """Deployment facts resolved before launch (image digest, identities)."""
    path = os.path.join(root, "o1_b200", "provider", "runpod",
                        "RUNPOD_SESSION_CONFIG.json")
    if not os.path.exists(path):
        raise AuthorizationError(
            "RUNPOD_SESSION_CONFIG.json missing: the authorized session "
            "requires the resolved image digest and identity hashes")
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    unresolved = [k for k, v in config.items()
                  if isinstance(v, str) and v.startswith("UNRESOLVED")]
    if unresolved:
        raise AuthorizationError(
            f"session config carries unresolved template fields {unresolved}")
    return config


def run_session(*, authorization_path: str, out_dir: str,
                cli_args: list[str], base_url: str = "https://api.runpod.io",
                root: str | None = None, sleep=time.sleep,
                config: dict | None = None,
                spawn_watchdog_fn=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    root = root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    status: dict = {"schema": "o1b200.runpod_session_status.v1",
                    "utc": utcnow_iso(), "outcome": None, "steps": []}

    def step(name, detail=None):
        status["steps"].append({"step": name, "utc": utcnow_iso(),
                                "detail": redact(str(detail)) if detail else None})

    def finish(outcome, **extra):
        status["outcome"] = outcome
        status.update(extra)
        path = os.path.join(out_dir, "RUNPOD_SESSION_STATUS.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return status

    config = config or load_session_config(root)
    # 1. interlock — refuse before ANY mutating call
    expected_identity = {
        "project": config["project"],
        "package_zip_sha256": config["package_zip_sha256"],
        "provider": "runpod",
        "budget_policy_sha256": config["budget_policy_sha256"],
        "deployment_spec_sha256": None,  # bound after render below
    }
    # 2. read-only preflight first (no authorization needed)
    pre = run_preflight(base_url=base_url)
    step("READONLY_PREFLIGHT", pre["verdict"])
    if pre["verdict"] != "PASS":
        return finish("REFUSED_PREFLIGHT", preflight=pre)

    # 3. fresh quote + canonical request + hash binding
    probe = RunpodV2Adapter(base_url=base_url, sleep=sleep)
    quote = probe.quote_instance(adapter_commit=config.get("adapter_commit",
                                                           "UNKNOWN"))
    step("QUOTE", f"{quote['gpu_type_id']} @ {quote['hourly_gpu_rate_usd']}/h "
                  f"in {quote['datacenter_id']}")
    req = build_pod_request(image_digest_ref=config["image_digest_ref"],
                            datacenter_id=quote["datacenter_id"])
    rendered = render_canonical_pod_request(req, config["identities"])
    expected_identity["deployment_spec_sha256"] = rendered["request_sha256"]
    try:
        auth = LiveMutationAuthorization.verify(
            path=authorization_path, expected_identity=expected_identity,
            cli_args=cli_args,
            nonce_ledger=os.path.join(out_dir, "consumed_nonces.txt"))
    except AuthorizationError as exc:
        step("AUTHORIZATION_REFUSED", exc)
        return finish("LIVE_MUTATION_NOT_AUTHORIZED", error=str(exc))
    step("AUTHORIZED", "interlock satisfied")

    adapter = RunpodV2Adapter(base_url=base_url, authorization=auth,
                              sleep=sleep)
    adapter.accepted_quote = quote
    controller_kw = {}
    if spawn_watchdog_fn is not None:
        controller_kw["spawn_watchdog_fn"] = spawn_watchdog_fn
    controller = PodLifecycleController(adapter, out_dir, sleep=sleep,
                                        **controller_kw)
    pod_id = None
    try:
        # 4. provision (reconciled, duplicate-safe) + watchdogs
        pod_id = controller.provision(req, rendered)
        # 5. run
        controller.wait_until_running(pod_id)

        def pod_done(pod):
            # the on-pod state machine emits this marker after writing its
            # machine-readable FINAL_STATUS and packaging results
            try:
                return "ZERO_TOUCH_COMPLETE" in adapter.get_container_logs(
                    pod.id, tail=50)
            except Exception:  # noqa: BLE001 - log endpoint may lag
                return False

        outcome = controller.monitor(pod_id, until=pod_done)
        step("MONITOR_RESULT", outcome)
        # 6/7. results: collect logs + download outputs
        controller.collect_logs(pod_id)
        dest = os.path.join(out_dir, "downloaded_results")
        adapter.download_results(config.get("result_source", dest), dest)
        step("RESULTS_DOWNLOADED", dest)
    except (LifecycleError, Exception) as exc:  # noqa: BLE001
        step("SESSION_FAILURE", exc)
        if pod_id is not None:
            confirmed = controller.terminate_and_confirm(pod_id)
            return finish("ABORTED_TERMINATED" if confirmed
                          else "ABORTED_TERMINATION_UNCONFIRMED",
                          error=redact(str(exc)))
        return finish("ABORTED_BEFORE_CREATE", error=redact(str(exc)))
    # 8. terminate + confirm (always; stop is never final)
    confirmed = controller.terminate_and_confirm(pod_id)
    return finish("COMPLETE" if confirmed else "TERMINATION_UNCONFIRMED",
                  termination_confirmed=confirmed)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--authorization", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--execute-authorized-rental", action="store_true")
    p.add_argument("--base-url", default="https://api.runpod.io")
    a = p.parse_args()
    cli = [CLI_FLAG] if a.execute_authorized_rental else []
    status = run_session(authorization_path=a.authorization, out_dir=a.out,
                         cli_args=cli, base_url=a.base_url)
    print(json.dumps({"outcome": status["outcome"]}, indent=2))
    return 0 if status["outcome"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
