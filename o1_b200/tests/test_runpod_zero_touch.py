"""Zero-touch RunPod session dress rehearsal against the mock server.

Covers the session-level halves of mandatory tests 17-24: any on-pod failure
(environment mismatch, backend-equivalence failure, benchmark failure,
affordability failure, git precommit push/verify failure, calibration crash)
manifests to the provider layer as a container EXITED/ERROR without the
completion marker — the session must terminate the pod and return a complete
local diagnostic package.  The state-level injections themselves are already
proven by test_state_machine.py (21-point injection matrix).
"""
from __future__ import annotations

import json
import os
import time

from _h import Runner, fresh_dir

os.environ.setdefault("RUNPOD_API_KEY", "rpa_MOCKKEY_1234567890abcdef")

from o1_b200.provider.runpod.authorization import (
    AUTH_SCHEMA, CLI_FLAG, ENV_FLAG, ENV_FLAG_VALUE,
)
from o1_b200.provider.runpod.mock_server import MockRunpodServer, Scenario, _b200
from o1_b200.provider.runpod.pod_request import (
    build_pod_request, render_canonical_pod_request,
)
from o1_b200.provider.runpod.zero_touch import run_session

NOSLEEP = lambda s: None  # noqa: E731
GOOD_IMAGE = "ghcr.io/x/o1@sha256:" + "a" * 64


class FakeWatchdog:
    def __init__(self):
        self.fired = False

    def terminate_now(self):
        self.fired = True

    def alive(self):
        return True


def _fake_spawn(**_kw):
    return FakeWatchdog()


def _session_setup(d, srv):
    config = {
        "project": "O1_B200", "package_zip_sha256": "p" * 64,
        "budget_policy_sha256": "b" * 64,
        "image_digest_ref": GOOD_IMAGE,
        "identities": {"package": "test"},
        "adapter_commit": "test",
        "result_source": os.path.join(d, "fake_results.tar.gz"),
    }
    with open(config["result_source"], "wb") as fh:
        fh.write(b"results")
    # the canonical request the driver will render (US-KS-2 sorts first among
    # non-NONE datacenters in the default scenario? EU-RO-1 sorts first)
    req = build_pod_request(image_digest_ref=GOOD_IMAGE,
                            datacenter_id=sorted(["US-KS-2", "EU-RO-1"])[0])
    rendered = render_canonical_pod_request(req, config["identities"])
    doc = {
        "schema": AUTH_SCHEMA, "project": "O1_B200",
        "package_zip_sha256": "p" * 64, "provider": "runpod",
        "budget_policy_sha256": "b" * 64,
        "expires_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(time.time() + 3600)),
        "launch_nonce": f"nonce-{time.time_ns()}",
        "deployment_spec_sha256": rendered["request_sha256"],
        "allow_create_pod": True, "allow_real_calibration": True,
        "allow_confirmation": False,
    }
    auth_path = os.path.join(d, "auth.json")
    with open(auth_path, "w") as fh:
        json.dump(doc, fh)
    return config, auth_path


def run() -> Runner:
    r = Runner("runpod_zero_touch")

    def refusal_without_interlock():
        d = fresh_dir("zt_refuse")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        with MockRunpodServer(sc) as srv:
            config, auth_path = _session_setup(d, srv)
            os.environ.pop(ENV_FLAG, None)
            status = run_session(
                authorization_path=auth_path, out_dir=d,
                cli_args=[CLI_FLAG], base_url=srv.base_url,
                sleep=NOSLEEP, config=config,
                spawn_watchdog_fn=_fake_spawn)
            assert status["outcome"] == "LIVE_MUTATION_NOT_AUTHORIZED"
            assert not sc.pods, "a pod was created without authorization!"
            posts = [p for m, p in sc.requests if m != "GET"]
            assert not posts, f"mutating requests sent: {posts}"
    r.check("session without the env flag refuses BEFORE any mutating "
            "request (no pod, zero non-GET traffic)", refusal_without_interlock)

    def full_rehearsal_complete():
        d = fresh_dir("zt_ok")
        sc = Scenario()
        sc.log_text = "state machine done\nZERO_TOUCH_COMPLETE\n"
        with MockRunpodServer(sc) as srv:
            config, auth_path = _session_setup(d, srv)
            os.environ[ENV_FLAG] = ENV_FLAG_VALUE
            try:
                status = run_session(
                    authorization_path=auth_path, out_dir=d,
                    cli_args=[CLI_FLAG], base_url=srv.base_url,
                    sleep=NOSLEEP, config=config,
                    spawn_watchdog_fn=_fake_spawn)
            finally:
                os.environ.pop(ENV_FLAG, None)
            assert status["outcome"] == "COMPLETE", status
            assert status["termination_confirmed"] is True
            pod = list(sc.pods.values())[0]
            assert pod["terminated"], "pod left alive after COMPLETE"
            assert os.path.exists(os.path.join(d, "RUNPOD_SESSION_STATUS.json"))
            assert os.path.exists(os.path.join(d, "LIFECYCLE_REPORT.json"))
            assert os.path.exists(os.path.join(d, "downloaded_results",
                                               "fake_results.tar.gz"))
    r.check("full mocked zero-touch session: quote -> create -> RUNNING -> "
            "completion marker -> results downloaded -> TERMINATED confirmed",
            full_rehearsal_complete)

    def preflight_refusal_no_pod():
        d = fresh_dir("zt_nopre")
        sc = Scenario()
        sc.gpus = [_b200(availability="NONE",
                         dcs=[{"id": "US-KS-2", "availability": "NONE"}])]
        with MockRunpodServer(sc) as srv:
            config, auth_path = _session_setup(d, srv)
            os.environ[ENV_FLAG] = ENV_FLAG_VALUE
            try:
                status = run_session(
                    authorization_path=auth_path, out_dir=d,
                    cli_args=[CLI_FLAG], base_url=srv.base_url,
                    sleep=NOSLEEP, config=config,
                    spawn_watchdog_fn=_fake_spawn)
            finally:
                os.environ.pop(ENV_FLAG, None)
            assert status["outcome"] == "REFUSED_PREFLIGHT"
            assert not sc.pods
    r.check("failed preflight (B200 unavailable) refuses before create",
            preflight_refusal_no_pod)

    def on_pod_failure_terminates():
        # stands in for 17-23: env mismatch / equivalence / benchmark /
        # affordability / git push / git verify / calibration crash — the
        # on-pod machine dies, container ERRORs, no completion marker
        d = fresh_dir("zt_fail")
        sc = Scenario()
        sc.lifecycle_plan = ["PROVISIONING", "STARTING", "RUNNING", "ERROR"]
        with MockRunpodServer(sc) as srv:
            config, auth_path = _session_setup(d, srv)
            os.environ[ENV_FLAG] = ENV_FLAG_VALUE
            try:
                status = run_session(
                    authorization_path=auth_path, out_dir=d,
                    cli_args=[CLI_FLAG], base_url=srv.base_url,
                    sleep=NOSLEEP, config=config,
                    spawn_watchdog_fn=_fake_spawn)
            finally:
                os.environ.pop(ENV_FLAG, None)
            assert status["outcome"] == "ABORTED_TERMINATED", status
            pod = list(sc.pods.values())[0]
            assert pod["terminated"]
            assert os.path.exists(os.path.join(d, "pod_container_logs.txt")), \
                "diagnostics not collected"
    r.check("17-23. any on-pod stage failure (ERROR before completion) -> "
            "logs collected, pod terminated, diagnostic package local",
            on_pod_failure_terminates)

    def result_transfer_failure_terminates():
        d = fresh_dir("zt_xferfail")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        with MockRunpodServer(sc) as srv:
            config, auth_path = _session_setup(d, srv)
            config["result_source"] = os.path.join(d, "missing.tar.gz")
            os.environ[ENV_FLAG] = ENV_FLAG_VALUE
            try:
                status = run_session(
                    authorization_path=auth_path, out_dir=d,
                    cli_args=[CLI_FLAG], base_url=srv.base_url,
                    sleep=NOSLEEP, config=config,
                    spawn_watchdog_fn=_fake_spawn)
            finally:
                os.environ.pop(ENV_FLAG, None)
            assert status["outcome"] == "ABORTED_TERMINATED", status
            assert list(sc.pods.values())[0]["terminated"]
    r.check("24. output-transfer failure -> pod still terminated, loud abort",
            result_transfer_failure_terminates)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
