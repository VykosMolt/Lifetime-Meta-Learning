"""Pod lifecycle: create/reconcile/monitor/terminate against the mock server,
plus artifact-store transfer semantics."""
from __future__ import annotations

import json
import os
import time

from _h import Runner, fresh_dir, hermetic_mock_credentials

MOCK_KEY = hermetic_mock_credentials()

from o1_b200.provider.runpod.adapter import RunpodAdapterError, RunpodV2Adapter
from o1_b200.provider.runpod.artifact_store import (
    LocalArtifactStore, MockArtifactStore, TransferFailure,
)
from o1_b200.provider.runpod.authorization import (
    AUTH_SCHEMA, CLI_FLAG, ENV_FLAG, ENV_FLAG_VALUE, LiveMutationAuthorization,
)
from o1_b200.provider.runpod.lifecycle import LifecycleError, PodLifecycleController
from o1_b200.provider.runpod.mock_server import MockRunpodServer, Scenario
from o1_b200.provider.runpod.pod_request import (
    build_pod_request, render_canonical_deployment,
)
from o1_b200.provider.runpod.policy import PROFILE_B200, PROFILE_B300
from o1_b200.provider.runpod.transport import ApiHttpError

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


def _authorized_adapter(srv, d, *, clock=None, max_pod_creations=4):
    reqs = {p.key: build_pod_request(profile=p, image_digest_ref=GOOD_IMAGE,
                                     datacenter_id="US-KS-2")
            for p in (PROFILE_B300, PROFILE_B200)}
    rendered = render_canonical_deployment(reqs, {"package": "test"})
    doc = {
        "schema": AUTH_SCHEMA, "project": "O1_B200",
        "package_zip_sha256": "p" * 64, "provider": "runpod",
        "budget_policy_sha256": "b" * 64,
        "expires_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(time.time() + 3600)),
        "launch_nonce": f"nonce-{time.time_ns()}",
        "deployment_spec_sha256": rendered["request_sha256"],
        "allow_create_pod": True, "allow_real_calibration": True,
        "allow_confirmation": False, "max_pod_creations": max_pod_creations,
    }
    path = os.path.join(d, "auth.json")
    with open(path, "w") as fh:
        json.dump(doc, fh)
    os.environ[ENV_FLAG] = ENV_FLAG_VALUE
    auth = LiveMutationAuthorization.verify(
        path=path, expected_identity={
            k: doc[k] for k in ("project", "package_zip_sha256", "provider",
                                "budget_policy_sha256",
                                "deployment_spec_sha256")},
        cli_args=[CLI_FLAG], nonce_ledger=os.path.join(d, "nonces.txt"))
    kw = {"clock": clock} if clock else {}
    ad = RunpodV2Adapter(base_url=srv.base_url, authorization=auth,
                         sleep=NOSLEEP, api_key=MOCK_KEY, **kw)
    ad.quote_instance(adapter_commit="test")
    return ad, reqs["B300"], rendered


def run() -> Runner:
    r = Runner("runpod_lifecycle")

    def full_lifecycle():
        d = fresh_dir("lc_full")
        with MockRunpodServer() as srv:
            ad, req, rendered = _authorized_adapter(srv, d)
            ctl = PodLifecycleController(ad, d, sleep=NOSLEEP,
                                         spawn_watchdog_fn=_fake_spawn)
            pod_id = ctl.provision(req, rendered)
            assert json.load(open(os.path.join(d, "POD_ID.json")))["pod_id"] \
                == pod_id
            ctl.wait_until_running(pod_id)
            confirmed = ctl.terminate_and_confirm(pod_id)
            assert confirmed
            report = json.load(open(os.path.join(d, "LIFECYCLE_REPORT.json")))
            assert report["termination_confirmed"] is True
            assert report["machine_readable"] is True
    r.check("provision -> RUNNING -> terminate -> confirmed, with atomic "
            "POD_ID record and machine-readable lifecycle report",
            full_lifecycle)

    def duplicate_create_protection():
        d = fresh_dir("lc_dup")
        with MockRunpodServer() as srv:
            ad, req, rendered = _authorized_adapter(srv, d)
            ad.create_instance(req, rendered)
            d2 = fresh_dir("lc_dup2")
            ad2, req2, rendered2 = _authorized_adapter(srv, d2)
            try:
                ad2.create_instance(req2, rendered2)
            except RunpodAdapterError as exc:
                assert "already exists" in str(exc)
                return
        raise AssertionError("duplicate create allowed")
    r.check("11. duplicate Pod creation prevented (reconcile by canonical "
            "name before creating)", duplicate_create_protection)

    def create_timeout_reconciles():
        d = fresh_dir("lc_lost")
        sc = Scenario()
        sc.drop_rent_response = True
        with MockRunpodServer(sc) as srv:
            ad, req, rendered = _authorized_adapter(srv, d)
            pod = ad.create_instance(req, rendered)   # response lost; reconcile
            assert pod.id in sc.pods, "did not attach to the real remote pod"
            assert len(sc.pods) == 1, "a second create was submitted"
            assert len(sc.rent_calls) == 1, \
                f"blind spot-create retry: {len(sc.rent_calls)}"
    r.check("12. spot-create response lost -> reconcile by nonce+name "
            "attaches to the unique remote pod; NO second create request",
            create_timeout_reconciles)

    def ambiguous_reconciliation_halts():
        d = fresh_dir("lc_ambig")
        sc = Scenario()
        with MockRunpodServer(sc) as srv:
            for i in range(2):
                sc.pods[f"pre{i}"] = {
                    "id": f"pre{i}", "name": "o1-b300-calibration",
                    "image": GOOD_IMAGE, "env": {}, "cloud": "SECURE",
                    "gpu": {"id": "NVIDIA B200", "count": 1},
                    "status": "RUNNING", "poll_count": 0,
                    "plan": ["RUNNING"], "terminated": False,
                    "terminate_polls_left": 0,
                    "createdAt": "2026-08-01T00:00:00Z"}
            ad, req, rendered = _authorized_adapter(srv, d)
            try:
                ad.create_instance(req, rendered)
            except RunpodAdapterError as exc:
                assert "ambiguous" in str(exc)
                assert all(p["terminated"] for p in sc.pods.values()), \
                    "ambiguous pods were not terminated"
                return
        raise AssertionError("ambiguous reconciliation did not halt")
    r.check("13. ambiguous reconciliation terminates all matches and halts",
            ambiguous_reconciliation_halts)

    def startup_timeout():
        d = fresh_dir("lc_stimeout")
        sc = Scenario()
        sc.lifecycle_plan = ["PROVISIONING", "STARTING", "STARTING", "STARTING"]
        with MockRunpodServer(sc) as srv:
            clock = {"t": 0.0}
            ad, req, rendered = _authorized_adapter(
                srv, d, clock=lambda: clock["t"])
            ctl = PodLifecycleController(ad, d, startup_timeout_seconds=10,
                                         sleep=lambda s: clock.__setitem__(
                                             "t", clock["t"] + 5),
                                         spawn_watchdog_fn=_fake_spawn)
            pod_id = ctl.provision(req, rendered)
            try:
                ctl.wait_until_running(pod_id)
            except LifecycleError:
                assert sc.pods[pod_id]["terminated"], \
                    "startup-timeout pod left running"
                return
        raise AssertionError("startup timeout not enforced")
    r.check("14. startup timeout terminates the pod and fails loudly",
            startup_timeout)

    def container_exits_before_healthy():
        d = fresh_dir("lc_exit")
        sc = Scenario()
        sc.lifecycle_plan = ["PROVISIONING", "STARTING", "ERROR"]
        with MockRunpodServer(sc) as srv:
            ad, req, rendered = _authorized_adapter(srv, d)
            ctl = PodLifecycleController(ad, d, sleep=NOSLEEP,
                                         spawn_watchdog_fn=_fake_spawn)
            pod_id = ctl.provision(req, rendered)
            try:
                ctl.wait_until_running(pod_id)
            except LifecycleError:
                assert os.path.exists(
                    os.path.join(d, "pod_container_logs.txt")), \
                    "logs were not collected on container failure"
                assert sc.pods[pod_id]["terminated"]
                return
        raise AssertionError("ERROR state not handled")
    r.check("15. container failure -> logs collected automatically -> "
            "terminate", container_exits_before_healthy)

    def termination_matrix():
        for label, plan, terminate_polls in (
                ("while provisioning", ["PROVISIONING"], 0),
                ("while starting", ["PROVISIONING", "STARTING"], 0),
                ("while running", ["PROVISIONING", "STARTING", "RUNNING"], 0),
                ("after container failure", ["ERROR"], 0),
                ("delayed termination", ["RUNNING"], 3)):
            d = fresh_dir(f"lc_term_{label.replace(' ', '_')}")
            sc = Scenario()
            sc.lifecycle_plan = plan
            sc.terminate_requires_polls = terminate_polls
            with MockRunpodServer(sc) as srv:
                ad, req, rendered = _authorized_adapter(srv, d)
                pod = ad.create_instance(req, rendered)
                ad.terminate_instance(pod.id)
                ad.terminate_instance(pod.id)  # repeated terminate: idempotent
                assert ad.confirm_terminated(pod.id, sleep=NOSLEEP), label
    r.check("terminate while provisioning/starting/running/after-failure; "
            "repeated terminate; delayed termination all confirm",
            termination_matrix)

    def terminate_transient_errors():
        d = fresh_dir("lc_term429")
        sc = Scenario()
        with MockRunpodServer(sc) as srv:
            ad, req, rendered = _authorized_adapter(srv, d)
            pod = ad.create_instance(req, rendered)
            sc.rate_limit_next = 1     # 429 on the terminate action
            try:
                ad.terminate_instance(pod.id)
            except ApiHttpError:
                pass                    # action rejected; DELETE path continues
            assert ad.confirm_terminated(pod.id, sleep=NOSLEEP)
    r.check("transient 429/5xx during termination still reaches confirmed "
            "termination via the redundant DELETE path",
            terminate_transient_errors)

    def unknown_pod_and_permission():
        d = fresh_dir("lc_unknown")
        with MockRunpodServer() as srv:
            ad, req, rendered = _authorized_adapter(srv, d)
            ad.terminate_instance("nonexistent-pod")   # 404 tolerated
            assert ad.confirm_terminated("nonexistent-pod", sleep=NOSLEEP), \
                "404 must count as terminated"
        sc = Scenario()
        sc.api_key = "rpa_DIFFERENT_KEY_9999999999"
        with MockRunpodServer(sc) as srv:
            ad2 = RunpodV2Adapter(base_url=srv.base_url, sleep=NOSLEEP,
                                  api_key=MOCK_KEY)
            try:
                ad2.get_instance("any")
            except ApiHttpError as exc:
                assert exc.status == 401
                return
        raise AssertionError("permission failure not surfaced")
    r.check("unknown Pod ID counts as already-terminated; permission failure "
            "is loud", unknown_pod_and_permission)

    def primary_fail_watchdog_success_and_both_fail():
        d = fresh_dir("lc_wdog")
        sc = Scenario()
        with MockRunpodServer(sc) as srv:
            ad, req, rendered = _authorized_adapter(srv, d)
            ctl = PodLifecycleController(ad, d, sleep=NOSLEEP,
                                         spawn_watchdog_fn=_fake_spawn)
            pod_id = ctl.provision(req, rendered)
            sc.fail_next = [(500, 4)]   # primary termination breaks

            class RealTerminatingWatchdog(FakeWatchdog):
                def terminate_now(self):
                    self.fired = True
                    sc.pods[pod_id]["terminated"] = True
            ctl.watchdog = RealTerminatingWatchdog()
            confirmed = ctl.terminate_and_confirm(pod_id)
            assert ctl.watchdog.fired, "watchdog was not invoked"
            assert confirmed, "watchdog termination did not confirm"
        d2 = fresh_dir("lc_wdog2")
        sc2 = Scenario()
        sc2.lifecycle_plan = ["RUNNING"]
        with MockRunpodServer(sc2) as srv:
            ad, req, rendered = _authorized_adapter(srv, d2)
            ctl = PodLifecycleController(ad, d2, sleep=NOSLEEP,
                                         spawn_watchdog_fn=_fake_spawn)
            pod_id = ctl.provision(req, rendered)
            sc2.fail_next = [(500, 50)]
            ctl.watchdog = FakeWatchdog()   # fires but terminates nothing
            confirmed = ctl.terminate_and_confirm(pod_id)
            assert not confirmed
            report = json.load(open(os.path.join(d2, "LIFECYCLE_REPORT.json")))
            assert report["termination_confirmed"] is False
            kinds = [e["event"] for e in report["events"]]
            assert "TERMINATION_UNCONFIRMED_LOUD_FAILURE" in kinds
    r.check("25/26/27. primary termination failure -> watchdog succeeds; "
            "both-fail is a loud, recorded failure",
            primary_fail_watchdog_success_and_both_fail)

    def watchdog_process_crash_detected():
        import subprocess
        import sys
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(97)"])
        proc.wait()
        from o1_b200.provider.runpod.watchdog_terminate import WatchdogHandle
        h = WatchdogHandle(proc)
        assert not h.alive(), "crashed watchdog reported alive"
    r.check("watchdog process crash is detectable via the handle",
            watchdog_process_crash_detected)

    def artifact_store_semantics():
        d = fresh_dir("store")
        payload = os.urandom(1 << 20)
        import hashlib
        want = hashlib.sha256(payload).hexdigest()
        store = MockArtifactStore({"obj": payload}, fail_reads=2)
        out = store.fetch("obj", d, expected_sha256=want, sleep=NOSLEEP)
        assert out["sha256"] == want and out["retries"] == 2
        assert not any(n.endswith(".partial") for n in os.listdir(d))
        store2 = MockArtifactStore({"obj2": payload}, corrupt=True)
        try:
            store2.fetch("obj2", d, expected_sha256=want, sleep=NOSLEEP)
        except TransferFailure:
            assert not any(n.endswith(".partial") for n in os.listdir(d))
        else:
            raise AssertionError("corrupt transfer accepted")
        src = os.path.join(d, "src.bin")
        with open(src, "wb") as fh:
            fh.write(payload)
        out3 = LocalArtifactStore().fetch(src, os.path.join(d, "dst"),
                                          expected_sha256=want, sleep=NOSLEEP)
        assert out3["sha256"] == want
    r.check("16/24. artifact transfer: chunked retry, hash verification, "
            "atomic rename, partial cleanup; corrupt transfer refused",
            artifact_store_semantics)

    os.environ.pop(ENV_FLAG, None)
    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
