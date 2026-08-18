"""Hostile preemption suite: eviction of INTERRUPTIBLE capacity at every
session phase must be recovered (or refused) mechanically — no operator.

Covers: eviction during provisioning/startup, during the running workload,
after the completion marker, repeated eviction to the acquisition ceiling,
authorization-slot exhaustion, budget-bounded reacquisition, zero-progress
repeat-failure abort, B300->B200 fallback ON reacquisition, and durable
row/checkpoint state surviving pod loss (restore + resume-at-boundary).
"""
from __future__ import annotations

import json
import os
import time

from _h import Runner, fresh_dir, hermetic_mock_credentials

MOCK_KEY = hermetic_mock_credentials()

from o1_b200.provider.runpod.authorization import (
    AUTH_SCHEMA, CLI_FLAG, ENV_FLAG, ENV_FLAG_VALUE,
)
from o1_b200.provider.runpod.mock_server import (
    MockRunpodServer, Scenario, _b200, _b300,
)
from o1_b200.provider.runpod.zero_touch import _render_all_profiles, run_session

NOSLEEP = lambda s: None  # noqa: E731
GOOD_IMAGE = "ghcr.io/x/o1@sha256:" + "a" * 64


class FakeWatchdog:
    def terminate_now(self):
        pass

    def alive(self):
        return True


def _fake_spawn(**_kw):
    return FakeWatchdog()


def _setup(d, *, max_pod_creations=4):
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
    rendered = _render_all_profiles(config)
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
    auth_path = os.path.join(d, "auth.json")
    with open(auth_path, "w") as fh:
        json.dump(doc, fh)
    return config, auth_path


def _run(d, sc, config, auth_path, **kw):
    os.environ[ENV_FLAG] = ENV_FLAG_VALUE
    try:
        with MockRunpodServer(sc) as srv:
            return run_session(
                authorization_path=auth_path, out_dir=d,
                cli_args=[CLI_FLAG], base_url=srv.base_url,
                api_key=MOCK_KEY, sleep=NOSLEEP, config=config,
                spawn_watchdog_fn=_fake_spawn, **kw), sc
    finally:
        os.environ.pop(ENV_FLAG, None)


class EvictOnceThenComplete:
    """Scenario hook: evict the first pod once RUNNING; later pods finish."""

    def __init__(self, sc, evictions=1, after_polls=2):
        self.sc = sc
        self.remaining = evictions
        self.after_polls = after_polls
        sc.evict_after_polls = after_polls

    def maybe_disarm(self):
        # called per created pod via rent_calls length check
        if len(self.sc.rent_calls) > self.remaining:
            self.sc.evict_after_polls = None


def run() -> Runner:
    r = Runner("preemption")

    def evicted_while_running_then_recovers():
        d = fresh_dir("pre_run")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        sc.evict_after_polls = 2

        # disarm eviction after the first pod so the second one completes:
        orig_rent = Scenario.spot_for  # noqa: F841 (documentation only)
        real_evict = sc.evict

        def evict_once(pod_id):
            real_evict(pod_id)
            sc.evict_after_polls = None
        sc.evict = evict_once

        config, auth_path = _setup(d)
        status, sc = _run(d, sc, config, auth_path)
        assert status["outcome"] == "COMPLETE", status["outcome"]
        assert status["acquisitions_used"] == 2, status
        assert len(sc.rent_calls) == 2
        assert all(p["terminated"] for p in sc.pods.values()), \
            "an evicted remnant or the final pod was left alive"
        assert len(status["acquisitions"]) == 2
        session = json.load(open(os.path.join(
            d, "RUNPOD_SESSION_STATUS.json")))
        assert session["outcome"] == "COMPLETE"
    r.check("eviction while RUNNING -> remnant terminated, fresh spot pod "
            "reacquired, session COMPLETE (2 acquisitions)",
            evicted_while_running_then_recovers)

    def evicted_during_startup_then_recovers():
        d = fresh_dir("pre_start")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        # first pod dies before RUNNING (spot reclaimed at provisioning)
        sc.lifecycle_plan = ["PROVISIONING", "STARTING", "EXITED"]

        created = {"n": 0}
        orig = sc.next_pod_status

        def status_fn(pod):
            if pod["id"] == list(sc.pods)[0]:
                return orig(pod)
            pod2_plan = ["PROVISIONING", "STARTING", "RUNNING"]
            idx = min(pod["poll_count"], len(pod2_plan) - 1)
            return "TERMINATED" if pod["terminated"] else pod2_plan[idx]
        sc.next_pod_status = status_fn

        config, auth_path = _setup(d)
        status, sc = _run(d, sc, config, auth_path)
        assert status["outcome"] == "COMPLETE", status["outcome"]
        assert status["acquisitions_used"] == 2
    r.check("eviction during provisioning/startup -> clean reacquisition, "
            "session COMPLETE", evicted_during_startup_then_recovers)

    def eviction_after_completion_marker_is_complete():
        d = fresh_dir("pre_done")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        # pod reaches RUNNING then EXITS (spot stop) WITH the marker present:
        # the work finished; the exit is not a loss
        sc.lifecycle_plan = ["PROVISIONING", "STARTING", "RUNNING", "EXITED"]
        config, auth_path = _setup(d)
        status, sc = _run(d, sc, config, auth_path)
        assert status["outcome"] == "COMPLETE", status["outcome"]
        assert status["acquisitions_used"] == 1
    r.check("EXITED with the completion marker is COMPLETE, not an eviction",
            eviction_after_completion_marker_is_complete)

    def acquisition_ceiling_bounds_thrash():
        d = fresh_dir("pre_thrash")
        sc = Scenario()
        sc.evict_after_polls = 1     # every pod gets evicted forever
        config, auth_path = _setup(d, max_pod_creations=8)
        status, sc = _run(d, sc, config, auth_path)
        assert status["outcome"] == "ABORTED_ACQUISITION_LIMIT", \
            status["outcome"]
        from o1_b200.provider.runpod.policy import MAX_POD_ACQUISITIONS
        assert len(sc.rent_calls) == MAX_POD_ACQUISITIONS
        assert all(p["terminated"] for p in sc.pods.values())
    r.check("perpetual eviction is bounded by MAX_POD_ACQUISITIONS; every "
            "remnant terminated; loud abort", acquisition_ceiling_bounds_thrash)

    def authorization_slots_bound_reacquisition():
        d = fresh_dir("pre_slots")
        sc = Scenario()
        sc.evict_after_polls = 1
        config, auth_path = _setup(d, max_pod_creations=2)
        status, sc = _run(d, sc, config, auth_path)
        assert status["outcome"] == "ABORTED_AUTHORIZATION_EXHAUSTED", \
            status["outcome"]
        assert len(sc.rent_calls) == 2, "creation beyond authorized slots"
    r.check("max_pod_creations exhaustion stops reacquisition with "
            "LIVE_MUTATION_NOT_AUTHORIZED semantics",
            authorization_slots_bound_reacquisition)

    def budget_bounds_reacquisition():
        """The budget — NOT the acquisition ceiling — must stop this run.

        A disjunctive assertion here previously passed via
        ABORTED_ACQUISITION_LIMIT while session spend stayed 0.00, hiding
        the fact that pre-RUNNING billing was uncounted.  The outcome is
        now pinned exactly.
        """
        d = fresh_dir("pre_budget")
        sc = Scenario()
        sc.evict_after_polls = 2
        clock = {"t": 0.0}

        def spend_clock():
            clock["t"] += 10800.0    # every spend sample advances 3 h
            return clock["t"]
        config, auth_path = _setup(d, max_pod_creations=8)
        status, sc = _run(d, sc, config, auth_path, spend_clock=spend_clock)
        assert status["outcome"] == "ABORTED_BUDGET", status["outcome"]
        assert all(p["terminated"] for p in sc.pods.values()), \
            "a pod was left alive when the budget stopped the session"
        from o1_b200.provider.runpod.policy import MAX_POD_ACQUISITIONS
        assert len(sc.rent_calls) < MAX_POD_ACQUISITIONS, (
            "the budget must bite BEFORE the acquisition ceiling in this "
            "scenario, otherwise it is not what stopped the run")
    r.check("the hard dollar budget — not the slot ceiling — stops a "
            "reacquisition sequence", budget_bounds_reacquisition)

    def evicted_before_running_pods_are_not_free():
        """Pods evicted before RUNNING still bill from provisioning."""
        d = fresh_dir("pre_prerun_billing")
        sc = Scenario()
        sc.lifecycle_plan = ["PROVISIONING", "STARTING", "EXITED"]
        clock = {"t": 0.0}

        def spend_clock():
            clock["t"] += 600.0
            return clock["t"]
        config, auth_path = _setup(d, max_pod_creations=8)
        status, sc = _run(d, sc, config, auth_path, spend_clock=spend_clock)
        assert status["outcome"] in ("ABORTED_BUDGET",
                                     "ABORTED_REPEATED_FAILURE_NO_PROGRESS",
                                     "ABORTED_ACQUISITION_LIMIT")
        report = json.load(open(os.path.join(d, "LIFECYCLE_REPORT.json")))
        frozen = [e for e in report["events"] if e["event"] == "SPEND_FROZEN"]
        assert frozen, "no spend was ever frozen for evicted pods"
        assert any(float(e["session_spend_usd"]) > 0 for e in frozen), (
            "pods evicted before RUNNING were recorded as free; RunPod "
            "bills from provisioning")
    r.check("a pod evicted before RUNNING is still charged (billing starts "
            "at provisioning, not at RUNNING)",
            evicted_before_running_pods_are_not_free)

    def watchdog_deadline_shrinks_with_session_spend():
        """Each pod's independent watchdog gets the REMAINING allocation."""
        d = fresh_dir("pre_wdog_budget")
        sc = Scenario()
        sc.evict_after_polls = 2
        clock = {"t": 0.0}

        def spend_clock():
            clock["t"] += 300.0
            return clock["t"]
        armed = []

        def spawn(**kw):
            armed.append(kw["hard_limit_seconds"])
            return FakeWatchdog()
        config, auth_path = _setup(d, max_pod_creations=3)
        os.environ[ENV_FLAG] = ENV_FLAG_VALUE
        try:
            with MockRunpodServer(sc) as srv:
                run_session(authorization_path=auth_path, out_dir=d,
                            cli_args=[CLI_FLAG], base_url=srv.base_url,
                            api_key=MOCK_KEY, sleep=NOSLEEP, config=config,
                            spawn_watchdog_fn=spawn, spend_clock=spend_clock)
        finally:
            os.environ.pop(ENV_FLAG, None)
        assert len(armed) >= 2, armed
        assert armed == sorted(armed, reverse=True) and armed[1] < armed[0], (
            f"later pods were armed at the same or a larger horizon: {armed}")
        # Only one pod is alive at a time (each remnant is terminated before
        # the next is created), so the exposure that matters is per pod:
        # spend-at-arm + horizon must stay inside the allocation.  Since the
        # horizon IS (allocation - spend)/rate, every individual horizon must
        # therefore be strictly less than the full allocation once anything
        # has been spent — which is exactly what the pre-repair code got
        # wrong by arming every pod at hard_compute_seconds().
        from o1_b200.provider.runpod.policy import MAX_COMPUTE_USD
        rate = 7.89
        full = float(MAX_COMPUTE_USD) / rate * 3600.0
        for i, horizon in enumerate(armed):
            assert horizon <= full + 1, (i, horizon, full)
            if i > 0:
                assert horizon < full, (
                    f"pod {i} was armed at the FULL allocation despite "
                    f"earlier spend: {horizon}s")
    r.check("each reacquired pod's watchdog is armed at the REMAINING "
            "allocation, so summed horizons stay inside the budget",
            watchdog_deadline_shrinks_with_session_spend)

    def transient_log_failure_is_not_an_eviction():
        """A 500ing log endpoint at exit must not cost a reacquisition."""
        d = fresh_dir("pre_logfail")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        sc.lifecycle_plan = ["PROVISIONING", "STARTING", "RUNNING"]
        sc.fail_next = [(500, 2)]     # the completion probe's first tries
        config, auth_path = _setup(d)
        status, sc = _run(d, sc, config, auth_path)
        assert status["outcome"] == "COMPLETE", status["outcome"]
        assert len(sc.rent_calls) == 1, (
            "a transient log-endpoint failure triggered a paid "
            "reacquisition")
    r.check("a transient completion-probe failure is retried, not read as "
            "an eviction", transient_log_failure_is_not_an_eviction)

    def soft_stop_and_external_termination_are_not_complete():
        """monitor() outcomes other than COMPLETE must never be COMPLETE."""
        from o1_b200.provider.runpod import lifecycle as lc
        for forced in ("SOFT_STOP", "TERMINATED"):
            d = fresh_dir(f"pre_{forced.lower()}")
            sc = Scenario()
            sc.log_text = "no marker here\n"
            config, auth_path = _setup(d)
            original = lc.PodLifecycleController.monitor
            lc.PodLifecycleController.monitor = (
                lambda self, pod_id, **kw: forced)
            try:
                status, sc = _run(d, sc, config, auth_path)
            finally:
                lc.PodLifecycleController.monitor = original
            assert status["outcome"] == f"ABORTED_{forced}", (
                f"{forced} was reported as {status['outcome']}")
            assert all(p["terminated"] for p in sc.pods.values())
    r.check("budget soft-stop and external termination are reported as "
            "ABORTED_*, never as COMPLETE",
            soft_stop_and_external_termination_are_not_complete)

    def zero_progress_repeat_aborts():
        d = fresh_dir("pre_noprog")
        sc = Scenario()
        sc.evict_after_polls = 1
        config, auth_path = _setup(d, max_pod_creations=8)
        status, sc = _run(d, sc, config, auth_path,
                          progress_probe=lambda: 0)
        assert status["outcome"] == "ABORTED_REPEATED_FAILURE_NO_PROGRESS", \
            status["outcome"]
        assert len(sc.rent_calls) == 2, \
            "should abort on the SECOND zero-progress interruption"
    r.check("a repeated interruption with zero durable progress aborts as a "
            "container defect instead of thrashing", zero_progress_repeat_aborts)

    def fallback_on_reacquisition():
        d = fresh_dir("pre_fall")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        sc.evict_after_polls = 2
        real_evict = sc.evict

        def evict_and_kill_b300(pod_id):
            real_evict(pod_id)
            sc.evict_after_polls = None
            # after the eviction, B300 capacity is gone entirely
            sc.gpus = [g for g in sc.gpus if "B300" not in g["id"]]
        sc.evict = evict_and_kill_b300

        config, auth_path = _setup(d)
        status, sc = _run(d, sc, config, auth_path)
        assert status["outcome"] == "COMPLETE", status["outcome"]
        acq = status["acquisitions"]
        assert acq[0]["profile"] == "B300" and acq[1]["profile"] == "B200"
        assert acq[1]["fallback_reason"], \
            "fallback reacquisition must record its reason"
        assert sc.rent_calls[0]["gpuTypeId"] == "NVIDIA B300 SXM6 AC"
        assert sc.rent_calls[1]["gpuTypeId"] == "NVIDIA B200"
    r.check("reacquisition re-runs profile selection: B300 evicted+gone -> "
            "explicit recorded B200 fallback -> COMPLETE",
            fallback_on_reacquisition)

    def spot_price_change_between_acquisitions():
        d = fresh_dir("pre_price")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        sc.evict_after_polls = 2
        real_evict = sc.evict

        def evict_and_reprice(pod_id):
            real_evict(pod_id)
            sc.evict_after_polls = None
            sc.spot["NVIDIA B300 SXM6 AC"] = {"secureSpotPrice": 7.25}
        sc.evict = evict_and_reprice

        config, auth_path = _setup(d)
        status, sc = _run(d, sc, config, auth_path)
        assert status["outcome"] == "COMPLETE"
        assert sc.rent_calls[0]["bidPerGpu"] == 7.89
        assert sc.rent_calls[1]["bidPerGpu"] == 7.25, \
            "reacquisition must bid the FRESH live spot price"
    r.check("reacquisition re-quotes: a changed interruptible price is "
            "re-derived, never cached", spot_price_change_between_acquisitions)

    def rows_survive_eviction_and_resume_at_boundary():
        from o1_b200.runner.durability import LocalDurableStore, RowDurability
        d = fresh_dir("pre_rows")
        store = LocalDurableStore(os.path.join(d, "durable"))
        # pod 1: commits rows 0..6, syncs every 3 -> 6 mirrored, 1 lost
        pod1 = os.path.join(d, "pod1_run")
        os.makedirs(os.path.join(pod1, "rows"))
        dur1 = RowDurability(pod1, store, sync_every_rows=3)
        for i in range(7):
            with open(os.path.join(pod1, "rows", f"row{i:03d}.json"),
                      "w") as fh:
                json.dump({"row_id": f"row{i:03d}", "payload": i}, fh)
            dur1.notify_commit()
        assert dur1.synced_row_count() == 6, dur1.synced_row_count()
        # EVICTION: pod1's disk is gone.  pod 2 restores durable state:
        pod2 = os.path.join(d, "pod2_run")
        os.makedirs(pod2)
        dur2 = RowDurability(pod2, store, sync_every_rows=3)
        rep = dur2.restore()
        assert rep["restored"] == 6 and rep["corrupt_quarantined"] == 0
        names = sorted(os.listdir(os.path.join(pod2, "rows")))
        assert names == [f"row{i:03d}.json" for i in range(6)], names
        # row006 was committed-but-unsynced: it is regenerated, not
        # reconstructed — the next missing canonical row is row006
        missing = [f"row{i:03d}" for i in range(7)
                   if f"row{i:03d}.json" not in names]
        assert missing == ["row006"]
        # a corrupt mirrored row is quarantined, never resumed from
        with open(os.path.join(d, "durable", "durable_rows",
                               "row999.json"), "w") as fh:
            fh.write("{not json")
        pod3 = os.path.join(d, "pod3_run")
        os.makedirs(pod3)
        rep3 = RowDurability(pod3, store).restore()
        assert rep3["corrupt_quarantined"] == 1
        assert os.path.exists(os.path.join(
            pod3, "quarantine", "row999.json.restore_corrupt"))
        assert not os.path.exists(os.path.join(pod3, "rows", "row999.json"))
    r.check("O1 rows: committed+synced rows survive eviction; unsynced work "
            "is regenerated from the next missing row; corrupt mirrors are "
            "quarantined", rows_survive_eviction_and_resume_at_boundary)

    def local_rows_win_over_mirror():
        from o1_b200.runner.durability import LocalDurableStore, RowDurability
        d = fresh_dir("pre_rows_win")
        store = LocalDurableStore(os.path.join(d, "durable"))
        run_dir = os.path.join(d, "run")
        os.makedirs(os.path.join(run_dir, "rows"))
        dur = RowDurability(run_dir, store, sync_every_rows=1)
        with open(os.path.join(run_dir, "rows", "rowA.json"), "w") as fh:
            json.dump({"row_id": "rowA", "v": "local"}, fh)
        dur.notify_commit()
        # tamper the MIRROR; the local atomic commit must win on restore
        with open(os.path.join(d, "durable", "durable_rows",
                               "rowA.json"), "w") as fh:
            json.dump({"row_id": "rowA", "v": "tampered"}, fh)
        rep = RowDurability(run_dir, store).restore()
        assert rep["skipped_existing"] == 1
        assert json.load(open(os.path.join(
            run_dir, "rows", "rowA.json")))["v"] == "local"
    r.check("restore never overwrites a locally committed row",
            local_rows_win_over_mirror)

    def checkpoint_roundtrip_and_corruption():
        from o1_b200.runner.durability import (
            CheckpointDurability, DurabilityError, LocalDurableStore,
        )
        d = fresh_dir("pre_ckpt")
        store = LocalDurableStore(os.path.join(d, "durable"))
        ck = CheckpointDurability(store)
        arch = os.path.join(d, "ckpt_step120.tar")
        with open(arch, "wb") as fh:
            fh.write(os.urandom(4096))
        man = ck.sync_checkpoint(arch, {"stage": "FL2", "step": 120,
                                        "rng": "state-hash"})
        got = ck.restore_latest(os.path.join(d, "restore"))
        assert got is not None and got["step"] == 120
        assert got["sha256"] == man["sha256"]
        # corrupt the durable archive: restore must refuse, never resume
        with open(os.path.join(d, "durable", "durable_checkpoint",
                               "ckpt_step120.tar"), "wb") as fh:
            fh.write(b"corrupted")
        try:
            ck.restore_latest(os.path.join(d, "restore2"))
        except DurabilityError as exc:
            assert "mismatch" in str(exc)
        else:
            raise AssertionError("corrupt durable checkpoint accepted")
        # empty store -> None (fresh start from the pristine bound model)
        ck2 = CheckpointDurability(LocalDurableStore(os.path.join(d, "d2")))
        assert ck2.restore_latest(os.path.join(d, "restore3")) is None
    r.check("FL checkpoint: atomic durable round-trip with manifest-last "
            "ordering; corrupt archive refused; empty store = pristine start",
            checkpoint_roundtrip_and_corruption)

    def terminate_after_armed_on_every_rent():
        d = fresh_dir("pre_termafter")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        config, auth_path = _setup(d)
        status, sc = _run(d, sc, config, auth_path)
        assert status["outcome"] == "COMPLETE"
        for call in sc.rent_calls:
            assert call.get("terminateAfter"), \
                "provider-side auto-terminate not armed"
            assert call.get("minCudaVersion") == "13.0"
            assert call.get("startSsh") is False
            assert call.get("startJupyter") is False
    r.check("every spot rent arms provider-side terminateAfter and pins "
            "minCudaVersion 13.0 with no SSH/Jupyter",
            terminate_after_armed_on_every_rent)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
