"""RunPod Pod lifecycle controller.

States come from the pinned schema (PROVISIONING, STARTING, RUNNING, EXITED,
ERROR, TERMINATED); anything else fails closed in models.PodModel.parse.

Responsibilities: reconcile-before-create (launch nonce + canonical name),
duplicate prevention, atomic Pod-ID record, asynchronous polling to RUNNING
or timeout, container-failure detection with automatic log collection,
budget-aware work gating (95% soft stop / 100% hard stop via SpendTracker),
terminate-on-any-hard-failure, redundant termination confirmation, final
billing query, machine-readable lifecycle report.
"""
from __future__ import annotations

import json
import os
import time

from .adapter import RunpodAdapterError, RunpodV2Adapter
from .billing import BudgetViolation, remaining_compute_seconds
from .identityutil import utcnow_iso
from .redaction import redact
from .watchdog_terminate import spawn_watchdog


class LifecycleError(RuntimeError):
    def __init__(self, msg: str):
        super().__init__(redact(msg))


class PodLifecycleController:
    def __init__(self, adapter: RunpodV2Adapter, out_dir: str,
                 *, startup_timeout_seconds: float = 900,
                 sleep=time.sleep, spawn_watchdog_fn=spawn_watchdog,
                 attempt: int = 1, monitor_timeout_seconds: float | None = None):
        self.adapter = adapter
        self.out_dir = out_dir
        self.startup_timeout = startup_timeout_seconds
        self.monitor_timeout = monitor_timeout_seconds
        self.sleep = sleep
        self.spawn_watchdog = spawn_watchdog_fn
        self.attempt = int(attempt)
        os.makedirs(out_dir, exist_ok=True)
        self.pod_id_path = os.path.join(out_dir, "POD_ID.json")
        self.report_path = os.path.join(out_dir, "LIFECYCLE_REPORT.json")
        # Per-acquisition evidence.  A session may create several pods; a
        # single shared report/id file keeps only the LAST one, silently
        # erasing (for example) a TERMINATION_UNCONFIRMED_LOUD_FAILURE on
        # an earlier pod — exactly the evidence an operator needs.
        self.history_dir = os.path.join(out_dir, "lifecycle")
        os.makedirs(self.history_dir, exist_ok=True)
        self.attempt_report_path = os.path.join(
            self.history_dir, f"LIFECYCLE_REPORT.attempt{self.attempt:02d}.json")
        self.pod_id_ledger = os.path.join(out_dir, "POD_IDS.jsonl")
        self.events: list[dict] = []

    def _event(self, kind: str, **fields):
        entry = {"event": kind, "utc": utcnow_iso(), **fields}
        self.events.append(json.loads(redact(json.dumps(entry))))

    def _record_pod_id(self, pod_id: str) -> None:
        record = {"pod_id": pod_id, "attempt": self.attempt,
                  "utc": utcnow_iso()}
        tmp = self.pod_id_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.pod_id_path)
        # append-only ledger of EVERY pod this session created, so an
        # earlier pod can never be lost from the record
        with open(self.pod_id_ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def provision(self, req, rendered) -> str:
        """Create exactly one Pod (reconciled, duplicate-safe)."""
        self._event("PROVISION_REQUESTED", name=req.name)
        pod = self.adapter.create_instance(req, rendered)
        self._record_pod_id(pod.id)
        self._event("POD_CREATED", pod_id=pod.id, status=pod.status)
        try:
            return self._arm_and_return(pod)
        except Exception as exc:  # noqa: BLE001
            # The pod EXISTS and is billing from here on.  Anything that
            # goes wrong while arming (an expired quote, a budget refusal)
            # must terminate it rather than leave a billable orphan whose
            # id the caller never received.
            self._event("ARMING_FAILED", pod_id=pod.id, error=str(exc)[:300])
            try:
                self.terminate_and_confirm(pod.id)
            finally:
                raise

    def _arm_and_return(self, pod) -> str:
        # arm the independent watchdog IMMEDIATELY, before waiting on startup.
        # Its deadline is the REMAINING compute allocation (spend from earlier
        # evicted pods already carried into the tracker), never the full
        # allocation: N pods each armed at the full budget would authorize
        # N x MAX_COMPUTE_USD of unattended runtime.
        self.adapter.validate_quote()
        limit = remaining_compute_seconds(
            self.adapter.accepted_quote["total_projected_hourly_usd"],
            self.adapter.session_spend_usd())
        if limit <= 0:
            # provision()'s handler terminates the pod for every failure in
            # here, so this must not terminate a second time
            self._event("BUDGET_EXHAUSTED_AT_PROVISION")
            raise BudgetViolation(
                "no compute allocation remains after provisioning")
        self.watchdog = self.spawn_watchdog(
            pod_id=pod.id, hard_limit_seconds=limit, out_dir=self.out_dir)
        self._event("WATCHDOG_ARMED", hard_limit_seconds=limit,
                    session_spend_usd=str(self.adapter.session_spend_usd()))
        return pod.id

    def wait_until_running(self, pod_id: str) -> str:
        """Returns "RUNNING", or "EVICTED" when interruptible capacity was
        reclaimed before the container came up (normal spot behavior: the
        remnant is terminated and the session may reacquire)."""
        try:
            pod = self.adapter.wait_for_state(
                pod_id, "RUNNING", timeout_seconds=self.startup_timeout,
                sleep=self.sleep)
        except RunpodAdapterError as exc:
            if "EVICTION_SUSPECTED" in str(exc):
                self._event("EVICTED_BEFORE_RUNNING", pod_id=pod_id)
                self.collect_logs(pod_id)
                self.terminate_and_confirm(pod_id)
                return "EVICTED"
            self._event("STARTUP_FAILED", error=str(exc))
            self.collect_logs(pod_id)
            self.terminate_and_confirm(pod_id)
            raise LifecycleError(f"startup failed: {exc}") from None
        self.adapter.spend.mark_pod_started()
        self._event("POD_RUNNING", pod_id=pod.id,
                    started_at=pod.started_at)
        return "RUNNING"

    def monitor(self, pod_id: str, *, poll_seconds: float = 20.0,
                until=None) -> str:
        """Poll status/budget; returns 'COMPLETE' | raises on failure.

        Bounded by monitor_timeout_seconds when supplied: a pod that stays
        RUNNING and never emits its completion witness would otherwise loop
        until the budget soft-stop, paying for the whole remaining
        allocation with nothing to show.
        """
        started = time.monotonic()
        while True:
            if self.monitor_timeout is not None and \
                    time.monotonic() - started > self.monitor_timeout:
                self._event("MONITOR_TIMEOUT",
                            seconds=round(time.monotonic() - started, 1))
                self.collect_logs(pod_id)
                self.terminate_and_confirm(pod_id)
                raise LifecycleError(
                    f"pod {pod_id} produced no completion witness within "
                    f"{self.monitor_timeout}s; logs collected, terminated")
            pod = self.adapter.get_instance(pod_id)
            if self.adapter.spend is not None:
                try:
                    bill = self.adapter.get_billing_usage(pod_id)
                    self._event("BILLING_SAMPLE", data=str(bill)[:200])
                except Exception:  # noqa: BLE001 - billing is supplementary
                    pass
                if self.adapter.spend.must_terminate():
                    self._event("BUDGET_HARD_STOP")
                    self.terminate_and_confirm(pod_id)
                    raise BudgetViolation(
                        "100% of compute allocation: immediate termination "
                        "requested; no further cleanup on the billable pod")
                if self.adapter.spend.state() == "SOFT_STOP":
                    self._event("BUDGET_SOFT_STOP")
                    return "SOFT_STOP"
            if pod.status == "EXITED":
                # On interruptible capacity, EXITED without the completion
                # marker is an eviction (spot stop), not a scientific
                # failure: collect what remains, terminate the remnant, and
                # let the session decide on bounded reacquisition.
                if until is not None and until(pod):
                    return "COMPLETE"
                self._event("EVICTED", pod_id=pod_id)
                self.collect_logs(pod_id)
                # spend is frozen INSIDE terminate_and_confirm, after the
                # confirmation poll: a pod still bills while terminating
                self.terminate_and_confirm(pod_id)
                return "EVICTED"
            if pod.status == "ERROR":
                self._event("CONTAINER_FAILURE", status=pod.status)
                self.collect_logs(pod_id)
                self.terminate_and_confirm(pod_id)
                raise LifecycleError(
                    f"container reached {pod.status}; logs collected, pod "
                    f"terminated")
            if pod.status == "TERMINATED":
                self._event("POD_TERMINATED_EXTERNALLY")
                return "TERMINATED"
            if until is not None and until(pod):
                return "COMPLETE"
            self.sleep(poll_seconds)

    def collect_logs(self, pod_id: str) -> None:
        for source, fn in (("container", self.adapter.get_container_logs),
                           ("system", self.adapter.get_system_logs)):
            try:
                text = fn(pod_id, tail=500)
                path = os.path.join(self.out_dir, f"pod_{source}_logs.txt")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(redact(text))
                self._event("LOGS_COLLECTED", source=source)
            except Exception as exc:  # noqa: BLE001
                self._event("LOG_COLLECTION_FAILED", source=source,
                            error=str(exc)[:200])

    def terminate_and_confirm(self, pod_id: str) -> bool:
        confirmed = False
        primary_error = None
        try:
            self.adapter.terminate_instance(pod_id)
            self._event("TERMINATE_REQUESTED", path="primary")
            confirmed = self.adapter.confirm_terminated(pod_id,
                                                        sleep=self.sleep)
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            self._event("PRIMARY_TERMINATION_FAILED", error=primary_error[:300])
        if not confirmed:
            try:
                wd = getattr(self, "watchdog", None)
                if wd is not None:
                    wd.terminate_now()
                    confirmed = self.adapter.confirm_terminated(
                        pod_id, sleep=self.sleep)
                    self._event("WATCHDOG_TERMINATION",
                                confirmed=confirmed)
            except Exception as exc:  # noqa: BLE001
                self._event("WATCHDOG_TERMINATION_FAILED",
                            error=str(exc)[:300])
        if not confirmed:
            self._event("TERMINATION_UNCONFIRMED_LOUD_FAILURE")
        # Freeze this pod's spend into the session carryover only NOW: the
        # pod kept billing throughout the terminate + confirmation poll
        # (up to confirm timeout), and that time must be charged before a
        # replacement pod's meter starts.
        if self.adapter.spend is not None:
            carried = self.adapter.spend.mark_pod_stopped()
            self._event("SPEND_FROZEN", session_spend_usd=str(carried))
        try:
            bill = self.adapter.get_billing_usage(pod_id)
            self._event("FINAL_BILLING", data=str(bill)[:300])
        except Exception:  # noqa: BLE001
            self._event("FINAL_BILLING_UNAVAILABLE")
        self.write_report(confirmed)
        return confirmed

    def write_report(self, termination_confirmed: bool) -> None:
        report = {
            "schema": "o1b300.runpod_lifecycle_report.v2",
            "attempt": self.attempt,
            "termination_confirmed": termination_confirmed,
            "events": self.events,
            "machine_readable": True,
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        # the per-attempt copy is the durable evidence; the shared path is
        # kept as the "latest" convenience view
        for path in (self.attempt_report_path, self.report_path):
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
