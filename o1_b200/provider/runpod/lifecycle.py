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
from .billing import BudgetViolation
from .identityutil import utcnow_iso
from .redaction import redact
from .watchdog_terminate import spawn_watchdog


class LifecycleError(RuntimeError):
    def __init__(self, msg: str):
        super().__init__(redact(msg))


class PodLifecycleController:
    def __init__(self, adapter: RunpodV2Adapter, out_dir: str,
                 *, startup_timeout_seconds: float = 900,
                 sleep=time.sleep, spawn_watchdog_fn=spawn_watchdog):
        self.adapter = adapter
        self.out_dir = out_dir
        self.startup_timeout = startup_timeout_seconds
        self.sleep = sleep
        self.spawn_watchdog = spawn_watchdog_fn
        os.makedirs(out_dir, exist_ok=True)
        self.pod_id_path = os.path.join(out_dir, "POD_ID.json")
        self.report_path = os.path.join(out_dir, "LIFECYCLE_REPORT.json")
        self.events: list[dict] = []

    def _event(self, kind: str, **fields):
        entry = {"event": kind, "utc": utcnow_iso(), **fields}
        self.events.append(json.loads(redact(json.dumps(entry))))

    def _record_pod_id(self, pod_id: str) -> None:
        tmp = self.pod_id_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"pod_id": pod_id, "utc": utcnow_iso()}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.pod_id_path)

    def provision(self, req, rendered) -> str:
        """Create exactly one Pod (reconciled, duplicate-safe)."""
        self._event("PROVISION_REQUESTED", name=req.name)
        pod = self.adapter.create_instance(req, rendered)
        self._record_pod_id(pod.id)
        self._event("POD_CREATED", pod_id=pod.id, status=pod.status)
        # arm the independent watchdog IMMEDIATELY, before waiting on startup
        limit = self.adapter.validate_quote()["hard_compute_seconds"]
        self.watchdog = self.spawn_watchdog(
            pod_id=pod.id, hard_limit_seconds=limit, out_dir=self.out_dir)
        self._event("WATCHDOG_ARMED", hard_limit_seconds=limit)
        return pod.id

    def wait_until_running(self, pod_id: str) -> None:
        try:
            pod = self.adapter.wait_for_state(
                pod_id, "RUNNING", timeout_seconds=self.startup_timeout,
                sleep=self.sleep)
        except RunpodAdapterError as exc:
            self._event("STARTUP_FAILED", error=str(exc))
            self.collect_logs(pod_id)
            self.terminate_and_confirm(pod_id)
            raise LifecycleError(f"startup failed: {exc}") from None
        self.adapter.spend.mark_pod_started()
        self._event("POD_RUNNING", pod_id=pod.id,
                    started_at=pod.started_at)

    def monitor(self, pod_id: str, *, poll_seconds: float = 20.0,
                until=None) -> str:
        """Poll status/budget; returns 'COMPLETE' | raises on failure."""
        while True:
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
            if pod.status in ("EXITED", "ERROR"):
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
        try:
            bill = self.adapter.get_billing_usage(pod_id)
            self._event("FINAL_BILLING", data=str(bill)[:300])
        except Exception:  # noqa: BLE001
            self._event("FINAL_BILLING_UNAVAILABLE")
        self.write_report(confirmed)
        return confirmed

    def write_report(self, termination_confirmed: bool) -> None:
        report = {
            "schema": "o1b200.runpod_lifecycle_report.v1",
            "termination_confirmed": termination_confirmed,
            "events": self.events,
            "machine_readable": True,
        }
        tmp = self.report_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, self.report_path)
