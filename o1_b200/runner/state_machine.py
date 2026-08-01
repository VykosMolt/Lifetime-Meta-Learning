"""Zero-touch execution state machine for the future B200 session.

States (in order):

  PRECHECK -> ARTIFACT_VERIFY -> ENVIRONMENT_VERIFY -> NON_O1_EQUIVALENCE ->
  NON_O1_BENCHMARK -> BACKEND_SELECT -> PRECOMMIT_BUILD ->
  EXTERNAL_COMMIT_VERIFY -> CALIBRATION_AFFORDABILITY_CHECK -> CALIBRATION ->
  RECORD_VERIFY -> RESULT_TRANSFER -> TERMINATE -> COMPLETE

During THIS task the machine only ever runs with the MockProviderAdapter (or
LocalProviderAdapter), the synthetic runtime, and the non-O1 validation
corpus.  The CALIBRATION state is hard-gated:

  * refuses any binding equal to the sealed O1 calibration precommit;
  * refuses any task whose ID appears in the O1 calibration manifest or the
    confirmatory candidate pool;
  * refuses a mock precommit for anything but the labeled dress rehearsal;
  * refuses confirmatory generation ALWAYS (confirmation_authorized=false is
    frozen in the budget policy and there is no code path that launches it).

On any state failure: later states never begin (except TERMINATE, which is
the abort path), completed synthetic records survive, invalid records are
quarantined by the persistence layer, instance termination is requested and
confirmed, and a machine-readable FINAL_STATUS.json is written.  No human
interaction is required at any point.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from .budget import BudgetRefusal, BudgetWatchdog, affordability_gate, load_policy
from .identity import canonical_json, domain_sha256
from .persistence import atomic_write_text
from .provider_adapter import ProviderAdapter

STATES = (
    "PRECHECK", "ARTIFACT_VERIFY", "ENVIRONMENT_VERIFY", "NON_O1_EQUIVALENCE",
    "NON_O1_BENCHMARK", "BACKEND_SELECT", "PRECOMMIT_BUILD",
    "EXTERNAL_COMMIT_VERIFY", "CALIBRATION_AFFORDABILITY_CHECK", "CALIBRATION",
    "RECORD_VERIFY", "RESULT_TRANSFER", "TERMINATE", "COMPLETE",
)

# The sealed O1 calibration precommit — the CALIBRATION state refuses it.
O1_CALIBRATION_PRECOMMIT_SHA256 = (
    "e819aeebc642fefde769f398b385f23f98dc3980412eeb67274d0f5891c5457d")


class StateMachineError(RuntimeError):
    def __init__(self, state: str, message: str):
        super().__init__(f"[{state}] {message}")
        self.state = state


class ZeroTouchStateMachine:
    """Handlers are injected so local tests exercise every transition.

    handlers: {state_name: callable(context) -> dict}.  A handler's return
    value is stored in context["results"][state].  Any exception aborts the
    run: no later state begins, TERMINATE runs, FINAL_STATUS.json is written.
    """

    def __init__(self, provider: ProviderAdapter, out_dir: str,
                 handlers: dict[str, Callable[[dict], dict]],
                 watchdog: BudgetWatchdog | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.provider = provider
        self.out_dir = out_dir
        self.handlers = handlers
        self.watchdog = watchdog
        self.clock = clock
        os.makedirs(out_dir, exist_ok=True)
        self.context: dict[str, Any] = {
            "results": {}, "instance_ref": None, "states_run": [],
        }

    # ---------------- built-in guards ----------------

    def _guard_calibration(self) -> None:
        ctx = self.context
        binding = ctx.get("calibration_binding_sha256")
        if binding == O1_CALIBRATION_PRECOMMIT_SHA256:
            raise StateMachineError(
                "CALIBRATION",
                "REFUSED: this machine may never run real O1 calibration in "
                "the local/mocked mode (binding equals the sealed O1 "
                "calibration precommit)")
        task_ids = ctx.get("calibration_task_ids") or []
        forbidden = ctx.get("o1_task_ids") or set()
        overlap = sorted(set(task_ids) & set(forbidden))
        if overlap:
            raise StateMachineError(
                "CALIBRATION",
                f"REFUSED: O1 tasks present in the calibration input: "
                f"{overlap[:3]}")
        precommit = ctx.get("finalized_precommit")
        if precommit is None:
            raise StateMachineError(
                "CALIBRATION", "no finalized precommit in context")
        if precommit.get("mock") and not ctx.get("dress_rehearsal"):
            raise StateMachineError(
                "CALIBRATION",
                "REFUSED: mock precommit outside a labeled dress rehearsal")
        policy = load_policy()
        if policy["confirmation_authorized"] is not False:
            raise StateMachineError(
                "CALIBRATION", "budget policy corrupted")
        if self.context.get("requested_mode") == "confirmatory":
            raise StateMachineError(
                "CALIBRATION",
                "REFUSED: confirmatory generation is not authorized "
                "(confirmation_authorized=false is frozen)")

    def _run_state(self, state: str) -> None:
        if self.watchdog is not None and state not in ("TERMINATE", "COMPLETE"):
            self.watchdog.check_may_launch_work(state)
        if state == "CALIBRATION":
            self._guard_calibration()
        handler = self.handlers.get(state)
        result = handler(self.context) if handler is not None else {"skipped": False}
        self.context["results"][state] = result

    def _terminate(self) -> dict:
        info = {"termination_requested": False, "termination_confirmed": False}
        ref = self.context.get("instance_ref")
        try:
            if ref is not None:
                self.provider.terminate_instance(ref)
                info["termination_requested"] = True
                info["termination_confirmed"] = bool(
                    self.provider.confirm_terminated(ref))
            else:
                info["termination_requested"] = True
                info["termination_confirmed"] = True  # nothing to terminate
        except Exception as exc:  # noqa: BLE001 - report, never raise past
            info["termination_error"] = repr(exc)
        return info

    def _write_status(self, status: dict) -> None:
        status["machine_readable"] = True
        status["human_interaction_required"] = status.get(
            "human_interaction_required", False)
        status["status_sha256"] = domain_sha256("o1b200.final_status.v1", {
            k: v for k, v in status.items() if k != "status_sha256"})
        atomic_write_text(os.path.join(self.out_dir, "FINAL_STATUS.json"),
                          json.dumps(status, indent=2, sort_keys=True) + "\n")

    def run(self) -> dict:
        failed_state = None
        failure = None
        for state in STATES:
            if state == "TERMINATE":
                break  # handled below in both success and failure paths
            try:
                self.context["states_run"].append(state)
                self._run_state(state)
            except BaseException as exc:  # noqa: BLE001
                failed_state = state
                failure = repr(exc)
                break
        term = self._terminate()
        self.context["results"]["TERMINATE"] = term
        if failed_state is None:
            self.context["states_run"].append("TERMINATE")
            self.context["states_run"].append("COMPLETE")
            status = {
                "outcome": "COMPLETE",
                "states_run": list(self.context["states_run"]),
                "termination": term,
                "human_interaction_required": not term.get(
                    "termination_confirmed", False),
            }
        else:
            status = {
                "outcome": f"ABORTED_AT_{failed_state}",
                "failed_state": failed_state,
                "failure": failure,
                "states_run": list(self.context["states_run"]),
                "states_never_started": [
                    s for s in STATES
                    if s not in self.context["states_run"]
                    and s not in ("TERMINATE", "COMPLETE")],
                "termination": term,
                "human_interaction_required": not term.get(
                    "termination_confirmed", False),
            }
        self._write_status(status)
        return status
