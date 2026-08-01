"""Budget controls: monotonic watchdog + affordability gate (mock-testable).

USD 45.00 total / USD 40.00 compute ceiling, frozen in
policies/B200_BUDGET_POLICY.template.json.  Nothing here spends money or
talks to a provider; the clock and rate are injected so every rule is locally
testable with a mocked clock and mocked provider.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Callable


POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "policies", "B200_BUDGET_POLICY.template.json")


class BudgetError(RuntimeError):
    pass


class BudgetRefusal(BudgetError):
    """A frozen budget rule refused the requested action."""


def load_policy(path: str | None = None) -> dict:
    with open(os.path.abspath(path or POLICY_PATH), encoding="utf-8") as fh:
        policy = json.load(fh)
    for key, want in (("total_authorized_usd", 45.00),
                      ("maximum_compute_spend_usd", 40.00),
                      ("reserved_noncompute_usd", 5.00)):
        if float(policy[key]) != want:
            raise BudgetError(f"budget policy {key} altered from frozen {want}")
    for key in ("confirmation_authorized", "automatic_extension_allowed",
                "operator_override_allowed", "auto_reload_allowed"):
        if policy[key] is not False:
            raise BudgetError(f"budget policy {key} must be false")
    return policy


def compute_runtime_limit_seconds(maximum_compute_spend_usd: float,
                                  actual_hourly_rate_usd: float) -> int:
    """floor(max_compute_usd / rate * 3600) — the frozen rule."""
    if not (actual_hourly_rate_usd > 0.0) or not math.isfinite(actual_hourly_rate_usd):
        raise BudgetError("hourly rate must be a positive finite number")
    return int(math.floor(
        maximum_compute_spend_usd / actual_hourly_rate_usd * 3600.0))


class BudgetWatchdog:
    """Monotonic in-process watchdog over a hard runtime limit.

    States: RUNNING -> SOFT_STOP (>=95%: no new work may launch) ->
    HARD_STOP (>=100%: immediate termination required).  There is no
    extension path and no override path — by policy those fields are frozen
    false and this class has no method to extend.
    """

    SOFT_FRACTION = 0.95

    def __init__(self, runtime_limit_seconds: int,
                 clock: Callable[[], float] | None = None,
                 on_hard_stop: Callable[[], None] | None = None):
        if runtime_limit_seconds <= 0:
            raise BudgetError("runtime limit must be positive")
        self.limit = float(runtime_limit_seconds)
        self.clock = clock or time.monotonic
        self.on_hard_stop = on_hard_stop
        self.t0 = self.clock()
        self._hard_stop_fired = False

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def fraction_used(self) -> float:
        return self.elapsed() / self.limit

    def state(self) -> str:
        f = self.fraction_used()
        if f >= 1.0:
            if not self._hard_stop_fired:
                self._hard_stop_fired = True
                if self.on_hard_stop is not None:
                    self.on_hard_stop()
            return "HARD_STOP"
        if f >= self.SOFT_FRACTION:
            return "SOFT_STOP"
        return "RUNNING"

    def remaining_seconds(self) -> float:
        return max(0.0, self.limit - self.elapsed())

    def check_may_launch_work(self, label: str = "work") -> None:
        s = self.state()
        if s != "RUNNING":
            raise BudgetRefusal(
                f"watchdog state {s}: launching new {label} is refused "
                f"({self.fraction_used():.1%} of authorized runtime used)")

    def check_must_terminate(self) -> bool:
        return self.state() == "HARD_STOP"


class ExternalWatchdogInterface:
    """Contract for the independent out-of-process watchdog.

    The future B200 session runs a separate process (or provider-side alarm)
    that terminates the instance if the in-process watchdog dies.  Locally a
    mock implements this interface; no real provider call exists here.
    """

    def arm(self, runtime_limit_seconds: int, instance_ref: str) -> None:
        raise NotImplementedError

    def heartbeat(self) -> None:
        raise NotImplementedError

    def confirm_armed(self) -> bool:
        raise NotImplementedError


class MockExternalWatchdog(ExternalWatchdogInterface):
    def __init__(self, clock: Callable[[], float]):
        self.clock = clock
        self.armed_at = None
        self.limit = None
        self.instance_ref = None
        self.heartbeats: list[float] = []
        self.terminated = False

    def arm(self, runtime_limit_seconds: int, instance_ref: str) -> None:
        self.armed_at = self.clock()
        self.limit = runtime_limit_seconds
        self.instance_ref = instance_ref

    def heartbeat(self) -> None:
        self.heartbeats.append(self.clock())

    def confirm_armed(self) -> bool:
        return self.armed_at is not None

    def tick(self) -> None:
        """Mock external check: terminate if over limit."""
        if (self.armed_at is not None
                and self.clock() - self.armed_at >= self.limit):
            self.terminated = True


def affordability_gate(*, projected_calibration_seconds: float,
                       verification_transfer_reserve_seconds: float,
                       termination_reserve_seconds: float,
                       remaining_authorized_runtime_seconds: float) -> dict:
    """The frozen pre-calibration affordability rule.

    projected * 1.30 + verification/transfer reserve + termination reserve
    must fit inside the remaining authorized runtime; otherwise the system
    REFUSES to start calibration.
    """
    for name, v in (("projected_calibration_seconds", projected_calibration_seconds),
                    ("verification_transfer_reserve_seconds",
                     verification_transfer_reserve_seconds),
                    ("termination_reserve_seconds", termination_reserve_seconds),
                    ("remaining_authorized_runtime_seconds",
                     remaining_authorized_runtime_seconds)):
        if not math.isfinite(v) or v < 0:
            raise BudgetError(f"{name} must be finite and non-negative")
    required = (projected_calibration_seconds * 1.30
                + verification_transfer_reserve_seconds
                + termination_reserve_seconds)
    ok = required <= remaining_authorized_runtime_seconds
    result = {
        "required_seconds": required,
        "remaining_seconds": remaining_authorized_runtime_seconds,
        "safety_factor": 1.30,
        "affordable": ok,
    }
    if not ok:
        raise BudgetRefusal(
            f"affordability gate REFUSES calibration: requires "
            f"{required:.0f}s but only "
            f"{remaining_authorized_runtime_seconds:.0f}s remain")
    return result
