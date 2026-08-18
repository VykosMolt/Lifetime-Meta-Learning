"""Hard budget binding with decimal arithmetic.

hard_compute_seconds = floor(40.00 / accepted_total_hourly_rate * 3600)

The monotonic watchdog is authoritative; live billing data is supplementary
(it may lag).  95%: stop launching work, preserve/transfer records, begin
termination.  100%: immediate termination; no further cleanup on the
billable Pod.
"""
from __future__ import annotations

import time
from decimal import ROUND_FLOOR, Decimal

from .policy import (
    MAX_COMPUTE_USD, MIN_VIABLE_SESSION_SECONDS, PolicyViolation,
    RESERVED_NONCOMPUTE_USD, SOFT_STOP_FRACTION, TOTAL_AUTHORIZED_USD,
)


class BudgetViolation(RuntimeError):
    pass


def as_money(v) -> Decimal:
    if isinstance(v, float):
        # floats are only accepted from live API JSON, converted via str to
        # avoid binary artifacts; all POLICY arithmetic stays Decimal.
        v = str(v)
    d = Decimal(v)
    if not d.is_finite():
        raise BudgetViolation(f"non-finite money value {v!r}")
    return d


def assert_policy_coherent() -> None:
    """The three budget constants must agree, or every derived limit lies."""
    if MAX_COMPUTE_USD + RESERVED_NONCOMPUTE_USD != TOTAL_AUTHORIZED_USD:
        raise BudgetViolation(
            f"budget policy incoherent: compute {MAX_COMPUTE_USD} + reserved "
            f"non-compute {RESERVED_NONCOMPUTE_USD} != total authorized "
            f"{TOTAL_AUTHORIZED_USD}")


def hard_compute_seconds(accepted_total_hourly_rate) -> int:
    """Runtime the FULL compute allocation buys at this rate (session start)."""
    assert_policy_coherent()
    rate = as_money(accepted_total_hourly_rate)
    if rate <= 0:
        raise BudgetViolation(f"hourly rate must be positive, got {rate}")
    if rate > TOTAL_AUTHORIZED_USD:
        raise BudgetViolation(
            f"hourly rate {rate} exceeds the entire authorized budget")
    seconds = (MAX_COMPUTE_USD / rate * Decimal(3600)).to_integral_value(
        rounding=ROUND_FLOOR)
    return int(seconds)


def remaining_compute_seconds(accepted_total_hourly_rate,
                              already_spent_usd="0") -> int:
    """Runtime the REMAINING compute allocation buys at this rate.

    Every per-pod deadline (the independent watchdog and the provider-side
    auto-terminate) must be derived from this, never from
    hard_compute_seconds: an eviction/reacquisition sequence that armed each
    pod at the full allocation would authorize N x MAX_COMPUTE_USD of
    unattended runtime if the driving orchestrator died.
    """
    rate = as_money(accepted_total_hourly_rate)
    if rate <= 0:
        raise BudgetViolation(f"hourly rate must be positive, got {rate}")
    spent = as_money(already_spent_usd)
    if spent < 0:
        raise BudgetViolation(f"spend cannot be negative, got {spent}")
    remaining = MAX_COMPUTE_USD - spent
    if remaining <= 0:
        return 0
    return int((remaining / rate * Decimal(3600)).to_integral_value(
        rounding=ROUND_FLOOR))


def validate_gpu_rate(gpu_hourly) -> Decimal:
    """Mechanical viability: the committed compute budget must buy at least
    MIN_VIABLE_SESSION_SECONDS at this rate.  There is no static price
    assumption; the live quote is authoritative and only the budget-derived
    viability floor can refuse it."""
    rate = as_money(gpu_hourly)
    if rate <= 0:
        raise PolicyViolation(f"quoted GPU rate {rate} is not positive")
    affordable = hard_compute_seconds(rate)
    if affordable < MIN_VIABLE_SESSION_SECONDS:
        raise PolicyViolation(
            f"quoted rate USD {rate}/h buys only {affordable}s of the "
            f"USD {MAX_COMPUTE_USD} compute allocation; below the "
            f"{MIN_VIABLE_SESSION_SECONDS}s minimum viable session — "
            f"refusing to provision")
    return rate


def projected_session_cost(total_hourly_rate, projected_seconds: int) -> Decimal:
    rate = as_money(total_hourly_rate)
    return (rate * Decimal(int(projected_seconds)) / Decimal(3600)
            ).quantize(Decimal("0.0001"))


def session_fits_policy(total_hourly_rate, projected_seconds: int) -> None:
    cost = projected_session_cost(total_hourly_rate, projected_seconds)
    if cost > MAX_COMPUTE_USD:
        raise BudgetViolation(
            f"projected session cost USD {cost} exceeds the USD "
            f"{MAX_COMPUTE_USD} compute allocation")


class SpendTracker:
    """Combines monotonic elapsed time (authoritative), the accepted quote,
    the Pod start timestamp, and supplementary live billing samples.

    Session-cumulative: spend from earlier pods in the same session
    (evicted interruptible capacity) is carried over via ``carryover_usd``
    or ``mark_pod_stopped()`` so reacquisition can never reset the budget.
    """

    def __init__(self, total_hourly_rate, clock=time.monotonic,
                 carryover_usd="0"):
        self.rate = as_money(total_hourly_rate)
        self.limit_seconds = hard_compute_seconds(self.rate)
        self.clock = clock
        self.pod_started_monotonic: float | None = None
        self.live_billed_usd: Decimal | None = None
        self.carryover_usd = as_money(carryover_usd)
        if self.carryover_usd < 0:
            raise BudgetViolation("carryover spend cannot be negative")

    def mark_pod_started(self) -> None:
        if self.pod_started_monotonic is None:
            self.pod_started_monotonic = self.clock()

    def mark_pod_stopped(self) -> Decimal:
        """Freeze the current pod's spend into carryover (eviction/stop);
        returns the cumulative session spend so far."""
        self.carryover_usd = self.effective_spend()
        self.pod_started_monotonic = None
        self.live_billed_usd = None
        return self.carryover_usd

    def record_live_billing(self, billed_usd) -> None:
        self.live_billed_usd = as_money(billed_usd)

    def elapsed_seconds(self) -> Decimal:
        if self.pod_started_monotonic is None:
            return Decimal(0)
        return Decimal(str(self.clock() - self.pod_started_monotonic))

    def monotonic_spend(self) -> Decimal:
        return (self.rate * self.elapsed_seconds() / Decimal(3600)
                ).quantize(Decimal("0.0001"))

    def effective_spend(self) -> Decimal:
        """carryover + max(monotonic projection, live billing) — never trust
        a lagging billing feed to extend the run, and never let a fresh pod
        forget what earlier pods in the session already spent."""
        m = self.monotonic_spend()
        if self.live_billed_usd is not None and self.live_billed_usd > m:
            m = self.live_billed_usd
        return (self.carryover_usd + m).quantize(Decimal("0.0001"))

    def state(self) -> str:
        spend = self.effective_spend()
        if spend >= MAX_COMPUTE_USD:
            return "HARD_STOP"
        if spend >= MAX_COMPUTE_USD * SOFT_STOP_FRACTION:
            return "SOFT_STOP"
        return "RUNNING"

    def check_may_launch_work(self, label: str = "work") -> None:
        s = self.state()
        if s != "RUNNING":
            raise BudgetViolation(
                f"{s}: launching new {label} refused at USD "
                f"{self.effective_spend()} of USD {MAX_COMPUTE_USD}")

    def must_terminate(self) -> bool:
        return self.state() == "HARD_STOP"
