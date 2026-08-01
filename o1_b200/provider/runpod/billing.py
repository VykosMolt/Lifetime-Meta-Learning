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
    MAX_COMPUTE_USD, MAX_GPU_HOURLY_USD, PolicyViolation, SOFT_STOP_FRACTION,
    TOTAL_AUTHORIZED_USD,
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


def hard_compute_seconds(accepted_total_hourly_rate) -> int:
    rate = as_money(accepted_total_hourly_rate)
    if rate <= 0:
        raise BudgetViolation(f"hourly rate must be positive, got {rate}")
    if rate > TOTAL_AUTHORIZED_USD:
        raise BudgetViolation(
            f"hourly rate {rate} exceeds the entire authorized budget")
    seconds = (MAX_COMPUTE_USD / rate * Decimal(3600)).to_integral_value(
        rounding=ROUND_FLOOR)
    return int(seconds)


def validate_gpu_rate(gpu_hourly) -> Decimal:
    rate = as_money(gpu_hourly)
    if rate <= 0:
        raise PolicyViolation(f"quoted GPU rate {rate} is not positive")
    if rate > MAX_GPU_HOURLY_USD:
        raise PolicyViolation(
            f"quoted GPU rate USD {rate}/h exceeds the frozen maximum "
            f"USD {MAX_GPU_HOURLY_USD}/h; refusing to provision")
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
    the Pod start timestamp, and supplementary live billing samples."""

    def __init__(self, total_hourly_rate, clock=time.monotonic):
        self.rate = as_money(total_hourly_rate)
        self.limit_seconds = hard_compute_seconds(self.rate)
        self.clock = clock
        self.pod_started_monotonic: float | None = None
        self.live_billed_usd: Decimal | None = None

    def mark_pod_started(self) -> None:
        if self.pod_started_monotonic is None:
            self.pod_started_monotonic = self.clock()

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
        """max(monotonic projection, live billing) — never trust a lagging
        billing feed to extend the run."""
        m = self.monotonic_spend()
        if self.live_billed_usd is not None and self.live_billed_usd > m:
            return self.live_billed_usd
        return m

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
