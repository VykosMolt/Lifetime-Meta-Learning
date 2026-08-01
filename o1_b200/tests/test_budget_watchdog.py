"""Budget policy, watchdog thresholds, affordability gate — mocked clock only."""
from __future__ import annotations

import json
import os

from _h import Runner, fresh_dir

from o1_b200.runner.budget import (
    BudgetError, BudgetRefusal, BudgetWatchdog, MockExternalWatchdog,
    affordability_gate, compute_runtime_limit_seconds, load_policy,
)


class MockClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def run() -> Runner:
    r = Runner("budget_watchdog")

    def policy_frozen():
        p = load_policy()
        assert p["total_authorized_usd"] == 45.00
        assert p["maximum_compute_spend_usd"] == 40.00
        assert p["confirmation_authorized"] is False
    r.check("budget policy loads with the frozen USD 45/40/5 values",
            policy_frozen)

    def policy_tamper_detected():
        import tempfile
        p = load_policy()
        p["maximum_compute_spend_usd"] = 400.0
        d = fresh_dir("budget_tamper")
        path = os.path.join(d, "policy.json")
        with open(path, "w") as fh:
            json.dump(p, fh)
        try:
            load_policy(path)
        except BudgetError:
            return
        raise AssertionError("tampered budget ceiling accepted")
    r.check("HOSTILE tampered budget ceiling is refused", policy_tamper_detected)

    def override_flag_tamper_detected():
        p = load_policy()
        p["operator_override_allowed"] = True
        d = fresh_dir("budget_tamper2")
        path = os.path.join(d, "policy.json")
        with open(path, "w") as fh:
            json.dump(p, fh)
        try:
            load_policy(path)
        except BudgetError:
            return
        raise AssertionError("operator override flag accepted")
    r.check("HOSTILE operator-override enablement is refused",
            override_flag_tamper_detected)

    def runtime_limit_rule():
        # simulated USD 40 ceiling at a mocked 2.99 USD/h quote
        limit = compute_runtime_limit_seconds(40.00, 2.99)
        assert limit == int(40.00 / 2.99 * 3600)  # floor
        assert compute_runtime_limit_seconds(40.00, 40.00) == 3600
        try:
            compute_runtime_limit_seconds(40.00, 0.0)
        except BudgetError:
            return
        raise AssertionError("zero rate accepted")
    r.check("runtime limit = floor(40 USD / rate * 3600); zero rate refused",
            runtime_limit_rule)

    def watchdog_thresholds():
        clock = MockClock()
        fired = []
        wd = BudgetWatchdog(1000, clock=clock,
                            on_hard_stop=lambda: fired.append(True))
        assert wd.state() == "RUNNING"
        wd.check_may_launch_work()
        clock.advance(949)
        assert wd.state() == "RUNNING"
        clock.advance(2)  # 951s -> over 95%
        assert wd.state() == "SOFT_STOP"
        try:
            wd.check_may_launch_work("benchmark stage")
        except BudgetRefusal:
            pass
        else:
            raise AssertionError("work launched past 95%")
        clock.advance(60)  # 1011s -> over 100%
        assert wd.state() == "HARD_STOP"
        assert wd.check_must_terminate()
        assert fired == [True], "hard-stop callback did not fire exactly once"
        assert wd.remaining_seconds() == 0.0
    r.check("watchdog: 95% stop-launching and 100% immediate-termination "
            "thresholds fire on the mocked USD-40 timeout",
            watchdog_thresholds)

    def no_extension_exists():
        wd = BudgetWatchdog(10, clock=MockClock())
        assert not hasattr(wd, "extend")
        assert not hasattr(wd, "override")
    r.check("no automatic extension / override method exists on the watchdog",
            no_extension_exists)

    def external_watchdog_covers_inprocess_death():
        clock = MockClock()
        ext = MockExternalWatchdog(clock)
        ext.arm(500, "mock-instance-0001")
        assert ext.confirm_armed()
        # in-process watchdog "dies" (no heartbeats); external still terminates
        clock.advance(501)
        ext.tick()
        assert ext.terminated, "external watchdog failed to terminate"
    r.check("watchdog failure: independent external watchdog terminates "
            "even when the in-process watchdog is dead",
            external_watchdog_covers_inprocess_death)

    def affordability_pass_and_refuse():
        ok = affordability_gate(
            projected_calibration_seconds=10_000,
            verification_transfer_reserve_seconds=1200,
            termination_reserve_seconds=300,
            remaining_authorized_runtime_seconds=15_000)
        assert ok["affordable"] and abs(ok["required_seconds"] - 14_500) < 1e-9
        try:
            affordability_gate(
                projected_calibration_seconds=11_000,
                verification_transfer_reserve_seconds=1200,
                termination_reserve_seconds=300,
                remaining_authorized_runtime_seconds=15_000)
        except BudgetRefusal:
            return
        raise AssertionError("unaffordable calibration was not refused")
    r.check("affordability gate: 1.30x + reserves rule passes and refuses "
            "exactly as frozen", affordability_pass_and_refuse)

    def affordability_bypass_attack():
        try:
            affordability_gate(
                projected_calibration_seconds=-5_000,  # forged projection
                verification_transfer_reserve_seconds=0,
                termination_reserve_seconds=0,
                remaining_authorized_runtime_seconds=100)
        except BudgetError:
            return
        raise AssertionError("negative projection bypassed the gate")
    r.check("HOSTILE affordability-gate bypass via forged inputs is refused",
            affordability_bypass_attack)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
