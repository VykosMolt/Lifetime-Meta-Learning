"""Compute ledger arithmetic and the matching comparison (contract §8)."""

from __future__ import annotations

import json

import pytest

from foundation_learner.training import compute_accounting as ca


def _ledger(arm_id: str, updates: int = 3, budget: int = 4096, loss_tokens: int = 100):
    ledger = ca.ComputeLedger(
        arm_id=arm_id, max_tokens_per_batch=budget, planned_updates=updates
    )
    ledger.start()
    for step in range(updates):
        ledger.record_step(
            n_examples=2,
            forward_tokens=1000,
            forward_tokens_unpadded=900,
            loss_tokens=loss_tokens,
            weighted_loss_mass=loss_tokens / 2.0,
            did_update=True,
            step_seconds=0.01,
            episode_ids=[f"{arm_id}-ep{step}", f"{arm_id}-ep{step}"],
        )
    ledger.stop()
    return ledger


def test_ledger_totals_and_schema():
    ledger = _ledger("FL3")
    payload = ledger.to_dict()
    assert payload["schema"] == "flb200.compute_ledger.v1"
    assert payload["optimizer_updates"] == 3
    assert payload["skipped_updates"] == 0
    assert payload["examples"] == 6
    assert payload["episodes"] == 3  # unique ids
    assert payload["forward_tokens"] == 3000
    assert payload["forward_tokens_unpadded"] == 2700
    assert payload["backward_tokens"] == 3000
    assert payload["loss_tokens"] == 300
    assert payload["weighted_loss_mass"] == pytest.approx(150.0)
    assert payload["wall_time_s"] >= 0.0
    assert payload["gpu_seconds"] == 0.0  # cpu device


def test_skipped_updates_are_recorded_not_hidden():
    ledger = ca.ComputeLedger(arm_id="FL3")
    ledger.record_step(
        n_examples=1,
        forward_tokens=10,
        forward_tokens_unpadded=10,
        loss_tokens=2,
        weighted_loss_mass=2.0,
        did_update=False,
        step_seconds=0.0,
    )
    assert ledger.to_dict()["optimizer_updates"] == 0
    assert ledger.to_dict()["skipped_updates"] == 1


def test_gpu_seconds_accumulate_only_on_cuda_devices():
    ledger = ca.ComputeLedger(arm_id="FL3", device="cuda:0")
    ledger.record_step(
        n_examples=1,
        forward_tokens=10,
        forward_tokens_unpadded=10,
        loss_tokens=2,
        weighted_loss_mass=2.0,
        did_update=True,
        step_seconds=1.5,
    )
    assert ledger.to_dict()["gpu_seconds"] == pytest.approx(1.5)


def test_write_and_load_round_trip(tmp_path):
    ledger = _ledger("FL2")
    path = str(tmp_path / "ledger.json")
    ledger.write(path)
    text = open(path, encoding="utf-8").read()
    assert text.endswith("\n")
    assert json.loads(text) == ledger.to_dict()
    assert ca.load_ledger(path)["arm_id"] == "FL2"


def test_matched_core_ledgers_report_intrinsic_differences():
    ledgers = [
        _ledger("FL1", loss_tokens=40).to_dict(),
        _ledger("FL2", loss_tokens=100).to_dict(),
        _ledger("FL3", loss_tokens=25).to_dict(),
    ]
    report = ca.compare_core_ledgers(ledgers)
    assert report["matched"]["optimizer_updates"] == 3
    assert report["matched"]["max_tokens_per_batch"] == 4096
    assert report["reported_differences"]["loss_tokens"] == {
        "FL1": 120,
        "FL2": 300,
        "FL3": 75,
    }


def test_unequal_updates_or_budget_is_refused():
    with pytest.raises(ca.ComputeMismatchError):
        ca.compare_core_ledgers(
            [_ledger("FL1").to_dict(), _ledger("FL3", updates=5).to_dict()]
        )
    with pytest.raises(ca.ComputeMismatchError):
        ca.compare_core_ledgers(
            [_ledger("FL1").to_dict(), _ledger("FL3", budget=2048).to_dict()]
        )
    with pytest.raises(ca.ComputeMismatchError):
        ca.compare_core_ledgers([_ledger("FL1").to_dict()])
    with pytest.raises(ca.ComputeMismatchError):
        ca.compare_core_ledgers(
            [_ledger("FL1").to_dict(), _ledger("FL1").to_dict()]
        )


def test_foreign_schema_is_refused():
    bad = _ledger("FL1").to_dict()
    bad["schema"] = "something.else.v1"
    with pytest.raises(ca.ComputeMismatchError):
        ca.compare_core_ledgers([bad, _ledger("FL3").to_dict()])
