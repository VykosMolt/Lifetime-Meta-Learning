"""HOSTILE: start the FL ladder before the O1 records are transferred.

Contract §13: FL never starts before the O1 artefacts are transferred and
verified, and never from an O1-mutated process state.  The attack jumps
straight to ``RUN_FL_LADDER``, forges the journal, and deletes the receipt.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from foundation_learner.campaign import o1_isolation
from foundation_learner.campaign import session_supervisor as ss
from foundation_learner.ecology.base import sha256_tree

LADDER_CALLS: list[float] = []


def config(tmp_path):
    o1_root = tmp_path / "o1_calibration"
    o1_root.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    payload = {
        "schema": ss.SESSION_CONFIG_SCHEMA,
        "session_id": "HOSTILE_0001",
        "rehearsal": True,
        "o1_entry_command": [sys.executable, "-c", "pass"],
        "o1_completion_markers": [str(o1_root / "O1_COMPLETE")],
        "o1_hash_manifests": [str(o1_root / "TRANSFER_MANIFEST.json")],
        "o1_transfer_command": [sys.executable, "-c", "pass"],
        "o1_roots": [str(o1_root)],
        "checkpoint_dir": str(checkpoint),
        "checkpoint_tree_sha256": sha256_tree(str(checkpoint)),
        "pregen_root": str(tmp_path / "pregen"),
        "fl_out_dir": str(tmp_path / "session"),
        "session_authorized_seconds": 3600.0,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def supervisor(tmp_path):
    LADDER_CALLS.clear()
    guard = o1_isolation.IsolationGuard(label="HOSTILE")

    def ladder_runner(scheduler, ctx):
        LADDER_CALLS.append(scheduler.available_foundation_learner_seconds)
        return {"states": {}}

    return ss.SessionSupervisor(
        config=ss.SessionConfig.load(config(tmp_path), guard=guard),
        out_dir=str(tmp_path / "session"), guard=guard,
        ladder_runner=ladder_runner,
        context_factory=lambda sup: None)


def test_jumping_straight_to_the_fl_ladder_is_refused(tmp_path):
    sup = supervisor(tmp_path)
    with pytest.raises(ss.SupervisorError) as exc:
        sup.state_RUN_FL_LADDER()
    assert "TRANSFER_O1_RECORDS" in str(exc.value)
    assert LADDER_CALLS == []


def test_a_forged_journal_without_the_receipt_is_refused(tmp_path):
    sup = supervisor(tmp_path)
    sup.completed = ["TRANSFER_O1_RECORDS", "CLOSE_O1_PROCESS",
                     "RELOAD_PRISTINE_OURO"]
    sup.state_results["available_foundation_learner_seconds"] = 3000.0
    with pytest.raises(ss.SupervisorError) as exc:
        sup.state_RUN_FL_LADDER()
    assert ss.O1_TRANSFER_RECEIPT in str(exc.value)
    assert LADDER_CALLS == []


def test_a_receipt_without_the_close_and_reload_states_is_refused(tmp_path):
    sup = supervisor(tmp_path)
    sup.guard.write_json(os.path.join(sup.out_dir, ss.O1_TRANSFER_RECEIPT),
                         {"forged": True})
    sup.completed = ["TRANSFER_O1_RECORDS"]
    sup.state_results["available_foundation_learner_seconds"] = 3000.0
    with pytest.raises(ss.SupervisorError) as exc:
        sup.state_RUN_FL_LADDER()
    assert "CLOSE_O1_PROCESS" in str(exc.value)
    assert LADDER_CALLS == []


def test_the_ladder_refuses_without_the_computed_remaining_time(tmp_path):
    sup = supervisor(tmp_path)
    sup.guard.write_json(os.path.join(sup.out_dir, ss.O1_TRANSFER_RECEIPT),
                         {"ok": True})
    sup.completed = ["TRANSFER_O1_RECORDS", "CLOSE_O1_PROCESS",
                     "RELOAD_PRISTINE_OURO"]
    with pytest.raises(ss.SupervisorError) as exc:
        sup.state_RUN_FL_LADDER()
    assert "COMPUTE_REMAINING_AUTHORIZED_TIME" in str(exc.value)
    assert LADDER_CALLS == []


def test_close_refuses_before_the_transfer(tmp_path):
    sup = supervisor(tmp_path)
    with pytest.raises(ss.SupervisorError):
        sup.state_CLOSE_O1_PROCESS()


def test_reload_refuses_before_the_o1_process_is_closed(tmp_path):
    sup = supervisor(tmp_path)
    sup.completed = ["TRANSFER_O1_RECORDS"]
    with pytest.raises(ss.SupervisorError) as exc:
        sup.state_RELOAD_PRISTINE_OURO()
    assert "CLOSE_O1_PROCESS" in str(exc.value)


def test_the_state_order_itself_is_frozen():
    order = list(ss.STATES)
    assert order.index("VERIFY_O1_RECORDS") < order.index("TRANSFER_O1_RECORDS")
    assert order.index("TRANSFER_O1_RECORDS") < order.index("CLOSE_O1_PROCESS")
    assert order.index("CLOSE_O1_PROCESS") < order.index("RELOAD_PRISTINE_OURO")
    assert order.index("RELOAD_PRISTINE_OURO") < order.index("RUN_FL_LADDER")
    assert order.index("RUN_FL_LADDER") < order.index("TRANSFER_FL_ARTIFACTS")
    assert order[-1] == "TERMINATE_ACCELERATOR"
