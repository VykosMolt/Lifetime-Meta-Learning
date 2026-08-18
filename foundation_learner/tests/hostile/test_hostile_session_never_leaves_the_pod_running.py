"""HOSTILE: a failed session leaves a rented accelerator billing.

Attack: make a state fail anywhere in the machine (an O1 halt, a bad checkpoint
hash, an exploding ladder, an interrupt) and check whether the supervisor still
terminates the accelerator.  Before the repair, ``run()`` simply ``break``-ed
out of the state loop, so ``TERMINATE_ACCELERATOR`` never ran on ANY failure
path: the pod kept billing until a human noticed.

Second attack: crash mid-ladder and resume.  Before the repair, resume marked
``COMPUTE_REMAINING_AUTHORIZED_TIME`` as "already completed" without restoring
its result, so ``RUN_FL_LADDER`` refused for want of an FL allowance — a crash
silently cost the whole FL half of the session.

Contract §11 (the reserve is never consumed), §13 (the state machine), §20.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from foundation_learner.campaign import o1_isolation
from foundation_learner.campaign import session_supervisor as ss
from foundation_learner.ecology.base import sha256_tree

STUB = (
    "import hashlib,json,os,sys;"
    "root=sys.argv[1];"
    "os.makedirs(root, exist_ok=True);"
    "open(os.path.join(root,'o1_records.jsonl'),'w').write('{\"row\": 0}\\n');"
    "open(os.path.join(root,'O1_COMPLETE'),'w').write('done\\n');"
    "d=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest();"
    "json.dump({'schema':'o1b200.transfer_manifest.v1','artifacts':{"
    "'records':{'path':'o1_records.jsonl','kind':'file',"
    "'sha256':d(os.path.join(root,'o1_records.jsonl'))}}},"
    "open(os.path.join(root,'TRANSFER_MANIFEST.json'),'w'))"
)


def fixtures(tmp_path, marker_path, **overrides):
    o1_root = tmp_path / "o1_calibration"
    o1_root.mkdir(exist_ok=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text('{"model_type": "ouro"}\n',
                                            encoding="utf-8")
    terminate = [sys.executable, "-c",
                 "import sys; open(sys.argv[1], 'a').write('TERMINATED\\n')",
                 str(marker_path)]
    payload = {
        "schema": ss.SESSION_CONFIG_SCHEMA,
        "session_id": "HOSTILE_TERMINATE",
        "label": "HOSTILE",
        "rehearsal": True,
        "o1_entry_command": [sys.executable, "-c", STUB, str(o1_root)],
        "o1_completion_markers": [str(o1_root / "O1_COMPLETE")],
        "o1_hash_manifests": [str(o1_root / "TRANSFER_MANIFEST.json")],
        "o1_transfer_command": [sys.executable, "-c", "print('t')"],
        "o1_close_command": [sys.executable, "-c", "print('c')"],
        "o1_roots": [str(o1_root)],
        "checkpoint_dir": str(checkpoint),
        "checkpoint_tree_sha256": sha256_tree(str(checkpoint)),
        "pregen_root": str(tmp_path / "pregen"),
        "fl_out_dir": str(tmp_path / "session"),
        "session_authorized_seconds": 3600.0,
        "fl_transfer_command": [sys.executable, "-c", "print('ft')"],
        "terminate_command": terminate,
    }
    payload.update(overrides)
    path = tmp_path / "session_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")
    return str(path)


def make_supervisor(tmp_path, config_path, *, ladder_runner=None,
                    context_factory=None):
    guard = o1_isolation.IsolationGuard(label="HOSTILE_TERMINATE")
    from foundation_learner.campaign.stage_definitions import StageContext

    def default_context(sup):
        return StageContext(out_dir=os.path.join(sup.out_dir, "ladder"),
                            pregen_root=sup.payload["pregen_root"],
                            bundle_factory=lambda: None, guard=sup.guard)

    def default_ladder(scheduler, ctx):
        os.makedirs(scheduler.out_dir, exist_ok=True)
        with open(os.path.join(scheduler.out_dir, "stub.json"), "w",
                  encoding="utf-8") as fh:
            fh.write('{"stub": true}\n')
        return {"states": {"BENCH": "COMPLETE"}}

    return ss.SessionSupervisor(
        config=ss.SessionConfig.load(config_path, guard=guard),
        out_dir=str(tmp_path / "session"), guard=guard,
        ladder_runner=ladder_runner or default_ladder,
        context_factory=context_factory or default_context)


def terminated(marker_path) -> int:
    if not os.path.exists(marker_path):
        return 0
    return len([l for l in open(marker_path, encoding="utf-8") if l.strip()])


# ---------------- termination on every failure path ----------------

@pytest.mark.parametrize("break_it,expected_state", [
    ("o1_halt", "O1_HALT_OR_COMPLETE"),
    ("bad_checkpoint", "RELOAD_PRISTINE_OURO"),
    ("ladder_explodes", "RUN_FL_LADDER"),
])
def test_every_failure_path_still_terminates_the_accelerator(
        tmp_path, break_it, expected_state):
    marker = tmp_path / "TERMINATED.log"
    overrides = {}
    if break_it == "o1_halt":
        overrides["o1_entry_command"] = [sys.executable, "-c",
                                         "raise SystemExit(3)"]
    if break_it == "bad_checkpoint":
        overrides["checkpoint_tree_sha256"] = "b" * 64
    path = fixtures(tmp_path, marker, **overrides)

    def exploding_ladder(scheduler, ctx):
        raise RuntimeError("the ladder exploded mid-run")

    sup = make_supervisor(
        tmp_path, path,
        ladder_runner=exploding_ladder if break_it == "ladder_explodes" else None)
    status = sup.run(resume=False)

    assert status["outcome"] == f"ABORTED_AT_{expected_state}", status
    assert terminated(marker) == 1, (
        "REFUSED: the session aborted without terminating the accelerator; a "
        "rented pod keeps billing until it is terminated")
    assert status["close_out"]["terminate"]["status"] == "COMPLETED"
    events = [(r["event"], r["state"]) for r in sup.read_journal()]
    assert ("EMERGENCY_STATE_COMPLETED", "TERMINATE_ACCELERATOR") in events


def test_an_interrupt_inside_a_state_still_terminates(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)

    def interrupting_ladder(scheduler, ctx):
        raise KeyboardInterrupt("operator interrupt")

    sup = make_supervisor(tmp_path, path, ladder_runner=interrupting_ladder)
    status = sup.run(resume=False)
    assert status["outcome"] == "ABORTED_AT_RUN_FL_LADDER"
    assert terminated(marker) == 1


def test_a_successful_session_terminates_exactly_once(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    sup = make_supervisor(tmp_path, path)
    status = sup.run(resume=False)
    assert status["outcome"] == "COMPLETE"
    assert terminated(marker) == 1
    assert status["close_out"].get("note")


def test_a_failed_transfer_does_not_prevent_termination(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker,
                    fl_transfer_command=[sys.executable, "-c",
                                         "raise SystemExit(9)"])

    def exploding_ladder(scheduler, ctx):
        os.makedirs(scheduler.out_dir, exist_ok=True)
        with open(os.path.join(scheduler.out_dir, "partial.json"), "w") as fh:
            fh.write("{}\n")
        raise RuntimeError("boom")

    sup = make_supervisor(tmp_path, path, ladder_runner=exploding_ladder)
    status = sup.run(resume=False)
    assert status["close_out"]["transfer"]["status"] == "FAILED"
    assert status["close_out"]["terminate"]["status"] == "COMPLETED"
    assert terminated(marker) == 1


# ---------------- mid-ladder crash -> resume ----------------

def test_a_mid_ladder_crash_resumes_and_completes_the_remaining_states(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    calls: list[float] = []

    def crashing_ladder(scheduler, ctx):
        raise RuntimeError("simulated mid-ladder crash")

    first = make_supervisor(tmp_path, path, ladder_runner=crashing_ladder)
    status = first.run(resume=False)
    assert status["outcome"] == "ABORTED_AT_RUN_FL_LADDER"
    assert "COMPUTE_REMAINING_AUTHORIZED_TIME" in status["states_completed"]
    assert terminated(marker) == 1              # the crash still terminated

    def resuming_ladder(scheduler, ctx):
        calls.append(scheduler.available_foundation_learner_seconds)
        os.makedirs(scheduler.out_dir, exist_ok=True)
        with open(os.path.join(scheduler.out_dir, "resumed.json"), "w") as fh:
            fh.write('{"resumed": true}\n')
        return {"states": {"BENCH": "COMPLETE"}}

    second = make_supervisor(tmp_path, path, ladder_runner=resuming_ladder)
    status = second.run(resume=True)

    assert status["outcome"] == "COMPLETE", status["failure"]
    assert calls, (
        "REFUSED: the resumed session never ran the ladder; state_results were "
        "not rebuilt from the journal")
    assert 0.0 < calls[0] <= 3600.0
    for state in ss.STATES:
        assert state in status["states_completed"]
    rebuilt = [r for r in second.read_journal()
               if r["event"] == "STATE_RESULTS_REBUILT"]
    assert rebuilt and "COMPUTE_REMAINING_AUTHORIZED_TIME" in \
        rebuilt[-1]["restored_states"]


def test_the_resumed_allowance_is_never_larger_than_the_journalled_one(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)

    def crashing_ladder(scheduler, ctx):
        raise RuntimeError("crash")

    first = make_supervisor(tmp_path, path, ladder_runner=crashing_ladder)
    first.run(resume=False)
    journalled = first.state_results["COMPUTE_REMAINING_AUTHORIZED_TIME"][
        "available_foundation_learner_seconds"]

    second = make_supervisor(tmp_path, path)
    report = second.rebuild_state_results()
    assert report["available_foundation_learner_seconds"] <= journalled
    assert report["resume_wall_clock_gap_seconds"] >= 0.0
