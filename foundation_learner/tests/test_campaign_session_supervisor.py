"""The combined-session state machine (contract §13, §22).

The O1 phase is a stub subprocess (a real ``python -c`` command) writing real
completion markers and a real hash manifest, so ``VERIFY_O1_RECORDS`` verifies
genuine digests.  The FL ladder is stubbed here; the real ladder is walked by
the dress rehearsal.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from foundation_learner.campaign import o1_isolation
from foundation_learner.campaign import session_supervisor as ss
from foundation_learner.ecology.base import sha256_file, sha256_tree

STUB = (
    "import hashlib,json,os,sys;"
    "root=sys.argv[1];"
    "os.makedirs(root, exist_ok=True);"
    "open(os.path.join(root,'o1_records.jsonl'),'w').write('{\"row\": 0}\\n');"
    "open(os.path.join(root,'O1_COMPLETE'),'w').write('done\\n');"
    "d=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest();"
    "json.dump({'schema':'o1b200.transfer_manifest.v1','artifacts':{"
    "'records':{'path':'o1_records.jsonl','kind':'file',"
    "'sha256':d(os.path.join(root,'o1_records.jsonl'))},"
    "'marker':{'path':'O1_COMPLETE','kind':'file',"
    "'sha256':d(os.path.join(root,'O1_COMPLETE'))}}},"
    "open(os.path.join(root,'TRANSFER_MANIFEST.json'),'w'))"
)


def fixtures(tmp_path, **overrides):
    o1_root = tmp_path / "o1_calibration"
    o1_root.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "ouro"}\n',
                                            encoding="utf-8")
    payload = {
        "schema": ss.SESSION_CONFIG_SCHEMA,
        "session_id": "TEST_0001",
        "label": "TEST",
        "rehearsal": True,
        "o1_entry_command": [sys.executable, "-c", STUB, str(o1_root)],
        "o1_completion_markers": [str(o1_root / "O1_COMPLETE")],
        "o1_hash_manifests": [str(o1_root / "TRANSFER_MANIFEST.json")],
        "o1_transfer_command": [sys.executable, "-c", "print('transferred')"],
        "o1_close_command": [sys.executable, "-c", "print('closed')"],
        "o1_roots": [str(o1_root)],
        "checkpoint_dir": str(checkpoint),
        "checkpoint_tree_sha256": sha256_tree(str(checkpoint)),
        "pregen_root": str(tmp_path / "pregen"),
        "fl_out_dir": str(tmp_path / "session"),
        "session_authorized_seconds": 3600.0,
        "terminate_command": [sys.executable, "-c", "print('terminated')"],
    }
    payload.update(overrides)
    path = tmp_path / "session_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")
    return str(path), str(o1_root), str(checkpoint)


def supervisor(tmp_path, config_path, **kwargs):
    guard = o1_isolation.IsolationGuard(label="TEST_SESSION")
    ladder_calls = []

    def ladder_runner(scheduler, ctx):
        ladder_calls.append(scheduler.available_foundation_learner_seconds)
        os.makedirs(scheduler.out_dir, exist_ok=True)
        with open(os.path.join(scheduler.out_dir, "stub_result.json"), "w",
                  encoding="utf-8") as fh:
            fh.write('{"stub": true}\n')
        return {"states": {"BENCH": "COMPLETE"}}

    sup = ss.SessionSupervisor(
        config=ss.SessionConfig.load(config_path, guard=guard),
        out_dir=str(tmp_path / "session"), guard=guard,
        ladder_runner=kwargs.pop("ladder_runner", ladder_runner),
        context_factory=kwargs.pop(
            "context_factory",
            lambda sup: __import__(
                "foundation_learner.campaign.stage_definitions",
                fromlist=["StageContext"]).StageContext(
                    out_dir=os.path.join(sup.out_dir, "ladder"),
                    pregen_root=sup.payload["pregen_root"],
                    bundle_factory=lambda: None, guard=sup.guard)),
        **kwargs)
    return sup, ladder_calls


# ---------------- configuration ----------------

def test_states_are_the_frozen_sequence():
    assert ss.STATES == (
        "START_SESSION", "RUN_O1_CALIBRATION", "O1_HALT_OR_COMPLETE",
        "VERIFY_O1_RECORDS", "TRANSFER_O1_RECORDS", "CLOSE_O1_PROCESS",
        "RELOAD_PRISTINE_OURO", "COMPUTE_REMAINING_AUTHORIZED_TIME",
        "RUN_FL_LADDER", "CHECKPOINT_AND_VERIFY", "TRANSFER_FL_ARTIFACTS",
        "TERMINATE_ACCELERATOR")


def test_unresolved_fields_refuse_a_real_session(tmp_path):
    path, _, _ = fixtures(
        tmp_path, rehearsal=False,
        o1_entry_command="UNRESOLVED_OPERATOR_BOUND",
        fl_transfer_command=[sys.executable, "-c", "print('t')"],
        terminate_command=[sys.executable, "-c", "print('x')"])
    with pytest.raises(ss.SessionConfigError) as exc:
        ss.SessionConfig.load(path).validate()
    assert "operator-bound" in str(exc.value)


def test_unresolved_fields_are_allowed_in_a_labelled_rehearsal(tmp_path):
    path, _, _ = fixtures(tmp_path, rehearsal=True,
                          o1_close_command="UNRESOLVED_SOMETHING")
    payload = ss.SessionConfig.load(path).validate()
    assert payload["_unresolved_fields"] == ["o1_close_command"]


def test_missing_fields_and_wrong_schema_are_refused(tmp_path):
    path, _, _ = fixtures(tmp_path)
    payload = json.loads(open(path, encoding="utf-8").read())
    payload.pop("o1_roots")
    open(path, "w", encoding="utf-8").write(json.dumps(payload))
    with pytest.raises(ss.SessionConfigError):
        ss.SessionConfig.load(path).validate()
    payload["schema"] = "something.else.v1"
    open(path, "w", encoding="utf-8").write(json.dumps(payload))
    with pytest.raises(ss.SessionConfigError):
        ss.SessionConfig.load(path).validate()


# ---------------- the O1 custodian ----------------

def test_the_custodian_refuses_paths_outside_the_o1_roots(tmp_path):
    custodian = ss.O1RecordCustodian([str(tmp_path / "o1")])
    (tmp_path / "o1").mkdir()
    (tmp_path / "elsewhere.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ss.SupervisorError):
        custodian.digest(str(tmp_path / "elsewhere.json"))


def test_the_custodian_has_no_content_returning_method():
    api = {name for name in dir(ss.O1RecordCustodian) if not name.startswith("_")}
    assert api == {"digest", "exists", "manifest_entries", "verify_manifest"}
    source = open(ss.__file__, encoding="utf-8").read()
    custodian_src = source.split("class O1RecordCustodian")[1].split(
        "\n# ---")[0]
    # the only file reads are the manifest parse and the hashing helpers
    assert custodian_src.count("open(") == 1
    assert "read_text" not in custodian_src


def test_the_custodian_verifies_a_real_o1_manifest(tmp_path):
    path, o1_root, _ = fixtures(tmp_path)
    import subprocess
    subprocess.run([sys.executable, "-c", STUB, o1_root], check=True)
    custodian = ss.O1RecordCustodian([o1_root])
    report = custodian.verify_manifest(
        os.path.join(o1_root, "TRANSFER_MANIFEST.json"))
    assert report["ok"] and report["checked"] == 2
    with open(os.path.join(o1_root, "o1_records.jsonl"), "a",
              encoding="utf-8") as fh:
        fh.write('{"row": 1}\n')
    assert custodian.verify_manifest(
        os.path.join(o1_root, "TRANSFER_MANIFEST.json"))["ok"] is False


def test_the_custodian_reads_sha256sums_format(tmp_path):
    o1_root = tmp_path / "o1"
    o1_root.mkdir()
    (o1_root / "a.json").write_text("{}\n", encoding="utf-8")
    digest = sha256_file(str(o1_root / "a.json"))
    (o1_root / "SHA256SUMS").write_text(f"{digest}  a.json\n", encoding="utf-8")
    custodian = ss.O1RecordCustodian([str(o1_root)])
    assert custodian.verify_manifest(str(o1_root / "SHA256SUMS"))["ok"]


# ---------------- the machine ----------------

def test_a_full_session_walks_every_state(tmp_path):
    path, o1_root, _ = fixtures(tmp_path)
    sup, ladder_calls = supervisor(tmp_path, path)
    status = sup.run(resume=False)
    assert status["outcome"] == "COMPLETE", status
    assert status["states_completed"] == list(ss.STATES)
    assert ladder_calls and ladder_calls[0] <= 3600.0
    assert os.path.isfile(os.path.join(sup.out_dir, ss.O1_TRANSFER_RECEIPT))
    assert os.path.isfile(os.path.join(sup.out_dir, ss.O1_CLOSE_RECEIPT))
    assert os.path.isfile(os.path.join(sup.out_dir, "FL_ARTIFACTS.zip"))
    assert os.path.isfile(os.path.join(sup.out_dir,
                                       "SESSION_FINAL_STATUS.json"))


def test_the_o1_roots_are_added_to_the_isolation_guard(tmp_path):
    path, o1_root, _ = fixtures(tmp_path)
    sup, _ = supervisor(tmp_path, path)
    with pytest.raises(o1_isolation.O1IsolationError):
        sup.guard.guard(os.path.join(o1_root, "o1_records.jsonl"),
                        o1_isolation.MODE_READ)


def test_journal_records_every_transition_and_supports_resume(tmp_path):
    path, _, _ = fixtures(tmp_path)
    sup, _ = supervisor(tmp_path, path)
    sup.run(resume=False)
    events = [(r["event"], r["state"]) for r in sup.read_journal()]
    for state in ss.STATES:
        assert ("STATE_STARTED", state) in events
        assert ("STATE_COMPLETED", state) in events
    resumed, ladder_calls = supervisor(tmp_path, path)
    assert set(resumed.completed) == set(ss.STATES)
    resumed.run(resume=True)
    assert ladder_calls == []          # nothing re-ran
    skipped = [r for r in resumed.read_journal()
               if r["event"] == "STATE_SKIPPED_RESUMED"]
    assert len(skipped) == len(ss.STATES)


def test_a_failed_o1_verification_aborts_before_fl(tmp_path):
    path, o1_root, _ = fixtures(tmp_path)
    sup, ladder_calls = supervisor(tmp_path, path)

    original = sup.custodian.verify_manifest
    sup.custodian.verify_manifest = lambda m: dict(
        original(m), ok=False, mismatched=["x"])
    status = sup.run(resume=False)
    assert status["outcome"] == "ABORTED_AT_VERIFY_O1_RECORDS"
    assert ladder_calls == []
    assert "RUN_FL_LADDER" in status["states_never_started"]


def test_checkpoint_tree_hash_mismatch_refuses(tmp_path):
    path, _, checkpoint = fixtures(tmp_path)
    payload = json.loads(open(path, encoding="utf-8").read())
    payload["checkpoint_tree_sha256"] = "b" * 64
    open(path, "w", encoding="utf-8").write(json.dumps(payload))
    sup, ladder_calls = supervisor(tmp_path, path)
    status = sup.run(resume=False)
    assert status["outcome"] == "ABORTED_AT_RELOAD_PRISTINE_OURO"
    assert ladder_calls == []


def test_a_real_session_requires_transfer_and_termination_commands(tmp_path):
    path, _, _ = fixtures(tmp_path, rehearsal=False)
    with pytest.raises(ss.SessionConfigError) as exc:
        ss.SessionConfig.load(path).validate()
    assert "fl_transfer_command" in str(exc.value)


def test_a_halted_o1_phase_aborts_before_fl(tmp_path):
    path, o1_root, _ = fixtures(tmp_path,
                                o1_entry_command=[sys.executable, "-c",
                                                  "raise SystemExit(3)"])
    sup, ladder_calls = supervisor(tmp_path, path)
    status = sup.run(resume=False)
    assert status["outcome"] == "ABORTED_AT_O1_HALT_OR_COMPLETE"
    assert "O1_HALT" in status["failure"]
    assert ladder_calls == []


def test_a_real_session_requires_the_frozen_backbone_hash(tmp_path):
    from foundation_learner.training import model_loading

    path, _, _ = fixtures(
        tmp_path, rehearsal=False,
        fl_transfer_command=[sys.executable, "-c", "print('t')"],
        terminate_command=[sys.executable, "-c", "print('x')"])
    sup, _ = supervisor(tmp_path, path)
    sup.completed = ["TRANSFER_O1_RECORDS", "CLOSE_O1_PROCESS"]
    with pytest.raises(ss.SupervisorError) as exc:
        sup.state_RELOAD_PRISTINE_OURO()
    assert "frozen §1 backbone hash" in str(exc.value)
    assert model_loading.FROZEN_CHECKPOINT_TREE_SHA256.startswith("a701f7a7")


def test_available_time_is_what_remains_after_o1(tmp_path):
    path, _, _ = fixtures(tmp_path)
    sup, _ = supervisor(tmp_path, path)
    sup._t0 = sup.clock.monotonic() - 600.0
    record = sup.state_COMPUTE_REMAINING_AUTHORIZED_TIME()
    assert 2990.0 <= record["available_foundation_learner_seconds"] <= 3000.0
