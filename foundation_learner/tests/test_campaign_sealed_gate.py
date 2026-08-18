"""The sealed gate (contract §12 + Amendment 1).

These tests use REAL enciphered shards written by the real ``data/shards.py``
writer with the real split-manifest digest, so the gate is exercised against
genuine sealed bytes rather than a stand-in.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from foundation_learner.campaign import o1_isolation, sealed_gate as sg
from foundation_learner.data.shards import read_shard, write_shard
from foundation_learner.ecology.manifests import write_split_manifest
from foundation_learner.ecology.split import compute_split

SEALED_FAMILY = compute_split()["SEALED_TEST"][0]
RECORDS = [{"episode_id": "e0", "family_id": SEALED_FAMILY, "x": 1},
           {"episode_id": "e1", "family_id": SEALED_FAMILY, "x": 2}]


def campaign_dir(tmp_path):
    """A pregen root with a real split manifest and one real sealed shard."""
    guard = o1_isolation.IsolationGuard(label="TEST_SEALED")
    pregen = tmp_path / "pregen"
    manifest = write_split_manifest(str(pregen / "family_split_manifest.json"))
    shard = pregen / "episodes" / "SEALED_TEST" / f"{SEALED_FAMILY}.scripted.jsonl"
    write_shard(str(shard), RECORDS, sealed=True,
                split_manifest_sha256_hex=manifest["split_manifest_sha256"])
    out = tmp_path / "run"
    out.mkdir()
    return guard, str(pregen), str(out), str(shard), manifest


def frozen_decisions(guard, out_dir):
    payload = sg.build_dev_decisions(
        chosen_learning_rate=1e-4, chosen_scope="PEFT_MODE", updates_U=600,
        stage_states={"FL3": "COMPLETE"}, checkpoint_tags={"FL3": ["final"]},
        hashes={"split_manifest_sha256": "0" * 64})
    path = os.path.join(out_dir, sg.DEV_DECISIONS_NAME)
    guard.write_json(path, payload)
    return path


def open_provisional(tmp_path):
    """Phase ONE only: nothing is written to the ledger yet."""
    guard, pregen, out, shard, manifest = campaign_dir(tmp_path)
    decisions = frozen_decisions(guard, out)
    unlock = sg.open_sealed(
        ledger_path=os.path.join(out, sg.LEDGER_NAME),
        dev_decisions_path=decisions,
        split_manifest_path=os.path.join(pregen, "family_split_manifest.json"),
        guard=guard)
    return guard, pregen, out, shard, unlock


def open_once(tmp_path):
    """Both phases: read the sealed shard, then COMMIT (the normal flow)."""
    guard, pregen, out, shard, unlock = open_provisional(tmp_path)
    records = unlock.read_shard(shard)
    unlock.commit(evaluation={"n_records": len(records)})
    return guard, pregen, out, shard, unlock


def reopen_kwargs(pregen, out, guard):
    return dict(ledger_path=os.path.join(out, sg.LEDGER_NAME),
                dev_decisions_path=os.path.join(out, sg.DEV_DECISIONS_NAME),
                split_manifest_path=os.path.join(pregen,
                                                 "family_split_manifest.json"),
                guard=guard)


# ---------------- the shard really is enciphered ----------------

def test_a_sealed_shard_is_not_readable_as_plain_text(tmp_path):
    _, _, _, shard, _ = campaign_dir(tmp_path)
    raw = open(shard, "rb").read()
    assert b"episode_id" not in raw
    with pytest.raises(ValueError):
        read_shard(shard)


# ---------------- loader-side refusal ----------------

def test_loader_guard_refuses_a_sealed_path_without_an_unlock(tmp_path):
    guard, _, _, shard, _ = campaign_dir(tmp_path)
    with pytest.raises(sg.SealedGateRefusal) as exc:
        sg.loader_guard(shard, unlock=None, guard=guard)
    assert "SEALED_TEST" in str(exc.value)


def test_sealed_path_recognition_is_conservative(tmp_path):
    assert sg.is_sealed_path(f"/x/episodes/SEALED_TEST/{SEALED_FAMILY}.jsonl")
    assert sg.is_sealed_path(f"/x/anywhere/{SEALED_FAMILY}.scripted.jsonl")
    dev_family = compute_split()["DEVELOPMENT"][0]
    assert not sg.is_sealed_path(f"/x/episodes/DEVELOPMENT/{dev_family}.jsonl")


# ---------------- the prerequisite ----------------

def test_opening_refuses_without_frozen_dev_decisions(tmp_path):
    guard, pregen, out, _, _ = campaign_dir(tmp_path)
    with pytest.raises(sg.DevDecisionsError):
        sg.open_sealed(
            ledger_path=os.path.join(out, sg.LEDGER_NAME),
            dev_decisions_path=os.path.join(out, sg.DEV_DECISIONS_NAME),
            split_manifest_path=os.path.join(pregen,
                                             "family_split_manifest.json"),
            guard=guard)
    assert not os.path.exists(os.path.join(out, sg.LEDGER_NAME))


def test_edited_dev_decisions_are_refused(tmp_path):
    guard, pregen, out, _, _ = campaign_dir(tmp_path)
    path = frozen_decisions(guard, out)
    payload = json.loads(open(path, encoding="utf-8").read())
    payload["chosen_learning_rate"] = 3e-4          # edited after freezing
    guard.write_json(path, payload)
    with pytest.raises(sg.DevDecisionsError):
        sg.read_dev_decisions(path, guard=guard)


def test_incomplete_dev_decisions_are_refused(tmp_path):
    guard, _, out, _, _ = campaign_dir(tmp_path)
    payload = sg.build_dev_decisions(
        chosen_learning_rate=1e-4, chosen_scope="PEFT_MODE", updates_U=600,
        stage_states={}, checkpoint_tags={}, hashes={"a": "b"})
    with pytest.raises(sg.DevDecisionsError):
        sg.validate_dev_decisions(payload)


# ---------------- opening, reading, ledgering ----------------

def test_a_valid_opening_reads_the_sealed_records_and_ledgers_it(tmp_path):
    guard, _, out, shard, unlock = open_once(tmp_path)
    assert unlock.read_shard(shard) == RECORDS
    entries = sg.read_ledger(os.path.join(out, sg.LEDGER_NAME), guard=guard)
    assert [e["event"] for e in entries] == [sg.EVENT_OPENED]
    assert entries[0]["dev_decisions_sha256"] == unlock.dev_decisions_sha256
    assert entries[0]["prev_sha256"] == "0" * 64
    assert entries[0]["stage_states"] == {"FL3": "COMPLETE"}
    assert "utc" in entries[0]
    # the committed entry proves the EVIDENCE existed first
    assert entries[0]["records_read"] == len(RECORDS)
    assert entries[0]["attempt_index"] == 0
    assert entries[0]["evaluation"]["n_records"] == len(RECORDS)


# ---------------- the two-phase opening (contract §12c + Amendment 12) ------

def test_phase_one_writes_nothing_to_the_ledger(tmp_path):
    guard, _, out, shard, unlock = open_provisional(tmp_path)
    assert unlock.committed is False
    assert not os.path.exists(os.path.join(out, sg.LEDGER_NAME))
    assert unlock.read_shard(shard) == RECORDS      # reading is allowed
    assert not os.path.exists(os.path.join(out, sg.LEDGER_NAME))


def test_a_provisional_unlock_may_not_write_a_result(tmp_path):
    _, _, out, _, unlock = open_provisional(tmp_path)
    with pytest.raises(sg.SealedGateRefusal) as exc:
        unlock.write_result(os.path.join(out, "r.json"), {"a": 1})
    assert "COMMITTED" in str(exc.value)
    with pytest.raises(sg.SealedGateRefusal):
        unlock.write_result_records(os.path.join(out, "r.jsonl"), [{"a": 1}])


def test_an_aborted_attempt_is_ledgered_and_leaves_exactly_one_retry(tmp_path):
    guard, pregen, out, shard, unlock = open_provisional(tmp_path)
    unlock.abort("simulated infrastructure failure")
    entries = sg.read_ledger(os.path.join(out, sg.LEDGER_NAME), guard=guard)
    assert [e["event"] for e in entries] == [sg.EVENT_ABORTED]
    assert entries[0]["retries_remaining"] == 1
    # the aborted unlock is dead
    with pytest.raises(sg.SealedGateRefusal):
        unlock.read_shard(shard)

    retry = sg.open_sealed(**reopen_kwargs(pregen, out, guard))
    assert retry.attempt_index == 1 and retry.prior_aborted == 1
    retry.read_shard(shard)
    retry.commit(evaluation={"n_records": len(RECORDS)})
    events = [e["event"] for e in
              sg.read_ledger(os.path.join(out, sg.LEDGER_NAME), guard=guard)]
    assert events == [sg.EVENT_ABORTED, sg.EVENT_OPENED]


def test_a_third_attempt_refuses_and_both_attempts_stay_recorded(tmp_path):
    guard, pregen, out, _, unlock = open_provisional(tmp_path)
    unlock.abort("attempt 1 failed")
    second = sg.open_sealed(**reopen_kwargs(pregen, out, guard))
    second.abort("attempt 2 failed")
    with pytest.raises(sg.LedgerError) as exc:
        sg.open_sealed(**reopen_kwargs(pregen, out, guard))
    assert "budget is 2" in str(exc.value)
    events = [e["event"] for e in
              sg.read_ledger(os.path.join(out, sg.LEDGER_NAME), guard=guard)]
    assert events == [sg.EVENT_ABORTED, sg.EVENT_ABORTED]


def test_the_context_manager_aborts_on_an_exception_and_reraises(tmp_path):
    guard, pregen, out, shard, _ = campaign_dir(tmp_path)
    frozen_decisions(guard, out)
    with pytest.raises(RuntimeError, match="boom"):
        with sg.sealed_opening(**reopen_kwargs(pregen, out, guard)) as unlock:
            unlock.read_shard(shard)
            raise RuntimeError("boom")
    entries = sg.read_ledger(os.path.join(out, sg.LEDGER_NAME), guard=guard)
    assert [e["event"] for e in entries] == [sg.EVENT_ABORTED]
    assert "boom" in entries[0]["reason"]


def test_the_context_manager_commits_on_success(tmp_path):
    guard, pregen, out, shard, _ = campaign_dir(tmp_path)
    frozen_decisions(guard, out)
    with sg.sealed_opening(**reopen_kwargs(pregen, out, guard)) as unlock:
        unlock.read_shard(shard)
        unlock.commit(evaluation={"n_records": len(RECORDS)})
        unlock.write_result(os.path.join(out, "sealed.json"), {"aulc": 0.1})
    entries = sg.read_ledger(os.path.join(out, sg.LEDGER_NAME), guard=guard)
    assert [e["event"] for e in entries] == [sg.EVENT_OPENED, sg.EVENT_RESULT]


def test_a_body_that_never_commits_is_recorded_as_an_aborted_attempt(tmp_path):
    guard, pregen, out, shard, _ = campaign_dir(tmp_path)
    frozen_decisions(guard, out)
    with sg.sealed_opening(**reopen_kwargs(pregen, out, guard)) as unlock:
        unlock.read_shard(shard)            # produced nothing, committed nothing
    entries = sg.read_ledger(os.path.join(out, sg.LEDGER_NAME), guard=guard)
    assert [e["event"] for e in entries] == [sg.EVENT_ABORTED]


def test_a_second_opening_refuses(tmp_path):
    guard, pregen, out, _, _ = open_once(tmp_path)
    with pytest.raises(sg.LedgerError) as exc:
        sg.open_sealed(**reopen_kwargs(pregen, out, guard))
    assert "already been opened" in str(exc.value)


def test_a_committed_opening_cannot_be_aborted(tmp_path):
    _, _, _, _, unlock = open_once(tmp_path)
    with pytest.raises(sg.SealedGateRefusal):
        unlock.abort("too late")


def test_a_revoked_unlock_cannot_read(tmp_path):
    _, _, _, shard, unlock = open_once(tmp_path)
    unlock.revoke()
    with pytest.raises(sg.SealedGateRefusal):
        unlock.read_shard(shard)


# ---------------- immutable results ----------------

def test_sealed_results_are_read_only_and_hash_ledgered(tmp_path):
    guard, _, out, _, unlock = open_once(tmp_path)
    path = os.path.join(out, "sealed_eval_report.json")
    record = unlock.write_result(path, {"macro_aulc": 0.42})
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o444
    entries = sg.read_ledger(os.path.join(out, sg.LEDGER_NAME), guard=guard)
    assert entries[-1]["event"] == sg.EVENT_RESULT
    assert entries[-1]["result_sha256"] == record["sha256"]
    assert entries[-1]["prev_sha256"] == entries[-2]["entry_sha256"]
    written = json.loads(open(path, encoding="utf-8").read())
    assert written["dev_decisions_sha256"] == unlock.dev_decisions_sha256
    with pytest.raises(sg.SealedGateRefusal):
        unlock.write_result(path, {"macro_aulc": 0.99})


# ---------------- the ledger is tamper-evident ----------------

def test_a_truncated_ledger_is_detected(tmp_path):
    guard, _, out, _, unlock = open_once(tmp_path)
    unlock.write_result(os.path.join(out, "r.json"), {"a": 1})
    ledger = os.path.join(out, sg.LEDGER_NAME)
    lines = open(ledger, encoding="utf-8").read().splitlines()
    with open(ledger, "w", encoding="utf-8") as fh:
        fh.write(lines[1] + "\n")          # first entry removed
    with pytest.raises(sg.LedgerError):
        sg.read_ledger(ledger, guard=guard)


def test_an_edited_ledger_entry_is_detected(tmp_path):
    guard, _, out, _, unlock = open_once(tmp_path)
    ledger = os.path.join(out, sg.LEDGER_NAME)
    entry = json.loads(open(ledger, encoding="utf-8").read().splitlines()[0])
    entry["stage_states"] = {"FL3": "TAMPERED"}
    with open(ledger, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(sg.LedgerError):
        sg.read_ledger(ledger, guard=guard)


# ---------------- the gate is the only decipher path ----------------

def test_no_other_campaign_module_derives_the_sealed_key():
    campaign = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "campaign")
    offenders = []
    for name in sorted(os.listdir(campaign)):
        if not name.endswith(".py") or name == "sealed_gate.py":
            continue
        text = open(os.path.join(campaign, name), encoding="utf-8").read()
        for token in ("unseal_bytes", "sealed_key("):
            if token in text:
                offenders.append((name, token))
    assert offenders == [], f"decipher path outside sealed_gate.py: {offenders}"
