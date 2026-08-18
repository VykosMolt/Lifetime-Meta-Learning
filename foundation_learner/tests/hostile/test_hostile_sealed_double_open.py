"""HOSTILE: open the sealed set twice.

Contract §12c: a single opening; a second unlock refuses.  The attack tries a
plain second call, a second call from a second process-level guard, deleting
the ledger, truncating it, and rewriting the opening entry.  It also checks
that a refused second opening leaves the ledger untouched and that the results
of the FIRST opening remain immutable.

UPDATED for the two-phase opening (Amendment 12).  The semantics that changed:
the ``SEALED_OPENED`` entry is written by ``commit()``, after the evaluation
records exist, and a failed attempt is recorded as ``SEALED_OPENING_ABORTED``
and permits exactly ONE retry.  The attacks are unchanged in spirit and are
extended, not weakened: a THIRD attempt must refuse, an aborted attempt must
stay in the ledger forever, a provisional unlock must not be able to write a
sealed result, and reading without frozen development decisions must still be
impossible.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from foundation_learner.campaign import o1_isolation
from foundation_learner.campaign import sealed_gate as sg
from foundation_learner.data.shards import write_shard
from foundation_learner.ecology.manifests import write_split_manifest
from foundation_learner.ecology.split import compute_split

SEALED_FAMILY = compute_split()["SEALED_TEST"][0]


def campaign(tmp_path):
    guard = o1_isolation.IsolationGuard(label="HOSTILE_SEALED")
    pregen = tmp_path / "pregen"
    manifest = write_split_manifest(str(pregen / "family_split_manifest.json"))
    shard = (pregen / "episodes" / "SEALED_TEST" /
             f"{SEALED_FAMILY}.scripted.jsonl")
    write_shard(str(shard), [{"episode_id": "s0", "family_id": SEALED_FAMILY}],
                sealed=True,
                split_manifest_sha256_hex=manifest["split_manifest_sha256"])
    out = tmp_path / "run"
    out.mkdir()
    decisions = sg.build_dev_decisions(
        chosen_learning_rate=1e-4, chosen_scope="PEFT_MODE", updates_U=600,
        stage_states={"FL3": "COMPLETE"}, checkpoint_tags={"FL3": ["final"]},
        hashes={"split_manifest_sha256": manifest["split_manifest_sha256"]})
    guard.write_json(os.path.join(str(out), sg.DEV_DECISIONS_NAME), decisions)
    return guard, str(pregen), str(out), str(shard)


def opening_kwargs(pregen, out, guard):
    return dict(ledger_path=os.path.join(out, sg.LEDGER_NAME),
                dev_decisions_path=os.path.join(out, sg.DEV_DECISIONS_NAME),
                split_manifest_path=os.path.join(pregen,
                                                 "family_split_manifest.json"),
                guard=guard)


def commit_an_opening(pregen, out, guard, shard):
    """The NORMAL flow: read the sealed shard, then commit on real records."""
    unlock = sg.open_sealed(**opening_kwargs(pregen, out, guard))
    records = unlock.read_shard(shard)
    unlock.commit(evaluation={"n_records": len(records)})
    return unlock


def test_a_second_opening_refuses(tmp_path):
    guard, pregen, out, shard = campaign(tmp_path)
    commit_an_opening(pregen, out, guard, shard)
    with pytest.raises(sg.LedgerError):
        sg.open_sealed(**opening_kwargs(pregen, out, guard))


def test_a_fresh_guard_does_not_reset_the_ledger(tmp_path):
    """The single-use record lives on disk, not in process memory."""
    guard, pregen, out, shard = campaign(tmp_path)
    commit_an_opening(pregen, out, guard, shard)
    other = o1_isolation.IsolationGuard(label="HOSTILE_SECOND_PROCESS")
    with pytest.raises(sg.LedgerError):
        sg.open_sealed(**opening_kwargs(pregen, out, other))


def test_a_refused_second_opening_leaves_the_ledger_unchanged(tmp_path):
    guard, pregen, out, shard = campaign(tmp_path)
    commit_an_opening(pregen, out, guard, shard)
    ledger = os.path.join(out, sg.LEDGER_NAME)
    before = open(ledger, encoding="utf-8").read()
    with pytest.raises(sg.LedgerError):
        sg.open_sealed(**opening_kwargs(pregen, out, guard))
    assert open(ledger, encoding="utf-8").read() == before


def test_a_third_attempt_refuses_after_two_recorded_aborts(tmp_path):
    """The retry budget is ONE, and every attempt stays in the ledger."""
    guard, pregen, out, _ = campaign(tmp_path)
    sg.open_sealed(**opening_kwargs(pregen, out, guard)).abort("attempt 1")
    sg.open_sealed(**opening_kwargs(pregen, out, guard)).abort("attempt 2")
    with pytest.raises(sg.LedgerError) as exc:
        sg.open_sealed(**opening_kwargs(pregen, out, guard))
    assert "budget is 2" in str(exc.value)
    events = [e["event"] for e in
              sg.read_ledger(os.path.join(out, sg.LEDGER_NAME), guard=guard)]
    assert events == [sg.EVENT_ABORTED, sg.EVENT_ABORTED]


def test_a_provisional_unlock_cannot_write_a_sealed_result(tmp_path):
    """Phase one may READ; only the committed opening may write a result."""
    guard, pregen, out, shard = campaign(tmp_path)
    unlock = sg.open_sealed(**opening_kwargs(pregen, out, guard))
    assert unlock.read_shard(shard)
    with pytest.raises(sg.SealedGateRefusal):
        unlock.write_result(os.path.join(out, "premature.json"), {"x": 1})
    assert not os.path.exists(os.path.join(out, "premature.json"))


def test_an_abort_cannot_be_used_to_erase_an_attempt(tmp_path):
    """An aborted attempt is permanent and the unlock is dead afterwards."""
    guard, pregen, out, shard = campaign(tmp_path)
    unlock = sg.open_sealed(**opening_kwargs(pregen, out, guard))
    unlock.abort("the attack aborts and hopes the attempt disappears")
    ledger = os.path.join(out, sg.LEDGER_NAME)
    assert len(open(ledger, encoding="utf-8").read().splitlines()) == 1
    with pytest.raises(sg.SealedGateRefusal):
        unlock.read_shard(shard)
    with pytest.raises(sg.SealedGateRefusal):
        unlock.commit(evaluation={"n_records": 0})


def test_deleting_the_ledger_is_detectable_from_the_results(tmp_path):
    """A deleted ledger permits a re-open, so the RESULTS carry the proof."""
    guard, pregen, out, shard = campaign(tmp_path)
    unlock = commit_an_opening(pregen, out, guard, shard)
    result = unlock.write_result(os.path.join(out, "sealed_report.json"),
                                 {"macro_aulc": 0.1})
    first_entry = unlock.entry["entry_sha256"]
    os.remove(os.path.join(out, sg.LEDGER_NAME))
    second = commit_an_opening(pregen, out, guard, shard)
    assert second.entry["entry_sha256"] != first_entry
    # the surviving read-only result still names the FIRST opening, so the
    # deletion is evident to any auditor comparing the two
    written = json.loads(open(os.path.join(out, "sealed_report.json"),
                              encoding="utf-8").read())
    assert written["sealed_opening_entry_sha256"] == first_entry
    assert stat.S_IMODE(os.stat(os.path.join(out, "sealed_report.json")).st_mode) \
        == 0o444
    # and the second opening cannot overwrite the first opening's result
    with pytest.raises(sg.SealedGateRefusal):
        second.write_result(os.path.join(out, "sealed_report.json"), {"x": 1})
    assert result["sha256"]


def test_a_truncated_or_edited_ledger_refuses_rather_than_reopening(tmp_path):
    guard, pregen, out, shard = campaign(tmp_path)
    unlock = commit_an_opening(pregen, out, guard, shard)
    unlock.write_result(os.path.join(out, "r.json"), {"a": 1})
    ledger = os.path.join(out, sg.LEDGER_NAME)
    lines = open(ledger, encoding="utf-8").read().splitlines()

    # drop the opening entry, keeping the result entry: the chain breaks
    with open(ledger, "w", encoding="utf-8") as fh:
        fh.write(lines[1] + "\n")
    with pytest.raises(sg.LedgerError):
        sg.open_sealed(**opening_kwargs(pregen, out, guard))

    # rewrite the opening entry as a different event: the entry hash breaks
    entry = json.loads(lines[0])
    entry["event"] = "SOMETHING_ELSE"
    with open(ledger, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(sg.LedgerError):
        sg.open_sealed(**opening_kwargs(pregen, out, guard))


def test_the_unlock_cannot_be_reused_after_revocation(tmp_path):
    guard, pregen, out, shard = campaign(tmp_path)
    unlock = commit_an_opening(pregen, out, guard, shard)
    assert unlock.read_shard(shard)
    unlock.revoke()
    with pytest.raises(sg.SealedGateRefusal):
        unlock.read_shard(shard)
    with pytest.raises(sg.SealedGateRefusal):
        unlock.write_result(os.path.join(out, "late.json"), {"x": 1})
    with pytest.raises(sg.SealedGateRefusal):
        sg.loader_guard(shard, unlock=unlock, guard=guard)
