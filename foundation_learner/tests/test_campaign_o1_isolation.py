"""O1 isolation guard (contract §13, §22).

Every check here runs against a FAKE root layout in a temporary directory, so
the fixtures exercise the real containment logic without depending on an
accelerator filesystem.  The frozen default list is additionally checked
against the campaign manifest template, so code and manifest cannot drift.
"""
from __future__ import annotations

import json
import os

import pytest

from foundation_learner.campaign import o1_isolation as iso

MANIFEST_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "FOUNDATION_LEARNER_V0_MANIFEST.template.json")


def fake_guard(tmp_path):
    o1 = tmp_path / "outputs"
    opt = tmp_path / "opt" / "o1_b200"
    shared = tmp_path / "artifacts" / "ouro_rltt_local"
    binding = tmp_path / "artifacts" / "TOKENIZER_BINDING.json"
    for d in (o1, opt, shared):
        d.mkdir(parents=True, exist_ok=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    binding.write_text("{}\n", encoding="utf-8")
    guard = iso.IsolationGuard(forbidden_roots=[str(o1), str(opt)],
                               shared_readonly=[str(shared), str(binding)],
                               label="TEST")
    return guard, {"o1": o1, "opt": opt, "shared": shared, "binding": binding}


def test_frozen_roots_match_the_manifest_template():
    with open(MANIFEST_TEMPLATE, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert tuple(manifest["isolation"]["o1_forbidden_roots"]) == \
        iso.FROZEN_FORBIDDEN_ROOTS
    assert tuple(manifest["isolation"]["shared_readonly_inputs"]) == \
        iso.SHARED_READONLY_INPUTS


def test_read_and_write_under_a_forbidden_root_are_refused(tmp_path):
    guard, paths = fake_guard(tmp_path)
    for mode in (iso.MODE_READ, iso.MODE_WRITE):
        with pytest.raises(iso.O1IsolationError):
            guard.guard(str(paths["o1"] / "records.jsonl"), mode)
        with pytest.raises(iso.O1IsolationError):
            guard.guard(str(paths["opt"]), mode)
    assert len(guard.refusals) == 4


def test_shared_inputs_are_readable_but_not_writable(tmp_path):
    guard, paths = fake_guard(tmp_path)
    assert guard.guard(str(paths["shared"] / "config.json"), iso.MODE_READ)
    with pytest.raises(iso.O1IsolationError) as exc:
        guard.guard(str(paths["shared"] / "config.json"), iso.MODE_WRITE)
    assert "READ-ONLY" in str(exc.value)
    with pytest.raises(iso.O1IsolationError):
        guard.guard(str(paths["binding"]), iso.MODE_WRITE)


def test_symlink_into_a_forbidden_root_is_resolved_and_refused(tmp_path):
    guard, paths = fake_guard(tmp_path)
    (paths["o1"] / "secret.json").write_text("{}\n", encoding="utf-8")
    link = tmp_path / "fl_workspace" / "innocent.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(paths["o1"] / "secret.json", link)
    with pytest.raises(iso.O1IsolationError):
        guard.guard(str(link), iso.MODE_READ)


def test_fl_owned_paths_are_allowed(tmp_path):
    guard, _ = fake_guard(tmp_path)
    target = tmp_path / "workspace" / "foundation_learner" / "run" / "x.json"
    guard.write_json(str(target), {"ok": True})
    assert json.loads(guard.read_text(str(target)))["ok"] is True


def test_discovery_only_widens_the_refusal_set(tmp_path):
    guard, paths = fake_guard(tmp_path)
    before = set(guard.forbidden_roots)
    extra = tmp_path / "workspace" / "o1_calibration"
    extra.mkdir(parents=True)
    manifest = tmp_path / "TRANSFER_MANIFEST.json"
    manifest.write_text(json.dumps({
        "schema": "o1b200.transfer_manifest.v1",
        "artifacts": {"results": {"path": str(extra), "kind": "tree",
                                  "sha256": "0" * 64}}}), encoding="utf-8")
    report = guard.discover_from_o1_manifests([str(manifest)])
    assert os.path.realpath(str(extra)) in report["added_roots"]
    assert before.issubset(set(guard.forbidden_roots))
    with pytest.raises(iso.O1IsolationError):
        guard.guard(str(extra / "anything"), iso.MODE_READ)


def test_missing_o1_manifest_is_recorded_not_silently_ignored(tmp_path):
    guard, _ = fake_guard(tmp_path)
    report = guard.discover_from_o1_manifests([str(tmp_path / "nope.json")])
    assert report["manifests"][0]["status"] == "UNREADABLE"
    assert report["added_roots"] == []


def test_atomic_write_and_append_are_fsynced(tmp_path):
    guard, _ = fake_guard(tmp_path)
    path = tmp_path / "fl" / "journal.jsonl"
    guard.append_line(str(path), '{"a": 1}')
    guard.append_line(str(path), '{"a": 2}')
    lines = guard.read_text(str(path)).splitlines()
    assert lines == ['{"a": 1}', '{"a": 2}']
    assert not os.path.exists(str(path) + ".tmp")


def test_walk_guards_every_visited_directory(tmp_path):
    guard, paths = fake_guard(tmp_path)
    root = tmp_path / "fl_tree"
    (root / "a").mkdir(parents=True)
    (root / "a" / "f.txt").write_text("x", encoding="utf-8")
    seen = [d for d, _, _ in guard.walk(str(root))]
    assert len(seen) == 2
    os.symlink(paths["o1"], root / "sneaky")
    with pytest.raises(iso.O1IsolationError):
        list(guard.walk(str(root)))
