"""HOSTILE: make the Foundation Learner touch O1 material.

A FAKE accelerator layout is built in a temporary directory (``/outputs``,
``/opt/o1_b200``, ``/workspace/o1_calibration``, the shared read-only
checkpoint) and every plausible route into it is attempted: direct read, direct
write, symlink, relative escape, a scheduler journal aimed at an O1 root, a
result archive that includes O1 files, and a shard loader pointed at O1 output.
"""
from __future__ import annotations

import json
import os

import pytest

from foundation_learner.campaign import o1_isolation as iso
from foundation_learner.campaign import result_verifier, scheduler as sched
from foundation_learner.campaign import sealed_gate


def fake_pod(tmp_path):
    layout = {
        "outputs": tmp_path / "outputs",
        "opt_o1": tmp_path / "opt" / "o1_b200",
        "axis": tmp_path / "artifacts" / "AXIS_PACKAGE_V2",
        "cohorts": tmp_path / "artifacts" / "COHORTS",
        "policies": tmp_path / "artifacts" / "policies",
        "calibration": tmp_path / "workspace" / "o1_calibration",
        "checkpoint": tmp_path / "artifacts" / "ouro_rltt_local",
        "fl": tmp_path / "workspace" / "foundation_learner",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    seeds = tmp_path / "artifacts" / "calibration_seed_matrix.json"
    seeds.write_text("{}\n", encoding="utf-8")
    binding = tmp_path / "artifacts" / "TOKENIZER_BINDING.json"
    binding.write_text("{}\n", encoding="utf-8")
    (layout["outputs"] / "records.jsonl").write_text('{"o1": 1}\n',
                                                     encoding="utf-8")
    (layout["calibration"] / "analysis.json").write_text('{"o1": 2}\n',
                                                         encoding="utf-8")
    forbidden = [str(layout[k]) for k in
                 ("outputs", "opt_o1", "axis", "cohorts", "policies",
                  "calibration")] + [str(seeds)]
    guard = iso.IsolationGuard(
        forbidden_roots=forbidden,
        shared_readonly=[str(layout["checkpoint"]), str(binding)],
        label="HOSTILE_FL")
    layout["seeds"] = seeds
    layout["binding"] = binding
    return guard, layout


def test_every_forbidden_root_refuses_read_and_write(tmp_path):
    guard, layout = fake_pod(tmp_path)
    targets = [layout["outputs"] / "records.jsonl",
               layout["calibration"] / "analysis.json",
               layout["opt_o1"] / "runner" / "records.py",
               layout["axis"], layout["cohorts"], layout["policies"],
               layout["seeds"]]
    for target in targets:
        for mode in (iso.MODE_READ, iso.MODE_WRITE):
            with pytest.raises(iso.O1IsolationError):
                guard.guard(str(target), mode)


def test_a_symlink_and_a_relative_escape_are_both_resolved(tmp_path):
    guard, layout = fake_pod(tmp_path)
    link = layout["fl"] / "harmless.jsonl"
    os.symlink(layout["outputs"] / "records.jsonl", link)
    with pytest.raises(iso.O1IsolationError):
        guard.guard(str(link), iso.MODE_READ)
    escape = os.path.join(str(layout["fl"]), "..", "..", "outputs",
                          "records.jsonl")
    with pytest.raises(iso.O1IsolationError):
        guard.guard(escape, iso.MODE_READ)


def test_the_shared_checkpoint_is_read_only(tmp_path):
    guard, layout = fake_pod(tmp_path)
    assert guard.guard(str(layout["checkpoint"] / "config.json"), iso.MODE_READ)
    with pytest.raises(iso.O1IsolationError):
        guard.guard(str(layout["checkpoint"] / "config.json"), iso.MODE_WRITE)
    with pytest.raises(iso.O1IsolationError):
        guard.write_text(str(layout["binding"]), "tampered")


def test_the_scheduler_cannot_journal_into_an_o1_root(tmp_path):
    guard, layout = fake_pod(tmp_path)
    with pytest.raises(iso.O1IsolationError):
        sched.Scheduler(available_foundation_learner_seconds=1000.0,
                        out_dir=str(layout["outputs"] / "fl_journal"),
                        guard=guard)


def test_a_transfer_archive_cannot_include_o1_files(tmp_path):
    guard, layout = fake_pod(tmp_path)
    run = layout["fl"] / "run"
    run.mkdir()
    (run / "a.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(iso.O1IsolationError):
        result_verifier.deterministic_zip(
            [str(layout["outputs"] / "records.jsonl")], str(tmp_path),
            str(run / "x.zip"), guard=guard)
    # a symlinked O1 file inside the FL tree is caught by the guarded walk
    os.symlink(layout["outputs"], run / "o1_link")
    with pytest.raises(iso.O1IsolationError):
        result_verifier.build_result_manifest(str(run), guard=guard)


def test_the_loader_guard_applies_isolation_before_anything_else(tmp_path):
    guard, layout = fake_pod(tmp_path)
    with pytest.raises(iso.O1IsolationError):
        sealed_gate.loader_guard(str(layout["outputs"] / "records.jsonl"),
                                 guard=guard)


def test_discovered_o1_result_roots_are_refused_too(tmp_path):
    guard, layout = fake_pod(tmp_path)
    late = tmp_path / "workspace" / "o1_results_late"
    late.mkdir(parents=True)
    manifest = layout["fl"] / "o1_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "o1b200.transfer_manifest.v1",
        "artifacts": {"late": {"path": str(late), "kind": "tree",
                               "sha256": "0" * 64}}}), encoding="utf-8")
    assert guard.guard(str(late / "x"), iso.MODE_READ)     # not yet known
    guard.discover_from_o1_manifests([str(manifest)])
    with pytest.raises(iso.O1IsolationError):
        guard.guard(str(late / "x"), iso.MODE_READ)


def test_fl_owned_paths_still_work(tmp_path):
    guard, layout = fake_pod(tmp_path)
    target = layout["fl"] / "runs" / "r.json"
    guard.write_json(str(target), {"ok": True})
    assert guard.read_json(str(target)) == {"ok": True}


def test_every_campaign_module_routes_file_access_through_the_guard():
    """No campaign module opens a path it has not guarded.

    ``open(`` is permitted only where the path came from ``guard(...)`` or from
    a guarded helper; the check below is deliberately blunt: every ``open(``
    in the campaign package must sit in the same function as a ``guard``/
    ``loader_guard``/``_accept`` call.
    """
    import ast

    campaign = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(iso.__file__)))),
        "foundation_learner", "campaign")
    # o1_isolation.py IS the guard.  hf_transfer.py is a standalone
    # byte-transfer helper that runs in its OWN subprocess (so the
    # supervisor can stay HF-offline-locked); it never imports the campaign
    # package and only ever receives paths the parent already guarded —
    # proven directly by test_hf_store_guards_every_path_before_transfer
    # below, which is a stronger assertion than this file-level scan.
    exempt = {"o1_isolation.py", "hf_transfer.py"}
    offenders = []
    for name in sorted(os.listdir(campaign)):
        if not name.endswith(".py") or name in exempt:
            continue
        tree = ast.parse(open(os.path.join(campaign, name), encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            body = ast.dump(node)
            opens = body.count("Name(id='open'")
            guarded = ("guard" in body) or ("_accept" in body)
            if opens and not guarded:
                offenders.append(f"{name}:{node.name}")
    assert offenders == [], f"unguarded file access: {offenders}"


def test_hf_store_guards_every_path_before_transfer():
    """The subprocess transfer helper is exempt from the file-level scan, so
    prove the exemption precisely: _HfStore must route every local path
    through the isolation guard BEFORE handing it to the helper, and must
    never hand the helper an unguarded path."""
    from foundation_learner.campaign.durability import _HfStore

    guarded = []
    handed = []

    def guard(path, mode="read"):
        guarded.append((os.path.abspath(path), mode))
        return path

    def runner(args):
        for flag in ("--local",):
            if flag in args:
                handed.append(os.path.abspath(args[args.index(flag) + 1]))
        if args[0] == "list":
            return {"files": []}
        return {"sha256": _digest_of(handed[-1])}

    import hashlib

    def _digest_of(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read())
        return h.hexdigest()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "journal.jsonl")
        with open(local, "wb") as fh:
            fh.write(b"line\n")
        store = _HfStore("ns/repo", guard=guard, runner=runner)
        store.push(local, "fl_durable/journal.jsonl")
        store.fetch("fl_durable/journal.jsonl", local)
        store.list_all()

    assert handed, "the helper received no path"
    guarded_paths = {p for p, _ in guarded}
    for path in handed:
        assert path in guarded_paths, (
            f"{path} reached the transfer helper without passing the "
            f"isolation guard")
    assert ("read" in {m for p, m in guarded if p == handed[0]}
            or "write" in {m for p, m in guarded})


def test_hf_store_list_never_exposes_non_fl_filenames():
    """Prefix filtering happens inside the helper subprocess, so a shared
    repository's O1 filenames never enter the FL process."""
    from foundation_learner.campaign.durability import _PREFIX, _HfStore

    seen = {}

    def runner(args):
        seen["prefix"] = args[args.index("--prefix") + 1]
        return {"files": [f"{_PREFIX}/session_journal.jsonl"]}

    store = _HfStore("ns/repo", runner=runner)
    out = store.list_all()
    assert seen["prefix"] == _PREFIX, (
        "list must filter server-side/in-helper by the FL prefix")
    assert all(f.startswith(_PREFIX) for f in out)
