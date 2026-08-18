"""Run-output verification and the deterministic transfer archive (§19)."""
from __future__ import annotations

import os
import zipfile

import pytest

from foundation_learner.campaign import o1_isolation
from foundation_learner.campaign import result_verifier as rv


def tree(tmp_path):
    root = tmp_path / "run"
    (root / "fl3").mkdir(parents=True)
    (root / "fl3" / "arm_result.json").write_text('{"a": 1}\n', encoding="utf-8")
    (root / "records.jsonl").write_text('{"r": 1}\n{"r": 2}\n', encoding="utf-8")
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_text("junk", encoding="utf-8")
    return o1_isolation.IsolationGuard(label="TEST"), str(root)


def test_tree_listing_excludes_bytecode(tmp_path):
    guard, root = tree(tmp_path)
    rels = rv.iter_tree_files(root, guard=guard)
    assert rels == ["fl3/arm_result.json", "records.jsonl"]


def test_manifest_then_verification_round_trips(tmp_path):
    guard, root = tree(tmp_path)
    manifest = rv.build_result_manifest(root, guard=guard)
    assert manifest["n_files"] == 2
    assert rv.verify_result_tree(root, manifest, guard=guard)["ok"] is True


def test_a_modified_file_fails_verification(tmp_path):
    guard, root = tree(tmp_path)
    manifest = rv.build_result_manifest(root, guard=guard)
    with open(os.path.join(root, "records.jsonl"), "a", encoding="utf-8") as fh:
        fh.write('{"r": 3}\n')
    with pytest.raises(rv.TransferError) as exc:
        rv.verify_result_tree(root, manifest, guard=guard)
    assert "hash mismatches" in str(exc.value)


def test_a_missing_or_extra_file_fails_verification(tmp_path):
    guard, root = tree(tmp_path)
    manifest = rv.build_result_manifest(root, guard=guard)
    os.remove(os.path.join(root, "records.jsonl"))
    with pytest.raises(rv.TransferError):
        rv.verify_result_tree(root, manifest, guard=guard)
    guard2, root2 = tree(tmp_path / "b")
    manifest2 = rv.build_result_manifest(root2, guard=guard2)
    open(os.path.join(root2, "surprise.json"), "w").write("{}")
    with pytest.raises(rv.TransferError):
        rv.verify_result_tree(root2, manifest2, guard=guard2)


def test_read_only_sealed_results_still_hash(tmp_path):
    guard, root = tree(tmp_path)
    sealed = os.path.join(root, "sealed_report.json")
    with open(sealed, "w", encoding="utf-8") as fh:
        fh.write("{}\n")
    os.chmod(sealed, 0o444)
    manifest = rv.build_result_manifest(root, guard=guard)
    assert manifest["files"]["sealed_report.json"]["mode"] == "0o444"
    assert rv.verify_result_tree(root, manifest, guard=guard)["ok"]


def test_the_zip_is_byte_deterministic(tmp_path):
    guard, root = tree(tmp_path)
    paths = [os.path.join(root, r) for r in rv.iter_tree_files(root, guard=guard)]
    a = rv.deterministic_zip(paths, root, str(tmp_path / "a.zip"), guard=guard)
    b = rv.deterministic_zip(list(reversed(paths)), root,
                             str(tmp_path / "b.zip"), guard=guard)
    assert a == b
    with zipfile.ZipFile(str(tmp_path / "a.zip")) as zf:
        infos = zf.infolist()
    assert [i.filename for i in infos] == sorted(i.filename for i in infos)
    for info in infos:
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.external_attr == 0o644 << 16


def test_secret_names_are_refused(tmp_path):
    guard, root = tree(tmp_path)
    secret = os.path.join(root, "deploy.pem")
    open(secret, "w").write("x")
    with pytest.raises(rv.TransferError) as exc:
        rv.deterministic_zip([secret], root, str(tmp_path / "s.zip"), guard=guard)
    assert "secret pattern" in str(exc.value)
    # tokenizer files are model assets, not credentials
    tok = os.path.join(root, "tokenizer.json")
    open(tok, "w").write("{}")
    rv.check_no_secrets(tok)


def test_sha256sums_round_trip_and_detect_edits(tmp_path):
    guard, root = tree(tmp_path)
    paths = [os.path.join(root, r) for r in rv.iter_tree_files(root, guard=guard)]
    sums = os.path.join(root, "SHA256SUMS")
    rv.write_sha256sums(paths, root, sums, guard=guard)
    text = open(sums, encoding="utf-8").read()
    assert "  fl3/arm_result.json" in text
    assert rv.verify_sha256sums(sums, root, guard=guard)["checked"] == 2
    with open(os.path.join(root, "records.jsonl"), "w", encoding="utf-8") as fh:
        fh.write("tampered\n")
    with pytest.raises(rv.TransferError):
        rv.verify_sha256sums(sums, root, guard=guard)


def test_build_transfer_archive_writes_a_sidecar(tmp_path):
    guard, root = tree(tmp_path)
    out = str(tmp_path / "FL_ARTIFACTS.zip")
    record = rv.build_transfer_archive(root, out, guard=guard)
    assert os.path.isfile(out)
    sidecar = open(out + ".sha256", encoding="utf-8").read()
    assert sidecar.startswith(record["zip_sha256"])
    assert sidecar.endswith("FL_ARTIFACTS.zip\n")


def test_an_archive_reaching_into_an_o1_root_is_refused(tmp_path):
    o1_root = tmp_path / "outputs"
    o1_root.mkdir()
    (o1_root / "o1.json").write_text("{}", encoding="utf-8")
    guard = o1_isolation.IsolationGuard(forbidden_roots=[str(o1_root)],
                                        shared_readonly=[], label="TEST")
    root = tmp_path / "run"
    root.mkdir()
    (root / "a.json").write_text("{}", encoding="utf-8")
    with pytest.raises(o1_isolation.O1IsolationError):
        rv.deterministic_zip([str(o1_root / "o1.json")], str(tmp_path),
                             str(tmp_path / "x.zip"), guard=guard)
