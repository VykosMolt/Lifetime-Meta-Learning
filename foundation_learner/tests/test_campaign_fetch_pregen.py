"""Pod-side pregen ingestion: the corpus arrives, or the session refuses."""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from foundation_learner.campaign import fetch_pregen as fp


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_tree(root: str, *, sealed_shard: bool = True) -> dict:
    """A minimal but structurally faithful pregen tree."""
    os.makedirs(os.path.join(root, "MANIFESTS"), exist_ok=True)
    os.makedirs(os.path.join(root, "episodes", "TRAIN"), exist_ok=True)
    os.makedirs(os.path.join(root, "episodes", "SEALED_TEST"), exist_ok=True)
    shards = []
    entries = [("episodes/TRAIN/family_a.jsonl", b'{"a": 1}\n', False)]
    if sealed_shard:
        entries.append(
            ("episodes/SEALED_TEST/family_a.scripted.jsonl", b"\x00sealed", True))
    for rel, payload, sealed in entries:
        path = os.path.join(root, rel)
        with open(path, "wb") as fh:
            fh.write(payload)
        shards.append({
            "path": rel, "bytes": len(payload), "records": 1,
            "sealed": sealed, "sha256": _sha(path),
            "plaintext_sha256": None if sealed else _sha(path),
        })
    with open(os.path.join(root, fp.SHARD_SUMS_REL), "w", encoding="utf-8") as fh:
        json.dump({"schema": "fl.shard_sums.v1", "shards": shards}, fh)
    with open(os.path.join(root, fp.PREGEN_MANIFEST_REL), "w",
              encoding="utf-8") as fh:
        json.dump({"schema": "fl.pregen_manifest.v1"}, fh)
    return {"shards": shards}


# ---------------------------------------------------------------- verification


def test_a_complete_tree_verifies(tmp_path):
    root = str(tmp_path / "pregen")
    build_tree(root)
    stats = fp.verify_tree(root)
    assert stats["shards_verified"] == 2
    assert stats["sealed_shards"] == 1


def test_a_sealed_shard_is_verified_without_being_opened(tmp_path, monkeypatch):
    """SEALED_TEST must never be read as episodes merely to check it."""
    root = str(tmp_path / "pregen")
    build_tree(root)
    sealed = os.path.join(root, "episodes/SEALED_TEST/family_a.scripted.jsonl")
    real_open = open
    opened: list[str] = []

    def watching_open(path, *a, **kw):
        opened.append(str(path))
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", watching_open)
    fp.verify_tree(root)
    # it is hashed (so it is opened in binary), but never decoded as records:
    # the manifest's plaintext_sha256 is None and nothing consults it.
    assert sealed in opened
    sums = json.load(real_open(os.path.join(root, fp.SHARD_SUMS_REL)))
    sealed_entry = [s for s in sums["shards"] if s["sealed"]][0]
    assert sealed_entry["plaintext_sha256"] is None


def test_a_missing_shard_refuses(tmp_path):
    root = str(tmp_path / "pregen")
    build_tree(root)
    os.remove(os.path.join(root, "episodes/TRAIN/family_a.jsonl"))
    with pytest.raises(fp.PregenFetchError, match="missing"):
        fp.verify_tree(root)


def test_a_corrupt_shard_refuses(tmp_path):
    root = str(tmp_path / "pregen")
    build_tree(root)
    with open(os.path.join(root, "episodes/TRAIN/family_a.jsonl"), "wb") as fh:
        fh.write(b'{"a": 2}\n')          # same length, different bytes
    with pytest.raises(fp.PregenFetchError, match="corrupt"):
        fp.verify_tree(root)


def test_absent_shard_sums_refuses(tmp_path):
    root = str(tmp_path / "pregen")
    build_tree(root)
    os.remove(os.path.join(root, fp.SHARD_SUMS_REL))
    with pytest.raises(fp.PregenFetchError, match="SHARD_SUMS"):
        fp.verify_tree(root)


def test_a_foreign_schema_refuses(tmp_path):
    root = str(tmp_path / "pregen")
    build_tree(root)
    path = os.path.join(root, fp.SHARD_SUMS_REL)
    with open(path, encoding="utf-8") as fh:
        sums = json.load(fh)
    sums["schema"] = "fl.shard_sums.v2"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sums, fh)
    with pytest.raises(fp.PregenFetchError, match="fl.shard_sums.v1"):
        fp.verify_tree(root)


def test_an_absent_pregen_manifest_refuses(tmp_path):
    root = str(tmp_path / "pregen")
    build_tree(root)
    os.remove(os.path.join(root, fp.PREGEN_MANIFEST_REL))
    with pytest.raises(fp.PregenFetchError, match="PREGEN_MANIFEST"):
        fp.verify_tree(root)


# ---------------------------------------------------------------------- source


def test_source_uri_parsing():
    assert fp.parse_source("hf://ns/repo") == ("ns/repo", fp.DEFAULT_REMOTE_PREFIX)
    assert fp.parse_source("hf://ns/repo/some/prefix") == ("ns/repo", "some/prefix")


@pytest.mark.parametrize("uri", ["", "s3://ns/repo", "hf://ns", "hf:///repo"])
def test_a_non_hf_source_refuses(uri):
    with pytest.raises(fp.PregenFetchError):
        fp.parse_source(uri)


# ----------------------------------------------------------------------- fetch


def test_an_already_present_tree_is_still_verified(tmp_path):
    root = str(tmp_path / "pregen")
    build_tree(root)

    def refuse(_args):
        raise AssertionError("must not fetch when the tree is already present")

    report = fp.fetch("hf://ns/repo", root, runner=refuse)
    assert report["action"] == "already_present"
    assert report["shards_verified"] == 2


def test_an_already_present_but_corrupt_tree_refuses(tmp_path):
    root = str(tmp_path / "pregen")
    build_tree(root)
    with open(os.path.join(root, "episodes/TRAIN/family_a.jsonl"), "wb") as fh:
        fh.write(b'{"a": 9}\n')
    with pytest.raises(fp.PregenFetchError):
        fp.fetch("hf://ns/repo", root, runner=lambda _a: {})


def test_fetch_publishes_only_after_verification(tmp_path):
    root = str(tmp_path / "pregen")

    def fake(args):
        local = args[args.index("--local") + 1]
        prefix = args[args.index("--prefix") + 1]
        build_tree(os.path.join(local, prefix))
        return {"count": 2}

    report = fp.fetch("hf://ns/repo", root, runner=fake)
    assert report["action"] == "fetched"
    assert report["repo"] == "ns/repo"
    assert os.path.isdir(root)
    assert not os.path.exists(root + ".incoming")
    fp.verify_tree(root)


def test_a_corrupt_fetch_leaves_the_root_absent(tmp_path):
    root = str(tmp_path / "pregen")

    def fake_corrupt(args):
        local = args[args.index("--local") + 1]
        prefix = args[args.index("--prefix") + 1]
        tree = os.path.join(local, prefix)
        build_tree(tree)
        os.remove(os.path.join(tree, "episodes/TRAIN/family_a.jsonl"))
        return {"count": 1}

    with pytest.raises(fp.PregenFetchError):
        fp.fetch("hf://ns/repo", root, runner=fake_corrupt)
    # the session must not find a half-fetched tree it would treat as valid
    assert not os.path.exists(root)
    assert os.path.isdir(root + ".incoming")


def test_a_fetch_that_delivers_nothing_refuses(tmp_path):
    root = str(tmp_path / "pregen")
    with pytest.raises(fp.PregenFetchError, match="absent from the staging"):
        fp.fetch("hf://ns/repo", root, runner=lambda _a: {"count": 0})


# ------------------------------------------------------------------------ main


def test_main_refuses_an_absent_tree_with_no_source(tmp_path, capsys, monkeypatch):
    root = str(tmp_path / "pregen")
    monkeypatch.setattr(
        "sys.argv", ["fetch_pregen", "--pregen-root", root, "--source", ""])
    assert fp.main() == 2
    assert "FL_PREGEN_SOURCE is" in capsys.readouterr().err


def test_main_refuses_an_unresolved_source(tmp_path, capsys, monkeypatch):
    root = str(tmp_path / "pregen")
    monkeypatch.setattr("sys.argv", [
        "fetch_pregen", "--pregen-root", root,
        "--source", "UNRESOLVED_OPERATOR_BOUND (hf://ns/repo)"])
    assert fp.main() == 2
    assert "FL_PREGEN_SOURCE is" in capsys.readouterr().err


def test_main_takes_the_pregen_root_from_the_session_config(tmp_path, monkeypatch):
    root = str(tmp_path / "pregen")
    build_tree(root)
    config = str(tmp_path / "session.json")
    with open(config, "w", encoding="utf-8") as fh:
        json.dump({"schema": "flb200.session_config.v1", "pregen_root": root}, fh)
    out = str(tmp_path / "report.json")
    monkeypatch.setattr(
        "sys.argv", ["fetch_pregen", "--config", config, "--out", out])
    assert fp.main() == 0
    with open(out, encoding="utf-8") as fh:
        report = json.load(fh)
    assert report["pregen_root"] == root
    assert report["shards_verified"] == 2
