"""HOSTILE: force the development selector to read a SEALED_TEST shard.

The attack is the realistic one: an operator (or a buggy caller) hands the
selector a sealed shard path, or hands it a sealed shard renamed to look like a
development file.  Both must be refused, and no sealed record may reach the
selection rule.
"""
from __future__ import annotations

import os

import pytest

from foundation_learner.campaign import dev_selector as ds
from foundation_learner.campaign import o1_isolation, sealed_gate
from foundation_learner.campaign.promotion import DevMetrics, SealedEvidenceRefused
from foundation_learner.data.shards import write_shard
from foundation_learner.ecology.manifests import write_split_manifest
from foundation_learner.ecology.split import compute_split

SEALED_FAMILY = compute_split()["SEALED_TEST"][0]
DEV_FAMILY = compute_split()["DEVELOPMENT"][0]


def sealed_shard(tmp_path, name=None):
    manifest = write_split_manifest(
        str(tmp_path / "pregen" / "family_split_manifest.json"))
    path = (tmp_path / "pregen" / "episodes" / "SEALED_TEST" /
            (name or f"{SEALED_FAMILY}.scripted.jsonl"))
    write_shard(str(path), [{"episode_id": "s0", "family_id": SEALED_FAMILY,
                             "split": "SEALED_TEST"}],
                sealed=True,
                split_manifest_sha256_hex=manifest["split_manifest_sha256"])
    return str(path)


def test_the_selector_refuses_a_sealed_shard_path(tmp_path):
    guard = o1_isolation.IsolationGuard(label="HOSTILE")
    path = sealed_shard(tmp_path)
    with pytest.raises(sealed_gate.SealedGateRefusal) as exc:
        ds.load_dev_episodes([path], guard=guard)
    assert "SEALED_TEST" in str(exc.value)


def test_a_sealed_shard_hidden_under_a_development_name_still_fails(tmp_path):
    """Renaming defeats the path heuristic — the CIPHER then defeats the read."""
    guard = o1_isolation.IsolationGuard(label="HOSTILE")
    manifest = write_split_manifest(
        str(tmp_path / "pregen" / "family_split_manifest.json"))
    disguised = (tmp_path / "pregen" / "episodes" / "DEVELOPMENT" /
                 f"{DEV_FAMILY}.scripted.jsonl")
    write_shard(str(disguised), [{"episode_id": "s0",
                                  "family_id": SEALED_FAMILY,
                                  "split": "SEALED_TEST"}],
                sealed=True,
                split_manifest_sha256_hex=manifest["split_manifest_sha256"])
    with pytest.raises(ValueError) as exc:
        ds.load_dev_episodes([str(disguised)], guard=guard)
    assert "sealed_gate" in str(exc.value)


def test_sealed_records_cannot_reach_the_selection_rule(tmp_path):
    with pytest.raises(SealedEvidenceRefused):
        DevMetrics.from_records(
            [{"split": "SEALED_TEST", "family_id": SEALED_FAMILY,
              "R": {"0": 1.0}}], stage="FL3_GRID")
    with pytest.raises(SealedEvidenceRefused):
        ds.GridRun(learning_rate=1e-4, scope="PEFT_MODE", updates=150,
                   dev=DevMetrics(stage="FL3_GRID", macro_aulc=1.0,
                                  split="SEALED_TEST"))


def test_the_selector_module_never_derives_a_sealed_key():
    source = open(ds.__file__, encoding="utf-8").read()
    assert "unseal_bytes" not in source
    assert "sealed_key(" not in source
    assert "loader_guard" in source


def _package_root():
    package = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(ds.__file__))))
    return os.path.join(package, "foundation_learner")


def _modules():
    root = _package_root()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", "tests", "docs")]
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                yield os.path.relpath(path, root).replace(os.sep, "/"), path


def test_only_the_sealed_gate_derives_the_sealed_key_for_reading():
    """The precise Amendment 1 / Amendment 7 invariant.

    ``ecology/base.py`` DEFINES the primitives and ``data/shards.py`` ENCIPHERS
    at write time (and accepts the contract §23 ``sealed_key`` parameter without
    ever deriving one).  Deriving the key and applying it to shard bytes — the
    act of opening the sealed set — happens in ``campaign/sealed_gate.py`` and
    nowhere else.
    """
    import ast

    derivers: dict[str, set[str]] = {}
    appliers: set[str] = set()
    for rel, path in _modules():
        if rel == "ecology/base.py":
            continue                      # defines the primitives
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    if inner.func.id == "sealed_key":
                        derivers.setdefault(rel, set()).add(node.name)
                    if inner.func.id == "unseal_bytes":
                        appliers.add(rel)
    assert derivers == {
        "data/shards.py": {"write_shard"},          # enciphering only
        "campaign/sealed_gate.py": {"open_sealed"},  # the one opening
    }, derivers
    assert appliers == {"data/shards.py", "campaign/sealed_gate.py"}, appliers


def test_read_shard_never_derives_a_key_of_its_own():
    from foundation_learner.data import shards

    import inspect
    source = inspect.getsource(shards.read_shard)
    assert "sealed_key(" not in source
    # and without a key it refuses enciphered bytes (proved above by the
    # disguised-shard fixture)
    assert "sealed_key: bytes | None = None" in source
