"""The frozen family split and its manifest (contract §5, Amendment 2)."""
from __future__ import annotations

import hashlib

import pytest

from foundation_learner.ecology import manifests, split


def test_split_reproduces_the_pinned_assignment():
    computed = split.compute_split()
    assert computed["TRAIN"] == [
        "grammar_classification", "set_operations", "dsl_execution",
        "string_rewrite", "finite_state_transducer", "modular_arithmetic"]
    assert computed["DEVELOPMENT"] == [
        "sequence_transform", "graph_edge_semantics", "boolean_rule"]
    assert computed["SEALED_TEST"] == [
        "constraint_rules", "permutation_composition", "propositional_transform"]
    assert {k: tuple(v) for k, v in computed.items()} == split.EXPECTED_SPLIT


def test_split_sizes_and_family_count():
    computed = split.compute_split()
    assert len(split.FAMILY_IDS) == 12
    assert len(set(split.FAMILY_IDS)) == 12
    assert [len(computed[s]) for s in split.SPLITS] == [6, 3, 3]


def test_family_hash_matches_the_frozen_rule():
    for fid in split.FAMILY_IDS:
        assert split.family_hash(fid) == hashlib.sha256(
            b"FOUNDATION_LEARNER_V0_FAMILY_SPLIT\0" + fid.encode()).hexdigest()
    order = [h for _, h in split.ordered_families()]
    assert order == sorted(order)


def test_every_family_is_in_exactly_one_split():
    computed = split.compute_split()
    seen: dict[str, str] = {}
    for name, fids in computed.items():
        for fid in fids:
            assert fid not in seen, f"{fid} in {seen.get(fid)} and {name}"
            seen[fid] = name
    assert set(seen) == set(split.FAMILY_IDS)
    for fid in split.FAMILY_IDS:
        assert split.split_of(fid) == seen[fid]
    with pytest.raises(KeyError):
        split.split_of("no_such_family")
    with pytest.raises(KeyError):
        split.families_in("NOT_A_SPLIT")


def test_split_manifest_records_hashes_sources_and_digest(tmp_path):
    path = str(tmp_path / manifests.SPLIT_MANIFEST_NAME)
    manifest = manifests.write_split_manifest(path)
    manifests.verify_split_manifest(manifest)
    assert manifest["assignment"] == {k: list(v)
                                      for k, v in split.compute_split().items()}
    assert set(manifest["generator_sources"]) == set(split.FAMILY_IDS)
    for entry in manifest["generator_sources"].values():
        assert len(entry["sha256"]) == 64
    assert manifest["split_manifest_sha256"] == \
        manifests.split_manifest_sha256(manifest)
    assert manifests.read_split_manifest(path) == manifest
    # the manifest is reproducible byte for byte
    assert manifests.build_split_manifest() == manifest


def test_tampered_manifest_is_rejected(tmp_path):
    manifest = manifests.build_split_manifest()
    tampered = dict(manifest)
    tampered["assignment"] = {
        "TRAIN": list(manifest["assignment"]["SEALED_TEST"]),
        "DEVELOPMENT": list(manifest["assignment"]["DEVELOPMENT"]),
        "SEALED_TEST": list(manifest["assignment"]["TRAIN"]),
    }
    with pytest.raises(ValueError):
        manifests.verify_split_manifest(tampered)
    # even with the digest recomputed, the frozen rule still refuses it
    tampered["split_manifest_sha256"] = manifests.split_manifest_sha256(tampered)
    with pytest.raises(ValueError):
        manifests.verify_split_manifest(tampered)
