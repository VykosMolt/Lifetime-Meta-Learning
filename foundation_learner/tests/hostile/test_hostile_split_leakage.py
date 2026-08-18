"""HOSTILE: try to move a family, an instance or a rule across the split.

Attacks, all of which must fail:

1. re-roll — call the split rule repeatedly, hoping it is not a pure function;
2. re-label — hand-edit a split manifest and re-hash it, hoping the verifier
   accepts a manifest that disagrees with the frozen rule;
3. source swap — change a generator source and hope the manifest digest does
   not notice;
4. rule reuse — generate the SAME episode index in two different splits and
   hope a latent rule or an instance identity is shared;
5. sealed family in a training pool — hope a SEALED_TEST family can be
   generated into a TRAIN shard path without the manifest disagreeing.
"""
from __future__ import annotations

import pytest

from foundation_learner.data import generate_shards as gen
from foundation_learner.ecology import manifests, split
from foundation_learner.ecology.base import derive_seed, make_rng
from foundation_learner.ecology.families import get_family


def test_repeated_rolls_never_change_the_split():
    first = split.compute_split()
    for _ in range(50):
        assert split.compute_split() == first
    assert {k: tuple(v) for k, v in first.items()} == split.EXPECTED_SPLIT


def test_relabelled_manifest_is_refused_even_after_rehashing():
    manifest = manifests.build_split_manifest()
    for victim in ("constraint_rules", "permutation_composition",
                   "propositional_transform"):
        forged = dict(manifest)
        assignment = {k: list(v) for k, v in manifest["assignment"].items()}
        assignment["SEALED_TEST"].remove(victim)
        assignment["TRAIN"].append(victim)
        forged["assignment"] = assignment
        forged["split_manifest_sha256"] = \
            manifests.split_manifest_sha256(forged)
        with pytest.raises(ValueError):
            manifests.verify_split_manifest(forged)


def test_a_tampered_digest_is_refused():
    manifest = manifests.build_split_manifest()
    forged = dict(manifest, split_manifest_sha256="0" * 64)
    with pytest.raises(ValueError):
        manifests.verify_split_manifest(forged)


def test_generator_source_hashes_bind_the_manifest():
    """Re-hashing the digest does not help: the sources themselves are checked."""
    manifest = manifests.build_split_manifest()
    forged = dict(manifest)
    sources = {k: dict(v) for k, v in manifest["generator_sources"].items()}
    sources["boolean_rule"]["sha256"] = "1" * 64
    forged["generator_sources"] = sources
    forged["split_manifest_sha256"] = manifests.split_manifest_sha256(forged)
    with pytest.raises(ValueError):
        manifests.verify_split_manifest(forged)
    dropped = dict(manifest)
    dropped["generator_sources"] = {
        k: v for k, v in manifest["generator_sources"].items()
        if k != "boolean_rule"}
    dropped["split_manifest_sha256"] = manifests.split_manifest_sha256(dropped)
    with pytest.raises(ValueError):
        manifests.verify_split_manifest(dropped)


def test_forged_family_hash_is_refused():
    manifest = manifests.build_split_manifest()
    forged = dict(manifest)
    hashes = dict(manifest["family_hashes"])
    hashes["boolean_rule"] = "2" * 64
    forged["family_hashes"] = hashes
    forged["split_manifest_sha256"] = manifests.split_manifest_sha256(forged)
    with pytest.raises(ValueError):
        manifests.verify_split_manifest(forged)


def test_no_latent_rule_or_instance_is_shared_across_splits():
    ledger = gen._UniquenessLedger()
    seen_rules: dict[str, tuple] = {}
    seen_instances: dict[str, tuple] = {}
    for split_name in split.SPLITS:
        for family_id in split.compute_split()[split_name]:
            family = get_family(family_id)
            for index in range(6):
                rule, rule_id, _, _ = gen._draw_unique_rule(
                    family, ledger, 20260809, split_name, index)
                owner = seen_rules.setdefault(rule_id, (split_name, family_id))
                assert owner == (split_name, family_id)
                rng = make_rng(derive_seed(20260809, family_id, split_name,
                                           index))
                for _ in range(4):
                    instance = family.sample_instance(rule, rng, 1)
                    owner = seen_instances.setdefault(instance.instance_id,
                                                      (split_name, family_id))
                    assert owner == (split_name, family_id)


def test_the_same_episode_index_in_two_splits_yields_different_rules():
    """The split name is part of every rule derivation, by construction."""
    for family_id in split.FAMILY_IDS:
        family = get_family(family_id)
        ids = set()
        for split_name in split.SPLITS:
            ledger = gen._UniquenessLedger()
            _, rule_id, _, _ = gen._draw_unique_rule(family, ledger, 20260809,
                                                     split_name, 0)
            ids.add(rule_id)
        assert len(ids) == 3


def test_a_sealed_family_cannot_be_smuggled_into_a_training_pool():
    sealed = split.compute_split()["SEALED_TEST"]
    train = split.compute_split()["TRAIN"]
    assert not set(sealed) & set(train)
    for family_id in sealed:
        assert split.split_of(family_id) == "SEALED_TEST"
    manifest = manifests.build_split_manifest()
    assert set(manifest["assignment"]["SEALED_TEST"]) == set(sealed)
    # the generator writes a family only under its own split's directory
    assignment = split.compute_split()
    for split_name, families in assignment.items():
        for family_id in families:
            assert split.split_of(family_id) == split_name
