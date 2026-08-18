"""HOSTILE: try to read sealed-test material without the opening gate.

Attacks, all of which must fail:

1. plaintext scan — search sealed shard bytes for prompts, answers, rule specs,
   field names and any recognisable JSON structure;
2. plain reader — open a sealed shard with the ordinary shard reader;
3. wrong key — derive the key from anything other than the split-manifest
   digest;
4. package search — look for a decipher path (a CLI, a helper, a call site)
   anywhere in the ecology / episodes / data / scripts code that a pre-gate
   caller could invoke;
5. summary leakage — look for sealed plaintext digests or contents inside the
   public manifests.
"""
from __future__ import annotations

import os
import re

import pytest

from foundation_learner.data import shards
from foundation_learner.data.shards import write_json, write_shard
from foundation_learner.ecology.base import (canonical_json, derive_seed,
                                             make_rng, sealed_key,
                                             unseal_bytes)
from foundation_learner.ecology.families import get_family
from foundation_learner.ecology.manifests import build_split_manifest
from foundation_learner.ecology.split import compute_split
from foundation_learner.episodes.assemble import assemble_episode

SPLIT_MANIFEST = build_split_manifest()
SEAL_HEX = SPLIT_MANIFEST["split_manifest_sha256"]
_PACKAGE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _sealed_episodes():
    episodes = []
    for family_id in compute_split()["SEALED_TEST"]:
        family = get_family(family_id)
        for index in range(3):
            seed = derive_seed(20260809, "hostile-sealed", family_id, index)
            rule = family.sample_rule(make_rng(seed))
            episodes.append(assemble_episode(
                family, rule, split="SEALED_TEST", difficulty=1,
                root_seed=20260809, episode_seed=seed, reveal=True))
    return episodes


def test_sealed_shard_bytes_contain_no_plaintext(tmp_path):
    episodes = _sealed_episodes()
    records = [e.to_dict() for e in episodes]
    path = str(tmp_path / "sealed.jsonl")
    write_shard(path, records, sealed=True, split_manifest_sha256_hex=SEAL_HEX)
    raw = open(path, "rb").read()

    needles = [b"episode_id", b"prompt_text", b"answer_canonical", b"rule_spec",
               b"ANSWER", b"TASK", b"HINT", b"family_id", b"instance_id",
               b"SEALED_TEST", b'{"', b'"}']
    for episode in episodes:
        needles.append(episode.episode_id.encode())
        needles.append(episode.rule_id.encode())
        for record in episode.items.values():
            instance = record["instance"]
            needles.append(instance["answer_canonical"].encode())
            needles.append(instance["prompt_text"][:40].encode())
    for needle in needles:
        if len(needle) < 3:
            continue
        assert needle not in raw, needle[:40]

    # and the bytes do not even look like text
    printable = sum(1 for b in raw if 32 <= b < 127)
    assert printable / len(raw) < 0.75


def test_plain_reader_refuses_sealed_shards(tmp_path):
    path = str(tmp_path / "sealed.jsonl")
    write_shard(path, [e.to_dict() for e in _sealed_episodes()], sealed=True,
                split_manifest_sha256_hex=SEAL_HEX)
    with pytest.raises(ValueError):
        shards.read_shard(path)


def test_only_the_split_manifest_digest_opens_a_sealed_shard(tmp_path):
    records = [e.to_dict() for e in _sealed_episodes()]
    path = str(tmp_path / "sealed.jsonl")
    write_shard(path, records, sealed=True, split_manifest_sha256_hex=SEAL_HEX)
    raw = open(path, "rb").read()
    plaintext = shards.shard_bytes(records)
    assert unseal_bytes(raw, sealed_key(SEAL_HEX)) == plaintext
    for wrong_hex in ("0" * 64, "f" * 64, SEAL_HEX[:-1] + "0"):
        assert unseal_bytes(raw, sealed_key(wrong_hex)) != plaintext
    with pytest.raises(ValueError):
        sealed_key("not a digest")


def test_no_pre_gate_module_ever_opens_sealed_data():
    """No module a pre-gate caller can reach may actually unseal anything.

    Contract §23 gives ``read_shard`` an optional ``sealed_key`` parameter, so
    the invariant is stated where it bites: the deciphering primitive exists in
    exactly two places (the pure function in ``ecology/base.py`` and the inert
    §23 reader parameter in ``data/shards.py``), no pre-gate module ever CALLS a
    sealed read or derives a key for reading, and no decipher CLI exists.  As
    Amendment 1 records, the protection is procedural; the key comes from a
    public digest, so no cryptographic unopenability is claimed.
    """
    allowed_definitions = {os.path.join("ecology", "base.py"),
                           os.path.join("data", "shards.py")}
    watched = []
    for package in ("ecology", "episodes", "data", "scripts"):
        for root, _dirs, files in os.walk(os.path.join(_PACKAGE, package)):
            if "__pycache__" in root:
                continue
            for name in files:
                if name.endswith(".py"):
                    watched.append(os.path.join(root, name))
    assert watched
    offenders = []
    for path in watched:
        source = open(path, encoding="utf-8").read()
        rel = os.path.relpath(path, _PACKAGE)
        if re.search(r"\bunseal_bytes\s*\(", source) and \
                rel not in allowed_definitions:
            offenders.append((rel, "unseal call"))
        if re.search(r"--decipher|--unseal", source):
            offenders.append((rel, "decipher CLI"))
        if re.search(r"def unseal", source) and rel not in allowed_definitions:
            offenders.append((rel, "decipher helper"))
        # nobody may actually perform a sealed read (definitions excluded)
        if re.search(r"(?<!def )\bread_shard\s*\([^)]*sealed_key", source):
            offenders.append((rel, "sealed read call"))
        # keys are derived only to WRITE sealed shards
        if re.search(r"\bsealed_key\s*\(", source) and \
                rel not in allowed_definitions:
            offenders.append((rel, "key derivation"))
    assert offenders == [], offenders
    # and the only key derivation in data/shards.py is on the write path
    source = open(os.path.join(_PACKAGE, "data", "shards.py"),
                  encoding="utf-8").read()
    write_body = source.split("def write_shard")[1].split("\ndef ")[0]
    assert "sealed_key(" in write_body
    read_body = source.split("def read_shard")[1].split("\ndef ")[0]
    assert "sealed_key(" not in read_body


def test_the_shard_reader_is_inert_without_an_explicit_key(tmp_path):
    """The §23 parameter opens nothing unless a caller supplies the key."""
    records = [e.to_dict() for e in _sealed_episodes()]
    path = str(tmp_path / "sealed.jsonl")
    write_shard(path, records, sealed=True, split_manifest_sha256_hex=SEAL_HEX)
    with pytest.raises(ValueError):
        shards.read_shard(path)
    with pytest.raises(ValueError):
        shards.read_shard(path, sealed_key=sealed_key("0" * 64))
    opened = shards.read_shard(path, sealed_key=sealed_key(SEAL_HEX))
    assert [r["episode_id"] for r in opened] == \
        [r["episode_id"] for r in records]


def test_public_manifests_carry_no_sealed_plaintext(tmp_path):
    records = [e.to_dict() for e in _sealed_episodes()]
    path = str(tmp_path / "sealed.jsonl")
    summary = write_shard(path, records, sealed=True,
                          split_manifest_sha256_hex=SEAL_HEX)
    assert summary["plaintext_sha256"] is None
    sums = shards.ShardSums()
    sums.add("episodes/SEALED_TEST/x.jsonl", summary)
    payload = canonical_json(sums.to_dict())
    for episode in records:
        assert episode["episode_id"] not in payload
        for record in episode["items"].values():
            answer = record["instance"]["answer_canonical"]
            if len(answer) >= 3:
                assert answer not in payload
    written = write_json(str(tmp_path / "sums.json"), sums.to_dict())
    assert written["sha256"]
