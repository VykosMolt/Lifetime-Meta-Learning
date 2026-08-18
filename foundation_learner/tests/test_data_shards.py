"""Shard writing, sealing and checksums (contract §16, Amendment 1)."""
from __future__ import annotations

import json
import os

import pytest

from foundation_learner.ecology.base import (canonical_json, sealed_key,
                                             sha256_bytes, sha256_file,
                                             unseal_bytes)
from foundation_learner.data import shards
from foundation_learner.data.pools import (FULL_POOLS, N_REMAP_VARIANTS,
                                           REMAP_SPLITS, ROOT_SEED,
                                           SECOND_SEED, tiny_pools)

RECORDS = [{"b": 2, "a": 1}, {"a": "x", "b": [1, 2]}]
SPLIT_HEX = "e" * 64


def test_pool_sizes_are_the_frozen_ones():
    assert (FULL_POOLS.train, FULL_POOLS.development, FULL_POOLS.sealed_test) \
        == (2000, 300, 300)
    assert FULL_POOLS.chains_per_pair == 200
    assert FULL_POOLS.remap_variants == N_REMAP_VARIANTS == 2
    assert REMAP_SPLITS == ("DEVELOPMENT", "SEALED_TEST")
    assert (ROOT_SEED, SECOND_SEED) == (20260809, 20260810)
    tiny = tiny_pools()
    assert (tiny.train, tiny.development, tiny.sealed_test) == (20, 3, 3)
    assert tiny.remap_variants == 2
    assert FULL_POOLS.for_split("TRAIN") == 2000
    with pytest.raises(KeyError):
        FULL_POOLS.for_split("NOPE")


def test_shard_bytes_are_canonical_and_stable():
    payload = shards.shard_bytes(RECORDS)
    assert payload == b'{"a":1,"b":2}\n{"a":"x","b":[1,2]}\n'
    assert payload == shards.shard_bytes(RECORDS)


def test_plain_shard_roundtrip(tmp_path):
    path = str(tmp_path / "plain.jsonl")
    summary = shards.write_shard(path, RECORDS)
    assert summary["records"] == 2 and not summary["sealed"]
    assert summary["sha256"] == sha256_file(path)
    assert summary["plaintext_sha256"] == summary["sha256"]
    assert summary["rewritten"] is True
    assert shards.read_shard(path) == [{"a": 1, "b": 2},
                                       {"a": "x", "b": [1, 2]}]


def test_rewriting_identical_bytes_is_a_no_op(tmp_path):
    path = str(tmp_path / "plain.jsonl")
    first = shards.write_shard(path, RECORDS)
    mtime = os.path.getmtime(path)
    second = shards.write_shard(path, RECORDS)
    assert second["rewritten"] is False
    assert second["sha256"] == first["sha256"]
    assert os.path.getmtime(path) == mtime


def test_sealed_shard_is_enciphered_on_disk_and_refuses_plain_reads(tmp_path):
    path = str(tmp_path / "sealed.jsonl")
    records = [{"prompt_text": "a hidden set expression", "answer": "b d g"}
               for _ in range(40)]
    summary = shards.write_shard(path, records, sealed=True,
                                 split_manifest_sha256_hex=SPLIT_HEX)
    assert summary["sealed"] and summary["plaintext_sha256"] is None
    raw = open(path, "rb").read()
    plaintext = shards.shard_bytes(records)
    assert raw != plaintext
    assert b"prompt_text" not in raw
    assert b"a hidden set expression" not in raw
    assert summary["sha256"] == sha256_bytes(raw)
    with pytest.raises(ValueError):
        shards.read_shard(path)
    # only the gate's pure function recovers it, and only with the right key
    assert unseal_bytes(raw, sealed_key(SPLIT_HEX)) == plaintext
    assert unseal_bytes(raw, sealed_key("f" * 64)) != plaintext


def test_sealing_requires_the_split_manifest_digest(tmp_path):
    with pytest.raises(ValueError):
        shards.write_shard(str(tmp_path / "x.jsonl"), RECORDS, sealed=True)


def test_json_helpers_and_shard_sums(tmp_path):
    path = str(tmp_path / "obj.json")
    summary = shards.write_json(path, {"b": 1, "a": 2})
    assert open(path, encoding="utf-8").read() == '{"a":2,"b":1}\n'
    assert shards.read_json(path) == {"a": 2, "b": 1}
    assert summary["sha256"] == sha256_file(path)

    sums = shards.ShardSums()
    sums.add("episodes/TRAIN/x.jsonl", summary)
    sums.add("episodes/TRAIN/a.jsonl", summary)
    with pytest.raises(ValueError):
        sums.add("episodes/TRAIN/x.jsonl", summary)
    payload = sums.to_dict()
    assert payload["schema"] == "fl.shard_sums.v1"
    assert [s["path"] for s in payload["shards"]] == [
        "episodes/TRAIN/a.jsonl", "episodes/TRAIN/x.jsonl"]


def test_episode_from_json_reconstructs_an_episode():
    from foundation_learner.ecology.base import derive_seed, make_rng
    from foundation_learner.ecology.families import get_family
    from foundation_learner.episodes.assemble import assemble_episode
    from foundation_learner.episodes.schema import Episode

    family = get_family("boolean_rule")
    seed = derive_seed(20260809, "shard-roundtrip")
    rule = family.sample_rule(make_rng(seed))
    episode = assemble_episode(family, rule, split="DEVELOPMENT", difficulty=1,
                               root_seed=20260809, episode_seed=seed)
    rebuilt = shards.episode_from_json(episode.to_dict())
    assert isinstance(rebuilt, Episode)
    assert rebuilt.to_dict() == episode.to_dict()


def test_atomic_write_leaves_no_temp_file(tmp_path):
    path = str(tmp_path / "atomic.jsonl")
    shards.write_shard(path, RECORDS)
    assert os.listdir(tmp_path) == ["atomic.jsonl"]


def test_corrupt_shard_reports_a_clear_failure(tmp_path):
    path = str(tmp_path / "broken.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not json}\n")
    with pytest.raises(ValueError):
        shards.read_shard(path)
    # a canonical record is still readable
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(canonical_json({"a": 1}) + "\n")
    assert shards.read_shard(path) == [json.loads('{"a": 1}')]
