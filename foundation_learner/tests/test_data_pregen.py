"""End-to-end tiny pre-generation: determinism, uniqueness, sealing (§16).

The tiny run (pools / 100) is generated ONCE per test session into a temporary
directory keyed by the hash of the generator sources, so a source change always
regenerates and a repeated session is cheap.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from foundation_learner.data import generate_shards as gen
from foundation_learner.data.pools import (EVAL_MAX_NEW_TOKENS,
                                           EVAL_ONLINE_MAX_SEQ_LEN,
                                           MAX_SEQ_LEN, ROOT_SEED, tiny_pools)
from foundation_learner.data.shards import (SHARD_SUMS_NAME, read_json,
                                            read_shard)
from foundation_learner.ecology.base import (TaskInstance, sealed_key,
                                             sha256_file, sha256_tree,
                                             unseal_bytes)
from foundation_learner.ecology.families import get_family
from foundation_learner.ecology.manifests import (SPLIT_MANIFEST_NAME,
                                                  verify_split_manifest)
from foundation_learner.ecology.poison import POISON_CONDITIONS
from foundation_learner.ecology.split import compute_split
from foundation_learner.episodes.parse import format_answer
from foundation_learner.episodes.schema import Episode, Role

_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(gen.__file__)))
_SOURCE_KEY = sha256_tree(os.path.join(_PACKAGE, "ecology"))[:16] + \
    sha256_tree(os.path.join(_PACKAGE, "episodes"))[:16] + \
    sha256_tree(os.path.join(_PACKAGE, "data"))[:16]
_ROOT = os.path.join(tempfile.gettempdir(), f"fl_v0_tiny_pregen_{_SOURCE_KEY}")

_CACHE: dict = {}


def _pregen() -> dict:
    if "manifest" not in _CACHE:
        if os.path.isdir(_ROOT) and not os.path.exists(
                os.path.join(_ROOT, gen.PREGEN_MANIFEST_NAME)):
            shutil.rmtree(_ROOT)
        _CACHE["manifest"] = gen.generate_all(
            _ROOT, root_seed=ROOT_SEED, pools=tiny_pools())
    return _CACHE["manifest"]


def _iter_episodes(manifest, split=None):
    for entry in manifest["files"]:
        rel = entry["path"]
        if not rel.startswith("episodes/"):
            continue
        _, entry_split, _ = rel.split("/", 2)
        if entry_split == "SEALED_TEST" or (split and entry_split != split):
            continue
        for record in read_shard(os.path.join(_ROOT, rel)):
            yield rel, Episode.from_dict(record)


def test_tiny_generation_produces_the_expected_layout():
    manifest = _pregen()
    assert manifest["schema"] == "fl.pregen_manifest.v1"
    assert manifest["root_seed"] == ROOT_SEED
    assert manifest["structure"] == "EPISODE_STRUCTURE_V0"
    assert manifest["poison_conditions"] == list(POISON_CONDITIONS)
    assignment = compute_split()
    assert manifest["assignment"] == {k: list(v) for k, v in assignment.items()}
    paths = {entry["path"] for entry in manifest["files"]}
    for family_id in assignment["TRAIN"]:
        assert f"episodes/TRAIN/{family_id}.scripted.jsonl" in paths
        assert f"episodes/TRAIN/{family_id}.successful.jsonl" in paths
        assert f"fl1_static/TRAIN/{family_id}.jsonl" in paths
        assert f"fl4_ablations/TRAIN/{family_id}.jsonl" in paths
        assert f"remaps/TRAIN/{family_id}.jsonl" not in paths
    for family_id in assignment["DEVELOPMENT"]:
        assert f"remaps/DEVELOPMENT/{family_id}.jsonl" in paths
    for family_id in assignment["SEALED_TEST"]:
        assert f"episodes/SEALED_TEST/{family_id}.scripted.jsonl" in paths
        assert f"remaps/SEALED_TEST/{family_id}.jsonl" in paths
        assert f"fl4_ablations/SEALED_TEST/{family_id}.jsonl" not in paths
    assert SHARD_SUMS_NAME in paths or os.path.exists(
        os.path.join(_ROOT, SHARD_SUMS_NAME))
    verify_split_manifest(read_json(os.path.join(_ROOT, SPLIT_MANIFEST_NAME)))


def test_manifest_hashes_match_the_files_on_disk():
    _pregen()
    report = gen.verify_manifest(_ROOT)
    assert report["ok"], report["bad"]
    assert report["checked"] > 20


def test_rerun_is_byte_identical_and_rewrites_nothing():
    first = _pregen()
    before = {entry["path"]: sha256_file(os.path.join(_ROOT, entry["path"]))
              for entry in first["files"]}
    second = gen.generate_all(_ROOT, root_seed=ROOT_SEED, pools=tiny_pools())
    after = {entry["path"]: sha256_file(os.path.join(_ROOT, entry["path"]))
             for entry in second["files"]}
    assert before == after
    assert second["_run"]["rewritten"] == []
    persisted = lambda m: {k: v for k, v in m.items()
                           if k not in ("files", "_run")}
    assert persisted(second) == persisted(first)
    assert not any("rewritten" in entry for entry in second["files"])


def test_pool_sizes_and_counts_match_the_request():
    manifest = _pregen()
    tiny = tiny_pools()
    for key, stats in manifest["per_family"].items():
        if key.startswith("chains/"):
            assert stats["chains"] == tiny.chains_per_pair
            continue
        split, _ = key.split("/", 1)
        expected = tiny.for_split(split)
        for mode, count in stats["episodes"].items():
            assert count == expected
        assert stats["poison_records"] == expected
        if split in ("DEVELOPMENT", "SEALED_TEST"):
            assert stats["remap_records"] == expected * tiny.remap_variants
        else:
            assert stats["remap_records"] == 0
            assert stats["fl1_items"] == expected * 3


def test_token_budget_is_enforced_and_recorded():
    manifest = _pregen()
    budget = manifest["token_budget"]
    assert budget["max_tokens_allowed"] == MAX_SEQ_LEN
    assert 0 < budget["max_tokens_seen"] <= MAX_SEQ_LEN
    assert budget["renders_measured"] > 0


def test_online_budget_projection_is_enforced_and_recorded():
    """Amendment 13: every episode must ALSO fit the eval-time allowance once
    each attempt slot carries real generated tokens."""
    manifest = _pregen()
    budget = manifest["token_budget"]
    assert budget["eval_online_allowance"] == EVAL_ONLINE_MAX_SEQ_LEN
    assert budget["eval_max_new_tokens"] == EVAL_MAX_NEW_TOKENS
    projected = budget["max_online_projected_seen"]
    assert budget["max_tokens_seen"] < projected <= EVAL_ONLINE_MAX_SEQ_LEN


def test_the_eval_allowance_is_the_one_the_online_walker_uses():
    """The constant is duplicated (data/ may not import torch); it may not
    drift from the evaluator's own value."""
    from foundation_learner.evaluation import learning_curve as LC

    assert EVAL_ONLINE_MAX_SEQ_LEN == LC.EVAL_ONLINE_MAX_SEQ_LEN
    assert LC.LearningCurveConfig().max_seq_len == EVAL_ONLINE_MAX_SEQ_LEN
    assert MAX_SEQ_LEN == LC.MAX_SEQ_LEN == 2048


def test_an_episode_that_would_overflow_online_is_refused_at_pregen_time():
    """The assertion is real: a budget whose allowance the SAME data exceeds
    hard-fails instead of shipping an episode the evaluator must exclude."""
    manifest = _pregen()
    text = "x " * 64
    tight = gen.TokenBudget(gen.DEFAULT_TOKENIZER_PATH,
                            online_max_tokens=256, eval_max_new_tokens=64)
    tight.check("fits", text, n_attempt_slots=1)          # 128 + 2*64 = 256
    with pytest.raises(ValueError, match="projected ONLINE context"):
        tight.check("overflows", text, n_attempt_slots=2)  # 128 + 3*64 = 320
    assert manifest["token_budget"]["max_online_projected_seen"] > 0


def test_no_instance_or_rule_crosses_a_split():
    manifest = _pregen()
    owner_instance: dict[str, str] = {}
    owner_rule: dict[str, str] = {}
    for rel, episode in _iter_episodes(manifest):
        for item_id in episode.items:
            assert owner_instance.setdefault(item_id, episode.split) == \
                episode.split
        assert owner_rule.setdefault(episode.rule_id, episode.split) == \
            episode.split
    assert len(owner_instance) == manifest["counts"]["instances"] - \
        _sealed_instance_count(manifest)


def _sealed_instance_count(manifest) -> int:
    sealed = 0
    for key, stats in manifest["per_family"].items():
        if key.startswith("SEALED_TEST/"):
            sealed += sum(stats["episodes"].values()) * 8
    return sealed


def test_rule_ids_are_unique_within_each_family():
    manifest = _pregen()
    seen: dict[str, set] = {}
    for _, episode in _iter_episodes(manifest):
        bucket = seen.setdefault((episode.family_id, episode.split), set())
        if episode.mode == "successful":
            continue  # the same rule intentionally backs both modes
        assert episode.rule_id not in bucket
        bucket.add(episode.rule_id)


def test_episode_contents_verify_against_their_family():
    manifest = _pregen()
    for _, episode in _iter_episodes(manifest):
        family = get_family(episode.family_id)
        for attempt in episode.meta["attempts"]:
            instance = TaskInstance.from_dict(
                episode.items[attempt["item_id"]]["instance"])
            text = episode.events[attempt["event_index"]].text
            correct = family.verify(_rule(episode), instance, text).correct
            assert correct != attempt["wrong"]


def _rule(episode):
    from foundation_learner.ecology.base import Rule
    return Rule(episode.family_id, episode.rule_spec["params"])


def test_fl1_static_items_are_isolated_and_correct():
    manifest = _pregen()
    for entry in manifest["files"]:
        if not entry["path"].startswith("fl1_static/"):
            continue
        records = read_shard(os.path.join(_ROOT, entry["path"]))
        assert records
        for record in records:
            assert set(record) >= {"prompt_text", "answer_canonical",
                                   "source_episode_id", "source_role"}
            assert "ATTEMPT:" not in record["prompt_text"]
            assert "FEEDBACK:" not in record["prompt_text"]
            assert record["answer_canonical"] not in record["prompt_text"] or \
                record["source_role"] in ("support", "related", "query")
            assert record["source_role"] in ("support", "related", "query")


def test_fl4_ablation_plans_reference_real_events():
    manifest = _pregen()
    episodes = {e.episode_id: e for _, e in _iter_episodes(manifest)}
    for entry in manifest["files"]:
        if not entry["path"].startswith("fl4_ablations/"):
            continue
        for record in read_shard(os.path.join(_ROOT, entry["path"])):
            episode = episodes[record["episode_id"]]
            assert record["n_events"] == len(episode.events)
            assert record["plans"][0]["ablation_id"] == "FULL"
            assert record["plans"][0]["ablated_event_indices"] == []
            for plan in record["plans"][1:]:
                for index in plan["ablated_event_indices"]:
                    assert episode.events[index].role in (
                        Role.FEEDBACK, Role.HINT, Role.REVEAL)
            for index in record["query_attempt_events"]:
                event = episode.events[index]
                assert event.role is Role.MODEL_ATTEMPT
                assert event.interaction_index == 5
            assert len(record["plans"]) == \
                1 + len(record["scoreable_feedback_events"])


def test_poison_variants_cover_every_condition_and_never_certify_poison():
    manifest = _pregen()
    episodes = {e.episode_id: e for _, e in _iter_episodes(manifest)}
    for entry in manifest["files"]:
        if not entry["path"].startswith("poison/") or \
                "SEALED_TEST" in entry["path"]:
            continue
        for record in read_shard(os.path.join(_ROOT, entry["path"])):
            episode = episodes[record["episode_id"]]
            assert set(record["variants"]) == set(POISON_CONDITIONS)
            for condition, slots in record["variants"].items():
                for slot in slots:
                    event = episode.events[slot["event_index"]]
                    assert event.role is Role.HINT
                    assert not event.certified


def test_remap_variants_verify_in_their_own_surface():
    manifest = _pregen()
    rules = {e.episode_id: _rule(e) for _, e in _iter_episodes(manifest)}
    canonical = {}
    for _, episode in _iter_episodes(manifest):
        for item_id, record in episode.items.items():
            canonical[item_id] = record["instance"]
    for entry in manifest["files"]:
        if not entry["path"].startswith("remaps/") or \
                "SEALED_TEST" in entry["path"]:
            continue
        for record in read_shard(os.path.join(_ROOT, entry["path"])):
            family = get_family(record["family_id"])
            rule = rules[record["episode_id"]]
            assert len(record["items"]) == 8
            for item_id, item in record["items"].items():
                instance = TaskInstance.from_dict(item)
                assert instance.surface_map_id == record["remap_id"]
                assert instance.rule_id == canonical[item_id]["rule_id"]
                assert instance.seed == canonical[item_id]["seed"]
                assert instance.prompt_text != canonical[item_id]["prompt_text"]
                assert family.verify(
                    rule, instance,
                    format_answer(instance.answer_canonical)).correct


def test_chains_reference_development_episodes_in_a_b_a_order():
    manifest = _pregen()
    dev_ids: dict[str, set] = {}
    for _, episode in _iter_episodes(manifest, split="DEVELOPMENT"):
        dev_ids.setdefault(episode.family_id, set()).add(episode.episode_id)
    seen_pairs = set()
    for entry in manifest["files"]:
        if not entry["path"].startswith("chains/"):
            continue
        for record in read_shard(os.path.join(_ROOT, entry["path"])):
            assert record["order"] == ["A", "B", "A"]
            a1, b1, a2 = record["episode_ids"]
            assert a1 in dev_ids[record["family_a"]]
            assert a2 in dev_ids[record["family_a"]]
            assert b1 in dev_ids[record["family_b"]]
            assert record["family_a"] != record["family_b"]
            seen_pairs.add((record["family_a"], record["family_b"]))
    assert len(seen_pairs) == 6


def test_arm_seed_manifest_is_complete():
    _pregen()
    manifest = read_json(os.path.join(_ROOT, "seeds", "arm_seed_manifest.json"))
    assert manifest["root_seed"] == ROOT_SEED
    assert set(manifest["arms"]) == set(gen.ARMS)
    for arm in manifest["arms"].values():
        assert len(arm["epoch_shuffle_seeds"]) == gen.N_EPOCH_SEEDS
        assert len(set(arm["epoch_shuffle_seeds"])) == gen.N_EPOCH_SEEDS
        assert arm["data_seed"] >= 0


def test_sealed_shards_hold_no_plaintext_and_open_only_with_the_key():
    manifest = _pregen()
    seal_hex = manifest["split_manifest_sha256"]
    key = sealed_key(seal_hex)
    sealed_paths = [e["path"] for e in manifest["files"]
                    if "SEALED_TEST" in e["path"]]
    assert sealed_paths
    for rel in sealed_paths:
        raw = open(os.path.join(_ROOT, rel), "rb").read()
        assert b"episode_id" not in raw
        assert b"prompt_text" not in raw
        assert b"ANSWER" not in raw
        with pytest.raises(ValueError):
            read_shard(os.path.join(_ROOT, rel))
        plaintext = unseal_bytes(raw, key)
        assert plaintext.startswith(b"{")
        for line in plaintext.decode("utf-8").splitlines():
            assert line.startswith("{")


def test_second_seed_produces_different_data(tmp_path):
    from foundation_learner.data.pools import SECOND_SEED

    other = str(tmp_path / "seed2")
    manifest = gen.generate_all(other, root_seed=SECOND_SEED,
                                pools=tiny_pools())
    base = _pregen()
    a = {e["path"]: e["sha256"] for e in base["files"]}
    b = {e["path"]: e["sha256"] for e in manifest["files"]}
    assert set(a) == set(b)
    changed = [p for p in a if a[p] != b[p]]
    assert len(changed) > len(a) // 2
    assert a[SPLIT_MANIFEST_NAME] == b[SPLIT_MANIFEST_NAME], \
        "the family split must not depend on the run seed"
