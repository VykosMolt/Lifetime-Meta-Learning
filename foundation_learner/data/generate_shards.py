"""Deterministic generation of every pre-rental data artefact (contract §16).

Output layout under ``<out_root>``::

    family_split_manifest.json
    episodes/<SPLIT>/<family_id>.<mode>.jsonl        (sealed for SEALED_TEST)
    remaps/<SPLIT>/<family_id>.jsonl                 (DEVELOPMENT + SEALED_TEST)
    poison/<SPLIT>/<family_id>.jsonl                 (all splits)
    fl4_ablations/<SPLIT>/<family_id>.jsonl          (TRAIN + DEVELOPMENT)
    fl1_static/TRAIN/<family_id>.jsonl
    chains/DEVELOPMENT/<A>__<B>.jsonl
    seeds/arm_seed_manifest.json
    SHARD_SUMS.json
    PREGEN_MANIFEST.json

Guarantees, all enforced here with HARD FAILURE:

* every episode renders to at most 2048 Ouro tokens, both in its canonical
  surface and with every item replaced by each of its remapped variants
  (contract §7: episodes are SIZED to fit, generation-side truncation is
  forbidden), AND its projected ONLINE context
  (``scripted_tokens + (n_attempt_slots + 1) * 64``) fits the frozen eval-time
  allowance of 4096 tokens (Amendment 13);
* ``instance_id`` never crosses a split, a family, or an episode key — the only
  permitted sharing is between the ``scripted`` and ``successful`` variants of
  ONE episode, which by construction hold the same items;
* ``rule_id`` is unique within a family (rules are re-derived deterministically
  on collision) and can never cross families because the family id is part of
  the rule spec;
* nothing here contains a wall-clock value, so re-running produces
  byte-identical output.
"""
from __future__ import annotations

import os
from dataclasses import replace

from ..ecology.base import (TaskInstance, canonical_json, derive_seed,
                            domain_sha256, make_rng, sha256_bytes)
from ..ecology.families import get_family
from ..ecology.manifests import (SPLIT_MANIFEST_NAME, build_split_manifest,
                                 write_split_manifest)
from ..ecology.poison import POISON_CONDITIONS
from ..ecology.split import SPLITS, compute_split
from ..ecology.surface_remap import remap_ids_for
from ..episodes.assemble import assemble_episode, build_hint_variants
from ..episodes.parse import format_answer
from ..episodes.render import render_events
from ..episodes.schema import EPISODE_STRUCTURE_V0, Episode, Role, TASK_ROLES
from .pools import (DEFAULT_DIFFICULTY, EVAL_MAX_NEW_TOKENS,
                    EVAL_ONLINE_MAX_SEQ_LEN, FULL_POOLS,
                    FL1_ITEMS_PER_EPISODE, MAX_SEQ_LEN, PoolSizes,
                    REMAP_SPLITS, ROOT_SEED)
from .shards import SHARD_SUMS_NAME, ShardSums, write_json, write_shard

__all__ = ["DEFAULT_TOKENIZER_PATH", "PREGEN_MANIFEST_NAME", "TRAIN_MODES",
           "EVAL_MODES", "ARMS", "N_EPOCH_SEEDS", "RULE_DRAW_ATTEMPTS",
           "TokenBudget", "generate_all", "verify_manifest"]

DEFAULT_TOKENIZER_PATH = "/home/moloch/ouro_project/models/ouro_rltt_local"
PREGEN_MANIFEST_NAME = "PREGEN_MANIFEST.json"

#: TRAIN pools carry both attempt-policy modes: FL2 imitates successful
#: histories, FL3 trains on ordered histories that include real mistakes.
TRAIN_MODES = ("scripted", "successful")
EVAL_MODES = ("scripted",)

ARMS = ("FL1", "FL2", "FL3", "FL4", "FL5", "FL6", "FL7", "FL8")
N_EPOCH_SEEDS = 8


class TokenBudget:
    """Ouro-tokenizer length assertion (lazily imported; tokenizer files only).

    Loading the model weights is never necessary and never done: only the
    tokenizer is read from the frozen checkpoint directory.

    TWO assertions, both hard-failing (Amendment 13):

    * the SCRIPTED rendering must fit the contract §7 budget (2048 tokens);
    * the projected ONLINE context must fit the frozen eval-time allowance
      (4096 tokens).  Online evaluation replaces every scripted attempt line
      with up to ``EVAL_MAX_NEW_TOKENS`` real generated tokens and appends one
      more slot's worth while the model writes, so the projection is
      ``scripted_tokens + (n_attempt_slots + 1) * EVAL_MAX_NEW_TOKENS``.  It is
      an over-estimate (the scripted answer lines are counted as well as the
      generations that replace them), which is the direction a budget assertion
      must err in.
    """

    def __init__(self, tokenizer_path: str, max_tokens: int = MAX_SEQ_LEN,
                 online_max_tokens: int = EVAL_ONLINE_MAX_SEQ_LEN,
                 eval_max_new_tokens: int = EVAL_MAX_NEW_TOKENS):
        self.tokenizer_path = tokenizer_path
        self.max_tokens = int(max_tokens)
        self.online_max_tokens = int(online_max_tokens)
        self.eval_max_new_tokens = int(eval_max_new_tokens)
        self._tokenizer = None
        self.max_seen = 0
        self.max_online_projected = 0
        self.total = 0
        self.count = 0

    def _tok(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # lazy: stdlib+numpy only above
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self._tokenizer

    def measure(self, text: str) -> int:
        n = len(self._tok()(text)["input_ids"])
        self.max_seen = max(self.max_seen, n)
        self.total += n
        self.count += 1
        return n

    def check(self, label: str, text: str, *, n_attempt_slots: int) -> int:
        n = self.measure(text)
        if n > self.max_tokens:
            raise ValueError(
                f"{label}: episode renders to {n} Ouro tokens (> "
                f"{self.max_tokens}); instances must be sized to fit, "
                f"truncation is forbidden")
        projected = n + (int(n_attempt_slots) + 1) * self.eval_max_new_tokens
        self.max_online_projected = max(self.max_online_projected, projected)
        if projected > self.online_max_tokens:
            raise ValueError(
                f"{label}: projected ONLINE context is {projected} Ouro tokens "
                f"({n} scripted + ({n_attempt_slots} + 1) x "
                f"{self.eval_max_new_tokens} generated) and exceeds the frozen "
                f"eval-time allowance {self.online_max_tokens}; online "
                f"evaluation may not truncate, so such an episode would have to "
                f"be recorded as ONLINE_BUDGET_EXCEEDED and excluded")
        return n

    def stats(self) -> dict:
        return {"tokenizer_path": self.tokenizer_path,
                "max_tokens_allowed": self.max_tokens,
                "max_tokens_seen": self.max_seen,
                "eval_online_allowance": self.online_max_tokens,
                "eval_max_new_tokens": self.eval_max_new_tokens,
                "max_online_projected_seen": self.max_online_projected,
                "mean_tokens": (self.total / self.count) if self.count else 0.0,
                "renders_measured": self.count}


# --------------------------------------------------------------------------
# uniqueness ledgers
# --------------------------------------------------------------------------

class _UniquenessLedger:
    """Hard-failing registries for ``instance_id`` and ``rule_id``."""

    def __init__(self) -> None:
        self.instances: dict[str, tuple[str, str, str]] = {}
        self.rules: dict[str, tuple[str, str, int]] = {}

    def claim_instance(self, instance_id: str, split: str, family_id: str,
                       episode_key: str) -> None:
        owner = self.instances.get(instance_id)
        if owner is None:
            self.instances[instance_id] = (split, family_id, episode_key)
            return
        if owner != (split, family_id, episode_key):
            raise ValueError(
                f"instance_id collision: {instance_id[:16]} claimed by "
                f"{owner} and by {(split, family_id, episode_key)}")

    def claim_rule(self, rule_id: str, split: str, family_id: str,
                   index: int) -> bool:
        owner = self.rules.get(rule_id)
        if owner is None:
            self.rules[rule_id] = (split, family_id, index)
            return True
        if owner == (split, family_id, index):
            return True
        if owner[1] != family_id:
            raise ValueError(
                f"rule_id collision across families: {rule_id[:16]} in "
                f"{owner} and {(split, family_id, index)}")
        return False


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

#: deterministic redraw budget when a latent rule repeats within a family.
#: Measured worst case over both predeclared seeds and the full pools is 12.
RULE_DRAW_ATTEMPTS = 256


def _draw_unique_rule(family, ledger: _UniquenessLedger, root_seed: int,
                      split: str, index: int):
    """Deterministically draw a rule that is fresh within its family."""
    for attempt in range(RULE_DRAW_ATTEMPTS):
        seed = derive_seed(root_seed, "FL_V0_RULE_DRAW", family.family_id,
                           split, index, attempt)
        rule = family.sample_rule(make_rng(seed))
        rule_id = family.rule_id(rule)
        if ledger.claim_rule(rule_id, split, family.family_id, index):
            return rule, rule_id, seed, attempt
    raise ValueError(
        f"{family.family_id}: could not draw a fresh rule for episode {index} "
        f"after {RULE_DRAW_ATTEMPTS} deterministic attempts; the rule space is "
        f"too small for the requested pool")


def _n_attempt_slots(events) -> int:
    """How many MODEL_ATTEMPT slots the ONLINE evaluator will really generate."""
    return sum(1 for event in events if event.role is Role.MODEL_ATTEMPT)


def _remapped_events(episode: Episode, remapped: dict):
    """Episode events with every item swapped to its remapped surface."""
    out = []
    for event in episode.events:
        item = remapped.get(event.item_id)
        if item is None:
            out.append(event)
        elif event.role in TASK_ROLES:
            out.append(replace(event, text=item.prompt_text))
        elif event.role is Role.MODEL_ATTEMPT:
            out.append(replace(event, text=format_answer(item.answer_canonical)))
        elif event.role is Role.REVEAL:
            out.append(replace(event, text=item.answer_canonical))
        else:
            out.append(event)
    return out


def _fl1_items(episode: Episode) -> list[dict]:
    """Isolated (TASK -> ANSWER) items for the FL1 static baseline."""
    wanted = ("support", "related", "query")
    picked: list[dict] = []
    for role in wanted:
        for item_id in sorted(episode.items):
            record = episode.items[item_id]
            if record["role"] == role:
                instance = record["instance"]
                picked.append({
                    "item_id": item_id,
                    "family_id": episode.family_id,
                    "split": episode.split,
                    "rule_id": episode.rule_id,
                    "difficulty": episode.difficulty,
                    "source_episode_id": episode.episode_id,
                    "source_slot": record["slot"],
                    "source_role": role,
                    "prompt_text": instance["prompt_text"],
                    "answer_canonical": instance["answer_canonical"],
                })
                break
    return picked[:FL1_ITEMS_PER_EPISODE]


def _ablation_plans(episode: Episode) -> dict:
    """FL4 presence/ablation PLANS: event-subset specs, never duplicated text."""
    feedback_events = [i for i, e in enumerate(episode.events)
                       if e.role in (Role.FEEDBACK, Role.HINT, Role.REVEAL)]
    query_attempts = [i for i, e in enumerate(episode.events)
                      if e.role is Role.MODEL_ATTEMPT
                      and e.interaction_index == EPISODE_STRUCTURE_V0.query_index]
    plans = [{"ablation_id": "FULL", "ablated_event_indices": [],
              "ablated_item_id": None, "ablated_role": None}]
    for index in feedback_events:
        event = episode.events[index]
        plans.append({
            "ablation_id": f"ABLATE_{index}",
            "ablated_event_indices": [index],
            "ablated_item_id": event.item_id,
            "ablated_role": event.role.value,
            "certified": event.certified,
            "poison": event.poison,
        })
    return {
        "episode_id": episode.episode_id,
        "family_id": episode.family_id,
        "split": episode.split,
        "n_events": len(episode.events),
        "scoreable_feedback_events": feedback_events,
        "query_attempt_events": query_attempts,
        "note": ("keep_event_indices = all event indices minus "
                 "ablated_event_indices; the text is never duplicated"),
        "plans": plans,
    }


def _arm_seed_manifest(root_seed: int) -> dict:
    arms = {}
    for arm in ARMS:
        arms[arm] = {
            "data_seed": derive_seed(root_seed, "FL_V0_ARM", arm),
            "epoch_shuffle_seeds": [
                derive_seed(root_seed, "FL_V0_ARM_EPOCH", arm, epoch)
                for epoch in range(N_EPOCH_SEEDS)],
            "sampling_policy": ("uniform without replacement per epoch; "
                                "reshuffle by the epoch seed"),
        }
    return {"schema": "fl.arm_seed_manifest.v1", "root_seed": root_seed,
            "n_epoch_seeds": N_EPOCH_SEEDS, "arms": arms}


def generate_all(out_root: str, *, root_seed: int = ROOT_SEED,
                 pools: PoolSizes = FULL_POOLS,
                 difficulty: int = DEFAULT_DIFFICULTY,
                 tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
                 max_tokens: int = MAX_SEQ_LEN,
                 online_max_tokens: int = EVAL_ONLINE_MAX_SEQ_LEN,
                 log=None) -> dict:
    """Generate every artefact deterministically; returns the pregen manifest."""
    def say(message: str) -> None:
        if log is not None:
            log(message)

    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)

    manifest_path = os.path.join(out_root, SPLIT_MANIFEST_NAME)
    split_manifest = write_split_manifest(manifest_path)
    if split_manifest != build_split_manifest():  # pragma: no cover - defensive
        raise AssertionError("split manifest is not reproducible")
    seal_hex = split_manifest["split_manifest_sha256"]

    budget = TokenBudget(tokenizer_path, max_tokens,
                         online_max_tokens=online_max_tokens,
                         eval_max_new_tokens=EVAL_MAX_NEW_TOKENS)
    ledger = _UniquenessLedger()
    sums = ShardSums()
    files: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    assignment = compute_split()
    dev_episode_ids: dict[str, list[str]] = {}

    rewritten: list[str] = []

    def record(relpath: str, summary: dict) -> None:
        summary = dict(summary)
        # 'rewritten' is a fact about THIS run, never about the artefact, so it
        # is tracked separately and never persisted; otherwise a second, no-op
        # run would change SHARD_SUMS.json and break byte-identity.
        if summary.pop("rewritten", False):
            rewritten.append(relpath)
        summary["path"] = relpath
        files[relpath] = summary
        sums.add(relpath, summary)

    for split in SPLITS:
        sealed = split == "SEALED_TEST"
        modes = TRAIN_MODES if split == "TRAIN" else EVAL_MODES
        n_episodes = pools.for_split(split)
        for family_id in assignment[split]:
            family = get_family(family_id)
            say(f"[{split}] {family_id}: {n_episodes} episodes x {len(modes)} modes")
            episodes: dict[str, list[Episode]] = {mode: [] for mode in modes}
            remap_records: list[dict] = []
            poison_records: list[dict] = []
            ablation_records: list[dict] = []
            fl1_records: list[dict] = []

            for index in range(n_episodes):
                rule, rule_id, _rule_seed, _ = _draw_unique_rule(
                    family, ledger, root_seed, split, index)
                episode_seed = derive_seed(root_seed, "FL_V0_EPISODE_SEED",
                                           family_id, split, index)
                built: dict[str, Episode] = {}
                for mode in modes:
                    episode = assemble_episode(
                        family, rule, split=split, difficulty=difficulty,
                        root_seed=root_seed, episode_seed=episode_seed,
                        mode=mode, condition="clean", structured=True,
                        reveal=False)
                    for item_id in episode.items:
                        ledger.claim_instance(item_id, split, family_id,
                                              episode.meta["episode_key"])
                    budget.check(f"{family_id}/{split}/{mode}/{index}",
                                 render_events(episode.events).text,
                                 n_attempt_slots=_n_attempt_slots(
                                     episode.events))
                    episodes[mode].append(episode)
                    built[mode] = episode

                primary = built[modes[0]]
                poison_records.append({
                    "episode_id": primary.episode_id,
                    "family_id": family_id,
                    "split": split,
                    "conditions": list(POISON_CONDITIONS),
                    "variants": build_hint_variants(family, rule, primary,
                                                    root_seed),
                })
                if split in REMAP_SPLITS:
                    for remap_id in remap_ids_for(primary.episode_id,
                                                  pools.remap_variants):
                        remapped = {}
                        for item_id, item_record in primary.items.items():
                            instance = TaskInstance.from_dict(item_record["instance"])
                            remapped[item_id] = family.surface_remap(
                                rule, instance, remap_id)
                        remapped_events = _remapped_events(primary, remapped)
                        budget.check(
                            f"{family_id}/{split}/{index}/{remap_id}",
                            render_events(remapped_events).text,
                            n_attempt_slots=_n_attempt_slots(remapped_events))
                        remap_records.append({
                            "episode_id": primary.episode_id,
                            "family_id": family_id,
                            "split": split,
                            "remap_id": remap_id,
                            "items": {item_id: instance.to_dict()
                                      for item_id, instance in remapped.items()},
                        })
                if split in ("TRAIN", "DEVELOPMENT"):
                    ablation_records.append(_ablation_plans(primary))
                if split == "TRAIN":
                    fl1_records.extend(_fl1_items(primary))
                if split == "DEVELOPMENT":
                    dev_episode_ids.setdefault(family_id, []).append(
                        primary.episode_id)

            for mode in modes:
                rel = f"episodes/{split}/{family_id}.{mode}.jsonl"
                record(rel, write_shard(
                    os.path.join(out_root, rel),
                    [e.to_dict() for e in episodes[mode]],
                    sealed=sealed, split_manifest_sha256_hex=seal_hex))
            rel = f"poison/{split}/{family_id}.jsonl"
            record(rel, write_shard(os.path.join(out_root, rel), poison_records,
                                    sealed=sealed,
                                    split_manifest_sha256_hex=seal_hex))
            if remap_records:
                rel = f"remaps/{split}/{family_id}.jsonl"
                record(rel, write_shard(os.path.join(out_root, rel),
                                        remap_records, sealed=sealed,
                                        split_manifest_sha256_hex=seal_hex))
            if ablation_records:
                rel = f"fl4_ablations/{split}/{family_id}.jsonl"
                record(rel, write_shard(os.path.join(out_root, rel),
                                        ablation_records))
            if fl1_records:
                rel = f"fl1_static/{split}/{family_id}.jsonl"
                record(rel, write_shard(os.path.join(out_root, rel),
                                        fl1_records))
            stats[f"{split}/{family_id}"] = {
                "episodes": {mode: len(episodes[mode]) for mode in modes},
                "poison_records": len(poison_records),
                "remap_records": len(remap_records),
                "ablation_records": len(ablation_records),
                "fl1_items": len(fl1_records),
            }

    # -- A->B->A interference chains over ordered DEVELOPMENT pairs --------
    dev_families = assignment["DEVELOPMENT"]
    for family_a in dev_families:
        for family_b in dev_families:
            if family_a == family_b:
                continue
            pool_a = dev_episode_ids.get(family_a, [])
            pool_b = dev_episode_ids.get(family_b, [])
            if not pool_a or not pool_b:
                continue
            rng = make_rng(derive_seed(root_seed, "FL_V0_CHAIN", family_a,
                                       family_b))
            chains = []
            for chain_index in range(pools.chains_per_pair):
                a1 = pool_a[int(rng.integers(0, len(pool_a)))]
                b1 = pool_b[int(rng.integers(0, len(pool_b)))]
                candidates = [e for e in pool_a if e != a1] or pool_a
                a2 = candidates[int(rng.integers(0, len(candidates)))]
                chains.append({
                    "chain_id": domain_sha256("FL_V0_CHAIN", [family_a, family_b,
                                                              chain_index]),
                    "family_a": family_a, "family_b": family_b,
                    "order": ["A", "B", "A"],
                    "episode_ids": [a1, b1, a2],
                })
            rel = f"chains/DEVELOPMENT/{family_a}__{family_b}.jsonl"
            record(rel, write_shard(os.path.join(out_root, rel), chains))
            stats[f"chains/{family_a}__{family_b}"] = {"chains": len(chains)}

    rel = "seeds/arm_seed_manifest.json"
    record(rel, write_json(os.path.join(out_root, rel),
                           _arm_seed_manifest(root_seed)))

    sums_payload = sums.to_dict()
    rel = SHARD_SUMS_NAME
    summary = write_json(os.path.join(out_root, rel), sums_payload)
    if summary.pop("rewritten", False):
        rewritten.append(rel)
    files[rel] = dict(summary, path=rel)

    manifest_payload = (canonical_json(split_manifest) + "\n").encode("utf-8")
    files[SPLIT_MANIFEST_NAME] = {
        "path": SPLIT_MANIFEST_NAME,
        "sha256": sha256_bytes(manifest_payload),
        "bytes": len(manifest_payload),
    }

    manifest = {
        "schema": "fl.pregen_manifest.v1",
        "root_seed": root_seed,
        "difficulty": difficulty,
        "structure": EPISODE_STRUCTURE_V0.name,
        "pools": {"train": pools.train, "development": pools.development,
                  "sealed_test": pools.sealed_test,
                  "chains_per_pair": pools.chains_per_pair,
                  "remap_variants": pools.remap_variants},
        "train_modes": list(TRAIN_MODES),
        "eval_modes": list(EVAL_MODES),
        "poison_conditions": list(POISON_CONDITIONS),
        "split_manifest_sha256": seal_hex,
        "assignment": {k: list(v) for k, v in assignment.items()},
        "token_budget": budget.stats(),
        "counts": {
            "instances": len(ledger.instances),
            "rules": len(ledger.rules),
            "files": len(files),
        },
        "per_family": stats,
        "files": [files[k] for k in sorted(files)],
        "notes": [
            "REVEAL events are not materialised in the standard pools; the "
            "assembler builds them on demand for the FL7 supervised inner loop "
            "from the stored item answers.",
            "The correctness-only condition is the standard episode rendered "
            "without its HINT events (episodes/render.render_subset).",
            "The algorithmic baseline histories of contract 16 are the "
            "scripted-attempt episodes themselves (episodes/*.scripted.jsonl) "
            "plus the isolated TASK->ANSWER items in fl1_static/ that FL1 "
            "consumes without any history.",
            "No wall-clock value appears in any generated artefact.",
        ],
    }
    rel = PREGEN_MANIFEST_NAME
    summary = write_json(os.path.join(out_root, rel), manifest)
    if summary.get("rewritten"):
        rewritten.append(rel)
    # underscore keys are run-local and never persisted
    return dict(manifest, _run={"rewritten": sorted(rewritten)})


def verify_manifest(out_root: str) -> dict:
    """Re-hash every file listed in ``PREGEN_MANIFEST.json``."""
    from .shards import read_json
    from ..ecology.base import sha256_file

    out_root = os.path.abspath(out_root)
    manifest = read_json(os.path.join(out_root, PREGEN_MANIFEST_NAME))
    bad = []
    for entry in manifest["files"]:
        path = os.path.join(out_root, entry["path"])
        if not os.path.exists(path):
            bad.append((entry["path"], "missing"))
            continue
        if sha256_file(path) != entry["sha256"]:
            bad.append((entry["path"], "hash mismatch"))
    return {"checked": len(manifest["files"]), "bad": bad,
            "ok": not bad}
