"""Shared plumbing for the FL4-FL8 campaign stage adapters (W3).

The campaign layer (``campaign/stage_definitions.py``) resolves each mechanism
rung lazily through a dotted entry point::

    foundation_learner.mechanisms.value_head:run_fl4_stage
    foundation_learner.mechanisms.fast_state:run_fl5_stage
    foundation_learner.mechanisms.value_gating:run_fl6_stage
    foundation_learner.mechanisms.fast_adapter:run_fl7_stage
    foundation_learner.mechanisms.consolidation:run_fl8_stage

Each entry point has the signature ``run_flN_stage(ctx, stage) -> dict`` and
returns a ``flb200.mechanism_stage.v1`` payload.  This module holds everything
those five thin adapters share so that no convention is implemented twice.

Rules the adapters obey
-----------------------
* **Fresh bundle per arm.**  Every independent arm calls ``ctx.bundle_factory()``
  again (contract §7); no arm inherits another arm's model.
* **All file access through ``ctx.guard``.**  Shard reads go through the
  campaign's own guarded loaders; every artefact is written with ``ctx.write``
  (or ``ctx.guard.open`` for the one binary artefact, the FL4 head tensors).
* **Scale from the context.**  ``ctx.eval_episode_cap`` / ``ctx.train_example_cap``
  / ``ctx.rehearsal`` decide how much work runs; a rehearsal is a MECHANICS
  walk, never a scientific result, and every payload says which it was.
* **No sealed material, ever.**  Only TRAIN and DEVELOPMENT shards are read;
  DEVELOPMENT records are tagged ``split="DEVELOPMENT"`` so that
  ``campaign.promotion.DevMetrics`` (whose constructor refuses sealed
  provenance) can be built from them.
* **The campaign's frozen conventions are imported, not re-implemented**:
  the guarded episode loader, the exact-verifier ``env_factory`` and the
  frozen greedy-decode budget all come from ``campaign.stage_definitions``.
  Imports happen INSIDE the functions so that importing ``mechanisms`` never
  pulls in the campaign package (and so the lazy stage table stays lazy).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

MECHANISM_STAGE_SCHEMA = "flb200.mechanism_stage.v1"

STATUS_COMPLETE = "COMPLETE"
STATUS_SKIPPED_INPUT = "SKIPPED_MISSING_INPUT"

#: rehearsal defaults — a dress rehearsal walks the mechanics at minimum size
REHEARSAL_UPDATES = 2
REHEARSAL_EPISODES = 1

__all__ = [
    "MECHANISM_STAGE_SCHEMA",
    "STATUS_COMPLETE",
    "STATUS_SKIPPED_INPUT",
    "REHEARSAL_UPDATES",
    "REHEARSAL_EPISODES",
    "stage_error",
    "mechanism_payload",
    "finish_stage",
    "generation_config",
    "env_factory",
    "prepare_for_evaluation",
    "load_split_episodes",
    "dev_episodes",
    "train_episodes",
    "count_scoreable_items",
    "stage_updates",
    "stage_episode_count",
    "trainable_params",
    "dev_records",
    "dev_metrics_dict",
    "macro_aulc",
    "post_reset_gain_records",
    "paired_persistence_evidence",
    "publish_persistence_evidence",
    "load_ablation_plans",
    "load_poison_records",
    "load_chain_specs",
    "reveal_variant",
    "save_head_state",
    "load_head_state",
    "value_head_from_ctx",
    "arm_checkpoint_state",
]


# --------------------------------------------------------------------------
# campaign interop (lazy imports, single source of truth)
# --------------------------------------------------------------------------
def _stage_definitions():
    from foundation_learner.campaign import stage_definitions

    return stage_definitions


def stage_error(message: str) -> Exception:
    """The campaign's own ``StageError`` (so a refusal is uniform)."""
    return _stage_definitions().StageError(message)


def generation_config(ctx: Any):
    """The campaign's frozen greedy-decode budget (64, rehearsal-overridable).

    Imported rather than re-derived: duplicating the rule would let the two
    copies drift on a FROZEN constant.  If the campaign renames the helper this
    raises loudly instead of silently decoding with a different budget.
    """
    return _stage_definitions()._generation_config(ctx)


def env_factory(episode: Any):
    """The campaign's exact-verifier environment factory (W4 API)."""
    return _stage_definitions().env_factory(episode)


def prepare_for_evaluation(bundle: Any) -> dict:
    """Force a bundle into the only state a decode is valid in (Amend. 16).

    ``model.eval()`` + layer-level gradient checkpointing off.  A mechanism
    rung that has just trained a module or run a fast-adaptation inner loop
    would otherwise decode with ``use_cache``/``past_key_values`` silently
    dropped by transformers' ``GradientCheckpointingLayer``.
    """
    return _stage_definitions().prepare_bundle_for_evaluation(bundle)


def load_split_episodes(ctx: Any, split: str, mode: str = "scripted", *,
                        limit_per_family: int | None = None) -> list[Any]:
    """Guarded shard load (never sealed: the loader guard refuses those)."""
    if str(split) == "SEALED_TEST":
        raise stage_error(
            "REFUSED: a mechanism stage may never read SEALED_TEST shards")
    return _stage_definitions().load_episodes(
        ctx.pregen_root, split, mode, guard=ctx.guard,
        limit_per_family=limit_per_family)


# --------------------------------------------------------------------------
# payloads
# --------------------------------------------------------------------------
def mechanism_payload(stage: Any, status: str = STATUS_COMPLETE,
                      **fields: Any) -> dict:
    payload = {
        "schema": MECHANISM_STAGE_SCHEMA,
        "stage": stage.stage_id,
        "status": status,
    }
    payload.update(fields)
    return payload


def finish_stage(ctx: Any, stage: Any, payload: dict) -> dict:
    """Write the stage's declared first output and return the payload."""
    payload.setdefault("rehearsal", bool(ctx.rehearsal))
    payload.setdefault("scope", ctx.scope)
    payload.setdefault("root_seed", int(ctx.root_seed))
    if ctx.rehearsal:
        payload.setdefault(
            "note", "DRESS REHEARSAL: mechanics only, never a scientific result")
    ctx.write(stage.stage_id, stage.outputs[0], _jsonable(payload))
    return payload


def _jsonable(value: Any) -> Any:
    """Drop non-JSON objects (tensors, modules) that a payload must not carry."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


# --------------------------------------------------------------------------
# scale
# --------------------------------------------------------------------------
def stage_updates(ctx: Any, key: str, *, rehearsal_default: int = REHEARSAL_UPDATES
                  ) -> int:
    """Optimizer updates for a mechanism arm.

    ``ctx.extra[key]`` wins; a rehearsal uses the tiny mechanics default;
    otherwise the affordability-selected ``ctx.updates`` is REQUIRED (a
    mechanism arm is projected with the MECHANISM formula, which already
    depends on U, so guessing one here would break the projection).
    """
    value = ctx.extra.get(key)
    if value is not None:
        return max(1, int(value))
    if ctx.rehearsal:
        return int(rehearsal_default)
    if ctx.updates is None:
        raise stage_error(
            f"{key}: no U selected; the affordability rule must choose U "
            "before a mechanism training arm runs")
    return int(ctx.updates)


def stage_episode_count(ctx: Any, stage: Any, key: str, *,
                        default: int | None = None) -> int:
    """Episode count for one sub-task of a mechanism stage."""
    value = ctx.extra.get(key)
    if value is not None:
        return max(1, int(value))
    if ctx.rehearsal:
        return max(1, ctx.eval_cap(stage))
    return max(1, int(default if default is not None else stage.eval_episodes))


def dev_episodes(ctx: Any, stage: Any, *, count: int | None = None,
                 mode: str = "scripted") -> list[Any]:
    """The DEVELOPMENT evaluation set for a mechanism rung.

    Delegates to the campaign's single evaluation-set assembler, which applies
    the cap PER FAMILY and asserts that the assembled set covers exactly the
    frozen DEVELOPMENT families.  The previous global ``episodes[:cap]`` slice
    silently collapsed every mechanism rung's evaluation onto whichever family
    sorts first in shard order, turning macro-averaged, family-clustered
    metrics into single-family ones (review finding R-C3).
    """
    sd = _stage_definitions()
    if count is None:
        return sd.load_balanced_eval_episodes(ctx, stage, mode=mode,
                                              split="DEVELOPMENT")
    plan = sd.eval_plan(ctx, stage, split="DEVELOPMENT")
    per_family = max(1, int(count) // int(plan["n_families"]))
    episodes = sd.load_episodes(ctx.pregen_root, "DEVELOPMENT", mode,
                                guard=ctx.guard, limit_per_family=per_family)
    sd.assert_eval_family_coverage(
        episodes, "DEVELOPMENT",
        context=f"{getattr(stage, 'stage_id', '?')} evaluation set")
    return episodes


def train_episodes(ctx: Any, count: int, *, mode: str = "scripted") -> list[Any]:
    episodes = load_split_episodes(ctx, "TRAIN", mode,
                                   limit_per_family=ctx.train_example_cap)
    return episodes[:int(count)]


def count_scoreable_items(episodes: Sequence[Any]) -> list[int]:
    """Feedback items per episode (the FL4 data-sufficiency evidence, §10)."""
    from foundation_learner.episodes.schema import FEEDBACK_ROLES

    return [sum(1 for e in ep.events if e.role in FEEDBACK_ROLES)
            for ep in episodes]


def trainable_params(ctx: Any, model: Any):
    """The arm's trainable-parameter scope (W2 ``peft_modes.apply_mode``)."""
    from foundation_learner.training.peft_modes import apply_mode

    return apply_mode(model, ctx.scope, seed=int(ctx.root_seed))


# --------------------------------------------------------------------------
# evaluation + promotion evidence
# --------------------------------------------------------------------------
def dev_records(bundle: Any, episodes: Sequence[Any], *, ctx: Any, arm_tag: str,
                callbacks_factory: Callable[[Any], Any] | None = None,
                context_mode: str = "history",
                reset_from_index: int = 4) -> list[dict]:
    """Online DEV evaluation, tagged ``split="DEVELOPMENT"`` for promotion.

    The bundle is forced into evaluation state first (eval mode, layer-level
    gradient checkpointing off).  A mechanism rung that has just trained a
    module would otherwise decode with ``use_cache``/``past_key_values``
    silently dropped by transformers' ``GradientCheckpointingLayer`` and
    produce different text than the same weights evaluated normally
    (Amendment 16).
    """
    from foundation_learner.evaluation.learning_curve import (
        LearningCurveConfig, annotate_records, run_episodes)

    _stage_definitions().prepare_bundle_for_evaluation(bundle)
    cfg = LearningCurveConfig(arm_tag=arm_tag, context_mode=context_mode,
                              reset_from_index=int(reset_from_index),
                              generation=generation_config(ctx))
    records = run_episodes(bundle, list(episodes), env_factory, cfg=cfg,
                           callbacks_factory=callbacks_factory)
    return annotate_records(records, split="DEVELOPMENT")


def dev_metrics_dict(records: Sequence[dict], stage: str,
                     stability_events: Sequence[str] = ()) -> dict:
    from foundation_learner.campaign import promotion

    return promotion.DevMetrics.from_records(
        records, stage=stage, stability_events=stability_events).to_dict()


def macro_aulc(records: Sequence[dict]) -> float | None:
    from foundation_learner.evaluation import metrics as _metrics

    return _metrics.macro_aulc(records)


def post_reset_gain_records(records: Sequence[dict], *, reset_from_index: int = 4
                            ) -> list[dict]:
    """Per-episode ``mean(R_k, k >= reset_from_index) - R_0``.

    This is the RETAINED IMPROVEMENT of one episode after its textual history
    was removed: the quantity contract §10's FL8 gate calls "context-reset
    persistence".  Returned as records so the family-clustered bootstrap can
    consume them unchanged.
    """
    from foundation_learner.evaluation import metrics as _metrics

    out = []
    for record in records:
        R = _metrics.record_R(record)
        if 0 not in R:
            continue
        post = [v for k, v in R.items() if k >= int(reset_from_index)]
        if not post:
            continue
        out.append(dict(record, _persistence_value=float(
            sum(post) / len(post) - R[0])))
    return out


def paired_persistence_evidence(mechanism_records: Sequence[dict],
                                baseline_records: Sequence[dict], *,
                                reset_from_index: int = 4,
                                label: str = "FL5",
                                seed: int = 0) -> dict:
    """Paired family-clustered bootstrap of retained gain, mechanism - baseline.

    Contract §10 admits FL8 on "FL5 or FL7 context-reset persistence > 0 with
    CI excluding 0 on DEV".  The estimand here is the DIFFERENCE in retained
    post-reset improvement between the mechanism arm and its matched
    no-mechanism arm on the SAME episodes, which is what makes the claim about
    the mechanism rather than about the task.  §14's clustered bootstrap
    (families with replacement, then episodes within family, one shared
    resample plan for both arms) is used unchanged.
    """
    from foundation_learner.analysis.stats import (clustered_sample_from_records,
                                                   paired_clustered_bootstrap)

    a = post_reset_gain_records(mechanism_records, reset_from_index=reset_from_index)
    b = post_reset_gain_records(baseline_records, reset_from_index=reset_from_index)
    value_fn = lambda r: r.get("_persistence_value")  # noqa: E731
    sample_a = clustered_sample_from_records(a, value_fn)
    sample_b = clustered_sample_from_records(b, value_fn)
    result = paired_clustered_bootstrap(sample_a, sample_b, seed=int(seed))
    return {
        "source": label,
        "split": "DEVELOPMENT",
        "reset_from_index": int(reset_from_index),
        "point": result["estimate"],
        "ci": [result["ci_low"], result["ci_high"]],
        "excludes_zero": result["excludes_zero"],
        "bootstrap": result,
    }


def publish_persistence_evidence(ctx: Any, evidence: dict, *, key: str) -> dict:
    """Record this rung's persistence evidence for the FL8 entry condition.

    ``campaign.stage_definitions.fl8_entry`` reads
    ``ctx.extra["persistence_evidence"]`` (``point`` + ``ci``).  §10 accepts
    FL5 OR FL7 evidence, so the first rung to produce a DEFINED estimate
    publishes it and every rung also records its own under its own key; a later
    rung never overwrites an earlier defined estimate.
    """
    ctx.extra[key] = evidence
    if evidence.get("point") is None:
        return evidence
    existing = ctx.extra.get("persistence_evidence")
    if not existing or existing.get("point") is None:
        ctx.extra["persistence_evidence"] = evidence
    return evidence


# --------------------------------------------------------------------------
# pre-generated side tables (all guarded reads)
# --------------------------------------------------------------------------
def _guarded_jsonl(ctx: Any, *parts: str) -> list[dict]:
    from foundation_learner.campaign import o1_isolation
    from foundation_learner.data.shards import read_shard

    root = ctx.guard.guard(os.path.join(ctx.pregen_root, *parts),
                           o1_isolation.MODE_READ)
    if not os.path.isdir(root):
        raise stage_error(f"missing pre-generated directory {root!r}")
    rows: list[dict] = []
    for name in ctx.guard.listdir(root):
        if name.endswith(".jsonl"):
            rows.extend(read_shard(os.path.join(root, name)))
    return rows


def load_ablation_plans(ctx: Any, split: str) -> dict[str, dict]:
    """``episode_id -> FL4 ablation plan`` from the pre-generated shards."""
    if str(split) == "SEALED_TEST":
        raise stage_error("no FL4 ablation plans exist for SEALED_TEST")
    return {str(row["episode_id"]): row
            for row in _guarded_jsonl(ctx, "fl4_ablations", split)}


def load_poison_records(ctx: Any, split: str) -> dict[str, dict]:
    """``episode_id -> poison variant table`` (the frozen §9 conditions)."""
    return {str(row["episode_id"]): row
            for row in _guarded_jsonl(ctx, "poison", split)}


def load_chain_specs(ctx: Any, episodes: Sequence[Any], *, split: str = "DEVELOPMENT",
                     limit: int | None = None) -> list[Any]:
    """Pre-generated A->B->A chains (the campaign's resolver).

    Delegates to ``campaign.stage_definitions.load_chain_specs``, which
    completes the episode pool from the split's own shards when a chain names
    an episode the caller does not hold, and selects chains ROUND-ROBIN over
    the ordered family pairs.  The previous implementation dropped every chain
    whose three episodes were not all inside the caller's capped subset and, on
    a limit, returned chains of a single ordered pair — neither of which is the
    family-balanced interference metric of contract §9 (review finding R-C3).
    """
    return _stage_definitions().load_chain_specs(ctx, episodes, split=split,
                                                 limit=limit)


def reveal_variant(episode: Any) -> Any:
    """The SAME episode re-assembled with its REVEAL events present.

    FL7's inner loop is supervised by the REVEALED support answer, but the
    standard pools are pre-generated with ``reveal=False`` (Amendment 7 item 7)
    rather than doubling every pool.  ``assemble_episode`` is a pure function of
    ``(family, rule, split, difficulty, root_seed, episode_seed, mode,
    condition, structured, reveal, arm_tag)``, so re-assembling with
    ``reveal=True`` reproduces the SAME latent rule, the SAME items and the
    SAME scripted attempts, plus the REVEAL events.  Nothing is fabricated:
    W1's own assembler builds it.
    """
    from foundation_learner.ecology.base import Rule
    from foundation_learner.ecology.families import get_family
    from foundation_learner.episodes.assemble import assemble_episode

    if getattr(episode, "reveal", False):
        return episode
    family = get_family(episode.family_id)
    spec = episode.rule_spec or {}
    rule = Rule(family_id=str(spec.get("family_id", episode.family_id)),
                params=dict(spec.get("params", {})))
    meta = episode.meta or {}
    return assemble_episode(
        family, rule, split=episode.split, difficulty=int(episode.difficulty),
        root_seed=int(meta.get("root_seed", 0)), episode_seed=int(episode.seed),
        mode=str(episode.mode), condition=str(episode.condition),
        structured=bool(episode.structured), reveal=True,
        arm_tag=str(meta.get("arm_tag", "")))


# --------------------------------------------------------------------------
# the FL4 head as a cross-stage artefact
# --------------------------------------------------------------------------
HEAD_STATE_NAME = "fl4_value_head.pt"
HEAD_CTX_KEY = "fl4_value_head"


def save_head_state(ctx: Any, stage_id: str, head: Any) -> str:
    """Persist the trained FL4 head next to its stage report (guarded write)."""
    import torch

    path = os.path.join(ctx.stage_dir(stage_id), HEAD_STATE_NAME)
    with ctx.guard.open(path, "wb") as fh:
        torch.save({"state_dict": head.state_dict(), "config": head.config(),
                    "config_hash": head.config_hash()}, fh)
    return path


def load_head_state(ctx: Any, path: str) -> Any | None:
    import torch

    from foundation_learner.mechanisms.value_head import ValueHead

    if not os.path.isfile(ctx.guard.guard(path)):
        return None
    with ctx.guard.open(path, "rb") as fh:
        payload = torch.load(fh, weights_only=False)
    config = payload["config"]
    head = ValueHead(int(config["hidden_size"]), inner=int(config["inner"]),
                     seed=int(config["seed"]))
    head.load_state_dict(payload["state_dict"])
    return head.freeze_()


def value_head_from_ctx(ctx: Any) -> Any | None:
    """The FL4 head produced earlier in this ladder, or ``None``.

    In-process first (the object the FL4 stage published), then the artefact
    the FL4 stage wrote.  ``None`` means FL4 did not run in this session: the
    caller must record a skip rather than invent a head.
    """
    head = ctx.extra.get(HEAD_CTX_KEY)
    if head is not None:
        return head
    path = os.path.join(ctx.out_dir, "fl4", HEAD_STATE_NAME)
    try:
        return load_head_state(ctx, path)
    except FileNotFoundError:
        return None


def arm_checkpoint_state(ctx: Any, bundle: Any, arm_id: str, *,
                        tag: str = "final") -> dict:
    """Load a completed core arm's trained state into ``bundle`` if it exists.

    Contract §7 computes the FL4 targets "with the final FL3 checkpoint".  The
    stage context hands out a FRESH BASE bundle, so the arm's trained state is
    restored here through W2's own checkpoint and LoRA primitives.  When the
    arm's checkpoint is absent (FL3 not run in this session) the base
    checkpoint is used and the payload RECORDS that fact — it is never passed
    off as the FL3 model.
    """
    from foundation_learner.training.checkpointing import load_checkpoint

    out_dir = os.path.join(ctx.out_dir, arm_id.lower(), "arm")
    manifest_path = os.path.join(out_dir, f"ckpt_{tag}.manifest.json")
    if not os.path.isfile(ctx.guard.guard(manifest_path)):
        return {"loaded": False, "arm_id": arm_id,
                "model": "BASE_CHECKPOINT_NO_TRAINED_ARM",
                "reason": f"no {tag!r} checkpoint for {arm_id} under {out_dir!r}"}
    payload = load_checkpoint(out_dir, tag)
    scope = payload["manifest"].get("trained_scope")
    state = payload["trained_state"]
    if scope == "FULL_MODEL_MODE":
        bundle.model.load_state_dict(state, strict=True)
        handles = None
    else:
        from foundation_learner.training.lora import load_lora_state_

        params = trainable_params(ctx, bundle.model)
        handles = params.lora_handles
        load_lora_state_(handles, state)
    return {"loaded": True, "arm_id": arm_id, "tag": tag,
            "trained_scope": scope,
            "manifest_hash": payload["manifest"].get("manifest_hash"),
            "model": f"{arm_id}_{tag}"}
