"""The production core-comparison audit (contract §8, §1, §15; V-Gap4).

``assert_core_arms_matched``, ``compare_core_ledgers`` and the base-identity
check existed but were invoked nowhere outside the test suite, so nothing in a
real campaign would have refused an unmatched comparison.  These tests exercise
the production invocation path (``campaign.core_matching`` +the
``CORE_MATCHING`` stage) with real ``ArmConfig`` objects, real compute ledgers
and real checkpoint manifests.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from foundation_learner.campaign import core_matching as cm
from foundation_learner.campaign import o1_isolation
from foundation_learner.campaign import stage_definitions as sd
from foundation_learner.training import checkpointing as cp
from foundation_learner.training import model_loading as ml
from foundation_learner.training.arms import make_arm_config
from foundation_learner.training.compute_accounting import (COMPUTE_LEDGER_SCHEMA,
                                                            ComputeMismatchError)

CORE = ("FL1", "FL2", "FL3")


class _Cfg:
    def __init__(self, **kw):
        self.model_type = "ouro"
        self.architectures = ["OuroForCausalLM"]
        self.hidden_size = 2048
        self.num_hidden_layers = 48
        self.num_attention_heads = 16
        self.total_ut_steps = 4
        self.early_exit_threshold = 1.0
        self.vocab_size = 49152
        self.__dict__.update(kw)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def identity(tree=ml.FROZEN_CHECKPOINT_TREE_SHA256, **overrides):
    return ml.build_identity(
        kind="FROZEN_BACKBONE", scientific=True,
        checkpoint_dir="/artifacts/ouro_rltt_local", tree_sha256=tree,
        tree_hash_verified=True, config=_Cfg(**overrides), dtype="bfloat16",
        device="cuda")


def ledger(arm_id, *, updates=600, tokens=4096, loss_tokens=1000):
    return {
        "schema": COMPUTE_LEDGER_SCHEMA,
        "arm_id": arm_id,
        "stage": "CORE",
        "optimizer_updates": updates,
        "max_tokens_per_batch": tokens,
        "loss_tokens": loss_tokens,
        "forward_tokens": 10_000,
        "backward_tokens": 10_000,
        "wall_time_s": 10.0,
        "gpu_seconds": 10.0,
        "examples": 10,
        "episodes": 10,
        "skipped_updates": 0,
        "device": "cpu",
    }


def build_ctx(tmp_path, *, arm_identities=None, ledgers=None, configs=None):
    guard = o1_isolation.IsolationGuard(label="TEST_CORE_MATCHING")
    ctx = sd.StageContext(out_dir=str(tmp_path / "ladder"),
                          pregen_root=str(tmp_path), bundle_factory=lambda: None,
                          guard=guard)
    arm_identities = arm_identities or {a: identity() for a in CORE}
    ledgers = ledgers or {a: ledger(a) for a in CORE}
    configs = configs or {
        a: make_arm_config(a, learning_rate=1e-4, updates=600,
                           max_tokens_per_batch=4096, peft_mode="PEFT_MODE",
                           seed=20260809, stage="CORE",
                           checkpoint_every_steps=200,
                           checkpoint_every_seconds=600.0, max_seq_len=2048)
        for a in CORE}
    for arm in CORE:
        out_dir = os.path.join(ctx.out_dir, arm.lower(), "arm")
        cp.save_checkpoint(out_dir, "final",
                           trained_state={"w": torch.zeros(2)},
                           trained_scope="PEFT_MODE", step=600,
                           arm_config={"arm_id": arm},
                           arm_config_hash=f"hash-{arm}",
                           base_identity=arm_identities[arm])
        ctx.results[arm] = {"arm_result": {"ledger": ledgers[arm],
                                           "arm_config_hash": f"hash-{arm}"}}
        ctx.results.setdefault("_arm_configs", {})[arm] = configs[arm]
        ctx.results.setdefault("_arm_out_dirs", {})[arm] = out_dir
    return ctx


def test_a_matched_comparison_produces_the_report_artifact(tmp_path):
    ctx = build_ctx(tmp_path)
    payload = sd.core_matching_work(ctx, sd.STAGES_BY_ID["CORE_MATCHING"])
    assert payload["schema"] == "flb200.core_matching_report.v1"
    assert payload["arms"] == list(CORE)
    assert payload["base_identity"]["base_checkpoint_tree_sha256"] == \
        ml.FROZEN_CHECKPOINT_TREE_SHA256
    assert payload["compute_comparison"]["matched"]["optimizer_updates"] == 600
    # the intrinsically differing quantities are REPORTED, never equalised
    assert "loss_tokens" in payload["compute_comparison"]["reported_differences"]
    written = json.loads(open(os.path.join(ctx.out_dir, "core_matching",
                                           "core_matching_report.json"),
                              encoding="utf-8").read())
    assert written["arms"] == list(CORE)


def test_unequal_realized_compute_is_refused(tmp_path):
    ledgers = {a: ledger(a) for a in CORE}
    ledgers["FL3"] = ledger("FL3", updates=1200)          # hidden extra compute
    ctx = build_ctx(tmp_path, ledgers=ledgers)
    with pytest.raises(ComputeMismatchError):
        cm.build_core_matching_report(ctx, CORE)


def test_a_mixed_trainable_mode_is_refused(tmp_path):
    configs = {
        a: make_arm_config(a, learning_rate=1e-4 if a != "FL3" else 1e-5,
                           updates=600, max_tokens_per_batch=4096,
                           peft_mode="PEFT_MODE" if a != "FL3" else "FULL_MODEL_MODE",
                           seed=20260809, stage="CORE",
                           checkpoint_every_steps=200,
                           checkpoint_every_seconds=600.0, max_seq_len=2048)
        for a in CORE}
    ctx = build_ctx(tmp_path, configs=configs)
    with pytest.raises(Exception) as exc:
        cm.build_core_matching_report(ctx, CORE)
    assert "REFUSED" in str(exc.value) or "mode" in str(exc.value).lower()


def test_different_base_checkpoints_are_refused(tmp_path):
    identities = {a: identity() for a in CORE}
    identities["FL2"] = identity(tree="0" * 64)
    ctx = build_ctx(tmp_path, arm_identities=identities)
    with pytest.raises(cm.CoreMatchingError) as exc:
        cm.build_core_matching_report(ctx, CORE)
    assert "one identical frozen checkpoint" in str(exc.value)


def test_unpinned_recurrence_is_a_different_base(tmp_path):
    identities = {a: identity() for a in CORE}
    identities["FL3"] = identity(total_ut_steps=1)
    ctx = build_ctx(tmp_path, arm_identities=identities)
    with pytest.raises(cm.CoreMatchingError):
        cm.build_core_matching_report(ctx, CORE)


def test_an_incomplete_comparison_is_refused(tmp_path):
    ctx = build_ctx(tmp_path)
    ctx.results.pop("FL2")
    with pytest.raises(cm.CoreMatchingError) as exc:
        cm.build_core_matching_report(ctx, CORE)
    assert "incomplete" in str(exc.value)


def test_a_nonscientific_rehearsal_may_have_no_tree_hash_but_says_so(tmp_path):
    """The tiny mechanics model has no frozen tree hash; that is recorded."""
    tiny = ml.build_identity(
        kind="TINY_NONSCIENTIFIC", scientific=False, checkpoint_dir=None,
        tree_sha256=None, tree_hash_verified=False,
        config=_Cfg(hidden_size=64, num_hidden_layers=2, total_ut_steps=2),
        dtype="float32", device="cpu")
    ctx = build_ctx(tmp_path, arm_identities={a: tiny for a in CORE})
    report = cm.build_core_matching_report(ctx, CORE)
    assert report["base_identity"]["base_checkpoint_tree_sha256"] is None
    assert "NONSCIENTIFIC" in report["base_identity"]["note"]


def test_a_scientific_arm_without_a_tree_hash_is_refused(tmp_path):
    """A SCIENTIFIC arm must name the frozen checkpoint it came from."""
    ctx = build_ctx(tmp_path,
                    arm_identities={a: identity(tree=None) for a in CORE})
    with pytest.raises(cm.CoreMatchingError) as exc:
        cm.build_core_matching_report(ctx, CORE)
    assert "claim to be scientific" in str(exc.value)


def test_a_missing_checkpoint_manifest_is_refused(tmp_path):
    ctx = build_ctx(tmp_path)
    os.remove(os.path.join(ctx.results["_arm_out_dirs"]["FL1"],
                           "ckpt_final.manifest.json"))
    with pytest.raises(cm.CoreMatchingError) as exc:
        cm.build_core_matching_report(ctx, CORE)
    assert "cannot be audited" in str(exc.value)
