"""Atomic checkpoints and bitwise-resumable training (contract §15, §17)."""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pytest
import torch

from foundation_learner.training import checkpointing as cp
from foundation_learner.training import model_loading as ml
from foundation_learner.training.arms import ArmConfig, STAGE_SMOKE
from foundation_learner.training.objectives import objective_for_arm
from foundation_learner.training.tiny_model import build_tiny_model
from foundation_learner.training.tokenization import iter_examples
from foundation_learner.training.trainer import TrainerHooks, run_training_arm

from foundation_learner.tests.test_training_stubs import (
    make_ordered_episode,
    render_episode,
)

requires_checkpoint_dir = pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory (code + tokenizer) not present",
)


# ---------------------------------------------------------------------------
# Unit level
# ---------------------------------------------------------------------------


def _save(tmp_path, tag="initial", **overrides):
    kwargs = dict(
        trained_state={"w": torch.arange(4, dtype=torch.float32)},
        trained_scope="PEFT_MODE",
        step=7,
        tokens=1234,
        loss_tokens=99,
        arm_config={"arm_id": "FL3", "learning_rate": 1e-4},
        arm_config_hash="deadbeef",
        base_identity={
            "identity_hash": "abc123",
            "checkpoint_tree_sha256": ml.FROZEN_CHECKPOINT_TREE_SHA256,
        },
        family_split_hash="split123",
        shard_hashes={"train-000.jsonl": "aa" * 32},
        stage="CORE",
    )
    kwargs.update(overrides)
    return cp.save_checkpoint(str(tmp_path), tag, **kwargs)


def test_checkpoint_records_every_contract_section_15_field(tmp_path):
    manifest = _save(tmp_path)
    for key in (
        "base_checkpoint_tree_sha256",
        "base_identity_hash",
        "arm_config_hash",
        "trained_scope",
        "stage",
        "step",
        "tokens",
        "loss_tokens",
        "family_split_hash",
        "shard_hashes",
        "code_commit",
        "environment",
        "payload_sha256",
    ):
        assert key in manifest, key
    assert manifest["base_checkpoint_tree_sha256"] == ml.FROZEN_CHECKPOINT_TREE_SHA256
    assert manifest["code_commit"]["commit"] is not None
    assert len(manifest["environment"]["digest"]) == 64

    payload = cp.load_checkpoint(str(tmp_path), "initial")
    assert set(payload["rng_state"]) >= {"python", "torch", "numpy_legacy", "named"}
    assert payload["manifest"]["step"] == 7


def test_writes_are_atomic_and_leave_no_temporary_files(tmp_path):
    _save(tmp_path)
    names = sorted(os.listdir(tmp_path))
    assert names == ["ckpt_initial.manifest.json", "ckpt_initial.pt"]
    assert not any(name.endswith(".tmp") for name in names)


def test_corrupted_payload_is_detected(tmp_path):
    _save(tmp_path)
    path = os.path.join(tmp_path, "ckpt_initial.pt")
    with open(path, "r+b") as fh:
        fh.seek(64)
        fh.write(b"\x00\x01\x02\x03")
    with pytest.raises(cp.CorruptCheckpointError):
        cp.load_checkpoint(str(tmp_path), "initial")


def test_missing_manifest_sidecar_is_refused(tmp_path):
    _save(tmp_path)
    os.remove(os.path.join(tmp_path, "ckpt_initial.manifest.json"))
    with pytest.raises(cp.CorruptCheckpointError):
        cp.load_checkpoint(str(tmp_path), "initial")


def test_rng_states_round_trip_bitwise(tmp_path):
    generator = np.random.Generator(np.random.PCG64(12345))
    torch_gen = torch.Generator().manual_seed(999)
    random.seed(7)
    torch.manual_seed(8)
    np.random.seed(9)
    _ = generator.random(3)
    _ = torch.rand(2, generator=torch_gen)

    state = cp.capture_rng_state({"data": generator, "torch_data": torch_gen})
    expected = (
        random.random(),
        float(torch.rand(1)),
        float(np.random.rand()),
        float(generator.random()),
        float(torch.rand(1, generator=torch_gen)),
    )
    # perturb everything
    random.seed(1)
    torch.manual_seed(2)
    np.random.seed(3)
    generator.random(50)
    torch.rand(10, generator=torch_gen)

    cp.restore_rng_state(state, {"data": generator, "torch_data": torch_gen})
    got = (
        random.random(),
        float(torch.rand(1)),
        float(np.random.rand()),
        float(generator.random()),
        float(torch.rand(1, generator=torch_gen)),
    )
    assert got == expected


def test_missing_generator_state_is_refused(tmp_path):
    state = cp.capture_rng_state({})
    with pytest.raises(cp.CheckpointError):
        cp.restore_rng_state(state, {"data": np.random.Generator(np.random.PCG64(1))})


def test_prune_keeps_the_retained_tags(tmp_path):
    for tag in ("initial", "step200", "step400", "best_dev", "final"):
        _save(tmp_path, tag=tag)
    removed = cp.prune_checkpoints(str(tmp_path))
    assert sorted(removed) == ["step200", "step400"]
    assert sorted(cp.list_checkpoints(str(tmp_path))) == ["best_dev", "final", "initial"]


def test_invalid_tag_is_refused(tmp_path):
    with pytest.raises(cp.CheckpointError):
        _save(tmp_path, tag="../escape")


def test_manifest_json_is_sorted_and_newline_terminated(tmp_path):
    _save(tmp_path)
    raw = open(os.path.join(tmp_path, "ckpt_initial.manifest.json"), encoding="utf-8").read()
    assert raw.endswith("\n")
    assert json.loads(raw)["schema"] == "flb200.checkpoint_manifest.v1"


# ---------------------------------------------------------------------------
# End-to-end: bitwise resume
# ---------------------------------------------------------------------------


def _smoke_cfg(updates: int, arm: str = "FL3", **overrides):
    kwargs = dict(
        arm_id=arm,
        data_mode="ORDERED_EPISODES",
        objective=objective_for_arm(arm),
        learning_rate=1e-3,
        updates=updates,
        max_tokens_per_batch=256,
        peft_mode="PEFT_MODE",
        seed=20260809,
        stage=STAGE_SMOKE,
        max_seq_len=256,
        gradient_checkpointing=False,
        bf16=False,
        checkpoint_every_steps=0,
        checkpoint_every_seconds=0.0,
    )
    kwargs.update(overrides)
    return ArmConfig(**kwargs)


@requires_checkpoint_dir
def test_resume_reproduces_the_uninterrupted_loss_trajectory(tmp_path):
    bundle = build_tiny_model(seed=31)
    episodes = [make_ordered_episode(k) for k in range(4)]
    examples = list(iter_examples(episodes, bundle.tokenizer, "FL3", render_fn=render_episode))

    cfg = _smoke_cfg(12)
    reference = run_training_arm(
        cfg,
        bundle,
        examples,
        str(tmp_path / "reference"),
        validate_config=False,
    )
    assert reference.steps == 12

    # interrupted run: stop after 10 steps, then resume from the final tag
    interrupted_bundle = build_tiny_model(seed=31)
    hooks = TrainerHooks(on_step=lambda state: "STOP" if state["step"] >= 6 else None)
    first_half = run_training_arm(
        cfg,
        interrupted_bundle,
        examples,
        str(tmp_path / "resumed"),
        hooks,
        validate_config=False,
    )
    assert first_half.steps == 6
    assert first_half.stopped_reason == "STOPPED_BY_HOOK"

    resumed_bundle = build_tiny_model(seed=31)
    second_half = run_training_arm(
        cfg,
        resumed_bundle,
        examples,
        str(tmp_path / "resumed"),
        validate_config=False,
        resume_from_tag="final",
    )
    assert second_half.steps == 12
    assert first_half.loss_history == reference.loss_history[:6]
    assert second_half.loss_history == reference.loss_history[6:]
    assert second_half.lr_history == reference.lr_history[6:]

    # and the parameters agree bitwise
    ref_state = cp.load_checkpoint(str(tmp_path / "reference"), "final")["trained_state"]
    res_state = cp.load_checkpoint(str(tmp_path / "resumed"), "final")["trained_state"]
    assert set(ref_state) == set(res_state)
    for key in ref_state:
        assert torch.equal(ref_state[key], res_state[key]), key


@requires_checkpoint_dir
def test_resume_refuses_a_different_arm_config_or_base_identity(tmp_path):
    bundle = build_tiny_model(seed=31)
    episodes = [make_ordered_episode(k) for k in range(2)]
    examples = list(iter_examples(episodes, bundle.tokenizer, "FL3", render_fn=render_episode))
    out = str(tmp_path / "arm")
    run_training_arm(_smoke_cfg(2), bundle, examples, out, validate_config=False)

    other = build_tiny_model(seed=31)
    with pytest.raises(Exception) as exc:
        run_training_arm(
            _smoke_cfg(2, learning_rate=2e-3),
            other,
            examples,
            out,
            validate_config=False,
            resume_from_tag="final",
        )
    assert "arm config hash" in str(exc.value)

    different_base = build_tiny_model(seed=999)
    with pytest.raises(Exception) as exc2:
        run_training_arm(
            _smoke_cfg(2),
            different_base,
            examples,
            out,
            validate_config=False,
            resume_from_tag="final",
        )
    assert "base model identity" in str(exc2.value)
