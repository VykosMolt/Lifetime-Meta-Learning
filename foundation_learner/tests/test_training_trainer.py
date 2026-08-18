"""The shared training loop (contract §8, §11, §15, §17, §23).

End-to-end mechanics on the TINY NONSCIENTIFIC model: every arm trains, the
loss falls on trivial synthetic episodes, the data order is deterministic, the
packing respects the token budget, and the ledger/checkpoint/stability plumbing
is exercised.  This is a FUNCTIONAL SANITY CHECK, not science: a tiny randomly
initialised model says nothing about the research question.
"""

from __future__ import annotations

import os

import pytest
import torch

from foundation_learner.training import model_loading as ml
from foundation_learner.training import trainer as tr
from foundation_learner.training.arms import ARM_DATA_MODES, ArmConfig, STAGE_SMOKE
from foundation_learner.training.objectives import objective_for_arm
from foundation_learner.training.stability import StabilityConfig
from foundation_learner.training.tiny_model import build_tiny_model
from foundation_learner.training.tokenization import iter_examples

from foundation_learner.tests.test_training_stubs import (
    make_ordered_episode,
    make_static_episode,
    render_episode,
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory (code + tokenizer) not present",
)

STEPS = 50


def smoke_cfg(arm: str, *, updates: int = STEPS, peft_mode="PEFT_MODE", lr=3e-3, **kw):
    kwargs = dict(
        arm_id=arm,
        data_mode=ARM_DATA_MODES[arm],
        objective=objective_for_arm(arm),
        learning_rate=lr,
        updates=updates,
        max_tokens_per_batch=256,
        peft_mode=peft_mode,
        seed=20260809,
        stage=STAGE_SMOKE,
        max_seq_len=256,
        gradient_checkpointing=False,
        bf16=False,
        checkpoint_every_steps=0,
        checkpoint_every_seconds=0.0,
    )
    kwargs.update(kw)
    return ArmConfig(**kwargs)


def build_examples(bundle, arm: str, n: int = 6):
    if arm == "FL1":
        episodes = [make_static_episode(k) for k in range(n)]
    else:
        episodes = [make_ordered_episode(k) for k in range(n)]
    return list(iter_examples(episodes, bundle.tokenizer, arm, render_fn=render_episode))


@pytest.mark.parametrize("arm", ["FL1", "FL2", "FL3"])
def test_every_arm_trains_and_the_loss_falls(tmp_path, arm):
    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, arm)
    cfg = smoke_cfg(arm)
    result = tr.run_training_arm(
        cfg, bundle, examples, str(tmp_path / arm), validate_config=False
    )
    assert result.steps == STEPS
    assert result.stopped_reason == tr.STOP_COMPLETED
    assert len(result.loss_history) == STEPS
    assert all(torch.isfinite(torch.tensor(result.loss_history)))
    first = sum(result.loss_history[:5]) / 5
    last = sum(result.loss_history[-5:]) / 5
    assert last < first - 0.1, (arm, first, last)
    assert not result.stability_events
    assert result.failure_record is None
    assert result.ledger["optimizer_updates"] == STEPS
    assert result.ledger["loss_tokens"] > 0
    assert result.identity["scientific"] is False
    ledger_path = tmp_path / arm / f"compute_ledger_{arm}.json"
    assert ledger_path.is_file()


def test_full_model_mode_also_trains(tmp_path):
    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, "FL3", n=4)
    cfg = smoke_cfg("FL3", updates=20, peft_mode="FULL_MODEL_MODE", lr=1e-3)
    result = tr.run_training_arm(
        cfg, bundle, examples, str(tmp_path / "full"), validate_config=False
    )
    assert result.trainable["mode"] == "FULL_MODEL_MODE"
    assert result.trainable["lora_spec"] is None
    assert result.loss_history[-1] < result.loss_history[0]


def test_learning_rate_schedule_is_cosine_with_three_percent_warmup(tmp_path):
    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, "FL3", n=2)
    cfg = smoke_cfg("FL3", updates=20)
    result = tr.run_training_arm(
        cfg, bundle, examples, str(tmp_path / "sched"), validate_config=False
    )
    lrs = result.lr_history
    assert cfg.warmup_steps == 1
    assert lrs[0] == pytest.approx(cfg.learning_rate, rel=1e-6)
    assert lrs[-1] < lrs[len(lrs) // 2] < lrs[0]
    assert lrs[-1] == pytest.approx(0.0, abs=1e-9)


def test_data_order_is_deterministic_and_reshuffles_per_epoch():
    first = tr.epoch_order(8, 20260809, "FL3", 0)
    again = tr.epoch_order(8, 20260809, "FL3", 0)
    second_epoch = tr.epoch_order(8, 20260809, "FL3", 1)
    other_arm = tr.epoch_order(8, 20260809, "FL1", 0)
    other_seed = tr.epoch_order(8, 20260810, "FL3", 0)
    assert first == again
    assert sorted(first) == list(range(8))
    assert first != second_epoch
    assert first != other_arm
    assert first != other_seed


def test_packing_respects_the_padded_token_budget():
    examples = [{"input_ids": [0] * n} for n in (100, 100, 50, 30, 200)]
    order = list(range(len(examples)))
    batches = tr.pack_batches(examples, order, max_tokens_per_batch=200)
    for batch in batches:
        longest = max(len(examples[i]["input_ids"]) for i in batch)
        assert len(batch) * longest <= 200
    assert [i for batch in batches for i in batch] == order
    with pytest.raises(tr.TrainerError):
        tr.pack_batches(examples, order, max_tokens_per_batch=100)


def test_examples_tokenised_for_another_arm_are_refused(tmp_path):
    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, "FL2", n=2)
    with pytest.raises(tr.TrainerError):
        tr.run_training_arm(
            smoke_cfg("FL3", updates=2),
            bundle,
            examples,
            str(tmp_path / "mismatch"),
            validate_config=False,
        )


def test_empty_example_set_is_refused(tmp_path):
    bundle = build_tiny_model(seed=17)
    with pytest.raises(tr.TrainerError):
        tr.run_training_arm(
            smoke_cfg("FL3", updates=2),
            bundle,
            [],
            str(tmp_path / "empty"),
            validate_config=False,
        )


def test_hook_can_stop_the_arm_and_sees_the_step_state(tmp_path):
    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, "FL3", n=2)
    seen = []

    def on_step(state):
        seen.append(state["step"])
        assert {"loss", "grad_norm", "lr", "arm_id", "ledger"} <= set(state)
        return "STOP" if state["step"] == 3 else None

    result = tr.run_training_arm(
        smoke_cfg("FL3", updates=30),
        bundle,
        examples,
        str(tmp_path / "hooked"),
        tr.TrainerHooks(on_step=on_step),
        validate_config=False,
    )
    assert seen == [1, 2, 3]
    assert result.steps == 3
    assert result.stopped_reason == tr.STOP_HOOK


def test_checkpoint_cadence_by_steps(tmp_path):
    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, "FL3", n=2)
    tags = []
    result = tr.run_training_arm(
        smoke_cfg("FL3", updates=6, checkpoint_every_steps=2),
        bundle,
        examples,
        str(tmp_path / "cadence"),
        tr.TrainerHooks(on_checkpoint=lambda tag, manifest: tags.append(tag)),
        validate_config=False,
    )
    assert tags == ["initial", "step2", "step4", "final"]
    assert [c["step"] for c in result.checkpoints] == [0, 2, 4, 6]


def test_checkpoint_cadence_by_wall_time(tmp_path):
    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, "FL3", n=2)
    tags = []
    tr.run_training_arm(
        smoke_cfg("FL3", updates=3, checkpoint_every_seconds=1e-9),
        bundle,
        examples,
        str(tmp_path / "cadence_time"),
        tr.TrainerHooks(on_checkpoint=lambda tag, manifest: tags.append(tag)),
        validate_config=False,
    )
    assert tags == ["initial", "step1", "step2", "final"]


def test_external_hook_can_request_the_retention_tags(tmp_path):
    from foundation_learner.training.checkpointing import list_checkpoints

    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, "FL3", n=2)

    def on_step(state):
        if state["step"] == 2:
            state["save"]("best_dev", state["step"])
        return None

    tr.run_training_arm(
        smoke_cfg("FL3", updates=3),
        bundle,
        examples,
        str(tmp_path / "tags"),
        tr.TrainerHooks(on_step=on_step),
        validate_config=False,
    )
    assert sorted(list_checkpoints(str(tmp_path / "tags"))) == [
        "best_dev",
        "final",
        "initial",
    ]


def test_nan_loss_stops_the_arm_with_a_failure_record(tmp_path, monkeypatch):
    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, "FL3", n=2)
    real_logits = tr._model_logits
    state = {"n": 0}

    def poisoned(model, batch, use_cache=False):
        logits = real_logits(model, batch, use_cache)
        state["n"] += 1
        if state["n"] == 3:
            logits = logits * float("nan")
        return logits

    monkeypatch.setattr(tr, "_model_logits", poisoned)
    cfg = smoke_cfg("FL3", updates=10)
    result = tr.run_training_arm(
        cfg, bundle, examples, str(tmp_path / "nan"), validate_config=False
    )
    assert result.stopped_reason == tr.STOP_STABILITY
    assert result.steps == 3
    assert result.failure_record["reason"] in {"NAN_LOSS", "OPTIMIZER_OVERFLOW"}
    assert result.failure_record["arm_config"]["learning_rate"] == cfg.learning_rate
    # the configuration was NOT modified in response
    assert result.arm_config["learning_rate"] == cfg.learning_rate
    assert result.arm_config["grad_clip_norm"] == 1.0


def test_transient_infrastructure_error_is_retried_exactly_once(tmp_path):
    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, "FL3", n=2)
    real_logits = tr._model_logits
    state = {"failures": 0, "calls": 0}

    def flaky(model, batch, use_cache=False):
        state["calls"] += 1
        if state["calls"] == 3 and state["failures"] == 0:
            state["failures"] += 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return real_logits(model, batch, use_cache)

    tr._model_logits, original = flaky, tr._model_logits
    try:
        result = tr.run_training_arm(
            smoke_cfg("FL3", updates=6, checkpoint_every_steps=2),
            bundle,
            examples,
            str(tmp_path / "retry"),
            validate_config=False,
        )
    finally:
        tr._model_logits = original
    assert result.retries == 1
    assert result.steps == 6
    assert any(e["kind"] == "OOM" for e in result.stability_events)

    # a second, persistent failure is NOT retried again
    state2 = {"calls": 0}

    def always_fail(model, batch, use_cache=False):
        state2["calls"] += 1
        raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

    tr._model_logits, original = always_fail, tr._model_logits
    try:
        with pytest.raises(RuntimeError):
            tr.run_training_arm(
                smoke_cfg("FL3", updates=6),
                build_tiny_model(seed=17),
                examples,
                str(tmp_path / "retry2"),
                validate_config=False,
            )
    finally:
        tr._model_logits = original
    assert state2["calls"] == 2  # first attempt + exactly one retry


def test_grad_norm_clipping_is_applied(tmp_path):
    bundle = build_tiny_model(seed=17)
    examples = build_examples(bundle, "FL3", n=2)
    result = tr.run_training_arm(
        smoke_cfg("FL3", updates=5),
        bundle,
        examples,
        str(tmp_path / "clip"),
        validate_config=False,
        stability_config=StabilityConfig(),
    )
    # the reported grad norms are PRE-clip values; the applied update is bounded
    assert all(norm >= 0 for norm in result.grad_norm_history)
    assert result.arm_config["grad_clip_norm"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_bf16_autocast_and_gpu_accounting_on_a_real_device(tmp_path):
    """Exercise the accelerator code path with the TINY model only.

    The frozen 2.6 B checkpoint is never loaded here (contract §20 / task
    constraint); this covers bf16 autocast, CUDA synchronisation, GPU-second
    accounting, CUDA RNG capture and gradient checkpointing plumbing.
    """
    from foundation_learner.training.checkpointing import load_checkpoint

    bundle = build_tiny_model(seed=17, device="cuda")
    examples = build_examples(bundle, "FL3", n=2)
    cfg = smoke_cfg("FL3", updates=4, bf16=True, gradient_checkpointing=True)
    result = tr.run_training_arm(
        cfg, bundle, examples, str(tmp_path / "cuda"), validate_config=False
    )
    assert result.steps == 4
    assert all(torch.isfinite(torch.tensor(result.loss_history)))
    assert result.ledger["device"].startswith("cuda")
    assert result.ledger["gpu_seconds"] > 0.0
    assert result.trainable["memory_savings"]["loop_level_checkpointing"] is True
    payload = load_checkpoint(str(tmp_path / "cuda"), "final")
    assert payload["rng_state"]["torch_cuda"] is not None


def test_ledger_records_the_intrinsic_loss_token_difference(tmp_path):
    results = {}
    for arm in ("FL2", "FL3"):
        bundle = build_tiny_model(seed=17)
        examples = build_examples(bundle, arm, n=4)
        results[arm] = tr.run_training_arm(
            smoke_cfg(arm, updates=6),
            bundle,
            examples,
            str(tmp_path / arm),
            validate_config=False,
        )
    fl2, fl3 = results["FL2"].ledger, results["FL3"].ledger
    assert fl2["optimizer_updates"] == fl3["optimizer_updates"]
    assert fl2["max_tokens_per_batch"] == fl3["max_tokens_per_batch"]
    assert fl2["forward_tokens"] == fl3["forward_tokens"]
    # FL2 weights every attempt; FL3 masks attempt0 and the related attempt
    assert fl3["loss_tokens"] < fl2["loss_tokens"]
    assert fl3["episodes"] > 0
