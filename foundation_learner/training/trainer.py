"""The single training loop used by EVERY arm (contract §8, §11, §15, §17, §23).

One loop, one optimizer, one schedule, one packing rule — the arms differ only
in their data (``ArmConfig.data_mode``) and their objective reduction.  Using
one loop for all arms is what makes the §8 compute matching meaningful.

Frozen mechanics
----------------
* AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0.01 (§8).
* Cosine schedule with 3 % warmup, grad-norm clip 1.0 (§8).
* bf16 autocast on CUDA; plain fp32 on CPU (the tiny mechanics model).
* Token-budget packing: examples are packed until the PADDED batch cost
  ``(n + 1) * max_len`` would exceed ``max_tokens_per_batch`` (§8).
* Deterministic data order from derived seeds; uniform sampling without
  replacement per epoch, reshuffled per epoch by a derived seed (§16).  No
  global RNG, no wall-clock, anywhere in the data path (§3, §20).
* Checkpoint cadence: every 200 optimizer steps or 600 s, whichever comes
  first (§11), atomic (§15).
* Stability monitors from §17, with EXACTLY ONE retry from the last valid
  checkpoint for a transient infrastructure error and no hyperparameter
  improvisation ever.
* Single accelerator: no DDP, no gradient accumulation.

``model.generate`` is never called here or anywhere in this package.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from foundation_learner.training.arms import ArmConfig, arm_config_hash, validate_arm_config
from foundation_learner.training.checkpointing import (
    TAG_FINAL,
    TAG_INITIAL,
    CorruptCheckpointError,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from foundation_learner.training.compute_accounting import ComputeLedger
from foundation_learner.training.model_loading import (
    ModelBundle,
    derive_seed,
    enable_training_memory_savings,
    set_evaluation_mode,
)
from foundation_learner.training.objectives import weighted_nll
from foundation_learner.training.peft_modes import TrainableParams, apply_mode
from foundation_learner.training.stability import (
    StabilityAbort,
    StabilityConfig,
    StabilityMonitor,
    SEVERITY_TRANSIENT_INFRA,
)
from foundation_learner.training.tokenization import examples_to_batch

ARM_RESULT_SCHEMA = "flb200.arm_result.v1"

STOP_COMPLETED = "COMPLETED"
STOP_HOOK = "STOPPED_BY_HOOK"
STOP_STABILITY = "STOPPED_BY_STABILITY"


class TrainerError(RuntimeError):
    """Raised on malformed trainer inputs."""


@dataclass
class TrainerHooks:
    """Optional external callbacks (scheduler heartbeat, dev eval, logging)."""

    #: ``on_step(state) -> str | None``; returning "STOP" ends the arm cleanly.
    on_step: Callable[[dict], str | None] | None = None
    #: ``on_checkpoint(tag, manifest) -> None``
    on_checkpoint: Callable[[str, dict], None] | None = None
    #: ``on_batch_start(state) -> None``
    on_batch_start: Callable[[dict], None] | None = None

    def call_on_step(self, state: dict) -> str | None:
        return self.on_step(state) if self.on_step else None


@dataclass
class ArmResult:
    arm_id: str
    out_dir: str
    steps: int
    stopped_reason: str
    final_loss: float | None
    loss_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    ledger: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    stability_events: list[dict[str, Any]] = field(default_factory=list)
    failure_record: dict[str, Any] | None = None
    identity: dict[str, Any] = field(default_factory=dict)
    arm_config: dict[str, Any] = field(default_factory=dict)
    arm_config_hash: str = ""
    trainable: dict[str, Any] = field(default_factory=dict)
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ARM_RESULT_SCHEMA,
            "arm_id": self.arm_id,
            "out_dir": self.out_dir,
            "steps": self.steps,
            "stopped_reason": self.stopped_reason,
            "final_loss": self.final_loss,
            "loss_history": self.loss_history,
            "grad_norm_history": self.grad_norm_history,
            "lr_history": self.lr_history,
            "ledger": self.ledger,
            "checkpoints": self.checkpoints,
            "stability_events": self.stability_events,
            "failure_record": self.failure_record,
            "identity": self.identity,
            "arm_config": self.arm_config,
            "arm_config_hash": self.arm_config_hash,
            "trainable": self.trainable,
            "retries": self.retries,
        }


# ---------------------------------------------------------------------------
# Data order + packing
# ---------------------------------------------------------------------------


def epoch_order(n: int, root_seed: int, arm_id: str, epoch: int) -> list[int]:
    """Deterministic permutation for one epoch (contract §3, §16)."""
    seed = derive_seed(root_seed, "DATA_ORDER", arm_id, epoch)
    rng = np.random.Generator(np.random.PCG64(seed))
    return [int(i) for i in rng.permutation(n)]


def pack_batches(
    examples: Sequence[dict], order: Sequence[int], max_tokens_per_batch: int
) -> list[list[int]]:
    """Greedy token-budget packing over a fixed order (padded cost)."""
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for idx in order:
        length = len(examples[idx]["input_ids"])
        if length > max_tokens_per_batch:
            raise TrainerError(
                f"example {idx} has {length} tokens > max_tokens_per_batch "
                f"{max_tokens_per_batch}"
            )
        candidate_max = max(current_max, length)
        if current and (len(current) + 1) * candidate_max > max_tokens_per_batch:
            batches.append(current)
            current = [idx]
            current_max = length
        else:
            current.append(idx)
            current_max = candidate_max
    if current:
        batches.append(current)
    return batches


def _batch_stream(
    examples: Sequence[dict], cfg: ArmConfig
) -> Iterable[tuple[int, list[int]]]:
    """Infinite deterministic stream of ``(epoch, batch_indices)``."""
    epoch = 0
    while True:
        order = epoch_order(len(examples), cfg.seed, cfg.arm_id, epoch)
        for batch in pack_batches(examples, order, cfg.max_tokens_per_batch):
            yield epoch, batch
        epoch += 1


# ---------------------------------------------------------------------------
# Optimizer / schedule
# ---------------------------------------------------------------------------


def build_optimizer(params: Sequence[torch.nn.Parameter], cfg: ArmConfig):
    return torch.optim.AdamW(
        params,
        lr=cfg.learning_rate,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(optimizer, cfg: ArmConfig):
    warmup = cfg.warmup_steps
    total = max(1, cfg.updates)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _model_logits(model, batch: dict, use_cache: bool = False) -> torch.Tensor:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=use_cache,
    )
    logits = getattr(out, "logits", None)
    if logits is None:
        raise TrainerError("model forward returned no logits")
    return logits


def _trainable_state(params: TrainableParams, model) -> dict[str, torch.Tensor]:
    if params.lora_handles is not None:
        from foundation_learner.training.lora import lora_state_dict

        return lora_state_dict(params.lora_handles)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_trainable_state(params: TrainableParams, model):
    def loader(state: dict[str, torch.Tensor]) -> None:
        if params.lora_handles is not None:
            from foundation_learner.training.lora import load_lora_state_

            load_lora_state_(params.lora_handles, state)
        else:
            model.load_state_dict(state, strict=True)

    return loader


def run_training_arm(
    cfg: ArmConfig,
    bundle: ModelBundle,
    examples_iter: Sequence[dict] | Iterable[dict],
    out_dir: str,
    hooks: TrainerHooks | None = None,
    *,
    trainable: TrainableParams | None = None,
    stability_config: StabilityConfig | None = None,
    family_split_hash: str | None = None,
    shard_hashes: Any = None,
    allow_retry: bool = True,
    validate_config: bool = True,
    resume_from_tag: str | None = None,
) -> ArmResult:
    """Contract §23: ``run_training_arm(cfg, bundle, examples_iter, out_dir, hooks)``.

    POST-CONDITION (Amendment 16): ``bundle.model`` is returned in **eval mode
    with layer-level gradient checkpointing disabled**, and the state change is
    recorded in ``result.trainable["post_training_mode"]``.  A model left in
    train mode with checkpointing on cannot be decoded correctly: transformers'
    ``GradientCheckpointingLayer.__call__`` drops ``use_cache`` and
    ``past_key_values`` in exactly that state.

    ``examples_iter`` may be any iterable of tokenised examples; it is
    materialised into a list so that the frozen data-order policy (uniform
    without replacement per epoch, reshuffled by a derived seed) can be applied
    deterministically.

    ``resume_from_tag`` restores parameters, optimizer, scheduler and every RNG
    stream from an existing checkpoint in ``out_dir`` and continues at exactly
    the recorded step; the resulting loss trajectory is identical to the
    uninterrupted run (pinned by a unit test).
    """
    if validate_config:
        validate_arm_config(cfg)
    hooks = hooks or TrainerHooks()
    os.makedirs(out_dir, exist_ok=True)

    examples = list(examples_iter)
    if not examples:
        raise TrainerError(f"{cfg.arm_id}: no training examples supplied")
    for example in examples:
        if example.get("arm") not in (None, cfg.arm_id):
            raise TrainerError(
                f"{cfg.arm_id}: example was tokenised for arm "
                f"{example.get('arm')!r}; arms must not share tokenised data"
            )

    model = bundle.model
    device = bundle.device
    pad_token_id = _pad_token_id(bundle)

    if trainable is None:
        trainable = apply_mode(model, cfg.peft_mode, seed=cfg.seed)
    memory_state = {}
    if cfg.gradient_checkpointing:
        memory_state = enable_training_memory_savings(model)

    optimizer = build_optimizer(trainable.parameters, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    monitor = StabilityMonitor(stability_config or StabilityConfig(), arm_id=cfg.arm_id)

    ledger = ComputeLedger(
        arm_id=cfg.arm_id,
        stage=cfg.stage,
        max_tokens_per_batch=cfg.max_tokens_per_batch,
        planned_updates=cfg.updates,
        device=device,
    )
    cfg_dict = cfg.to_dict()
    cfg_hash = arm_config_hash(cfg)
    result = ArmResult(
        arm_id=cfg.arm_id,
        out_dir=out_dir,
        steps=0,
        stopped_reason=STOP_COMPLETED,
        final_loss=None,
        identity=dict(bundle.identity),
        arm_config=cfg_dict,
        arm_config_hash=cfg_hash,
        trainable=trainable.to_dict() | {"memory_savings": memory_state},
    )

    def _save(tag: str, step: int) -> dict[str, Any]:
        manifest = save_checkpoint(
            out_dir,
            tag,
            trained_state=_trainable_state(trainable, model),
            trained_scope=trainable.mode.name,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            tokens=ledger.forward_tokens,
            loss_tokens=ledger.loss_tokens,
            arm_config=cfg_dict,
            arm_config_hash=cfg_hash,
            base_identity=bundle.identity,
            family_split_hash=family_split_hash,
            shard_hashes=shard_hashes,
            stage=cfg.stage,
        )
        result.checkpoints.append(
            {"tag": tag, "step": step, "manifest_hash": manifest.get("manifest_hash")}
        )
        if hooks.on_checkpoint:
            hooks.on_checkpoint(tag, manifest)
        return manifest

    if resume_from_tag is None:
        _save(TAG_INITIAL, 0)
    else:
        payload = load_checkpoint(out_dir, resume_from_tag)
        # verify BEFORE touching any state, so a refused resume leaves the
        # freshly loaded model untouched
        stored = payload["manifest"]
        if stored.get("arm_config_hash") != cfg_hash:
            raise TrainerError(
                f"refusing to resume {cfg.arm_id!r} from checkpoint "
                f"{resume_from_tag!r}: arm config hash "
                f"{stored.get('arm_config_hash')} != {cfg_hash}"
            )
        base_id = (stored.get("base_identity") or {}).get("identity_hash")
        if base_id != bundle.identity.get("identity_hash"):
            raise TrainerError(
                f"refusing to resume {cfg.arm_id!r}: base model identity "
                f"{base_id} != {bundle.identity.get('identity_hash')}"
            )
        manifest = restore_training_state(
            payload,
            load_trained_state=_load_trainable_state(trainable, model),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        result.steps = int(manifest["step"])
        result.checkpoints.append(
            {
                "tag": resume_from_tag,
                "step": result.steps,
                "manifest_hash": manifest.get("manifest_hash"),
                "resumed": True,
            }
        )

    attempt = 0
    while True:
        try:
            _train_loop(
                cfg=cfg,
                model=model,
                trainable=trainable,
                optimizer=optimizer,
                scheduler=scheduler,
                monitor=monitor,
                ledger=ledger,
                result=result,
                examples=examples,
                device=device,
                pad_token_id=pad_token_id,
                hooks=hooks,
                save=_save,
                start_step=result.steps,
            )
            break
        except StabilityAbort as abort:
            result.stopped_reason = STOP_STABILITY
            result.failure_record = abort.record
            break
        except Exception as exc:  # noqa: BLE001 - classified, never swallowed
            severity = StabilityMonitor.classify_exception(exc)
            if severity != SEVERITY_TRANSIENT_INFRA or not allow_retry or attempt >= 1:
                ledger.stop()
                result.ledger = ledger.to_dict()
                result.stability_events = monitor.events_as_dicts()
                raise
            attempt += 1
            result.retries = attempt
            if "memory" in repr(exc).lower():
                monitor.record_oom(result.steps, repr(exc))
            else:
                monitor.record_transient_infra(
                    result.steps,
                    f"retrying once from the last valid checkpoint after {exc!r}",
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            resume_step = _resume_from_last_checkpoint(
                out_dir=out_dir,
                result=result,
                trainable=trainable,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                monitor=monitor,
            )
            result.steps = resume_step

    ledger.stop()
    result.ledger = ledger.to_dict()
    result.stability_events = monitor.events_as_dicts()
    if result.stopped_reason != STOP_STABILITY:
        _save(TAG_FINAL, result.steps)
    ledger.write(os.path.join(out_dir, f"compute_ledger_{cfg.arm_id}.json"))
    # POST-CONDITION (Amendment 16): the arm's model is handed back in EVAL
    # mode with layer-level gradient checkpointing DISABLED.  While a module is
    # in train mode with that flag set, transformers'
    # GradientCheckpointingLayer.__call__ drops use_cache/past_key_values, so a
    # manual greedy decode silently produces DIFFERENT text than the same
    # weights evaluated normally.  Every caller that evaluates a trained arm
    # therefore receives a model that is already valid to decode with.
    result.trainable = dict(result.trainable or {})
    result.trainable["post_training_mode"] = set_evaluation_mode(model)
    return result


def _pad_token_id(bundle: ModelBundle) -> int:
    tokenizer = bundle.tokenizer
    pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    if pad is None:
        pad = getattr(bundle.model.config, "pad_token_id", None)
    if pad is None:
        raise TrainerError("no pad_token_id available for batching")
    return int(pad)


def _resume_from_last_checkpoint(
    *, out_dir, result, trainable, model, optimizer, scheduler, monitor
) -> int:
    tags = [c["tag"] for c in result.checkpoints]
    if not tags:
        raise TrainerError("no checkpoint available to retry from")
    for tag in reversed(tags):
        try:
            payload = load_checkpoint(out_dir, tag)
        except CorruptCheckpointError as exc:
            monitor.record_corrupted_checkpoint(result.steps, tag, str(exc))
            continue
        manifest = restore_training_state(
            payload,
            load_trained_state=_load_trainable_state(trainable, model),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        return int(manifest["step"])
    raise TrainerError("every available checkpoint failed verification")


def _train_loop(
    *,
    cfg: ArmConfig,
    model,
    trainable: TrainableParams,
    optimizer,
    scheduler,
    monitor: StabilityMonitor,
    ledger: ComputeLedger,
    result: ArmResult,
    examples: Sequence[dict],
    device: str,
    pad_token_id: int,
    hooks: TrainerHooks,
    save,
    start_step: int,
) -> None:
    model.train()
    stream = _batch_stream(examples, cfg)
    # Fast-forward the deterministic stream to the resume point so that a
    # retry consumes exactly the batches the first attempt had not consumed.
    for _ in range(start_step):
        next(stream)

    ledger.start()
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    autocast_enabled = bool(cfg.bf16 and use_cuda)
    last_checkpoint_time = time.monotonic()
    step = start_step

    while step < cfg.updates:
        epoch, indices = next(stream)
        batch_examples = [examples[i] for i in indices]
        batch = examples_to_batch(batch_examples, pad_token_id, device=device)
        if hooks.on_batch_start:
            hooks.on_batch_start({"step": step, "epoch": epoch, "batch": batch})

        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = _model_logits(model, batch)
            loss, stats = weighted_nll(
                logits.float(), batch["input_ids"], batch["loss_mask"], cfg.objective
            )
        else:
            logits = _model_logits(model, batch)
            loss, stats = weighted_nll(
                logits, batch["input_ids"], batch["loss_mask"], cfg.objective
            )
        loss.backward()

        grads_finite = all(
            torch.isfinite(p.grad).all().item()
            for p in trainable.parameters
            if p.grad is not None
        )
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable.parameters, cfg.grad_clip_norm)
        )
        loss_value = float(loss.detach())
        did_update = grads_finite and math.isfinite(loss_value) and math.isfinite(grad_norm)
        if did_update:
            optimizer.step()
        scheduler.step()
        if use_cuda:
            torch.cuda.synchronize()
        step_seconds = time.monotonic() - t0

        step += 1
        result.steps = step
        result.loss_history.append(loss_value)
        result.grad_norm_history.append(grad_norm)
        result.lr_history.append(float(optimizer.param_groups[0]["lr"]))
        result.final_loss = loss_value

        ledger.record_step(
            n_examples=batch["n_examples"],
            forward_tokens=batch["n_padded_tokens"],
            forward_tokens_unpadded=batch["n_real_tokens"],
            loss_tokens=batch["n_loss_tokens"],
            weighted_loss_mass=stats["sum_weight"],
            did_update=did_update,
            step_seconds=step_seconds,
            episode_ids=[ex.get("episode_id") for ex in batch_examples],
        )

        events = monitor.observe_step(
            step=step,
            loss=loss_value,
            grad_norm=grad_norm,
            tokens=batch["n_padded_tokens"],
            step_seconds=step_seconds,
            memory_bytes=(
                float(torch.cuda.memory_allocated()) if use_cuda else None
            ),
            grads_finite=grads_finite,
        )
        del events

        now = time.monotonic()
        due_by_steps = cfg.checkpoint_every_steps > 0 and step % cfg.checkpoint_every_steps == 0
        due_by_time = (
            cfg.checkpoint_every_seconds > 0
            and (now - last_checkpoint_time) >= cfg.checkpoint_every_seconds
        )
        if step < cfg.updates and (due_by_steps or due_by_time):
            save(f"step{step}", step)
            last_checkpoint_time = now

        if monitor.fatal:
            record = monitor.failure_record(
                arm_id=cfg.arm_id,
                step=step,
                reason=monitor.fatal_event.kind if monitor.fatal_event else "UNKNOWN",
                config_snapshot=result.arm_config,
            )
            ledger.stop()
            raise StabilityAbort(
                f"{cfg.arm_id}: scientific instability at step {step} "
                f"({record['reason']}); recorded, no hyperparameter improvisation",
                record,
            )

        signal = hooks.call_on_step(
            {
                "step": step,
                "epoch": epoch,
                "loss": loss_value,
                "grad_norm": grad_norm,
                "lr": result.lr_history[-1],
                "arm_id": cfg.arm_id,
                "ledger": ledger,
                "n_examples": batch["n_examples"],
                "step_seconds": step_seconds,
                # lets the campaign layer request the §15 retention tags
                # (``best_dev`` at a scheduled DEV eval, ``pre_promotion``)
                "save": save,
            }
        )
        if signal == "STOP":
            result.stopped_reason = STOP_HOOK
            ledger.stop()
            return

    result.stopped_reason = STOP_COMPLETED
    ledger.stop()
