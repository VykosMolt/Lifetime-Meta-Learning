"""FL7 — bounded resettable low-rank fast adapter (contract §7).

Frozen mechanism
================
* LoRA ``r=8``, ``alpha=16``, dropout 0, on ALL attention ``q_proj`` /
  ``v_proj`` modules (``training.peft_modes.FL7_FAST_ADAPTER_SPEC`` — W2's
  frozen spec, imported, never re-declared).
* Inner loop: per REVEAL-supervised SUPPORT item, ``FL7_INNER_STEPS = 4`` SGD
  steps at ``FL7_INNER_LR = 1e-3`` on the NLL of the REVEALED support answer.
* After each item, the GLOBAL trainable-delta Frobenius norm is clipped to
  ``FL7_FAST_DELTA_MAX_FROBENIUS = 1.0`` with
  ``training.lora.clip_lora_frobenius_`` (which returns the PRE-clip norm; it
  is recorded).
* :meth:`FastAdapter.reset_episode` zeroes the fast delta exactly
  (``training.lora.zero_lora_``), per episode.
* The fast delta persists across a CONTEXT RESET inside an episode: nothing in
  the context-reset path touches parameters, and the adapter only resets when
  ``reset_episode`` is called.
* Base weights are never written: ``attach_lora`` freezes them and the
  optimizer only ever sees ``lora_A`` / ``lora_B``.
  :meth:`FastAdapter.base_fingerprint` hashes every base parameter so a test
  can assert BITWISE equality before and after an episode.
* Deterministic checkpoint: :meth:`FastAdapter.state_dict` = W2's
  ``lora_state_dict`` plus the configuration hash.

The supervised inner-loop example
---------------------------------
Built with W1's renderer and W2's tokenizer primitives: the SUPPORT task event
and an answer event carrying the revealed canonical answer are rendered as a
two-event episode through ``episodes/render.py`` (instruction header included,
identical surface to training) and masked by
``training.tokenization.episode_to_training_examples(..., arm="FL1")``, i.e.
loss on the ``ANSWER:`` line only.

REVEAL supervision is ENFORCED, not assumed: :meth:`FastAdapter.adapt_item`
refuses any item that is not a SUPPORT item with a ``REVEAL`` event in the
episode.  A query or transfer item therefore cannot enter the inner loop, so
the hidden QUERY/TRANSFER labels can never train the adapter.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from foundation_learner.ecology.base import domain_sha256
from foundation_learner.episodes.parse import format_answer
from foundation_learner.episodes.schema import Episode, Event, Role
from foundation_learner.evaluation.learning_curve import NullCallbacks
from foundation_learner.training.lora import (
    LoraSpec,
    attach_lora,
    clip_lora_frobenius_,
    lora_frobenius_norm,
    lora_state_dict,
    zero_lora_,
)
from foundation_learner.training.objectives import OBJ_TOKEN_MEAN, weighted_nll
from foundation_learner.training.peft_modes import (
    FL7_FAST_ADAPTER_SPEC,
    FL7_FAST_DELTA_MAX_FROBENIUS,
    FL7_INNER_LR,
    FL7_INNER_STEPS,
)
from foundation_learner.training.tokenization import (
    episode_to_training_examples,
    examples_to_batch,
)

__all__ = [
    "FastAdapterError",
    "FastAdapter",
    "FastAdapterCallbacks",
    "revealed_support_items",
    "supervised_example",
    "run_fl7_stage",
]


class FastAdapterError(RuntimeError):
    """Raised on malformed FL7 usage (never silently repaired)."""


def _tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw bytes of a tensor (bfloat16 is viewed as int16: numpy has no bf16)."""
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def revealed_support_items(episode: Episode) -> dict[str, int]:
    """``item_id -> REVEAL event index`` for the episode's supervised items."""
    out: dict[str, int] = {}
    for index, event in enumerate(episode.events):
        if event.role is Role.REVEAL and event.reveal:
            out[str(event.item_id)] = index
    return out


def _support_task_event(episode: Episode, item_id: str) -> tuple[int, Event]:
    for index, event in enumerate(episode.events):
        if event.role is Role.TASK and str(event.item_id) == str(item_id):
            return index, event
    raise FastAdapterError(
        f"episode {episode.episode_id!r} has no SUPPORT TASK event for item "
        f"{item_id!r}")


def _first_attempt_event(episode: Episode, item_id: str) -> Event:
    for event in episode.events:
        if (event.role is Role.MODEL_ATTEMPT
                and str(event.item_id) == str(item_id)):
            return event
    raise FastAdapterError(
        f"episode {episode.episode_id!r} has no MODEL_ATTEMPT event for item "
        f"{item_id!r}")


def _revealed_answer(episode: Episode, item_id: str, reveal_index: int) -> str:
    text = str(episode.events[reveal_index].text).strip()
    record = (episode.items or {}).get(item_id)
    canonical = None
    if isinstance(record, dict):
        instance = record.get("instance", record)
        if isinstance(instance, dict):
            canonical = instance.get("answer_canonical")
    if canonical is not None and str(canonical).strip() != text:
        raise FastAdapterError(
            f"REVEAL text for item {item_id!r} disagrees with the episode's "
            "item table; refusing to train on an ambiguous supervision target")
    return text


def supervised_example(episode: Episode, item_id: str, tokenizer: Any) -> dict:
    """The two-event (TASK -> revealed ANSWER) example for one support item."""
    reveals = revealed_support_items(episode)
    if item_id not in reveals:
        raise FastAdapterError(
            f"REFUSED: item {item_id!r} has no REVEAL event in episode "
            f"{episode.episode_id!r}; the FL7 inner loop is supervised by the "
            "REVEALED SUPPORT answer only (contract §7), so query, transfer "
            "and related items can never enter it")
    record = (episode.items or {}).get(item_id)
    role = record.get("role") if isinstance(record, dict) else None
    if role is not None and role != "support":
        raise FastAdapterError(
            f"REFUSED: item {item_id!r} has role {role!r}; FL7 adapts on "
            "SUPPORT items only")
    task_index, task_event = _support_task_event(episode, item_id)
    attempt_event = _first_attempt_event(episode, item_id)
    answer = _revealed_answer(episode, item_id, reveals[item_id])
    answer_event = dataclasses.replace(attempt_event, text=format_answer(answer))
    pair = dataclasses.replace(episode, events=[task_event, answer_event])
    examples = episode_to_training_examples(pair, tokenizer, "FL1")
    if len(examples) != 1:
        raise FastAdapterError(
            f"expected exactly one supervised example for item {item_id!r}, "
            f"got {len(examples)}")
    example = examples[0]
    example["item_id"] = item_id
    example["task_event_index"] = task_index
    return example


@dataclass
class InnerUpdateRecord:
    """One item's inner loop (recorded, never averaged away)."""

    item_id: str
    episode_id: str
    applied: bool
    steps: int
    losses: list[float] = field(default_factory=list)
    pre_clip_norm: float | None = None
    post_clip_norm: float | None = None
    gate: dict[str, Any] | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class FastAdapter:
    """The FL7 fast adapter attached to ONE model."""

    def __init__(self, bundle: Any, *, seed: int = 0,
                 spec: LoraSpec | None = None,
                 inner_steps: int = FL7_INNER_STEPS,
                 inner_lr: float = FL7_INNER_LR,
                 max_frobenius: float = FL7_FAST_DELTA_MAX_FROBENIUS,
                 gate: Any = None) -> None:
        base = spec or FL7_FAST_ADAPTER_SPEC
        self.spec = LoraSpec(
            target_suffixes=base.target_suffixes, rank=base.rank,
            alpha=base.alpha, dropout=base.dropout, seed=int(seed),
            freeze_base=True, name=base.name)
        self.bundle = bundle
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer
        self.device = getattr(bundle, "device", None) or next(
            self.model.parameters()).device
        self.inner_steps = int(inner_steps)
        self.inner_lr = float(inner_lr)
        self.max_frobenius = float(max_frobenius)
        self.gate = gate
        self.handles = attach_lora(self.model, self.spec)
        self.records: list[InnerUpdateRecord] = []
        self.episodes_seen = 0
        zero_lora_(self.handles)

    # -- lifecycle --------------------------------------------------------
    def reset_episode(self) -> None:
        """Contract §7: per-episode reset of the fast delta to EXACT zero."""
        zero_lora_(self.handles)
        self.episodes_seen += 1

    def delta_norm(self) -> float:
        return lora_frobenius_norm(self.handles)

    # -- the inner loop ---------------------------------------------------
    def adapt_item(self, episode: Episode, item_id: str, *,
                   hidden_summary: torch.Tensor | None = None
                   ) -> InnerUpdateRecord:
        """4 SGD steps at lr 1e-3 on the NLL of the REVEALED support answer."""
        gate_record = None
        if self.gate is not None:
            if hidden_summary is None:
                raise FastAdapterError(
                    "a gated FL7 adapter needs the item's hidden summary; the "
                    "gate consumes nothing else")
            decision = self.gate.decide(hidden_summary)
            gate_record = decision.to_dict()
            if not decision.incorporate:
                record = InnerUpdateRecord(
                    item_id=str(item_id), episode_id=episode.episode_id,
                    applied=False, steps=0, gate=gate_record,
                    reason="value gate declined")
                self.records.append(record)
                return record

        example = supervised_example(episode, item_id, self.tokenizer)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.model.config, "pad_token_id", None)
        if pad_id is None:
            raise FastAdapterError("no pad_token_id available for the inner loop")
        batch = examples_to_batch([example], int(pad_id), device=str(self.device))
        params = list(self.handles.parameters())
        optimizer = torch.optim.SGD(params, lr=self.inner_lr)
        was_training = self.model.training
        self.model.train()
        losses: list[float] = []
        for _ in range(self.inner_steps):
            optimizer.zero_grad(set_to_none=True)
            out = self.model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"],
                             use_cache=False)
            loss, _stats = weighted_nll(out.logits, batch["input_ids"],
                                        batch["loss_mask"], OBJ_TOKEN_MEAN)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        # ALWAYS return to evaluation state, never merely to whatever the mode
        # happened to be.  The fast adapter runs INSIDE an online evaluation
        # walk, so the operation right after an inner update is a greedy
        # decode, which requires eval mode (Amendment 16); `was_training` is
        # kept only for the record.
        del was_training
        self.model.eval()
        pre = clip_lora_frobenius_(self.handles, self.max_frobenius)
        record = InnerUpdateRecord(
            item_id=str(item_id), episode_id=episode.episode_id, applied=True,
            steps=self.inner_steps, losses=losses, pre_clip_norm=float(pre),
            post_clip_norm=self.delta_norm(), gate=gate_record)
        self.records.append(record)
        return record

    def adapt_episode(self, episode: Episode, *,
                      item_ids: Sequence[str] | None = None
                      ) -> list[InnerUpdateRecord]:
        """Run the inner loop on every REVEAL-supervised support item, in order."""
        reveals = revealed_support_items(episode)
        if item_ids is None:
            item_ids = [i for i in sorted(reveals, key=lambda k: reveals[k])]
        if self.gate is not None:
            raise FastAdapterError(
                "adapt_episode() is the UNGATED batch entry point; a gated "
                "adapter must call adapt_item() with the item's hidden summary")
        return [self.adapt_item(episode, item_id) for item_id in item_ids]

    # -- checkpointing / identity -----------------------------------------
    def config(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "inner_steps": self.inner_steps,
            "inner_lr": self.inner_lr,
            "max_frobenius": self.max_frobenius,
            "gated": self.gate is not None,
            "n_modules": len(self.handles),
        }

    def config_hash(self) -> str:
        return domain_sha256("FL_V0_FL7_FAST_ADAPTER", self.config())

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "flb200.fl7_fast_adapter.v1",
            "config": self.config(),
            "config_hash": self.config_hash(),
            "lora": lora_state_dict(self.handles),
        }

    def base_fingerprint(self) -> str:
        """SHA-256 over every NON-adapter parameter (bitwise evidence).

        Covers the whole model, not only the wrapped layers, so "the base
        checkpoint artifact stays untouched" is checked globally.
        """
        digest = hashlib.sha256()
        for name, param in sorted(self.model.named_parameters()):
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                continue
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_tensor_bytes(param))
        return digest.hexdigest()

    def summary(self) -> dict[str, Any]:
        applied = [r for r in self.records if r.applied]
        return {
            "schema": "flb200.fl7_summary.v1",
            "config_hash": self.config_hash(),
            "episodes_seen": self.episodes_seen,
            "n_updates_applied": len(applied),
            "n_updates_skipped": len(self.records) - len(applied),
            "delta_norm": self.delta_norm(),
            "records": [r.to_dict() for r in self.records],
        }


class FastAdapterCallbacks(NullCallbacks):
    """FL7 adapter for W4's callback protocol.

    ``on_feedback`` runs the inner loop when the event is the item's ``REVEAL``
    (the only supervised channel).  ``FEEDBACK`` and ``HINT`` events never
    trigger a parameter update, because their content does not contain the
    supervision target.  The gate (FL6) sees only the item's hidden summary.

    ``prefix_provider`` returns ``None``: FL7 adapts PARAMETERS, not a prefix.

    IMPORTANT (a real constraint of parameter-level adaptation): the fast delta
    lives in the MODEL, so exactly one FL7 episode may be in flight at a time.
    FL7 evaluation therefore runs ``run_episode`` (or ``run_episodes`` with a
    single episode) per episode; batching several episodes across one shared
    adapter would mix their adaptations.  ``training.lora.attach_lora`` refuses
    a second attachment to the same model, so the mistake cannot be made
    silently.
    """

    def __init__(self, adapter: FastAdapter, episode: Episode, *,
                 carry_across_episodes: bool = False,
                 gate_reduce: str = "last") -> None:
        self.adapter = adapter
        self.episode = episode
        self.carry_across_episodes = bool(carry_across_episodes)
        self.gate_reduce = str(gate_reduce)
        self.episodes_seen = 0
        self.records: list[InnerUpdateRecord] = []

    def on_episode_start(self, ctx: Any) -> None:
        if not (self.carry_across_episodes and self.episodes_seen > 0):
            self.adapter.reset_episode()
        self.episodes_seen += 1

    def on_feedback(self, event: Any, hidden_summary_fn: Any) -> None:
        if str(getattr(event, "role", "")).upper() != "REVEAL":
            return None
        summary = None
        if self.adapter.gate is not None:
            summary = hidden_summary_fn(reduce=self.gate_reduce,
                                        include_prefix=False)
        record = self.adapter.adapt_item(self.episode,
                                         str(getattr(event, "item_id", "")),
                                         hidden_summary=summary)
        self.records.append(record)
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "flb200.fl7_callbacks.v1",
            "episodes_seen": self.episodes_seen,
            "carry_across_episodes": self.carry_across_episodes,
            "records": [r.to_dict() for r in self.records],
            "delta_norm": self.adapter.delta_norm(),
        }


# ---------------------------------------------------------------------------
# campaign stage adapter (contract §7 FL7, §9 context reset)
# ---------------------------------------------------------------------------
def run_fl7_stage(ctx: Any, stage: Any) -> dict:
    """FL7 rung entry point for ``campaign/stage_definitions.py``.

    Two independently predeclared variants, each on its OWN fresh bundle
    (the fast delta lives in the model, so one adapter per bundle):

    * ``ungated`` — the inner loop runs on every REVEAL-supervised support item;
    * ``value_gated`` — the same inner loop behind the FROZEN FL4 head gate.
      It is SKIPPED, and recorded as skipped, when FL4 did not run: §10 says
      FL7 may run even when FL5 is null, but its gated variant depends on FL4.

    Each variant is evaluated on DEVELOPMENT in the ``history`` and the
    ``context_reset`` condition, which is where FL7's persistence claim lives
    (parameters survive the removal of the textual history).  The paired
    clustered bootstrap of retained post-reset improvement (adapter minus the
    no-adapter baseline) is published as FL8's ``persistence_evidence`` when
    FL5 did not already publish a defined estimate (§10 accepts either).
    """
    from foundation_learner.mechanisms import stage_support as _s

    dev_eps = _s.dev_episodes(ctx, stage)
    if not dev_eps:
        return _s.finish_stage(ctx, stage, _s.mechanism_payload(
            stage, _s.STATUS_SKIPPED_INPUT,
            reason="no DEVELOPMENT episodes are available"))
    reveal_eps = [_s.reveal_variant(ep) for ep in dev_eps]

    head = _s.value_head_from_ctx(ctx)
    gates: dict[str, Any] = {"ungated": None}
    if head is not None:
        from foundation_learner.mechanisms.value_gating import ValueGate

        gates["value_gated"] = ValueGate(head)

    variants: dict[str, Any] = {}
    records: dict[str, dict[str, list]] = {}
    for variant, gate in gates.items():
        bundle = ctx.bundle_factory()                   # FRESH load per variant
        adapter = FastAdapter(bundle, seed=int(ctx.root_seed), gate=gate)
        base_before = adapter.base_fingerprint()
        per_mode: dict[str, list] = {}
        for mode in ("history", "context_reset"):
            callbacks: list[FastAdapterCallbacks] = []

            def factory(ep, _adapter=adapter, _seen=callbacks):
                cb = FastAdapterCallbacks(_adapter, ep)
                _seen.append(cb)
                return cb

            # one episode at a time: the fast delta is model-global
            recs: list[dict] = []
            for episode in reveal_eps:
                recs.extend(_s.dev_records(bundle, [episode], ctx=ctx,
                                           arm_tag=f"FL7_{variant}_{mode}",
                                           callbacks_factory=factory,
                                           context_mode=mode))
            per_mode[mode] = recs
            variants.setdefault(variant, {}).setdefault("callbacks", {})[mode] = [
                cb.summary() for cb in callbacks]
        records[variant] = per_mode
        summary = adapter.summary()
        variants[variant].update({
            "config": adapter.config(),
            "config_hash": adapter.config_hash(),
            "n_updates_applied": summary["n_updates_applied"],
            "n_updates_skipped": summary["n_updates_skipped"],
            "final_delta_norm": summary["delta_norm"],
            "delta_norm_bound": adapter.max_frobenius,
            "base_weights_unchanged": adapter.base_fingerprint() == base_before,
            "dev": {mode: _s.dev_metrics_dict(recs, f"FL7_{variant}_{mode}")
                    for mode, recs in per_mode.items()},
            "macro_aulc": {mode: _s.macro_aulc(recs)
                           for mode, recs in per_mode.items()},
        })

    # matched no-adapter baseline for the persistence estimand
    baseline_bundle = ctx.bundle_factory()
    baseline = _s.dev_records(baseline_bundle, reveal_eps, ctx=ctx,
                              arm_tag="FL7_baseline_context_reset",
                              context_mode="context_reset")
    persistence = _s.paired_persistence_evidence(
        records["ungated"]["context_reset"], baseline, label="FL7",
        seed=int(ctx.root_seed))
    _s.publish_persistence_evidence(ctx, persistence,
                                    key="fl7_persistence_evidence")
    stable = all(v["base_weights_unchanged"]
                 and v["final_delta_norm"] <= v["delta_norm_bound"] * (1 + 1e-6)
                 for v in variants.values())
    ctx.extra.setdefault("fast_update_stable", stable)

    payload = _s.mechanism_payload(
        stage, _s.STATUS_COMPLETE,
        variants=variants,
        n_dev_episodes=len(dev_eps),
        gated_variant_available=bool(head is not None),
        fast_update_stable=stable,
        baseline_macro_aulc=_s.macro_aulc(baseline),
        persistence_evidence=persistence,
        inner_loop={"steps": FL7_INNER_STEPS, "lr": FL7_INNER_LR,
                    "max_frobenius": FL7_FAST_DELTA_MAX_FROBENIUS},
    )
    return _s.finish_stage(ctx, stage, payload)
