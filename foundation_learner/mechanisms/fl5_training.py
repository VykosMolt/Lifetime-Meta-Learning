"""FL5 training: segmented-episode forward under the FL3 objective (§7, §8).

The pipeline (identical for FAST_STATE_ON and FAST_STATE_OFF)
=============================================================
One episode is rendered ONCE with W1's renderer, tokenised ONCE, and then split
into SEGMENTS at the feedback events.  Segment ``t`` is forwarded on its own::

    inputs_embeds = [ prefix_embeds(s_t) ; embed(segment_t tokens) ]

Every token of the episode belongs to exactly one segment (tokens are assigned
to the segment containing their START offset), so the segments partition the
episode's token sequence without duplication or loss.  After the forward, each
feedback item in the segment contributes

    u_j = W_in . mean(final-layer, FINAL-LOOP hidden states over item-j tokens)
    s  <- GRUCell(u_j, s)

with gradients flowing (BPTT within the episode).  The FL3-weighted NLL is
accumulated over ALL segments and reduced exactly as ``OBJ_EPISODE_MEAN``:
per-episode weighted token mean, then batch mean.

Why segments, and what the arms actually contrast (corrected, Amendment 13)
--------------------------------------------------------------------------
Because each segment attends only to its own tokens (plus the prefix), the ONLY
channel by which information from an earlier interaction can reach a later
segment is the fast state ``s``.  That is a statement about the SEGMENTED
pipeline, and it is the only statement this design supports:

    within the segmented forward, FAST_STATE_ON has a segment-to-segment
    channel and FAST_STATE_OFF has none.

The earlier claim that the ON/OFF contrast is "exactly 'the state carried the
learning' and nothing else" OVERCLAIMED.  Three confounds sit inside that
difference and none of them is the state carrying learning:

1. **OFF is an impossible-task control.**  With the history removed by
   segmentation and no state, the OFF arm cannot use ANY earlier interaction;
   part of the ON-minus-OFF gap is OFF's degradation on a task it cannot do,
   not ON's retention.
2. **ON trains ~25M extra parameters** (``W_in``, ``GRUCell``, ``W_p``) that OFF
   does not have, so the arms are not parameter-matched.
3. **The ``s = 0`` prefix is a LEARNED STATIC PREFIX.**  ``prefix_embeds(0)`` is
   a trained, episode-independent set of 8 vectors, i.e. ordinary prefix
   tuning; any gain it produces is not persistence of within-episode learning.

``FAST_STATE_ON_S0`` (``mechanisms.fast_state``) isolates confound 3 at
evaluation time: it is the TRAINED ON module with ``s`` pinned to 0 and never
updated, so ON minus ON_S0 is the contribution of the STATE UPDATES on top of
the static prefix, while ON_S0 minus OFF is the static prefix plus the extra
parameters.  Confounds 1 and 2 are recorded, not removed — FL5 is not a
compute- or parameter-matched comparison and is never comparable to the FL3
core arm (every FL5 record carries ``comparable_to_fl3: false`` with this
reason).

Data order, seeds, weights, segmentation and reduction ARE shared between the
training arms.

The FL3 weights are ``training.tokenization.FL3_INTERACTION_WEIGHTS`` (QUERY
1.0, TRANSFER 1.0, revisions 0.25, everything else 0.0), applied to the same
``ANSWER:``-line spans W2 derives, with W2's frozen straddle policy.  Revision
spans live in earlier segments and therefore contribute from those segments;
that is what makes this objective identical to FL3's, not merely similar.

Final-loop hidden states come from ``mechanisms.hidden_states`` (see that
module's docstring for exactly which ``OuroForCausalLM`` output is used and
why); logits and hidden states come from ONE differentiable forward.

Plugging into the W2 training core
----------------------------------
``run_training_arm`` is built around ``examples_to_batch`` + a single
whole-sequence forward, which a segmented, stateful, per-episode forward cannot
express through its external step hook.  This module therefore runs its OWN
loop and REUSES W2's components rather than copying them:

* optimizer / schedule: ``training.trainer.build_optimizer`` /
  ``build_scheduler`` (AdamW betas (0.9, 0.95), eps 1e-8, wd 0.01, cosine with
  3 % warmup, grad-norm clip 1.0);
* data order: ``training.trainer.epoch_order`` (uniform without replacement per
  epoch, reshuffled by a derived seed);
* accounting: ``training.compute_accounting.ComputeLedger``;
* stability: ``training.stability.StabilityMonitor`` (with the FL5-specific
  fast-state norm bound of Amendment 6);
* checkpoints: ``training.checkpointing.save_checkpoint`` (atomic, RNG states,
  manifest hash), storing the fast-state module together with the trainable
  backbone state;
* objective mathematics: ``training.objectives`` (the per-token NLL convention
  is re-derived here only because of the prefix offset, and a unit test pins it
  against ``objectives.per_token_nll`` for ``prefix_len == 0``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from foundation_learner.ecology.base import domain_sha256
from foundation_learner.episodes.render import render_episode
from foundation_learner.episodes.schema import Episode, Role
from foundation_learner.mechanisms.fast_state import (
    FL5_STATE_NORM_LIMIT,
    FastStateModule,
)
from foundation_learner.mechanisms.hidden_states import forward_final_loop
from foundation_learner.training.checkpointing import save_checkpoint
from foundation_learner.training.compute_accounting import ComputeLedger
from foundation_learner.training.stability import StabilityConfig, StabilityMonitor
from foundation_learner.training.tokenization import (
    FL3_INTERACTION_WEIGHTS,
    answer_spans_for_event,
    char_span_to_token_span,
    encode_with_offsets,
    MASK_STRADDLE_POLICY,
    MAX_SEQ_LEN,
)
from foundation_learner.training.trainer import (
    build_optimizer,
    build_scheduler,
    epoch_order,
)

__all__ = [
    "ARM_FAST_STATE_ON",
    "ARM_FAST_STATE_OFF",
    "FL5_COMPARABLE_TO_FL3",
    "FL5_NOT_COMPARABLE_REASON",
    "FL5TrainConfig",
    "Fl5TrainingError",
    "EpisodeSegments",
    "Segment",
    "segment_episode",
    "segment_token_nll",
    "episode_loss",
    "batch_loss",
    "run_fast_state_arm",
]

ARM_FAST_STATE_ON = "FAST_STATE_ON"
ARM_FAST_STATE_OFF = "FAST_STATE_OFF"
FEEDBACK_ROLES = (Role.FEEDBACK, Role.HINT, Role.REVEAL)

#: FL5 is NEVER comparable to the FL3 core arm, and every FL5 payload and
#: per-arm record says so in its own body (Amendment 13).
FL5_COMPARABLE_TO_FL3 = False
FL5_NOT_COMPARABLE_REASON = (
    "FL5 runs a SEGMENTED forward under the FL3 objective, so its numbers are "
    "not on the same scale as FL3's whole-episode forward. Three confounds sit "
    "inside the FAST_STATE_ON minus FAST_STATE_OFF difference: (1) OFF is an "
    "impossible-task control - segmentation removes the textual history and OFF "
    "has no state, so it cannot use any earlier interaction at all; (2) ON "
    "trains ~25M parameters (W_in, GRUCell, W_p) that OFF does not have, so the "
    "arms are not parameter-matched; (3) prefix_embeds(0) is a LEARNED STATIC "
    "PREFIX, i.e. prefix tuning, whose gain is not persistence of within-episode "
    "learning. FAST_STATE_ON_S0 (the trained ON module with s pinned to 0, "
    "eval-only) isolates (3): ON minus ON_S0 is the contribution of the state "
    "updates over the static prefix, and ON_S0 minus OFF is the static prefix "
    "plus the extra parameters. (1) and (2) are recorded, not removed.")


class Fl5TrainingError(RuntimeError):
    """Raised on malformed FL5 training inputs (never silently repaired)."""


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Segment:
    """One forward unit: a contiguous token range of the rendered episode."""

    index: int
    event_indices: tuple[int, ...]
    token_start: int
    token_end: int
    #: (event_index, item_id, local_t0, local_t1) of each feedback item inside
    feedback_items: tuple[tuple[int, str, int, int], ...]
    #: (event_index, interaction_index, weight, local_t0, local_t1) loss spans
    loss_spans: tuple[tuple[int, int, float, int, int], ...]

    @property
    def n_tokens(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class EpisodeSegments:
    episode_id: str
    family_id: str
    split: str
    text: str
    input_ids: tuple[int, ...]
    segments: tuple[Segment, ...]

    @property
    def total_weight(self) -> float:
        return float(sum(w * (t1 - t0)
                         for seg in self.segments
                         for _e, _i, w, t0, t1 in seg.loss_spans))


def segment_episode(episode: Episode, tokenizer: Any, *,
                    max_seq_len: int = MAX_SEQ_LEN,
                    on_straddle: str = MASK_STRADDLE_POLICY) -> EpisodeSegments:
    """Split a rendered episode into segments at its feedback events.

    A segment ends after the maximal run of FEEDBACK / HINT / REVEAL events
    that follows its content, so the state update happens exactly where the
    learner receives feedback.  Tokens are assigned to the segment containing
    their start offset, which partitions the token sequence exactly.
    """
    events = list(episode.events)
    if not events:
        raise Fl5TrainingError(f"episode {episode.episode_id!r} has no events")
    rendered = render_episode(episode)
    text = rendered.text
    input_ids, offsets = encode_with_offsets(tokenizer, text)
    if len(input_ids) > max_seq_len:
        raise Fl5TrainingError(
            f"episode {episode.episode_id!r} renders to {len(input_ids)} tokens "
            f"> max_seq_len {max_seq_len}; truncation is forbidden (contract §7)")
    spans = {int(s.event_index): (int(s.start), int(s.end)) for s in rendered.spans}

    # group event indices into segments: cut AFTER a run of feedback events
    groups: list[list[int]] = []
    current: list[int] = []
    for index, event in enumerate(events):
        current.append(index)
        is_feedback = event.role in FEEDBACK_ROLES
        next_is_feedback = (index + 1 < len(events)
                            and events[index + 1].role in FEEDBACK_ROLES)
        if is_feedback and not next_is_feedback:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    if not groups:  # pragma: no cover - guarded by the empty-events check
        raise Fl5TrainingError("segmentation produced no segments")

    # segment char boundaries: segment g covers [start of its first event block
    # (or 0 for the first segment, which owns the instruction header),
    #  end of its last event block)
    boundaries: list[tuple[int, int]] = []
    for gi, group in enumerate(groups):
        start = 0 if gi == 0 else spans[group[0]][0]
        end = (len(text) if gi == len(groups) - 1 else spans[groups[gi + 1][0]][0])
        boundaries.append((start, end))

    # token assignment by START offset -> exact partition
    token_bounds: list[tuple[int, int]] = []
    cursor = 0
    for gi, (_start, end) in enumerate(boundaries):
        if gi == len(boundaries) - 1:
            stop = len(input_ids)
        else:
            stop = cursor
            while stop < len(input_ids) and offsets[stop][0] < end:
                stop += 1
        if stop <= cursor:
            raise Fl5TrainingError(
                f"segment {gi} of episode {episode.episode_id!r} contains no "
                "tokens; the rendering and the tokenisation disagree")
        token_bounds.append((cursor, stop))
        cursor = stop
    if cursor != len(input_ids):  # pragma: no cover - arithmetic invariant
        raise Fl5TrainingError("segment token partition is not exhaustive")

    segments: list[Segment] = []
    for gi, group in enumerate(groups):
        t_start, t_end = token_bounds[gi]
        feedback_items: list[tuple[int, str, int, int]] = []
        loss_spans: list[tuple[int, int, float, int, int]] = []
        for index in group:
            event = events[index]
            char_start, char_end = spans[index]
            if event.role in FEEDBACK_ROLES:
                a, b = char_span_to_token_span(
                    offsets, char_start, char_end, on_straddle="expand",
                    what=f"FL5 item span (event {index})", text=text)
                a, b = max(a, t_start), min(b, t_end)
                if b <= a:
                    raise Fl5TrainingError(
                        f"feedback event {index} has no tokens inside its own "
                        f"segment ({t_start}, {t_end})")
                feedback_items.append((index, str(event.item_id),
                                       a - t_start, b - t_start))
            elif event.role is Role.MODEL_ATTEMPT:
                interaction = event.interaction_index
                if interaction is None:
                    raise Fl5TrainingError(
                        f"MODEL_ATTEMPT event {index} has no interaction index; "
                        "the FL3 weights are keyed by it")
                weight = FL3_INTERACTION_WEIGHTS.get(int(interaction))
                if weight is None:
                    raise Fl5TrainingError(
                        f"no frozen FL3 weight for interaction index {interaction}")
                if weight <= 0.0:
                    continue
                for line_start, line_end in answer_spans_for_event(
                        text, char_start, char_end, event.text):
                    a, b = char_span_to_token_span(
                        offsets, line_start, line_end, on_straddle=on_straddle,
                        what=f"FL5 answer span (event {index})", text=text)
                    if a < t_start or b > t_end:
                        raise Fl5TrainingError(
                            f"answer span of event {index} crosses a segment "
                            "boundary; the segmentation is malformed")
                    if a - t_start == 0:
                        raise Fl5TrainingError(
                            f"answer span of event {index} starts at token 0 of "
                            "its segment; nothing would predict it")
                    loss_spans.append((index, int(interaction), float(weight),
                                       a - t_start, b - t_start))
        segments.append(Segment(
            index=gi, event_indices=tuple(group), token_start=t_start,
            token_end=t_end, feedback_items=tuple(feedback_items),
            loss_spans=tuple(loss_spans)))

    out = EpisodeSegments(
        episode_id=episode.episode_id, family_id=episode.family_id,
        split=str(episode.split), text=text, input_ids=tuple(input_ids),
        segments=tuple(segments))
    if out.total_weight <= 0.0:
        raise Fl5TrainingError(
            f"episode {episode.episode_id!r} carries zero FL3 loss weight; it "
            "would contribute nothing and must not enter a batch")
    return out


# ---------------------------------------------------------------------------
# per-segment loss
# ---------------------------------------------------------------------------
def segment_token_nll(logits: torch.Tensor, input_ids: torch.Tensor,
                      prefix_len: int) -> torch.Tensor:
    """Target-aligned per-token NLL of ONE segment, shape ``[T]``.

    ``logits`` covers ``[prefix; tokens]``.  Position 0 is 0.0 by the same
    convention W2 uses (``objectives.per_token_nll``): nothing in the segment
    predicts its own first token, and the FL3 mask never weights it.  A unit
    test pins equality with ``objectives.per_token_nll`` for ``prefix_len=0``.
    """
    if logits.dim() != 3 or logits.shape[0] != 1:
        raise Fl5TrainingError(
            f"segment logits must be [1, L, V], got {tuple(logits.shape)}")
    n_tokens = int(input_ids.shape[-1])
    if n_tokens < 2:
        raise Fl5TrainingError("a segment must contain at least 2 tokens")
    start = int(prefix_len)
    shifted = logits[0, start:start + n_tokens - 1, :].float()
    targets = input_ids.reshape(-1)[1:]
    nll = F.cross_entropy(shifted, targets, reduction="none")
    return torch.cat([torch.zeros(1, dtype=nll.dtype, device=nll.device), nll])


def episode_loss(bundle: Any, module: FastStateModule | None,
                 segments: EpisodeSegments, *, inject: bool,
                 detach_state_between_segments: bool = False,
                 ) -> tuple[torch.Tensor, dict[str, Any]]:
    """FL3-weighted, per-episode weighted-token-mean loss over the segments.

    ``inject=True`` is FAST_STATE_ON (prefix + state updates);
    ``inject=False`` is FAST_STATE_OFF (identical pipeline, no prefix, no state
    usage).  Returns ``(loss, stats)``; the loss keeps its graph.
    """
    model = bundle.model
    device = getattr(bundle, "device", None) or next(model.parameters()).device
    embed = model.get_input_embeddings()
    if inject and module is None:
        raise Fl5TrainingError("FAST_STATE_ON requires a FastStateModule")

    s = module.reset_state(1) if (inject and module is not None) else None
    total_weighted = None
    total_weight = 0.0
    prefix_len = 0
    n_updates = 0
    state_norms: list[float] = []
    forward_tokens = 0

    for segment in segments.segments:
        ids = torch.tensor([segments.input_ids[segment.token_start:segment.token_end]],
                           dtype=torch.long, device=device)
        if inject:
            prefix = module.prefix_embeds(s[0]).to(dtype=embed.weight.dtype)
            prefix_len = int(prefix.shape[0])
            embeds = torch.cat([prefix.unsqueeze(0), embed(ids)], dim=1)
            kwargs = {"inputs_embeds": embeds}
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
        else:
            prefix_len = 0
            kwargs = {"input_ids": ids}
            mask = torch.ones_like(ids)
        tap = forward_final_loop(model, attention_mask=mask, need_logits=True,
                                 **kwargs)
        forward_tokens += int(mask.shape[1])
        if tap.logits is None:  # pragma: no cover - guarded by need_logits
            raise Fl5TrainingError("segment forward produced no logits")

        if segment.loss_spans:
            nll = segment_token_nll(tap.logits, ids, prefix_len)
            weights = torch.zeros_like(nll)
            for _event, _interaction, weight, t0, t1 in segment.loss_spans:
                weights[t0:t1] = float(weight)
            weighted = (nll * weights).sum()
            total_weighted = weighted if total_weighted is None else total_weighted + weighted
            total_weight += float(weights.sum().item())

        if inject:
            hidden = tap.hidden[0]
            for _event, _item, t0, t1 in segment.feedback_items:
                u = hidden[prefix_len + t0: prefix_len + t1].mean(dim=0)
                s = module.update(s, u,
                                  detach_state=detach_state_between_segments)
                n_updates += 1
                state_norms.append(FastStateModule.state_norm(s))

    if total_weighted is None or total_weight <= 0.0:
        raise Fl5TrainingError(
            f"episode {segments.episode_id!r} produced no weighted loss tokens")
    loss = total_weighted / total_weight
    stats = {
        "episode_id": segments.episode_id,
        "family_id": segments.family_id,
        "n_segments": len(segments.segments),
        "n_state_updates": n_updates,
        "loss_weight": total_weight,
        "n_loss_tokens": int(sum(t1 - t0 for seg in segments.segments
                                 for *_x, t0, t1 in seg.loss_spans)),
        "forward_tokens": forward_tokens,
        "state_norms": state_norms,
        "final_state_norm": state_norms[-1] if state_norms else 0.0,
        "inject": bool(inject),
    }
    return loss, stats


def batch_loss(bundle: Any, module: FastStateModule | None,
               batch: Sequence[EpisodeSegments], *, inject: bool,
               detach_state_between_segments: bool = False
               ) -> tuple[torch.Tensor, dict[str, Any]]:
    """``OBJ_EPISODE_MEAN``: per-episode weighted token mean, then batch mean."""
    if not batch:
        raise Fl5TrainingError("empty batch")
    losses = []
    per_episode = []
    for segments in batch:
        loss, stats = episode_loss(
            bundle, module, segments, inject=inject,
            detach_state_between_segments=detach_state_between_segments)
        losses.append(loss)
        per_episode.append(stats)
    total = torch.stack(losses).mean()
    return total, {
        "objective": "OBJ_EPISODE_MEAN",
        "n_examples": len(batch),
        "loss": float(total.detach()),
        "per_episode": per_episode,
        "sum_weight": float(sum(s["loss_weight"] for s in per_episode)),
        "n_loss_tokens": int(sum(s["n_loss_tokens"] for s in per_episode)),
        "forward_tokens": int(sum(s["forward_tokens"] for s in per_episode)),
        "n_state_updates": int(sum(s["n_state_updates"] for s in per_episode)),
    }


# ---------------------------------------------------------------------------
# the arm loop
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FL5TrainConfig:
    """FL5 arm configuration.

    The optimizer / schedule fields are duck-typed for W2's
    ``build_optimizer`` / ``build_scheduler`` (they read exactly
    ``learning_rate``, ``betas``, ``eps``, ``weight_decay``, ``warmup_steps``
    and ``updates``), so FL5 uses the SAME frozen optimizer as every other arm
    without registering a fake core arm in ``training/arms.py``.
    """

    arm_id: str = ARM_FAST_STATE_ON
    learning_rate: float = 1e-4
    updates: int = 10
    episodes_per_batch: int = 2
    seed: int = 20260809
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_fraction: float = 0.03
    stage: str = "FL5"
    checkpoint_every_steps: int = 200
    checkpoint_every_seconds: float = 600.0
    detach_state_between_segments: bool = False
    train_backbone: bool = True

    def __post_init__(self) -> None:
        if self.arm_id not in (ARM_FAST_STATE_ON, ARM_FAST_STATE_OFF):
            raise Fl5TrainingError(
                f"arm_id must be {ARM_FAST_STATE_ON} or {ARM_FAST_STATE_OFF}")
        if self.updates < 1:
            raise Fl5TrainingError("updates must be >= 1")

    @property
    def inject(self) -> bool:
        return self.arm_id == ARM_FAST_STATE_ON

    @property
    def warmup_steps(self) -> int:
        return max(1, int(round(self.warmup_fraction * self.updates)))

    def to_dict(self) -> dict[str, Any]:
        data = {k: getattr(self, k) for k in (
            "arm_id", "learning_rate", "updates", "episodes_per_batch", "seed",
            "eps", "weight_decay", "grad_clip_norm", "warmup_fraction", "stage",
            "checkpoint_every_steps", "checkpoint_every_seconds",
            "detach_state_between_segments", "train_backbone")}
        data["betas"] = list(self.betas)
        return data

    def config_hash(self) -> str:
        return domain_sha256("FL_V0_FL5_ARM", self.to_dict())


@dataclass
class FL5ArmResult:
    arm_id: str
    out_dir: str
    steps: int
    stopped_reason: str
    final_loss: float | None
    loss_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    stats_history: list[dict[str, Any]] = field(default_factory=list)
    ledger: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    stability_events: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flb200.fl5_arm_result.v1",
            "arm_id": self.arm_id,
            "comparable_to_fl3": FL5_COMPARABLE_TO_FL3,
            "comparable_to_fl3_reason": FL5_NOT_COMPARABLE_REASON,
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
            "config": self.config,
            "config_hash": self.config_hash,
        }


def run_fast_state_arm(
    cfg: FL5TrainConfig,
    bundle: Any,
    module: FastStateModule,
    episodes: Sequence[Episode] | Sequence[EpisodeSegments],
    out_dir: str,
    *,
    backbone_parameters: Sequence[torch.nn.Parameter] | None = None,
    stability_config: StabilityConfig | None = None,
    on_step: Callable[[dict], str | None] | None = None,
) -> FL5ArmResult:
    """Run one FL5 arm (ON or OFF) with W2's optimizer / accounting / checkpoints.

    ``episodes`` may be raw episodes (segmented here, once) or pre-segmented
    :class:`EpisodeSegments`.  ``backbone_parameters`` are the trainable
    backbone parameters (e.g. ``training.peft_modes.apply_mode(...).parameters``);
    when omitted, only the fast-state module trains.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    prepared: list[EpisodeSegments] = []
    for episode in episodes:
        prepared.append(episode if isinstance(episode, EpisodeSegments)
                        else segment_episode(episode, bundle.tokenizer))
    if not prepared:
        raise Fl5TrainingError(f"{cfg.arm_id}: no episodes supplied")

    params: list[torch.nn.Parameter] = []
    if cfg.inject:
        params.extend(p for p in module.parameters() if p.requires_grad)
    if cfg.train_backbone and backbone_parameters:
        params.extend(p for p in backbone_parameters if p.requires_grad)
    if not params:
        raise Fl5TrainingError(
            f"{cfg.arm_id}: no trainable parameters. FAST_STATE_OFF trains the "
            "backbone only, so it needs backbone_parameters")

    optimizer = build_optimizer(params, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    # Amendment 6 leaves the fast-state bound to FL5; the mechanism supplies
    # its own tighter value through W2's ``fast_state_norm_max`` field.
    stability = stability_config or StabilityConfig(
        fast_state_norm_max=FL5_STATE_NORM_LIMIT)
    monitor = StabilityMonitor(stability, arm_id=cfg.arm_id)
    device = str(getattr(bundle, "device", "cpu"))
    ledger = ComputeLedger(arm_id=cfg.arm_id, stage=cfg.stage,
                           max_tokens_per_batch=0, planned_updates=cfg.updates,
                           device=device)
    result = FL5ArmResult(
        arm_id=cfg.arm_id, out_dir=out_dir, steps=0,
        stopped_reason="COMPLETED", final_loss=None,
        config=cfg.to_dict() | {"fast_state": module.config()},
        config_hash=cfg.config_hash())

    def _save(tag: str, step: int) -> dict[str, Any]:
        state = {f"fast_state.{k}": v for k, v in module.state_dict().items()}
        manifest = save_checkpoint(
            out_dir, tag, trained_state=state,
            trained_scope=f"{cfg.arm_id}+FAST_STATE", optimizer=optimizer,
            scheduler=scheduler, step=step, tokens=ledger.forward_tokens,
            loss_tokens=ledger.loss_tokens, arm_config=result.config,
            arm_config_hash=result.config_hash,
            base_identity=getattr(bundle, "identity", None), stage=cfg.stage,
            extra={"fast_state_config": module.config(),
                   "fast_state_config_hash": module.config_hash()})
        result.checkpoints.append({"tag": tag, "step": step,
                                   "manifest_hash": manifest.get("manifest_hash")})
        return manifest

    _save("initial", 0)
    ledger.start()
    model = bundle.model
    model.train()
    last_checkpoint = time.monotonic()
    step = 0
    epoch = 0
    order: list[int] = []
    cursor = 0
    while step < cfg.updates:
        if cursor >= len(order):
            order = epoch_order(len(prepared), cfg.seed, cfg.arm_id, epoch)
            cursor = 0
            epoch += 1
        batch_indices = order[cursor:cursor + cfg.episodes_per_batch]
        cursor += cfg.episodes_per_batch
        batch = [prepared[i] for i in batch_indices]

        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss, stats = batch_loss(
            bundle, module if cfg.inject else None, batch, inject=cfg.inject,
            detach_state_between_segments=cfg.detach_state_between_segments)
        loss.backward()
        grads_finite = all(torch.isfinite(p.grad).all().item()
                           for p in params if p.grad is not None)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm))
        loss_value = float(loss.detach())
        did_update = (grads_finite and math.isfinite(loss_value)
                      and math.isfinite(grad_norm))
        if did_update:
            optimizer.step()
        scheduler.step()
        step_seconds = time.monotonic() - t0

        step += 1
        result.steps = step
        result.loss_history.append(loss_value)
        result.grad_norm_history.append(grad_norm)
        result.lr_history.append(float(optimizer.param_groups[0]["lr"]))
        result.stats_history.append(stats)
        result.final_loss = loss_value

        ledger.record_step(
            n_examples=len(batch), forward_tokens=stats["forward_tokens"],
            forward_tokens_unpadded=stats["forward_tokens"],
            loss_tokens=stats["n_loss_tokens"],
            weighted_loss_mass=stats["sum_weight"], did_update=did_update,
            step_seconds=step_seconds,
            episode_ids=[s.episode_id for s in batch])
        monitor.observe_step(
            step=step, loss=loss_value, grad_norm=grad_norm,
            tokens=stats["forward_tokens"], step_seconds=step_seconds,
            grads_finite=grads_finite)
        if cfg.inject:
            monitor.observe_fast_state_norm(
                step, max((s["final_state_norm"] for s in stats["per_episode"]),
                          default=0.0))
        now = time.monotonic()
        due = (cfg.checkpoint_every_steps > 0 and step % cfg.checkpoint_every_steps == 0) \
            or (cfg.checkpoint_every_seconds > 0
                and (now - last_checkpoint) >= cfg.checkpoint_every_seconds)
        if step < cfg.updates and due:
            _save(f"step{step}", step)
            last_checkpoint = now
        if monitor.fatal:
            result.stopped_reason = "STOPPED_BY_STABILITY"
            break
        if on_step is not None and on_step({
                "step": step, "loss": loss_value, "grad_norm": grad_norm,
                "arm_id": cfg.arm_id, "stats": stats}) == "STOP":
            result.stopped_reason = "STOPPED_BY_HOOK"
            break

    ledger.stop()
    result.ledger = ledger.to_dict()
    result.stability_events = monitor.events_as_dicts()
    if result.stopped_reason != "STOPPED_BY_STABILITY":
        _save("final", result.steps)
    ledger.write(f"{out_dir}/compute_ledger_{cfg.arm_id}.json")
    # POST-CONDITION (Amendment 16): hand the backbone back in EVAL mode with
    # layer-level gradient checkpointing OFF, exactly as
    # ``training.trainer.run_training_arm`` does.  FL5's arms are evaluated
    # immediately afterwards, and a train-mode model with the checkpointing
    # flag set decodes with use_cache/past_key_values silently dropped by
    # transformers' GradientCheckpointingLayer.
    from foundation_learner.training.model_loading import set_evaluation_mode

    set_evaluation_mode(model)
    return result
