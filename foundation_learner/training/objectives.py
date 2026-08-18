"""Training objectives (contract §7).

Two reductions, both over the SAME per-token weighted NLL:

``OBJ_TOKEN_MEAN``
    ``sum_{b,t} w_bt * nll_bt / sum_{b,t} w_bt`` — the standard weighted NLL
    used by FL1 (isolated answer tokens) and FL2 (behaviour cloning over all
    MODEL_ATTEMPT answer tokens).

``OBJ_EPISODE_MEAN``
    per-episode weighted token mean, then batch mean:
    ``mean_b ( sum_t w_bt * nll_bt / sum_t w_bt )`` — contract §7 states this
    explicitly for FL3's ``L_meta``.

Weights are TARGET-aligned (see ``tokenization``): ``loss_mask[b, t]`` is the
weight of predicting ``input_ids[b, t]``.  The shift is performed here, once.

Losses are always accumulated in float32 even when the forward pass runs in
bfloat16 autocast.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

OBJ_TOKEN_MEAN = "OBJ_TOKEN_MEAN"
OBJ_EPISODE_MEAN = "OBJ_EPISODE_MEAN"
OBJECTIVES = (OBJ_TOKEN_MEAN, OBJ_EPISODE_MEAN)

#: Frozen arm -> objective binding (contract §7).
ARM_OBJECTIVES = {
    "FL1": OBJ_TOKEN_MEAN,
    "FL2": OBJ_TOKEN_MEAN,
    "FL3": OBJ_EPISODE_MEAN,
}


class ObjectiveError(RuntimeError):
    """Raised on malformed objective inputs (never silently repaired)."""


class ZeroLossWeightError(ObjectiveError):
    """An example carries no loss weight at all — masking is broken."""


def per_token_nll(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """Target-aligned per-token NLL, shape ``(B, T)``.

    Position 0 is 0.0 by construction (nothing predicts the first token); its
    weight is 0.0 by the tokenisation convention as well.
    """
    if logits.shape[:2] != input_ids.shape[:2]:
        raise ObjectiveError(
            f"logits {tuple(logits.shape)} incompatible with input_ids "
            f"{tuple(input_ids.shape)}"
        )
    batch, seq = input_ids.shape
    if seq < 2:
        raise ObjectiveError("sequences must have at least 2 tokens")
    shifted_logits = logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    nll = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(batch, seq - 1)
    zero = torch.zeros(batch, 1, dtype=nll.dtype, device=nll.device)
    return torch.cat([zero, nll], dim=1)


def weighted_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    objective: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``(loss, stats)`` for the requested reduction."""
    if objective not in OBJECTIVES:
        raise ObjectiveError(f"unknown objective {objective!r}; frozen: {list(OBJECTIVES)}")
    if loss_mask.shape != input_ids.shape:
        raise ObjectiveError(
            f"loss_mask {tuple(loss_mask.shape)} != input_ids {tuple(input_ids.shape)}"
        )
    weights = loss_mask.float()
    if torch.any(weights < 0):
        raise ObjectiveError("negative loss weights are not defined")
    if torch.any(weights[:, 0] != 0):
        raise ObjectiveError(
            "position 0 carries loss weight; the tokenisation convention "
            "forbids it (nothing predicts the first token)"
        )
    per_example_weight = weights.sum(dim=1)
    if torch.any(per_example_weight <= 0):
        bad = torch.nonzero(per_example_weight <= 0).flatten().tolist()
        raise ZeroLossWeightError(
            f"examples {bad} carry zero total loss weight; they contribute "
            "nothing and must not enter a batch"
        )

    nll = per_token_nll(logits, input_ids)
    weighted = nll * weights
    per_example_sum = weighted.sum(dim=1)
    per_example_mean = per_example_sum / per_example_weight

    if objective == OBJ_TOKEN_MEAN:
        loss = per_example_sum.sum() / per_example_weight.sum()
    else:
        loss = per_example_mean.mean()

    stats = {
        "objective": objective,
        "loss": float(loss.detach()),
        "sum_weight": float(per_example_weight.sum().detach()),
        "n_weighted_tokens": int((weights > 0).sum().item()),
        "n_examples": int(input_ids.shape[0]),
        "per_example_loss": [float(v) for v in per_example_mean.detach().cpu()],
        "mean_token_nll": float(
            (per_example_sum.sum() / per_example_weight.sum()).detach()
        ),
    }
    return loss, stats


def objective_for_arm(arm_id: str) -> str:
    try:
        return ARM_OBJECTIVES[arm_id]
    except KeyError:
        raise ObjectiveError(
            f"no frozen objective for arm {arm_id!r}; core arms are "
            f"{sorted(ARM_OBJECTIVES)}"
        ) from None
