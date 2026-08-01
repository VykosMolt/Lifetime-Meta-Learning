"""Batched one-shot residual edit at the frozen writable boundary.

Frozen locus (identical to the sealed serial hook, which remains the semantic
authority): paper loop L3 = zero-based loop index 2; physical layer 24 =
``model.model.layers[23]``; post-layer residual output; one-shot during
prefill; token position = final non-padding prompt token of EACH row.

The batched engine uses LEFT padding, so every row's final non-padding prompt
token sits at sequence position -1 — the same position the sealed serial hook
edits.  This hook therefore applies, per row b:

    delta[b] = sign[b] * alpha[b] * rms(h[b, -1, :]) * axis[direction[b]]

with per-row [1, HIDDEN] slices so each row's arithmetic is shape-identical
to the sealed serial hook's (bitwise-equal kernels at any batch size).

Guarantees enforced here and proven by tests/test_intervention.py:

  * no delta on padding (position -1 is never padding under left padding;
    an all-padding row is refused);
  * no delta on another row's position (per-row slice writes only row b);
  * baseline / zero-alpha rows take the identity path (no clone-plus-zero);
  * exactly one edit visit per prefill (current_ut == 2);
  * intended injected RMS (sealed float32-of-bf16 semantics) stored per row,
    separately from the measured realized delta RMS;
  * axis tensors are consumed as given — byte-identity to the frozen package
    is verified upstream by the sealed loader.
"""
from __future__ import annotations

import numpy as np
import torch

from . import sealed_import

sealed_import.ensure_sealed_path()

from run_o1_v2_generation import (  # noqa: E402 (sealed)
    HIDDEN, LOOP_INDEX, MODULE_INDEX, _rms,
)


class BatchedInterventionError(RuntimeError):
    pass


class BatchedOneShotResidualEdit:
    """Per-row one-shot residual edits for a left-padded batch.

    directions: list of per-row axis vectors (np.ndarray unit-RMS) or None.
    signed_alphas: list of per-row signed alpha (0.0 with None direction on
    baseline / zero-alpha rows).
    """

    def __init__(self, directions: list, signed_alphas: list[float],
                 attention_mask: torch.Tensor):
        if len(directions) != len(signed_alphas):
            raise BatchedInterventionError("directions/alphas length mismatch")
        self.directions = directions
        self.signed_alphas = [float(a) for a in signed_alphas]
        for direction, alpha in zip(directions, self.signed_alphas):
            if (direction is None) != (alpha == 0.0):
                raise BatchedInterventionError(
                    "direction and signed alpha must be null together")
        # LEFT padding invariant: last column all real tokens.
        if not bool(attention_mask[:, -1].all().item()):
            raise BatchedInterventionError(
                "left-padding invariant violated: a row's final position is "
                "padding — the frozen boundary position would be wrong")
        self.applied = 0
        self.injected_rms = [0.0] * len(directions)
        self.realized_delta_rms = [None] * len(directions)

    def __call__(self, _module, _args, kwargs, output):
        current_ut = kwargs.get("current_ut")
        if current_ut is None or int(current_ut) != LOOP_INDEX:
            return output
        self.applied += 1
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[-1] != HIDDEN:
            raise BatchedInterventionError(
                f"unexpected L3_24 output shape {tuple(hidden.shape)}")
        if hidden.shape[0] != len(self.directions):
            raise BatchedInterventionError(
                f"batch size {hidden.shape[0]} != {len(self.directions)} rows")
        if all(d is None for d in self.directions):
            return output  # identity path, exactly like the sealed hook
        changed = hidden.clone()
        for b, (direction, signed_alpha) in enumerate(
                zip(self.directions, self.signed_alphas)):
            if direction is None or signed_alpha == 0.0:
                continue  # this row keeps its untouched values
            d_t = torch.as_tensor(np.asarray(direction, dtype=np.float64),
                                  device=hidden.device, dtype=hidden.dtype)
            if not torch.isfinite(d_t).all():
                raise BatchedInterventionError(f"row {b}: non-finite direction")
            target = hidden[b:b + 1, -1, :]              # [1, HIDDEN] slice
            delta = signed_alpha * _rms(target).to(hidden.dtype) * d_t
            changed[b:b + 1, -1, :] = target + delta
            # intended RMS: sealed semantics (float32 RMS of the dtype delta)
            self.injected_rms[b] = float(_rms(delta).item())
            realized = changed[b:b + 1, -1, :] - target
            self.realized_delta_rms[b] = float(_rms(realized).item())
        if isinstance(output, tuple):
            return (changed, *output[1:])
        if isinstance(output, list):
            return [changed, *output[1:]]
        return changed


def register(model, hook: BatchedOneShotResidualEdit):
    return model.model.layers[MODULE_INDEX].register_forward_hook(
        hook, with_kwargs=True)
