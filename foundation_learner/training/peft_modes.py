"""Trainable-parameter modes (contract §8) and the FL7/FL8 adapter specs (§7).

Frozen, predeclared:

* ``PEFT_MODE``       LoRA r=16, alpha=32, dropout 0.0 on ALL attention
                      ``q_proj``/``v_proj`` plus MLP ``down_proj``.
                      Development LR grid {1e-4, 3e-4}.
* ``FULL_MODEL_MODE`` every parameter trainable.  Development LR grid
                      {1e-5, 3e-5}.

The SAME mode is used for all of FL1/FL2/FL3 — never mixed (§8, §20).  The
selection between the two modes is mechanical (§11 affordability rule, owned by
``campaign/scheduler.py``); nothing here may choose a mode from an outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from foundation_learner.training.lora import (
    LoraHandles,
    LoraSpec,
    attach_lora,
)

PEFT_MODE_NAME = "PEFT_MODE"
FULL_MODEL_MODE_NAME = "FULL_MODEL_MODE"

PEFT_LORA_SPEC = LoraSpec(
    target_suffixes=("q_proj", "v_proj", "down_proj"),
    rank=16,
    alpha=32.0,
    dropout=0.0,
    name="fl_core_peft",
)

#: FL7 bounded resettable fast adapter (contract §7): r=8, alpha=16, q/v only.
FL7_FAST_ADAPTER_SPEC = LoraSpec(
    target_suffixes=("q_proj", "v_proj"),
    rank=8,
    alpha=16.0,
    dropout=0.0,
    name="fl7_fast_adapter",
)
FL7_FAST_DELTA_MAX_FROBENIUS = 1.0
FL7_INNER_STEPS = 4
FL7_INNER_LR = 1e-3

#: FL8 slow bank (contract §7); consolidation scale eta = 0.5.
FL8_SLOW_BANK_SPEC = LoraSpec(
    target_suffixes=("q_proj", "v_proj"),
    rank=8,
    alpha=16.0,
    dropout=0.0,
    name="fl8_slow_bank",
)
FL8_CONSOLIDATION_SCALE = 0.5


class TrainableModeError(RuntimeError):
    """Raised on an unknown mode or on a mode mismatch across core arms."""


@dataclass(frozen=True)
class TrainableParamMode:
    name: str
    lora_spec: LoraSpec | None
    learning_rates: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "learning_rates": list(self.learning_rates),
            "lora_spec": self.lora_spec.to_dict() if self.lora_spec else None,
        }


PEFT_MODE = TrainableParamMode(
    name=PEFT_MODE_NAME, lora_spec=PEFT_LORA_SPEC, learning_rates=(1e-4, 3e-4)
)
FULL_MODEL_MODE = TrainableParamMode(
    name=FULL_MODEL_MODE_NAME, lora_spec=None, learning_rates=(1e-5, 3e-5)
)

MODES: dict[str, TrainableParamMode] = {
    PEFT_MODE_NAME: PEFT_MODE,
    FULL_MODEL_MODE_NAME: FULL_MODEL_MODE,
}


def get_mode(name: str) -> TrainableParamMode:
    try:
        return MODES[name]
    except KeyError:
        raise TrainableModeError(
            f"unknown trainable-parameter mode {name!r}; frozen modes are "
            f"{sorted(MODES)}"
        ) from None


@dataclass
class TrainableParams:
    mode: TrainableParamMode
    parameters: list
    lora_handles: LoraHandles | None
    num_trainable: int

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.name,
            "num_trainable_parameters": self.num_trainable,
            "num_trainable_tensors": len(self.parameters),
            "lora_spec": (
                self.lora_handles.spec.to_dict() if self.lora_handles else None
            ),
            "lora_modules": (
                self.lora_handles.names() if self.lora_handles else None
            ),
        }


def apply_mode(model, mode: TrainableParamMode | str, seed: int = 0) -> TrainableParams:
    """Configure ``model`` for the given mode and return the trainable params.

    PEFT_MODE attaches the frozen LoRA spec (base parameters frozen); the LoRA
    init is derived deterministically from ``seed``.
    """
    if isinstance(mode, str):
        mode = get_mode(mode)
    if mode.lora_spec is None:
        for param in model.parameters():
            param.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        return TrainableParams(
            mode=mode,
            parameters=params,
            lora_handles=None,
            num_trainable=sum(p.numel() for p in params),
        )
    spec = LoraSpec(
        target_suffixes=mode.lora_spec.target_suffixes,
        rank=mode.lora_spec.rank,
        alpha=mode.lora_spec.alpha,
        dropout=mode.lora_spec.dropout,
        seed=int(seed),
        freeze_base=mode.lora_spec.freeze_base,
        name=mode.lora_spec.name,
    )
    handles = attach_lora(model, spec)
    params = list(handles.parameters())
    return TrainableParams(
        mode=mode,
        parameters=params,
        lora_handles=handles,
        num_trainable=sum(p.numel() for p in params),
    )


def assert_same_mode_across_arms(modes: Sequence[str]) -> str:
    """Contract §8/§20: PEFT and FULL may never be mixed across core arms."""
    names = [m.name if isinstance(m, TrainableParamMode) else str(m) for m in modes]
    if not names:
        raise TrainableModeError("no arms supplied to the mode-matching check")
    for name in names:
        get_mode(name)
    unique = sorted(set(names))
    if len(unique) != 1:
        raise TrainableModeError(
            "REFUSED: core arms must share one trainable-parameter mode "
            f"(contract §8); got {names}"
        )
    return unique[0]
