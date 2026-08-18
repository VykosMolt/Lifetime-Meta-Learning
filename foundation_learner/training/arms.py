"""Frozen arm configurations and compute-matching rules (contract §7, §8).

Compute matching is contract option B: all core arms (FL1, FL2, FL3) run the
SAME number of optimizer updates ``U`` and the SAME per-update token budget
``max_tokens_per_batch``, i.e. FLOP-matched at equal updates.  Loss-token counts
differ across objectives by construction; they are recorded and reported, never
hidden (see ``compute_accounting``).

``U`` is a B200-derived mechanical quantity: the largest value of the frozen
ladder {600, 1200, 2400, 4800} that fits the §11 affordability inequality.  It
is passed in, never chosen here from an outcome.  The development grid runs on
FL3 only at 25 % of the core ``U``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

from foundation_learner.training.objectives import objective_for_arm
from foundation_learner.training.peft_modes import (
    FULL_MODEL_MODE_NAME,
    PEFT_MODE_NAME,
    assert_same_mode_across_arms,
    get_mode,
)
from foundation_learner.training.tokenization import MAX_SEQ_LEN

ARM_FL1 = "FL1"
ARM_FL2 = "FL2"
ARM_FL3 = "FL3"
CORE_ARMS = (ARM_FL1, ARM_FL2, ARM_FL3)

DATA_MODE_ISOLATED_PAIRS = "ISOLATED_PAIRS"
DATA_MODE_SUCCESS_EPISODES = "SUCCESS_EPISODES"
DATA_MODE_ORDERED_EPISODES = "ORDERED_EPISODES"

#: Frozen arm -> data mode binding (contract §7).
ARM_DATA_MODES = {
    ARM_FL1: DATA_MODE_ISOLATED_PAIRS,
    ARM_FL2: DATA_MODE_SUCCESS_EPISODES,
    ARM_FL3: DATA_MODE_ORDERED_EPISODES,
}

#: Contract §11: mechanical update ladder.
UPDATE_LADDER = (600, 1200, 2400, 4800)
#: Contract §8: the development grid runs at 25 % of the core update count.
GRID_UPDATE_FRACTION = 0.25

#: Contract §8 development grid.
LEARNING_RATE_GRID = {
    PEFT_MODE_NAME: (1e-4, 3e-4),
    FULL_MODEL_MODE_NAME: (1e-5, 3e-5),
}

STAGE_CORE = "CORE"
STAGE_GRID = "GRID"
STAGE_SMOKE = "SMOKE"  # local mechanics only; never a scientific result
STAGES = (STAGE_CORE, STAGE_GRID, STAGE_SMOKE)

# Frozen optimizer settings (contract §8).
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
WARMUP_FRACTION = 0.03
LR_SCHEDULE = "cosine"

ROOT_SEED_PRIMARY = 20260809
ROOT_SEED_SECONDARY = 20260810


class ArmConfigError(RuntimeError):
    """Raised on an arm configuration that violates the frozen contract."""


@dataclass(frozen=True)
class ArmConfig:
    """Frozen per-arm configuration."""

    arm_id: str
    data_mode: str
    objective: str
    learning_rate: float
    updates: int
    max_tokens_per_batch: int
    peft_mode: str
    seed: int
    stage: str = STAGE_CORE
    max_seq_len: int = MAX_SEQ_LEN
    betas: tuple[float, float] = ADAMW_BETAS
    eps: float = ADAMW_EPS
    weight_decay: float = WEIGHT_DECAY
    grad_clip_norm: float = GRAD_CLIP_NORM
    warmup_fraction: float = WARMUP_FRACTION
    lr_schedule: str = LR_SCHEDULE
    gradient_checkpointing: bool = True
    bf16: bool = True
    checkpoint_every_steps: int = 200  # contract §11
    checkpoint_every_seconds: float = 600.0  # contract §11
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["betas"] = list(self.betas)
        return data

    @property
    def warmup_steps(self) -> int:
        return max(1, int(round(self.warmup_fraction * self.updates)))


def validate_arm_config(cfg: ArmConfig) -> ArmConfig:
    """Hard-fail on anything outside the frozen contract."""
    problems: list[str] = []
    if cfg.arm_id not in CORE_ARMS:
        problems.append(f"arm_id {cfg.arm_id!r} is not a core arm {list(CORE_ARMS)}")
    elif cfg.data_mode != ARM_DATA_MODES[cfg.arm_id]:
        problems.append(
            f"data_mode {cfg.data_mode!r} != frozen {ARM_DATA_MODES[cfg.arm_id]!r}"
        )
    if cfg.arm_id in CORE_ARMS and cfg.objective != objective_for_arm(cfg.arm_id):
        problems.append(
            f"objective {cfg.objective!r} != frozen {objective_for_arm(cfg.arm_id)!r}"
        )
    if cfg.stage not in STAGES:
        problems.append(f"stage {cfg.stage!r} not in {list(STAGES)}")
    try:
        get_mode(cfg.peft_mode)
    except Exception as exc:  # noqa: BLE001 - re-raised below as a config problem
        problems.append(str(exc))
    else:
        allowed = LEARNING_RATE_GRID[cfg.peft_mode]
        if cfg.stage in (STAGE_CORE, STAGE_GRID) and cfg.learning_rate not in allowed:
            problems.append(
                f"learning_rate {cfg.learning_rate} is outside the frozen grid "
                f"{list(allowed)} for {cfg.peft_mode}; hyperparameter invention "
                "is forbidden (contract §8/§20)"
            )
    if cfg.updates <= 0:
        problems.append(f"updates must be positive, got {cfg.updates}")
    elif cfg.stage == STAGE_CORE and cfg.updates not in UPDATE_LADDER:
        problems.append(
            f"updates {cfg.updates} is not on the frozen ladder {list(UPDATE_LADDER)}"
        )
    elif cfg.stage == STAGE_GRID and cfg.updates not in grid_update_counts():
        problems.append(
            f"grid updates {cfg.updates} is not 25 % of a ladder value "
            f"{sorted(grid_update_counts())}"
        )
    if cfg.max_tokens_per_batch <= 0:
        problems.append("max_tokens_per_batch must be positive")
    if cfg.max_seq_len > MAX_SEQ_LEN:
        problems.append(
            f"max_seq_len {cfg.max_seq_len} exceeds the frozen budget {MAX_SEQ_LEN}"
        )
    if cfg.max_tokens_per_batch < cfg.max_seq_len:
        problems.append(
            f"max_tokens_per_batch {cfg.max_tokens_per_batch} cannot hold one "
            f"max-length example ({cfg.max_seq_len} tokens)"
        )
    if tuple(cfg.betas) != ADAMW_BETAS or cfg.eps != ADAMW_EPS:
        problems.append("optimizer betas/eps are frozen at (0.9, 0.95) / 1e-8")
    if cfg.weight_decay != WEIGHT_DECAY or cfg.grad_clip_norm != GRAD_CLIP_NORM:
        problems.append("weight_decay and grad clip are frozen at 0.01 / 1.0")
    if cfg.warmup_fraction != WARMUP_FRACTION or cfg.lr_schedule != LR_SCHEDULE:
        problems.append("schedule is frozen at cosine with 3 % warmup")
    if problems:
        raise ArmConfigError(
            f"invalid ArmConfig for {cfg.arm_id!r}: " + "; ".join(problems)
        )
    return cfg


def grid_update_counts() -> set[int]:
    return {int(round(u * GRID_UPDATE_FRACTION)) for u in UPDATE_LADDER}


def make_arm_config(
    arm_id: str,
    *,
    learning_rate: float,
    updates: int,
    max_tokens_per_batch: int,
    peft_mode: str,
    seed: int = ROOT_SEED_PRIMARY,
    stage: str = STAGE_CORE,
    validate: bool = True,
    **overrides,
) -> ArmConfig:
    """Build a frozen :class:`ArmConfig` for a core arm."""
    if arm_id not in CORE_ARMS:
        raise ArmConfigError(f"unknown arm {arm_id!r}; core arms are {list(CORE_ARMS)}")
    cfg = ArmConfig(
        arm_id=arm_id,
        data_mode=ARM_DATA_MODES[arm_id],
        objective=objective_for_arm(arm_id),
        learning_rate=float(learning_rate),
        updates=int(updates),
        max_tokens_per_batch=int(max_tokens_per_batch),
        peft_mode=peft_mode,
        seed=int(seed),
        stage=stage,
        **overrides,
    )
    return validate_arm_config(cfg) if validate else cfg


def make_core_arm_configs(
    *,
    learning_rate: float,
    updates: int,
    max_tokens_per_batch: int,
    peft_mode: str,
    seed: int = ROOT_SEED_PRIMARY,
    **overrides,
) -> dict[str, ArmConfig]:
    """All three matched core arms from one mechanical (U, budget, mode, LR)."""
    configs = {
        arm: make_arm_config(
            arm,
            learning_rate=learning_rate,
            updates=updates,
            max_tokens_per_batch=max_tokens_per_batch,
            peft_mode=peft_mode,
            seed=seed,
            **overrides,
        )
        for arm in CORE_ARMS
    }
    assert_core_arms_matched(list(configs.values()))
    return configs


def assert_core_arms_matched(configs: Sequence[ArmConfig]) -> None:
    """Contract §8/§20: refuse unequal compute or mixed modes across core arms."""
    if not configs:
        raise ArmConfigError("no arm configurations supplied to the matching check")
    arms = [c.arm_id for c in configs]
    if len(set(arms)) != len(arms):
        raise ArmConfigError(f"duplicate arms in the core comparison: {arms}")
    assert_same_mode_across_arms([c.peft_mode for c in configs])
    problems = []
    for field_name in (
        "updates",
        "max_tokens_per_batch",
        "learning_rate",
        "max_seq_len",
        "stage",
        "seed",
    ):
        values = {getattr(c, field_name) for c in configs}
        if len(values) != 1:
            problems.append(
                f"{field_name} differs across core arms: "
                + ", ".join(
                    f"{c.arm_id}={getattr(c, field_name)!r}" for c in configs
                )
            )
    if problems:
        raise ArmConfigError(
            "REFUSED: core arms are not compute-matched (contract §8): "
            + "; ".join(problems)
        )


def arm_config_hash(cfg: ArmConfig) -> str:
    from foundation_learner.training.model_loading import domain_sha256

    return domain_sha256("FL_V0_ARM_CONFIG", cfg.to_dict())
