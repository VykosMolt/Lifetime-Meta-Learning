"""Per-arm compute ledger (contract §8).

Records, per arm: optimizer updates, training (loss) tokens, forward tokens,
backward tokens, wall time, GPU seconds, examples, episodes.  Loss-token counts
differ across objectives by construction; they are RECORDED and REPORTED, never
hidden or equalised.

Written as ``flb200.compute_ledger.v1`` JSON (``indent=2, sort_keys=True`` plus
a trailing newline, mirroring the O1 packaging conventions of §22).

Wall time and GPU seconds are METADATA.  They never influence data order,
optimisation, or any scientific quantity; they feed the §11 scheduler and the
§17 throughput detector only.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

COMPUTE_LEDGER_SCHEMA = "flb200.compute_ledger.v1"


class ComputeMismatchError(RuntimeError):
    """Raised when core arms are not compute-matched (contract §8/§20)."""


@dataclass
class ComputeLedger:
    """Mutable accumulator; :meth:`to_dict` yields the frozen schema."""

    arm_id: str
    stage: str = "CORE"
    max_tokens_per_batch: int = 0
    planned_updates: int = 0
    optimizer_updates: int = 0
    skipped_updates: int = 0
    loss_tokens: int = 0
    weighted_loss_mass: float = 0.0
    forward_tokens: int = 0
    forward_tokens_unpadded: int = 0
    backward_tokens: int = 0
    examples: int = 0
    episodes: int = 0
    unique_episode_ids: set = field(default_factory=set)
    batches: int = 0
    wall_time_s: float = 0.0
    gpu_seconds: float = 0.0
    device: str = "cpu"
    notes: list[str] = field(default_factory=list)
    _t0: float | None = field(default=None, repr=False)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._t0 is not None:  # an interrupted segment is banked, not lost
            self.stop()
        self._t0 = time.monotonic()

    def stop(self) -> None:
        if self._t0 is not None:
            self.wall_time_s += time.monotonic() - self._t0
            self._t0 = None

    # -- recording ---------------------------------------------------------
    def record_step(
        self,
        *,
        n_examples: int,
        forward_tokens: int,
        forward_tokens_unpadded: int,
        loss_tokens: int,
        weighted_loss_mass: float,
        did_update: bool,
        step_seconds: float,
        episode_ids: Sequence[Any] = (),
        backward: bool = True,
    ) -> None:
        self.batches += 1
        self.examples += int(n_examples)
        self.forward_tokens += int(forward_tokens)
        self.forward_tokens_unpadded += int(forward_tokens_unpadded)
        if backward:
            # Backward covers exactly the tokens of the forward pass it belongs
            # to (no gradient accumulation in this campaign); kept as a separate
            # field because §8 requires it to be reported separately.
            self.backward_tokens += int(forward_tokens)
        self.loss_tokens += int(loss_tokens)
        self.weighted_loss_mass += float(weighted_loss_mass)
        if did_update:
            self.optimizer_updates += 1
        else:
            self.skipped_updates += 1
        for episode_id in episode_ids:
            if episode_id is not None:
                self.unique_episode_ids.add(episode_id)
        self.episodes = len(self.unique_episode_ids)
        if self.device.startswith("cuda"):
            self.gpu_seconds += float(step_seconds)

    def note(self, message: str) -> None:
        self.notes.append(str(message))

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPUTE_LEDGER_SCHEMA,
            "arm_id": self.arm_id,
            "stage": self.stage,
            "device": self.device,
            "planned_updates": int(self.planned_updates),
            "optimizer_updates": int(self.optimizer_updates),
            "skipped_updates": int(self.skipped_updates),
            "max_tokens_per_batch": int(self.max_tokens_per_batch),
            "batches": int(self.batches),
            "examples": int(self.examples),
            "episodes": int(self.episodes),
            "loss_tokens": int(self.loss_tokens),
            "weighted_loss_mass": float(self.weighted_loss_mass),
            "forward_tokens": int(self.forward_tokens),
            "forward_tokens_unpadded": int(self.forward_tokens_unpadded),
            "backward_tokens": int(self.backward_tokens),
            "wall_time_s": float(self.wall_time_s),
            "gpu_seconds": float(self.gpu_seconds),
            "tokens_per_second": (
                float(self.forward_tokens) / self.wall_time_s
                if self.wall_time_s > 0
                else None
            ),
            "notes": list(self.notes),
        }

    def write(self, path: str) -> str:
        payload = self.to_dict()
        tmp = f"{path}.tmp"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return path


def load_ledger(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") != COMPUTE_LEDGER_SCHEMA:
        raise ComputeMismatchError(
            f"{path}: unexpected schema {payload.get('schema')!r}"
        )
    return payload


#: Quantities that MUST be identical across the core arms (contract §8).
MATCHED_FIELDS = ("optimizer_updates", "max_tokens_per_batch")

#: Quantities that legitimately differ and are reported, never equalised.
REPORTED_FIELDS = (
    "loss_tokens",
    "weighted_loss_mass",
    "forward_tokens",
    "examples",
    "episodes",
    "wall_time_s",
    "gpu_seconds",
)


def compare_core_ledgers(ledgers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Refuse hidden unequal compute across core arms (contract §8, §18 hostile).

    Raises :class:`ComputeMismatchError` when the matched quantities differ.
    Returns a report of the intrinsically differing quantities so they can be
    printed alongside the headline comparison.
    """
    if len(ledgers) < 2:
        raise ComputeMismatchError(
            "core-arm compute comparison needs at least two ledgers"
        )
    for ledger in ledgers:
        if ledger.get("schema") != COMPUTE_LEDGER_SCHEMA:
            raise ComputeMismatchError(
                f"ledger for {ledger.get('arm_id')!r} has schema "
                f"{ledger.get('schema')!r}, expected {COMPUTE_LEDGER_SCHEMA}"
            )
    arms = [ledger.get("arm_id") for ledger in ledgers]
    if len(set(arms)) != len(arms):
        raise ComputeMismatchError(f"duplicate arm ids in ledger set: {arms}")
    problems = []
    for field_name in MATCHED_FIELDS:
        values = {ledger.get("arm_id"): ledger.get(field_name) for ledger in ledgers}
        if len(set(values.values())) != 1:
            problems.append(f"{field_name}: {values}")
    if problems:
        raise ComputeMismatchError(
            "REFUSED: core arms are not compute-matched (contract §8, option B: "
            "equal optimizer updates and equal per-update token budget); "
            + "; ".join(problems)
        )
    return {
        "schema": "flb200.compute_comparison.v1",
        "arms": arms,
        "matched": {
            field_name: ledgers[0].get(field_name) for field_name in MATCHED_FIELDS
        },
        "reported_differences": {
            field_name: {
                ledger.get("arm_id"): ledger.get(field_name) for ledger in ledgers
            }
            for field_name in REPORTED_FIELDS
        },
    }
