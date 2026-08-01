"""Immutable sealed row specifications and their deterministic enumeration.

A RowSpec is the complete, immutable identity of one scientific row.  Every
backend receives RowSpecs and may not derive any scientific choice itself.
The enumeration order defined here is the CANONICAL execution-order index:
backends may execute in any order, but final records are always merged back
into this order.

The calibration enumeration mirrors the sealed v2.1 orchestrator loop
byte-for-byte in its identity choices:

  * per task (sealed manifest order), baseline rows for streams 0..7 first;
  * then for each alpha in the sealed grid, the eight signed actions
    (d, s) with the cyclic Latin square stream(a, g) = (a + rank_g) mod 8;
  * seeds via the sealed sha256_domain_uint64_be_v1 derivation.

The validation enumeration (non-O1 corpus) has the confirmatory shape —
baseline / zero_alpha / structured / random arms — so that every execution
path of every backend is exercised locally without touching any O1 task.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Iterable

from . import sealed_import
from .identity import (
    BANK_SIZE, action_index, canonical_json, derive_stream_seed, domain_sha256,
    text_sha256,
)

sealed_import.ensure_sealed_path()

from o1_prompt_template_v2 import render_prompt  # noqa: E402 (sealed)

SIGNED_ACTIONS = tuple(
    (d, s) for d in range(1, 5) for s in (1, -1))

ROW_ID_DOMAIN = "o1b200.row_id.v1"


@dataclasses.dataclass(frozen=True)
class RowSpec:
    """Immutable identity of one scientific row."""
    row_id: str                 # sealed row ID (domain-tagged digest, below)
    exec_index: int             # canonical execution-order index
    task_id: str
    task_content_sha256: str
    prompt_sha256: str          # digest of the rendered model-visible prompt
    generator_setting_id: str
    bank: str | None            # "structured" | "random" | None
    arm: str                    # "baseline" | "structured" | "zero_alpha" | "random" | ...
    direction_id: int | None    # 1..4 or None
    sign: int | None            # +1 / -1 or None
    alpha: float                # 0.0 on baseline / zero_alpha
    stream_index: int           # 0..7
    seed: int                   # sealed CRN stream seed (uint64)
    binding_key: str            # e.g. "calibration_precommit_sha256"
    binding_sha256: str         # the precommit / seal identity every record must carry

    def identity_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_content_sha256": self.task_content_sha256,
            "prompt_sha256": self.prompt_sha256,
            "generator_setting_id": self.generator_setting_id,
            "bank": self.bank,
            "arm": self.arm,
            "direction_id": self.direction_id,
            "sign": self.sign,
            "alpha": f"{float(self.alpha):.12g}",
            "stream_index": self.stream_index,
            "seed": self.seed,
            "binding_key": self.binding_key,
            "binding_sha256": self.binding_sha256,
        }

    def signed_alpha(self) -> float:
        if self.sign is None:
            return 0.0
        return float(self.sign) * float(self.alpha)

    def legacy_key(self) -> tuple:
        """The sealed orchestrator's completion key (task, arm, alpha, stream)."""
        return (self.task_id, self.arm, f"{float(self.alpha):.12g}",
                int(self.stream_index))


def row_id_of(identity: dict) -> str:
    return domain_sha256(ROW_ID_DOMAIN, identity)


class RowSpecError(ValueError):
    pass


def _mk(exec_index: int, task: dict, *, bank, arm, direction_id, sign, alpha,
        stream_index, seed, binding_key, binding_sha256) -> RowSpec:
    prompt_sha = text_sha256(render_prompt(task))
    base = {
        "task_id": task["task_id"],
        "task_content_sha256": task["task_content_sha256"],
        "prompt_sha256": prompt_sha,
        "generator_setting_id": task["generator_setting_id"],
        "bank": bank,
        "arm": arm,
        "direction_id": direction_id,
        "sign": sign,
        "alpha": f"{float(alpha):.12g}",
        "stream_index": stream_index,
        "seed": seed,
        "binding_key": binding_key,
        "binding_sha256": binding_sha256,
    }
    return RowSpec(
        row_id=row_id_of(base), exec_index=exec_index,
        task_id=task["task_id"],
        task_content_sha256=task["task_content_sha256"],
        prompt_sha256=prompt_sha,
        generator_setting_id=task["generator_setting_id"],
        bank=bank, arm=arm, direction_id=direction_id, sign=sign,
        alpha=float(alpha), stream_index=int(stream_index), seed=int(seed),
        binding_key=binding_key, binding_sha256=binding_sha256,
    )


def calibration_rowspecs(tasks: list[dict], alpha_grid: list[float],
                         master_seed: int, precommit_sha256: str) -> list[RowSpec]:
    """Canonical enumeration of the sealed calibration design (96*48 rows)."""
    specs: list[RowSpec] = []
    idx = 0
    for g, task in enumerate(tasks):
        rank = g % BANK_SIZE
        seeds = {i: derive_stream_seed(master_seed, task["task_id"], i)
                 for i in range(BANK_SIZE)}
        for i in range(BANK_SIZE):
            specs.append(_mk(idx, task, bank=None, arm="baseline",
                             direction_id=None, sign=None, alpha=0.0,
                             stream_index=i, seed=seeds[i],
                             binding_key="calibration_precommit_sha256",
                             binding_sha256=precommit_sha256))
            idx += 1
        for alpha in alpha_grid:
            for d, s in SIGNED_ACTIONS:
                stream = (action_index(d, s) + rank) % BANK_SIZE
                specs.append(_mk(idx, task, bank="structured", arm="structured",
                                 direction_id=d, sign=s, alpha=float(alpha),
                                 stream_index=stream, seed=seeds[stream],
                                 binding_key="calibration_precommit_sha256",
                                 binding_sha256=precommit_sha256))
                idx += 1
    _assert_unique(specs)
    return specs


def validation_rowspecs(tasks: list[dict], config: dict) -> list[RowSpec]:
    """Enumeration for the non-O1 validation corpus (confirmatory shape).

    config: {"master_seed": int, "binding_sha256": str,
             "alpha_structured": float, "alpha_random": float,
             "arms": [...]}  — sealed by the corpus manifest hash.
    """
    master = int(config["master_seed"])
    binding = str(config["binding_sha256"])
    arms = list(config["arms"])
    specs: list[RowSpec] = []
    idx = 0
    for g, task in enumerate(tasks):
        rank = g % BANK_SIZE
        seeds = {i: derive_stream_seed(master, task["task_id"], i)
                 for i in range(BANK_SIZE)}
        for arm in arms:
            if arm == "baseline":
                for i in range(BANK_SIZE):
                    specs.append(_mk(idx, task, bank=None, arm="baseline",
                                     direction_id=None, sign=None, alpha=0.0,
                                     stream_index=i, seed=seeds[i],
                                     binding_key="validation_corpus_sha256",
                                     binding_sha256=binding))
                    idx += 1
            elif arm == "zero_alpha":
                for i in range(BANK_SIZE):
                    specs.append(_mk(idx, task, bank=None, arm="zero_alpha",
                                     direction_id=None, sign=None, alpha=0.0,
                                     stream_index=i, seed=seeds[i],
                                     binding_key="validation_corpus_sha256",
                                     binding_sha256=binding))
                    idx += 1
            elif arm in ("structured", "random"):
                alpha = float(config[f"alpha_{arm}"])
                for d, s in SIGNED_ACTIONS:
                    stream = (action_index(d, s) + rank) % BANK_SIZE
                    specs.append(_mk(idx, task, bank=arm, arm=arm,
                                     direction_id=d, sign=s, alpha=alpha,
                                     stream_index=stream, seed=seeds[stream],
                                     binding_key="validation_corpus_sha256",
                                     binding_sha256=binding))
                    idx += 1
            else:
                raise RowSpecError(f"unknown arm {arm!r}")
    _assert_unique(specs)
    return specs


def validation_sweep_rowspecs(tasks: list[dict], config: dict) -> list[RowSpec]:
    """Calibration-shaped enumeration over the validation corpus: every alpha
    of the sealed grid is exercised (structured bank), plus baselines."""
    master = int(config["master_seed"])
    binding = str(config["binding_sha256"])
    grid = [float(a) for a in config["alpha_grid"]]
    specs: list[RowSpec] = []
    idx = 0
    for g, task in enumerate(tasks):
        rank = g % BANK_SIZE
        seeds = {i: derive_stream_seed(master, task["task_id"], i)
                 for i in range(BANK_SIZE)}
        for i in range(BANK_SIZE):
            specs.append(_mk(idx, task, bank=None, arm="baseline",
                             direction_id=None, sign=None, alpha=0.0,
                             stream_index=i, seed=seeds[i],
                             binding_key="validation_corpus_sha256",
                             binding_sha256=binding))
            idx += 1
        for alpha in grid:
            for d, s in SIGNED_ACTIONS:
                stream = (action_index(d, s) + rank) % BANK_SIZE
                specs.append(_mk(idx, task, bank="structured", arm="structured",
                                 direction_id=d, sign=s, alpha=alpha,
                                 stream_index=stream, seed=seeds[stream],
                                 binding_key="validation_corpus_sha256",
                                 binding_sha256=binding))
                idx += 1
    _assert_unique(specs)
    return specs


def _assert_unique(specs: Iterable[RowSpec]) -> None:
    specs = list(specs)
    ids = [s.row_id for s in specs]
    keys = [s.legacy_key() for s in specs]
    if len(set(ids)) != len(ids):
        raise RowSpecError("duplicate row_id in enumeration")
    if len(set(keys)) != len(keys):
        raise RowSpecError("duplicate (task, arm, alpha, stream) in enumeration")
    for want, spec in enumerate(specs):
        if spec.exec_index != want:
            raise RowSpecError("exec_index is not the enumeration position")


def spec_manifest_sha256(specs: list[RowSpec]) -> str:
    """Digest of the full sealed row-spec list (order-sensitive)."""
    return domain_sha256("o1b200.rowspec_manifest.v1",
                         [s.identity_dict() for s in specs])
