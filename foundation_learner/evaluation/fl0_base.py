"""FL0 — base model, no training (contract section 7).

Runs the ONLINE learning-curve harness over DEVELOPMENT episodes for the
frozen base bundle across the four frozen cells

    {structured, correctness_only} x {history, context_reset}

and emits the episode records plus an ``flb200.fl0_base_report.v1`` summary.
FL0 establishes ``R_0``, how much the base model improves from context alone,
its baseline related/whole-family transfer, and the history-only context-reset
control.  FL0 is NOT a treatment and is never described as one.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Callable, Sequence

from . import write_json_document, write_jsonl_records
from .context_reset import run_context_reset_eval
from .generation import GenerationConfig, run_equivalence_gate, generation_policy_from_gate
from .learning_curve import LearningCurveConfig, annotate_records, run_episodes
from . import metrics as _metrics

FL0_REPORT_SCHEMA = "flb200.fl0_base_report.v1"

FEEDBACK_CONDITIONS: tuple[str, ...] = ("structured", "correctness_only")
CONTEXT_MODES: tuple[str, ...] = ("history", "context_reset")


def load_episodes(shard_paths: Sequence[str], *, shards_module: Any = None,
                  episode_from_json: Any = None) -> list[Any]:
    """Read episodes from DEVELOPMENT shard files through the data layer.

    Adapts to the producer's actual API: ``data.shards.read_shard(path)``
    returns plain dicts and ``episodes.schema.Episode.from_dict`` rebuilds the
    dataclass (contract section 23 named the latter ``episode_from_json``).
    SEALED shards are NOT readable here by construction: deciphering lives only
    in ``campaign/sealed_gate.py``.
    """
    if shards_module is None:
        from foundation_learner.data import shards as shards_module  # type: ignore
    if episode_from_json is None:
        episode_from_json = getattr(shards_module, "episode_from_json", None)
    if episode_from_json is None:
        from foundation_learner.episodes.schema import Episode  # type: ignore
        episode_from_json = Episode.from_dict
    episodes: list[Any] = []
    for path in shard_paths:
        for row in shards_module.read_shard(path):
            episodes.append(episode_from_json(row))
    return episodes


def run_fl0_base(
    bundle: Any,
    episodes: Sequence[Any],
    env_factory: Callable[[Any], Any],
    *,
    generation: GenerationConfig | None = None,
    out_dir: str | None = None,
    arm_tag: str = "FL0_BASE",
    reset_from_index: int = 4,
    render_module: Any = None,
    parser: Callable[[str], str | None] | None = None,
    callbacks_factory: Callable[[Any], Any] | None = None,
    run_gate: bool = True,
    trained_family_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the four FL0 cells; returns the report and all records.

    The batched/unbatched equivalence gate (contract section 22) runs FIRST
    when ``run_gate`` is set, and its verdict determines the generation policy
    for every cell.  A failed gate does not stop FL0: it pins batch size 1.
    """
    gate = run_equivalence_gate(bundle) if run_gate else None
    gen_cfg = generation or GenerationConfig()
    if gate is not None:
        gen_cfg = generation_policy_from_gate(gate, gen_cfg)

    cells: dict[str, dict[str, Any]] = {}
    records_by_cell: dict[str, list[dict[str, Any]]] = {}

    for feedback_condition in FEEDBACK_CONDITIONS:
        base_cfg = LearningCurveConfig(
            feedback_condition=feedback_condition,
            context_mode="history",
            reset_from_index=reset_from_index,
            generation=gen_cfg,
            arm_tag=arm_tag,
        )
        history_records = run_episodes(
            bundle, episodes, env_factory, cfg=base_cfg,
            callbacks_factory=callbacks_factory, render_module=render_module,
            parser=parser)
        history_records = annotate_records(
            history_records, generalization_scope=None)
        key_h = f"{feedback_condition}__history"
        records_by_cell[key_h] = history_records
        cells[key_h] = _metrics.summarize(history_records, trained_family_ids)

        reset_bundle = run_context_reset_eval(
            bundle, episodes, env_factory,
            cfg=replace(base_cfg, context_mode="context_reset"),
            callbacks_factory=callbacks_factory, render_module=render_module,
            parser=parser, run_history_control=False)
        reset_records = reset_bundle["reset_records"]
        key_r = f"{feedback_condition}__context_reset"
        records_by_cell[key_r] = reset_records
        cells[key_r] = _metrics.summarize(reset_records, trained_family_ids)
        cells[key_r]["persistence"] = _metrics.context_reset_persistence(
            reset_records, history_records, reset_from_index=reset_from_index)

    report: dict[str, Any] = {
        "schema": FL0_REPORT_SCHEMA,
        "arm_tag": arm_tag,
        "n_episodes": len(episodes),
        "reset_from_index": int(reset_from_index),
        "generation": {
            "max_new_tokens": int(gen_cfg.max_new_tokens),
            "batch_size": gen_cfg.batch_size,
            "force_unbatched": bool(gen_cfg.force_unbatched),
        },
        "equivalence_gate": gate,
        "cells": cells,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for cell, recs in records_by_cell.items():
            write_jsonl_records(os.path.join(out_dir, f"fl0_{cell}_records.jsonl"), recs)
        write_json_document(os.path.join(out_dir, "fl0_base_report.json"), report)

    return {"report": report, "records_by_cell": records_by_cell}


__all__ = [
    "FL0_REPORT_SCHEMA",
    "FEEDBACK_CONDITIONS",
    "CONTEXT_MODES",
    "load_episodes",
    "run_fl0_base",
]
