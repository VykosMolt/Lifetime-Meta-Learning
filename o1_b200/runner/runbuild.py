"""Builders for validation-run bundles (non-O1 corpus only).

The ONLY bundle builders in this package target the validation corpus.  Real
O1 calibration bundles are deliberately NOT constructible here: the future
B200 calibration runs through the sealed v2.1 orchestrator semantics with the
selected backend, after the frozen selection policy and the external
precommit — see docs/B200_ACCESS_RUNBOOK.md.
"""
from __future__ import annotations

import os

import numpy as np

from . import sealed_import
from .backend_interface import RunBundle
from .backends import model_artifact_sha256
from .identity import domain_sha256, sha256_tree
from .row_specs import (
    spec_manifest_sha256, validation_rowspecs, validation_sweep_rowspecs,
)
from .validation_corpus import load_corpus

sealed_import.ensure_sealed_path()

from run_o1_v2_orchestrator import load_axis_tensor  # noqa: E402 (sealed)

_HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
AXIS_DIR = os.path.join(WORKTREE_ROOT, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN",
                        "AXIS_PACKAGE_V2")
AXIS_PACKAGE_TREE_SHA256 = (
    "8c3b34a3574fbca9e63de23a88831e076e317fec33236405fb3af72f0ff96fdd")
STRUCTURED_TENSOR_SHA256 = (
    "72107b8eec817930652016c768fa90529a5989d82129b3d4a44f8d0b593946db")
RANDOM_TENSOR_SHA256 = (
    "1ff455edd3f2982ea45b6c11085887e46ece392e4e5eafab7916262f6fd7660b")

SYNTHETIC_TOKENIZER_SHA256 = domain_sha256(
    "o1b200.tokenizer.v1", {"kind": "synthetic_tokenizer_v1"})

O1_MANIFEST_PATHS = [
    os.path.join(WORKTREE_ROOT, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN",
                 "COHORTS", "calibration_tasks.jsonl"),
    os.path.join(WORKTREE_ROOT, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN",
                 "COHORTS", "confirmatory_candidate_pool.jsonl"),
]


def load_frozen_axes(verify_tree: bool = True) -> dict:
    if verify_tree and sha256_tree(AXIS_DIR) != AXIS_PACKAGE_TREE_SHA256:
        raise RuntimeError("frozen axis package tree hash mismatch")
    structured = load_axis_tensor(
        os.path.join(AXIS_DIR, "axes_l3_24.npy"),
        STRUCTURED_TENSOR_SHA256, "structured")
    random_axes = load_axis_tensor(
        os.path.join(AXIS_DIR, "random_axes_l3_24.npy"),
        RANDOM_TENSOR_SHA256, "random")
    return {"structured": structured, "random": random_axes}


def build_validation_bundle(corpus_dir: str, model_artifact: dict, *,
                            shape: str = "confirmatory",
                            task_subset: list[str] | None = None,
                            axes: dict | None = None) -> RunBundle:
    tasks, config = load_corpus(corpus_dir)
    if task_subset is not None:
        chosen = [t for t in tasks if t["task_id"] in set(task_subset)]
        if len(chosen) != len(task_subset):
            raise ValueError("task_subset contains unknown task ids")
        tasks = chosen
    spec_config = {
        "master_seed": config["master_seed"],
        "binding_sha256": config["binding_sha256"],
        "arms": config["arms"],
        "alpha_structured": config["alpha_structured"],
        "alpha_random": config["alpha_random"],
        "alpha_grid": config["alpha_grid"],
    }
    if shape == "confirmatory":
        specs = validation_rowspecs(tasks, spec_config)
    elif shape == "sweep":
        specs = validation_sweep_rowspecs(tasks, spec_config)
    else:
        raise ValueError(f"unknown shape {shape!r}")
    if axes is None:
        axes = load_frozen_axes()
    artifact_ids = {
        "package_zip_sha256": sealed_import.BASE_PACKAGE_ZIP_SHA256,
        "source_commit": sealed_import.SOURCE_COMMIT,
        "axis_package_sha256": AXIS_PACKAGE_TREE_SHA256,
        "model_artifact_sha256": model_artifact_sha256(model_artifact),
        "tokenizer_sha256": SYNTHETIC_TOKENIZER_SHA256
        if model_artifact.get("kind") == "synthetic"
        else model_artifact.get("tokenizer_sha256", "UNKNOWN"),
    }
    return RunBundle(
        tasks_by_id={t["task_id"]: t for t in tasks},
        specs=specs,
        artifact_ids=artifact_ids,
        axes=axes,
        row_spec_manifest_sha256=spec_manifest_sha256(specs),
        binding_key="validation_corpus_sha256",
        binding_sha256=config["binding_sha256"],
        task_manifest_sha256=config["tasks_sha256"],
        seed_matrix_sha256=domain_sha256(
            "o1b200.seed_matrix.v1",
            {"scheme": "sha256_domain_uint64_be_v1",
             "master_seed": config["master_seed"]}),
        seal_sha256=config["binding_sha256"],
    )
