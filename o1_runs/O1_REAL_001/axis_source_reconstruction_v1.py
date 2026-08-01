#!/usr/bin/env python3
"""Reconstruct the real pre-O1 d1, d2, and d3 L3_24 source vectors.

d4 is intentionally not approximated here. It requires live model capture with
both real adapter checkpoints on the separately frozen reference set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT = Path("/home/moloch/ouro_project").resolve()
D_MODEL = 2048
BOOTSTRAP_SEED = 20260617
BOOTSTRAP_DRAWS = 10_000

XLOOP_TAPS = (
    PROJECT
    / "artifacts/reports/cross_loop_early_layer_taps_20260720/xloop_taps.pt"
)
XLOOP_FEATURE_MANIFEST = (
    PROJECT
    / "artifacts/reports/cross_loop_early_layer_taps_20260720/"
    "features/feature_manifest.json"
)
XLOOP_SPLIT_MANIFEST = (
    PROJECT
    / "artifacts/reports/cross_loop_early_layer_taps_20260720/split_integrity.json"
)
XLOOP_FIT_SCRIPT = (
    PROJECT / "utilities/tests/manual/bg_xloop_early_v1_train_eval.py"
)
EXACT_BUNDLE = (
    PROJECT
    / "artifacts/reports/proto_introspection/"
    "s1_s3_exact_injection_orthogonality_2026-06-17/"
    "s1_s3_exact_injection_delta_bundle_2026-06-17.pt"
)
EXACT_TASK_MANIFEST = (
    PROJECT
    / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13/s1_1_task_set.json"
)
EXACT_RECONSTRUCTION_LOG = (
    PROJECT
    / "artifacts/reports/proto_introspection/"
    "s1_s3_exact_injection_orthogonality_2026-06-17/"
    "s1_s3_exact_injection_orthogonality_null_audit_2026-06-17.json"
)

EXPECTED_HASHES = {
    XLOOP_TAPS: "f50b5c0da8ead0a998d8a6b499a54fccd36d9fa05da17d0392d0fa2be7f5fd29",
    XLOOP_FEATURE_MANIFEST: "7288502e90e62cddc26314fb1d03690adbfb8c34404644fa9a30b61a19e1fca6",
    XLOOP_SPLIT_MANIFEST: "74c146e78e68daa20e2e6d129e57635141c71f19660ffc02d85ffd3123742c7d",
    XLOOP_FIT_SCRIPT: "214ab1bc5e00ff7afb0396f7abd7e0e097d9eaadd54452feac5f3e70800610dc",
    EXACT_BUNDLE: "ebb47f3975c5ea670b62678e7330c9a92e700a641b9f3ae6af51dc1320a92a38",
    EXACT_TASK_MANIFEST: "1f91a814a16c8f9d65c56fbc514c2bce0f9da1180a72c3cd19a23090a0d0d2be",
    EXACT_RECONSTRUCTION_LOG: "5136b06326fe0fb6c0caed46c55b8d12e4de1506e5332fe535be3f87efd4a768",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_row_sha256(vector: np.ndarray) -> str:
    row = np.ascontiguousarray(vector, dtype="<f8")
    if row.shape != (D_MODEL,):
        raise ValueError(f"canonical row must have shape [{D_MODEL}], got {row.shape}")
    return hashlib.sha256(row.tobytes(order="C")).hexdigest()


def unit_rms(vector: np.ndarray) -> tuple[np.ndarray, float]:
    row = np.asarray(vector, dtype=np.float64).reshape(-1)
    if row.shape != (D_MODEL,):
        raise ValueError(f"axis must have shape [{D_MODEL}], got {row.shape}")
    if not np.isfinite(row).all():
        raise ValueError("axis contains non-finite values")
    before = float(np.sqrt(np.mean(row * row)))
    if not np.isfinite(before) or before <= 1e-12:
        raise ValueError(f"axis RMS is invalid: {before}")
    normalized = np.ascontiguousarray(row / before, dtype="<f8")
    after = float(np.sqrt(np.mean(normalized * normalized)))
    if abs(after - 1.0) > 1e-12:
        raise ValueError(f"unit-RMS normalization failed: {after}")
    return normalized, before


def verify_expected_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        got = sha256_file(path)
        observed[str(path)] = got
        if got != expected:
            raise RuntimeError(
                f"frozen source hash mismatch for {path}: expected {expected}, got {got}"
            )
    return observed


def reconstruct_d1(taps_path: Path = XLOOP_TAPS) -> tuple[np.ndarray, dict[str, Any]]:
    taps = torch.load(taps_path, map_location="cpu", weights_only=False)
    if "24_L3" not in taps:
        raise KeyError("xloop_taps.pt has no 24_L3 cell")
    cell = taps["24_L3"]
    if cell.get("arch") != "AntisymLinearNoNorm":
        raise ValueError(f"unexpected d1 architecture: {cell.get('arch')!r}")
    raw = cell["weight"].detach().cpu().to(torch.float64).numpy().reshape(-1)
    vector, before = unit_rms(raw)
    return vector, {
        "source_key": "24_L3.weight",
        "source_shape": list(cell["weight"].shape),
        "source_dtype": str(cell["weight"].dtype),
        "selected_grid": cell.get("selected"),
        "trajectory_phase": "completed_candidate_state",
        "token_position": "all attention-mask-valid tokens",
        "pooling_rule": "attention-mask-valid mean; fp32 compute",
        "coordinate_mapping": {
            "standardization_inverse": "not_applicable",
            "pca_inverse": "not_applicable",
            "pooling_inverse": "not_applicable; the linear tap was fit directly on pooled raw residual vectors",
            "coefficient_to_residual": "identity; AntisymLinearNoNorm weight is already a 2048-d raw-residual covector",
        },
        "rms_before_normalization": before,
    }


def task_balanced_bootstrap_pc(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != D_MODEL or x.shape[0] != len(rows):
        raise ValueError(f"bad d2 feature/row geometry: {x.shape}, rows={len(rows)}")
    by_task_class: dict[tuple[str, bool], list[np.ndarray]] = defaultdict(list)
    for vector, row in zip(x, rows):
        by_task_class[(str(row["task_id"]), bool(row["correct"]))].append(vector)
    positive = {
        task: np.mean(np.stack(vectors), axis=0)
        for (task, label), vectors in by_task_class.items()
        if label
    }
    negative = {
        task: np.mean(np.stack(vectors), axis=0)
        for (task, label), vectors in by_task_class.items()
        if not label
    }
    if not positive or not negative:
        raise ValueError(
            f"d2 requires both verifier classes; positive_tasks={len(positive)} "
            f"negative_tasks={len(negative)}"
        )
    pos_tasks = sorted(positive)
    neg_tasks = sorted(negative)
    pos = np.stack([positive[task] for task in pos_tasks])
    neg = np.stack([negative[task] for task in neg_tasks])
    global_diff = pos.mean(axis=0) - neg.mean(axis=0)

    # Freeze the task bootstrap: sample each class's task pool separately, with
    # replacement, preserving equal task weight within each class. Compute the
    # leading right singular direction in the small exact span of task means.
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    pos_draw = rng.integers(0, len(pos_tasks), size=(int(draws), len(pos_tasks)))
    neg_draw = rng.integers(0, len(neg_tasks), size=(int(draws), len(neg_tasks)))
    ensemble = pos[pos_draw].mean(axis=1) - neg[neg_draw].mean(axis=1)
    source_span = np.concatenate([pos, neg], axis=0)
    q, _ = np.linalg.qr(source_span.T, mode="reduced")
    coordinates = ensemble @ q
    gram = coordinates.T @ coordinates
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    raw_pc = q @ eigenvectors[:, order[0]]
    if float(np.dot(raw_pc, global_diff)) < 0:
        raw_pc = -raw_pc
    vector, before = unit_rms(raw_pc)
    return vector, {
        "construction": "first right singular vector of uncentered task-bootstrap mean-difference ensemble",
        "bootstrap_seed": int(seed),
        "bootstrap_bit_generator": "numpy.PCG64",
        "bootstrap_draws": int(draws),
        "class_task_weighting": "equal task weight within verifier class",
        "positive_tasks": pos_tasks,
        "negative_tasks": neg_tasks,
        "positive_rows": int(sum(bool(row["correct"]) for row in rows)),
        "negative_rows": int(sum(not bool(row["correct"]) for row in rows)),
        "leading_singular_values": [
            float(np.sqrt(max(eigenvalues[index], 0.0))) for index in order[:5]
        ],
        "sign_alignment": "positive dot with task-balanced mu_correct-minus-mu_incorrect",
        "rms_before_normalization": before,
    }


def reconstruct_d2(
    bundle: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = list(bundle["branch_rows"])
    features = (
        bundle["branch_features"][:, 0, 2, :]
        .detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )
    vector, report = task_balanced_bootstrap_pc(features, rows)
    report.update(
        {
            "source_tensor_key": "branch_features[:, layer_slot_24=0, loop_slot_L3=2, :]",
            "source_tensor_shape": list(bundle["branch_features"].shape),
            "source_tensor_dtype": str(bundle["branch_features"].dtype),
            "physical_layer": 24,
            "loop": 3,
            "trajectory_phase": "completed_generated_candidate_state",
            "token_position": "all attention-mask-valid candidate-text tokens",
            "pooling_rule": "BGTransformerFeatureExtractor attention-mask-valid mean; max_length=512",
            "label_key": "branch_rows[].correct",
            "task_id_key": "branch_rows[].task_id",
        }
    )
    return vector, report


def reconstruct_d3(
    bundle: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = list(bundle["branch_rows"])
    selected = [
        index
        for index, row in enumerate(rows)
        if int(row["loop_1indexed"]) == 3 and int(row["layer"]) == 24
    ]
    if not selected:
        raise ValueError("exact delta bundle contains no L3_24 rows")
    delta = (
        bundle["delta_locus"][selected].detach().cpu().to(torch.float64).numpy()
    )
    calc = (
        bundle["h_injected_locus"][selected].detach().cpu().to(torch.float64)
        - bundle["h_clean_locus"][selected].detach().cpu().to(torch.float64)
    ).numpy()
    quantized_delta_max_abs_error = float(np.max(np.abs(delta - calc)))
    if quantized_delta_max_abs_error > 1e-4:
        raise ValueError(
            "stored delta_locus does not reproduce h_injected_locus-h_clean_locus "
            f"within float16 tolerance: {quantized_delta_max_abs_error}"
        )
    centered = delta - delta.mean(axis=0, keepdims=True)
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    raw_pc = vh[0]
    mean_delta = delta.mean(axis=0)
    if float(np.dot(raw_pc, mean_delta)) < 0:
        raw_pc = -raw_pc
    vector, before = unit_rms(raw_pc)
    explained = (singular * singular) / np.sum(singular * singular)
    return vector, {
        "source_tensor_key": "delta_locus[branch_rows loop_1indexed==3 and layer==24]",
        "source_tensor_shape": list(bundle["delta_locus"].shape),
        "source_tensor_dtype": str(bundle["delta_locus"].dtype),
        "selected_rows": len(selected),
        "selected_row_indices": selected,
        "physical_layer": 24,
        "loop": 3,
        "trajectory_phase": "one-shot injection at frozen last-token boundary",
        "token_position": "fixed last-token suffix (plen-1, plen)",
        "pooling_rule": "none beyond the one-token range mean",
        "centering": "column mean over exact L3_24 injection deltas",
        "construction": "leading right singular vector of centered L3_24 delta matrix",
        "sign_alignment": "positive dot with uncentered mean L3_24 delta",
        "quantized_delta_max_abs_error": quantized_delta_max_abs_error,
        "leading_singular_values": [float(value) for value in singular[:5]],
        "leading_explained_variance": float(explained[0]),
        "rms_before_normalization": before,
    }


def save_axis(path: Path, vector: np.ndarray) -> dict[str, Any]:
    np.save(path, np.ascontiguousarray(vector, dtype="<f8"), allow_pickle=False)
    reloaded = np.load(path, allow_pickle=False)
    if reloaded.shape != (D_MODEL,) or reloaded.dtype != np.dtype("<f8"):
        raise ValueError(f"saved axis has wrong geometry: {reloaded.shape} {reloaded.dtype}")
    if not np.array_equal(reloaded, vector):
        raise ValueError("saved axis differs from reconstruction bytes")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "canonical_row_sha256": canonical_row_sha256(reloaded),
        "shape": list(reloaded.shape),
        "dtype": str(reloaded.dtype),
        "finite": bool(np.isfinite(reloaded).all()),
        "rms": float(np.sqrt(np.mean(reloaded * reloaded))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    observed = verify_expected_hashes()

    d1, d1_report = reconstruct_d1()
    bundle = torch.load(EXACT_BUNDLE, map_location="cpu", weights_only=False)
    d2, d2_report = reconstruct_d2(bundle)
    d3, d3_report = reconstruct_d3(bundle)
    matrix = np.stack([d1, d2, d3])
    cosine = matrix @ matrix.T / D_MODEL

    saved = {
        "d1": save_axis(output / "d1_l3_24.npy", d1),
        "d2": save_axis(output / "d2_l3_24.npy", d2),
        "d3": save_axis(output / "d3_l3_24.npy", d3),
    }
    report = {
        "schema_version": "o1.axis_source_reconstruction.v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "scientific_scope": "real pre-O1 artifact reconstruction only; no O1 calibration or confirmatory outcomes",
        "locus": "L3_24",
        "coordinate_system": "raw 2048-dimensional Ouro-RLTT residual coordinates",
        "source_hashes": observed,
        "axes": {
            "d1_readable_process_quality": {**d1_report, **saved["d1"]},
            "d2_empirical_verifier_outcome": {**d2_report, **saved["d2"]},
            "d3_writable_transport": {**d3_report, **saved["d3"]},
            "d4_learned_actuator": {
                "status": "PENDING_LIVE_CAPTURE",
                "substitution_used": False,
            },
        },
        "d1_d3_cosine_matrix": cosine.tolist(),
    }
    report_path = output / "RECONSTRUCTION_REPORT.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

