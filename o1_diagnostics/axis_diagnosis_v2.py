#!/usr/bin/env python3
"""Bounded AXIS_DIAGNOSIS_V2 for the failed real O1 axis reconstruction.

This script consumes frozen pre-O1 artifacts and the preserved O1_REAL_001
capture. It does not construct axes, calibration manifests, or generations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


BASE_COMMIT = "e4776dd41a85cad699ac36f309b5986ab48bd171"
PRESERVED_CHECKPOINT_TREE_SHA256 = (
    "7dbbe4ef91de369bd712bad587c1af4f7c5dc42deff5377d4d70086c4c08a802"
)
CAUSAL_SHA256 = "f0831a9abf846cdf702d1211da346e2de3d32fe2ce11275247bad8400af215f4"
SEQUENCE_SHA256 = "75140bf3af7dfba4ee6851a500693b019f884e25dad0a2112f493dc99630597c"
EXACT_BUNDLE_SHA256 = "ebb47f3975c5ea670b62678e7330c9a92e700a641b9f3ae6af51dc1320a92a38"
REFERENCE_SHA256 = "d145955fdd15a7c413fae26276d4a61a0d4eab05f5241b73365982e4ed49c75b"
CAPTURE_REPORT_SHA256 = (
    "56e525c99f98fb1ab5f6aff0e99d99f50b8b49643d96c4ea9a134e2d0e80e2e3"
)
CAUSAL_UPDATES_SHA256 = (
    "6388d7fc1477311ab6f75c02736451e60be1a925f5f4a46c0edba2eb23adfc92"
)
SEQUENCE_UPDATES_SHA256 = (
    "484cfd9adf8b4c6bb11a80fa6aab149711c52af7ce9bc25a06a7fb0979c87775"
)
HISTORICAL_PROXY_REPORTED = 0.951094388961792
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260729
REQUESTED_K = (1, 2, 4, 8, 16)
D_MODEL = 2048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"frozen source hash mismatch: {path}; expected {expected}, got {observed}"
        )
    return observed


def checkpoint_tree_hash(source_root: Path) -> str:
    model = source_root / "models/ouro_rltt_local"
    rows = []
    for path in sorted(child for child in model.rglob("*") if child.is_file()):
        relative = path.relative_to(source_root).as_posix()
        rows.append(f"{relative}\0{sha256_file(path)}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-15:
        return None
    return float(np.dot(aa, bb) / denom)


def summarize(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0, "status": "UNAVAILABLE"}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "minimum": float(np.min(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
        "maximum": float(np.max(finite)),
    }


def centered_basis(matrix: np.ndarray) -> dict[str, Any]:
    x = np.asarray(matrix, dtype=np.float64)
    mean = np.mean(x, axis=0)
    centered = x - mean
    gram = centered @ centered.T
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    singular = np.sqrt(eigvals)
    tol = (
        float(singular[0])
        * max(centered.shape)
        * math.sqrt(np.finfo(np.float64).eps)
        if singular.size and singular[0] > 0
        else 0.0
    )
    rank = int(np.sum(singular > tol))
    rows = []
    for index in range(rank):
        row = eigvecs[:, order[index]].T @ centered
        row = row / singular[index]
        rows.append(row)
    basis = np.stack(rows) if rows else np.zeros((0, x.shape[1]), dtype=np.float64)
    if rank:
        if cosine(basis[0], mean) is not None and float(np.dot(basis[0], mean)) < 0:
            basis[0] *= -1
    total = float(np.sum(singular * singular))
    explained = (
        singular * singular / total
        if total > 0
        else np.zeros_like(singular, dtype=np.float64)
    )
    return {
        "matrix": x,
        "mean": mean,
        "centered": centered,
        "singular": singular,
        "basis": basis,
        "effective_rank": rank,
        "explained": explained,
        "tolerance": tol,
    }


def subspace_comparison(a: np.ndarray, b: np.ndarray, k: int) -> dict[str, Any]:
    if a.shape[0] < k or b.shape[0] < k:
        return {
            "status": "UNAVAILABLE_INSUFFICIENT_EFFECTIVE_RANK",
            "requested_k": k,
            "rank_a": int(a.shape[0]),
            "rank_b": int(b.shape[0]),
        }
    canonical = np.linalg.svd(a[:k] @ b[:k].T, compute_uv=False)
    clipped = np.clip(canonical, 0.0, 1.0)
    return {
        "status": "ESTIMABLE",
        "requested_k": k,
        "canonical_cosines": [float(value) for value in clipped],
        "principal_angles_degrees": [
            float(value) for value in np.degrees(np.arccos(clipped))
        ],
        "projection_overlap_mean_cosine_squared": float(np.mean(clipped * clipped)),
        "smallest_canonical_cosine": float(np.min(clipped)),
        "largest_principal_angle_degrees": float(
            np.max(np.degrees(np.arccos(clipped)))
        ),
    }


def requested_explained(decomp: dict[str, Any]) -> dict[str, Any]:
    explained = decomp["explained"]
    rank = int(decomp["effective_rank"])
    out: dict[str, Any] = {}
    for k in REQUESTED_K:
        key = f"pc_{k}"
        if k > rank:
            out[key] = {
                "status": "UNAVAILABLE_INSUFFICIENT_EFFECTIVE_RANK",
                "requested_pc": k,
                "effective_rank": rank,
                "maximum_centered_rank": int(decomp["matrix"].shape[0] - 1),
            }
        else:
            out[key] = {
                "status": "ESTIMABLE",
                "individual_explained_variance": float(explained[k - 1]),
                "cumulative_explained_variance_through_pc": float(
                    np.sum(explained[:k])
                ),
            }
    return out


def bootstrap_adapter(
    matrix: np.ndarray,
    full: dict[str, Any],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed))
    n_tasks = int(matrix.shape[0])
    pc1_values: list[float] = []
    mean_values: list[float] = []
    overlaps: dict[int, list[float]] = {k: [] for k in REQUESTED_K}
    invalid: Counter[int] = Counter()
    for _ in range(draws):
        indices = rng.integers(0, n_tasks, size=n_tasks)
        sample = matrix[indices]
        sample_decomp = centered_basis(sample)
        if sample_decomp["effective_rank"] >= 1 and full["effective_rank"] >= 1:
            value = cosine(sample_decomp["basis"][0], full["basis"][0])
            if value is not None:
                pc1_values.append(abs(value))
        mean_value = cosine(sample_decomp["mean"], full["mean"])
        if mean_value is not None:
            mean_values.append(mean_value)
        for k in REQUESTED_K:
            result = subspace_comparison(
                sample_decomp["basis"], full["basis"], k
            )
            if result["status"] == "ESTIMABLE":
                overlaps[k].append(
                    float(result["projection_overlap_mean_cosine_squared"])
                )
            else:
                invalid[k] += 1
    return {
        "unit": "task_id; one frozen reference example per task",
        "draws": draws,
        "seed": seed,
        "bit_generator": "numpy.PCG64",
        "row_independence_assumed": False,
        "pc1_absolute_cosine_to_full": summarize(pc1_values),
        "mean_update_cosine_to_full": summarize(mean_values),
        "top_k_subspace_projection_overlap_to_full": {
            str(k): {
                **summarize(overlaps[k]),
                "invalid_draws_insufficient_rank": int(invalid[k]),
            }
            for k in REQUESTED_K
        },
    }


def leave_one_task_out(
    matrix: np.ndarray, full: dict[str, Any], task_ids: list[str]
) -> dict[str, Any]:
    rows = []
    for index, task_id in enumerate(task_ids):
        subset = np.delete(matrix, index, axis=0)
        decomp = centered_basis(subset)
        pc1 = (
            abs(cosine(decomp["basis"][0], full["basis"][0]) or 0.0)
            if decomp["effective_rank"] and full["effective_rank"]
            else None
        )
        mean_cos = cosine(decomp["mean"], full["mean"])
        subspaces = {
            str(k): subspace_comparison(decomp["basis"], full["basis"], k)
            for k in REQUESTED_K
        }
        rows.append(
            {
                "left_out_task_id": task_id,
                "remaining_tasks": len(task_ids) - 1,
                "effective_rank": int(decomp["effective_rank"]),
                "pc1_absolute_cosine_to_full": pc1,
                "mean_update_cosine_to_full": mean_cos,
                "top_k_subspaces": subspaces,
            }
        )
    pc1_values = [
        float(row["pc1_absolute_cosine_to_full"])
        for row in rows
        if row["pc1_absolute_cosine_to_full"] is not None
    ]
    mean_values = [
        float(row["mean_update_cosine_to_full"])
        for row in rows
        if row["mean_update_cosine_to_full"] is not None
    ]
    return {
        "rows": rows,
        "pc1_absolute_cosine_summary": summarize(pc1_values),
        "mean_update_cosine_summary": summarize(mean_values),
    }


def adapter_report(
    name: str, matrix: np.ndarray, task_ids: list[str], seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    decomp = centered_basis(matrix)
    rms = np.sqrt(np.mean(matrix * matrix, axis=1))
    mean_rms = float(np.sqrt(np.mean(decomp["mean"] * decomp["mean"])))
    sign_dot = (
        float(np.dot(decomp["basis"][0], decomp["mean"]))
        if decomp["effective_rank"]
        else None
    )
    report = {
        "adapter": name,
        "matrix_shape": list(matrix.shape),
        "matrix_dtype_after_load": str(matrix.dtype),
        "finite": bool(np.isfinite(matrix).all()),
        "task_ids_in_frozen_order": task_ids,
        "task_count": len(task_ids),
        "one_row_per_task": len(set(task_ids)) == len(task_ids) == matrix.shape[0],
        "update_rms_per_task": [float(value) for value in rms],
        "mean_update_rms": mean_rms,
        "singular_values": [float(value) for value in decomp["singular"]],
        "effective_rank": int(decomp["effective_rank"]),
        "rank_tolerance": float(decomp["tolerance"]),
        "explained_variance_requested": requested_explained(decomp),
        "pc1_sign_rule": "positive dot with the uncentered mean update",
        "pc1_sign_dot_mean_after_alignment": sign_dot,
        "task_clustered_bootstrap": bootstrap_adapter(
            matrix, decomp, seed=seed, draws=BOOTSTRAP_DRAWS
        ),
        "leave_one_task_out": leave_one_task_out(matrix, decomp, task_ids),
    }
    return report, decomp


def partial_procrustes_predict(
    train_a: np.ndarray,
    train_b: np.ndarray,
    test_a: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean_a = np.mean(train_a, axis=0)
    mean_b = np.mean(train_b, axis=0)
    a = train_a - mean_a
    b = train_b - mean_b
    qa, ra = np.linalg.qr(a.T, mode="reduced")
    qb, rb = np.linalg.qr(b.T, mode="reduced")
    core = ra @ rb.T
    u_small, singular, vt_small = np.linalg.svd(core, full_matrices=False)
    tolerance = (
        float(singular[0]) * max(core.shape) * np.finfo(np.float64).eps
        if singular.size and singular[0] > 0
        else 0.0
    )
    rank = int(np.sum(singular > tolerance))
    left = qa @ u_small[:, :rank]
    right = qb @ vt_small.T[:, :rank]
    centered_test = test_a - mean_a
    prediction = (centered_test @ left) @ right.T + mean_b
    return prediction, {
        "mapping": "minimum-norm thin orthogonal Procrustes map in the train-task span",
        "train_rank": rank,
        "singular_values": [float(value) for value in singular[:rank]],
        "unidentified_orthogonal_complement": True,
    }


def cross_validated_procrustes(
    a: np.ndarray, b: np.ndarray, task_ids: list[str]
) -> dict[str, Any]:
    rows = []
    for index, task_id in enumerate(task_ids):
        mask = np.ones(len(task_ids), dtype=bool)
        mask[index] = False
        pred_b, meta_ab = partial_procrustes_predict(a[mask], b[mask], a[index])
        pred_a, meta_ba = partial_procrustes_predict(b[mask], a[mask], b[index])
        cos_ab = cosine(pred_b, b[index])
        cos_ba = cosine(pred_a, a[index])
        rows.append(
            {
                "held_out_task_id": task_id,
                "a_to_b_cosine": cos_ab,
                "b_to_a_cosine": cos_ba,
                "a_to_b_relative_l2_error": float(
                    np.linalg.norm(pred_b - b[index])
                    / max(np.linalg.norm(b[index]), 1e-15)
                ),
                "b_to_a_relative_l2_error": float(
                    np.linalg.norm(pred_a - a[index])
                    / max(np.linalg.norm(a[index]), 1e-15)
                ),
                "a_to_b_fit": meta_ab,
                "b_to_a_fit": meta_ba,
            }
        )
    return {
        "cross_validation": "leave-one-task-out",
        "row_independence_assumed": False,
        "rows": rows,
        "a_to_b_cosine_summary": summarize(
            [float(row["a_to_b_cosine"]) for row in rows if row["a_to_b_cosine"] is not None]
        ),
        "b_to_a_cosine_summary": summarize(
            [float(row["b_to_a_cosine"]) for row in rows if row["b_to_a_cosine"] is not None]
        ),
        "limitation": (
            "seven train tasks identify at most a six-dimensional centered map; "
            "the 2048-dimensional orthogonal complement is not identified"
        ),
    }


def inverse_sqrt(matrix: np.ndarray, ridge: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, ridge)
    return (vectors / np.sqrt(values)) @ vectors.T


def cross_validated_cca(
    a: np.ndarray, b: np.ndarray, task_ids: list[str], rank: int = 4
) -> dict[str, Any]:
    folds = [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    heldout_a: dict[int, list[float]] = defaultdict(list)
    heldout_b: dict[int, list[float]] = defaultdict(list)
    fold_rows = []
    for heldout in folds:
        train = [idx for idx in range(len(task_ids)) if idx not in heldout]
        atr = a[train]
        btr = b[train]
        mean_a = np.mean(atr, axis=0)
        mean_b = np.mean(btr, axis=0)
        ua, sa, vha = np.linalg.svd(atr - mean_a, full_matrices=False)
        ub, sb, vhb = np.linalg.svd(btr - mean_b, full_matrices=False)
        fit_rank = min(rank, len(train) - 1)
        xa = ua[:, :fit_rank] * sa[:fit_rank]
        xb = ub[:, :fit_rank] * sb[:fit_rank]
        scale = max(len(train) - 1, 1)
        ridge = 1e-6
        caa = xa.T @ xa / scale
        cbb = xb.T @ xb / scale
        cab = xa.T @ xb / scale
        wa = inverse_sqrt(caa, ridge)
        wb = inverse_sqrt(cbb, ridge)
        left, train_corr, right_t = np.linalg.svd(wa @ cab @ wb)
        coef_a = wa @ left
        coef_b = wb @ right_t.T
        scores_a = ((a[heldout] - mean_a) @ vha[:fit_rank].T) @ coef_a
        scores_b = ((b[heldout] - mean_b) @ vhb[:fit_rank].T) @ coef_b
        for component in range(fit_rank):
            heldout_a[component].extend(
                float(value) for value in scores_a[:, component]
            )
            heldout_b[component].extend(
                float(value) for value in scores_b[:, component]
            )
        fold_rows.append(
            {
                "held_out_task_ids": [task_ids[index] for index in heldout],
                "train_task_ids": [task_ids[index] for index in train],
                "fit_rank": fit_rank,
                "train_canonical_correlations": [
                    float(value) for value in np.clip(train_corr, 0.0, 1.0)
                ],
            }
        )
    evaluated = {}
    for component in range(rank):
        av = np.asarray(heldout_a[component], dtype=np.float64)
        bv = np.asarray(heldout_b[component], dtype=np.float64)
        if av.size < 3 or np.std(av) <= 1e-12 or np.std(bv) <= 1e-12:
            evaluated[str(component + 1)] = {
                "status": "UNAVAILABLE",
                "heldout_tasks": int(av.size),
            }
        else:
            evaluated[str(component + 1)] = {
                "status": "ESTIMABLE_BUT_SMALL_N_EXPLORATORY",
                "heldout_tasks": int(av.size),
                "heldout_pearson_correlation": float(np.corrcoef(av, bv)[0, 1]),
            }
    return {
        "method": (
            "four domain-balanced two-task holdout folds; train-only PCA and "
            "ridge-whitened CCA; each task evaluated exactly once"
        ),
        "folds": fold_rows,
        "heldout_canonical_correlations": evaluated,
        "requested_k_8": "UNAVAILABLE: six train tasks permit at most five centered components",
        "requested_k_16": "UNAVAILABLE: six train tasks permit at most five centered components",
        "limitation": (
            "only eight reference tasks exist; held-out canonical correlations "
            "are reconstruction diagnostics, not population estimates"
        ),
    }


def compare_adapters(
    a: np.ndarray,
    b: np.ndarray,
    da: dict[str, Any],
    db: dict[str, Any],
    task_ids: list[str],
) -> dict[str, Any]:
    row_cosines = [cosine(x, y) for x, y in zip(a, b)]
    subspaces = {
        str(k): subspace_comparison(da["basis"], db["basis"], k)
        for k in REQUESTED_K
    }
    projection_energy = {}
    for k in REQUESTED_K:
        if da["basis"].shape[0] < k or db["basis"].shape[0] < k:
            projection_energy[str(k)] = {
                "status": "UNAVAILABLE_INSUFFICIENT_EFFECTIVE_RANK"
            }
            continue
        basis_a = da["basis"][:k]
        basis_b = db["basis"][:k]
        projected_a = (da["centered"] @ basis_b.T) @ basis_b
        projected_b = (db["centered"] @ basis_a.T) @ basis_a
        projection_energy[str(k)] = {
            "status": "ESTIMABLE",
            "causal_energy_captured_by_sequence_subspace": float(
                np.sum(projected_a * projected_a)
                / max(np.sum(da["centered"] * da["centered"]), 1e-15)
            ),
            "sequence_energy_captured_by_causal_subspace": float(
                np.sum(projected_b * projected_b)
                / max(np.sum(db["centered"] * db["centered"]), 1e-15)
            ),
        }
    return {
        "pc1_signed_cosine": cosine(da["basis"][0], db["basis"][0]),
        "pc1_absolute_cosine": abs(cosine(da["basis"][0], db["basis"][0]) or 0.0),
        "mean_update_cosine": cosine(da["mean"], db["mean"]),
        "per_example_update_cosines": [
            {"task_id": task_id, "cosine": value}
            for task_id, value in zip(task_ids, row_cosines)
        ],
        "per_example_update_cosine_summary": summarize(
            [float(value) for value in row_cosines if value is not None]
        ),
        "principal_angles_and_projection_overlap": subspaces,
        "cross_projection_energy": projection_energy,
        "cross_validated_procrustes": cross_validated_procrustes(a, b, task_ids),
        "heldout_task_canonical_correlations": cross_validated_cca(
            a, b, task_ids
        ),
    }


def import_adapter_class(source_root: Path) -> Any:
    module_path = source_root / "src/evaluator/bg_causal_adapter.py"
    spec = importlib.util.spec_from_file_location("o1_diag_bg_adapter", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import adapter implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LowRankDeltaAdapter


def reproduce_proxy(
    source_root: Path,
    causal_path: Path,
    sequence_path: Path,
    output: Path,
    captured_a: np.ndarray,
    captured_b: np.ndarray,
    da: dict[str, Any],
    db: dict[str, Any],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("historical proxy reproduction requires the original CUDA path")
    seed = 20260518
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    adapter_class = import_adapter_class(source_root)
    sequence_payload = torch.load(
        sequence_path, map_location=device, weights_only=False
    )
    causal_payload = torch.load(causal_path, map_location=device, weights_only=False)
    sequence = adapter_class(rank=32).to(device)
    sequence.load_state_dict(sequence_payload["adapter_state_dict"], strict=True)
    causal = adapter_class(rank=32).to(device)
    causal.load_state_dict(causal_payload["adapter_state_dict"], strict=True)
    sequence.eval()
    causal.eval()
    probe = torch.randn(64, D_MODEL, device=device)
    with torch.no_grad():
        sequence_proxy = sequence(probe).detach().float().mean(dim=0).cpu().numpy()
        causal_proxy = causal(probe).detach().float().mean(dim=0).cpu().numpy()
    proxy_matrix = np.stack([causal_proxy, sequence_proxy]).astype("<f4")
    proxy_path = output / "historical_adapter_proxy_vectors.npy"
    np.save(proxy_path, proxy_matrix, allow_pickle=False)
    reproduced = cosine(causal_proxy, sequence_proxy)
    comparisons = {
        "causal_proxy_vs_causal_induced_pc1": cosine(
            causal_proxy, da["basis"][0]
        ),
        "sequence_proxy_vs_sequence_induced_pc1": cosine(
            sequence_proxy, db["basis"][0]
        ),
        "causal_proxy_vs_causal_induced_mean_update": cosine(
            causal_proxy, da["mean"]
        ),
        "sequence_proxy_vs_sequence_induced_mean_update": cosine(
            sequence_proxy, db["mean"]
        ),
        "causal_proxy_vs_sequence_induced_pc1": cosine(
            causal_proxy, db["basis"][0]
        ),
        "sequence_proxy_vs_causal_induced_pc1": cosine(
            sequence_proxy, da["basis"][0]
        ),
    }
    return {
        "status": (
            "REPRODUCED_FROM_CHECKPOINT_BINARIES"
            if reproduced is not None
            and abs(reproduced - HISTORICAL_PROXY_REPORTED) <= 1e-6
            else "NOT_REPRODUCED"
        ),
        "reported_historical_cosine_for_comparison_only": HISTORICAL_PROXY_REPORTED,
        "recomputed_cosine_from_checkpoint_binaries": reproduced,
        "absolute_difference": (
            abs(reproduced - HISTORICAL_PROXY_REPORTED)
            if reproduced is not None
            else None
        ),
        "tolerance": 1e-6,
        "seed": seed,
        "probe": {
            "distribution": "torch.randn standard Gaussian",
            "shape": [64, D_MODEL],
            "device": "cuda:0",
            "gpu": torch.cuda.get_device_name(0),
        },
        "construction": (
            "instantiate and load sequence adapter, instantiate and load causal "
            "adapter, evaluate both directly on the same 64 seeded Gaussian "
            "vectors, normalize inside each adapter per example, then mean"
        ),
        "proxy_vectors_path": str(proxy_path),
        "proxy_vectors_sha256": sha256_file(proxy_path),
        "proxy_vectors_shape": list(proxy_matrix.shape),
        "proxy_vectors_dtype": str(proxy_matrix.dtype),
        "differences_from_real_induced_update_capture": [
            "Gaussian hidden probes rather than frozen prompt residual states",
            "direct adapter outputs rather than model-induced downstream residual updates",
            "native layer-36 adapter output geometry rather than transported post-layer L3_24 geometry",
            "no alpha, loop scaling, recurrent propagation, or residual subtraction",
            "uncentered mean of 64 normalized adapter outputs rather than PC1 of eight centered task updates",
            "no task identities, prompt ordering, or leave-task-out support",
        ],
        "comparisons_to_captured_geometry": comparisons,
        "captured_mean_update_cosine": cosine(
            np.mean(captured_a, axis=0), np.mean(captured_b, axis=0)
        ),
        "captured_pc1_cosine": cosine(da["basis"][0], db["basis"][0]),
    }


def parity_report(
    worktree: Path,
    source_root: Path,
    report: dict[str, Any],
    reference: list[dict[str, Any]],
    causal_path: Path,
    sequence_path: Path,
    capture_report_path: Path,
    causal_updates_path: Path,
    sequence_updates_path: Path,
) -> dict[str, Any]:
    settings = report["settings"]
    diagnostics = report["task_diagnostics"]
    task_ids = [str(row["task_id"]) for row in reference]
    diagnostic_ids = [str(row["task_id"]) for row in diagnostics]
    expected_causal_scales = [0.0025, 0.005, 0.0075, 0.01]
    expected_sequence_scales = [0.005, 0.01, 0.015, 0.02]
    observed_checkpoint_tree = checkpoint_tree_hash(source_root)
    causal_code = source_root / "src/evaluator/bg_causal_adapter.py"
    sequence_code = source_root / "src/evaluator/bg_sequence_adapter.py"
    recorded_sources = report["source_hashes"]
    checks = {
        "causal_checkpoint_sha256": {
            "passed": sha256_file(causal_path) == CAUSAL_SHA256,
            "observed": sha256_file(causal_path),
            "expected": CAUSAL_SHA256,
        },
        "sequence_checkpoint_sha256": {
            "passed": sha256_file(sequence_path) == SEQUENCE_SHA256,
            "observed": sha256_file(sequence_path),
            "expected": SEQUENCE_SHA256,
        },
        "checkpoint_tree_binding": {
            "passed": (
                settings["model_tree_sha256"]
                == PRESERVED_CHECKPOINT_TREE_SHA256
                == observed_checkpoint_tree
            ),
            "recorded": settings["model_tree_sha256"],
            "recomputed_from_current_local_checkpoint": observed_checkpoint_tree,
            "expected": PRESERVED_CHECKPOINT_TREE_SHA256,
            "note": "tree was hash-bound in O1_REAL_001; the 5 GB checkpoint is not copied",
        },
        "adapter_and_hook_implementation_hashes": {
            "passed": (
                recorded_sources[str(causal_code)] == sha256_file(causal_code)
                and recorded_sources[str(sequence_code)] == sha256_file(sequence_code)
            ),
            "causal_adapter_code_sha256": sha256_file(causal_code),
            "sequence_hook_code_sha256": sha256_file(sequence_code),
        },
        "capture_report_sha256": {
            "passed": sha256_file(capture_report_path) == CAPTURE_REPORT_SHA256,
            "observed": sha256_file(capture_report_path),
            "expected": CAPTURE_REPORT_SHA256,
        },
        "capture_matrix_hashes": {
            "passed": (
                sha256_file(causal_updates_path) == CAUSAL_UPDATES_SHA256
                and sha256_file(sequence_updates_path) == SEQUENCE_UPDATES_SHA256
            ),
            "causal_observed": sha256_file(causal_updates_path),
            "sequence_observed": sha256_file(sequence_updates_path),
        },
        "adapter_application_and_scaling": {
            "passed": all(
                [entry["loop"] for entry in row["causal_capture"]["records"]]
                == [1, 2, 3, 4]
                and np.allclose(
                    [
                        entry["alpha_eff"]
                        for entry in row["causal_capture"]["records"]
                    ],
                    expected_causal_scales,
                    atol=0,
                    rtol=0,
                )
                and [entry["loop"] for entry in row["sequence_capture"]["records"]]
                == [1, 2, 3, 4]
                and np.allclose(
                    [
                        entry["alpha_eff"]
                        for entry in row["sequence_capture"]["records"]
                    ],
                    expected_sequence_scales,
                    atol=0,
                    rtol=0,
                )
                for row in diagnostics
            ),
            "intervention_target": "physical layer 36 post-layer hook",
            "mode": settings["intervention_mode"],
            "causal_effective_alpha_by_loop": expected_causal_scales,
            "sequence_effective_alpha_by_loop": expected_sequence_scales,
        },
        "paper_l3_loop_mapping": {
            "passed": settings["capture_loop_1indexed"] == 3,
            "zero_indexed_loop": settings["capture_loop_1indexed"] - 1,
            "paper_loop_label": "L3",
            "physical_layer": settings["capture_physical_layer"],
            "expected": "zero-indexed loop 2 / paper L3 / physical layer 24",
        },
        "post_layer_residual_boundary": {
            "passed": settings["capture_tensor"]
            == "post-block output of model.model.layers[23]",
            "observed": settings["capture_tensor"],
        },
        "final_nonpadding_prompt_position": {
            "passed": all(
                int(row["capture_position"]) == int(row["prompt_tokens"]) - 1
                for row in diagnostics
            )
            and settings["capture_position"]
            == "final non-padding prompt token",
            "positions": [
                {
                    "task_id": row["task_id"],
                    "prompt_tokens": row["prompt_tokens"],
                    "capture_position": row["capture_position"],
                }
                for row in diagnostics
            ],
        },
        "identical_reference_examples_and_order": {
            "passed": task_ids == diagnostic_ids and len(task_ids) == 8,
            "reference_sha256": sha256_file(
                worktree
                / "o1_runs/O1_REAL_001/axis_source_bundle_v1/D4_REFERENCE_SET.jsonl"
            ),
            "expected_reference_sha256": REFERENCE_SHA256,
            "task_ids": task_ids,
        },
        "dtype_and_normalization": {
            "passed": report["runtime"]["model_parameter_dtype"] == "torch.bfloat16"
            and report["saved_artifacts"]["causal_induced_updates"]["dtype"]
            == "float32"
            and report["saved_artifacts"]["sequence_induced_updates"]["dtype"]
            == "float32"
            and report["settings"]["component_normalization"] == "unit RMS"
            and report["runtime"]["transformers"] == "4.54.1"
            and __import__("transformers").__version__ == "4.54.1",
            "model_dtype": report["runtime"]["model_parameter_dtype"],
            "stored_update_dtype": "float32",
            "pc_compute_dtype": "float64",
            "component_normalization": report["settings"]["component_normalization"],
            "transformers_capture_version": report["runtime"]["transformers"],
            "transformers_current_version": __import__("transformers").__version__,
        },
        "sign_convention": {
            "passed": settings["pc_sign_rule"]
            == "positive dot with that adapter's mean induced update",
            "observed": settings["pc_sign_rule"],
            "gate_uses_absolute_cosine": True,
        },
        "trajectory_phase": {
            "passed": settings["trajectory_phase"] == "prompt_only",
            "observed": settings["trajectory_phase"],
            "generated_state_used": False,
        },
        "hook_and_zero_control": {
            "passed": all(
                row["baseline_capture"]["capture_layer_calls"] == 4
                and row["causal_capture"]["capture_layer_calls"] == 4
                and row["sequence_capture"]["capture_layer_calls"] == 4
                and row["causal_capture"]["hook_loop_index_source"] == "current_ut"
                and row["sequence_capture"]["hook_loop_index_source"] == "current_ut"
                and not row["causal_capture"]["nan_or_inf_activations"]
                and not row["sequence_capture"]["nan_or_inf_activations"]
                for row in diagnostics
            )
            and diagnostics[0]["zero_alpha_max_abs"] == 0.0,
            "zero_alpha_max_abs": diagnostics[0]["zero_alpha_max_abs"],
        },
    }
    failed = [name for name, item in checks.items() if not item["passed"]]
    return {
        "schema_version": "o1.axis_diagnosis_v2.capture_parity.v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "scientific_scope": "reconstruction diagnostic; no O1 outcomes",
        "source_root_read_only": str(source_root),
        "worktree": str(worktree),
        "base_commit": BASE_COMMIT,
        "checks": checks,
        "failed_checks": failed,
        "overall": "PASS" if not failed else "FAIL",
        "capture_or_reconstruction_defect_detected": bool(failed),
    }


def d2_task_means(
    features: np.ndarray, rows: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    grouped: dict[tuple[str, bool], list[np.ndarray]] = defaultdict(list)
    for feature, row in zip(features, rows):
        grouped[(str(row["task_id"]), bool(row["correct"]))].append(feature)
    positive = {
        task: np.mean(np.stack(values), axis=0)
        for (task, label), values in grouped.items()
        if label
    }
    negative = {
        task: np.mean(np.stack(values), axis=0)
        for (task, label), values in grouped.items()
        if not label
    }
    return positive, negative


def d2_report(
    worktree: Path,
    source_root: Path,
    exact_bundle_path: Path,
    d2_axis_path: Path,
) -> dict[str, Any]:
    require_hash(exact_bundle_path, EXACT_BUNDLE_SHA256)
    bundle = torch.load(exact_bundle_path, map_location="cpu", weights_only=False)
    rows = list(bundle["branch_rows"])
    features = (
        bundle["branch_features"][:, 0, 2, :]
        .detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )
    positive_counts = Counter(
        str(row["task_id"]) for row in rows if bool(row["correct"])
    )
    negative_counts = Counter(
        str(row["task_id"]) for row in rows if not bool(row["correct"])
    )
    positive, negative = d2_task_means(features, rows)
    frozen_axis = np.load(d2_axis_path, allow_pickle=False).astype(np.float64)
    global_diff = np.mean(np.stack(list(positive.values())), axis=0) - np.mean(
        np.stack(list(negative.values())), axis=0
    )

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED + 100))
    pos_names = sorted(positive)
    neg_names = sorted(negative)
    conditional_cosines = []
    for _ in range(BOOTSTRAP_DRAWS):
        pos_sample = rng.choice(pos_names, size=len(pos_names), replace=True)
        neg_sample = rng.choice(neg_names, size=len(neg_names), replace=True)
        diff = np.mean(np.stack([positive[name] for name in pos_sample]), axis=0)
        diff -= np.mean(np.stack([negative[name] for name in neg_sample]), axis=0)
        value = cosine(diff, frozen_axis)
        if value is not None:
            conditional_cosines.append(abs(value))

    loto = []
    all_tasks = sorted(set(pos_names) | set(neg_names))
    for task_id in all_tasks:
        remaining_pos = {k: v for k, v in positive.items() if k != task_id}
        remaining_neg = {k: v for k, v in negative.items() if k != task_id}
        if not remaining_pos or not remaining_neg:
            loto.append(
                {
                    "left_out_task_id": task_id,
                    "status": "UNDEFINED_ONE_VERIFIER_CLASS_ABSENT",
                    "remaining_positive_tasks": len(remaining_pos),
                    "remaining_negative_tasks": len(remaining_neg),
                }
            )
            continue
        diff = np.mean(np.stack(list(remaining_pos.values())), axis=0)
        diff -= np.mean(np.stack(list(remaining_neg.values())), axis=0)
        loto.append(
            {
                "left_out_task_id": task_id,
                "status": "ESTIMABLE_CONDITIONAL_ON_SOLE_POSITIVE_TASK",
                "remaining_positive_tasks": len(remaining_pos),
                "remaining_negative_tasks": len(remaining_neg),
                "direct_mean_difference_cosine_to_frozen_d2": cosine(
                    diff, frozen_axis
                ),
                "direct_mean_difference_cosine_to_full_direct_difference": cosine(
                    diff, global_diff
                ),
            }
        )

    xloop = (
        source_root
        / "artifacts/reports/cross_loop_early_layer_taps_20260720"
    )
    s3b2_path = xloop / "s3b2_features.pt"
    s3b2 = torch.load(s3b2_path, map_location="cpu", weights_only=False)
    items = list(s3b2["items"])
    s3_pos = Counter(str(row["task_id"]) for row in items if bool(row["correct"]))
    s3_neg = Counter(
        str(row["task_id"]) for row in items if not bool(row["correct"])
    )
    feature_manifest = xloop / "features/feature_manifest.json"
    mixed_manifest = json.loads(feature_manifest.read_text(encoding="utf-8"))
    mixed_group_count = sum(int(row["groups"]) for row in mixed_manifest["shards"])
    mixed_candidate_count = sum(
        int(row["candidates"]) for row in mixed_manifest["shards"]
    )
    empirical_directions = (
        source_root
        / "artifacts/reports/probes/bg_empirical_steering_direction_2026-05-18/directions.pt"
    )
    empirical_payload = torch.load(
        empirical_directions, map_location="cpu", weights_only=False
    )
    empirical_layers = sorted(
        {
            int(target["layer"])
            for target in empirical_payload.get("targets", [])
            if "layer" in target
        }
    )
    broader = [
        {
            "path": str(s3b2_path),
            "sha256": sha256_file(s3b2_path),
            "status": "FOUND_CANDIDATE_FOR_NEW_PREREGISTRATION",
            "tensor": "items[*].features[:, loop, :] with physical-layer index 2 and paper-L3 loop index 2",
            "shape_per_item": [5, 4, D_MODEL],
            "label": "items[*].correct; saved verifier correctness",
            "rows": len(items),
            "tasks": len(set(s3_pos) | set(s3_neg)),
            "positive_rows": int(sum(s3_pos.values())),
            "negative_rows": int(sum(s3_neg.values())),
            "positive_tasks": len(s3_pos),
            "negative_tasks": len(s3_neg),
            "positive_task_counts": dict(sorted(s3_pos.items())),
            "negative_task_counts": dict(sorted(s3_neg.items())),
            "caveat": (
                "small-N exploratory source; coding has zero positives; requires "
                "a new split/leakage and task-balance preregistration before axis use"
            ),
            "used_to_replace_current_d2": False,
        },
        {
            "path": str(feature_manifest),
            "sha256": sha256_file(feature_manifest),
            "status": "FOUND_BUT_NOT_UNIFORM_VERIFIER_OUTCOME_CORPUS",
            "groups": mixed_group_count,
            "candidates": mixed_candidate_count,
            "reason": (
                "contains heterogeneous canonical/mutant correctness, preference, "
                "and alignment rewards across domains; not one deterministic "
                "verifier-labelled outcome definition without a new audit"
            ),
            "used_to_replace_current_d2": False,
        },
        {
            "path": str(empirical_directions),
            "sha256": sha256_file(empirical_directions),
            "status": "EXCLUDED_WRONG_LOCUS",
            "physical_layers": empirical_layers,
            "reason": "empirical outcome directions are at physical layer 36, not L3_24",
            "used_to_replace_current_d2": False,
        },
    ]
    return {
        "schema_version": "o1.axis_diagnosis_v2.d2_adequacy.v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "scientific_scope": "frozen pre-O1 artifacts only; no O1 outcomes",
        "D2_ADEQUACY_VERDICT": "INSUFFICIENT_POSITIVE_TASK_SUPPORT",
        "current_source": {
            "path": str(exact_bundle_path),
            "sha256": sha256_file(exact_bundle_path),
            "feature_tensor": "branch_features[:,0,2,:]",
            "feature_shape": list(features.shape),
            "positive_rows": int(sum(positive_counts.values())),
            "negative_rows": int(sum(negative_counts.values())),
            "positive_tasks": len(positive_counts),
            "negative_tasks": len(negative_counts),
            "sole_positive_task": next(iter(positive_counts)),
            "positive_task_counts": dict(sorted(positive_counts.items())),
            "negative_task_counts": dict(sorted(negative_counts.items())),
            "frozen_d2_path": str(d2_axis_path),
            "frozen_d2_sha256": sha256_file(d2_axis_path),
            "direct_task_balanced_mean_difference_cosine_to_frozen_d2": cosine(
                global_diff, frozen_axis
            ),
        },
        "task_clustered_bootstrap": {
            "unit": "task_id",
            "draws": BOOTSTRAP_DRAWS,
            "positive_task_resampling": (
                "degenerate: the only positive task is selected in every draw"
            ),
            "negative_task_resampling": "four negative tasks sampled with replacement",
            "conditional_absolute_cosine_to_frozen_d2": summarize(
                conditional_cosines
            ),
            "general_positive-task_stability": "NOT_ESTIMABLE",
            "certifies_task_general_axis": False,
        },
        "leave_one_task_out": loto,
        "broader_source_search": broader,
        "current_vector_certified_as_general_outcome_axis": False,
        "blocker": (
            "one positive task cannot identify between-task positive outcome "
            "variation; deleting that task makes the estimator undefined"
        ),
    }


def choose_d4_verdict(
    parity: dict[str, Any],
    causal: dict[str, Any],
    sequence: dict[str, Any],
    comparison: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if parity["overall"] != "PASS":
        return "CAPTURE_OR_RECONSTRUCTION_DEFECT", {
            "reason": "one or more capture-parity checks failed",
            "failed_checks": parity["failed_checks"],
        }
    c_boot = causal["task_clustered_bootstrap"]["pc1_absolute_cosine_to_full"]
    s_boot = sequence["task_clustered_bootstrap"]["pc1_absolute_cosine_to_full"]
    c_loto = causal["leave_one_task_out"]["pc1_absolute_cosine_summary"]
    s_loto = sequence["leave_one_task_out"]["pc1_absolute_cosine_summary"]
    stable_c = (
        c_boot.get("q05", 0.0) >= 0.50
        and c_boot.get("median", 0.0) >= 0.80
        and c_loto.get("minimum", 0.0) >= 0.50
    )
    stable_s = (
        s_boot.get("q05", 0.0) >= 0.50
        and s_boot.get("median", 0.0) >= 0.80
        and s_loto.get("minimum", 0.0) >= 0.50
    )
    c_mean_boot = causal["task_clustered_bootstrap"]["mean_update_cosine_to_full"]
    s_mean_boot = sequence["task_clustered_bootstrap"]["mean_update_cosine_to_full"]
    c_mean_loto = causal["leave_one_task_out"]["mean_update_cosine_summary"]
    s_mean_loto = sequence["leave_one_task_out"]["mean_update_cosine_summary"]
    stable_c_mean = (
        c_mean_boot.get("q05", 0.0) >= 0.60
        and c_mean_boot.get("median", 0.0) >= 0.75
        and c_mean_loto.get("minimum", 0.0) >= 0.90
    )
    stable_s_mean = (
        s_mean_boot.get("q05", 0.0) >= 0.60
        and s_mean_boot.get("median", 0.0) >= 0.75
        and s_mean_loto.get("minimum", 0.0) >= 0.90
    )
    shared_pc1 = stable_c and stable_s and comparison["pc1_absolute_cosine"] >= 0.90
    top4 = comparison["principal_angles_and_projection_overlap"]["4"]
    procrustes = comparison["cross_validated_procrustes"]
    shared_higher = (
        top4["status"] == "ESTIMABLE"
        and top4["projection_overlap_mean_cosine_squared"] >= 0.50
        and procrustes["a_to_b_cosine_summary"].get("median", -1.0) >= 0.50
        and procrustes["b_to_a_cosine_summary"].get("median", -1.0) >= 0.50
    )
    criteria = {
        "diagnostic_rule_not_a_change_to_frozen_d4_gate": True,
        "stable_pc1_rule": (
            "task-bootstrap q05>=0.50, median>=0.80, and LOTO minimum>=0.50"
        ),
        "stable_mean_update_rule": (
            "task-bootstrap q05>=0.60, median>=0.75, and LOTO minimum>=0.90"
        ),
        "shared_pc1_rule": "both PC1s stable and cross-adapter |cosine|>=0.90",
        "shared_higher_subspace_rule": (
            "top-4 projection overlap>=0.50 and bidirectional held-out "
            "Procrustes median cosine>=0.50"
        ),
        "causal_pc1_stable": stable_c,
        "sequence_pc1_stable": stable_s,
        "causal_mean_update_stable": stable_c_mean,
        "sequence_mean_update_stable": stable_s_mean,
        "shared_pc1": shared_pc1,
        "shared_higher_subspace": shared_higher,
    }
    if shared_pc1:
        return "STABLE_SHARED_ONE_DIMENSIONAL_AXIS", criteria
    if shared_higher:
        return "NO_STABLE_PC1_BUT_SHARED_HIGHER_DIMENSIONAL_SUBSPACE", criteria
    if (stable_c or stable_c_mean) and (stable_s or stable_s_mean):
        return "STABLE_BUT_DISTINCT_ADAPTER_GEOMETRIES", criteria
    return "NO_STABLE_ACTUATOR_GEOMETRY", criteria


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    worktree = Path(__file__).resolve().parents[1]
    source_root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnosis output: {output}")
    output.mkdir(parents=True)

    preserved = worktree / "o1_runs/O1_REAL_001"
    capture_root = (
        preserved / "axis_source_bundle_v1/d4_capture_v1_1"
    )
    capture_report_path = capture_root / "D4_CAPTURE_REPORT.json"
    causal_updates_path = capture_root / "d4_adapter1_induced_updates_l3_24.npy"
    sequence_updates_path = (
        capture_root / "d4_adapter2_induced_updates_l3_24.npy"
    )
    reference_path = (
        preserved / "axis_source_bundle_v1/D4_REFERENCE_SET.jsonl"
    )
    d2_axis_path = preserved / "axis_source_bundle_v1/d2_l3_24.npy"
    causal_path = (
        source_root
        / "artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/"
        "adapter_checkpoints/best_adapter.pt"
    )
    sequence_path = (
        source_root
        / "artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/"
        "sequence_adapter_checkpoints/best_sequence_adapter.pt"
    )
    exact_bundle_path = (
        source_root
        / "artifacts/reports/proto_introspection/"
        "s1_s3_exact_injection_orthogonality_2026-06-17/"
        "s1_s3_exact_injection_delta_bundle_2026-06-17.pt"
    )
    require_hash(causal_path, CAUSAL_SHA256)
    require_hash(sequence_path, SEQUENCE_SHA256)
    require_hash(capture_report_path, CAPTURE_REPORT_SHA256)
    require_hash(causal_updates_path, CAUSAL_UPDATES_SHA256)
    require_hash(sequence_updates_path, SEQUENCE_UPDATES_SHA256)
    require_hash(reference_path, REFERENCE_SHA256)
    require_hash(exact_bundle_path, EXACT_BUNDLE_SHA256)

    capture_report = json.loads(capture_report_path.read_text(encoding="utf-8"))
    reference = [
        json.loads(line)
        for line in reference_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_ids = [str(row["task_id"]) for row in reference]
    causal_updates = np.load(causal_updates_path, allow_pickle=False).astype(
        np.float64
    )
    sequence_updates = np.load(
        sequence_updates_path, allow_pickle=False
    ).astype(np.float64)

    parity = parity_report(
        worktree,
        source_root,
        capture_report,
        reference,
        causal_path,
        sequence_path,
        capture_report_path,
        causal_updates_path,
        sequence_updates_path,
    )
    write_json(output / "D4_CAPTURE_PARITY_REPORT.json", parity)

    causal_report, causal_decomp = adapter_report(
        "causal_teacher_forced",
        causal_updates,
        task_ids,
        BOOTSTRAP_SEED + 1,
    )
    sequence_report, sequence_decomp = adapter_report(
        "sequence_level",
        sequence_updates,
        task_ids,
        BOOTSTRAP_SEED + 2,
    )
    comparison = compare_adapters(
        causal_updates,
        sequence_updates,
        causal_decomp,
        sequence_decomp,
        task_ids,
    )
    geometry = {
        "schema_version": "o1.axis_diagnosis_v2.d4_geometry.v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "scientific_scope": "real captured pre-O1 updates; reconstruction diagnostics only",
        "reference_task_count": len(task_ids),
        "reference_task_ids": task_ids,
        "support_limit": (
            "eight independent task clusters imply centered rank at most seven; "
            "PC/subspace requests k=8 and k=16 are not mathematically estimable"
        ),
        "causal_adapter": causal_report,
        "sequence_adapter": sequence_report,
        "cross_adapter": comparison,
    }
    write_json(output / "D4_GEOMETRY_DIAGNOSTICS.json", geometry)

    proxy = reproduce_proxy(
        source_root,
        causal_path,
        sequence_path,
        output,
        causal_updates,
        sequence_updates,
        causal_decomp,
        sequence_decomp,
    )
    proxy.update(
        {
            "schema_version": "o1.axis_diagnosis_v2.historical_proxy.v1",
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "causal_checkpoint": str(causal_path),
            "causal_checkpoint_sha256": sha256_file(causal_path),
            "sequence_checkpoint": str(sequence_path),
            "sequence_checkpoint_sha256": sha256_file(sequence_path),
            "historical_diagnostics_source": str(
                source_root
                / "artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/diagnostics.json"
            ),
            "historical_diagnostics_source_sha256": sha256_file(
                source_root
                / "artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/diagnostics.json"
            ),
        }
    )
    write_json(output / "HISTORICAL_PROXY_REPRODUCTION.json", proxy)

    d2 = d2_report(worktree, source_root, exact_bundle_path, d2_axis_path)
    write_json(output / "D2_ADEQUACY_REPORT.json", d2)

    d4_verdict, decision = choose_d4_verdict(
        parity, causal_report, sequence_report, comparison
    )
    final = {
        "schema_version": "o1.axis_diagnosis_v2.final.v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "base_commit": BASE_COMMIT,
        "preserved_snapshot_commit_in_worktree": "31ef40b",
        "scientific_scope": (
            "bounded axis diagnosis only; no replacement axes, bank changes, "
            "calibration precommit, calibration, or confirmation"
        ),
        "D4_VERDICT": d4_verdict,
        "D4_DECISION_DIAGNOSTICS": decision,
        "D2_ADEQUACY_VERDICT": d2["D2_ADEQUACY_VERDICT"],
        "historical_proxy_reproduction": proxy["status"],
        "historical_proxy_recomputed_cosine": proxy[
            "recomputed_cosine_from_checkpoint_binaries"
        ],
        "real_measurements": [
            "two real 8x2048 adapter-induced L3_24 update matrices",
            "real checkpoint-binary proxy recomputation on the RTX 5070 Ti",
            "frozen d2 source labels/task IDs and broader-source metadata",
        ],
        "reconstruction_diagnostics": [
            "SVD/explained variance",
            "task-clustered bootstrap",
            "leave-one-task-out stability",
            "principal angles and projection overlaps",
            "held-out Procrustes and canonical-correlation diagnostics",
        ],
        "frozen_d4_gate": {
            "required_absolute_cosine": 0.90,
            "observed_absolute_cosine": comparison["pc1_absolute_cosine"],
            "passed": False,
            "altered_retroactively": False,
        },
        "calibration_started": False,
        "confirmatory_generation_started": False,
        "replacement_d4_emitted": False,
        "replacement_d2_emitted": False,
        "recommended_redesign": (
            "write a new preregistration that first expands the frozen d4 "
            "reference to enough independent tasks to estimate the requested "
            "subspaces, treats adapters as related because sequence training "
            "initialized from the causal checkpoint, and independently "
            "preregisters a d2 source using task-diverse verifier-labelled "
            "L3_24 states such as the found 16-task s3b2 artifact only after "
            "split/leakage and domain-support audits"
        ),
        "exact_blockers": [
            "d4 real induced-update PC1 absolute cosine remains below 0.90",
            "d4 has only eight task clusters, so centered k=8 and k=16 geometry is unidentifiable",
            "current d2 has one positive task and cannot support a task-general outcome axis",
        ],
    }
    write_json(output / "AXIS_DIAGNOSIS_V2_REPORT.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
