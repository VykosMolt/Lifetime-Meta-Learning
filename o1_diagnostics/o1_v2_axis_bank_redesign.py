#!/usr/bin/env python3
"""Build diagnostics and frozen source inputs for O1_V2_AXIS_BANK_REDESIGN.

This program consumes only preserved, pre-O1 material. It constructs no
calibration records and performs no model generation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


D_MODEL = 2048
REFERENCE_TASKS = 8
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026072902
S3B2_BOOTSTRAP_SEED = 2026072903

EXPECTED = {
    "a1": "2ef03d84304f838bcf749132007b4481010f40e7ba3db5dd59e586abb84f3584",
    "a2": "900d56f02b2f0a82060aad8519950cd0b07f70f1320b714890ab7cee6c06211d",
    "causal_updates": "6388d7fc1477311ab6f75c02736451e60be1a925f5f4a46c0edba2eb23adfc92",
    "sequence_updates": "484cfd9adf8b4c6bb11a80fa6aab149711c52af7ce9bc25a06a7fb0979c87775",
    "reference": "d145955fdd15a7c413fae26276d4a61a0d4eab05f5241b73365982e4ed49c75b",
    "capture_report": "56e525c99f98fb1ab5f6aff0e99d99f50b8b49643d96c4ea9a134e2d0e80e2e3",
    "capture_parity": "d77f409738981ee134c482d0162417fecf5380f184b467f83fa64fb298f58d17",
    "causal_checkpoint": "f0831a9abf846cdf702d1211da346e2de3d32fe2ce11275247bad8400af215f4",
    "sequence_checkpoint": "75140bf3af7dfba4ee6851a500693b019f884e25dad0a2112f493dc99630597c",
    "checkpoint_tree": "7dbbe4ef91de369bd712bad587c1af4f7c5dc42deff5377d4d70086c4c08a802",
    "s3b2": "0bf994770ee6e14fa834680d5f621211a2758de92e55d0dc45e5db0926d3f34d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"frozen source hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_row_hash(vector: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(vector, dtype="<f8").tobytes()
    ).hexdigest()


def rms(vector: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(vector, dtype=np.float64))))


def unit_rms(vector: np.ndarray) -> np.ndarray:
    value = rms(vector)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("cannot normalize zero/non-finite vector")
    return np.asarray(vector, dtype=np.float64) / value


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return None
    return float(np.dot(left, right) / denominator)


def summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray([value for value in values if math.isfinite(value)])
    if not array.size:
        return {"n": 0, "status": "UNAVAILABLE"}
    return {
        "n": int(array.size),
        "minimum": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_reference(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != REFERENCE_TASKS:
        raise ValueError("frozen actuator reference must contain eight tasks")
    if [row["reference_rank"] for row in rows] != list(range(REFERENCE_TASKS)):
        raise ValueError("frozen actuator reference ordering changed")
    if len({row["task_id"] for row in rows}) != REFERENCE_TASKS:
        raise ValueError("frozen actuator task IDs are not unique")
    return rows


def mean_stability(
    matrix: np.ndarray, task_ids: list[str], seed: int
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    full = np.mean(matrix, axis=0)
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, len(task_ids), size=len(task_ids))
        value = cosine(np.mean(matrix[indices], axis=0), full)
        if value is not None:
            bootstrap.append(value)
    loto = []
    for index, task_id in enumerate(task_ids):
        value = cosine(np.mean(np.delete(matrix, index, axis=0), axis=0), full)
        loto.append(
            {
                "left_out_task_id": task_id,
                "cosine_to_full_mean": value,
            }
        )
    return {
        "bootstrap": {
            "unit": "task_id; one frozen reference example per task",
            "draws": BOOTSTRAP_DRAWS,
            "seed": seed,
            "bit_generator": "numpy.PCG64",
            "cosine_to_full_mean": summary(bootstrap),
        },
        "leave_one_task_out": {
            "rows": loto,
            "cosine_to_full_mean": summary(
                [
                    float(row["cosine_to_full_mean"])
                    for row in loto
                    if row["cosine_to_full_mean"] is not None
                ]
            ),
        },
    }


def actuator_diagnostics(
    causal: np.ndarray,
    sequence: np.ndarray,
    references: list[dict[str, Any]],
    parity: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    task_ids = [str(row["task_id"]) for row in references]
    causal64 = causal.astype(np.float64)
    sequence64 = sequence.astype(np.float64)
    causal_mean = np.mean(causal64, axis=0)
    sequence_mean = np.mean(sequence64, axis=0)
    a3 = unit_rms(causal_mean)
    a4 = unit_rms(sequence_mean)
    per_example = [
        {
            "reference_rank": index,
            "task_id": task_id,
            "cosine": cosine(causal64[index], sequence64[index]),
        }
        for index, task_id in enumerate(task_ids)
    ]
    return {
        "schema_version": "o1.v2.actuator_mean_diagnostics.v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "scientific_scope": (
            "real pre-O1 induced-update reconstruction diagnostics; no O1 "
            "calibration or confirmatory outcomes"
        ),
        "construction": {
            "causal": "unitRMS(mean(float64(exact float32 causal matrix), axis=0))",
            "sequence": (
                "unitRMS(mean(float64(exact float32 sequence matrix), axis=0))"
            ),
            "centering": False,
            "pc_extraction": False,
            "cross_adapter_averaging": False,
            "reference_conditioned": True,
        },
        "reference": {
            "file_sha256": sha256_file(source_paths["reference"]),
            "task_count": len(task_ids),
            "task_ids_in_frozen_order": task_ids,
            "unchanged_from_failed_capture": True,
        },
        "capture_parity": {
            "report_sha256": sha256_file(source_paths["capture_parity"]),
            "overall": parity["overall"],
            "failed_checks": parity["failed_checks"],
            "exact_checkpoint_and_adapter_hashes": parity["checks"][
                "checkpoint_tree_binding"
            ],
            "causal_checkpoint": parity["checks"]["causal_checkpoint_sha256"],
            "sequence_checkpoint": parity["checks"]["sequence_checkpoint_sha256"],
            "application_and_scaling": parity["checks"][
                "adapter_application_and_scaling"
            ],
            "paper_l3_loop_mapping": parity["checks"]["paper_l3_loop_mapping"],
            "post_layer_boundary": parity["checks"][
                "post_layer_residual_boundary"
            ],
            "final_prompt_position": parity["checks"][
                "final_nonpadding_prompt_position"
            ],
            "reference_identity_and_order": parity["checks"][
                "identical_reference_examples_and_order"
            ],
            "dtype_and_normalization": parity["checks"][
                "dtype_and_normalization"
            ],
            "sign_convention": parity["checks"]["sign_convention"],
            "trajectory_phase": parity["checks"]["trajectory_phase"],
        },
        "A3_CAUSAL_MEAN": {
            "source_matrix_path": str(source_paths["causal_updates"]),
            "source_matrix_file_sha256": sha256_file(
                source_paths["causal_updates"]
            ),
            "source_matrix_shape": list(causal.shape),
            "source_matrix_dtype": str(causal.dtype),
            "raw_mean_l2_norm": float(np.linalg.norm(causal_mean)),
            "raw_mean_rms": rms(causal_mean),
            "unit_rms_row_sha256": canonical_row_hash(a3),
            "stability": mean_stability(
                causal64, task_ids, BOOTSTRAP_SEED + 1
            ),
            "adapter_checkpoint_sha256": EXPECTED["causal_checkpoint"],
        },
        "A4_SEQUENCE_MEAN": {
            "source_matrix_path": str(source_paths["sequence_updates"]),
            "source_matrix_file_sha256": sha256_file(
                source_paths["sequence_updates"]
            ),
            "source_matrix_shape": list(sequence.shape),
            "source_matrix_dtype": str(sequence.dtype),
            "raw_mean_l2_norm": float(np.linalg.norm(sequence_mean)),
            "raw_mean_rms": rms(sequence_mean),
            "unit_rms_row_sha256": canonical_row_hash(a4),
            "stability": mean_stability(
                sequence64, task_ids, BOOTSTRAP_SEED + 2
            ),
            "adapter_checkpoint_sha256": EXPECTED["sequence_checkpoint"],
        },
        "cross_adapter": {
            "mean_direction_cosine": cosine(causal_mean, sequence_mean),
            "interpretation": "moderate alignment; axes remain separate",
            "per_example_update_cosines": per_example,
            "per_example_update_cosine_summary": summary(
                [float(row["cosine"]) for row in per_example]
            ),
        },
    }


def _task_label_means(
    features: np.ndarray, items: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    grouped: dict[tuple[str, bool], list[np.ndarray]] = defaultdict(list)
    for feature, item in zip(features, items):
        grouped[(str(item["task_id"]), bool(item["correct"]))].append(feature)
    positive = {
        task_id: np.mean(np.stack(rows), axis=0)
        for (task_id, label), rows in grouped.items()
        if label
    }
    negative = {
        task_id: np.mean(np.stack(rows), axis=0)
        for (task_id, label), rows in grouped.items()
        if not label
    }
    return positive, negative


def _task_balanced_direction(
    positive: dict[str, np.ndarray], negative: dict[str, np.ndarray]
) -> np.ndarray | None:
    if not positive or not negative:
        return None
    return np.mean(np.stack(list(positive.values())), axis=0) - np.mean(
        np.stack(list(negative.values())), axis=0
    )


def _d2_cluster_bootstrap(
    positive: dict[str, np.ndarray],
    negative: dict[str, np.ndarray],
    full: np.ndarray,
) -> dict[str, Any]:
    tasks = sorted(set(positive) | set(negative))
    rng = np.random.Generator(np.random.PCG64(S3B2_BOOTSTRAP_SEED))
    signed: list[float] = []
    invalid = 0
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = list(rng.choice(tasks, size=len(tasks), replace=True))
        pos_rows = [positive[task] for task in sampled if task in positive]
        neg_rows = [negative[task] for task in sampled if task in negative]
        if not pos_rows or not neg_rows:
            invalid += 1
            continue
        direction = np.mean(np.stack(pos_rows), axis=0) - np.mean(
            np.stack(neg_rows), axis=0
        )
        value = cosine(direction, full)
        if value is not None:
            signed.append(value)
    return {
        "unit": (
            "task_id cluster; a sampled task carries all of its positive and/or "
            "negative branch rows together"
        ),
        "draws": BOOTSTRAP_DRAWS,
        "seed": S3B2_BOOTSTRAP_SEED,
        "bit_generator": "numpy.PCG64",
        "valid_draws": len(signed),
        "invalid_draws_missing_class": invalid,
        "signed_cosine_to_full_direction": summary(signed),
        "absolute_cosine_to_full_direction": summary([abs(value) for value in signed]),
    }


def _cohort_leakage(
    task_ids: set[str],
    future_cohorts: list[Path],
) -> dict[str, Any]:
    if not future_cohorts:
        return {
            "status": "NOT_TESTABLE_FUTURE_COHORTS_NOT_YET_CONSTRUCTED",
            "checked_manifests": [],
            "task_id_overlap": [],
            "content_hash_overlap": "NOT_TESTABLE_SOURCE_HAS_NO_O1_CANONICAL_TASK_HASH",
            "primary_bank_impact": "NONE; D2_OUTCOME_AXIS_V2 is excluded from primary bank",
            "required_future_gate": (
                "exclude all 16 source task IDs and prove canonical task-content "
                "hash disjointness before any secondary outcome-axis experiment"
            ),
        }
    checked = []
    overlap: set[str] = set()
    for path in future_cohorts:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids = {str(row["task_id"]) for row in rows}
        overlap |= task_ids & ids
        checked.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
                "task_id_overlap": sorted(task_ids & ids),
            }
        )
    return {
        "status": "PASS_TASK_ID_DISJOINT" if not overlap else "FAIL_TASK_ID_OVERLAP",
        "checked_manifests": checked,
        "task_id_overlap": sorted(overlap),
        "content_hash_overlap": (
            "NOT_DIRECTLY_COMPARABLE: s3b2 stores generated branch text and task "
            "IDs, not Horizon canonical task JSON hashes"
        ),
        "primary_bank_impact": "NONE; D2_OUTCOME_AXIS_V2 is excluded from primary bank",
        "required_future_gate": (
            "retain task-ID exclusion and audit original prompt/spec hashes before "
            "any secondary outcome-axis experiment"
        ),
    }


def d2_outcome_diagnosis(
    s3b2_path: Path,
    future_cohorts: list[Path],
) -> tuple[dict[str, Any], np.ndarray]:
    require_hash(s3b2_path, EXPECTED["s3b2"])
    payload = torch.load(s3b2_path, map_location="cpu", weights_only=False)
    items = list(payload["items"])
    if payload.get("layout") != "(5,4,2048) layers[8,16,24,36,47] x loops[1..4]":
        raise RuntimeError("unexpected s3b2 tensor layout")
    features = np.stack(
        [
            item["features"][2, 2, :]
            .detach()
            .cpu()
            .to(torch.float64)
            .numpy()
            for item in items
        ]
    )
    positive, negative = _task_label_means(features, items)
    full = _task_balanced_direction(positive, negative)
    if full is None:
        raise RuntimeError("s3b2 has no estimable outcome direction")
    axis = unit_rms(full)
    task_ids = sorted(set(positive) | set(negative))
    domain_by_task: dict[str, str] = {}
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        task_id = str(item["task_id"])
        domain = str(item["domain"])
        if task_id in domain_by_task and domain_by_task[task_id] != domain:
            raise RuntimeError(f"{task_id}: inconsistent domains")
        domain_by_task[task_id] = domain
        rows_by_task[task_id].append(item)

    positive_rows = Counter(
        str(item["task_id"]) for item in items if bool(item["correct"])
    )
    negative_rows = Counter(
        str(item["task_id"]) for item in items if not bool(item["correct"])
    )
    domains = sorted(set(domain_by_task.values()))
    counts_by_domain = {}
    for domain in domains:
        domain_tasks = {
            task for task, value in domain_by_task.items() if value == domain
        }
        counts_by_domain[domain] = {
            "tasks": len(domain_tasks),
            "positive_rows": sum(positive_rows[task] for task in domain_tasks),
            "negative_rows": sum(negative_rows[task] for task in domain_tasks),
            "positive_tasks": sum(task in positive for task in domain_tasks),
            "negative_tasks": sum(task in negative for task in domain_tasks),
            "task_ids": sorted(domain_tasks),
        }

    leave_positive = []
    for task_id in sorted(positive):
        remaining_positive = {
            task: vector for task, vector in positive.items() if task != task_id
        }
        direction = _task_balanced_direction(remaining_positive, negative)
        leave_positive.append(
            {
                "left_out_positive_task_id": task_id,
                "domain": domain_by_task[task_id],
                "remaining_positive_tasks": len(remaining_positive),
                "cosine_to_full": cosine(direction, full) if direction is not None else None,
            }
        )

    domain_held_out = []
    for domain in domains:
        retained = {
            task for task in task_ids if domain_by_task[task] != domain
        }
        retained_positive = {
            task: vector for task, vector in positive.items() if task in retained
        }
        retained_negative = {
            task: vector for task, vector in negative.items() if task in retained
        }
        direction = _task_balanced_direction(retained_positive, retained_negative)
        domain_held_out.append(
            {
                "left_out_domain": domain,
                "remaining_positive_tasks": len(retained_positive),
                "remaining_negative_tasks": len(retained_negative),
                "status": "ESTIMABLE" if direction is not None else "UNAVAILABLE_CLASS_ABSENT",
                "cosine_to_full": cosine(direction, full) if direction is not None else None,
            }
        )

    contribution = []
    for task_id in task_ids:
        pos_term = (
            positive[task_id] / len(positive)
            if task_id in positive
            else np.zeros(D_MODEL)
        )
        neg_term = (
            -negative[task_id] / len(negative)
            if task_id in negative
            else np.zeros(D_MODEL)
        )
        vector = pos_term + neg_term
        remaining_positive = {
            task: value for task, value in positive.items() if task != task_id
        }
        remaining_negative = {
            task: value for task, value in negative.items() if task != task_id
        }
        loto_direction = _task_balanced_direction(
            remaining_positive, remaining_negative
        )
        contribution.append(
            {
                "task_id": task_id,
                "domain": domain_by_task[task_id],
                "positive_rows": positive_rows[task_id],
                "negative_rows": negative_rows[task_id],
                "signed_estimator_coefficient_positive": (
                    1.0 / len(positive) if task_id in positive else 0.0
                ),
                "signed_estimator_coefficient_negative": (
                    -1.0 / len(negative) if task_id in negative else 0.0
                ),
                "contribution_l2_norm": float(np.linalg.norm(vector)),
                "contribution_cosine_with_full": cosine(vector, full),
                "leave_one_task_out_cosine_to_full": (
                    cosine(loto_direction, full)
                    if loto_direction is not None
                    else None
                ),
            }
        )

    source_items = []
    for task_id in task_ids:
        canonical_rows = [
            {
                "branch_idx": int(item["branch_idx"]),
                "correct": bool(item["correct"]),
                "domain": str(item["domain"]),
                "provenance": str(item["provenance"]),
                "text": str(item["text"]),
            }
            for item in sorted(
                rows_by_task[task_id], key=lambda row: int(row["branch_idx"])
            )
        ]
        source_items.append(
            {
                "task_id": task_id,
                "domain": domain_by_task[task_id],
                "artifact_source_item_surrogate_sha256": canonical_json_hash(
                    {"task_id": task_id, "rows": canonical_rows}
                ),
                "note": (
                    "hash binds task ID and stored generated branches; it is not "
                    "an O1 canonical Horizon task-spec hash"
                ),
            }
        )

    bootstrap = _d2_cluster_bootstrap(positive, negative, full)
    loto_values = [
        float(row["cosine_to_full"])
        for row in leave_positive
        if row["cosine_to_full"] is not None
    ]
    domain_values = [
        float(row["cosine_to_full"])
        for row in domain_held_out
        if row["cosine_to_full"] is not None
    ]
    noncoding_domains = [domain for domain in domains if domain != "coding"]
    positive_domain_support = all(
        counts_by_domain[domain]["positive_tasks"] >= 2
        for domain in noncoding_domains
    )
    stability_rule = {
        "bootstrap_signed_cosine_q05_minimum": 0.70,
        "leave_one_positive_task_out_minimum": 0.80,
        "domain_held_out_minimum": 0.70,
        "note": (
            "diagnostic rule fixed in code for this bounded audit; it does not "
            "authorize insertion into the v2 primary bank"
        ),
    }
    stability_pass = (
        bootstrap["signed_cosine_to_full_direction"]["q05"] >= 0.70
        and min(loto_values) >= 0.80
        and min(domain_values) >= 0.70
    )
    leakage = _cohort_leakage(set(task_ids), future_cohorts)
    if leakage["status"] == "FAIL_TASK_ID_OVERLAP":
        verdict = "LEAKAGE_OR_SPLIT_FAILURE"
    elif not positive_domain_support:
        verdict = "INSUFFICIENT_DOMAIN_SUPPORT"
    elif not stability_pass:
        verdict = "INSUFFICIENT_STABILITY"
    else:
        # No Horizon Logic examples occur in the source, so even a stable result
        # is necessarily restricted and cannot be target-domain adequate.
        verdict = "ADEQUATE_RESTRICTED_NONCODING_AXIS"

    report = {
        "schema_version": "o1.d2_outcome_axis_v2.diagnosis.v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "scientific_scope": (
            "real frozen pre-O1 verifier-labelled artifact; reconstruction "
            "diagnostics only; excluded from O1 v2 primary bank"
        ),
        "source": {
            "path": str(s3b2_path),
            "sha256": sha256_file(s3b2_path),
            "layout": payload["layout"],
            "selected_tensor": "items[*].features[layer_slot_24=2, loop_slot_L3=2, :]",
            "shape": list(features.shape),
            "dtype_after_load": str(features.dtype),
            "rows": len(items),
            "tasks": len(task_ids),
        },
        "D2_OUTCOME_AXIS_V2_VERDICT": verdict,
        "primary_bank_modified": False,
        "outcome_axis": {
            "construction": (
                "unitRMS(mean of per-task positive branch means minus mean of "
                "per-task negative branch means)"
            ),
            "canonical_row_sha256": canonical_row_hash(axis),
            "raw_difference_l2_norm": float(np.linalg.norm(full)),
            "raw_difference_rms": rms(full),
            "shape": list(axis.shape),
            "dtype": str(axis.dtype),
        },
        "labels": {
            "positive_rows": sum(positive_rows.values()),
            "negative_rows": sum(negative_rows.values()),
            "positive_tasks": len(positive),
            "negative_tasks": len(negative),
            "positive_task_ids_and_rows": dict(sorted(positive_rows.items())),
            "negative_task_ids_and_rows": dict(sorted(negative_rows.items())),
            "by_domain": counts_by_domain,
            "horizon_logic_or_target_domain_positive_rows": 0,
            "horizon_logic_or_target_domain_positive_tasks": 0,
            "target_domain_reason": (
                "all source domains are coding/logic/math/reasoning pre-O1 canaries; "
                "none are generated by the frozen Horizon Logic O1 task generator"
            ),
        },
        "source_item_and_task_disjointness": {
            "unique_task_ids": len(task_ids),
            "one_domain_per_task": True,
            "branch_indices_0_through_9_once_per_task": all(
                sorted(int(row["branch_idx"]) for row in rows_by_task[task])
                == list(range(10))
                for task in task_ids
            ),
            "positive_negative_task_sets_disjoint": not bool(
                set(positive) & set(negative)
            ),
            "tasks_with_both_outcomes": sorted(set(positive) & set(negative)),
            "interpretation": (
                "branch outcome classes deliberately share eight source tasks; "
                "task-balanced estimation clusters all branches by task. There is "
                "no independent fit/evaluation split in this 16-task artifact."
            ),
            "source_item_surrogate_hashes": source_items,
        },
        "task_clustered_bootstrap": bootstrap,
        "leave_one_positive_task_out": {
            "rows": leave_positive,
            "cosine_to_full_summary": summary(loto_values),
        },
        "domain_held_out": {
            "rows": domain_held_out,
            "cosine_to_full_summary": summary(domain_values),
        },
        "task_contributions": contribution,
        "leakage_against_future_o1_cohorts": leakage,
        "adequacy_rule": {
            "positive_domain_support": (
                "at least two positive tasks in each identifiable noncoding "
                "source domain (logic, math, reasoning)"
            ),
            "positive_domain_support_pass": positive_domain_support,
            "stability": stability_rule,
            "stability_pass": stability_pass,
            "target_domain_adequacy_requires_horizon_logic_support": True,
        },
        "restriction": (
            "D2_OUTCOME_AXIS_V2 cannot enter the O1 v2 primary bank. A later "
            "secondary experiment requires a new preregistration and completed "
            "prompt/spec-hash leakage audit."
        ),
    }
    return report, axis


def build(args: argparse.Namespace) -> dict[str, Any]:
    worktree = Path(__file__).resolve().parents[1]
    source_root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_bundle = output / "source_bundle_v2"
    d2_output = output / "D2_OUTCOME_AXIS_V2"
    source_bundle.mkdir()
    d2_output.mkdir()

    failed = worktree / "o1_runs/O1_REAL_001/axis_source_bundle_v1"
    diagnosis = worktree / "o1_runs/AXIS_DIAGNOSIS_V2"
    source_paths = {
        "a1": failed / "d1_l3_24.npy",
        "a2": failed / "d3_l3_24.npy",
        "causal_updates": (
            failed
            / "d4_capture_v1_1/d4_adapter1_induced_updates_l3_24.npy"
        ),
        "sequence_updates": (
            failed
            / "d4_capture_v1_1/d4_adapter2_induced_updates_l3_24.npy"
        ),
        "reference": failed / "D4_REFERENCE_SET.jsonl",
        "capture_report": failed / "d4_capture_v1_1/D4_CAPTURE_REPORT.json",
        "capture_parity": diagnosis / "D4_CAPTURE_PARITY_REPORT.json",
        "causal_checkpoint": (
            source_root
            / "artifacts/reports/probes/"
            "bg_causal_intervention_adapter_2026-05-18/"
            "adapter_checkpoints/best_adapter.pt"
        ),
        "sequence_checkpoint": (
            source_root
            / "artifacts/reports/probes/"
            "bg_sequence_level_adapter_2026-05-18/"
            "sequence_adapter_checkpoints/best_sequence_adapter.pt"
        ),
        "s3b2": (
            source_root
            / "artifacts/reports/cross_loop_early_layer_taps_20260720/"
            "s3b2_features.pt"
        ),
    }
    for key, expected in EXPECTED.items():
        if key == "checkpoint_tree":
            continue
        require_hash(source_paths[key], expected)

    references = load_reference(source_paths["reference"])
    causal = np.load(source_paths["causal_updates"], allow_pickle=False)
    sequence = np.load(source_paths["sequence_updates"], allow_pickle=False)
    if (
        causal.shape != (REFERENCE_TASKS, D_MODEL)
        or sequence.shape != (REFERENCE_TASKS, D_MODEL)
        or causal.dtype != np.dtype("<f4")
        or sequence.dtype != np.dtype("<f4")
    ):
        raise RuntimeError("actuator source matrix identity/shape/dtype changed")
    parity = json.loads(source_paths["capture_parity"].read_text())
    if parity["overall"] != "PASS":
        raise RuntimeError("accepted capture parity is not PASS")

    copy_map = {
        "a1_readable_source_l3_24.npy": source_paths["a1"],
        "a2_writable_source_l3_24.npy": source_paths["a2"],
        "causal_induced_updates_l3_24.npy": source_paths["causal_updates"],
        "sequence_induced_updates_l3_24.npy": source_paths["sequence_updates"],
        "actuator_reference_tasks.jsonl": source_paths["reference"],
        "D4_CAPTURE_REPORT.json": source_paths["capture_report"],
        "D4_CAPTURE_PARITY_REPORT.json": source_paths["capture_parity"],
    }
    for name, source in copy_map.items():
        shutil.copyfile(source, source_bundle / name)

    diagnostics = actuator_diagnostics(
        causal, sequence, references, parity, source_paths
    )
    write_json(output / "ACTUATOR_MEAN_DIAGNOSTICS.json", diagnostics)

    provenance_fields = {
        "A1_READABLE": {
            "construction": (
                "accepted real leakage-clean L3_24 process-quality tap covector, "
                "carried byte-for-byte; unit RMS"
            ),
            "provenance": {
                "source_loop": "paper L3",
                "source_layer": 24,
                "source_token_position": "all attention-mask-valid tokens",
                "source_pooling_rule": "attention-mask-valid mean; fp32 compute",
                "source_phase": "completed_candidate_state",
                "target_injection_loop": "paper L3 / zero-based loop 2",
                "target_injection_layer": 24,
                "target_token_position": "final_nonpadding_prompt_position",
                "coefficient_mapping": (
                    "identity; AntisymLinearNoNorm raw 2048-d residual covector; "
                    "no PCA or standardization inversion"
                ),
            },
        },
        "A2_WRITABLE": {
            "construction": (
                "accepted leading direction of centered exact-protocol S1/S3 "
                "L3_24 injection deltas; reconstructed because historical tensor "
                "was absent; unit RMS"
            ),
            "provenance": {
                "source_loop": "paper L3",
                "source_layer": 24,
                "source_token_position": "fixed last-token suffix",
                "source_pooling_rule": "none beyond one-token range mean",
                "source_phase": "exact-protocol one-shot injection delta",
                "target_injection_loop": "paper L3 / zero-based loop 2",
                "target_injection_layer": 24,
                "target_token_position": "final_nonpadding_prompt_position",
                "coefficient_mapping": (
                    "leading right singular vector of centered exact L3_24 "
                    "delta matrix; sign aligned to mean delta"
                ),
            },
        },
        "A3_CAUSAL_MEAN": {
            "construction": (
                "unit RMS raw uncentered mean of eight causal-adapter-induced "
                "L3_24 updates in frozen reference order; no PC"
            ),
            "provenance": {
                "source_loop": "paper L3 / zero-based loop 2",
                "source_layer": 24,
                "source_token_position": "final non-padding prompt position",
                "source_pooling_rule": "none; one position per task then raw task mean",
                "source_phase": "prompt_only",
                "target_injection_loop": "paper L3 / zero-based loop 2",
                "target_injection_layer": 24,
                "target_token_position": "final_nonpadding_prompt_position",
                "coefficient_mapping": (
                    "float64 arithmetic mean of exact stored float32 rows, then "
                    "unit RMS; no centering, PCA, SVD, or cross-adapter averaging"
                ),
            },
        },
        "A4_SEQUENCE_MEAN": {
            "construction": (
                "unit RMS raw uncentered mean of eight sequence-adapter-induced "
                "L3_24 updates in frozen reference order; no PC"
            ),
            "provenance": {
                "source_loop": "paper L3 / zero-based loop 2",
                "source_layer": 24,
                "source_token_position": "final non-padding prompt position",
                "source_pooling_rule": "none; one position per task then raw task mean",
                "source_phase": "prompt_only",
                "target_injection_loop": "paper L3 / zero-based loop 2",
                "target_injection_layer": 24,
                "target_token_position": "final_nonpadding_prompt_position",
                "coefficient_mapping": (
                    "float64 arithmetic mean of exact stored float32 rows, then "
                    "unit RMS; no centering, PCA, SVD, or cross-adapter averaging"
                ),
            },
        },
    }
    write_json(source_bundle / "AXIS_SOURCE_PROVENANCE.json", provenance_fields)

    source_bundle_record = {
        "schema_version": "o1.axis_source_bundle.v2.0",
        "scientific_scope": (
            "real frozen pre-O1 source reconstruction; no O1 calibration or "
            "confirmatory outcomes"
        ),
        "accepted_predecessors": {
            "failed_reconstruction_preservation_commit": (
                "ea40a48f7f41fdf60866eb2067033c62dd14be0f"
            ),
            "axis_diagnosis_v2_commit": (
                "38c474ebea3c6173d48d659d3edc0317c36acfd4"
            ),
        },
        "files": {
            name: {
                "sha256": sha256_file(source_bundle / name),
                "copied_byte_for_byte": True,
                "source_path_read_only": str(source),
            }
            for name, source in copy_map.items()
        },
        "model_checkpoint": {
            "path": str(source_root / "models/ouro_rltt_local"),
            "tree_sha256": EXPECTED["checkpoint_tree"],
            "copied": False,
        },
        "adapters": {
            "causal_path": str(source_paths["causal_checkpoint"]),
            "causal_sha256": EXPECTED["causal_checkpoint"],
            "sequence_path": str(source_paths["sequence_checkpoint"]),
            "sequence_sha256": EXPECTED["sequence_checkpoint"],
        },
        "A1_READABLE": {
            "source_file_sha256": EXPECTED["a1"],
            "canonical_row_sha256": canonical_row_hash(
                np.load(source_paths["a1"], allow_pickle=False)
            ),
            "status": "REAL_RECONSTRUCTED_AND_CARRIED_FORWARD",
        },
        "A2_WRITABLE": {
            "source_file_sha256": EXPECTED["a2"],
            "canonical_row_sha256": canonical_row_hash(
                np.load(source_paths["a2"], allow_pickle=False)
            ),
            "status": (
                "REAL_EXACT_PROTOCOL_RECONSTRUCTION; HISTORICAL_TENSOR_MISSING"
            ),
        },
        "A3_CAUSAL_MEAN": diagnostics["A3_CAUSAL_MEAN"],
        "A4_SEQUENCE_MEAN": diagnostics["A4_SEQUENCE_MEAN"],
    }
    write_json(source_bundle / "AXIS_SOURCE_BUNDLE.json", source_bundle_record)

    future_cohorts = [Path(path).resolve() for path in args.future_cohort]
    d2_report, d2_axis = d2_outcome_diagnosis(
        source_paths["s3b2"], future_cohorts
    )
    np.save(d2_output / "d2_outcome_candidate_l3_24.npy", d2_axis.astype("<f8"))
    write_json(d2_output / "D2_OUTCOME_AXIS_V2_REPORT.json", d2_report)
    write_json(
        d2_output / "D2_SOURCE_TASK_EXCLUSIONS.json",
        {
            "schema_version": "o1.d2_source_task_exclusions.v1",
            "source_sha256": EXPECTED["s3b2"],
            "task_ids": sorted(
                d2_report["labels"]["positive_task_ids_and_rows"].keys()
                | d2_report["labels"]["negative_task_ids_and_rows"].keys()
            ),
            "applies_to": "any later secondary outcome-axis experiment",
            "does_not_modify_v2_primary_bank": True,
        },
    )
    return {
        "actuator_diagnostics": str(output / "ACTUATOR_MEAN_DIAGNOSTICS.json"),
        "source_bundle": str(source_bundle),
        "d2_verdict": d2_report["D2_OUTCOME_AXIS_V2_VERDICT"],
        "d2_report": str(d2_output / "D2_OUTCOME_AXIS_V2_REPORT.json"),
        "no_o1_generation": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--future-cohort",
        action="append",
        default=[],
        help="future O1 task JSONL to test for task-ID overlap; may repeat",
    )
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
