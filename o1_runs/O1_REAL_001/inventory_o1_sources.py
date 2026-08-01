#!/usr/bin/env python3
"""Build the O1 real-source inventory without copying checkpoint weights."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT = Path("/home/moloch/ouro_project").resolve()
XLOOP = PROJECT / "artifacts/reports/cross_loop_early_layer_taps_20260720"
EXACT = (
    PROJECT
    / "artifacts/reports/proto_introspection/"
    "s1_s3_exact_injection_orthogonality_2026-06-17"
)
CAUSAL = (
    PROJECT
    / "artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18"
)
SEQUENCE = (
    PROJECT
    / "artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18"
)
MODEL = PROJECT / "models/ouro_rltt_local"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(PROJECT).as_posix()
    except ValueError:
        return None


def normalize_tensor_path(value: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", value)


def tensor_schemas_pt(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    top_keys = sorted(str(key) for key in payload) if isinstance(payload, dict) else []
    schemas: dict[tuple[str, tuple[int, ...], str], int] = defaultdict(int)

    def visit(value: Any, at: str) -> None:
        if torch.is_tensor(value):
            key = (normalize_tensor_path(at), tuple(value.shape), str(value.dtype))
            schemas[key] += 1
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{at}.{key}" if at else str(key))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{at}[{index}]")

    visit(payload, "")
    rows = [
        {
            "tensor_key": key,
            "shape": list(shape),
            "dtype": dtype,
            "occurrences": count,
        }
        for (key, shape, dtype), count in sorted(schemas.items())
    ]
    return top_keys, rows


def tensor_schemas_npy(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    import numpy as np

    arr = np.load(path, allow_pickle=False)
    return [], [
        {
            "tensor_key": "<array>",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "occurrences": 1,
        }
    ]


def tensor_schemas_npz(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        keys = sorted(payload.files)
        rows = [
            {
                "tensor_key": key,
                "shape": list(payload[key].shape),
                "dtype": str(payload[key].dtype),
                "occurrences": 1,
            }
            for key in keys
        ]
    return keys, rows


def tensor_schemas_safetensors(
    path: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    from safetensors import safe_open

    keys: list[str] = []
    rows: list[dict[str, Any]] = []
    with safe_open(path, framework="pt", device="cpu") as payload:
        for key in sorted(payload.keys()):
            keys.append(key)
            slc = payload.get_slice(key)
            rows.append(
                {
                    "tensor_key": key,
                    "shape": list(slc.get_shape()),
                    "dtype": str(slc.get_dtype()),
                    "occurrences": 1,
                }
            )
    return keys, rows


def parse_sha256sums(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split(maxsplit=1)
        if len(parts) == 2:
            name = parts[1].lstrip("* ")
            if name.startswith("./"):
                name = name[2:]
            out[name] = parts[0].lower()
    return out


XLOOP_SUMS = parse_sha256sums(XLOOP / "SHA256SUMS")
XLOOP_FEATURE_MANIFEST_SHA = sha256_file(XLOOP / "features/feature_manifest.json")
XLOOP_SPLIT_MANIFEST_SHA = sha256_file(XLOOP / "split_integrity.json")
CORECONTENT_MANIFEST = (
    PROJECT / "data/corecontent_v2/processed/candidate_groups_deduped.jsonl"
)
CORECONTENT_MANIFEST_SHA = sha256_file(CORECONTENT_MANIFEST)
EXACT_TASK_MANIFEST = (
    PROJECT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13/s1_1_task_set.json"
)
EXACT_TASK_MANIFEST_SHA = sha256_file(EXACT_TASK_MANIFEST)
CAUSAL_DATASET = CAUSAL / "adapter_dataset.json"
SEQUENCE_DATASET = SEQUENCE / "sequence_adapter_dataset.json"
CAUSAL_DATASET_SHA = sha256_file(CAUSAL_DATASET)
SEQUENCE_DATASET_SHA = sha256_file(SEQUENCE_DATASET)


def semantics(path: Path) -> dict[str, Any]:
    rp = rel(path) or str(path)
    result: dict[str, Any] = {
        "loop": None,
        "physical_layer": None,
        "token_position": None,
        "pooling_rule": None,
        "trajectory_phase": None,
        "source_task_split": None,
        "task_manifest_sha256": None,
        "artifact_status": "directly_verified",
        "historical_digest_status": "not_found",
    }
    if path.is_relative_to(XLOOP):
        result.update(
            {
                "loop": [1, 2, 3, 4],
                "physical_layer": [8, 16, 24, 36, 47],
                "token_position": "all attention-mask-valid tokens",
                "pooling_rule": "attention-mask-valid mean; fp32 compute",
                "trajectory_phase": "completed_candidate_state",
                "source_task_split": ["train", "val", "heldout"],
                "task_manifest_sha256": CORECONTENT_MANIFEST_SHA,
                "feature_manifest_sha256": XLOOP_FEATURE_MANIFEST_SHA,
                "split_manifest_sha256": XLOOP_SPLIT_MANIFEST_SHA,
            }
        )
        local = path.relative_to(XLOOP).as_posix()
        if local in XLOOP_SUMS:
            result["historical_digest_status"] = (
                "matched_shipped_SHA256SUMS"
                if XLOOP_SUMS[local] == sha256_file(path)
                else "MISMATCH_shipped_SHA256SUMS"
            )
    if path.is_relative_to(EXACT):
        result.update(
            {
                "loop": [1, 2, 3, 4],
                "physical_layer": [24, 36, 47],
                "token_position": "fixed last-token suffix",
                "pooling_rule": "none; last-token raw residual",
                "trajectory_phase": "partial_generation_branch_state",
                "source_task_split": "exact S1/S3 four-task source set",
                "task_manifest_sha256": EXACT_TASK_MANIFEST_SHA,
                "artifact_status": (
                    "reconstructed"
                    if path.name == "s1_s3_exact_injection_delta_bundle_2026-06-17.pt"
                    else "directly_verified"
                ),
            }
        )
    if path.is_relative_to(CAUSAL):
        result.update(
            {
                "loop": [1, 2, 3, 4],
                "physical_layer": 36,
                "token_position": "prefix_last_token",
                "pooling_rule": "none",
                "trajectory_phase": "partial_generation_prefix_state",
                "source_task_split": ["train", "val", "heldout"],
                "task_manifest_sha256": CAUSAL_DATASET_SHA,
            }
        )
    if path.is_relative_to(SEQUENCE):
        result.update(
            {
                "loop": [1, 2, 3, 4],
                "physical_layer": 36,
                "token_position": "prompt_last_token",
                "pooling_rule": "none",
                "trajectory_phase": "prompt_only_state",
                "source_task_split": ["train", "val", "heldout"],
                "task_manifest_sha256": SEQUENCE_DATASET_SHA,
            }
        )
    if path.is_relative_to(MODEL):
        result.update(
            {
                "trajectory_phase": "model_binding_only",
                "artifact_status": "directly_verified",
            }
        )
    if rp.startswith("utilities/tests/manual/bg_xloop_early_v1"):
        result.update(
            {
                "loop": [1, 2, 3, 4],
                "physical_layer": [8, 16, 24, 36, 47],
                "pooling_rule": "attention-mask-valid mean; fp32 compute",
                "trajectory_phase": "reconstruction_code",
            }
        )
    if rp.startswith("tools/s1_s3_exact_injection"):
        result.update(
            {
                "loop": [1, 2, 3, 4],
                "physical_layer": [24, 36, 47],
                "token_position": "fixed last-token suffix",
                "trajectory_phase": "reconstruction_code",
            }
        )
    if "adapter" in path.name or "adapter" in rp:
        if rp.startswith(("src/", "utilities/")):
            result["trajectory_phase"] = "capture_or_training_code"
    return result


def describe(path: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    row: dict[str, Any] = {
        "absolute_path": str(path.resolve()),
        "repository_relative_path": rel(path),
        "file_size": path.stat().st_size,
        "sha256": digest,
        "tensor_keys": [],
        "tensors": [],
        **semantics(path),
    }
    suffix = path.suffix.lower()
    try:
        if suffix in {".pt", ".pth"}:
            row["tensor_keys"], row["tensors"] = tensor_schemas_pt(path)
        elif suffix == ".npy":
            row["tensor_keys"], row["tensors"] = tensor_schemas_npy(path)
        elif suffix == ".npz":
            row["tensor_keys"], row["tensors"] = tensor_schemas_npz(path)
        elif suffix == ".safetensors":
            row["tensor_keys"], row["tensors"] = tensor_schemas_safetensors(path)
    except Exception as exc:
        row["tensor_inspection_error"] = f"{type(exc).__name__}: {exc}"
    return row


def selected_paths() -> list[Path]:
    paths: set[Path] = set()
    for root in (XLOOP, EXACT, CAUSAL, SEQUENCE):
        paths.update(path for path in root.rglob("*") if path.is_file())
    for root in (
        PROJECT / "artifacts/reports/probes/mpn_s1_baseline_2026-06-13",
        PROJECT / "artifacts/reports/probes/mpn_s3_closure_2026-06-17",
    ):
        paths.update(path for path in root.rglob("*") if path.is_file())
    for path in MODEL.rglob("*"):
        if path.is_file():
            paths.add(path)
    paths.update(
        {
            CORECONTENT_MANIFEST,
            PROJECT / "data/corecontent_v2/processed/candidate_groups_deduped.parquet",
            PROJECT
            / "artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/task_suite.json",
            PROJECT
            / "artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/continued_prefixes.json",
            PROJECT
            / "artifacts/reports/probes/bg_trajectory_prediction_2026-05-18/prefix_features.pt",
            PROJECT / "tools/s1_s3_exact_injection_orthogonality_2026_06_17.py",
            PROJECT / "tools/s1_s3_exact_injection_null_audit_2026_06_17.py",
            PROJECT / "src/evaluator/bg_causal_adapter.py",
            PROJECT / "src/evaluator/bg_sequence_adapter.py",
            PROJECT / "src/evaluator/bg_steering_hook.py",
        }
    )
    paths.update(
        PROJECT / "utilities/tests/manual" / name
        for name in (
            "bg_xloop_early_v1_common.py",
            "bg_xloop_early_v1_extract.py",
            "bg_xloop_early_v1_train_eval.py",
            "bg_xloop_early_v1_controls.py",
            "bg_xloop_early_v1_stats.py",
            "bg_xloop_early_v1_tests.py",
            "bg_causal_adapter_common.py",
            "train_bg_causal_intervention_adapter.py",
            "bg_sequence_adapter_common.py",
            "train_bg_sequence_level_adapter.py",
            "evaluate_bg_causal_adapter_teacher_forced.py",
            "evaluate_bg_sequence_adapter_heldout.py",
        )
    )
    return sorted(path for path in paths if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = selected_paths()
    entries = []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {rel(path) or path}", flush=True)
        entries.append(describe(path))
    model_entries = [row for row in entries if str(row["absolute_path"]).startswith(str(MODEL))]
    tree_material = "".join(
        f"{row['repository_relative_path']}\0{row['sha256']}\n"
        for row in sorted(model_entries, key=lambda value: value["repository_relative_path"])
    ).encode("utf-8")
    output = {
        "schema_version": "o1.real_source_inventory.v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT),
        "collector_package": {
            "expected": "/home/moloch/Downloads/O1_axis_source_collector_v1.zip",
            "status": "NOT_FOUND",
            "fallback": "inventory_o1_sources.py",
        },
        "checkpoint_binding": {
            "path": str(MODEL),
            "copy_included": False,
            "tree_hash_scheme": "sha256(sorted(repository_relative_path + NUL + file_sha256 + LF))",
            "tree_sha256": hashlib.sha256(tree_material).hexdigest(),
            "file_count": len(model_entries),
        },
        "source_manifest_hashes": {
            "corecontent_candidate_groups_sha256": CORECONTENT_MANIFEST_SHA,
            "xloop_feature_manifest_sha256": XLOOP_FEATURE_MANIFEST_SHA,
            "xloop_split_manifest_sha256": XLOOP_SPLIT_MANIFEST_SHA,
            "exact_s1_task_manifest_sha256": EXACT_TASK_MANIFEST_SHA,
            "causal_adapter_dataset_sha256": CAUSAL_DATASET_SHA,
            "sequence_adapter_dataset_sha256": SEQUENCE_DATASET_SHA,
        },
        "entry_count": len(entries),
        "entries": entries,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {out} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
