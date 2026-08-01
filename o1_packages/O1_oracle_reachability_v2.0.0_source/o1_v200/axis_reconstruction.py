#!/usr/bin/env python3
"""Deterministic assembler for the O1 v2.0 L3_24 axis package.

The v2 primary bank is:

0. A1_READABLE
1. A2_WRITABLE
2. A3_CAUSAL_MEAN
3. A4_SEQUENCE_MEAN

A1 and A2 are carried forward byte-for-byte from the accepted real
reconstruction. A3 and A4 are recomputed from the uncentered arithmetic means
of the two frozen [8, 2048] induced-update matrices. No PC is extracted and the
two adapter directions are never averaged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from verify_axis_artifact import (
    ACTUATOR_CONSTRUCTION,
    AXIS_ORDER,
    D_MODEL,
    LOCUS,
    N_REFERENCE_TASKS,
    PKG_FILES,
    PROVENANCE_FIELDS,
    canonical_row_bytes,
    regenerate_random,
    rms,
    row_hash,
    unit_rms,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_axis(path: str | Path) -> np.ndarray:
    """Load an already frozen unit-RMS row without renormalizing its bytes."""
    vector = np.load(path, allow_pickle=False)
    if vector.dtype != np.dtype("<f8"):
        raise ValueError(f"{path}: expected little-endian float64, got {vector.dtype}")
    vector = np.asarray(vector).reshape(-1)
    if vector.shape != (D_MODEL,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{path}: expected finite ({D_MODEL},), got {vector.shape}")
    if abs(rms(vector) - 1.0) > 1e-12:
        raise ValueError(f"{path}: frozen source row is not unit RMS")
    return vector


def load_update_matrix(path: str | Path) -> np.ndarray:
    matrix = np.load(path, allow_pickle=False)
    if matrix.dtype != np.dtype("<f4"):
        raise ValueError(f"{path}: expected little-endian float32, got {matrix.dtype}")
    if matrix.shape != (N_REFERENCE_TASKS, D_MODEL):
        raise ValueError(
            f"{path}: expected {(N_REFERENCE_TASKS, D_MODEL)}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path}: induced-update matrix contains non-finite values")
    return matrix


def load_reference_rows(path: str | Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != N_REFERENCE_TASKS:
        raise ValueError(
            f"{path}: expected {N_REFERENCE_TASKS} reference tasks, got {len(rows)}"
        )
    task_ids = [row.get("task_id") for row in rows]
    ranks = [row.get("reference_rank") for row in rows]
    if len(set(task_ids)) != N_REFERENCE_TASKS:
        raise ValueError(f"{path}: reference task IDs are not unique")
    if ranks != list(range(N_REFERENCE_TASKS)):
        raise ValueError(f"{path}: reference ranks/order are not 0..7")
    return rows


def _axis_record(
    name: str,
    row: np.ndarray,
    source_file: str | Path,
    record: dict,
) -> dict:
    missing = [
        key
        for key in ("construction", "provenance")
        if key not in record
    ]
    if missing:
        raise ValueError(f"{name}: provenance record missing {missing}")
    provenance = record["provenance"]
    missing_provenance = [
        key for key in PROVENANCE_FIELDS if key not in provenance
    ]
    if missing_provenance:
        raise ValueError(f"{name}: provenance missing {missing_provenance}")
    return {
        "name": name,
        "sha256": row_hash(row),
        "construction": record["construction"],
        "source_artifact_sha256": sha256_file(source_file),
        "locus": LOCUS,
        "provenance": provenance,
    }


def assemble(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    a1 = load_frozen_axis(args.a1_readable)
    a2 = load_frozen_axis(args.a2_writable)
    causal_source = load_update_matrix(args.causal_updates)
    sequence_source = load_update_matrix(args.sequence_updates)
    reference_rows = load_reference_rows(args.reference_tasks)

    # Mean in float64 over the exact stored float32 capture. The mean is neither
    # centered nor passed through PCA/SVD.
    causal_mean = np.mean(causal_source.astype(np.float64), axis=0)
    sequence_mean = np.mean(sequence_source.astype(np.float64), axis=0)
    if rms(causal_mean) <= 0.0 or rms(sequence_mean) <= 0.0:
        raise ValueError("an actuator raw mean has zero RMS")
    a3 = unit_rms(causal_mean)
    a4 = unit_rms(sequence_mean)

    structured = np.stack((a1, a2, a3, a4)).astype("<f8", copy=False)
    random = regenerate_random(structured).astype("<f8", copy=False)

    with open(args.provenance, encoding="utf-8") as handle:
        provenance = json.load(handle)
    for name in AXIS_ORDER:
        if name not in provenance:
            raise ValueError(f"provenance JSON missing {name}")

    source_inputs = (
        args.a1_readable,
        args.a2_writable,
        args.causal_updates,
        args.sequence_updates,
        args.reference_tasks,
    )
    source_names = (
        "a1_readable_source_l3_24.npy",
        "a2_writable_source_l3_24.npy",
        "causal_induced_updates_l3_24.npy",
        "sequence_induced_updates_l3_24.npy",
        "actuator_reference_tasks.jsonl",
    )
    for source, destination in zip(source_inputs, source_names):
        shutil.copyfile(source, output / destination)

    axes_meta = [
        _axis_record(AXIS_ORDER[0], structured[0], args.a1_readable,
                     provenance[AXIS_ORDER[0]]),
        _axis_record(AXIS_ORDER[1], structured[1], args.a2_writable,
                     provenance[AXIS_ORDER[1]]),
        _axis_record(AXIS_ORDER[2], structured[2], args.causal_updates,
                     provenance[AXIS_ORDER[2]]),
        _axis_record(AXIS_ORDER[3], structured[3], args.sequence_updates,
                     provenance[AXIS_ORDER[3]]),
    ]

    actuator_meta = {
        "construction_method": ACTUATOR_CONSTRUCTION,
        "centered": False,
        "pc_extraction": False,
        "reference_conditioned": True,
        "reference_task_count": N_REFERENCE_TASKS,
        "reference_task_ids_in_order": [
            str(row["task_id"]) for row in reference_rows
        ],
        "reference_tasks_file_sha256": sha256_file(args.reference_tasks),
        "causal": {
            "source_matrix_file_sha256": sha256_file(args.causal_updates),
            "source_matrix_shape": list(causal_source.shape),
            "source_matrix_dtype": str(causal_source.dtype),
            "raw_mean_l2_norm": float(np.linalg.norm(causal_mean)),
            "raw_mean_rms": rms(causal_mean),
            "unit_rms_row_sha256": row_hash(a3),
        },
        "sequence": {
            "source_matrix_file_sha256": sha256_file(args.sequence_updates),
            "source_matrix_shape": list(sequence_source.shape),
            "source_matrix_dtype": str(sequence_source.dtype),
            "raw_mean_l2_norm": float(np.linalg.norm(sequence_mean)),
            "raw_mean_rms": rms(sequence_mean),
            "unit_rms_row_sha256": row_hash(a4),
        },
        "mean_direction_cosine": float(
            np.dot(a3, a4) / (np.linalg.norm(a3) * np.linalg.norm(a4))
        ),
    }

    np.save(output / "axes_l3_24.npy", structured)
    np.save(output / "random_axes_l3_24.npy", random)
    with (output / "AXIS_MANIFEST.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "o1.axis_manifest.v2.0",
                "axis_order": list(AXIS_ORDER),
                "axes": axes_meta,
                "actuator_mean_construction": actuator_meta,
                "explicit_exclusions": [
                    "original inadequate d2 is not in the primary bank",
                    "causal and sequence centered PC1 vectors are not used",
                    "A3 and A4 are not averaged into a shared direction",
                    "no O1 calibration or confirmatory outcomes were used",
                ],
            },
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    with (output / "GRAM_MATRIX.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "o1.axis_gram.v2.0",
                "axis_order": list(AXIS_ORDER),
                "gram": (structured @ structured.T).tolist(),
            },
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    shutil.copyfile(os.path.abspath(__file__), output / "axis_reconstruction.py")
    verifier = Path(__file__).with_name("verify_axis_artifact.py")
    shutil.copyfile(verifier, output / "verify_axis_artifact.py")

    input_artifacts = {
        "a1_readable": sha256_file(args.a1_readable),
        "a2_writable": sha256_file(args.a2_writable),
        "causal_induced_updates": sha256_file(args.causal_updates),
        "sequence_induced_updates": sha256_file(args.sequence_updates),
        "actuator_reference_tasks": sha256_file(args.reference_tasks),
        "provenance": sha256_file(args.provenance),
    }
    with (output / "RECONSTRUCTION_ATTESTATION.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "schema_version": "o1.axis_reconstruction_attestation.v2.0",
                "scientific_scope": (
                    "real frozen pre-O1 source reconstruction; no O1 calibration "
                    "or confirmatory outcomes"
                ),
                "reconstruction_code_sha256": sha256_file(
                    output / "axis_reconstruction.py"
                ),
                "verifier_code_sha256": sha256_file(
                    output / "verify_axis_artifact.py"
                ),
                "input_artifacts": input_artifacts,
                "embedded_source_files": {
                    "a1_readable_source_l3_24.npy": input_artifacts["a1_readable"],
                    "a2_writable_source_l3_24.npy": input_artifacts["a2_writable"],
                    "causal_induced_updates_l3_24.npy": input_artifacts[
                        "causal_induced_updates"
                    ],
                    "sequence_induced_updates_l3_24.npy": input_artifacts[
                        "sequence_induced_updates"
                    ],
                    "actuator_reference_tasks.jsonl": input_artifacts[
                        "actuator_reference_tasks"
                    ],
                },
                "assurance_boundary": {
                    "structurally_recomputed": [
                        "A1 and A2 identity to embedded frozen source-axis bytes",
                        "A3 and A4 as unit-RMS uncentered means of embedded matrices",
                        "reference task identity and order",
                        "axis rows, Gram matrix, and deterministic random bank",
                    ],
                    "attested_or_hash_bound": [
                        "upstream A1 fitted-tap reconstruction",
                        "upstream A2 exact-protocol delta reconstruction",
                        "adapter capture parity and upstream checkpoint execution",
                    ],
                },
            },
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    with (output / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for name in PKG_FILES:
            handle.write(f"{sha256_file(output / name)}  {name}\n")

    return {
        "output": str(output),
        "axis_row_sha256": [row_hash(row) for row in structured],
        "actuator_mean_direction_cosine": actuator_meta["mean_direction_cosine"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-readable", required=True)
    parser.add_argument("--a2-writable", required=True)
    parser.add_argument("--causal-updates", required=True)
    parser.add_argument("--sequence-updates", required=True)
    parser.add_argument("--reference-tasks", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(assemble(parser.parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
