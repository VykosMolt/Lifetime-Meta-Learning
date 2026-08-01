#!/usr/bin/env python3
"""Structural verifier for the O1 v2.0 four-axis L3_24 package.

The verifier recomputes the two reference-conditioned actuator means from the
embedded real source matrices. It contains no shared-PC cosine gate: v2 uses
two distinct mean-update axes and never averages them.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

N_AXES, D_MODEL, N_REFERENCE_TASKS = 4, 2048, 8
AXIS_ORDER = (
    "A1_READABLE",
    "A2_WRITABLE",
    "A3_CAUSAL_MEAN",
    "A4_SEQUENCE_MEAN",
)
LOCUS = "L3_24_raw_residual_2048"
DUP_COS = 0.98
GRAM_TOL, RMS_TOL, RECON_TOL = 1e-6, 1e-6, 1e-8
RANDOM_DOMAIN = b"O1-v2.0-fixed-random-domain"
ACTUATOR_CONSTRUCTION = (
    "unit_rms(mean(float64(exact_float32_induced_update_matrix), axis=0))"
)
PROVENANCE_FIELDS = (
    "source_loop",
    "source_layer",
    "source_token_position",
    "source_pooling_rule",
    "source_phase",
    "target_injection_loop",
    "target_injection_layer",
    "target_token_position",
    "coefficient_mapping",
)
SOURCE_FILES = (
    "a1_readable_source_l3_24.npy",
    "a2_writable_source_l3_24.npy",
    "causal_induced_updates_l3_24.npy",
    "sequence_induced_updates_l3_24.npy",
    "actuator_reference_tasks.jsonl",
)
PKG_FILES = (
    "axes_l3_24.npy",
    "random_axes_l3_24.npy",
    *SOURCE_FILES,
    "AXIS_MANIFEST.json",
    "GRAM_MATRIX.json",
    "RECONSTRUCTION_ATTESTATION.json",
    "axis_reconstruction.py",
    "verify_axis_artifact.py",
)


class AxisArtifactError(RuntimeError):
    def __init__(self, code: str, message: str, detail=None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.detail = detail


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row_bytes(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(vector, dtype="<f8").tobytes()


def row_hash(vector: np.ndarray) -> str:
    return hashlib.sha256(canonical_row_bytes(vector)).hexdigest()


def rms(vector: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(vector, dtype=np.float64))))


def unit_rms(vector: np.ndarray) -> np.ndarray:
    value = rms(vector)
    if not np.isfinite(value) or value <= 0.0:
        raise AxisArtifactError("A3_UNIT_RMS", "cannot normalize a zero/non-finite row")
    return np.asarray(vector, dtype=np.float64) / value


def random_seed_int(structured: np.ndarray) -> int:
    digest = hashlib.sha256()
    digest.update(RANDOM_DOMAIN)
    digest.update(hashlib.sha256(canonical_row_bytes(structured)).digest())
    return int(digest.hexdigest()[:16], 16)


def regenerate_random(structured: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(random_seed_int(structured))
    q, r = np.linalg.qr(
        rng.normal(size=(structured.shape[1], structured.shape[0]))
    )
    signs = np.sign(np.where(np.diag(r) == 0, 1.0, np.diag(r)))
    q = q * signs
    gram = structured @ structured.T
    chol = np.linalg.cholesky(gram + 1e-12 * np.eye(structured.shape[0]))
    return chol @ q.T


def canonical_correlations(a: np.ndarray, b: np.ndarray) -> list[float]:
    qa = np.linalg.qr(a.T)[0]
    qb = np.linalg.qr(b.T)[0]
    return sorted(
        np.linalg.svd(qa.T @ qb, compute_uv=False).round(6).tolist(),
        reverse=True,
    )


def _load_reference_rows(path: str | Path) -> list[dict]:
    try:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as exc:
        raise AxisArtifactError(
            "A19_REFERENCE_IDENTITY", f"invalid reference JSONL: {exc}"
        ) from exc
    if len(rows) != N_REFERENCE_TASKS:
        raise AxisArtifactError(
            "A19_REFERENCE_IDENTITY",
            f"reference set has {len(rows)} rows, expected {N_REFERENCE_TASKS}",
        )
    task_ids = [row.get("task_id") for row in rows]
    ranks = [row.get("reference_rank") for row in rows]
    if len(set(task_ids)) != N_REFERENCE_TASKS or ranks != list(
        range(N_REFERENCE_TASKS)
    ):
        raise AxisArtifactError(
            "A19_REFERENCE_IDENTITY",
            "reference tasks must have eight unique IDs in frozen rank order 0..7",
        )
    return rows


def _verify_hash_coverage(package: Path) -> dict[str, str]:
    sums_path = package / "SHA256SUMS"
    sums: dict[str, str] = {}
    for line_number, raw in enumerate(sums_path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise AxisArtifactError(
                "A15_PACKAGE_HASH_COVERAGE",
                f"malformed SHA256SUMS line {line_number}",
            )
        digest, name = parts
        name = os.path.basename(name)
        if name in sums:
            raise AxisArtifactError(
                "A15_PACKAGE_HASH_COVERAGE", f"duplicate checksum entry {name}"
            )
        sums[name] = digest
    on_disk = {
        path.name
        for path in package.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if on_disk != set(sums):
        raise AxisArtifactError(
            "A15_PACKAGE_HASH_COVERAGE",
            f"checksum coverage mismatch: unlisted={sorted(on_disk-set(sums))}, "
            f"missing={sorted(set(sums)-on_disk)}",
        )
    for name, expected in sums.items():
        if _sha(package / name) != expected:
            raise AxisArtifactError("A11_HASH", f"{name}: SHA256SUMS mismatch")
    forbidden = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if (path.is_dir() and path.name == "__pycache__")
        or (path.is_file() and path.suffix.lower() in {".pyc", ".pyo"})
    ]
    if forbidden:
        raise AxisArtifactError(
            "A20_BYTECODE", f"forbidden packaged bytecode: {forbidden[:10]}"
        )
    return sums


def verify(package_path: str) -> dict:
    package = Path(package_path)
    for name in (*PKG_FILES, "SHA256SUMS"):
        if not (package / name).is_file():
            raise AxisArtifactError("A0_MISSING_FILE", f"package lacks {name}")
    sums = _verify_hash_coverage(package)

    try:
        manifest = json.loads((package / "AXIS_MANIFEST.json").read_text())
        attestation = json.loads(
            (package / "RECONSTRUCTION_ATTESTATION.json").read_text()
        )
    except Exception as exc:
        raise AxisArtifactError("A5_AXIS_RECORD", f"invalid package JSON: {exc}") from exc

    structured = np.load(package / "axes_l3_24.npy", allow_pickle=False)
    random = np.load(package / "random_axes_l3_24.npy", allow_pickle=False)
    a1_source = np.load(
        package / "a1_readable_source_l3_24.npy", allow_pickle=False
    )
    a2_source = np.load(
        package / "a2_writable_source_l3_24.npy", allow_pickle=False
    )
    causal = np.load(
        package / "causal_induced_updates_l3_24.npy", allow_pickle=False
    )
    sequence = np.load(
        package / "sequence_induced_updates_l3_24.npy", allow_pickle=False
    )
    references = _load_reference_rows(package / "actuator_reference_tasks.jsonl")

    matrices = (
        ("structured", structured, (N_AXES, D_MODEL), np.dtype("<f8")),
        ("random", random, (N_AXES, D_MODEL), np.dtype("<f8")),
        ("A1 source", a1_source, (D_MODEL,), np.dtype("<f8")),
        ("A2 source", a2_source, (D_MODEL,), np.dtype("<f8")),
        (
            "causal source",
            causal,
            (N_REFERENCE_TASKS, D_MODEL),
            np.dtype("<f4"),
        ),
        (
            "sequence source",
            sequence,
            (N_REFERENCE_TASKS, D_MODEL),
            np.dtype("<f4"),
        ),
    )
    for name, matrix, shape, dtype in matrices:
        if matrix.shape != shape:
            raise AxisArtifactError(
                "A1_SHAPE", f"{name} shape {matrix.shape}, expected {shape}"
            )
        if matrix.dtype != dtype:
            raise AxisArtifactError(
                "A1_DTYPE", f"{name} dtype {matrix.dtype}, expected {dtype}"
            )
        if not np.all(np.isfinite(matrix)):
            raise AxisArtifactError("A2_NONFINITE", f"{name} contains non-finite values")

    for index in range(N_AXES):
        if abs(rms(structured[index]) - 1.0) > RMS_TOL:
            raise AxisArtifactError(
                "A3_UNIT_RMS",
                f"structured axis {index} has RMS {rms(structured[index]):.9f}",
            )

    if manifest.get("axis_order") != list(AXIS_ORDER) or [
        axis.get("name") for axis in manifest.get("axes", [])
    ] != list(AXIS_ORDER):
        raise AxisArtifactError(
            "A4_AXIS_ORDER", f"axis order != {list(AXIS_ORDER)}"
        )

    for index, axis in enumerate(manifest["axes"]):
        for key in (
            "name",
            "sha256",
            "construction",
            "source_artifact_sha256",
            "locus",
        ):
            if key not in axis:
                raise AxisArtifactError(
                    "A5_AXIS_RECORD", f"{axis.get('name')}: missing {key}"
                )
        if axis["locus"] != LOCUS:
            raise AxisArtifactError(
                "A6_CROSS_LOCUS", f"{axis['name']}: wrong locus {axis['locus']}"
            )
        actual = row_hash(structured[index])
        if axis["sha256"] != actual:
            raise AxisArtifactError(
                "A13_AXIS_ROW_HASH",
                f"{axis['name']}: claimed {axis['sha256']} but row hashes to {actual}",
            )
        missing = [
            field
            for field in PROVENANCE_FIELDS
            if field not in axis.get("provenance", {})
        ]
        if missing:
            raise AxisArtifactError(
                "A16_AXIS_PROVENANCE",
                f"{axis['name']}: provenance missing {missing}",
            )

    cosine = structured @ structured.T
    cosine = cosine / np.sqrt(
        np.outer(np.diag(cosine), np.diag(cosine))
    )
    for left in range(N_AXES):
        for right in range(left + 1, N_AXES):
            if abs(cosine[left, right]) >= DUP_COS:
                raise AxisArtifactError(
                    "A7_DUPLICATE_AXIS",
                    f"|cos({AXIS_ORDER[left]}, {AXIS_ORDER[right]})|="
                    f"{abs(cosine[left, right]):.6f} >= {DUP_COS}",
                )

    gram = structured @ structured.T
    gram_record = json.loads((package / "GRAM_MATRIX.json").read_text())
    stored_gram = np.asarray(gram_record.get("gram"))
    if (
        gram_record.get("axis_order") != list(AXIS_ORDER)
        or stored_gram.shape != (N_AXES, N_AXES)
        or np.max(np.abs(stored_gram - gram)) > GRAM_TOL
    ):
        raise AxisArtifactError("A8_GRAM_MISMATCH", "stored Gram != recomputed Gram")

    gram_deviation = float(
        np.max(np.abs(random @ random.T - gram)) / D_MODEL
    )
    if gram_deviation > 1e-6:
        raise AxisArtifactError(
            "A9_RANDOM_GRAM",
            f"random bank Gram mismatch {gram_deviation:.3e}",
        )
    expected_random = regenerate_random(structured)
    random_deviation = float(np.max(np.abs(random - expected_random)))
    if random_deviation > RECON_TOL:
        raise AxisArtifactError(
            "A12_RANDOM_ORIENTATION",
            f"random bank differs from fixed regeneration by {random_deviation:.3e}",
        )

    embedded = attestation.get("embedded_source_files", {})
    for name in SOURCE_FILES:
        actual = _sha(package / name)
        if embedded.get(name) != actual:
            raise AxisArtifactError(
                "A18_SOURCE_MATRIX_IDENTITY",
                f"{name}: attested identity does not match embedded bytes",
            )

    source_expected = (
        a1_source,
        a2_source,
        unit_rms(np.mean(causal.astype(np.float64), axis=0)),
        unit_rms(np.mean(sequence.astype(np.float64), axis=0)),
    )
    for index, expected in enumerate(source_expected):
        deviation = float(np.max(np.abs(structured[index] - expected)))
        if deviation > RECON_TOL:
            code = (
                "A21_MEAN_UPDATE_CONSTRUCTION"
                if index >= 2
                else "A18_SOURCE_MATRIX_IDENTITY"
            )
            raise AxisArtifactError(
                code,
                f"{AXIS_ORDER[index]} differs from its embedded source "
                f"construction by {deviation:.3e}",
            )

    actuator = manifest.get("actuator_mean_construction", {})
    if (
        actuator.get("construction_method") != ACTUATOR_CONSTRUCTION
        or actuator.get("centered") is not False
        or actuator.get("pc_extraction") is not False
        or actuator.get("reference_conditioned") is not True
        or actuator.get("reference_task_count") != N_REFERENCE_TASKS
    ):
        raise AxisArtifactError(
            "A21_MEAN_UPDATE_CONSTRUCTION",
            "actuator construction must be uncentered mean, no PC, reference-conditioned",
        )
    task_ids = [str(row["task_id"]) for row in references]
    if actuator.get("reference_task_ids_in_order") != task_ids:
        raise AxisArtifactError(
            "A19_REFERENCE_IDENTITY", "manifest reference ordering disagrees with JSONL"
        )
    identity_checks = (
        (
            actuator.get("reference_tasks_file_sha256"),
            _sha(package / "actuator_reference_tasks.jsonl"),
        ),
        (
            actuator.get("causal", {}).get("source_matrix_file_sha256"),
            _sha(package / "causal_induced_updates_l3_24.npy"),
        ),
        (
            actuator.get("sequence", {}).get("source_matrix_file_sha256"),
            _sha(package / "sequence_induced_updates_l3_24.npy"),
        ),
    )
    if any(claimed != actual for claimed, actual in identity_checks):
        raise AxisArtifactError(
            "A18_SOURCE_MATRIX_IDENTITY",
            "manifest source/reference identity does not match embedded bytes",
        )
    for key, matrix, row in (
        ("causal", causal, structured[2]),
        ("sequence", sequence, structured[3]),
    ):
        raw_mean = np.mean(matrix.astype(np.float64), axis=0)
        record = actuator.get(key, {})
        expected_values = {
            "source_matrix_shape": list(matrix.shape),
            "source_matrix_dtype": str(matrix.dtype),
            "raw_mean_l2_norm": float(np.linalg.norm(raw_mean)),
            "raw_mean_rms": rms(raw_mean),
            "unit_rms_row_sha256": row_hash(row),
        }
        for field, actual in expected_values.items():
            claimed = record.get(field)
            if isinstance(actual, float):
                if claimed is None or abs(float(claimed) - actual) > 1e-12:
                    raise AxisArtifactError(
                        "A21_MEAN_UPDATE_CONSTRUCTION",
                        f"{key}.{field} is not reproducible",
                    )
            elif claimed != actual:
                raise AxisArtifactError(
                    "A21_MEAN_UPDATE_CONSTRUCTION",
                    f"{key}.{field} is not reproducible",
                )
    mean_cosine = float(
        np.dot(structured[2], structured[3])
        / (np.linalg.norm(structured[2]) * np.linalg.norm(structured[3]))
    )
    if abs(float(actuator.get("mean_direction_cosine", 9.0)) - mean_cosine) > 1e-12:
        raise AxisArtifactError(
            "A21_MEAN_UPDATE_CONSTRUCTION", "mean-direction cosine is not reproducible"
        )

    if _sha(package / "verify_axis_artifact.py") != _sha(Path(__file__).resolve()):
        raise AxisArtifactError(
            "A17_VERIFIER_IDENTITY",
            "packaged verifier differs from the executing verifier",
        )
    if attestation.get("reconstruction_code_sha256") != _sha(
        package / "axis_reconstruction.py"
    ):
        raise AxisArtifactError(
            "A17_VERIFIER_IDENTITY", "reconstruction code hash is false"
        )
    if attestation.get("verifier_code_sha256") != _sha(
        package / "verify_axis_artifact.py"
    ):
        raise AxisArtifactError(
            "A17_VERIFIER_IDENTITY", "verifier code attestation is false"
        )

    return {
        "verdict": "SEALABLE",
        "schema_version": "o1.axis_verification.v2.0",
        "structurally_verified": [
            "shape, dtype, finiteness, and unit RMS",
            "v2 semantic order and canonical row hashes",
            "A1/A2 identity to embedded frozen source-axis bytes",
            "A3/A4 exact uncentered mean construction from embedded source matrices",
            "source-matrix and eight-task reference-order identity",
            "no centering, no PC extraction, and no A3/A4 averaging",
            "pairwise |cosine| < 0.98",
            "stored and recomputed structured Gram matrices",
            "random bank exact non-discretionary regeneration and Gram match",
            "complete SHA256SUMS coverage and no bytecode",
        ],
        "attested_or_hash_bound_only": [
            "upstream A1 tap fitting and leakage audit",
            "upstream A2 exact-protocol delta reconstruction (historical tensor absent)",
            "adapter capture execution and checkpoint/model parity",
        ],
        "axis_order": list(AXIS_ORDER),
        "axis_row_sha256": [row_hash(row) for row in structured],
        "source_file_sha256": {name: _sha(package / name) for name in SOURCE_FILES},
        "cosines": cosine.round(10).tolist(),
        "mean_actuator_cosine": mean_cosine,
        "random_seed_int": random_seed_int(structured),
        "random_max_abs_regeneration_deviation": random_deviation,
        "random_structured_gram_max_abs_per_dimension": gram_deviation,
        "structured_random_canonical_correlations": canonical_correlations(
            structured, random
        ),
        "sha256sums_entries": len(sums),
    }


def _rewrite_sums(package: str | Path) -> None:
    package = Path(package)
    with (package / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for name in PKG_FILES:
            handle.write(f"{_sha(package / name)}  {name}\n")


def _mk(
    destination: str,
    *,
    random_mode: str = "deterministic",
    duplicate: bool = False,
    wrong_locus: bool = False,
) -> str:
    """Create a complete synthetic package used only by shipped fixtures."""
    package = Path(destination)
    package.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260729)
    a1 = unit_rms(rng.normal(size=D_MODEL))
    a2 = unit_rms(rng.normal(size=D_MODEL))
    causal = rng.normal(size=(N_REFERENCE_TASKS, D_MODEL)).astype("<f4")
    sequence = (0.5 * causal + rng.normal(size=causal.shape)).astype("<f4")
    a3 = unit_rms(np.mean(causal.astype(np.float64), axis=0))
    a4 = unit_rms(np.mean(sequence.astype(np.float64), axis=0))
    structured = np.stack((a1, a2, a3, a4)).astype("<f8")
    if duplicate:
        structured[1] = structured[0]
    random = regenerate_random(structured)
    if random_mode == "copy":
        random = structured.copy()
    elif random_mode == "orthogonal":
        q = np.linalg.qr(rng.normal(size=(D_MODEL, N_AXES)))[0]
        random = np.sqrt(D_MODEL) * q.T
    np.save(package / "axes_l3_24.npy", structured)
    np.save(package / "random_axes_l3_24.npy", random.astype("<f8"))
    np.save(package / "a1_readable_source_l3_24.npy", a1.astype("<f8"))
    np.save(package / "a2_writable_source_l3_24.npy", a2.astype("<f8"))
    np.save(package / "causal_induced_updates_l3_24.npy", causal)
    np.save(package / "sequence_induced_updates_l3_24.npy", sequence)
    references = [
        {
            "reference_rank": index,
            "task_id": f"synthetic_reference_{index}",
            "trajectory_phase": "prompt_only",
        }
        for index in range(N_REFERENCE_TASKS)
    ]
    with (package / "actuator_reference_tasks.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in references:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    common_provenance = {
        field: "synthetic_fixture"
        for field in PROVENANCE_FIELDS
    }
    axes = [
        {
            "name": name,
            "sha256": row_hash(structured[index]),
            "construction": "synthetic fixture",
            "source_artifact_sha256": "0" * 64,
            "locus": "WRONG_LOCUS" if wrong_locus and index == 0 else LOCUS,
            "provenance": common_provenance,
        }
        for index, name in enumerate(AXIS_ORDER)
    ]
    causal_mean = np.mean(causal.astype(np.float64), axis=0)
    sequence_mean = np.mean(sequence.astype(np.float64), axis=0)
    actuator = {
        "construction_method": ACTUATOR_CONSTRUCTION,
        "centered": False,
        "pc_extraction": False,
        "reference_conditioned": True,
        "reference_task_count": N_REFERENCE_TASKS,
        "reference_task_ids_in_order": [row["task_id"] for row in references],
        "reference_tasks_file_sha256": _sha(
            package / "actuator_reference_tasks.jsonl"
        ),
        "causal": {
            "source_matrix_file_sha256": _sha(
                package / "causal_induced_updates_l3_24.npy"
            ),
            "source_matrix_shape": list(causal.shape),
            "source_matrix_dtype": str(causal.dtype),
            "raw_mean_l2_norm": float(np.linalg.norm(causal_mean)),
            "raw_mean_rms": rms(causal_mean),
            "unit_rms_row_sha256": row_hash(structured[2]),
        },
        "sequence": {
            "source_matrix_file_sha256": _sha(
                package / "sequence_induced_updates_l3_24.npy"
            ),
            "source_matrix_shape": list(sequence.shape),
            "source_matrix_dtype": str(sequence.dtype),
            "raw_mean_l2_norm": float(np.linalg.norm(sequence_mean)),
            "raw_mean_rms": rms(sequence_mean),
            "unit_rms_row_sha256": row_hash(structured[3]),
        },
        "mean_direction_cosine": float(
            np.dot(structured[2], structured[3])
            / (np.linalg.norm(structured[2]) * np.linalg.norm(structured[3]))
        ),
    }
    (package / "AXIS_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": "o1.axis_manifest.v2.0",
                "axis_order": list(AXIS_ORDER),
                "axes": axes,
                "actuator_mean_construction": actuator,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (package / "GRAM_MATRIX.json").write_text(
        json.dumps(
            {
                "schema_version": "o1.axis_gram.v2.0",
                "axis_order": list(AXIS_ORDER),
                "gram": (structured @ structured.T).tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    import shutil

    shutil.copyfile(Path(__file__).resolve(), package / "verify_axis_artifact.py")
    (package / "axis_reconstruction.py").write_text(
        "# synthetic reconstruction fixture\n"
    )
    embedded = {name: _sha(package / name) for name in SOURCE_FILES}
    (package / "RECONSTRUCTION_ATTESTATION.json").write_text(
        json.dumps(
            {
                "schema_version": "o1.axis_reconstruction_attestation.v2.0",
                "reconstruction_code_sha256": _sha(
                    package / "axis_reconstruction.py"
                ),
                "verifier_code_sha256": _sha(
                    package / "verify_axis_artifact.py"
                ),
                "embedded_source_files": embedded,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _rewrite_sums(package)
    return str(package)


def selftest() -> int:
    import shutil
    import tempfile

    failed = 0

    def case(name, expected_code, mutate=None, **kwargs):
        nonlocal failed
        package = _mk(tempfile.mkdtemp(), **kwargs)
        if mutate:
            mutate(Path(package))
        try:
            result = verify(package)
            actual = None
            detail = result["verdict"]
        except AxisArtifactError as exc:
            actual = exc.code
            detail = str(exc)
        passed = actual == expected_code
        failed += not passed
        print(f"  {'ok  ' if passed else 'FAIL'} {name}: {actual or detail}")

    def copy_random(package):
        shutil.copyfile(
            package / "axes_l3_24.npy", package / "random_axes_l3_24.npy"
        )
        _rewrite_sums(package)

    def lie_row_hash(package):
        manifest = json.loads((package / "AXIS_MANIFEST.json").read_text())
        manifest["axes"][0]["sha256"] = "0" * 64
        (package / "AXIS_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        _rewrite_sums(package)

    def replace_causal_source(package):
        source = np.load(
            package / "causal_induced_updates_l3_24.npy", allow_pickle=False
        )
        source[0, 0] += np.float32(0.25)
        np.save(package / "causal_induced_updates_l3_24.npy", source)
        _rewrite_sums(package)

    def center_causal_axis(package):
        structured = np.load(package / "axes_l3_24.npy", allow_pickle=False)
        source = np.load(
            package / "causal_induced_updates_l3_24.npy", allow_pickle=False
        ).astype(np.float64)
        centered = source - np.mean(source, axis=0)
        structured[2] = unit_rms(np.linalg.svd(centered, full_matrices=False)[2][0])
        np.save(package / "axes_l3_24.npy", structured)
        manifest = json.loads((package / "AXIS_MANIFEST.json").read_text())
        manifest["axes"][2]["sha256"] = row_hash(structured[2])
        (package / "AXIS_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        gram = json.loads((package / "GRAM_MATRIX.json").read_text())
        gram["gram"] = (structured @ structured.T).tolist()
        (package / "GRAM_MATRIX.json").write_text(
            json.dumps(gram, indent=2, sort_keys=True) + "\n"
        )
        np.save(package / "random_axes_l3_24.npy", regenerate_random(structured))
        _rewrite_sums(package)

    def reorder_reference(package):
        path = package / "actuator_reference_tasks.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(reversed(lines)) + "\n")
        att = json.loads(
            (package / "RECONSTRUCTION_ATTESTATION.json").read_text()
        )
        att["embedded_source_files"]["actuator_reference_tasks.jsonl"] = _sha(path)
        (package / "RECONSTRUCTION_ATTESTATION.json").write_text(
            json.dumps(att, indent=2, sort_keys=True) + "\n"
        )
        _rewrite_sums(package)

    def wrong_gram(package):
        record = json.loads((package / "GRAM_MATRIX.json").read_text())
        record["gram"][0][1] += 1.0
        (package / "GRAM_MATRIX.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        _rewrite_sums(package)

    def tamper_verifier(package):
        with (package / "verify_axis_artifact.py").open("a") as handle:
            handle.write("\n# tamper\n")
        _rewrite_sums(package)

    def omit_manifest_sum(package):
        lines = [
            line
            for line in (package / "SHA256SUMS").read_text().splitlines()
            if "AXIS_MANIFEST.json" not in line
        ]
        (package / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    def missing_provenance(package):
        manifest = json.loads((package / "AXIS_MANIFEST.json").read_text())
        del manifest["axes"][0]["provenance"]["source_phase"]
        (package / "AXIS_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        _rewrite_sums(package)

    def claim_pc(package):
        manifest = json.loads((package / "AXIS_MANIFEST.json").read_text())
        manifest["actuator_mean_construction"]["pc_extraction"] = True
        (package / "AXIS_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        _rewrite_sums(package)

    def fake_recon_hash(package):
        att = json.loads(
            (package / "RECONSTRUCTION_ATTESTATION.json").read_text()
        )
        att["reconstruction_code_sha256"] = "0" * 64
        (package / "RECONSTRUCTION_ATTESTATION.json").write_text(
            json.dumps(att, indent=2, sort_keys=True) + "\n"
        )
        _rewrite_sums(package)

    case("clean v2 package", None)
    case("copied structured random bank", "A12_RANDOM_ORIENTATION", copy_random)
    case("orthogonal but not Gram-matched random bank", "A9_RANDOM_GRAM",
         random_mode="orthogonal")
    case("fabricated canonical row hash", "A13_AXIS_ROW_HASH", lie_row_hash)
    case("source matrix changed after assembly", "A18_SOURCE_MATRIX_IDENTITY",
         replace_causal_source)
    case("centered PC substituted for causal mean", "A21_MEAN_UPDATE_CONSTRUCTION",
         center_causal_axis)
    case("reference task order reversed", "A19_REFERENCE_IDENTITY",
         reorder_reference)
    case("duplicate structured axes", "A7_DUPLICATE_AXIS", duplicate=True)
    case("wrong L3_24 locus", "A6_CROSS_LOCUS", wrong_locus=True)
    case("stored Gram edited", "A8_GRAM_MISMATCH", wrong_gram)
    case("packaged verifier edited", "A17_VERIFIER_IDENTITY", tamper_verifier)
    case("checksum coverage omits manifest", "A15_PACKAGE_HASH_COVERAGE",
         omit_manifest_sum)
    case("axis provenance missing", "A16_AXIS_PROVENANCE", missing_provenance)
    case("manifest claims PC extraction", "A21_MEAN_UPDATE_CONSTRUCTION", claim_pc)
    case("fabricated reconstruction hash", "A17_VERIFIER_IDENTITY",
         fake_recon_hash)
    print(f"\n  {15-failed}/15 axis-artifact checks behaved as specified")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(verify(sys.argv[1]), indent=2, sort_keys=True))
