#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("axis_diagnosis_v2.py")
SPEC = importlib.util.spec_from_file_location("axis_diagnosis_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_centered_rank_and_requested_support() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(8, 32))
    decomp = MOD.centered_basis(matrix)
    assert decomp["effective_rank"] == 7
    requested = MOD.requested_explained(decomp)
    assert requested["pc_4"]["status"] == "ESTIMABLE"
    assert requested["pc_8"]["status"].startswith("UNAVAILABLE")
    assert requested["pc_16"]["status"].startswith("UNAVAILABLE")


def test_subspace_identity_and_orthogonality() -> None:
    identity = np.eye(4, 16)
    same = MOD.subspace_comparison(identity, identity, 4)
    assert np.isclose(same["projection_overlap_mean_cosine_squared"], 1.0)
    other = np.eye(16)[4:8]
    separate = MOD.subspace_comparison(identity, other, 4)
    assert np.isclose(separate["projection_overlap_mean_cosine_squared"], 0.0)


def test_partial_procrustes_on_exact_rotation() -> None:
    rng = np.random.default_rng(2)
    train_a = rng.normal(size=(7, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    offset = rng.normal(size=12)
    train_b = train_a @ q + offset
    test_a = rng.normal(size=12)
    prediction, metadata = MOD.partial_procrustes_predict(
        train_a, train_b, test_a
    )
    assert metadata["train_rank"] <= 6
    assert prediction.shape == (12,)
    assert np.isfinite(prediction).all()


def test_defect_verdict_has_priority() -> None:
    verdict, details = MOD.choose_d4_verdict(
        {"overall": "FAIL", "failed_checks": ["x"]},
        {},
        {},
        {},
    )
    assert verdict == "CAPTURE_OR_RECONSTRUCTION_DEFECT"
    assert details["failed_checks"] == ["x"]


def test_stable_means_can_yield_distinct_geometry_verdict() -> None:
    adapter = {
        "task_clustered_bootstrap": {
            "pc1_absolute_cosine_to_full": {"q05": 0.1, "median": 0.4},
            "mean_update_cosine_to_full": {"q05": 0.7, "median": 0.8},
        },
        "leave_one_task_out": {
            "pc1_absolute_cosine_summary": {"minimum": 0.4},
            "mean_update_cosine_summary": {"minimum": 0.95},
        },
    }
    comparison = {
        "pc1_absolute_cosine": 0.1,
        "principal_angles_and_projection_overlap": {
            "4": {
                "status": "ESTIMABLE",
                "projection_overlap_mean_cosine_squared": 0.2,
            }
        },
        "cross_validated_procrustes": {
            "a_to_b_cosine_summary": {"median": 0.2},
            "b_to_a_cosine_summary": {"median": 0.2},
        },
    }
    verdict, details = MOD.choose_d4_verdict(
        {"overall": "PASS", "failed_checks": []},
        adapter,
        adapter,
        comparison,
    )
    assert verdict == "STABLE_BUT_DISTINCT_ADAPTER_GEOMETRIES"
    assert details["causal_pc1_stable"] is False
    assert details["causal_mean_update_stable"] is True


def test_bounded_scope_markers() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "replacement_d4_emitted" in text
    assert "calibration_started" in text
    assert '"INSUFFICIENT_POSITIVE_TASK_SUPPORT"' in text
    assert "STABLE_SHARED_ONE_DIMENSIONAL_AXIS" in text
    assert "NO_STABLE_ACTUATOR_GEOMETRY" in text


if __name__ == "__main__":
    tests = [
        test_centered_rank_and_requested_support,
        test_subspace_identity_and_orthogonality,
        test_partial_procrustes_on_exact_rotation,
        test_defect_verdict_has_priority,
        test_stable_means_can_yield_distinct_geometry_verdict,
        test_bounded_scope_markers,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)}/{len(tests)}")
