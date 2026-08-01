#!/usr/bin/env python3
"""Regression fixtures for the real-axis source reconstruction primitives."""
from __future__ import annotations

import numpy as np

import axis_source_reconstruction_v1 as r


def test_unit_rms_and_canonical_bytes() -> None:
    source = np.linspace(-3.0, 5.0, r.D_MODEL, dtype=np.float64)
    axis, before = r.unit_rms(source)
    assert before > 0
    assert axis.shape == (r.D_MODEL,)
    assert axis.dtype == np.dtype("<f8")
    assert np.isfinite(axis).all()
    assert abs(float(np.sqrt(np.mean(axis * axis))) - 1.0) < 1e-12
    assert r.canonical_row_sha256(axis) == r.canonical_row_sha256(axis.copy())


def test_task_bootstrap_deterministic_and_oriented() -> None:
    e0 = np.zeros(r.D_MODEL)
    e0[0] = 1.0
    e1 = np.zeros(r.D_MODEL)
    e1[1] = 0.2
    features = np.stack(
        [
            3 * e0 + e1,
            2 * e0 - e1,
            -2 * e0,
            -e0 + e1,
            -3 * e0,
            -2 * e0 - e1,
        ]
    )
    rows = [
        {"task_id": "a", "correct": True},
        {"task_id": "a", "correct": True},
        {"task_id": "a", "correct": False},
        {"task_id": "a", "correct": False},
        {"task_id": "b", "correct": False},
        {"task_id": "b", "correct": False},
    ]
    a, meta_a = r.task_balanced_bootstrap_pc(
        features, rows, seed=17, draws=256
    )
    b, meta_b = r.task_balanced_bootstrap_pc(
        features, rows, seed=17, draws=256
    )
    assert np.array_equal(a, b)
    assert meta_a == meta_b
    positive_mean = features[:2].mean(axis=0)
    negative_mean = np.stack(
        [features[2:4].mean(axis=0), features[4:6].mean(axis=0)]
    ).mean(axis=0)
    assert float(a @ (positive_mean - negative_mean)) > 0


def main() -> int:
    test_unit_rms_and_canonical_bytes()
    print("PASS test_unit_rms_and_canonical_bytes")
    test_task_bootstrap_deterministic_and_oriented()
    print("PASS test_task_bootstrap_deterministic_and_oriented")
    print("2/2 axis-source reconstruction regression fixtures passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
