#!/usr/bin/env python3
"""Regression fixtures for the v1.1 standalone d4 capture."""
from __future__ import annotations

import sys

import numpy as np

import d4_induced_update_capture_v1_1 as d4


def test_standalone_project_import_path() -> None:
    assert str(d4.PROJECT) in sys.path
    from src.evaluator.bg_causal_adapter import LowRankDeltaAdapter

    assert LowRankDeltaAdapter.__name__ == "LowRankDeltaAdapter"


def test_leading_pc_is_deterministic_unit_rms_and_sign_oriented() -> None:
    rng = np.random.default_rng(19)
    direction = np.zeros(d4.D_MODEL)
    direction[:3] = [3.0, -1.0, 0.5]
    coefficients = np.arange(1.0, 9.0)[:, None]
    updates = coefficients * direction[None, :]
    updates += rng.normal(0, 1e-4, size=updates.shape)
    a, meta_a = d4.leading_update_pc(updates)
    b, meta_b = d4.leading_update_pc(updates)
    assert np.array_equal(a, b)
    assert meta_a == meta_b
    assert abs(float(np.sqrt(np.mean(a * a))) - 1.0) < 1e-12
    assert float(a @ updates.mean(axis=0)) > 0


def main() -> int:
    test_standalone_project_import_path()
    print("PASS test_standalone_project_import_path")
    test_leading_pc_is_deterministic_unit_rms_and_sign_oriented()
    print("PASS test_leading_pc_is_deterministic_unit_rms_and_sign_oriented")
    print("2/2 d4 v1.1 regression fixtures passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
