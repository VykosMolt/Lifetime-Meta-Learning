#!/usr/bin/env python3
"""Regression fixtures for d4 induced-update PC reconstruction."""
from __future__ import annotations

import numpy as np

import d4_induced_update_capture_v1 as d4


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
    test_leading_pc_is_deterministic_unit_rms_and_sign_oriented()
    print("PASS test_leading_pc_is_deterministic_unit_rms_and_sign_oriented")
    print("1/1 d4 induced-update regression fixture passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
