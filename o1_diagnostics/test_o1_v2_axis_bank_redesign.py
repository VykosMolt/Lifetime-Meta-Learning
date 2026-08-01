#!/usr/bin/env python3
"""Unit tests for pure O1 v2 axis redesign calculations."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

import o1_v2_axis_bank_redesign as redesign


def test_unit_rms_and_canonical_hash() -> None:
    vector = np.arange(1, 2049, dtype=np.float64)
    normalized = redesign.unit_rms(vector)
    assert abs(redesign.rms(normalized) - 1.0) < 1e-12
    expected = hashlib.sha256(
        np.ascontiguousarray(normalized, dtype="<f8").tobytes()
    ).hexdigest()
    assert redesign.canonical_row_hash(normalized) == expected


def test_mean_stability_clusters_tasks() -> None:
    rng = np.random.default_rng(4)
    shared = rng.normal(size=2048)
    matrix = shared + 0.01 * rng.normal(size=(8, 2048))
    result = redesign.mean_stability(
        matrix, [f"task_{index}" for index in range(8)], 7
    )
    assert result["bootstrap"]["unit"].startswith("task_id")
    assert result["bootstrap"]["cosine_to_full_mean"]["q05"] > 0.99
    assert result["leave_one_task_out"]["cosine_to_full_mean"]["minimum"] > 0.99


def test_task_balanced_direction_not_row_balanced() -> None:
    positive = {
        "p1": np.array([1.0, 0.0]),
        "p2": np.array([3.0, 0.0]),
    }
    negative = {"n1": np.array([0.0, 2.0])}
    got = redesign._task_balanced_direction(positive, negative)
    assert np.array_equal(got, np.array([2.0, -2.0]))


def test_leakage_detects_task_overlap() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tasks.jsonl"
        path.write_text(
            json.dumps({"task_id": "source_a", "task_content_sha256": "x"})
            + "\n"
        )
        report = redesign._cohort_leakage({"source_a", "source_b"}, [path])
        assert report["status"] == "FAIL_TASK_ID_OVERLAP"
        assert report["task_id_overlap"] == ["source_a"]


def main() -> int:
    names = [name for name in sorted(globals()) if name.startswith("test_")]
    failed = 0
    for name in names:
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(names)-failed}/{len(names)} passed")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
