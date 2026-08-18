"""End-to-end tiny regression for the local nonscientific smoke sequence."""

from __future__ import annotations

import json

from foundation_learner.scripts.local_smoke_test import run_smoke


def test_tiny_smoke_runs_the_real_sequence_and_writes_a_passing_report(tmp_path):
    report = run_smoke(
        tiny=True,
        out_dir=str(tmp_path),
        device="cpu",
        budget_seconds=180,
        log=lambda _message: None,
    )

    report_path = tmp_path / "SMOKE_REPORT.json"
    assert report_path.is_file()
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["verdict"] == "PASS"
    assert on_disk["verdict"] == "PASS"
    assert report["mode"] == "TINY_SELF_TEST"
    assert report["classification"] == "NONSCIENTIFIC"
    assert on_disk["mode"] == "TINY_SELF_TEST"
    assert on_disk["classification"] == "NONSCIENTIFIC"
    assert len(report["checks"]) == 10
    assert all(value is not False for value in report["checks"].values())
    assert len(on_disk["checks"]) == 10
    assert all(value is not False for value in on_disk["checks"].values())
    assert isinstance(report["generation_preview"], str)
    assert report["checks"]["generation_returned_text"] is True
    assert report["checks"]["optimizer_step_taken"] is True
    assert on_disk["checks"]["generation_returned_text"] is True
    assert on_disk["checks"]["optimizer_step_taken"] is True
