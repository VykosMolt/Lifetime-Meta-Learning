#!/usr/bin/env python3
"""Build outcome-blind O1 v2 calibration and confirmatory-candidate cohorts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from horizon_logic_generator_v2 import REVISION, generate_tasks

SETTINGS = ("rules_2", "rules_3", "rules_4")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path, calibration_per_setting: int,
          candidate_per_setting: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    calibration = []
    candidates = []
    for setting in SETTINGS:
        for ordinal, task in enumerate(generate_tasks(
                "calibration", setting, calibration_per_setting)):
            task["sealed_rank_within_generator_setting"] = ordinal
            task["stratum"] = "CALIBRATION_SETTING"
            calibration.append(task)
        for ordinal, task in enumerate(generate_tasks(
                "confirmatory_candidate", setting, candidate_per_setting)):
            task["sealed_rank_within_generator_setting"] = ordinal
            task["stratum"] = "PENDING_MECHANICAL_CALIBRATION_BANDING"
            candidates.append(task)

    all_hashes = [x["task_content_sha256"] for x in calibration + candidates]
    if len(set(all_hashes)) != len(all_hashes):
        raise AssertionError("task-content collision within/across cohorts")
    calibration.sort(key=lambda x: (SETTINGS.index(x["generator_setting_id"]),
                                    x["sealed_rank_within_generator_setting"]))
    candidates.sort(key=lambda x: (SETTINGS.index(x["generator_setting_id"]),
                                   x["sealed_rank_within_generator_setting"]))
    cal_path = out / "calibration_tasks.jsonl"
    candidate_path = out / "confirmatory_candidate_pool.jsonl"
    _write_jsonl(cal_path, calibration)
    _write_jsonl(candidate_path, candidates)
    report = {
        "schema_version": "2.0",
        "generator_revision": REVISION,
        "selection_uses_model_outcomes": False,
        "calibration": {
            "path": cal_path.name,
            "sha256": _sha(cal_path),
            "count": len(calibration),
            "count_per_setting": calibration_per_setting,
        },
        "confirmatory_candidate_pool": {
            "path": candidate_path.name,
            "sha256": _sha(candidate_path),
            "count": len(candidates),
            "count_per_setting": candidate_per_setting,
            "status": "CANDIDATE_POOL_ONLY",
            "final_selection": (
                "After calibration only: retain settings assigned to the two "
                "frozen baseline-rate bands; take each setting in sealed rank "
                "order until the mechanically derived G_confirm allocation is met."
            ),
        },
        "content_hash_disjoint": True,
        "note": (
            "The confirmatory cohort cannot truthfully carry final difficulty "
            "strata or ranks before calibration derives generator-level bands. "
            "This file freezes the larger outcome-blind candidate population."
        ),
    }
    report_path = out / "COHORT_BUILD_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--calibration-per-setting", type=int, default=32)
    p.add_argument("--candidate-per-setting", type=int, default=800)
    a = p.parse_args()
    print(json.dumps(build(
        Path(a.output_dir), a.calibration_per_setting,
        a.candidate_per_setting,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
