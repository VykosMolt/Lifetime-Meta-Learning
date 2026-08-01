#!/usr/bin/env python3
"""Refresh only the prospective D2 leakage audit after cohorts are frozen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from o1_v2_axis_bank_redesign import d2_outcome_diagnosis, write_json


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--future-cohort", action="append", required=True)
    a = p.parse_args()
    out = Path(a.output_dir)
    report, axis = d2_outcome_diagnosis(
        Path(a.source), [Path(x).resolve() for x in a.future_cohort],
    )
    np.save(out / "d2_outcome_candidate_l3_24.npy", axis.astype("<f8"))
    write_json(out / "D2_OUTCOME_AXIS_V2_REPORT.json", report)
    result = {
        "verdict": report["D2_OUTCOME_AXIS_V2_VERDICT"],
        "leakage": report["leakage_against_future_o1_cohorts"],
        "primary_bank_modified": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
