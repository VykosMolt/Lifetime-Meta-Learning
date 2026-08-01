#!/usr/bin/env python3
"""Freeze the outcome-independent d4 reference cohort from pre-O1 heldout sets."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT = Path("/home/moloch/ouro_project").resolve()
CAUSAL = (
    PROJECT
    / "artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/"
    "adapter_dataset.json"
)
SEQUENCE = (
    PROJECT
    / "artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/"
    "sequence_adapter_dataset.json"
)
EXPECTED = {
    CAUSAL: "35fb682289794e3f8c93da28e508773587238ea2322719bb119f26150f1a5c05",
    SEQUENCE: "c0198a7f26df5935292e8ec44faebb782fb723387b680eac6c2d975db0b39d65",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    jsonl = Path(args.jsonl).resolve()
    manifest_path = Path(args.manifest).resolve()
    for path in (jsonl, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen d4 reference file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    observed = {str(path): sha256_file(path) for path in EXPECTED}
    for path, expected in EXPECTED.items():
        if observed[str(path)] != expected:
            raise RuntimeError(f"source hash mismatch for {path}")

    causal = json.loads(CAUSAL.read_text(encoding="utf-8"))
    sequence = json.loads(SEQUENCE.read_text(encoding="utf-8"))
    causal_heldout = {
        str(row["task_id"])
        for row in causal["examples"]
        if row.get("split") == "heldout"
    }
    sequence_rows = {
        str(row["task_id"]): row
        for row in sequence["tasks"]
        if row.get("split") == "heldout"
    }
    common = sorted(causal_heldout & set(sequence_rows))
    if len(common) != 8:
        raise ValueError(f"expected 8 common heldout task IDs, got {len(common)}: {common}")
    rows = []
    for task_id in common:
        source = sequence_rows[task_id]
        task = {
            "reference_rank": len(rows),
            "task_id": task_id,
            "domain": str(source["domain"]),
            "source_dataset": str(source.get("source_dataset") or ""),
            "prompt": str(source["prompt"]),
            "options": source["options"],
            "correct_option": str(source["correct_option"]),
            "trajectory_phase": "prompt_only",
            "capture_token_position": "final non-padding prompt token",
            "selection_provenance": "intersection of causal-adapter and sequence-adapter heldout task IDs",
        }
        task["task_content_sha256"] = canonical_hash(
            {
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "options": task["options"],
                "correct_option": task["correct_option"],
            }
        )
        rows.append(task)
    if len({row["task_content_sha256"] for row in rows}) != len(rows):
        raise ValueError("duplicate task content in d4 reference set")
    jsonl.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "o1.d4_reference_set.v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "selection_rule": "sorted intersection of the two adapters' frozen heldout task-ID sets",
        "selection_uses_adapter_outcomes": False,
        "selection_uses_o1_outcomes": False,
        "trajectory_phase": "prompt_only",
        "capture_token_position": "final non-padding prompt position",
        "task_count": len(rows),
        "task_ids": [row["task_id"] for row in rows],
        "domain_counts": {
            domain: sum(row["domain"] == domain for row in rows)
            for domain in sorted({row["domain"] for row in rows})
        },
        "source_hashes": observed,
        "reference_jsonl": str(jsonl),
        "reference_jsonl_sha256": sha256_file(jsonl),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
