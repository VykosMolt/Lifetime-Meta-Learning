"""Build the deterministic O1 confirmatory seed matrix from the sealed manifest."""
from __future__ import annotations

import argparse
import json
import os

from o1_analysis import Manifest, derive_stream_seed, load_task_manifest, sha256_file


def _load_calibration_tasks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        tasks = [json.loads(line) for line in fh if line.strip()]
    ids = [row.get("task_id") for row in tasks]
    hashes = [row.get("task_content_sha256") for row in tasks]
    if (not tasks or any(not isinstance(x, str) or not x for x in ids + hashes)
            or len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes)):
        raise ValueError("invalid calibration task manifest identities")
    return tasks


def build_seed_matrix(manifest_path: str, task_manifest_path: str, output_path: str,
                      calibration: bool = False) -> dict:
    if os.path.exists(output_path):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if calibration:
        with open(manifest_path, encoding="utf-8") as fh:
            raw_manifest = json.load(fh)
        master = raw_manifest["cohorts"]["master_seed"]
    else:
        man = Manifest(manifest_path)
        master = man.get("cohorts.master_seed")
    tasks = (_load_calibration_tasks(task_manifest_path)
             if calibration else load_task_manifest(task_manifest_path))
    matrix = {
        t["task_id"]: {str(i): derive_stream_seed(master, t["task_id"], i)
                       for i in range(8)}
        for t in tasks
    }
    tmp = output_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(matrix, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, output_path)
    return {"output": output_path, "sha256": sha256_file(output_path),
            "n_tasks": len(tasks)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--confirmatory-task-manifest", required=True)
    p.add_argument(
        "--calibration", action="store_true",
        help="accept an outcome-blind calibration manifest before difficulty banding",
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(build_seed_matrix(
        a.manifest, a.confirmatory_task_manifest, a.output,
        calibration=a.calibration,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
