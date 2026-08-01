"""Build the deterministic O1 confirmatory seed matrix from the sealed manifest."""
from __future__ import annotations

import argparse
import json
import os

from o1_analysis import Manifest, derive_stream_seed, load_task_manifest, sha256_file


def build_seed_matrix(manifest_path: str, task_manifest_path: str, output_path: str) -> dict:
    if os.path.exists(output_path):
        raise FileExistsError(f"refusing to overwrite {output_path}")
    man = Manifest(manifest_path)
    tasks = load_task_manifest(task_manifest_path)
    master = man.get("cohorts.master_seed")
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
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(build_seed_matrix(a.manifest, a.confirmatory_task_manifest,
                                       a.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
