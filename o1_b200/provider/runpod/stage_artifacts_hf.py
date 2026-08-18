#!/usr/bin/env python3
"""Stage the large private artifacts to a PRIVATE Hugging Face repo.

Free, private, reachable from RunPod pods (with a read token), and staged
BEFORE the B200 billing clock starts — so the pod only ever downloads.

Usage (after `hf auth login` with a WRITE-capable token; never paste tokens
into chat):

    python -m o1_b200.provider.runpod.stage_artifacts_hf \
        --repo Vykos/o1-b200-staging upload
    python -m o1_b200.provider.runpod.stage_artifacts_hf \
        --repo Vykos/o1-b200-staging verify

`upload` creates the private repo if needed and uploads: the Ouro-RLTT
checkpoint tree (5.0 GB), tokenizer binding, axis package, verified O1
package zips, cohort manifests, and seed matrix — each path mirrored under
the repo root exactly as the transfer manifest destinations expect.

`verify` is the ARTIFACT DOWNLOAD TEST: it re-downloads every staged file to
a scratch directory through the same code path the pod will use
(huggingface_hub with a token) and recomputes every SHA-256 against
RUNPOD_ARTIFACT_TRANSFER_MANIFEST.json, writing HF_STAGING_RECORD.json.
The pod refuses model loading unless the same verification passes there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)

from o1_b200.runner.identity import sha256_file, sha256_tree  # noqa: E402

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "RUNPOD_ARTIFACT_TRANSFER_MANIFEST.json")
RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "HF_STAGING_RECORD.json")

# logical name -> path-in-repo (mirrors /artifacts/ destinations)
STAGED = {
    "ouro_rltt_checkpoint": "ouro_rltt_local",
    "tokenizer_binding": "TOKENIZER_BINDING.json",
    "o1_v210_package_zip": "O1_oracle_reachability_v2.1.0_verified.zip",
    "axis_package": "AXIS_PACKAGE_V2",
    "calibration_task_manifest": "COHORTS/calibration_tasks.jsonl",
    "calibration_seed_matrix": "calibration_seed_matrix.json",
}


def _load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)["artifacts"]


def upload(repo: str) -> int:
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print(f"authenticated as {who.get('name')}")
    api.create_repo(repo, private=True, repo_type="model", exist_ok=True)
    info = api.repo_info(repo)
    if not info.private:
        raise SystemExit(f"REFUSED: {repo} is not private; the checkpoint "
                         f"must not be published")
    artifacts = _load_manifest()
    for name, dest in STAGED.items():
        src = artifacts[name]["local_path"]
        print(f"uploading {name}: {src} -> {repo}/{dest}")
        if os.path.isdir(src):
            api.upload_folder(repo_id=repo, folder_path=src, path_in_repo=dest)
        else:
            api.upload_file(repo_id=repo, path_or_fileobj=src,
                            path_in_repo=dest)
    print("upload complete; now run: ... verify")
    return 0


def verify(repo: str) -> int:
    """The artifact download test (same fetch path the pod uses)."""
    from huggingface_hub import snapshot_download
    artifacts = _load_manifest()
    with tempfile.TemporaryDirectory(prefix="o1_hf_verify_") as tmp:
        local = snapshot_download(repo_id=repo, local_dir=tmp)
        results = {}
        ok = True
        for name, dest in STAGED.items():
            want = artifacts[name]["sha256"]
            path = os.path.join(local, dest)
            got = sha256_tree(path) if os.path.isdir(path) else sha256_file(path)
            results[name] = {"expected": want, "downloaded": got,
                             "match": got == want}
            ok &= got == want
            print(f"{'OK  ' if got == want else 'FAIL'} {name}")
    record = {
        "schema": "o1b200.hf_staging_record.v1",
        "repo": repo,
        "private": True,
        "artifact_download_test": "PASS" if ok else "FAIL",
        "results": results,
        "pod_fetch_note": ("the pod downloads via huggingface_hub with a "
                           "READ-scoped token in RUNPOD_API-independent env "
                           "var HF_TOKEN (never baked into the image), then "
                           "deploy/verify_artifacts.py re-verifies before "
                           "any model load"),
    }
    with open(RECORD, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"ARTIFACT DOWNLOAD TEST: {'PASS' if ok else 'FAIL'} -> {RECORD}")
    return 0 if ok else 1


RESULT_ROUNDTRIP_RECORD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "RESULT_ROUNDTRIP_RECORD.json")


def result_roundtrip(results_repo: str) -> int:
    """RESULT DESTINATION ROUND-TRIP test.

    Creates the PRIVATE results repo (kept SEPARATE from the artifact
    staging repo so the pod's fine-grained WRITE token is scoped to results
    only and can never touch the checkpoint), uploads a probe archive
    through the pod's upload path, re-downloads it through the driver's
    download path (hf:// scheme), and hash-compares.
    """
    import hashlib
    import tarfile
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    api.create_repo(results_repo, private=True, repo_type="model",
                    exist_ok=True)
    if not api.repo_info(results_repo).private:
        raise SystemExit(f"REFUSED: {results_repo} is not private")
    with tempfile.TemporaryDirectory(prefix="o1_result_probe_") as tmp:
        payload = os.path.join(tmp, "probe.txt")
        with open(payload, "w") as fh:
            fh.write("o1 b200 result-destination round-trip probe\n")
        archive = os.path.join(tmp, "roundtrip_probe.tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(payload, arcname="probe.txt")
        want = sha256_file(archive)
        api.upload_file(repo_id=results_repo, path_or_fileobj=archive,
                        path_in_repo="probe/roundtrip_probe.tar.gz")
        got_path = hf_hub_download(repo_id=results_repo,
                                   filename="probe/roundtrip_probe.tar.gz",
                                   local_dir=os.path.join(tmp, "down"))
        got = sha256_file(got_path)
    ok = got == want
    record = {
        "schema": "o1b200.result_roundtrip_record.v1",
        "results_repo": results_repo,
        "private": True,
        "result_destination_round_trip": "PASS" if ok else "FAIL",
        "probe_sha256_uploaded": want,
        "probe_sha256_downloaded": got,
        "separation_note": ("results repo is separate from the artifact "
                            "staging repo; the pod's write token must be "
                            "fine-grained to the results repo ONLY"),
    }
    with open(RESULT_ROUNDTRIP_RECORD, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"RESULT DESTINATION ROUND-TRIP: {'PASS' if ok else 'FAIL'} "
          f"-> {RESULT_ROUNDTRIP_RECORD}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True,
                   help="private HF repo: staging repo for upload/verify, "
                        "results repo for result-roundtrip")
    p.add_argument("action", choices=("upload", "verify", "result-roundtrip"))
    a = p.parse_args()
    if a.action == "upload":
        return upload(a.repo)
    if a.action == "verify":
        return verify(a.repo)
    return result_roundtrip(a.repo)


if __name__ == "__main__":
    raise SystemExit(main())
