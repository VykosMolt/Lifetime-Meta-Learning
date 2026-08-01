"""End-to-end command-line smoke tests for the O1 v2.0 package."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from o1_analysis import sha256_file
import test_o1_fixtures as base

HERE = os.path.dirname(os.path.abspath(__file__))


def run_cli(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run([sys.executable, *args], cwd=HERE, text=True,
                        capture_output=True)
    if cp.returncode != expect:
        raise AssertionError(
            f"CLI {' '.join(args)} returned {cp.returncode}, expected {expect}\n"
            f"STDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")
    return cp


def fresh(tag: str) -> dict:
    return base.write_run(base.make_clean(G=8), tag=tag)


def test_00_seed_matrix_cli_reproduces_sealed_bytes():
    r = fresh("cli00")
    out = os.path.join(r["dir"], "seed_matrix_cli.json")
    run_cli("build_seed_matrix.py", "--manifest", r["manifest"],
            "--confirmatory-task-manifest", r["conf"], "--output", out)
    assert json.load(open(out)) == json.load(open(r["seeds"]))
    print("  deterministic seed-matrix CLI reproduces the sealed matrix")


def test_01_provenance_cli_hashes_real_artifacts():
    r = fresh("cli01")
    out = os.path.join(r["dir"], "RUN_PROVENANCE.cli.json")
    run_cli("build_run_provenance.py", "--manifest", r["manifest"],
            "--artifact-paths", r["artifact_paths"], "--output", out)
    got = json.load(open(out))
    sealed = json.load(open(r["prov"]))
    assert got["resolved_generation_config"] == sealed["resolved_generation_config"]
    assert got["runtime_attestation"] == sealed["runtime_attestation"]
    print("  runtime-provenance CLI recursively binds the real artifacts")


def test_02_calibration_cli_recomputes_exact_result():
    r = fresh("cli02")
    out = os.path.join(r["dir"], "CALIBRATION_RESULTS.cli.json")
    run_cli("calibration_analysis.py", r["cal_records"], r["cal_metadata"],
            r["manifest"], r["cal"], r["cal_precommit"], out)
    assert json.load(open(out)) == json.load(open(r["cal_results"]))
    print("  calibration CLI reproduces CALIBRATION_RESULTS.json exactly")


def test_03_calibration_precommit_cli_succeeds_before_records():
    r = fresh("cli03")
    out = os.path.join(r["dir"], "CALIBRATION_PRECOMMIT.cli.json")
    planned = os.path.join(r["dir"], "not_yet_generated_calibration.jsonl")
    run_cli("calibration_precommit.py", "--output", out,
            "--manifest-design", r["manifest"], "--artifact-paths", r["artifact_paths"],
            "--calibration-task-manifest", r["cal"], "--records", planned)
    obj = json.load(open(out))
    assert obj["calibration_task_manifest_sha256"] == sha256_file(r["cal"])
    print("  calibration-precommit CLI writes only before the planned records exist")


def test_04_pregeneration_cli_preflights_and_commits():
    r = fresh("cli04")
    out = os.path.join(r["dir"], "PREGENERATION_SEAL.cli.json")
    planned = os.path.join(r["dir"], "not_yet_generated_confirmatory.jsonl")
    cp = run_cli(
        "pregeneration_seal.py", "--output", out,
        "--manifest", r["manifest"], "--provenance", r["prov"],
        "--confirmatory-task-manifest", r["conf"],
        "--calibration-task-manifest", r["cal"],
        "--calibration-precommit", r["cal_precommit"],
        "--seed-matrix", r["seeds"], "--calibration-records", r["cal_records"],
        "--calibration-metadata", r["cal_metadata"],
        "--calibration-results", r["cal_results"],
        "--axis-package", r["axis_package"], "--records", planned,
    )
    payload = json.loads(cp.stdout)
    assert payload["preflight"] == "PASS" and os.path.exists(out)
    print("  pre-generation CLI reruns every upstream preflight before committing")


def test_05_primary_cli_opens_once_on_the_committed_run():
    r = fresh("cli05")
    cp = run_cli(
        "run_o1_primary.py", "--records", r["records"],
        "--manifest", r["manifest"], "--confirmatory-task-manifest", r["conf"],
        "--calibration-task-manifest", r["cal"],
        "--calibration-precommit", r["cal_precommit"],
        "--seed-matrix", r["seeds"], "--result-seal", r["seal"],
        "--run-provenance", r["prov"], "--pregeneration-seal", r["pregen"],
        "--calibration-records", r["cal_records"],
        "--calibration-metadata", r["cal_metadata"],
        "--calibration-results", r["cal_results"],
        "--axis-package", r["axis_package"],
    )
    payload = json.loads(cp.stdout)
    assert payload["verdict"] == payload["primary"]["verdict"]
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["primary"] is not None and os.path.exists(r["seal"])
    print("  primary CLI verifies the full chain and writes one result seal")


def test_06_axis_verifier_cli_returns_sealable():
    r = fresh("cli06")
    cp = run_cli("verify_axis_artifact.py", r["axis_package"])
    payload = json.loads(cp.stdout)
    assert payload["verdict"] == "SEALABLE"
    print("  axis-verifier CLI returns SEALABLE for the canonical synthetic package")


def main() -> int:
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = 0
    for name in names:
        print(f"\n{name}")
        try:
            globals()[name]()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL: {type(exc).__name__}: {exc}")
    print(f"\n{'=' * 70}\n{len(names)-failed}/{len(names)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
