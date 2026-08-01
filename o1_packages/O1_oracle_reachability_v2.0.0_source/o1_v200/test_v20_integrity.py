"""v2.0 integration fixtures: precommit, runtime, calibration, and axis binding."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from o1_analysis import (
    IntegrityError, Manifest, SealViolation, create_pregeneration_seal,
    run_primary, sha256_file, gate_G14_provenance, sha256_tree,
    required_G, budget_fallback,
)
import test_o1_fixtures as base
from build_run_provenance import build_run_provenance
from run_all_tests import verify_package_checksums, reject_bytecode_artifacts
from verify_axis_artifact import AXIS_ORDER, _mk as make_axis_package, verify
from horizon_logic_generator_v2 import generate_task, truth_table_label
from o1_answer_parser_v2 import parse_final_answer
from o1_truth_table_verifier_v2 import verify_letter
from run_o1_v2_generation import (
    LOOP_INDEX, MODULE_INDEX, PHYSICAL_LAYER, OneShotResidualEdit,
)


def expect(code, fn, *args, exc=IntegrityError):
    try:
        fn(*args)
    except exc as e:
        if isinstance(e, IntegrityError):
            assert e.code == code, f"expected {code}, got {e.code}: {e}"
        print(f"  caught: {e}")
        return e
    raise AssertionError(f"expected {exc.__name__}[{code}]")


def RP(r):
    return run_primary(
        r["records"], r["manifest"], r["conf"], r["cal"],
        r["cal_precommit"], r["seeds"],
        r["seal"], r["prov"], r["pregen"], r["cal_records"],
        r["cal_metadata"], r["cal_results"], r["axis_package"],
    )


def test_00_clean_v20_pipeline():
    r = base.write_run(base.make_clean(G=40), tag="v1500")
    out = RP(r)
    assert out["axis_verification"]["verdict"] == "SEALABLE"
    assert out["pregeneration_seal_sha256"] == sha256_file(r["pregen"])
    print("  preseal + runtime + calibration + axis package verified before primary")


def test_01_manifest_changed_after_precommit():
    r = base.write_run(base.make_clean(G=8), tag="v1501")
    m = json.load(open(r["manifest"])); m["decoding"]["answer_marker"] = "CHANGED"
    json.dump(m, open(r["manifest"], "w"), indent=2)
    expect("G18_PREGENERATION_SEAL", RP, r)


def test_02_calibration_changed_after_precommit():
    r = base.write_run(base.make_clean(G=8), tag="v1502")
    with open(r["cal_results"], "a") as fh:
        fh.write("\n")
    expect("G18_PREGENERATION_SEAL", RP, r)


def test_03_axis_package_changed_after_precommit():
    r = base.write_run(base.make_clean(G=8), tag="v1503")
    with open(os.path.join(r["axis_package"], "AXIS_MANIFEST.json"), "a") as fh:
        fh.write("\n")
    expect("G18_PREGENERATION_SEAL", RP, r)


def test_04_preseal_refuses_existing_records():
    r = base.write_run(base.make_clean(G=8), tag="v1504")
    out = os.path.join(r["dir"], "SECOND_PRESEAL.json")
    expect("", create_pregeneration_seal,
           out, r["manifest"], r["prov"], r["conf"], r["cal"],
           r["cal_precommit"], r["seeds"], r["cal_records"],
           r["cal_metadata"], r["cal_results"],
           r["axis_package"], r["records"], exc=SealViolation)


def test_05_extra_runtime_configuration_surface_rejected():
    r = base.write_run(base.make_clean(G=8), tag="v1505")
    p = json.load(open(r["prov"])); p["shadow_temperature"] = 1.3
    json.dump(p, open(r["prov"], "w"), indent=2)
    os.remove(r["pregen"])
    create_pregeneration_seal(r["pregen"], r["manifest"], r["prov"],
                              r["conf"], r["cal"], r["cal_precommit"], r["seeds"],
                              r["cal_records"], r["cal_metadata"],
                              r["cal_results"], r["axis_package"])
    expect("G14_PROVENANCE", RP, r)


def test_06_dirty_worktree_attestation_rejected():
    r = base.write_run(base.make_clean(G=8), tag="v1506")
    p = json.load(open(r["prov"])); p["runtime_attestation"]["repository_clean"] = False
    json.dump(p, open(r["prov"], "w"), indent=2)
    os.remove(r["pregen"])
    create_pregeneration_seal(r["pregen"], r["manifest"], r["prov"],
                              r["conf"], r["cal"], r["cal_precommit"], r["seeds"],
                              r["cal_records"], r["cal_metadata"],
                              r["cal_results"], r["axis_package"])
    expect("G14_PROVENANCE", RP, r)


def test_07_required_power_fields_are_not_optional():
    r = base.write_run(base.make_clean(G=8), tag="v1507")
    m = json.load(open(r["manifest"])); del m["power"]["q_disc_upper_confidence_bound"]
    json.dump(m, open(r["manifest"], "w"), indent=2)
    expect("G0_MANIFEST_INCOMPLETE", Manifest, r["manifest"])


def _builder_fixture(tag: str):
    r = base.write_run(base.make_clean(G=8), tag=tag)
    return r, r["artifact_paths"]


def test_08_provenance_builder_hashes_real_artifacts_and_emits_canonical_config():
    r, paths = _builder_fixture("v1508")
    out_path = os.path.join(r["dir"], "BUILT_RUN_PROVENANCE.json")
    report = build_run_provenance(r["manifest"], paths, out_path)
    assert report["sha256"] == sha256_file(out_path)
    prov = gate_G14_provenance(Manifest(r["manifest"]), out_path)
    assert prov["runtime_attestation"]["repository_clean"] is True
    print("  provenance built only after hashing all eight real runtime artifacts")


def test_09_provenance_builder_rejects_artifact_mismatch():
    r, paths = _builder_fixture("v1509")
    pmap = json.load(open(paths))
    with open(pmap["artifact_hashes.parser"], "ab") as fh:
        fh.write(b"tamper")
    expect("P2_ARTIFACT_HASH", build_run_provenance,
           r["manifest"], paths, os.path.join(r["dir"], "BAD_PROVENANCE.json"))


def test_10_required_G_infeasible_returns_machine_readable_sentinel():
    assert required_G(0.0, 0.05) == -2
    assert required_G(0.03, 0.05) == -2
    decision = budget_fallback(0.03, G_max=100, delta_target=0.05)
    assert decision.G_power == -2
    assert decision.G_confirm == 0
    assert decision.q_disc_basis == "target excluded by discordance upper bound"
    print("  infeasible effect target returns -2 sentinel; budget planning halts at G=0")


def test_11_sha256sums_detects_modified_package_file():
    src = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "package"
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        with (dst / "o1_analysis.py").open("a") as fh:
            fh.write("\n# adversarial post-package mutation\n")
        try:
            verify_package_checksums(dst)
        except RuntimeError as exc:
            assert "mismatch for o1_analysis.py" in str(exc), exc
            print(f"  caught: {exc}")
        else:
            raise AssertionError("modified package file passed SHA256SUMS verification")


def test_12_packaged_bytecode_is_forbidden_even_when_source_metadata_matches():
    src = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "package"
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        cache = dst / "__pycache__"
        cache.mkdir()
        # The payload need not be valid: the package rule is categorical because
        # timestamp/size-valid bytecode can shadow reviewed source.
        (cache / "o1_analysis.cpython-313.pyc").write_bytes(b"forged-bytecode")
        try:
            reject_bytecode_artifacts(dst)
        except RuntimeError as exc:
            assert "forbidden executable bytecode artifacts" in str(exc), exc
            print(f"  caught: {exc}")
        else:
            raise AssertionError("packaged bytecode passed the no-bytecode gate")


def test_13_v2_axis_semantics_and_embedded_mean_sources_are_bound():
    assert AXIS_ORDER == (
        "A1_READABLE",
        "A2_WRITABLE",
        "A3_CAUSAL_MEAN",
        "A4_SEQUENCE_MEAN",
    )
    with tempfile.TemporaryDirectory() as td:
        package = Path(td) / "axis"
        make_axis_package(str(package))
        report = verify(str(package))
        assert report["verdict"] == "SEALABLE"
        assert "A3/A4 exact uncentered mean construction from embedded source matrices" in (
            report["structurally_verified"]
        )
        assert not (package / "d4_components_l3_24.npy").exists()
        assert (package / "causal_induced_updates_l3_24.npy").is_file()
        assert (package / "sequence_induced_updates_l3_24.npy").is_file()
        print("  v2 bank binds distinct causal/sequence means; no shared-PC artifact")


def test_14_frozen_generator_parser_verifier_and_physical_boundary():
    import numpy as np
    import torch

    task = generate_task("calibration", "rules_3", 4)
    assert task == generate_task("calibration", "rules_3", 4)
    assert task != generate_task("confirmatory_candidate", "rules_3", 4)
    assert task["rule_count"] == 3
    assert truth_table_label(task["symbolic"]) == task["gold_answer"]
    letter = chr(65 + task["answer_key"])
    assert verify_letter(task, letter)
    assert parse_final_answer(f"reasoning\nFINAL ANSWER: {letter}.") == letter
    assert LOOP_INDEX == 2 and PHYSICAL_LAYER == 24 and MODULE_INDEX == 23

    hidden = torch.randn(1, 7, 2048)
    zero = OneShotResidualEdit(np.ones(2048), 0.0)
    output = zero(None, (), {"current_ut": 2}, hidden)
    assert output is hidden
    assert zero.applied == 1 and zero.injected_rms == 0.0
    off_loop = OneShotResidualEdit(np.ones(2048), 0.1)
    assert off_loop(None, (), {"current_ut": 1}, hidden) is hidden
    assert off_loop.applied == 0
    print("  frozen tasks are disjoint; verifier agrees; physical 24 maps to module 23")


def main() -> int:
    names = [k for k in sorted(globals()) if k.startswith("test_")]
    failed = 0
    for name in names:
        print(f"\n{name}")
        try:
            globals()[name]()
        except Exception as exc:
            failed += 1
            print(f"  FAIL: {type(exc).__name__}: {exc}")
    print(f"\n{'=' * 70}\n{len(names)-failed}/{len(names)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
