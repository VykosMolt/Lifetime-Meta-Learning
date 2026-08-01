"""Deterministic fixtures for calibration_analysis.py, v2.1."""
from __future__ import annotations

import copy
import json
import os
import tempfile

from calibration_precommit import create_calibration_precommit
from calibration_analysis import (
    CalibrationError, apply_results_to_manifest, compute_calibration,
    verify_calibration_binding, write_results,
)
from o1_analysis import sha256_file
import test_o1_fixtures as base


def expect(code, fn, *args):
    try:
        fn(*args)
    except CalibrationError as e:
        assert e.code == code, f"expected {code}, got {e.code}: {e}"
        print(f"  caught: {e}")
        return e
    raise AssertionError(f"expected CalibrationError[{code}]")


def fresh(tag="cal"):
    return base.write_run(base.make_clean(G=8), tag=tag)


def recommit_calibration(r, manifest_obj):
    json.dump(manifest_obj, open(r["manifest"], "w"), indent=2)
    if os.path.exists(r["cal_precommit"]):
        os.remove(r["cal_precommit"])
    create_calibration_precommit(
        r["cal_precommit"], r["manifest"], r["artifact_paths"], r["cal"],
        records_path=None)
    prehash = sha256_file(r["cal_precommit"])
    rows = [json.loads(x) for x in open(r["cal_records"])]
    for row in rows:
        row["calibration_precommit_sha256"] = prehash
    with open(r["cal_records"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = json.load(open(r["cal_metadata"]))
    meta["calibration_precommit_sha256"] = prehash
    json.dump(meta, open(r["cal_metadata"], "w"))
    return prehash


def test_00_clean_recomputation_matches_stored():
    r = fresh("c00")
    out = verify_calibration_binding(
        r["cal_records"], r["cal_metadata"], r["cal_results"],
        r["manifest"], r["cal"], r["cal_precommit"], r["pool"],
    )
    assert out["verdict"] == "PROCEED_POWERED"
    assert out["alpha"]["alpha_star"] == 0.02
    assert out["alpha"]["passing_alphas"] == [0.005, 0.01, 0.02]
    print("  clean calibration recomputes byte-for-byte")


def test_01_raw_record_tamper_breaks_results():
    r = fresh("c01")
    rows = [json.loads(x) for x in open(r["cal_records"])]
    rows[0]["verifier_correct"] = not rows[0]["verifier_correct"]
    with open(r["cal_records"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    # v2.1: a bare boolean flip is caught by scoring recomputation, before it
    # can even reach the byte-comparison of results.
    expect("C10_TEXT_RECOMPUTATION", verify_calibration_binding,
           r["cal_records"], r["cal_metadata"], r["cal_results"],
           r["manifest"], r["cal"], r["cal_precommit"], r["pool"])
    # a text-consistent forgery (text AND booleans rewritten together) passes
    # C10 but still breaks the deterministic results byte comparison
    r2 = fresh("c01b")
    rows = [json.loads(x) for x in open(r2["cal_records"])]
    base._set_scoring(rows[0], True, not rows[0]["verifier_correct"])
    with open(r2["cal_records"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    expect("C8_RESULTS_MISMATCH", verify_calibration_binding,
           r2["cal_records"], r2["cal_metadata"], r2["cal_results"],
           r2["manifest"], r2["cal"], r2["cal_precommit"], r2["pool"])


def test_02_stored_result_fabrication_rejected():
    r = fresh("c02")
    obj = json.load(open(r["cal_results"]))
    obj["alpha"]["alpha_star"] = 0.08
    json.dump(obj, open(r["cal_results"], "w"), indent=2)
    expect("C8_RESULTS_MISMATCH", verify_calibration_binding,
           r["cal_records"], r["cal_metadata"], r["cal_results"],
           r["manifest"], r["cal"], r["cal_precommit"], r["pool"])


def test_03_manifest_copy_error_rejected():
    r = fresh("c03")
    m = json.load(open(r["manifest"]))
    m["power"]["q_disc_calibrated"] = 0.999
    json.dump(m, open(r["manifest"], "w"), indent=2)
    expect("C9_MANIFEST_BINDING", verify_calibration_binding,
           r["cal_records"], r["cal_metadata"], r["cal_results"],
           r["manifest"], r["cal"], r["cal_precommit"], r["pool"])


def test_04_incomplete_alpha_bank_rejected():
    r = fresh("c04")
    rows = [json.loads(x) for x in open(r["cal_records"])]
    dropped = False
    keep = []
    for row in rows:
        if (not dropped and row["task_id"] == "cal_0000" and
                row["arm"] == "structured" and row["alpha"] == 0.02):
            dropped = True
            continue
        keep.append(row)
    with open(r["cal_records"], "w") as fh:
        for row in keep:
            fh.write(json.dumps(row) + "\n")
    expect("C2_GEOMETRY", compute_calibration,
           r["cal_records"], r["cal_metadata"], r["manifest"], r["cal"],
           r["cal_precommit"], r["pool"])


def test_05_task_manifest_binding_rejected():
    r = fresh("c05")
    rows = [json.loads(x) for x in open(r["cal"])]
    rows[0]["task_content_sha256"] = "fabricated_content"
    with open(r["cal"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    expect("C0_PRECOMMIT", compute_calibration,
           r["cal_records"], r["cal_metadata"], r["manifest"], r["cal"],
           r["cal_precommit"], r["pool"])


def test_06_invalid_elapsed_time_rejected():
    r = fresh("c06")
    json.dump({"elapsed_wall_seconds": 0, "calibration_precommit_sha256": sha256_file(r["cal_precommit"])}, open(r["cal_metadata"], "w"))
    expect("C6_METADATA", compute_calibration,
           r["cal_records"], r["cal_metadata"], r["manifest"], r["cal"],
           r["cal_precommit"], r["pool"])


def test_07_no_difficulty_settings_is_explicit():
    r = fresh("c07")
    rows = [json.loads(x) for x in open(r["cal_records"])]
    # Make every baseline bank correct; p_base=1.0 falls outside both bands.
    for row in rows:
        if row["arm"] == "baseline":
            base._set_scoring(row, True, row["stream_index"] == 0)
    with open(r["cal_records"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    out = compute_calibration(r["cal_records"], r["cal_metadata"],
                              r["manifest"], r["cal"], r["cal_precommit"],
                              r["pool"])
    assert out["verdict"] == "NO_DIFFICULTY_SETTINGS"
    print("  no qualifying generator settings -> NO_DIFFICULTY_SETTINGS")


def test_08_discordance_bound_can_halt_without_crash():
    r = fresh("c08")
    m = json.load(open(r["manifest"]))
    # Replace the calibration cohort with 80 tasks and no bank-level discordance.
    with open(r["cal"], "w") as fh:
        for i in range(80):
            st = "hard_medium" if i < 40 else "medium"
            fh.write(json.dumps({
                "task_id": f"cal_{i:04d}", "stratum": st,
                "generator_setting_id": "depth_hm" if st == "hard_medium" else "depth_m",
                "sealed_rank_within_stratum": i if i < 40 else i - 40,
                "task_content_sha256": f"cal_content_{i}",
                "options": base.FIXTURE_OPTIONS,
                "symbolic": base.FIXTURE_SYMBOLIC,
            }) + "\n")
    # Generate raw records using the helper, then force structured outcomes to
    # match their task's baseline oracle exactly.
    cal_rec, cal_meta = base._write_calibration(r["dir"], m, r["cal_precommit"], G_cal=80)
    rows = [json.loads(x) for x in open(cal_rec)]
    base_y = {}
    for row in rows:
        if row["arm"] == "baseline":
            base_y.setdefault(row["task_id"], False)
            base_y[row["task_id"]] |= bool(row["well_formed"] and row["verifier_correct"])
    first = set()
    for row in rows:
        if row["arm"] == "structured" and row["alpha"] == 0.02:
            key = row["task_id"]
            vc = bool(base_y[key] and key not in first)
            base._set_scoring(row, True, vc)
            if vc:
                first.add(key)
    with open(cal_rec, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    m["cohorts"]["calibration_task_manifest_sha256"] = sha256_file(r["cal"])
    # Recreate precommit for the enlarged sealed cohort, then bind new raw rows.
    json.dump(m, open(r["manifest"], "w"), indent=2)
    os.remove(r["cal_precommit"])
    create_calibration_precommit(r["cal_precommit"], r["manifest"],
                                 r["artifact_paths"], r["cal"])
    prehash = sha256_file(r["cal_precommit"])
    for row in rows:
        row["calibration_precommit_sha256"] = prehash
    with open(cal_rec, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = json.load(open(cal_meta))
    meta["calibration_precommit_sha256"] = prehash
    json.dump(meta, open(cal_meta, "w"))
    out = compute_calibration(cal_rec, cal_meta, r["manifest"], r["cal"],
                              r["cal_precommit"], r["pool"])
    assert out["discordance"]["n_disc"] == 0
    assert out["discordance"]["q_disc_upper_confidence_bound"] < 0.05
    assert out["verdict"] == "TARGET_HEADROOM_EXCLUDED_BY_DISCORDANCE_BOUND"
    assert out["power"]["G_confirm"] == 0
    print(f"  q_upper={out['discordance']['q_disc_upper_confidence_bound']:.4f} -> explicit halt")


def test_09_results_file_is_immutable():
    r = fresh("c09")
    out = compute_calibration(r["cal_records"], r["cal_metadata"],
                              r["manifest"], r["cal"], r["cal_precommit"],
                              r["pool"])
    expect("C7_IMMUTABLE", write_results, out, r["cal_results"])


def test_10_budget_drops_only_the_first_optional_arm_when_that_restores_power():
    r = fresh("c10")
    m = json.load(open(r["manifest"]))
    m["bank"]["arms_required"] = ["baseline", "structured", "zero_alpha"]
    m["bank"]["arms_conditional_on_budget"] = ["random", "structured_secondary"]
    m["power"]["target_power"] = 0.80
    # v2.1: the candidate pool must cover the powered G, otherwise the pool
    # cap (correctly) forces PROCEED_BUDGET_CONSTRAINED before arm planning.
    big_pool = os.path.join(r["dir"], "big_pool.jsonl")
    with open(big_pool, "w") as fh:
        for i in range(4000):
            setting = "depth_hm" if i % 2 == 0 else "depth_m"
            fh.write(json.dumps({
                "task_id": f"pool_{i:05d}",
                "generator_setting_id": setting,
                "sealed_rank_within_generator_setting": i // 2,
                "task_content_sha256": f"pool_content_{i}",
                "options": base.FIXTURE_OPTIONS,
                "symbolic": base.FIXTURE_SYMBOLIC}) + "\n")
    r["pool"] = big_pool
    m["cohorts"]["confirmatory_candidate_pool_sha256"] = sha256_file(big_pool)
    # First precommit the changed design, then compute q_upper and powered G.
    recommit_calibration(r, m)
    probe = compute_calibration(r["cal_records"], r["cal_metadata"],
                                r["manifest"], r["cal"], r["cal_precommit"],
                                r["pool"])
    g_power = probe["power"]["G_power"]
    assert g_power and g_power > 0
    # Candidate budget: insufficient for five arms, sufficient after dropping
    # only the first optional arm (random), while retaining secondary alpha.
    available_hours = (m["budget"]["max_wall_clock_days"] * 24
                       * (1 - m["budget"]["rerun_reserve_fraction"]))
    desired_candidates = (g_power + 1) * 8 * 4
    throughput = desired_candidates / available_hours
    n_candidates = sum(1 for line in open(r["cal_records"]) if line.strip())
    elapsed = n_candidates / throughput * 3600
    json.dump({"elapsed_wall_seconds": elapsed, "calibration_precommit_sha256": sha256_file(r["cal_precommit"])}, open(r["cal_metadata"], "w"))
    out = compute_calibration(r["cal_records"], r["cal_metadata"],
                              r["manifest"], r["cal"], r["cal_precommit"],
                              r["pool"])
    full_key = "baseline|structured|zero_alpha|random|structured_secondary"
    four_key = "baseline|structured|zero_alpha|structured_secondary"
    assert out["throughput"]["capacity_by_arm_plan"][full_key] < g_power
    assert out["throughput"]["capacity_by_arm_plan"][four_key] >= g_power
    assert out["power"]["selected_confirmatory_arms"] == [
        "baseline", "structured", "zero_alpha", "structured_secondary"]
    assert out["power"]["dropped_arms"] == ["random"]
    assert out["verdict"] == "PROCEED_POWERED"
    print(f"  full five-arm capacity insufficient; dropped random only, G={g_power}")


def test_11_raw_task_content_mismatch_is_rejected_after_precommit_passes():
    r = fresh("c11")
    rows = [json.loads(x) for x in open(r["cal_records"])]
    for row in rows:
        if row["task_id"] == "cal_0000":
            row["task_content_sha256"] = "fabricated_content"
    with open(r["cal_records"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    expect("C3_TASK_BINDING", compute_calibration,
           r["cal_records"], r["cal_metadata"], r["manifest"],
           r["cal"], r["cal_precommit"], r["pool"])


def test_12_calibration_precommit_refuses_existing_records():
    r = fresh("c12")
    out = os.path.join(r["dir"], "SECOND_CALIBRATION_PRECOMMIT.json")
    expect("C0_PRECOMMIT", create_calibration_precommit,
           out, r["manifest"], r["artifact_paths"], r["cal"],
           None, r["cal_records"])


def test_13_calibration_design_change_after_precommit_is_rejected():
    r = fresh("c13")
    m = json.load(open(r["manifest"]))
    m["decoding"]["top_p"] = 0.80
    json.dump(m, open(r["manifest"], "w"), indent=2)
    expect("C0_PRECOMMIT", compute_calibration,
           r["cal_records"], r["cal_metadata"], r["manifest"],
           r["cal"], r["cal_precommit"], r["pool"])


def test_14_calibration_rows_are_bound_to_precommit():
    r = fresh("c14")
    rows = [json.loads(x) for x in open(r["cal_records"])]
    rows[0]["calibration_precommit_sha256"] = "0" * 64
    with open(r["cal_records"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    expect("C0_PRECOMMIT", compute_calibration,
           r["cal_records"], r["cal_metadata"], r["manifest"],
           r["cal"], r["cal_precommit"], r["pool"])


def test_15_calibration_seed_cannot_be_rechosen():
    r = fresh("c15")
    rows = [json.loads(x) for x in open(r["cal_records"])]
    task = rows[0]["task_id"]
    stream = rows[0]["stream_index"]
    for row in rows:
        if row["task_id"] == task and row["stream_index"] == stream:
            row["seed"] = (row["seed"] + 1) % (2**64)
    with open(r["cal_records"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    expect("C4_CRN", compute_calibration,
           r["cal_records"], r["cal_metadata"], r["manifest"],
           r["cal"], r["cal_precommit"], r["pool"])


def test_16_apply_results_copies_calibration_precommit_hash():
    r = fresh("c16")
    out = os.path.join(r["dir"], "APPLIED_MANIFEST.json")
    apply_results_to_manifest(r["manifest"], r["cal_results"], out)
    applied = json.load(open(out))
    results = json.load(open(r["cal_results"]))
    assert applied["calibration"]["precommit_sha256"] == results["source_hashes"]["precommit_sha256"]
    assert applied["calibration"]["precommit_sha256"] == sha256_file(r["cal_precommit"])
    print("  applied manifest binds the exact calibration precommit hash")


def test_17_band_edges_half_open():
    """Sealed half-open bands: 0.40 in hard_medium, 0.55 in medium, 0.70 out."""
    from calibration_analysis import _difficulty_assignment

    def mk(setting, n_correct, n=20):
        tasks = {}
        for i in range(n):
            tasks[f"{setting}_{i}"] = {
                "setting": setting,
                "baseline": [{"well_formed": True,
                              "verifier_correct": bool(i < n_correct and j == 0)}
                             for j in range(8)],
            }
        return tasks

    tasks = {}
    tasks.update(mk("edge_040", 8))     # 8/20  = 0.40 exactly
    tasks.update(mk("edge_055", 11))    # 11/20 = 0.55 exactly
    tasks.update(mk("edge_070", 14))    # 14/20 = 0.70 exactly
    bands = {"hard_medium": [0.4, 0.55], "medium": [0.55, 0.7]}
    selected, metrics = _difficulty_assignment(tasks, bands)
    assert metrics["edge_040"]["baseline_any_correct_of_8"] == 0.4
    assert "edge_040" in selected["hard_medium"]
    assert "edge_055" in selected["medium"]
    assert "edge_055" not in selected["hard_medium"]
    assert "edge_070" not in selected["medium"]
    assert "edge_070" not in selected["hard_medium"]
    print("  p=0.40 -> hard_medium; p=0.55 -> medium only; p=0.70 -> neither")


def test_18_pool_capacity_constrained_is_declared():
    r = fresh("c18")
    m = json.load(open(r["manifest"]))
    m["power"]["target_power"] = 0.80
    recommit_calibration(r, m)
    out = compute_calibration(r["cal_records"], r["cal_metadata"],
                              r["manifest"], r["cal"], r["cal_precommit"],
                              r["pool"])
    assert out["verdict"] == "PROCEED_BUDGET_CONSTRAINED", out["verdict"]
    assert out["power"]["G_pool"] == 8
    assert out["power"]["G_confirm"] == 8
    assert "pool" in out["power"]["constrained_by"]
    alloc = out["confirmatory_allocation"]
    assert alloc["pool_capacity_status"] == "POOL_CAPACITY_CONSTRAINED"
    assert alloc["expected_counts"] == {"hard_medium": 4, "medium": 4}
    assert out["power"]["budget_mde_if_constrained"] is None
    assert out["power"]["mde_status"] == "NO_DETECTABLE_EFFECT_AT_G"
    print(f"  G_power={out['power']['G_power']} > pool 8 -> declared "
          f"POOL_CAPACITY_CONSTRAINED, mde=NO_DETECTABLE_EFFECT_AT_G")


def test_19_budget_mde_infeasible_state():
    from o1_analysis import budget_mde
    assert budget_mde(10, 0.30) is None
    assert budget_mde(12, 0.10) is None
    v = budget_mde(817, 0.25)
    assert v is not None and abs(v - 0.05) < 0.002, v
    print("  small-G regime returns None, not q_disc dressed as an MDE")


def test_20_real_schema_calibration_manifest_loads():
    """v2.0 latent deadlock: the REAL calibration cohort (placeholder stratum,
    per-setting ranks) was routed through the confirmatory loader."""
    from o1_analysis import (IntegrityError, load_calibration_task_manifest,
                             load_task_manifest)
    d = tempfile.mkdtemp(dir=base._TMP, prefix="realcal_")
    p = os.path.join(d, "calibration_tasks.jsonl")
    with open(p, "w") as fh:
        for i in range(3):
            fh.write(json.dumps({
                "task_id": f"o1v2-task{i}",
                "task_content_sha256": f"{i:064d}",
                "generator_setting_id": "rules_2",
                "sealed_rank_within_generator_setting": i,
                "stratum": "CALIBRATION_SETTING",
                "options": base.FIXTURE_OPTIONS,
                "symbolic": base.FIXTURE_SYMBOLIC}) + "\n")
    rows = load_calibration_task_manifest(p)
    assert len(rows) == 3
    try:
        load_task_manifest(p)
    except IntegrityError as e:
        print(f"  confirmatory loader rejects the real calibration schema "
              f"({e.code}); dedicated loader accepts it")
    else:
        raise AssertionError("load_task_manifest accepted CALIBRATION_SETTING")


def test_21_forged_repetition_flag_rejected():
    """Hiding a genuine repetition failure could flip alpha_star; C10 recomputes
    the mechanical flag from token ids."""
    r = fresh("c21")
    rows = [json.loads(x) for x in open(r["cal_records"])]
    for row in rows:
        if (row["arm"] == "structured" and row["alpha"] == 0.08
                and row["catastrophic_repetition"]):
            row["catastrophic_repetition"] = False
            break
    with open(r["cal_records"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    expect("C10_TEXT_RECOMPUTATION", compute_calibration,
           r["cal_records"], r["cal_metadata"], r["manifest"],
           r["cal"], r["cal_precommit"], r["pool"])


def test_22_implausible_throughput_rejected():
    """Red-team regression: fabricated tiny elapsed must not inflate G_max."""
    r = fresh("c22")
    json.dump({"elapsed_wall_seconds": 1e-3,
               "calibration_precommit_sha256": sha256_file(r["cal_precommit"])},
              open(r["cal_metadata"], "w"))
    e = expect("C6_METADATA", compute_calibration,
               r["cal_records"], r["cal_metadata"], r["manifest"],
               r["cal"], r["cal_precommit"], r["pool"])
    assert "plausibility ceiling" in str(e)


def test_23_zero_time_capacity_is_declared_not_crashed():
    """Red-team regression: throughput collapse yields NO_TIME_CAPACITY, not an
    uncaught AllocationError or a proceed verdict with G_confirm=0."""
    r = fresh("c23")
    json.dump({"elapsed_wall_seconds": 1e12,
               "calibration_precommit_sha256": sha256_file(r["cal_precommit"])},
              open(r["cal_metadata"], "w"))
    out = compute_calibration(r["cal_records"], r["cal_metadata"],
                              r["manifest"], r["cal"], r["cal_precommit"],
                              r["pool"])
    assert out["verdict"] == "NO_TIME_CAPACITY", out["verdict"]
    assert out["confirmatory_allocation"] is None
    assert out["power"]["budget_mde_if_constrained"] is None
    assert out["power"]["mde_status"] == "NO_TIME_CAPACITY"


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
