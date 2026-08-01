"""Fixtures for the frozen deterministic confirmatory-cohort builder.

Covers the preregistered allocation rule (50/50 target, waterfall,
round-robin, lexicographic remainders, hash-ordered prefixes), the pool
capacity cap, rerun determinism, and hostile mutations (out-of-population
tasks, manual stratum mixes, forged allocations).

Run:  python test_cohort_builder_fixtures.py
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_o1_fixtures as base
from build_confirmatory_cohort import build_confirmatory_cohort
from cohort_allocation import AllocationError, allocate, materialize
from o1_analysis import (
    IntegrityError,
    gate_G23_pool_membership,
    load_jsonl,
    load_task_manifest,
    sha256_file,
)

PASSED = 0
FAILED: list[str] = []


def check(name, fn):
    global PASSED
    try:
        fn()
        PASSED += 1
        print(f"PASS  {name}")
    except Exception as exc:                                       # noqa: BLE001
        FAILED.append(name)
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")


def expect(exc_type, fn, code=None):
    try:
        fn()
    except exc_type as e:
        if code is not None and isinstance(e, IntegrityError):
            assert e.code == code, f"expected {code}, got {e.code}: {e}"
        return e
    raise AssertionError(f"expected {exc_type.__name__}")


# ---------------------------------------------------------------- allocate()

def t00_two_settings_even_split():
    out = allocate({"hard_medium": ["s_a"], "medium": ["s_b"]},
                   {"s_a": 50, "s_b": 50}, 40)
    assert out["expected_counts"] == {"hard_medium": 20, "medium": 20}
    assert out["per_setting_quota"] == {"s_a": 20, "s_b": 20}


def t01_three_settings_round_robin_lexicographic():
    out = allocate({"hard_medium": ["s_b", "s_a"], "medium": ["s_c"]},
                   {"s_a": 50, "s_b": 50, "s_c": 50}, 41)
    # odd remainder -> hard_medium (21); within hm, 21 over {s_a, s_b} round
    # robin from lexicographically first: s_a gets 11, s_b gets 10
    assert out["expected_counts"] == {"hard_medium": 21, "medium": 20}
    assert out["per_setting_quota"] == {"s_a": 11, "s_b": 10, "s_c": 20}
    assert out["settings_per_stratum"]["hard_medium"] == ["s_a", "s_b"]


def t02_uneven_capacity_clamps_and_redistributes():
    out = allocate({"hard_medium": ["s_a", "s_b"], "medium": ["s_c"]},
                   {"s_a": 3, "s_b": 50, "s_c": 50}, 40)
    # hm target 20: round robin fills s_a to its 3-task cap, rest to s_b
    assert out["per_setting_quota"] == {"s_a": 3, "s_b": 17, "s_c": 20}


def t03_undersupplied_stratum_waterfalls():
    out = allocate({"hard_medium": ["s_a"], "medium": ["s_b"]},
                   {"s_a": 5, "s_b": 100}, 40)
    assert out["expected_counts"] == {"hard_medium": 5, "medium": 35}
    assert out["per_setting_quota"] == {"s_a": 5, "s_b": 35}


def t04_exact_capacity():
    out = allocate({"hard_medium": ["s_a"], "medium": ["s_b"]},
                   {"s_a": 4, "s_b": 4}, 8)
    assert out["expected_counts"] == {"hard_medium": 4, "medium": 4}


def t05_over_capacity_is_upstream_defect():
    expect(AllocationError,
           lambda: allocate({"hard_medium": ["s_a"], "medium": ["s_b"]},
                            {"s_a": 4, "s_b": 4}, 9))


def t06_no_eligible_setting_refused():
    expect(AllocationError,
           lambda: allocate({"hard_medium": [], "medium": ["s_b"]},
                            {"s_b": 10}, 5))


def t07_setting_in_both_strata_refused():
    expect(AllocationError,
           lambda: allocate({"hard_medium": ["s_a"], "medium": ["s_a"]},
                            {"s_a": 10}, 5))


def t08_materialize_prefix_and_ranks():
    pool = []
    for s, n in (("s_a", 6), ("s_b", 6)):
        for i in range(n):
            pool.append({"task_id": f"{s}_t{i}", "generator_setting_id": s,
                         "sealed_rank_within_generator_setting": i,
                         "task_content_sha256": f"c_{s}_{i}"})
    alloc = allocate({"hard_medium": ["s_a"], "medium": ["s_b"]},
                     {"s_a": 6, "s_b": 6}, 8)
    rows = materialize(pool, alloc)
    hm = [r for r in rows if r["stratum"] == "hard_medium"]
    m = [r for r in rows if r["stratum"] == "medium"]
    assert [r["task_id"] for r in hm] == ["s_a_t0", "s_a_t1", "s_a_t2", "s_a_t3"]
    assert [r["sealed_rank_within_stratum"] for r in hm] == [0, 1, 2, 3]
    assert [r["task_id"] for r in m] == ["s_b_t0", "s_b_t1", "s_b_t2", "s_b_t3"]
    # hash-ordered PREFIX only: the later ranks are untouched
    assert not any(r["task_id"].endswith("t5") for r in rows)


# ------------------------------------------------------- builder end-to-end

def _built(tag):
    r = base.write_run(base.make_clean(G=8), tag=tag)
    out = os.path.join(r["dir"], "confirmatory_tasks.built.jsonl")
    rep = os.path.join(r["dir"], "COHORT_DERIVATION_REPORT.json")
    report = build_confirmatory_cohort(
        r["manifest"], r["cal_results"], r["pool"], out, rep)
    return r, out, rep, report


def t09_builder_end_to_end_matches_sealed_expectations():
    r, out, rep, report = _built("b09")
    rows = load_jsonl(out)
    assert len(rows) == 8 and report["n_rows"] == 8
    assert report["expected_counts"] == {"hard_medium": 4, "medium": 4}
    assert report["confirmatory_task_manifest_sha256"] == sha256_file(out)
    # the built cohort is loadable by the sealed confirmatory loader and is a
    # full member of the sealed pool
    manifest_rows = load_task_manifest(out)
    gate_G23_pool_membership(manifest_rows, load_jsonl(r["pool"]))
    # builder output must agree with the fixture's independently written
    # confirmatory manifest on (task, stratum, rank)
    fixture = {(t["task_id"], t["stratum"], t["sealed_rank_within_stratum"])
               for t in load_task_manifest(r["conf"])}
    built = {(t["task_id"], t["stratum"], t["sealed_rank_within_stratum"])
             for t in manifest_rows}
    assert fixture == built


def t10_rerun_determinism_byte_identical():
    r, out, rep, _ = _built("b10")
    out2 = os.path.join(r["dir"], "confirmatory_tasks.rerun.jsonl")
    rep2 = os.path.join(r["dir"], "COHORT_DERIVATION_REPORT.rerun.json")
    build_confirmatory_cohort(r["manifest"], r["cal_results"], r["pool"],
                              out2, rep2)
    assert open(out, "rb").read() == open(out2, "rb").read()
    a = json.load(open(rep)); b = json.load(open(rep2))
    a.pop("confirmatory_task_manifest"); b.pop("confirmatory_task_manifest")
    assert a == b


def t11_refuses_overwrite():
    r, out, rep, _ = _built("b11")
    expect(IntegrityError,
           lambda: build_confirmatory_cohort(
               r["manifest"], r["cal_results"], r["pool"], out, rep),
           code="B0_COHORT_BUILD")


def t12_pool_tamper_refused():
    r = base.write_run(base.make_clean(G=8), tag="b12")
    with open(r["pool"], "a") as fh:
        fh.write(json.dumps({"task_id": "smuggled",
                             "generator_setting_id": "depth_hm",
                             "sealed_rank_within_generator_setting": 99,
                             "task_content_sha256": "smuggled"}) + "\n")
    expect(IntegrityError,
           lambda: build_confirmatory_cohort(
               r["manifest"], r["cal_results"], r["pool"],
               os.path.join(r["dir"], "x.jsonl"),
               os.path.join(r["dir"], "x.json")),
           code="B0_COHORT_BUILD")


def t13_forged_allocation_refused():
    r = base.write_run(base.make_clean(G=8), tag="b13")
    results = json.load(open(r["cal_results"]))
    results["confirmatory_allocation"]["per_setting_quota"] = {
        "depth_hm": 8, "depth_m": 0}      # a manually chosen stratum mix
    results["confirmatory_allocation"]["expected_counts"] = {
        "hard_medium": 8, "medium": 0}
    forged = os.path.join(r["dir"], "FORGED_RESULTS.json")
    json.dump(results, open(forged, "w"), indent=2, sort_keys=True)
    m = json.load(open(r["manifest"]))
    m["calibration"]["results_sha256"] = sha256_file(forged)
    json.dump(m, open(r["manifest"], "w"), indent=2)
    expect(IntegrityError,
           lambda: build_confirmatory_cohort(
               r["manifest"], forged, r["pool"],
               os.path.join(r["dir"], "y.jsonl"),
               os.path.join(r["dir"], "y.json")),
           code="B1_ALLOCATION_MISMATCH")


def t14_non_proceed_verdict_refused():
    r = base.write_run(base.make_clean(G=8), tag="b14")
    results = json.load(open(r["cal_results"]))
    results["verdict"] = "NO_COHERENT_ALPHA"
    halted = os.path.join(r["dir"], "HALTED_RESULTS.json")
    json.dump(results, open(halted, "w"), indent=2, sort_keys=True)
    m = json.load(open(r["manifest"]))
    m["calibration"]["results_sha256"] = sha256_file(halted)
    json.dump(m, open(r["manifest"], "w"), indent=2)
    expect(IntegrityError,
           lambda: build_confirmatory_cohort(
               r["manifest"], halted, r["pool"],
               os.path.join(r["dir"], "z.jsonl"),
               os.path.join(r["dir"], "z.json")),
           code="B0_COHORT_BUILD")


def t15_out_of_population_task_caught_by_G23():
    r, out, rep, _ = _built("b15")
    rows = load_jsonl(out)
    rows[0]["task_content_sha256"] = "not_in_the_pool"
    tampered = os.path.join(r["dir"], "tampered_cohort.jsonl")
    with open(tampered, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    expect(IntegrityError,
           lambda: gate_G23_pool_membership(load_task_manifest(tampered),
                                            load_jsonl(r["pool"])),
           code="G23_POOL_MEMBERSHIP")


def t16_allocation_errors_surface_as_gate_aborts():
    """Red-team regression: over-capacity / defective pools abort with a coded
    B1 gate, never an uncaught AllocationError."""
    r = base.write_run(base.make_clean(G=8), tag="b16")
    results = json.load(open(r["cal_results"]))
    results["power"]["G_confirm"] = 999          # exceeds the 8-task pool
    results["confirmatory_allocation"]["G_confirm"] = 999
    forged = os.path.join(r["dir"], "OVER_RESULTS.json")
    json.dump(results, open(forged, "w"), indent=2, sort_keys=True)
    m = json.load(open(r["manifest"]))
    m["calibration"]["results_sha256"] = sha256_file(forged)
    json.dump(m, open(r["manifest"], "w"), indent=2)
    expect(IntegrityError,
           lambda: build_confirmatory_cohort(
               r["manifest"], forged, r["pool"],
               os.path.join(r["dir"], "w.jsonl"),
               os.path.join(r["dir"], "w.json")),
           code="B1_ALLOCATION_MISMATCH")


def main() -> int:
    for name in sorted(k for k in globals() if k.startswith("t") and k[1:3].isdigit()):
        check(name, globals()[name])
    total = PASSED + len(FAILED)
    print(f"\n{'=' * 70}\ncohort builder fixtures: {PASSED}/{total} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
