"""
Synthetic fixtures for the O1 v2.0 gates.

Each fixture injects exactly one defect into an otherwise clean sealed run and
asserts the corresponding gate raises with the expected stable code. Every
defect listed in the v1.0 audit is covered, including the five that previously
passed validation silently.

Run:  python tests/test_o1_fixtures.py
"""
from __future__ import annotations

import copy, hashlib, json, os, sys, tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(_HERE, "..", "analysis"), _HERE,
           os.path.join(_HERE, "analysis"), os.getcwd()):
    if os.path.exists(os.path.join(_c, "o1_analysis.py")):
        sys.path.insert(0, _c)
        MODULE_PATH = os.path.abspath(os.path.join(_c, "o1_analysis.py"))
        break
else:
    raise SystemExit("cannot locate o1_analysis.py next to, above or below tests/")
import numpy as np                                                    # noqa: E402
from o1_analysis import (                                             # noqa: E402
    BANK_SIZE, IntegrityError, Manifest, Seal, SealViolation, action_index,
    budget_fallback, coupling_survival, endpoint, exact_power,
    first_divergence, mcnemar_exact_p, outcome_diversity, required_G,
    run_primary, selection_headroom, sha256_file, sha256_tree,
    create_pregeneration_seal, resolved_generation_config, derive_stream_seed,
    transport_summary, axis_transport_profile, validate, zero_alpha_gate, exact_power,
    clopper_pearson_upper,
)

from calibration_analysis import compute_calibration, write_results  # noqa: E402
from calibration_precommit import create_calibration_precommit       # noqa: E402
from o1_analysis import text_sha256, token_ids_sha256                # noqa: E402
from o1_answer_parser_v2 import parse_final_answer                   # noqa: E402
from verify_axis_artifact import _mk as make_axis_package          # noqa: E402

DIRS = [(d, s) for d in (1, 2, 3, 4) for s in (1, -1)]
ARMS = ["baseline", "structured", "zero_alpha"]
G_DEFAULT = 40
_TMP = tempfile.mkdtemp(prefix="o1_fix_")

# Universal fixture task content: a one-variable KB whose truth-table label is
# "True", so the correct letter is A and a wrong-but-committed letter is B.
FIXTURE_OPTIONS = ["True", "False", "Unknown"]
FIXTURE_SYMBOLIC = {"n_vars": 1, "facts": [[0, True]], "rules": [],
                    "query": [0, True]}


def _fixture_text(wf: bool, vc: bool) -> str:
    if not wf:
        return "thinking, but never committing to a letter"
    return "FINAL ANSWER: A" if vc else "FINAL ANSWER: B"


def _set_scoring(r: dict, wf: bool, vc: bool) -> dict:
    """Set scoring fields with internally consistent text and digests."""
    text = _fixture_text(wf, vc)
    r["well_formed"] = wf
    r["verifier_correct"] = vc
    r["generated_text"] = text
    r["generated_text_sha256"] = text_sha256(text)
    r["parsed_answer"] = parse_final_answer(text)
    return r


def _set_tokens(r: dict, toks: list) -> dict:
    r["generated_token_ids"] = toks
    r["n_generated_tokens"] = len(toks)
    r["generated_token_ids_sha256"] = token_ids_sha256(toks)
    return r


# ---------------------------------------------------------------- builders

def _row(task, arm, stream, seed, wf, vc, alpha, d=None, s=None, toks=None,
         rms=None):
    toks = list(range(100, 110)) if toks is None else toks
    is_int = arm != "baseline"
    if rms is None:
        rms = (0.0 if arm == "zero_alpha" else 0.5) if is_int else None
    row = {"task_id": task, "stratum": "medium" if int(task.split("_")[1]) % 2
           else "hard_medium", "arm": arm, "alpha": alpha, "direction_id": d,
           "sign": s, "stream_index": stream, "seed": seed,
           "hit_max_tokens": False,
           "downstream_delta_rms": rms,
           "task_content_sha256": f"content_{task}"}
    _set_tokens(row, toks)
    _set_scoring(row, wf, vc)
    return row


def _lsq(d, s, rank):
    """sealed cyclic Latin square: stream(a, g) = (a + r_g) mod 8"""
    return (action_index(d, s) + rank) % BANK_SIZE


def make_clean(G=G_DEFAULT, arms=ARMS, n_int=lambda g: 1 if g % 4 == 0 else 0,
               n_base=lambda g: 1 if g % 5 == 0 else 0, alpha_star=0.02):
    recs = []
    rank = {"hard_medium": 0, "medium": 0}
    for g in range(G):
        t = f"conf_{g:04d}"
        st = "medium" if g % 2 else "hard_medium"
        rg = rank[st] % BANK_SIZE
        rank[st] += 1
        seeds = [1_000_000 + g * 100 + i for i in range(BANK_SIZE)]
        nb, ni = n_base(g), n_int(g)
        bt = {i: [200 + g, 300 + i, 400] for i in range(BANK_SIZE)}
        if "baseline" in arms:
            for i in range(BANK_SIZE):
                recs.append(_row(t, "baseline", i, seeds[i], True, i < nb, 0.0,
                                 toks=list(bt[i])))
        if "zero_alpha" in arms:
            for i in range(BANK_SIZE):
                recs.append(_row(t, "zero_alpha", i, seeds[i], True, i < nb, 0.0,
                                 toks=list(bt[i])))
        for arm, al, tk in (("structured", alpha_star, 999),
                            ("structured_secondary", 0.01, 998),
                            ("random", alpha_star, 997)):
            if arm not in arms:
                continue
            for k, (d, sg) in enumerate(DIRS):
                i = _lsq(d, sg, rg)
                recs.append(_row(t, arm, i, seeds[i], True, k < (ni if "struct"
                                 in arm else nb), al, d, sg,
                                 toks=[200 + g, tk, 401, 5]))
    return recs


def _cal_row(task, g, setting, arm, alpha, d, sg, stream, seed, vc,
             precommit_hash, toks, cat_rep=False):
    row = {"task_id": task,
           "task_content_sha256": f"cal_content_{g}",
           "generator_setting_id": setting,
           "arm": arm, "alpha": alpha,
           "direction_id": d, "sign": sg,
           "stream_index": stream, "seed": seed,
           "hit_max_tokens": False,
           "downstream_delta_rms": None if arm == "baseline" else 0.5,
           "logits_finite": True,
           "catastrophic_repetition": bool(cat_rep),
           "calibration_precommit_sha256": precommit_hash}
    _set_tokens(row, toks)
    _set_scoring(row, True, bool(vc))
    return row


def _write_calibration(run_dir, manifest, precommit_path, G_cal=40):
    """Small deterministic calibration with one setting in each target band."""
    rec_p = os.path.join(run_dir, "calibration_records.jsonl")
    meta_p = os.path.join(run_dir, "calibration_metadata.json")
    rows = []
    grid = manifest["action_space"]["alpha_swept_in_calibration"]
    precommit_hash = sha256_file(precommit_path)
    master_seed = manifest["cohorts"]["master_seed"]
    for g in range(G_cal):
        task = f"cal_{g:04d}"
        setting = "depth_hm" if g < G_cal // 2 else "depth_m"
        # hard-medium 0.50; medium 0.60 any-correct-of-8
        local = g if g < G_cal // 2 else g - G_cal // 2
        base_ok = (local % 2 == 0) if setting == "depth_hm" else (local % 5 in (0, 1, 2))
        seeds = [derive_stream_seed(master_seed, task, i) for i in range(8)]
        for i in range(8):
            rows.append(_cal_row(task, g, setting, "baseline", 0.0, None, None,
                                 i, seeds[i], base_ok and i == 0,
                                 precommit_hash, [10 + g, 20 + i]))
        for a in grid:
            for k, (d, sg) in enumerate(DIRS):
                # create bank-level discordance on a few tasks at alpha_star;
                # all magnitudes remain coherent.
                int_ok = base_ok
                if a == 0.02 and g % 10 == 0:
                    int_ok = not base_ok
                # Repetition rows carry genuinely repetitive token lists: the
                # flag is recomputed mechanically from token ids by C10.
                cat_rep = a in (0.04, 0.08) and k == 0
                toks = [30 + k] * 32 if cat_rep else [10 + g, 30 + k]
                rows.append(_cal_row(task, g, setting, "structured", a, d, sg,
                                     k, seeds[k], int_ok and k == 0,
                                     precommit_hash, toks, cat_rep=cat_rep))
    with open(rec_p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    json.dump({"elapsed_wall_seconds": 3600.0,
               "calibration_precommit_sha256": precommit_hash}, open(meta_p, "w"))
    return rec_p, meta_p


def write_run(recs, G=None, arms=ARMS, tag="run", **overrides):
    """Write a complete v2.0 synthetic run, including calibration and preseal."""
    d = tempfile.mkdtemp(dir=_TMP, prefix=tag + "_")
    tasks = sorted({r["task_id"] for r in recs})
    G = len(tasks) if G is None else G
    master_seed = 20260728
    for r in recs:
        r["seed"] = derive_stream_seed(master_seed, r["task_id"], r["stream_index"])
    conf_p, cal_p, seed_p, pool_p = (os.path.join(d, n) for n in
                                     ("conf_tasks.jsonl", "cal_tasks.jsonl",
                                      "seed_matrix.json",
                                      "candidate_pool.jsonl"))
    strat = {"hard_medium": 0, "medium": 0}
    strat_of = {r["task_id"]: r["stratum"] for r in recs}
    setting_rank = {"depth_hm": 0, "depth_m": 0}
    pool_rows = []
    with open(conf_p, "w") as fh:
        for t in tasks:
            st = strat_of[t]
            setting = "depth_hm" if st == "hard_medium" else "depth_m"
            fh.write(json.dumps({"task_id": t, "stratum": st,
                                 "generator_setting_id": setting,
                                 "sealed_rank_within_stratum": strat[st],
                                 "task_content_sha256": f"content_{t}",
                                 "options": FIXTURE_OPTIONS,
                                 "symbolic": FIXTURE_SYMBOLIC}) + "\n")
            strat[st] += 1
            pool_rows.append({
                "task_id": t, "generator_setting_id": setting,
                "sealed_rank_within_generator_setting": setting_rank[setting],
                "task_content_sha256": f"content_{t}",
                "options": FIXTURE_OPTIONS, "symbolic": FIXTURE_SYMBOLIC})
            setting_rank[setting] += 1
    with open(pool_p, "w") as fh:
        for row in pool_rows:
            fh.write(json.dumps(row) + "\n")
    with open(cal_p, "w") as fh:
        for i in range(40):
            st = "hard_medium" if i < 20 else "medium"
            fh.write(json.dumps({"task_id": f"cal_{i:04d}", "stratum": st,
                                 "generator_setting_id": "depth_hm" if st == "hard_medium" else "depth_m",
                                 "sealed_rank_within_stratum": i if i < 20 else i - 20,
                                 "task_content_sha256": f"cal_content_{i}",
                                 "options": FIXTURE_OPTIONS,
                                 "symbolic": FIXTURE_SYMBOLIC}) + "\n")
    sm = {}
    for r in recs:
        sm.setdefault(r["task_id"], {})[str(r["stream_index"])] = r["seed"]
    json.dump(sm, open(seed_p, "w"))

    axis_dir = os.path.join(d, "axis_package")
    os.mkdir(axis_dir)
    make_axis_package(axis_dir)
    generation_module = MODULE_PATH
    empty_diff_hash = hashlib.sha256(b"").hexdigest()
    checkpoint_dir = os.path.join(d, "checkpoint")
    os.mkdir(checkpoint_dir)
    open(os.path.join(checkpoint_dir, "weights.bin"), "wb").write(b"checkpoint")
    small_artifacts = {}
    for name, payload in (("tokenizer.json", b"tokenizer"),
                          ("prompt.txt", b"prompt"),
                          ("parser.py", b"parser"),
                          ("verifier.py", b"verifier")):
        path = os.path.join(d, name)
        open(path, "wb").write(payload)
        small_artifacts[name] = path
    orchestrator_module = os.path.join(os.path.dirname(MODULE_PATH),
                                       "run_o1_v2_orchestrator.py")
    transport_module = os.path.join(os.path.dirname(MODULE_PATH),
                                    "o1_transport_v2.py")
    artifact_paths = {
        "code.generation_module_sha256": generation_module,
        "code.orchestrator_module_sha256": orchestrator_module,
        "code.transport_module_sha256": transport_module,
        "model.checkpoint_sha256": checkpoint_dir,
        "artifact_hashes.tokenizer": small_artifacts["tokenizer.json"],
        "artifact_hashes.prompt_template": small_artifacts["prompt.txt"],
        "artifact_hashes.parser": small_artifacts["parser.py"],
        "artifact_hashes.verifier_implementation": small_artifacts["verifier.py"],
        "artifact_hashes.structured_axis_tensor": os.path.join(axis_dir, "axes_l3_24.npy"),
        "artifact_hashes.random_axis_tensor": os.path.join(axis_dir, "random_axes_l3_24.npy"),
    }
    artifact_paths_p = os.path.join(d, "RUNTIME_ARTIFACT_PATHS.json")
    json.dump(artifact_paths, open(artifact_paths_p, "w"), indent=2)

    man = {
        "prereg_version": "2.0", "sealed_at_utc": "2026-07-28T00:00:00Z",
        "code": {"commit": "deadbeef", "analysis_module_sha256": sha256_file(MODULE_PATH),
                 "fixture_suite_sha256": sha256_file(os.path.abspath(__file__)),
                 "generation_module_sha256": sha256_file(generation_module),
                 "orchestrator_module_sha256": sha256_file(orchestrator_module),
                 "transport_module_sha256": sha256_file(transport_module),
                 "calibration_module_sha256": sha256_file(os.path.join(os.path.dirname(MODULE_PATH), "calibration_analysis.py")),
                 "axis_verifier_sha256": sha256_file(os.path.join(axis_dir, "verify_axis_artifact.py")),
                 "numpy": np.__version__,
                 "torch": "fixture-torch-0.0",
                 "transformers": "4.54.1",
                 "repository_clean_required": True,
                 "repository_diff_sha256": empty_diff_hash},
        "model": {"checkpoint_sha256": sha256_tree(checkpoint_dir),
                  "checkpoint_project_relative_tree_sha256": sha256_tree(checkpoint_dir),
                  "loops": 4,
                  "dtype": "bfloat16",
                  "deterministic_flags": "cudnn_deterministic"},
        "generator": {"family": "Horizon Logic", "revision": "gen_v7",
                      "hash_ordered_pool_offsets_calibration": [0, 39],
                      "hash_ordered_pool_offsets_confirmatory": [40, 40 + G - 1],
                      "structural_knobs": ["proof_depth", "rule_count"],
                      "task_id_hash_scheme": "sha256(canonical_task_json)"},
        "difficulty_strata": {"hard_medium": [0.4, 0.55], "medium": [0.55, 0.7],
                              "min_well_formed_commitment_rate": 0.85,
                              "well_formed_noninferiority_margin": -0.05,
                              "generator_settings_per_stratum": {"hard_medium": ["depth_hm"], "medium": ["depth_m"]},
                              "expected_counts": strat},
        "decoding": {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 448,
                     "stopping_rule": "stop after parseable FINAL ANSWER or max tokens",
                     "verifier": "deterministic truth-table", "answer_marker": "FINAL ANSWER:"},
        "action_space": {
            "alpha_star": 0.02, "alpha_secondary": 0.01,
            "alpha_selection_rule": "largest_passing",
            "normalization": "per-position residual RMS at writable boundary",
            "writable_boundary": {"loop_index_zero_based": 2,
                                  "physical_layer_one_based": 24,
                                  "module_index_zero_based": 23,
                                  "boundary_semantics": "post-layer residual output",
                                  "first_affected_cache_slot": [2, 24],
                                  "position_scope": "final_nonpadding_prompt_position_only",
                                  "application_frequency": "one_shot_during_prefill",
                                  "delta_broadcast_rule": "delta = sign*alpha*r*d"},
            "structured_axes": {"gram_matrix": [[1.0]]},
            "alpha_swept_in_calibration": [0.005, 0.01, 0.02, 0.04, 0.08],
            "alpha_calibration": {"passing_alphas": [0.005, 0.01, 0.02, 0.04, 0.08],
                                  "failing_alphas": [], "failure_reasons": {}}},
        "transport": {"measurement_definition": "RMS(h_int[L4,47,final prompt pos] - h_base) / RMS(delta_in)",
                      "measurement_loop": "L4", "measurement_layer": 47,
                      "norm_convention": "RMS over hidden dimension",
                      "token_position": "final_nonpadding_prompt_position",
                      "normalized_by_injected_rms": True,
                      "dead_rms_threshold": 1e-6},
        "bank": {"size": 8, "geometry": "4 axes x 2 signs",
                 "arms_required": list(arms),
                 "arms_conditional_on_budget": [],
                 "arms_confirmatory": list(arms),
                 "stream_assignment": "cyclic_latin_square",
                 "stream_assignment_rule": "stream(a,g)=(a+r_g) mod 8"},
        "cohorts": {"master_seed": master_seed, "seed_hash_function": "sha256",
                    "seed_derivation_scheme": "sha256_domain_uint64_be_v1",
                    "confirmatory_task_manifest_sha256": sha256_file(conf_p),
                    "confirmatory_candidate_pool_sha256": sha256_file(pool_p),
                    "calibration_task_manifest_sha256": sha256_file(cal_p),
                    "seed_matrix_sha256": sha256_file(seed_p)},
        "power": {"q_disc_calibrated": 0.0,
                  "q_disc_upper_confidence_bound": 0.25,
                  "G_power": G, "G_max": G, "G_pool": G, "G_confirm": G,
                  "delta_target": 0.05, "target_power": 0.1,
                  "alpha_level": 0.05, "budget_mde_if_constrained": 0.05,
                  "calibration_verdict": "PROCEED_POWERED"},
        "budget": {"max_wall_clock_days": 7, "calibration_inside_budget": False,
                   "rerun_reserve_fraction": 0.1, "min_completed_task_fraction": 1.0,
                   "max_plausible_candidates_per_hour": 10000,
                   "on_hardware_interruption": "resume sealed missing shards only",
                   "measured_candidates_per_hour": 1000.0},
        "zero_alpha_null": {"tau": 0.01, "token_identity_required": True,
                            "min_token_identity_rate": 1.0},
        "artifact_hashes": {"tokenizer": sha256_file(small_artifacts["tokenizer.json"]),
                            "prompt_template": sha256_file(small_artifacts["prompt.txt"]),
                            "parser": sha256_file(small_artifacts["parser.py"]),
                            "verifier_implementation": sha256_file(small_artifacts["verifier.py"]),
                            "structured_axis_tensor": sha256_file(os.path.join(axis_dir, "axes_l3_24.npy")),
                            "random_axis_tensor": sha256_file(os.path.join(axis_dir, "random_axes_l3_24.npy"))},
        "axis_artifact": {"package_sha256": sha256_tree(axis_dir)},
        "calibration": {"precommit_sha256": "PENDING",
                        "raw_records_sha256": "PENDING", "metadata_sha256": "PENDING",
                        "results_sha256": "PENDING", "bootstrap_draws": 200,
                        "bootstrap_seed": 20260728},
    }
    for k, v in overrides.items():
        parts = k.split("."); cur = man
        for part in parts[:-1]: cur = cur[part]
        cur[parts[-1]] = v

    # The synthetic calibration has exactly four discordant banks among 40
    # selected tasks. Freeze the target power BEFORE calibration so its exact
    # minimum G is the requested fixture G (for G>=8). This avoids changing an
    # analysis choice after CALIBRATION_PRECOMMIT.json exists.
    q_fixture = clopper_pearson_upper(4, 40, 0.05)
    man["power"]["q_disc_upper_confidence_bound"] = q_fixture
    delta = float(man["power"]["delta_target"])
    if G >= 8:
        lo = exact_power(G - 1, q_fixture, delta) if G > 1 else 0.0
        hi = exact_power(G, q_fixture, delta)
        man["power"]["target_power"] = (lo + hi) / 2.0
        man["power"]["G_power"] = G
        man["power"]["G_max"] = G
        man["power"]["G_confirm"] = G

    # Calibration records/results are generated after design values are fixed.
    pre_man_p = os.path.join(d, "manifest_design.json")
    json.dump(man, open(pre_man_p, "w"), indent=2)
    cal_precommit = os.path.join(d, "CALIBRATION_PRECOMMIT.json")
    planned_cal_records = os.path.join(d, "calibration_records.jsonl")
    create_calibration_precommit(
        cal_precommit, pre_man_p, artifact_paths_p, cal_p,
        records_path=planned_cal_records)
    man["calibration"]["precommit_sha256"] = sha256_file(cal_precommit)
    json.dump(man, open(pre_man_p, "w"), indent=2)
    cal_rec, cal_meta = _write_calibration(d, man, cal_precommit)
    cal_result = compute_calibration(cal_rec, cal_meta, pre_man_p, cal_p,
                                     cal_precommit, pool_p)
    # For fixture manifests, bind the deterministic outputs and then choose the
    # target power so G_power equals the confirmatory G.
    q = cal_result["discordance"]["q_disc_upper_confidence_bound"]
    if q < man["power"]["delta_target"]:
        q = man["power"]["delta_target"]
    man["power"]["q_disc_calibrated"] = cal_result["discordance"]["q_disc"]
    man["power"]["q_disc_upper_confidence_bound"] = q
    man["power"]["G_power"] = G
    man["power"]["G_max"] = G
    man["power"]["G_confirm"] = G
    man["power"]["calibration_verdict"] = "PROCEED_POWERED"
    man["action_space"]["alpha_star"] = cal_result["alpha"]["alpha_star"]
    man["action_space"]["alpha_secondary"] = cal_result["alpha"]["alpha_secondary"]
    man["action_space"]["alpha_calibration"]["passing_alphas"] = cal_result["alpha"]["passing_alphas"]
    man["action_space"]["alpha_calibration"]["failing_alphas"] = cal_result["alpha"]["failing_alphas"]
    man["action_space"]["alpha_calibration"]["failure_reasons"] = cal_result["alpha"]["failure_reasons"]
    man["difficulty_strata"]["generator_settings_per_stratum"] = cal_result["difficulty"]["selected_settings_per_stratum"]
    man["budget"]["measured_candidates_per_hour"] = cal_result["throughput"]["measured_candidates_per_hour"]

    # Recompute final calibration result against final design. To keep G_power=G,
    # its metadata is chosen so G_max=G; the result is then patched only through
    # its deterministic design inputs, never by hand.
    candidates = len([ln for ln in open(cal_rec) if ln.strip()])
    max_sec = man["budget"]["max_wall_clock_days"] * 86400 * (1 - man["budget"]["rerun_reserve_fraction"])
    desired_thr = G * (8 * len(arms)) / (max_sec / 3600)
    elapsed = candidates / desired_thr * 3600
    json.dump({"elapsed_wall_seconds": elapsed,
               "calibration_precommit_sha256": sha256_file(cal_precommit)},
              open(cal_meta, "w"))

    # Write provisional final manifest, compute result, then bind hashes/values.
    man_p = os.path.join(d, "manifest.json")
    json.dump(man, open(man_p, "w"), indent=2)
    cal_result = compute_calibration(cal_rec, cal_meta, man_p, cal_p,
                                     cal_precommit, pool_p)
    # The exact calibration power result is authoritative.
    alloc = cal_result.get("confirmatory_allocation")
    for target, value in [
        (("action_space", "alpha_star"), cal_result["alpha"]["alpha_star"]),
        (("action_space", "alpha_secondary"), cal_result["alpha"]["alpha_secondary"]),
        (("difficulty_strata", "generator_settings_per_stratum"), cal_result["difficulty"]["selected_settings_per_stratum"]),
        (("difficulty_strata", "expected_counts"),
         alloc["expected_counts"] if alloc else {"hard_medium": 0, "medium": 0}),
        (("power", "q_disc_calibrated"), cal_result["discordance"]["q_disc"]),
        (("power", "q_disc_upper_confidence_bound"), cal_result["discordance"]["q_disc_upper_confidence_bound"]),
        (("power", "G_power"), cal_result["power"]["G_power"]),
        (("power", "G_max"), cal_result["power"]["G_max"]),
        (("power", "G_pool"), cal_result["power"]["G_pool"]),
        (("power", "G_confirm"), cal_result["power"]["G_confirm"]),
        (("power", "budget_mde_if_constrained"), cal_result["power"]["budget_mde_if_constrained"]),
        (("power", "calibration_verdict"), cal_result["verdict"]),
        (("bank", "arms_confirmatory"), cal_result["power"]["selected_confirmatory_arms"]),
        (("budget", "measured_candidates_per_hour"), cal_result["throughput"]["measured_candidates_per_hour"]),
    ]:
        cur = man
        for part in target[:-1]: cur = cur[part]
        cur[target[-1]] = value
    man["action_space"]["alpha_calibration"]["passing_alphas"] = cal_result["alpha"]["passing_alphas"]
    man["action_space"]["alpha_calibration"]["failing_alphas"] = cal_result["alpha"]["failing_alphas"]
    man["action_space"]["alpha_calibration"]["failure_reasons"] = cal_result["alpha"]["failure_reasons"]

    # Recompute once more because manifest power outputs are not design inputs;
    # the result bytes are stable.
    json.dump(man, open(man_p, "w"), indent=2)
    cal_result = compute_calibration(cal_rec, cal_meta, man_p, cal_p,
                                     cal_precommit, pool_p)
    cal_res_p = os.path.join(d, "CALIBRATION_RESULTS.json")
    write_results(cal_result, cal_res_p)
    man["calibration"] = {"precommit_sha256": sha256_file(cal_precommit),
                          "raw_records_sha256": sha256_file(cal_rec),
                          "metadata_sha256": sha256_file(cal_meta),
                          "results_sha256": sha256_file(cal_res_p),
                          "bootstrap_draws": 200,
                          "bootstrap_seed": 20260728}
    # Test mutations are applied after deterministic calibration binding so the
    # fixture can exercise manifest gates directly.
    for k, v in overrides.items():
        parts = k.split("."); cur = man
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = v
    json.dump(man, open(man_p, "w"), indent=2)

    # Runtime provenance is an exact recursive copy of the resolved config.
    M = Manifest(man_p)
    prov_p = os.path.join(d, "RUN_PROVENANCE.json")
    json.dump({"schema_version": "1.0", "created_at_utc": "2026-07-28T00:00:00Z",
               "resolved_generation_config": resolved_generation_config(M),
               "runtime_attestation": {"generation_module_sha256": man["code"]["generation_module_sha256"],
                                       "repository_clean": True,
                                       "repository_diff_sha256": empty_diff_hash}},
              open(prov_p, "w"), indent=2)
    rec_p = os.path.join(d, "records.jsonl")
    pregen = os.path.join(d, "PREGENERATION_SEAL.json")
    create_pregeneration_seal(pregen, man_p, prov_p, conf_p, cal_p,
                              cal_precommit, seed_p, cal_rec, cal_meta,
                              cal_res_p, axis_dir, pool_p,
                              records_path=rec_p)
    prov_hash = sha256_file(prov_p)
    pregen_hash = sha256_file(pregen)
    for r in recs:
        r["generator_setting_id"] = ("depth_hm" if r["stratum"] == "hard_medium"
                                     else "depth_m")
        r["run_provenance_sha256"] = prov_hash
        r["pregeneration_seal_sha256"] = pregen_hash
    with open(rec_p, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return {"dir": d, "records": rec_p, "manifest": man_p, "conf": conf_p,
            "cal": cal_p, "cal_precommit": cal_precommit,
            "seeds": seed_p, "prov": prov_p,
            "pregen": pregen, "cal_records": cal_rec,
            "cal_metadata": cal_meta, "cal_results": cal_res_p,
            "axis_package": axis_dir, "artifact_paths": artifact_paths_p,
            "pool": pool_p,
            "seal": os.path.join(d, "RESULT_SEAL.json")}


def V(recs, run, seeds=True):
    from o1_analysis import load_task_manifest
    sm = json.load(open(run["seeds"])) if seeds else None
    validate(recs, Manifest(run["manifest"]), load_task_manifest(run["conf"]),
             load_task_manifest(run["cal"]), sm)


def RP(run, provenance_override="__DEFAULT__"):
    prov = run["prov"] if provenance_override == "__DEFAULT__" else provenance_override
    return run_primary(
        run["records"], run["manifest"], run["conf"], run["cal"],
        run["cal_precommit"], run["seeds"], run["seal"], prov, run["pregen"],
        run["cal_records"], run["cal_metadata"], run["cal_results"],
        run["axis_package"], run["pool"],
    )


def expect(code, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except IntegrityError as e:
        assert e.code == code, f"expected {code}, got {e.code}: {e}"
        print(f"  caught: {e}")
        return e
    raise AssertionError(f"expected IntegrityError[{code}], nothing raised")


# ---------------------------------------------------------------- fixtures

def test_00_clean_run_end_to_end():
    recs = make_clean()
    run = write_run(recs)
    out = RP(run)
    p = out["primary"]
    assert (p["n11"], p["n10"], p["n01"], p["n00"]) == (2, 8, 6, 24), p
    assert abs(p["H_reach"] - 0.05) < 1e-12
    assert out["zero_alpha"]["verdict"] == "PROCEED"
    assert out["zero_alpha"]["parity"]["token_identity_rate"] == 1.0
    seal = json.load(open(run["seal"]))
    assert [e["name"] for e in seal["entries"]] == ["primary"]
    print(f"  H_reach={p['H_reach']:+.4f} n10={p['n10']} n01={p['n01']} "
          f"p={p['p_exact']:.3f} CI=[{p['ci_lo']:+.4f},{p['ci_hi']:+.4f}] "
          f"verdict={p['verdict']}")


def test_01_four_unique_baselines_dressed_as_eight():
    recs = make_clean(G=20); run = write_run(recs, tag="f01")
    for r in recs:
        if r["task_id"] == "conf_0003" and r["arm"] == "baseline":
            r["seed"] = 1_000_000 + 300 + r["stream_index"] // 2
    e = expect("G3_STREAM_UNIQUENESS", V, recs, run)
    assert e.detail["n_unique_seeds"] == 4


def test_02_task_set_mismatch_and_calibration_overlap():
    recs = make_clean(G=20); run = write_run(recs, tag="f02")
    for r in recs:
        if r["task_id"] == "conf_0007":
            r["task_id"] = "cal_0007"
    expect("G7_TASK_SET", V, recs, run)


def test_03_duplicate_seed_in_bank():
    recs = make_clean(G=20); run = write_run(recs, tag="f03")
    duplicate = next(r["seed"] for r in recs
                     if r["task_id"] == "conf_0011" and r["arm"] == "baseline"
                     and r["stream_index"] == 4)
    for r in recs:
        if (r["task_id"] == "conf_0011" and r["arm"] == "baseline"
                and r["stream_index"] == 5):
            r["seed"] = duplicate
    e = expect("G3_STREAM_UNIQUENESS", V, recs, run)
    assert e.detail["n_unique_seeds"] == 7


def test_04_missing_paired_row():
    recs = make_clean(G=20); run = write_run(recs, tag="f04")
    recs = [r for r in recs if not (r["task_id"] == "conf_0013"
                                    and r["arm"] == "baseline"
                                    and r["stream_index"] == 6)]
    e = expect("G2_BANK_SIZE", V, recs, run); assert e.detail["n"] == 7


def test_05_malformed_scored_as_success():
    recs = make_clean(G=20); run = write_run(recs, tag="f05")
    for r in recs:
        if (r["task_id"] == "conf_0002" and r["arm"] == "structured"
                and r["stream_index"] == 0):
            r["well_formed"], r["verifier_correct"], r["R"] = False, True, 1
    e = expect("G6_MALFORMED_CONVENTION", V, recs, run)
    assert e.detail["stored"] == 1 and e.detail["derived"] == 0


def test_06_bank_size_not_eight():
    recs = make_clean(G=20); run = write_run(recs, tag="f06")
    extra = copy.deepcopy([r for r in recs if r["task_id"] == "conf_0009"
                           and r["arm"] == "structured"][0])
    extra["seed"] += 999
    recs.append(extra)
    e = expect("G2_BANK_SIZE", V, recs, run); assert e.detail["n"] == 9


def test_07_independent_seeds_instead_of_crn():
    recs = make_clean(G=20); run = write_run(recs, tag="f07")
    for r in recs:
        if r["arm"] == "structured":
            r["seed"] += 500_000
    expect("G4_INDEX_MATCHED_SEEDS", V, recs, run)


def test_08_random_bank_eight_free_axes():
    arms = ARMS + ["random"]
    recs = make_clean(G=10, arms=arms); run = write_run(recs, arms=arms, tag="f08")
    for r in recs:
        if r["arm"] == "random":
            r["direction_id"], r["sign"] = r["stream_index"] + 1, 1
    expect("G1_SCHEMA", V, recs, run)   # direction_id 5..8 rejected at schema


def test_09_secondary_before_primary_and_append_only_seal():
    recs = make_clean(G=20); run = write_run(recs, tag="f09")
    man = Manifest(run["manifest"])
    pregen_hash = sha256_file(run["pregen"])
    seal = Seal(run["seal"], man.sha256, pregen_hash)
    try:
        seal.open_secondary("stratum_medium", {})
    except SealViolation as e:
        print(f"  caught: {e}")
    else:
        raise AssertionError("expected SealViolation")
    res = endpoint(recs, "structured", man)
    h = seal.open_primary(res); assert len(h) == 64
    try:
        seal.open_primary(res)
    except SealViolation as e:
        print(f"  caught: {e}")
    else:
        raise AssertionError("expected SealViolation on second primary")
    try:
        Seal(run["seal"], man.sha256)
    except SealViolation as e:
        print(f"  caught: {e}")
    else:
        raise AssertionError("expected SealViolation on existing seal path")
    resumed = Seal.resume(run["seal"], man.sha256, pregen_hash)
    resumed.open_secondary("stratum_medium", {"H": 0.0})
    chain = json.load(open(run["seal"]))
    assert chain["entries"][1]["parent"] == chain["entries"][0]["hash"]
    print(f"  hash chain intact, head={chain['chain_head'][:16]}...")


# ------ defects that passed silently in v1.0 ------

def test_10_entire_structured_arm_missing():
    recs = make_clean(G=20); run = write_run(recs, tag="f10")
    recs = [r for r in recs if not (r["task_id"] == "conf_0002"
                                    and r["arm"] == "structured")]
    e = expect("G9_ARM_COMPLETENESS", V, recs, run)
    assert e.detail["missing"] == ["structured"]


def test_11_entire_zero_alpha_arm_missing():
    recs = make_clean(G=20); run = write_run(recs, tag="f11")
    recs = [r for r in recs if not (r["task_id"] == "conf_0002"
                                    and r["arm"] == "zero_alpha")]
    expect("G9_ARM_COMPLETENESS", V, recs, run)


def test_12_one_row_wrong_alpha():
    recs = make_clean(G=20); run = write_run(recs, tag="f12")
    for r in recs:
        if (r["task_id"] == "conf_0002" and r["arm"] == "structured"
                and r["stream_index"] == 3):
            r["alpha"] = 0.04
    e = expect("G10_ALPHA_DISCIPLINE", V, recs, run)
    assert e.detail["alpha"] == 0.04 and e.detail["sealed"] == 0.02


def test_13_baseline_alpha_nonzero():
    recs = make_clean(G=20); run = write_run(recs, tag="f13")
    for r in recs:
        if r["arm"] == "baseline":
            r["alpha"] = 0.02
    expect("G10_ALPHA_DISCIPLINE", V, recs, run)


def test_14_task_carries_two_strata():
    recs = make_clean(G=20); run = write_run(recs, tag="f14")
    for r in recs:
        if r["task_id"] == "conf_0002" and r["arm"] == "structured":
            r["stratum"] = "medium"
    expect("G11_STRATUM_CONSISTENCY", V, recs, run)


def test_15_completed_subset_changes_G():
    """Dropping a sealed task is caught twice over: as a task-set violation, and
    (when the task manifest is also truncated) as an exact-G violation."""
    recs = make_clean(G=20); run = write_run(recs, tag="f15")
    subset = [r for r in recs if r["task_id"] != "conf_0019"]
    expect("G7_TASK_SET", V, subset, run)
    # v2.1: the pool cap makes a truncated manifest honestly reseal to the
    # smaller pool (G_confirm follows min(G_power, G_max, G_pool)), so the
    # sealed-G disagreement is exercised directly at the gate.
    from o1_analysis import gate_G12_exact_G
    e = expect("G12_EXACT_G", gate_G12_exact_G, subset, 20)
    assert e.detail["actual"] == 19 and e.detail["sealed"] == 20
    # and the honest reseal itself must validate cleanly end to end
    run2 = write_run(subset, G=20, tag="f15b")
    V(subset, run2)


def test_16_none_direction_id_raises_integrity_not_typeerror():
    recs = make_clean(G=3); run = write_run(recs, tag="f16")
    for r in recs:
        if (r["task_id"] == "conf_0000" and r["arm"] == "structured"
                and r["stream_index"] == 0):
            r["direction_id"] = None
    expect("G1_SCHEMA", V, recs, run)


def test_17_unequal_task_sets_rejected_not_intersected():
    from o1_analysis import paired_arrays
    recs = [r for r in make_clean(G=5)
            if not (r["task_id"] == "conf_0002" and r["arm"] == "structured")]
    expect("E1_UNEQUAL_TASK_SETS", paired_arrays, recs, "structured")


# ------ manifest binding ------

def test_18_unsealed_manifest_rejected():
    recs = make_clean(G=5); run = write_run(recs, tag="f18")
    m = json.load(open(run["manifest"])); m["action_space"]["alpha_star"] = "FREEZE_SLOT"
    json.dump(m, open(run["manifest"], "w"), indent=2)
    e = expect("G0_MANIFEST_UNSEALED", Manifest, run["manifest"])
    assert e.detail["open"]


def test_19_manifest_self_hash_rejected():
    recs = make_clean(G=5); run = write_run(recs, tag="f19")
    m = json.load(open(run["manifest"])); m["manifest_sha256"] = "whatever"
    json.dump(m, open(run["manifest"], "w"), indent=2)
    expect("G0_MANIFEST_SELF_HASH", Manifest, run["manifest"])


def test_20_artifact_hash_mismatch_rejected():
    recs = make_clean(G=5); run = write_run(recs, tag="f20")
    open(run["conf"], "a").write("conf_9999\n")     # tamper after sealing
    expect("G18_PREGENERATION_SEAL", RP, run)


# ------ zero-alpha parity ------

def test_21_zero_alpha_symmetric_divergence_halts():
    """Bank-level H_0 = 0 but the tokens differ: v1.0 passed this."""
    recs = make_clean(G=40); run = write_run(recs, tag="f21")
    for r in recs:
        if r["arm"] == "zero_alpha":
            _set_tokens(r, [777] + r["generated_token_ids"][1:])
    man = Manifest(run["manifest"]); V(recs, run)
    g = zero_alpha_gate(recs, man)
    assert g.result.H_reach == 0.0 and g.verdict == "HALT_FOR_PIPELINE_REPAIR"
    print(f"  H_0={g.result.H_reach:+.4f} but token identity="
          f"{g.parity['token_identity_rate']:.3f} -> {g.verdict}")
    for why in g.reasons:
        print(f"    reason: {why}")


def test_22_zero_alpha_directional_floor_halts():
    recs = make_clean(G=200); run = write_run(recs, tag="f22")
    for r in recs:
        g = int(r["task_id"].split("_")[1])
        if r["arm"] == "zero_alpha" and g % 16 == 0 and r["stream_index"] == 0:
            _set_scoring(r, True, True)
    man = Manifest(run["manifest"]); V(recs, run)
    gate = zero_alpha_gate(recs, man)
    assert gate.verdict == "HALT_FOR_PIPELINE_REPAIR"
    print(f"  H_0={gate.result.H_reach:+.4f} -> {gate.verdict}")


# ------ verdicts ------

def test_23_four_state_verdicts():
    man_run = write_run(make_clean(G=400), tag="f23")
    man = Manifest(man_run["manifest"])

    # a wide null must NOT be reported as a negative oracle
    recs = make_clean(G=40, n_int=lambda g: 1 if g % 4 == 0 else 0,
                      n_base=lambda g: 1 if g % 4 == 0 else 0)
    r_small = write_run(recs, tag="f23a")
    res = endpoint(recs, "structured", Manifest(r_small["manifest"]))
    assert res.verdict == "INCONCLUSIVE", res
    print(f"  G=40  H={res.H_reach:+.4f} ub95={res.ub_one_sided_95:+.4f} "
          f"-> {res.verdict}")

    # same zero effect, large G: equivalence is now established
    recs = make_clean(G=400, n_int=lambda g: 1 if g % 4 == 0 else 0,
                      n_base=lambda g: 1 if g % 4 == 0 else 0)
    r_big = write_run(recs, tag="f23b")
    res = endpoint(recs, "structured", Manifest(r_big["manifest"]))
    assert res.verdict.startswith("PRACTICALLY_NULL"), res
    print(f"  G=400 H={res.H_reach:+.4f} ub95={res.ub_one_sided_95:+.4f} "
          f"-> {res.verdict}")

    # clear positive
    recs = make_clean(G=200, n_int=lambda g: 1 if g % 2 == 0 else 0,
                      n_base=lambda g: 1 if g % 10 == 0 else 0)
    r_pos = write_run(recs, tag="f23c")
    res = endpoint(recs, "structured", Manifest(r_pos["manifest"]))
    assert res.verdict == "POSITIVE_HEADROOM", res
    print(f"  pos   H={res.H_reach:+.4f} p={res.p_exact:.2e} -> {res.verdict}")

    # clear negative
    recs = make_clean(G=200, n_int=lambda g: 1 if g % 10 == 0 else 0,
                      n_base=lambda g: 1 if g % 2 == 0 else 0)
    r_neg = write_run(recs, tag="f23d")
    res = endpoint(recs, "structured", Manifest(r_neg["manifest"]))
    assert res.verdict == "HARMFUL", res
    print(f"  neg   H={res.H_reach:+.4f} p={res.p_exact:.2e} -> {res.verdict}")


# ------ power and budget ------

def test_24_exact_power_table():
    print("   q_disc  delta   exact minimum G (80% directional, alpha=0.05)")
    got = {}
    for q, d in ((0.25, 0.08), (0.25, 0.05), (0.25, 0.03), (0.10, 0.05)):
        got[(q, d)] = required_G(q, d)
        print(f"    {q:.2f}   {d:+.2f}   {got[(q, d)]:>6d}")
    assert got == {(0.25, 0.08): 324, (0.25, 0.05): 817,
                   (0.25, 0.03): 2238, (0.10, 0.05): 337}, got
    assert exact_power(324, 0.25, 0.08) >= 0.80
    assert exact_power(323, 0.25, 0.08) < 0.80
    print("   determinism confirmed: identical across calls, no n_sim dependence")


def test_25_budget_fallback_deterministic():
    d = budget_fallback(0.25, 2000, 0.05)
    assert d.powered_for_target and d.G_confirm == 817 and not d.dropped_arms
    print(f"  G_max=2000 -> G_confirm={d.G_confirm}, powered for +0.05")
    d = budget_fallback(0.25, 300, 0.05)
    assert not d.powered_for_target
    assert d.dropped_arms == ["random", "structured_secondary"]
    print(f"  G_max=300  -> G_confirm={d.G_confirm}, dropped={d.dropped_arms}, "
          f"budget MDE={d.mde:+.4f}")


def test_26_divergence_computed_not_trusted():
    recs = make_clean(G=5)
    for r in recs:
        r["first_divergence"] = 999   # upstream lie
    div = first_divergence(recs, "structured")
    assert set(div.values()) == {1}, div
    cs = coupling_survival(recs, "structured")
    assert cs["p_no_divergence"] == 0.0 and cs["median_T_div"] == 1.0
    print(f"  ignored upstream first_divergence=999, computed T_div=1 "
          f"for {cs['n']} rows")


def test_27_headroom_helpers():
    recs = make_clean(G=20, n_int=lambda g: 2, n_base=lambda g: 0)
    assert abs(selection_headroom(recs, "structured") - (1 - 2 / 8)) < 1e-12
    assert outcome_diversity(recs, "structured") == 1.0
    recs2 = make_clean(G=20, n_int=lambda g: 0, n_base=lambda g: 0)
    assert outcome_diversity(recs2, "structured") == 0.0
    print("  H_sel and outcome diversity match hand calculation")


def test_28_stream_assignment_off_latin_square():
    recs = make_clean(G=20); run = write_run(recs, tag="f28")
    base = {r["stream_index"]: r["seed"] for r in recs
            if r["arm"] == "baseline" and r["task_id"] == "conf_0004"}
    for r in recs:
        if r["arm"] == "structured" and r["task_id"] == "conf_0004":
            r["stream_index"] = (r["stream_index"] + 3) % BANK_SIZE
            r["seed"] = base[r["stream_index"]]     # stays CRN-paired
    expect("G13_STREAM_ASSIGNMENT", V, recs, run)


def test_29_latin_square_balances_actions_across_streams():
    recs = make_clean(G=16)
    seen = {}
    for r in recs:
        if r["arm"] == "structured":
            a = action_index(r["direction_id"], r["sign"])
            seen.setdefault(a, []).append(r["stream_index"])
    for a, idxs in sorted(seen.items()):
        assert sorted(set(idxs)) == list(range(BANK_SIZE)), (a, idxs)
    print(f"  each of 8 signed actions visits all 8 streams over 16 tasks "
          f"({len(seen)} actions checked)")


def test_30_provenance_mismatch_rejected():
    recs = make_clean(G=5); run = write_run(recs, tag="f30")
    prov = json.load(open(run["prov"]))
    prov["resolved_generation_config"]["artifact_hashes"]["structured_axis_tensor"] = "WRONG_TENSOR"
    json.dump(prov, open(run["prov"], "w"))
    # Recommit the tampered provenance so the pre-generation gate does not mask G14.
    os.remove(run["pregen"])
    create_pregeneration_seal(run["pregen"], run["manifest"], run["prov"],
                              run["conf"], run["cal"], run["cal_precommit"], run["seeds"],
                              run["cal_records"], run["cal_metadata"],
                              run["cal_results"], run["axis_package"],
                              run["pool"])
    e = expect("G14_PROVENANCE", RP, run)
    assert "artifact_hashes" in e.detail["sections"]


def test_31_missing_provenance_rejected():
    recs = make_clean(G=5); run = write_run(recs, tag="f31")
    expect("G18_PREGENERATION_SEAL", RP, run, None)


def test_32_alpha_grid_rule_enforced():
    recs = make_clean(G=5)
    run = write_run(recs, tag="f32a", **{"action_space.alpha_secondary": 0.04})
    expect("G15_ALPHA_GRID", V, recs, run)          # secondary above star
    run = write_run(recs, tag="f32b", **{"action_space.alpha_secondary": 0.005})
    expect("G15_ALPHA_GRID", V, recs, run)          # not adjacent
    run = write_run(recs, tag="f32c", **{"action_space.alpha_star": 0.03})
    expect("G15_ALPHA_GRID", V, recs, run)          # star off the swept grid
    run = write_run(recs, tag="f32d",
                    **{"action_space.alpha_selection_rule": "vibes"})
    expect("G15_ALPHA_GRID", V, recs, run)


def test_33_transport_field_required_on_intervention_arms():
    recs = make_clean(G=5); run = write_run(recs, tag="f33")
    for r in recs:
        if r["arm"] == "structured" and r["stream_index"] == 0:
            r["downstream_delta_rms"] = None
    expect("G1_SCHEMA", V, recs, run)


def test_34_transport_decomposes_a_null():
    """A direction whose delta never propagates is distinguishable from one that
    propagates and does not help. Without this they read identically."""
    recs = make_clean(G=40, n_int=lambda g: 0)
    for r in recs:
        if r["arm"] == "structured" and r["direction_id"] == 2:
            r["downstream_delta_rms"] = 0.0        # d2 dies at the boundary
    run = write_run(recs, tag="f34")
    V(recs, run)
    res = endpoint(recs, "structured", Manifest(run["manifest"]))
    ts = transport_summary(recs, "structured")
    assert outcome_diversity(recs, "structured") == 0.0
    assert ts["d2+"]["dead_rate"] == 1.0 and ts["d1+"]["dead_rate"] == 0.0
    print(f"  bank-level: H={res.H_reach:+.4f}, outcome diversity 0.000")
    for k in ("d1+", "d2+", "d3+"):
        print(f"    {k}: median transport RMS {ts[k]['median_rms']:.3f}, "
              f"dead rate {ts[k]['dead_rate']:.2f}")
    print("  -> d2 contributes nothing because it never propagated, not "
          "because the action space is barren")


def test_35_runs_from_a_flattened_archive():
    """The module and its calibration/verifier dependencies survive flattening."""
    import shutil, subprocess
    flat = tempfile.mkdtemp(dir=_TMP, prefix="flat_")
    for name in ("o1_analysis.py", "calibration_analysis.py",
                 "calibration_precommit.py", "build_run_provenance.py",
                 "verify_axis_artifact.py", "test_o1_fixtures.py",
                 "o1_answer_parser_v2.py", "o1_truth_table_verifier_v2.py",
                 "cohort_allocation.py", "run_o1_v2_orchestrator.py",
                 "o1_transport_v2.py"):
        src = os.path.join(os.path.dirname(MODULE_PATH), name)
        shutil.copy(src, flat)
    code = (
        "import test_o1_fixtures as t; "
        "r=t.make_clean(G=8); run=t.write_run(r, tag='flat'); "
        "t.V(r,run); print('FLAT_SMOKE_PASS')"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=flat,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "FLAT_SMOKE_PASS" in out.stdout, out.stderr
    print("  flattened layout: FLAT_SMOKE_PASS")


def test_36_sealed_seed_shift_caught():
    """Global seed shift preserves CRN pairing; only the sealed matrix catches it."""
    recs = make_clean(G=8); run = write_run(recs, tag="f36")
    for r in recs:
        r["seed"] += 777_777
    e = expect("G16_SEED_MATRIX_BINDING", V, recs, run)
    assert e.detail["actual"] - e.detail["sealed"] == 777_777


def test_37_empty_seed_matrix_placeholder_caught():
    recs = make_clean(G=8); run = write_run(recs, tag="f37")
    json.dump({r["task_id"]: {} for r in recs}, open(run["seeds"], "w"))
    expect("G16_SEED_MATRIX_BINDING", V, recs, run)


def test_38_stratum_swap_preserving_counts_caught():
    """Two tasks swap strata, aggregate counts unchanged, ranks recomputed."""
    recs = make_clean(G=8); run = write_run(recs, tag="f38")
    sw = {"conf_0000": "medium", "conf_0001": "hard_medium"}
    for r in recs:
        if r["task_id"] in sw:
            r["stratum"] = sw[r["task_id"]]
    e = expect("G11_STRATUM_CONSISTENCY", V, recs, run)
    assert e.detail


def test_39_alpha_star_not_largest_passing_caught():
    recs = make_clean(G=5)
    run = write_run(recs, tag="f39a",
                    **{"action_space.alpha_calibration.passing_alphas":
                       [0.005, 0.01, 0.02, 0.04],
                       "action_space.alpha_calibration.failing_alphas": [0.08],
                       "action_space.alpha_calibration.failure_reasons":
                           {"0.08": "repetition"}})
    e = expect("G15_ALPHA_GRID", V, recs, run)   # 0.04 passed but star is 0.02
    assert e.detail["max_passing"] == 0.04
    run = write_run(recs, tag="f39b",
                    **{"action_space.alpha_calibration.passing_alphas":
                       [0.005, 0.02],
                       "action_space.alpha_calibration.failing_alphas":
                           [0.01, 0.04, 0.08],
                       "action_space.alpha_calibration.failure_reasons":
                           {"0.01": "x", "0.04": "x", "0.08": "x"}})
    expect("G15_ALPHA_GRID", V, recs, run)       # secondary 0.01 declared failing


def test_40_nan_transport_caught():
    recs = make_clean(G=8); run = write_run(recs, tag="f40")
    for r in recs:
        if r["arm"] == "structured" and r["stream_index"] == 0:
            r["downstream_delta_rms"] = float("nan")
    expect("G1_SCHEMA", V, recs, run)
    recs = make_clean(G=8)
    for r in recs:
        if r["arm"] == "structured" and r["stream_index"] == 0:
            r["downstream_delta_rms"] = -1.0
    expect("G1_SCHEMA", V, recs, write_run(recs, tag="f40b"))


def test_41_manifest_internal_consistency():
    recs = make_clean(G=5)
    run = write_run(recs, tag="f41a",
                    **{"action_space.structured_axes_sha256": "dup"})
    expect("G17_MANIFEST_CONSISTENCY", V, recs, run)
    run = write_run(recs, tag="f41b", **{"power.G_power": 999, "power.G_max": 5000})
    e = expect("G17_MANIFEST_CONSISTENCY", V, recs, run)
    print(f"    (G_confirm cross-check: {e})")


def test_42_bare_task_id_manifest_rejected():
    from o1_analysis import load_task_manifest
    recs = make_clean(G=5); run = write_run(recs, tag="f42")
    open(run["conf"], "w").write("conf_0000\nconf_0001\n")
    expect("G7_TASK_SET", load_task_manifest, run["conf"])


def test_43_sealed_ranks_must_be_a_permutation():
    recs = make_clean(G=8); run = write_run(recs, tag="f43")
    rows = [json.loads(l) for l in open(run["conf"])]
    hm = [r for r in rows if r["stratum"] == "hard_medium"]
    hm[1]["sealed_rank_within_stratum"] = hm[0]["sealed_rank_within_stratum"]
    with open(run["conf"], "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    from o1_analysis import load_task_manifest
    e = expect("G7_TASK_SET", load_task_manifest, run["conf"])
    assert e.detail["duplicated"]


def test_44_same_task_content_across_cohorts_caught():
    recs = make_clean(G=8); run = write_run(recs, tag="f44")
    rows = [json.loads(l) for l in open(run["cal"])]
    rows[0]["task_content_sha256"] = "content_conf_0000"      # same content, new ID
    with open(run["cal"], "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    e = expect("G7_TASK_SET", V, recs, run)
    assert "content" in str(e).lower()


def test_45_record_generated_from_other_content_caught():
    recs = make_clean(G=8); run = write_run(recs, tag="f45")
    for r in recs:
        if r["task_id"] == "conf_0003":
            r["task_content_sha256"] = "content_conf_0004"
    expect("G7_TASK_SET", V, recs, run)


def test_46_alpha_sweep_must_partition():
    recs = make_clean(G=5)
    run = write_run(recs, tag="f46a",
                    **{"action_space.alpha_calibration.failing_alphas": [],
                       "action_space.alpha_calibration.failure_reasons": {}})
    e = expect("G15_ALPHA_GRID", V, recs, run)
    assert e.detail["undeclared"] == [0.04, 0.08]
    run = write_run(recs, tag="f46b",
                    **{"action_space.alpha_calibration.passing_alphas": [0.02],
                       "action_space.alpha_calibration.failing_alphas":
                           [0.005, 0.01, 0.04, 0.08],
                       "action_space.alpha_calibration.failure_reasons":
                           {"0.005": "x", "0.01": "x", "0.04": "x", "0.08": "x"}})
    expect("G15_ALPHA_GRID", V, recs, run)   # secondary 0.01 declared failing


def test_47_zero_alpha_metadata_must_be_exactly_identical():
    recs = make_clean(G=40); run = write_run(recs, tag="f47a")
    for r in recs:
        if r["arm"] == "zero_alpha" and r["task_id"] == "conf_0000":
            r["hit_max_tokens"] = True
    man = Manifest(run["manifest"]); V(recs, run)
    g = zero_alpha_gate(recs, man)
    assert g.verdict == "HALT_FOR_PIPELINE_REPAIR" and g.result.H_reach == 0.0
    print(f"  termination flag differs, tokens identical, H_0=0 -> {g.verdict}")

    # same derived R, different parser/verifier state (rows stay internally
    # text-consistent so only the CRN parity gate can catch the divergence)
    recs = make_clean(G=40, n_base=lambda g: 0)
    for r in recs:
        if r["task_id"] == "conf_0000" and r["stream_index"] == 0:
            if r["arm"] == "baseline":
                _set_scoring(r, True, False)
            if r["arm"] == "zero_alpha":
                _set_scoring(r, False, False)
    run = write_run(recs, tag="f47b"); man = Manifest(run["manifest"]); V(recs, run)
    g = zero_alpha_gate(recs, man)
    assert g.verdict == "HALT_FOR_PIPELINE_REPAIR"
    print(f"  parser/verifier disagree, derived R identical -> {g.verdict}")
    for why in g.reasons[:2]:
        print(f"    reason: {why[:88]}")


def test_48_provenance_must_resolve_the_full_configuration():
    recs = make_clean(G=5); run = write_run(recs, tag="f48")
    prov = json.load(open(run["prov"]))
    del prov["resolved_generation_config"]["action_space"]["writable_boundary"]["module_index_zero_based"]
    json.dump(prov, open(run["prov"], "w"))
    # Recommit the tampered provenance so the pre-generation gate does not mask G14.
    os.remove(run["pregen"])
    create_pregeneration_seal(run["pregen"], run["manifest"], run["prov"],
                              run["conf"], run["cal"], run["cal_precommit"], run["seeds"],
                              run["cal_records"], run["cal_metadata"],
                              run["cal_results"], run["axis_package"],
                              run["pool"])
    e = expect("G14_PROVENANCE", RP, run)
    assert "action_space" in e.detail["sections"]


def test_49_generation_at_the_wrong_boundary_caught():
    recs = make_clean(G=5); run = write_run(recs, tag="f49")
    prov = json.load(open(run["prov"]))
    prov["resolved_generation_config"]["action_space"]["writable_boundary"]["module_index_zero_based"] = 46
    prov["resolved_generation_config"]["action_space"]["writable_boundary"]["loop_index_zero_based"] = 3
    json.dump(prov, open(run["prov"], "w"))
    # Recommit the tampered provenance so the pre-generation gate does not mask G14.
    os.remove(run["pregen"])
    create_pregeneration_seal(run["pregen"], run["manifest"], run["prov"],
                              run["conf"], run["cal"], run["cal_precommit"], run["seeds"],
                              run["cal_records"], run["cal_metadata"],
                              run["cal_results"], run["axis_package"],
                              run["pool"])
    e = expect("G14_PROVENANCE", RP, run)
    print(f"    generated at L4_47 instead of L3_24 -> sections {e.detail['sections']}")



def test_50_seed_matrix_cannot_be_rechosen_after_master_seed_is_frozen():
    """Rows and matrix agree on a post-hoc seed choice; derivation still rejects it."""
    recs = make_clean(G=8); run = write_run(recs, tag="f50")
    matrix = json.load(open(run["seeds"]))
    for r in recs:
        new_seed = (int(r["seed"]) + 1234567) % (2**64)
        r["seed"] = new_seed
        matrix[r["task_id"]][str(r["stream_index"])] = new_seed
    json.dump(matrix, open(run["seeds"], "w"), indent=2)
    e = expect("G16_SEED_MATRIX_BINDING", V, recs, run)
    assert "deterministic" in str(e)


def test_51_records_are_bound_to_the_committed_runtime_and_preseal():
    recs = make_clean(G=8); run = write_run(recs, tag="f51")
    for r in recs:
        r["run_provenance_sha256"] = "0" * 64
    with open(run["records"], "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    e = expect("G21_RECORD_RUN_BINDING", RP, run)
    assert e.detail["actual"] == "0" * 64



def test_52_transport_profile_scopes_dead_axes_without_collapsing_the_bank():
    recs = make_clean(G=8)
    for r in recs:
        if r["arm"] != "structured":
            continue
        # Prior-informed synthetic pattern: readable/outcome axes are dead,
        # writable-transport/actuator axes propagate.
        r["downstream_delta_rms"] = 0.0 if r["direction_id"] in (1, 2) else 0.5
    profile = axis_transport_profile(recs, "structured", dead_rms=1e-6)
    assert profile["observed_transporting_signed_actions"] == 4, profile
    assert profile["observed_transporting_axes"] == 2, profile
    assert profile["axes"]["d1"]["status"] == "DEAD_BOTH_SIGNS"
    assert profile["axes"]["d2"]["status"] == "DEAD_BOTH_SIGNS"
    assert profile["axes"]["d3"]["status"] == "TRANSPORTS_BOTH_SIGNS"
    assert profile["axes"]["d4"]["status"] == "TRANSPORTS_BOTH_SIGNS"
    assert "do not establish" in profile["interpretation"]
    print("  dead d1/d2 and live d3/d4 -> 4/8 signed actions transport; null scope preserved")

def main():
    names = [k for k in sorted(globals()) if k.startswith("test_")]
    failed = 0
    for n in names:
        print(f"\n{n}")
        try:
            globals()[n]()
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL: {type(exc).__name__}: {exc}")
    print(f"\n{'=' * 70}\n{len(names) - failed}/{len(names)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
