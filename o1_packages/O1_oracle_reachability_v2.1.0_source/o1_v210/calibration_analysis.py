"""
Executable calibration derivation for O1, v2.1.

The module turns raw calibration continuations plus run metadata into the exact
values later frozen in FREEZE_MANIFEST.json. It is deliberately deterministic:
fixed task-level bootstrap seed, exact Clopper-Pearson discordance bound, and
exact McNemar power from o1_analysis.

v2.1 changes, closing pre-calibration audit defects:
  * difficulty bands are half-open on BOTH strata: hard_medium [0.40, 0.55),
    medium [0.55, 0.70).  v2.0 made the medium top edge inclusive in code
    while the sealed design said [0.55, 0.70); the sealed design wins.
  * rows carry generated_text plus text/token digests; well_formed,
    verifier_correct and the bank oracle are recomputed here from the text
    with the sealed parser and verifier, and stored booleans that disagree are
    rejected (C10).
  * the sealed confirmatory candidate pool enters calibration: eligible pool
    capacity after banding (G_pool) caps G_confirm alongside G_power and
    G_max, the deterministic confirmatory allocation is derived HERE (no
    post-calibration operator discretion), and pool exhaustion produces the
    declared POOL_CAPACITY_CONSTRAINED state instead of a later deadlock.
  * budget_mde may return None (no detectable effect at the available G);
    that state is surfaced as mde_status, never coerced to a number.
  * per-signed-action transport and coupling-survival diagnostics are derived
    deterministically into the results (diagnostic only; nothing selects on
    them).

Raw calibration JSONL, one row per continuation:
    task_id, task_content_sha256, generator_setting_id,
    arm ("baseline" or "structured"), alpha,
    direction_id/sign for structured, stream_index, seed,
    well_formed, verifier_correct, generated_token_ids,
    generated_text, generated_text_sha256, generated_token_ids_sha256,
    n_generated_tokens, hit_max_tokens, downstream_delta_rms
    (null on baseline, finite >= 0 on structured),
    logits_finite, catastrophic_repetition, calibration_precommit_sha256

Each task has one baseline bank of K=8 and one structured bank of K=8 for every
alpha in action_space.alpha_swept_in_calibration. Seeds are index-matched across
all banks. Malformed continuations count as failures inside the bank oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from o1_analysis import (
    BANK_SIZE,
    MAX_GENERATED_TEXT_CHARS,
    IntegrityError,
    budget_mde,
    canonical_json_bytes,
    clopper_pearson_upper,
    derive_stream_seed,
    required_G,
    sha256_file,
    text_sha256,
    token_ids_sha256,
)
from o1_answer_parser_v2 import has_catastrophic_repetition, parse_final_answer
from o1_truth_table_verifier_v2 import verify_letter
from cohort_allocation import ALLOCATION_RULE_TEXT, allocate

CALIBRATION_SCHEMA_VERSION = "1.1"
ALPHA_LEVEL = 0.05
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260728
CALIBRATION_PRECOMMIT_SCHEMA_VERSION = "1.0"
CALIBRATION_CONFIG_PATHS = (
    "code.commit", "code.generation_module_sha256",
    "code.orchestrator_module_sha256", "code.transport_module_sha256",
    "code.numpy", "code.torch", "code.transformers",
    "code.repository_clean_required", "code.repository_diff_sha256",
    "model.checkpoint_sha256", "model.checkpoint_project_relative_tree_sha256",
    "model.loops", "model.dtype",
    "model.deterministic_flags",
    "generator.family", "generator.revision",
    "generator.hash_ordered_pool_offsets_calibration",
    "generator.structural_knobs", "generator.task_id_hash_scheme",
    "difficulty_strata.hard_medium", "difficulty_strata.medium",
    "difficulty_strata.min_well_formed_commitment_rate",
    "difficulty_strata.well_formed_noninferiority_margin",
    "decoding.temperature", "decoding.top_p", "decoding.max_new_tokens",
    "decoding.stopping_rule", "decoding.verifier", "decoding.answer_marker",
    "action_space.alpha_selection_rule",
    "action_space.alpha_swept_in_calibration",
    "action_space.writable_boundary", "action_space.normalization",
    "action_space.structured_axes",
    "bank.size", "bank.geometry", "bank.arms_required",
    "bank.arms_conditional_on_budget", "bank.stream_assignment",
    "bank.stream_assignment_rule",
    "cohorts.master_seed", "cohorts.seed_hash_function",
    "cohorts.seed_derivation_scheme",
    "cohorts.calibration_task_manifest_sha256",
    "cohorts.confirmatory_candidate_pool_sha256",
    "cohorts.seed_matrix_sha256",
    "artifact_hashes.tokenizer", "artifact_hashes.prompt_template",
    "artifact_hashes.parser", "artifact_hashes.verifier_implementation",
    "artifact_hashes.structured_axis_tensor",
    "artifact_hashes.random_axis_tensor",
    "transport.measurement_definition", "transport.measurement_loop",
    "transport.measurement_layer", "transport.norm_convention",
    "transport.token_position", "transport.normalized_by_injected_rms",
    "transport.dead_rms_threshold",
    "budget.max_wall_clock_days", "budget.calibration_inside_budget",
    "budget.rerun_reserve_fraction", "budget.max_plausible_candidates_per_hour",
    "power.delta_target", "power.target_power", "power.alpha_level",
    "calibration.bootstrap_draws", "calibration.bootstrap_seed",
)


class CalibrationError(IntegrityError):
    pass


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _get(raw: dict, dotted: str) -> Any:
    node: Any = raw
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise CalibrationError("C0_SPEC", f"missing calibration design key {dotted!r}")
        node = node[part]
    if node == "FREEZE_SLOT":
        raise CalibrationError("C0_SPEC", f"design key {dotted!r} is still FREEZE_SLOT")
    return node


def calibration_generation_config(manifest: dict) -> dict:
    """Exact pre-calibration design object; calibrated outputs are excluded."""
    return {path: _get(manifest, path) for path in CALIBRATION_CONFIG_PATHS}


def verify_calibration_precommit(precommit_path: str, manifest: dict,
                                 task_manifest_path: str) -> dict:
    if not precommit_path or not os.path.exists(precommit_path):
        raise CalibrationError("C0_PRECOMMIT",
                               "CALIBRATION_PRECOMMIT.json is required")
    obj = _load_json(precommit_path)
    if obj.get("schema_version") != CALIBRATION_PRECOMMIT_SCHEMA_VERSION:
        raise CalibrationError("C0_PRECOMMIT", "unsupported calibration precommit schema")
    want_cfg = calibration_generation_config(manifest)
    if obj.get("resolved_calibration_config") != want_cfg:
        bad = [k for k in sorted(set(want_cfg) | set(obj.get("resolved_calibration_config", {})))
               if obj.get("resolved_calibration_config", {}).get(k) != want_cfg.get(k)]
        raise CalibrationError("C0_PRECOMMIT",
                               f"calibration configuration disagrees in {bad}",
                               detail={"keys": bad})
    want_task = sha256_file(task_manifest_path)
    if obj.get("calibration_task_manifest_sha256") != want_task:
        raise CalibrationError("C0_PRECOMMIT",
                               "calibration task manifest hash disagrees")
    expected_att = {
        "generation_module_sha256": _get(manifest, "code.generation_module_sha256"),
        "repository_clean": bool(_get(manifest, "code.repository_clean_required")),
        "repository_diff_sha256": _get(manifest, "code.repository_diff_sha256"),
    }
    if obj.get("runtime_attestation") != expected_att:
        raise CalibrationError("C0_PRECOMMIT",
                               "calibration runtime attestation disagrees")
    allowed = {"schema_version", "created_at_utc",
               "resolved_calibration_config", "runtime_attestation",
               "calibration_task_manifest_sha256"}
    extra = sorted(set(obj) - allowed)
    if extra:
        raise CalibrationError("C0_PRECOMMIT",
                               f"undeclared calibration precommit fields {extra}")
    return obj


def _R(r: dict) -> int:
    return int(bool(r["well_formed"]) and bool(r["verifier_correct"]))


def _validate_records(records: list[dict], grid: list[float], manifest: dict,
                      precommit_sha256: str) -> None:
    required = {
        "task_id": str,
        "task_content_sha256": str,
        "generator_setting_id": str,
        "arm": str,
        "alpha": (int, float),
        "stream_index": int,
        "seed": int,
        "well_formed": bool,
        "verifier_correct": bool,
        "generated_token_ids": list,
        "generated_text": str,
        "generated_text_sha256": str,
        "generated_token_ids_sha256": str,
        "n_generated_tokens": int,
        "hit_max_tokens": bool,
        "logits_finite": bool,
        "catastrophic_repetition": bool,
        "calibration_precommit_sha256": str,
    }
    if not records:
        raise CalibrationError("C1_SCHEMA", "calibration record set is empty")
    for idx, r in enumerate(records):
        for k, typ in required.items():
            if k not in r or not isinstance(r[k], typ) or (
                    typ in (int, (int, float)) and isinstance(r[k], bool)):
                raise CalibrationError("C1_SCHEMA", f"row {idx}: invalid or missing {k!r}")
        if r["arm"] not in ("baseline", "structured"):
            raise CalibrationError("C1_SCHEMA", f"row {idx}: unknown arm {r['arm']!r}")
        if not (0 <= r["stream_index"] < BANK_SIZE):
            raise CalibrationError("C1_SCHEMA", f"row {idx}: stream_index outside 0..7")
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in r["generated_token_ids"]):
            raise CalibrationError("C1_SCHEMA", f"row {idx}: generated_token_ids must be ints")
        if r["n_generated_tokens"] != len(r["generated_token_ids"]):
            raise CalibrationError(
                "C1_SCHEMA",
                f"row {idx}: n_generated_tokens != len(generated_token_ids)")
        if len(r["generated_text"]) > MAX_GENERATED_TEXT_CHARS:
            raise CalibrationError(
                "C1_SCHEMA",
                f"row {idx}: generated_text is {len(r['generated_text'])} chars, "
                f"above the {MAX_GENERATED_TEXT_CHARS}-char schema ceiling")
        if r["n_generated_tokens"] > int(_get(manifest, "decoding.max_new_tokens")):
            raise CalibrationError(
                "C1_SCHEMA",
                f"row {idx}: {r['n_generated_tokens']} generated tokens exceed "
                f"the sealed decoding.max_new_tokens")
        if text_sha256(r["generated_text"]) != r["generated_text_sha256"]:
            raise CalibrationError(
                "C10_TEXT_RECOMPUTATION",
                f"row {idx}: generated_text does not hash to generated_text_sha256")
        if token_ids_sha256(r["generated_token_ids"]) != r["generated_token_ids_sha256"]:
            raise CalibrationError(
                "C10_TEXT_RECOMPUTATION",
                f"row {idx}: token ids do not hash to generated_token_ids_sha256")
        tv = r.get("downstream_delta_rms")
        if r["arm"] == "baseline":
            if tv is not None:
                raise CalibrationError(
                    "C1_SCHEMA", f"row {idx}: baseline rows carry null "
                    f"downstream_delta_rms")
        else:
            if (tv is None or isinstance(tv, bool)
                    or not isinstance(tv, (int, float))
                    or not math.isfinite(float(tv)) or float(tv) < 0.0):
                raise CalibrationError(
                    "C1_SCHEMA",
                    f"row {idx}: structured rows require a finite non-negative "
                    f"downstream_delta_rms (got {tv!r})")
        if r["calibration_precommit_sha256"] != precommit_sha256:
            raise CalibrationError("C0_PRECOMMIT",
                                   f"row {idx}: calibration_precommit_sha256 disagrees")
        want_seed = derive_stream_seed(_get(manifest, "cohorts.master_seed"),
                                       r["task_id"], r["stream_index"])
        if r["seed"] != want_seed:
            raise CalibrationError("C4_CRN",
                                   f"row {idx}: seed {r['seed']} != deterministic {want_seed}")
        if r["arm"] == "baseline":
            if not math.isclose(float(r["alpha"]), 0.0, abs_tol=1e-12):
                raise CalibrationError("C2_GEOMETRY", f"row {idx}: baseline alpha must be 0")
            if r.get("direction_id") is not None or r.get("sign") is not None:
                raise CalibrationError("C2_GEOMETRY", f"row {idx}: baseline direction/sign must be null")
        else:
            if float(r["alpha"]) not in grid:
                raise CalibrationError("C2_GEOMETRY", f"row {idx}: alpha not in sealed sweep")
            if r.get("direction_id") not in (1, 2, 3, 4) or r.get("sign") not in (-1, 1):
                raise CalibrationError("C2_GEOMETRY", f"row {idx}: structured geometry invalid")

    tasks: dict[str, dict] = {}
    for r in records:
        t = tasks.setdefault(r["task_id"], {
            "content": r["task_content_sha256"],
            "setting": r["generator_setting_id"],
            "baseline": [],
            "structured": {a: [] for a in grid},
        })
        if t["content"] != r["task_content_sha256"] or t["setting"] != r["generator_setting_id"]:
            raise CalibrationError("C3_TASK_BINDING", f"task {r['task_id']}: content/setting changes across rows")
        if r["arm"] == "baseline":
            t["baseline"].append(r)
        else:
            t["structured"][float(r["alpha"])].append(r)

    content = [t["content"] for t in tasks.values()]
    if len(content) != len(set(content)):
        raise CalibrationError("C3_TASK_BINDING", "duplicate task_content_sha256 within calibration cohort")

    want_geom = sorted((d, s) for d in range(1, 5) for s in (1, -1))
    for task_id, t in tasks.items():
        base = t["baseline"]
        if len(base) != BANK_SIZE:
            raise CalibrationError("C2_GEOMETRY", f"task {task_id}: baseline bank size {len(base)} != 8")
        base_idx = sorted(r["stream_index"] for r in base)
        if base_idx != list(range(BANK_SIZE)) or len({r["seed"] for r in base}) != BANK_SIZE:
            raise CalibrationError("C2_GEOMETRY", f"task {task_id}: baseline streams/seeds are not eight unique values")
        bseed = {r["stream_index"]: r["seed"] for r in base}
        for a in grid:
            rows = t["structured"][a]
            if len(rows) != BANK_SIZE:
                raise CalibrationError("C2_GEOMETRY", f"task {task_id} alpha {a}: bank size {len(rows)} != 8")
            if sorted(r["stream_index"] for r in rows) != list(range(BANK_SIZE)):
                raise CalibrationError("C2_GEOMETRY", f"task {task_id} alpha {a}: streams are not 0..7")
            if sorted((r["direction_id"], r["sign"]) for r in rows) != want_geom:
                raise CalibrationError("C2_GEOMETRY", f"task {task_id} alpha {a}: not four antipodal pairs")
            for r in rows:
                if r["seed"] != bseed[r["stream_index"]]:
                    raise CalibrationError("C4_CRN", f"task {task_id} alpha {a} stream {r['stream_index']}: seed not baseline-matched")


def _task_banks(records: list[dict], grid: list[float]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        t = out.setdefault(r["task_id"], {
            "content": r["task_content_sha256"],
            "setting": r["generator_setting_id"],
            "baseline": [],
            "structured": {a: [] for a in grid},
        })
        if r["arm"] == "baseline":
            t["baseline"].append(r)
        else:
            t["structured"][float(r["alpha"])].append(r)
    return out


def _bootstrap_lower(values: np.ndarray, seed: int = BOOTSTRAP_SEED,
                     n_draws: int = BOOTSTRAP_DRAWS, alpha: float = ALPHA_LEVEL) -> float:
    if values.size == 0:
        return float("-inf")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_draws, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha))


def _difficulty_assignment(tasks: dict[str, dict], bands: dict[str, list[float]]) -> tuple[dict[str, list[str]], dict[str, dict]]:
    settings: dict[str, list[dict]] = {}
    for t in tasks.values():
        settings.setdefault(t["setting"], []).append(t)
    metrics: dict[str, dict] = {}
    selected = {"hard_medium": [], "medium": []}
    for setting, rows in sorted(settings.items()):
        y = [max(_R(r) for r in t["baseline"]) for t in rows]
        wf = [np.mean([r["well_formed"] for r in t["baseline"]]) for t in rows]
        p = float(np.mean(y))
        metrics[setting] = {
            "n_tasks": len(rows),
            "baseline_any_correct_of_8": p,
            "baseline_well_formed_rate": float(np.mean(wf)),
        }
        for stratum in ("hard_medium", "medium"):
            lo, hi = [float(x) for x in bands[stratum]]
            # Half-open [lo, hi) on BOTH strata. The sealed design states
            # medium = [0.55, 0.70); v2.0 code made the medium top edge
            # inclusive, contradicting the precommitted band. The design wins.
            if lo <= p < hi:
                selected[stratum].append(setting)
    return selected, metrics


def _selected_tasks(tasks: dict[str, dict], selected: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    setting_to_stratum = {}
    for st, vals in selected.items():
        for s in vals:
            if s in setting_to_stratum:
                raise CalibrationError("C5_DIFFICULTY", f"setting {s!r} falls in multiple strata")
            setting_to_stratum[s] = st
    tids = [tid for tid, t in tasks.items() if t["setting"] in setting_to_stratum]
    return sorted(tids), setting_to_stratum


def _coherence_for_alpha(tasks: dict[str, dict], tids: list[str], setting_to_stratum: dict[str, str], alpha: float,
                         wf_floor: float, margin: float, bootstrap_draws: int,
                         bootstrap_seed: int) -> dict:
    by_st = {"hard_medium": [], "medium": []}
    hard_failures: list[str] = []
    for tid in tids:
        t = tasks[tid]
        st = setting_to_stratum[t["setting"]]
        rows = t["structured"][alpha]
        base = t["baseline"]
        if not all(r["logits_finite"] for r in rows):
            hard_failures.append(f"{tid}:nonfinite_logits")
        if any(r["catastrophic_repetition"] for r in rows):
            hard_failures.append(f"{tid}:catastrophic_repetition")
        # Text-bound, not token-bound: the coherence gates that select alpha
        # must rest on the same quantity the scoring gates recompute.
        if any(r["generated_text"] == "" for r in rows):
            hard_failures.append(f"{tid}:empty_continuation")
        by_st[st].append((
            float(np.mean([r["well_formed"] for r in rows])),
            float(np.mean([r["well_formed"] for r in base])),
        ))
    strata = {}
    passed = not hard_failures
    reasons = list(hard_failures)
    for st, vals in by_st.items():
        if not vals:
            passed = False
            reasons.append(f"{st}:no_selected_tasks")
            strata[st] = {"n_tasks": 0}
            continue
        arr = np.asarray(vals, dtype=float)
        int_rate = float(arr[:, 0].mean())
        base_rate = float(arr[:, 1].mean())
        diff = arr[:, 0] - arr[:, 1]
        lower = _bootstrap_lower(
            diff, seed=bootstrap_seed + int(round(alpha * 1e6)),
            n_draws=bootstrap_draws,
        )
        st_pass = int_rate >= wf_floor and lower >= margin
        if not st_pass:
            passed = False
            if int_rate < wf_floor:
                reasons.append(f"{st}:well_formed_rate={int_rate:.6f}<floor={wf_floor:.6f}")
            if lower < margin:
                reasons.append(f"{st}:noninferiority_lower={lower:.6f}<margin={margin:.6f}")
        strata[st] = {
            "n_tasks": int(arr.shape[0]),
            "intervention_well_formed_rate": int_rate,
            "baseline_well_formed_rate": base_rate,
            "paired_difference": float(diff.mean()),
            "one_sided_95_lower_bound": lower,
            "pass": st_pass,
        }
    return {"pass": passed, "strata": strata, "reasons": sorted(set(reasons))}


def _recompute_scoring(records: list[dict], sealed: dict[str, dict]) -> None:
    """C10: recompute parser/verifier outcomes from stored text; reject drift."""
    for tid, row in sealed.items():
        for key in ("options", "symbolic"):
            if key not in row:
                raise CalibrationError(
                    "C10_TEXT_RECOMPUTATION",
                    f"sealed calibration task {tid} lacks {key!r}; the manifest "
                    f"must carry full task content so scoring is recomputed, "
                    f"not trusted")
    for idx, r in enumerate(records):
        task = sealed[r["task_id"]]
        parsed = parse_final_answer(r["generated_text"])
        if "parsed_answer" in r and r.get("parsed_answer") != parsed:
            raise CalibrationError(
                "C10_TEXT_RECOMPUTATION",
                f"row {idx}: stored parsed_answer={r.get('parsed_answer')!r} "
                f"but the sealed parser returns {parsed!r}")
        if bool(r["well_formed"]) != (parsed is not None):
            raise CalibrationError(
                "C10_TEXT_RECOMPUTATION",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): stored "
                f"well_formed={r['well_formed']} but the sealed parser derives "
                f"{parsed is not None}")
        want_vc = verify_letter(task, parsed)
        if bool(r["verifier_correct"]) != want_vc:
            raise CalibrationError(
                "C10_TEXT_RECOMPUTATION",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): stored "
                f"verifier_correct={r['verifier_correct']} but the sealed "
                f"verifier derives {want_vc}")
        want_rep = has_catastrophic_repetition(r["generated_token_ids"])
        if bool(r["catastrophic_repetition"]) != want_rep:
            raise CalibrationError(
                "C10_TEXT_RECOMPUTATION",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): stored "
                f"catastrophic_repetition={r['catastrophic_repetition']} but "
                f"the sealed mechanical flag derives {want_rep}; a forged flag "
                f"could flip alpha selection")


def _transport_by_alpha(records: list[dict], grid: list[float],
                        dead_rms: float) -> dict:
    """Descriptive per-signed-action transport at every swept magnitude."""
    out: dict[str, dict] = {}
    for a in grid:
        per: dict[str, list[float]] = {}
        for r in records:
            if r["arm"] != "structured" or float(r["alpha"]) != a:
                continue
            key = f"d{r['direction_id']}{'+' if r['sign'] == 1 else '-'}"
            per.setdefault(key, []).append(float(r["downstream_delta_rms"]))
        summary = {}
        for key, vals in sorted(per.items()):
            arr = np.asarray(vals, dtype=np.float64)
            summary[key] = {
                "n": int(arr.size),
                "median_rms": float(np.median(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "dead_rate": float(np.mean(arr <= dead_rms)),
            }
        out[str(a)] = summary
    return out


def _coupling_by_alpha(records: list[dict], grid: list[float]) -> dict:
    """Descriptive CRN coupling survival: first token divergence vs baseline."""
    base = {(r["task_id"], r["stream_index"]): r["generated_token_ids"]
            for r in records if r["arm"] == "baseline"}
    out: dict[str, dict] = {}
    for a in grid:
        divs: list[float] = []
        for r in records:
            if r["arm"] != "structured" or float(r["alpha"]) != a:
                continue
            b = base[(r["task_id"], r["stream_index"])]
            g = r["generated_token_ids"]
            d = next((t for t, (x, y) in enumerate(zip(g, b)) if x != y), None)
            if d is None and len(g) != len(b):
                d = min(len(g), len(b))
            divs.append(math.inf if d is None else float(d))
        arr = np.asarray(divs, dtype=np.float64)
        fin = arr[np.isfinite(arr)]
        out[str(a)] = {
            "n": int(arr.size),
            "p_no_divergence": float(np.mean(~np.isfinite(arr))) if arr.size else None,
            "survival": {str(t): float(np.mean(arr > t)) for t in (0, 1, 8, 32, 128)},
            "median_T_div": float(np.median(fin)) if fin.size else None,
        }
    return out


def _load_candidate_pool(path: str, manifest: dict,
                         calibration_tasks: dict[str, dict]) -> list[dict]:
    """Load and validate the sealed outcome-blind confirmatory candidate pool."""
    want = _get(manifest, "cohorts.confirmatory_candidate_pool_sha256")
    got = sha256_file(path)
    if got != want:
        raise CalibrationError(
            "C11_CANDIDATE_POOL",
            f"candidate pool hashes {got[:16]}... but the sealed value is "
            f"{str(want)[:16]}...")
    pool = _load_jsonl(path)
    if not pool:
        raise CalibrationError("C11_CANDIDATE_POOL", "candidate pool is empty")
    ids = [r.get("task_id") for r in pool]
    ch = [r.get("task_content_sha256") for r in pool]
    if (any(not isinstance(x, str) or not x for x in ids + ch)
            or len(set(ids)) != len(ids) or len(set(ch)) != len(ch)):
        raise CalibrationError(
            "C11_CANDIDATE_POOL", "candidate pool identities are invalid or duplicated")
    for r in pool:
        if not isinstance(r.get("generator_setting_id"), str):
            raise CalibrationError(
                "C11_CANDIDATE_POOL", f"pool task {r.get('task_id')} lacks a setting")
        rk = r.get("sealed_rank_within_generator_setting")
        if isinstance(rk, bool) or not isinstance(rk, int) or rk < 0:
            raise CalibrationError(
                "C11_CANDIDATE_POOL",
                f"pool task {r.get('task_id')} lacks a valid "
                f"sealed_rank_within_generator_setting")
    cal_content = {t["content"] for t in calibration_tasks.values()}
    shared = cal_content & set(ch)
    if shared:
        raise CalibrationError(
            "C11_CANDIDATE_POOL",
            f"{len(shared)} candidate task(s) share content with the "
            f"calibration cohort")
    return pool


def compute_calibration(records_path: str, metadata_path: str, manifest_path: str,
                        task_manifest_path: str, precommit_path: str,
                        candidate_pool_path: str) -> dict:
    manifest = _load_json(manifest_path)
    records = _load_jsonl(records_path)
    metadata = _load_json(metadata_path)
    verify_calibration_precommit(precommit_path, manifest, task_manifest_path)
    precommit_sha = sha256_file(precommit_path)
    allowed_meta = {"elapsed_wall_seconds", "calibration_precommit_sha256"}
    if set(metadata) != allowed_meta:
        raise CalibrationError("C6_METADATA",
                               f"calibration metadata keys must be exactly {sorted(allowed_meta)}")
    if metadata.get("calibration_precommit_sha256") != precommit_sha:
        raise CalibrationError("C0_PRECOMMIT",
                               "calibration metadata is not bound to CALIBRATION_PRECOMMIT.json")

    grid = [float(x) for x in _get(manifest, "action_space.alpha_swept_in_calibration")]
    _validate_records(records, grid, manifest, precommit_sha)
    tasks = _task_banks(records, grid)
    task_manifest = _load_jsonl(task_manifest_path)
    sealed = {r.get("task_id"): r for r in task_manifest}
    if len(sealed) != len(task_manifest) or set(sealed) != set(tasks):
        raise CalibrationError(
            "C3_TASK_BINDING",
            "raw calibration task set does not equal the sealed calibration task manifest")
    for tid, t in tasks.items():
        rec = sealed[tid]
        if rec.get("task_content_sha256") != t["content"]:
            raise CalibrationError("C3_TASK_BINDING", f"task {tid}: content hash differs from sealed calibration manifest")
        if rec.get("generator_setting_id") != t["setting"]:
            raise CalibrationError("C3_TASK_BINDING", f"task {tid}: generator setting differs from sealed calibration manifest")

    _recompute_scoring(records, sealed)
    pool = _load_candidate_pool(candidate_pool_path, manifest, tasks)

    bands = {
        "hard_medium": _get(manifest, "difficulty_strata.hard_medium"),
        "medium": _get(manifest, "difficulty_strata.medium"),
    }
    selected, setting_metrics = _difficulty_assignment(tasks, bands)
    tids, setting_to_stratum = _selected_tasks(tasks, selected)

    wf_floor = float(_get(manifest, "difficulty_strata.min_well_formed_commitment_rate"))
    margin = float(_get(manifest, "difficulty_strata.well_formed_noninferiority_margin"))
    bootstrap_draws = int(_get(manifest, "calibration.bootstrap_draws"))
    bootstrap_seed = int(_get(manifest, "calibration.bootstrap_seed"))
    if bootstrap_draws < 100:
        raise CalibrationError("C0_SPEC", "calibration.bootstrap_draws must be >= 100")
    alpha_metrics = {
        str(a): _coherence_for_alpha(
            tasks, tids, setting_to_stratum, a, wf_floor, margin,
            bootstrap_draws, bootstrap_seed,
        ) for a in grid
    }
    passing = sorted(a for a in grid if alpha_metrics[str(a)]["pass"])
    failing = sorted(a for a in grid if a not in passing)
    failure_reasons = {str(a): "; ".join(alpha_metrics[str(a)]["reasons"]) or "coherence criterion failed" for a in failing}

    if not selected["hard_medium"] or not selected["medium"]:
        calibration_verdict = "NO_DIFFICULTY_SETTINGS"
        alpha_star = None
        alpha_secondary = None
    elif not passing:
        calibration_verdict = "NO_COHERENT_ALPHA"
        alpha_star = None
        alpha_secondary = None
    else:
        alpha_star = max(passing)
        lower = [a for a in passing if a < alpha_star]
        alpha_secondary = max(lower) if lower else None
        calibration_verdict = "PENDING_POWER"

    n10 = n01 = 0
    q_disc = q_upper = 0.0
    if alpha_star is not None:
        for tid in tids:
            t = tasks[tid]
            yb = max(_R(r) for r in t["baseline"])
            yi = max(_R(r) for r in t["structured"][alpha_star])
            n10 += int(yi == 1 and yb == 0)
            n01 += int(yi == 0 and yb == 1)
        G_cal = len(tids)
        n_disc = n10 + n01
        q_disc = n_disc / G_cal if G_cal else 0.0
        q_upper = clopper_pearson_upper(n_disc, G_cal, ALPHA_LEVEL) if G_cal else 1.0
    else:
        G_cal = len(tids)
        n_disc = 0

    elapsed = float(metadata.get("elapsed_wall_seconds", 0.0))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise CalibrationError("C6_METADATA", "elapsed_wall_seconds must be finite and > 0")
    total_candidates = len(records)
    throughput = total_candidates / elapsed * 3600.0
    # elapsed_wall_seconds is an operator-side measurement with no independent
    # attestation; an implausibly small value would inflate G_max and could
    # flip PROCEED_BUDGET_CONSTRAINED into PROCEED_POWERED. The sealed
    # plausibility ceiling bounds that channel (red-team finding, v2.1).
    thr_ceiling = float(_get(manifest, "budget.max_plausible_candidates_per_hour"))
    if not math.isfinite(thr_ceiling) or thr_ceiling <= 0:
        raise CalibrationError("C0_SPEC",
                               "budget.max_plausible_candidates_per_hour must be finite and > 0")
    if throughput > thr_ceiling:
        raise CalibrationError(
            "C6_METADATA",
            f"measured throughput {throughput:.1f} candidates/hour exceeds the "
            f"sealed plausibility ceiling {thr_ceiling:.1f}; elapsed_wall_seconds "
            f"is not credible")
    max_days = float(_get(manifest, "budget.max_wall_clock_days"))
    reserve = float(_get(manifest, "budget.rerun_reserve_fraction"))
    inside = bool(_get(manifest, "budget.calibration_inside_budget"))
    if not (0 <= reserve < 1):
        raise CalibrationError("C6_METADATA", "rerun_reserve_fraction must lie in [0,1)")
    available_seconds = max_days * 86400.0 - (elapsed if inside else 0.0)
    available_seconds = max(0.0, available_seconds) * (1.0 - reserve)
    required_arms = list(_get(manifest, "bank.arms_required"))
    conditional_arms = list(_get(manifest, "bank.arms_conditional_on_budget"))
    if len(set(required_arms + conditional_arms)) != len(required_arms + conditional_arms):
        raise CalibrationError("C0_SPEC", "required and conditional arm lists overlap or contain duplicates")
    if set(("baseline", "structured", "zero_alpha")) - set(required_arms):
        raise CalibrationError("C0_SPEC", "bank.arms_required must include baseline, structured, zero_alpha")
    full_arms = required_arms + conditional_arms
    arm_plans = [full_arms]
    remaining = list(conditional_arms)
    for drop in conditional_arms:
        remaining = [a for a in remaining if a != drop]
        arm_plans.append(required_arms + remaining)
    # Remove duplicate plans while preserving order.
    unique_plans = []
    for plan in arm_plans:
        if plan not in unique_plans:
            unique_plans.append(plan)
    capacity_by_plan = {}
    for plan in unique_plans:
        candidates_per_task = BANK_SIZE * len(plan)
        capacity_by_plan[tuple(plan)] = int(math.floor(
            available_seconds * throughput / 3600.0 / candidates_per_task))

    delta = float(_get(manifest, "power.delta_target"))
    target_power = float(_get(manifest, "power.target_power"))
    alpha_level = float(_get(manifest, "power.alpha_level"))

    # Eligible banded pool capacity. Whole settings are banded, so eligibility
    # is per setting; G_pool is the total candidate count over the selected
    # settings. It caps G_confirm alongside time capacity (v2.0 had no pool
    # cap and could size a cohort the sealed pool cannot supply).
    setting_to_stratum_pool = {}
    for st, vals in selected.items():
        for s in vals:
            setting_to_stratum_pool[s] = st
    pool_counts: dict[str, int] = {}
    for row in pool:
        s = row["generator_setting_id"]
        if s in setting_to_stratum_pool:
            pool_counts[s] = pool_counts.get(s, 0) + 1
    G_pool = sum(pool_counts.values())

    G_power = None
    G_confirm = None
    G_max = capacity_by_plan[tuple(required_arms)]
    selected_arms = list(required_arms)
    mde = None
    mde_status = "NOT_APPLICABLE"
    dropped: list[str] = list(conditional_arms)
    powered = False
    constrained_by: list[str] = []
    allocation = None

    def _mde_at(g: int):
        if g < 1:
            return None, "NO_DETECTABLE_EFFECT_AT_G"
        value = budget_mde(g, q_upper, target_power, alpha_level)
        if value is None:
            return None, "NO_DETECTABLE_EFFECT_AT_G"
        return value, "CONSTRAINED_MDE"

    if calibration_verdict == "PENDING_POWER":
        if q_upper < delta:
            calibration_verdict = "TARGET_HEADROOM_EXCLUDED_BY_DISCORDANCE_BOUND"
            G_power = 0
            G_confirm = 0
            mde, mde_status = None, "TARGET_EXCLUDED_BY_DISCORDANCE_BOUND"
        elif G_pool < 1:
            calibration_verdict = "NO_ELIGIBLE_POOL_CAPACITY"
            G_power = 0
            G_confirm = 0
            mde, mde_status = None, "NO_ELIGIBLE_POOL_CAPACITY"
        else:
            G_power = required_G(q_upper, delta, target_power, alpha_level)
            if G_power == -1:
                calibration_verdict = "POWER_TARGET_UNREACHABLE"
                G_confirm = min(G_max, G_pool)
                constrained_by = (["time"] if G_max <= G_pool else []) + (
                    ["pool"] if G_pool <= G_max else [])
                mde, mde_status = _mde_at(G_confirm)
            elif G_power <= G_pool:
                chosen = None
                for plan in unique_plans:
                    cap = capacity_by_plan[tuple(plan)]
                    if G_power <= cap:
                        chosen = plan
                        break
                if chosen is not None:
                    selected_arms = list(chosen)
                    G_max = capacity_by_plan[tuple(chosen)]
                    dropped = [a for a in conditional_arms if a not in selected_arms]
                    calibration_verdict = "PROCEED_POWERED"
                    G_confirm = G_power
                    mde, mde_status = delta, "TARGET_MET"
                    powered = True
                else:
                    selected_arms = list(required_arms)
                    G_max = capacity_by_plan[tuple(required_arms)]
                    dropped = list(conditional_arms)
                    calibration_verdict = "PROCEED_BUDGET_CONSTRAINED"
                    G_confirm = min(G_max, G_pool)
                    constrained_by = ["time"] + (["pool"] if G_pool < G_max else [])
                    mde, mde_status = _mde_at(G_confirm)
            else:
                # Pool-constrained below the powered size: keep the largest arm
                # plan whose time capacity covers the pool-sized run.
                chosen = None
                for plan in unique_plans:
                    cap = capacity_by_plan[tuple(plan)]
                    if G_pool <= cap:
                        chosen = plan
                        break
                if chosen is not None:
                    selected_arms = list(chosen)
                    G_max = capacity_by_plan[tuple(chosen)]
                    dropped = [a for a in conditional_arms if a not in selected_arms]
                    G_confirm = G_pool
                    constrained_by = ["pool"]
                else:
                    selected_arms = list(required_arms)
                    G_max = capacity_by_plan[tuple(required_arms)]
                    dropped = list(conditional_arms)
                    G_confirm = min(G_max, G_pool)
                    constrained_by = ["pool", "time"]
                calibration_verdict = "PROCEED_BUDGET_CONSTRAINED"
                mde, mde_status = _mde_at(G_confirm)

    if (calibration_verdict in ("PROCEED_POWERED", "PROCEED_BUDGET_CONSTRAINED")
            and int(G_confirm) < 1):
        # Time capacity collapsed to zero tasks: an explicit halt, not a
        # "proceed" verdict with an unallocatable cohort (red-team finding).
        calibration_verdict = "NO_TIME_CAPACITY"
        mde, mde_status = None, "NO_TIME_CAPACITY"

    if calibration_verdict in ("PROCEED_POWERED", "PROCEED_BUDGET_CONSTRAINED"):
        allocation = allocate(selected, pool_counts, int(G_confirm))
        allocation["G_pool"] = G_pool
        allocation["eligible_pool_counts"] = dict(sorted(pool_counts.items()))
        allocation["constrained_by"] = constrained_by
        allocation["pool_capacity_status"] = (
            "POOL_CAPACITY_CONSTRAINED" if "pool" in constrained_by
            else "POOL_CAPACITY_OK")
        allocation["allocation_rule"] = ALLOCATION_RULE_TEXT

    result = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_hashes": {
            "records_sha256": sha256_file(records_path),
            "metadata_sha256": sha256_file(metadata_path),
            "task_manifest_sha256": sha256_file(task_manifest_path),
            "candidate_pool_sha256": sha256_file(candidate_pool_path),
            "precommit_sha256": precommit_sha,
            "manifest_design_sha256": hashlib.sha256(_canonical_json_bytes({
                "alpha_grid": grid,
                "bands": bands,
                "wf_floor": wf_floor,
                "wf_margin": margin,
                "budget": {
                    "max_wall_clock_days": max_days,
                    "calibration_inside_budget": inside,
                    "rerun_reserve_fraction": reserve,
                },
                "power": {
                    "delta_target": delta,
                    "target_power": target_power,
                    "alpha_level": alpha_level,
                },
                "bootstrap": {
                    "draws": bootstrap_draws,
                    "seed": bootstrap_seed,
                },
            })).hexdigest(),
        },
        "difficulty": {
            "selected_settings_per_stratum": selected,
            "setting_metrics": setting_metrics,
            "n_selected_tasks": G_cal,
        },
        "alpha": {
            "grid": grid,
            "metrics": alpha_metrics,
            "passing_alphas": passing,
            "failing_alphas": failing,
            "failure_reasons": failure_reasons,
            "alpha_star": alpha_star,
            "alpha_secondary": alpha_secondary,
            "selection_rule": "largest_passing",
        },
        "discordance": {
            "n10": n10,
            "n01": n01,
            "n_disc": n_disc,
            "G_calibration": G_cal,
            "q_disc": q_disc,
            "q_disc_upper_confidence_bound": q_upper,
            "confidence_level": 0.95,
            "bound_method": "exact Clopper-Pearson upper bound on n_disc/G",
        },
        "throughput": {
            "total_candidates": total_candidates,
            "elapsed_wall_seconds": elapsed,
            "measured_candidates_per_hour": throughput,
            "available_confirmatory_seconds_after_reserve": available_seconds,
            "capacity_by_arm_plan": {"|".join(k): v for k, v in capacity_by_plan.items()},
            "G_max": G_max,
        },
        "power": {
            "delta_target": delta,
            "target_power": target_power,
            "alpha_level": alpha_level,
            "G_power": G_power,
            "G_max": G_max,
            "G_pool": G_pool,
            "G_confirm": G_confirm,
            "constrained_by": constrained_by,
            "budget_mde_if_constrained": mde,
            "mde_status": mde_status,
            "powered_for_target": powered,
            "dropped_arms": dropped,
            "selected_confirmatory_arms": selected_arms,
        },
        "confirmatory_allocation": allocation,
        "transport_by_alpha": _transport_by_alpha(
            records, grid, float(_get(manifest, "transport.dead_rms_threshold"))),
        "coupling_by_alpha": _coupling_by_alpha(records, grid),
        "verdict": calibration_verdict,
    }
    return result


def write_results(result: dict, path: str) -> None:
    if os.path.exists(path):
        raise CalibrationError("C7_IMMUTABLE", f"refusing to overwrite existing {path}")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def apply_results_to_manifest(manifest_path: str, results_path: str, output_path: str) -> None:
    if os.path.exists(output_path):
        raise CalibrationError("C7_IMMUTABLE", f"refusing to overwrite existing {output_path}")
    m = _load_json(manifest_path)
    r = _load_json(results_path)
    m["action_space"]["alpha_star"] = r["alpha"]["alpha_star"]
    m["action_space"]["alpha_secondary"] = r["alpha"]["alpha_secondary"]
    m["action_space"]["alpha_calibration"]["passing_alphas"] = r["alpha"]["passing_alphas"]
    m["action_space"]["alpha_calibration"]["failing_alphas"] = r["alpha"]["failing_alphas"]
    m["action_space"]["alpha_calibration"]["failure_reasons"] = r["alpha"]["failure_reasons"]
    m["difficulty_strata"]["generator_settings_per_stratum"] = r["difficulty"]["selected_settings_per_stratum"]
    alloc = r.get("confirmatory_allocation")
    m["difficulty_strata"]["expected_counts"] = (
        alloc["expected_counts"] if alloc else {"hard_medium": 0, "medium": 0})
    m["power"]["q_disc_calibrated"] = r["discordance"]["q_disc"]
    m["power"]["q_disc_upper_confidence_bound"] = r["discordance"]["q_disc_upper_confidence_bound"]
    m["power"]["G_power"] = r["power"]["G_power"]
    m["power"]["G_max"] = r["power"]["G_max"]
    m["power"]["G_pool"] = r["power"]["G_pool"]
    m["power"]["G_confirm"] = r["power"]["G_confirm"]
    m["power"]["budget_mde_if_constrained"] = r["power"]["budget_mde_if_constrained"]
    m["power"]["calibration_verdict"] = r["verdict"]
    m["bank"]["arms_confirmatory"] = r["power"]["selected_confirmatory_arms"]
    m["budget"]["measured_candidates_per_hour"] = r["throughput"]["measured_candidates_per_hour"]
    m.setdefault("calibration", {})["precommit_sha256"] = r["source_hashes"]["precommit_sha256"]
    m["calibration"]["raw_records_sha256"] = r["source_hashes"]["records_sha256"]
    m["calibration"]["metadata_sha256"] = r["source_hashes"]["metadata_sha256"]
    m["calibration"]["results_sha256"] = sha256_file(results_path)
    with open(output_path + ".tmp", "w") as fh:
        json.dump(m, fh, indent=2)
        fh.write("\n")
    os.replace(output_path + ".tmp", output_path)


def verify_calibration_binding(records_path: str, metadata_path: str, results_path: str,
                               manifest_path: str, task_manifest_path: str,
                               precommit_path: str, candidate_pool_path: str) -> dict:
    recomputed = compute_calibration(records_path, metadata_path, manifest_path,
                                     task_manifest_path, precommit_path,
                                     candidate_pool_path)
    stored = _load_json(results_path)
    if _canonical_json_bytes(stored) != _canonical_json_bytes(recomputed):
        raise CalibrationError("C8_RESULTS_MISMATCH", "CALIBRATION_RESULTS.json != deterministic recomputation")
    m = _load_json(manifest_path)
    alloc = stored.get("confirmatory_allocation")
    checks = {
        "action_space.alpha_calibration.passing_alphas": stored["alpha"]["passing_alphas"],
        "action_space.alpha_calibration.failing_alphas": stored["alpha"]["failing_alphas"],
        "action_space.alpha_calibration.failure_reasons": stored["alpha"]["failure_reasons"],
        "action_space.alpha_star": stored["alpha"]["alpha_star"],
        "action_space.alpha_secondary": stored["alpha"]["alpha_secondary"],
        "difficulty_strata.generator_settings_per_stratum": stored["difficulty"]["selected_settings_per_stratum"],
        "power.q_disc_calibrated": stored["discordance"]["q_disc"],
        "power.q_disc_upper_confidence_bound": stored["discordance"]["q_disc_upper_confidence_bound"],
        "power.G_power": stored["power"]["G_power"],
        "power.G_max": stored["power"]["G_max"],
        "power.G_pool": stored["power"]["G_pool"],
        "power.G_confirm": stored["power"]["G_confirm"],
        "power.budget_mde_if_constrained": stored["power"]["budget_mde_if_constrained"],
        "power.calibration_verdict": stored["verdict"],
        "bank.arms_confirmatory": stored["power"]["selected_confirmatory_arms"],
        "budget.measured_candidates_per_hour": stored["throughput"]["measured_candidates_per_hour"],
        "difficulty_strata.expected_counts": (
            alloc["expected_counts"] if alloc else {"hard_medium": 0, "medium": 0}),
        "calibration.raw_records_sha256": sha256_file(records_path),
        "calibration.metadata_sha256": sha256_file(metadata_path),
        "calibration.results_sha256": sha256_file(results_path),
        "cohorts.calibration_task_manifest_sha256": sha256_file(task_manifest_path),
        "cohorts.confirmatory_candidate_pool_sha256": sha256_file(candidate_pool_path),
    }
    for dotted, want in checks.items():
        node: Any = m
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise CalibrationError("C9_MANIFEST_BINDING", f"manifest missing {dotted}")
            node = node[part]
        if node != want:
            raise CalibrationError("C9_MANIFEST_BINDING", f"manifest {dotted}={node!r} != calibration {want!r}")
    return stored


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("records")
    p.add_argument("metadata")
    p.add_argument("manifest")
    p.add_argument("task_manifest")
    p.add_argument("precommit")
    p.add_argument("candidate_pool")
    p.add_argument("output")
    p.add_argument("--apply-manifest", metavar="PATH")
    ns = p.parse_args(argv)
    result = compute_calibration(ns.records, ns.metadata, ns.manifest,
                                 ns.task_manifest, ns.precommit,
                                 ns.candidate_pool)
    write_results(result, ns.output)
    if ns.apply_manifest:
        apply_results_to_manifest(ns.manifest, ns.output, ns.apply_manifest)
    print(json.dumps({"verdict": result["verdict"], "results_sha256": sha256_file(ns.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
