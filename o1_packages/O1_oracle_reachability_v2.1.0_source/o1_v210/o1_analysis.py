"""
O1 oracle-reachability gate — executable analysis, v2.1.

Changes from v2.0, all closing pre-calibration audit defects:
  * records carry generated_text plus text/token digests; gate G22 recomputes
    well_formed, verifier_correct and R from the stored text with the sealed
    parser and verifier and rejects any stored metadata that disagrees — a
    record can no longer claim well_formed=true when the sealed parser says
    otherwise
  * gate G23 requires every confirmatory task to be a byte-identical member of
    the sealed 2,400-task candidate pool
  * G17 additionally binds power.G_pool (eligible banded-pool capacity) and
    requires G_confirm == min(G_power, G_max, G_pool)
  * budget_mde returns None with an explicit NO_DETECTABLE_EFFECT state when
    no delta <= q_disc reaches target power at the available G, instead of
    returning q_disc and calling it a sensitivity

Changes from v1.0, all closing audit defects:
  * BOUND to the sealed manifest; alpha, arms, G, task set, strata and artifact
    hashes are derived from it, never accepted as caller arguments
  * arms are explicit labels with declared magnitudes; no alpha-filtering, so a
    mis-tagged row cannot silently shrink a bank
  * completeness gates: exact task-set equality, exact arm set per task, exact
    G, one stratum per task, alpha discipline per arm
  * paired_arrays REJECTS unequal task sets instead of intersecting them
  * power is an exact finite sum, not Monte Carlo
  * four-state verdict with a predeclared equivalence rule; a null test is no
    longer reported as a negative oracle
  * zero-alpha gate checks rowwise token identity first; first_divergence is
    COMPUTED here, never trusted from upstream
  * the seal is append-only, single-write, atomic and hash-chained
  * optional fields are type-checked in G1, so schema defects raise
    IntegrityError rather than a raw TypeError deep inside a gate

stdlib + numpy only.

Record schema (JSONL, one row per generated continuation):

    task_id             str
    stratum             str    "hard_medium" | "medium"
    arm                 str    see ARM_SPEC
    alpha               float  must equal the arm's sealed magnitude
    direction_id        int?   1..4 for structured/random arms; null otherwise
    sign                int?   +1|-1 for structured/random arms; null otherwise
    stream_index        int    0..7 — the CRN pairing key
    seed                int    must match the baseline seed at the same index
    well_formed         bool   recomputed from generated_text by gate G22
    verifier_correct    bool   recomputed from generated_text by gate G22
    generated_token_ids list[int]
    generated_text      str    exact decoded continuation
    generated_text_sha256      str  sha256 of UTF-8 generated_text
    generated_token_ids_sha256 str  sha256 of canonical JSON token list
    n_generated_tokens  int    must equal len(generated_token_ids)
    hit_max_tokens      bool
    parsed_answer       str?   OPTIONAL; must equal the sealed parser's output
    R                   int?   OPTIONAL; if present must equal the derivation

first_divergence is NOT a schema field. It is computed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import numpy as np

from o1_answer_parser_v2 import parse_final_answer
from o1_truth_table_verifier_v2 import verify_letter

BANK_SIZE = 8
N_DIRECTIONS = 4
# Hard schema ceiling on stored continuation text. Honest rows are bounded by
# decoding.max_new_tokens (448); this cap exists so a single oversized forged
# row cannot be used to stall the sealed parser inside the scoring gates.
MAX_GENERATED_TEXT_CHARS = 65536
SIGNS = (1, -1)
STRATA = ("hard_medium", "medium")
SEED_DERIVATION_SCHEME = "sha256_domain_uint64_be_v1"
SEED_DOMAIN = b"O1_STREAM_SEED_V1\0"

# arm -> (is_intervention, requires_axis_geometry, manifest alpha key)
ARM_SPEC: dict[str, tuple[bool, bool, str | None]] = {
    "baseline": (False, False, None),
    "zero_alpha": (True, False, None),
    "structured": (True, True, "alpha_star"),
    "structured_secondary": (True, True, "alpha_secondary"),
    "random": (True, True, "alpha_star"),
}
ARMS = tuple(ARM_SPEC)

REQUIRED_FIELDS: dict[str, Any] = {
    "task_id": str, "stratum": str, "arm": str, "alpha": float,
    "stream_index": int, "seed": int, "well_formed": bool,
    "verifier_correct": bool, "generated_token_ids": list,
    "generated_text": str, "generated_text_sha256": str,
    "generated_token_ids_sha256": str,
    "n_generated_tokens": int, "hit_max_tokens": bool,
    "task_content_sha256": str, "generator_setting_id": str,
    "run_provenance_sha256": str, "pregeneration_seal_sha256": str,
}
# required non-null on every intervention arm; null on baseline
TRANSPORT_FIELD = "downstream_delta_rms"
OPTIONAL_FIELDS: dict[str, Any] = {"direction_id": int, "sign": int, "R": int,
                                   "parsed_answer": str}


def text_sha256(text: str) -> str:
    """Digest of the exact UTF-8 generated text."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Digest of the canonical JSON encoding of the token-id list."""
    return hashlib.sha256(
        json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class IntegrityError(RuntimeError):
    """A sealed gate failed. Analysis must not proceed."""

    def __init__(self, code: str, message: str, detail: Any = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.detail = detail


class SealViolation(RuntimeError):
    """The sealed opening order or the append-only seal was violated."""


# ==========================================================================
# exact binomial machinery
# ==========================================================================

_LF: dict[int, np.ndarray] = {}
_CRIT_CACHE: dict[tuple[int, float], np.ndarray] = {}


def _logfact(n_max: int) -> np.ndarray:
    key = max(int(n_max), 1)
    for n, arr in _LF.items():
        if n >= key:
            return arr
    arr = np.zeros(key + 1, dtype=np.float64)
    arr[1:] = np.cumsum(np.log(np.arange(1, key + 1, dtype=np.float64)))
    _LF.clear()
    _LF[key] = arr
    return arr


def _log_binom_pmf(n: int, p: float, k: np.ndarray, lf: np.ndarray) -> np.ndarray:
    if p <= 0.0:
        return np.where(k == 0, 0.0, -np.inf)
    if p >= 1.0:
        return np.where(k == n, 0.0, -np.inf)
    return lf[n] - lf[k] - lf[n - k] + k * math.log(p) + (n - k) * math.log1p(-p)


def mcnemar_exact_p(n10: int, n01: int) -> float:
    """Two-sided exact McNemar p. Conditional on n_disc, n10 ~ Bin(n_disc, 1/2)."""
    n = int(n10) + int(n01)
    if n == 0:
        return 1.0
    k = min(int(n10), int(n01))
    lf = _logfact(n)
    i = np.arange(0, k + 1)
    return min(1.0, 2.0 * float(
        np.exp(lf[n] - lf[i] - lf[n - i] - n * math.log(2.0)).sum()
    ))


def mcnemar_critical_table(n_max: int, alpha: float = 0.05) -> np.ndarray:
    """k_crit[n] = largest k <= n//2 rejected two-sided at alpha, else -1."""
    key = (int(n_max), float(alpha))
    cached = _CRIT_CACHE.get(key)
    if cached is not None:
        return cached
    lf = _logfact(n_max)
    out = np.full(int(n_max) + 1, -1, dtype=np.int64)
    log2 = math.log(2.0)
    for n in range(1, int(n_max) + 1):
        i = np.arange(0, n // 2 + 1)
        cdf = np.cumsum(np.exp(lf[n] - lf[i] - lf[n - i] - n * log2))
        ok = np.nonzero(2.0 * cdf <= alpha)[0]
        if ok.size:
            out[n] = int(ok[-1])
    out.setflags(write=False)
    _CRIT_CACHE[key] = out
    return out


def exact_power(
    G: int, q_disc: float, delta: float, alpha_level: float = 0.05,
    k_crit: np.ndarray | None = None,
) -> float:
    """
    EXACT directional power of the two-sided exact McNemar test. No simulation.

        N_disc ~ Bin(G, q),  N10 | n ~ Bin(n, pi),  pi = (q + delta) / (2q)
        Power  = sum_n P(N_disc = n) * P(N10 >= n - k_crit(n) | n)

    k_crit(n) < n//2 always (2*P(X <= n/2) >= 1 > alpha), so the inner event
    implies n10 > n01 and the sum is directional by construction.
    """
    if delta > q_disc:
        raise ValueError(f"delta={delta} exceeds q_disc={q_disc}: infeasible")
    if k_crit is None or k_crit.size <= G:
        k_crit = mcnemar_critical_table(G, alpha_level)
    lf = _logfact(G)
    pi = (q_disc + delta) / (2.0 * q_disc)
    p_n = np.exp(_log_binom_pmf(G, q_disc, np.arange(G + 1), lf))
    total = 0.0
    for nn in range(1, G + 1):
        if p_n[nn] < 1e-16:
            continue
        k = int(k_crit[nn])
        if k < 0:
            continue
        i = np.arange(nn - k, nn + 1)
        total += p_n[nn] * float(np.exp(_log_binom_pmf(nn, pi, i, lf)).sum())
    return total


def required_G(
    q_disc: float, delta: float, target_power: float = 0.80,
    alpha_level: float = 0.05, G_hi: int = 20_000,
) -> int:
    """Smallest G reaching ``target_power`` under exact McNemar planning.

    Return codes:
      * ``>= 1``: smallest powered task count;
      * ``-1``: target power was not reached by ``G_hi``;
      * ``-2``: the requested effect is infeasible because ``delta > q_disc``.

    The confirmatory pipeline guards the infeasible case earlier with the
    explicit ``TARGET_HEADROOM_EXCLUDED_BY_DISCORDANCE_BOUND`` verdict.  The
    ``-2`` sentinel is retained for direct library callers so they receive a
    machine-readable planning verdict rather than a raw ``ValueError``.
    """
    if G_hi < 1:
        raise ValueError("G_hi must be positive")
    if not (0.0 <= q_disc <= 1.0):
        raise ValueError(f"q_disc={q_disc} outside [0,1]")
    if delta < 0.0:
        raise ValueError(f"delta={delta} must be non-negative")
    if delta > q_disc:
        return -2
    lo = 1
    hi = min(256, int(G_hi))
    while True:
        k_crit = mcnemar_critical_table(hi, alpha_level)
        if exact_power(hi, q_disc, delta, alpha_level, k_crit) >= target_power:
            break
        if hi >= G_hi:
            return -1
        lo = hi + 1
        hi = min(int(G_hi), hi * 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if exact_power(mid, q_disc, delta, alpha_level, k_crit) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def budget_mde(
    G: int, q_disc: float, target_power: float = 0.80,
    alpha_level: float = 0.05, tol: float = 1e-4,
) -> float | None:
    """Smallest delta detectable at target_power under a constrained G.

    Returns ``None`` when NO delta <= q_disc reaches the target power at this
    G.  The v2.0 implementation returned q_disc itself in that regime (the
    bisection never moved), reporting an unachievable value as a sensitivity;
    callers must surface the ``None`` state as NO_DETECTABLE_EFFECT_AT_G, not
    coerce it to a number.
    """
    if G < 1 or q_disc <= 0.0:
        return None
    k_crit = mcnemar_critical_table(G, alpha_level)
    if exact_power(G, q_disc, q_disc, alpha_level, k_crit) < target_power:
        return None
    lo, hi = 0.0, q_disc
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if exact_power(G, q_disc, mid, alpha_level, k_crit) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass
class BudgetDecision:
    G_power: int
    G_max: int
    G_confirm: int
    dropped_arms: list[str]
    mde: float | None
    powered_for_target: bool
    q_disc_basis: str


def budget_fallback(
    q_disc_upper: float, G_max: int, delta_target: float = 0.05,
    target_power: float = 0.80, alpha_level: float = 0.05,
) -> BudgetDecision:
    """
    Deterministic, sealed before calibration opens: G_confirm = min(G_t, G_max).

    q_disc_upper must be a predeclared CONSERVATIVE UPPER confidence bound on the
    calibrated discordance, not its point estimate: greater symmetric discordance
    demands more tasks for a fixed directional difference.
    """
    g = required_G(q_disc_upper, delta_target, target_power, alpha_level)
    if g == -2:
        return BudgetDecision(
            G_power=-2, G_max=G_max, G_confirm=0,
            dropped_arms=["random", "structured_secondary"],
            mde=q_disc_upper, powered_for_target=False,
            q_disc_basis="target excluded by discordance upper bound",
        )
    if g != -1 and g <= G_max:
        return BudgetDecision(g, G_max, g, [], delta_target, True,
                              "upper confidence bound")
    return BudgetDecision(
        G_power=g, G_max=G_max, G_confirm=G_max,
        dropped_arms=["random", "structured_secondary"],
        mde=budget_mde(G_max, q_disc_upper, target_power, alpha_level),
        powered_for_target=False, q_disc_basis="upper confidence bound",
    )


# ==========================================================================
# manifest binding
# ==========================================================================


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON bytes used for configuration and seal hashing."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_tree(path: str) -> str:
    """Deterministic digest of a file or directory tree, including paths."""
    if os.path.isfile(path):
        return sha256_file(path)
    if not os.path.isdir(path):
        raise IntegrityError("G0_ARTIFACT_HASH", f"artifact path does not exist: {path}")
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(bytes.fromhex(sha256_file(full)))
            h.update(b"\n")
    return h.hexdigest()


def resolved_generation_config(manifest: "Manifest") -> dict:
    """
    Canonical generation-relevant manifest object. RUN_PROVENANCE must carry an
    exact recursive copy; selected-key comparisons are intentionally forbidden.
    """
    return {k: manifest.get(k) for k in GENERATION_CONFIG_SECTIONS}


def create_pregeneration_seal(
    output_path: str,
    manifest_path: str,
    provenance_path: str,
    confirmatory_task_manifest_path: str,
    calibration_task_manifest_path: str,
    calibration_precommit_path: str,
    seed_matrix_path: str,
    calibration_records_path: str,
    calibration_metadata_path: str,
    calibration_results_path: str,
    axis_package_path: str,
    candidate_pool_path: str,
    records_path: str | None = None,
) -> dict:
    """
    Create the external pre-generation commitment. Refuses to overwrite and,
    when records_path is supplied, refuses if confirmatory records already
    exist. A remote timestamp/Git commit can strengthen chronology, but the
    analysis always consumes this exact artifact.
    """
    if os.path.exists(output_path):
        raise SealViolation(f"pre-generation seal already exists at {output_path}")
    if records_path and os.path.exists(records_path) and os.path.getsize(records_path) > 0:
        raise SealViolation(
            "confirmatory records already exist; PREGENERATION_SEAL must be "
            "created before the first outcome is generated"
        )
    man = Manifest(manifest_path)
    actual_axis_tree = sha256_tree(axis_package_path)
    if man.get("axis_artifact.package_sha256") != actual_axis_tree:
        raise IntegrityError(
            "G18_PREGENERATION_SEAL",
            "axis_artifact.package_sha256 does not match the supplied package tree")
    for key, filename in (("artifact_hashes.structured_axis_tensor", "axes_l3_24.npy"),
                          ("artifact_hashes.random_axis_tensor", "random_axes_l3_24.npy")):
        actual = sha256_file(os.path.join(axis_package_path, filename))
        if man.get(key) != actual:
            raise IntegrityError("G18_PREGENERATION_SEAL",
                                 f"{key} does not match {filename}")
    want_pool = man.get("cohorts.confirmatory_candidate_pool_sha256")
    got_pool = sha256_file(candidate_pool_path)
    if want_pool != got_pool:
        raise IntegrityError(
            "G18_PREGENERATION_SEAL",
            "cohorts.confirmatory_candidate_pool_sha256 does not match the "
            "supplied candidate pool")
    payload = {
        "schema_version": PREGEN_SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": man.get("code.commit"),
        "manifest_sha256": sha256_file(manifest_path),
        "run_provenance_sha256": sha256_file(provenance_path),
        "confirmatory_task_manifest_sha256": sha256_file(confirmatory_task_manifest_path),
        "calibration_task_manifest_sha256": sha256_file(calibration_task_manifest_path),
        "calibration_precommit_sha256": sha256_file(calibration_precommit_path),
        "seed_matrix_sha256": sha256_file(seed_matrix_path),
        "calibration_records_sha256": sha256_file(calibration_records_path),
        "calibration_metadata_sha256": sha256_file(calibration_metadata_path),
        "calibration_results_sha256": sha256_file(calibration_results_path),
        "axis_package_sha256": actual_axis_tree,
        "candidate_pool_sha256": got_pool,
        "records_absent_at_creation": True,
    }
    tmp = output_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, output_path)
    return payload


def verify_pregeneration_seal(
    seal_path: str,
    manifest: "Manifest",
    provenance_path: str,
    confirmatory_task_manifest_path: str,
    calibration_task_manifest_path: str,
    calibration_precommit_path: str,
    seed_matrix_path: str,
    calibration_records_path: str,
    calibration_metadata_path: str,
    calibration_results_path: str,
    axis_package_path: str,
    candidate_pool_path: str,
) -> dict:
    if not seal_path or not os.path.exists(seal_path):
        raise IntegrityError(
            "G18_PREGENERATION_SEAL",
            "PREGENERATION_SEAL.json is required; analysis-time agreement does "
            "not prove the configuration was frozen before outcomes existed",
        )
    required_paths = {
        "run_provenance": provenance_path,
        "confirmatory_task_manifest": confirmatory_task_manifest_path,
        "calibration_task_manifest": calibration_task_manifest_path,
        "calibration_precommit": calibration_precommit_path,
        "seed_matrix": seed_matrix_path,
        "calibration_records": calibration_records_path,
        "calibration_metadata": calibration_metadata_path,
        "calibration_results": calibration_results_path,
        "axis_package": axis_package_path,
        "candidate_pool": candidate_pool_path,
    }
    absent = [k for k, v in required_paths.items()
              if not v or not os.path.exists(v)]
    if absent:
        raise IntegrityError(
            "G18_PREGENERATION_SEAL",
            f"pre-generation artifact path(s) missing: {absent}", detail={"missing": absent})
    with open(seal_path) as fh:
        seal = json.load(fh)
    if seal.get("schema_version") != PREGEN_SCHEMA_VERSION:
        raise IntegrityError("G18_PREGENERATION_SEAL", "unsupported pre-generation seal schema")
    actual_axis_tree = sha256_tree(axis_package_path)
    if manifest.get("axis_artifact.package_sha256") != actual_axis_tree:
        raise IntegrityError(
            "G18_PREGENERATION_SEAL",
            "sealed axis package hash disagrees with the supplied package tree")
    for key, filename in (("artifact_hashes.structured_axis_tensor", "axes_l3_24.npy"),
                          ("artifact_hashes.random_axis_tensor", "random_axes_l3_24.npy")):
        actual = sha256_file(os.path.join(axis_package_path, filename))
        if manifest.get(key) != actual:
            raise IntegrityError("G18_PREGENERATION_SEAL",
                                 f"{key} disagrees with {filename}")
    expected = {
        "code_commit": manifest.get("code.commit"),
        "manifest_sha256": manifest.sha256,
        "run_provenance_sha256": sha256_file(provenance_path),
        "confirmatory_task_manifest_sha256": sha256_file(confirmatory_task_manifest_path),
        "calibration_task_manifest_sha256": sha256_file(calibration_task_manifest_path),
        "calibration_precommit_sha256": sha256_file(calibration_precommit_path),
        "seed_matrix_sha256": sha256_file(seed_matrix_path),
        "calibration_records_sha256": sha256_file(calibration_records_path),
        "calibration_metadata_sha256": sha256_file(calibration_metadata_path),
        "calibration_results_sha256": sha256_file(calibration_results_path),
        "axis_package_sha256": actual_axis_tree,
        "candidate_pool_sha256": sha256_file(candidate_pool_path),
        "records_absent_at_creation": True,
    }
    bad = {k: {"sealed": seal.get(k), "actual": v}
           for k, v in expected.items() if seal.get(k) != v}
    if bad:
        raise IntegrityError(
            "G18_PREGENERATION_SEAL",
            f"pre-generation commitment disagrees on {sorted(bad)}",
            detail=bad,
        )
    return seal


def _walk(o: Any, p: str = ""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{p}[{i}]")
    else:
        yield p, o


MANIFEST_REQUIRED = (
    "code.commit", "code.analysis_module_sha256", "code.fixture_suite_sha256",
    "code.generation_module_sha256", "code.calibration_module_sha256",
    "code.orchestrator_module_sha256", "code.transport_module_sha256",
    "code.axis_verifier_sha256", "code.repository_clean_required",
    "code.repository_diff_sha256",
    "model.checkpoint_sha256", "model.dtype", "model.deterministic_flags",
    "generator.family", "generator.revision", "generator.structural_knobs",
    "generator.task_id_hash_scheme",
    "decoding.temperature", "decoding.top_p", "decoding.max_new_tokens",
    "decoding.stopping_rule", "decoding.verifier", "decoding.answer_marker",
    "action_space.alpha_star", "action_space.alpha_selection_rule",
    "action_space.alpha_calibration.passing_alphas",
    "action_space.alpha_calibration.failing_alphas",
    "action_space.alpha_calibration.failure_reasons",
    "action_space.writable_boundary.loop_index_zero_based",
    "action_space.writable_boundary.physical_layer_one_based",
    "action_space.writable_boundary.module_index_zero_based",
    "action_space.writable_boundary.boundary_semantics",
    "action_space.writable_boundary.first_affected_cache_slot",
    "action_space.writable_boundary.position_scope",
    "action_space.writable_boundary.application_frequency",
    "action_space.writable_boundary.delta_broadcast_rule",
    "action_space.normalization", "action_space.structured_axes.gram_matrix",
    "bank.size", "bank.geometry", "bank.arms_required",
    "bank.arms_conditional_on_budget", "bank.arms_confirmatory",
    "bank.stream_assignment", "bank.stream_assignment_rule",
    "cohorts.master_seed", "cohorts.seed_hash_function",
    "cohorts.seed_derivation_scheme",
    "cohorts.confirmatory_task_manifest_sha256",
    "cohorts.confirmatory_candidate_pool_sha256",
    "cohorts.calibration_task_manifest_sha256", "cohorts.seed_matrix_sha256",
    "action_space.alpha_swept_in_calibration",
    "transport.measurement_definition", "transport.measurement_loop",
    "transport.measurement_layer", "transport.norm_convention",
    "transport.token_position", "transport.normalized_by_injected_rms",
    "transport.dead_rms_threshold",
    "power.q_disc_calibrated", "power.q_disc_upper_confidence_bound",
    "power.G_power", "power.G_max", "power.G_pool", "power.G_confirm",
    "power.delta_target", "power.target_power", "power.alpha_level",
    "power.budget_mde_if_constrained", "power.calibration_verdict",
    "budget.max_wall_clock_days", "budget.calibration_inside_budget",
    "budget.rerun_reserve_fraction", "budget.min_completed_task_fraction",
    "budget.max_plausible_candidates_per_hour",
    "budget.on_hardware_interruption", "budget.measured_candidates_per_hour",
    "zero_alpha_null.tau", "zero_alpha_null.token_identity_required",
    "zero_alpha_null.min_token_identity_rate",
    "difficulty_strata.expected_counts",
    "difficulty_strata.generator_settings_per_stratum",
    "calibration.precommit_sha256", "calibration.raw_records_sha256",
    "calibration.metadata_sha256",
    "calibration.results_sha256", "calibration.bootstrap_draws",
    "calibration.bootstrap_seed",
    "artifact_hashes.tokenizer", "artifact_hashes.prompt_template",
    "artifact_hashes.parser", "artifact_hashes.verifier_implementation",
    "artifact_hashes.structured_axis_tensor", "artifact_hashes.random_axis_tensor",
    "axis_artifact.package_sha256",
)

GENERATION_CONFIG_SECTIONS = (
    "code", "model", "generator", "difficulty_strata", "decoding",
    "action_space", "bank", "cohorts", "artifact_hashes", "transport",
    "zero_alpha_null",
)
PREGEN_SCHEMA_VERSION = "1.0"
PROVENANCE_SCHEMA_VERSION = "1.0"



class Manifest:
    """The sealed manifest. Everything the analysis needs comes from here."""

    def __init__(self, path: str):
        self.path = path
        with open(path) as fh:
            self.raw = json.load(fh)
        self.sha256 = sha256_file(path)
        self._check()

    def _check(self) -> None:
        flat = dict(_walk(self.raw))
        open_slots = [k for k, v in flat.items() if v == "FREEZE_SLOT"]
        if open_slots:
            raise IntegrityError(
                "G0_MANIFEST_UNSEALED",
                f"{len(open_slots)} FREEZE_SLOT(s) still open: {open_slots[:6]}",
                detail={"open": open_slots},
            )
        if "manifest_sha256" in self.raw:
            raise IntegrityError(
                "G0_MANIFEST_SELF_HASH",
                "manifest carries its own sha256, which cannot be well defined; "
                "the manifest hash belongs in an external SEAL.json / SHA256SUMS",
            )
        for key in MANIFEST_REQUIRED:
            # Resolve the key directly. Empty lists/dicts are legitimate values
            # and produce no leaves under _walk, so leaf-path matching is unsafe.
            node: Any = self.raw
            missing = False
            for part in key.split("."):
                if not isinstance(node, dict) or part not in node:
                    missing = True
                    break
                node = node[part]
            if missing:
                raise IntegrityError(
                    "G0_MANIFEST_INCOMPLETE", f"missing required key {key!r}"
                )

    def get(self, dotted: str, default: Any = "__RAISE__") -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default == "__RAISE__":
                    raise IntegrityError(
                        "G0_MANIFEST_INCOMPLETE", f"missing key {dotted!r}"
                    )
                return default
            node = node[part]
        return node

    def arm_alpha(self, arm: str) -> float:
        key = ARM_SPEC[arm][2]
        return 0.0 if key is None else float(self.get(f"action_space.{key}"))

    def verify_artifacts(self, paths: dict[str, str]) -> dict[str, str]:
        out = {}
        for key, p in paths.items():
            want, got = self.get(key), sha256_file(p)
            if got != want:
                raise IntegrityError(
                    "G0_ARTIFACT_HASH",
                    f"{key}: sealed {want[:16]}... but {os.path.basename(p)} "
                    f"hashes {got[:16]}...",
                    detail={"key": key, "path": p, "sealed": want, "actual": got},
                )
            out[key] = got
        return out


# ==========================================================================
# gates
# ==========================================================================


def _group(records, *keys):
    out: dict[tuple, list[dict]] = {}
    for r in records:
        out.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    return out


def gate_G1_schema(records: Sequence[dict]) -> None:
    if not records:
        raise IntegrityError("G1_SCHEMA", "record set is empty")
    for idx, r in enumerate(records):
        for fname, ftype in REQUIRED_FIELDS.items():
            if fname not in r:
                raise IntegrityError("G1_SCHEMA", f"row {idx}: missing {fname!r}")
            v = r[fname]
            if ftype is float and isinstance(v, int) and not isinstance(v, bool):
                continue
            if ftype is int and isinstance(v, bool):
                raise IntegrityError(
                    "G1_SCHEMA", f"row {idx}: {fname!r} must be int, got bool"
                )
            if not isinstance(v, ftype):
                raise IntegrityError(
                    "G1_SCHEMA",
                    f"row {idx}: {fname!r} must be {ftype.__name__}, "
                    f"got {type(v).__name__}",
                )
        for fname, ftype in OPTIONAL_FIELDS.items():
            if fname in r and r[fname] is not None:
                if isinstance(r[fname], bool) or not isinstance(r[fname], ftype):
                    raise IntegrityError(
                        "G1_SCHEMA",
                        f"row {idx}: optional {fname!r} must be {ftype.__name__} "
                        f"or null, got {type(r[fname]).__name__}",
                    )
        if r["arm"] not in ARMS:
            raise IntegrityError("G1_SCHEMA", f"row {idx}: unknown arm {r['arm']!r}")
        if r["stratum"] not in STRATA:
            raise IntegrityError(
                "G1_SCHEMA", f"row {idx}: unknown stratum {r['stratum']!r}"
            )
        if not (0 <= r["stream_index"] < BANK_SIZE):
            raise IntegrityError(
                "G1_SCHEMA", f"row {idx}: stream_index {r['stream_index']} out of range"
            )
        is_int = ARM_SPEC[r["arm"]][0]
        tv = r.get(TRANSPORT_FIELD)
        if is_int and (tv is None or isinstance(tv, bool)
                       or not isinstance(tv, (int, float))
                       or not math.isfinite(float(tv)) or float(tv) < 0.0):
            raise IntegrityError(
                "G1_SCHEMA",
                f"row {idx}: arm {r['arm']!r} requires a finite, non-negative "
                f"{TRANSPORT_FIELD} (got {tv!r}); it is the measured downstream "
                f"residual change, and a NaN silently poisons every summary; "
                f"without it a null cannot be decomposed into 'the delta died' "
                f"versus 'the delta moved nothing useful'",
            )
        if not is_int and tv is not None:
            raise IntegrityError(
                "G1_SCHEMA", f"row {idx}: baseline rows must carry null "
                f"{TRANSPORT_FIELD}")
        if len(r["generated_token_ids"]) != r["n_generated_tokens"]:
            raise IntegrityError(
                "G1_SCHEMA",
                f"row {idx}: n_generated_tokens={r['n_generated_tokens']} but "
                f"{len(r['generated_token_ids'])} token ids present",
            )
        if len(r["generated_text"]) > MAX_GENERATED_TEXT_CHARS:
            raise IntegrityError(
                "G1_SCHEMA",
                f"row {idx}: generated_text is {len(r['generated_text'])} chars, "
                f"above the {MAX_GENERATED_TEXT_CHARS}-char schema ceiling",
            )
        if not all(isinstance(t, int) and not isinstance(t, bool)
                   for t in r["generated_token_ids"]):
            raise IntegrityError(
                "G1_SCHEMA", f"row {idx}: generated_token_ids must be ints")
        needs_geom = ARM_SPEC[r["arm"]][1]
        d_null = r.get("direction_id") is None
        s_null = r.get("sign") is None
        if needs_geom and (d_null or s_null):
            raise IntegrityError(
                "G1_SCHEMA",
                f"row {idx}: arm {r['arm']!r} requires BOTH direction_id and sign "
                f"non-null (direction_id={r.get('direction_id')!r}, "
                f"sign={r.get('sign')!r})",
            )
        if not needs_geom and not (d_null and s_null):
            raise IntegrityError(
                "G1_SCHEMA", f"row {idx}: arm {r['arm']!r} must carry null geometry"
            )
        if needs_geom and r["sign"] not in SIGNS:
            raise IntegrityError(
                "G1_SCHEMA", f"row {idx}: sign {r['sign']} not in {SIGNS}"
            )
        if needs_geom and not (1 <= r["direction_id"] <= N_DIRECTIONS):
            raise IntegrityError(
                "G1_SCHEMA",
                f"row {idx}: direction_id {r['direction_id']} outside 1..{N_DIRECTIONS}",
            )


def gate_G2_bank_size(records) -> None:
    for (task, arm), rows in _group(records, "task_id", "arm").items():
        if len(rows) != BANK_SIZE:
            raise IntegrityError(
                "G2_BANK_SIZE",
                f"task {task} arm {arm}: {len(rows)} rows, expected {BANK_SIZE}",
                detail={"task_id": task, "arm": arm, "n": len(rows)},
            )


def gate_G3_stream_uniqueness(records) -> None:
    """
    Eight distinct stream indices AND eight distinct seeds per (task, arm).

    Catches four unique baselines dressed as eight: had +a*d_k and -a*d_k shared
    a stream, the baseline bank would hold four unique continuations across eight
    rows and its max_i would be a best-of-4 against a best-of-8.
    """
    for (task, arm), rows in _group(records, "task_id", "arm").items():
        idxs = sorted(r["stream_index"] for r in rows)
        if idxs != list(range(BANK_SIZE)):
            raise IntegrityError(
                "G3_STREAM_UNIQUENESS",
                f"task {task} arm {arm}: stream_index multiset {idxs} is not 0..7",
            )
        n_uni = len({r["seed"] for r in rows})
        if n_uni != BANK_SIZE:
            raise IntegrityError(
                "G3_STREAM_UNIQUENESS",
                f"task {task} arm {arm}: {n_uni} distinct seeds across {BANK_SIZE} "
                f"rows — effective bank is max-over-{n_uni}, not max-over-{BANK_SIZE}",
                detail={"task_id": task, "arm": arm, "n_unique_seeds": n_uni},
            )


def gate_G4_index_matched_seeds(records) -> None:
    base = {(r["task_id"], r["stream_index"]): r["seed"]
            for r in records if r["arm"] == "baseline"}
    for r in records:
        if r["arm"] == "baseline":
            continue
        k = (r["task_id"], r["stream_index"])
        if k in base and r["seed"] != base[k]:
            raise IntegrityError(
                "G4_INDEX_MATCHED_SEEDS",
                f"task {r['task_id']} arm {r['arm']} stream {r['stream_index']}: "
                f"seed {r['seed']} != baseline {base[k]} — not CRN-paired",
            )


def gate_G5_pair_completeness(records) -> None:
    base = {(r["task_id"], r["stream_index"]) for r in records if r["arm"] == "baseline"}
    for t in {r["task_id"] for r in records}:
        for i in range(BANK_SIZE):
            if (t, i) not in base:
                raise IntegrityError(
                    "G5_PAIR_COMPLETENESS",
                    f"task {t}: no baseline row at stream_index {i}",
                )
    for r in records:
        if r["arm"] != "baseline" and (r["task_id"], r["stream_index"]) not in base:
            raise IntegrityError(
                "G5_PAIR_COMPLETENESS",
                f"task {r['task_id']} arm {r['arm']} stream {r['stream_index']}: unpaired",
            )


def gate_G6_malformed_convention(records) -> None:
    for idx, r in enumerate(records):
        derived = int(bool(r["well_formed"]) and bool(r["verifier_correct"]))
        if r.get("R") is not None and int(r["R"]) != derived:
            raise IntegrityError(
                "G6_MALFORMED_CONVENTION",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): stored R={r['R']} "
                f"but well_formed={r['well_formed']} verifier_correct="
                f"{r['verifier_correct']} implies R={derived}",
                detail={"row": idx, "stored": r["R"], "derived": derived},
            )


def gate_G7_task_set(records, confirmatory, calibration) -> None:
    """Exact equality with the sealed confirmatory set, plus disjointness."""
    got = {r["task_id"] for r in records}
    want = {t["task_id"] for t in confirmatory}
    calibration_ids = {t["task_id"] for t in calibration}
    missing, extra = sorted(want - got), sorted(got - want)
    if missing or extra:
        raise IntegrityError(
            "G7_TASK_SET",
            f"record task set != sealed confirmatory set: {len(missing)} sealed "
            f"task(s) absent, {len(extra)} unsealed task(s) present",
            detail={"missing": missing[:10], "extra": extra[:10]},
        )
    overlap = sorted(got & calibration_ids)
    if overlap:
        raise IntegrityError(
            "G7_TASK_SET",
            f"{len(overlap)} confirmatory task(s) also in calibration: {overlap[:5]}",
            detail={"overlap": overlap})
    ch_conf = {t["task_content_sha256"] for t in confirmatory}
    ch_cal = {t["task_content_sha256"] for t in calibration}
    shared = sorted(ch_conf & ch_cal)
    if shared:
        raise IntegrityError(
            "G7_TASK_SET",
            f"{len(shared)} task(s) appear in both cohorts under different IDs: "
            f"identical task_content_sha256 {[h[:12] for h in shared[:3]]}. "
            f"ID disjointness is not content disjointness.",
            detail={"shared_content": shared})
    sealed_ch = {t["task_id"]: t["task_content_sha256"] for t in confirmatory}
    for r in records:
        if r["task_content_sha256"] != sealed_ch.get(r["task_id"]):
            raise IntegrityError(
                "G7_TASK_SET",
                f"task {r['task_id']} arm {r['arm']}: record content hash "
                f"{r['task_content_sha256'][:12]} != sealed "
                f"{str(sealed_ch.get(r['task_id']))[:12]} — the row carries the "
                f"right task ID but was generated from different content",
                detail={"task_id": r["task_id"]})


def gate_G8_bank_geometry(records) -> None:
    want = sorted((d, s) for d in range(1, N_DIRECTIONS + 1) for s in SIGNS)
    for (task, arm), rows in _group(records, "task_id", "arm").items():
        if not ARM_SPEC[arm][1]:
            continue
        got = sorted((r["direction_id"], r["sign"]) for r in rows)
        if got != want:
            raise IntegrityError(
                "G8_BANK_GEOMETRY",
                f"task {task} arm {arm}: geometry {got} is not {N_DIRECTIONS} "
                f"antipodal axis pairs",
                detail={"task_id": task, "arm": arm},
            )


def action_index(direction_id: int, sign: int) -> int:
    """a in 0..7 over the eight signed actions."""
    return 2 * (int(direction_id) - 1) + (0 if int(sign) == 1 else 1)


def gate_G13_stream_assignment(records, confirmatory, manifest) -> None:
    """
    Sealed cyclic Latin square: stream(a, g) = (a + r_g) mod 8, where r_g is the
    task's rank within its stratum in sealed manifest order, mod 8.

    A fixed direction-to-stream map would tie every per-direction secondary to
    one stream index. The Latin square gives every signed action every stream
    once per block of eight tasks.
    """
    if manifest.get("bank.stream_assignment", "") != "cyclic_latin_square":
        return
    rank = {t["task_id"]: int(t["sealed_rank_within_stratum"])
            for t in confirmatory}
    for r in records:
        if not ARM_SPEC[r["arm"]][1]:
            continue
        a = action_index(r["direction_id"], r["sign"])
        want = (a + rank[r["task_id"]]) % BANK_SIZE
        if r["stream_index"] != want:
            raise IntegrityError(
                "G13_STREAM_ASSIGNMENT",
                f"task {r['task_id']} arm {r['arm']} d{r['direction_id']}"
                f"{'+' if r['sign'] == 1 else '-'}: stream {r['stream_index']} "
                f"but the sealed Latin square requires {want}",
                detail={"task_id": r["task_id"], "action": a,
                        "rank": rank[r["task_id"]], "want": want},
            )


def gate_G14_provenance(manifest: Manifest, provenance_path: str | None) -> dict:
    """
    Compare one canonical resolved generation object recursively. This replaces
    the v1.4 selected-key list, which could miss boundary semantics, the answer
    marker, transport definition, structural knobs, or dirty generation code.
    """
    if provenance_path is None or not os.path.exists(provenance_path):
        raise IntegrityError(
            "G14_PROVENANCE",
            "no RUN_PROVENANCE.json supplied; generation configuration would "
            "remain documentary rather than executable",
        )
    with open(provenance_path) as fh:
        prov = json.load(fh)
    if prov.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise IntegrityError("G14_PROVENANCE", "unsupported RUN_PROVENANCE schema")
    expected_cfg = resolved_generation_config(manifest)
    got_cfg = prov.get("resolved_generation_config")
    if got_cfg != expected_cfg:
        # report top-level sections first; the full objects remain in detail
        sections = sorted(set(expected_cfg) | set(got_cfg or {}))
        bad = [k for k in sections
               if not isinstance(got_cfg, dict) or got_cfg.get(k) != expected_cfg.get(k)]
        raise IntegrityError(
            "G14_PROVENANCE",
            f"resolved generation configuration disagrees in {bad}",
            detail={"sections": bad, "expected": expected_cfg, "actual": got_cfg},
        )
    att = prov.get("runtime_attestation", {})
    expected_att = {
        "generation_module_sha256": manifest.get("code.generation_module_sha256"),
        "repository_clean": bool(manifest.get("code.repository_clean_required")),
        "repository_diff_sha256": manifest.get("code.repository_diff_sha256"),
    }
    bad_att = {k: {"sealed": v, "actual": att.get(k)}
               for k, v in expected_att.items() if att.get(k) != v}
    if bad_att:
        raise IntegrityError(
            "G14_PROVENANCE",
            f"runtime code/worktree attestation disagrees on {sorted(bad_att)}",
            detail=bad_att,
        )
    allowed = {"schema_version", "created_at_utc", "resolved_generation_config",
               "runtime_attestation"}
    extras = sorted(set(prov) - allowed)
    if extras:
        raise IntegrityError(
            "G14_PROVENANCE",
            f"RUN_PROVENANCE contains undeclared top-level fields {extras}; "
            "use the canonical schema rather than a second configuration surface",
        )
    return prov


def gate_G15_alpha_grid(manifest, records) -> None:
    """Enforce the sealed magnitude-selection rule, not merely record it."""
    rule = manifest.get("action_space.alpha_selection_rule")
    if rule not in ("largest_passing", "largest_passing_minus_one_grid_step"):
        raise IntegrityError(
            "G15_ALPHA_GRID", f"unknown alpha_selection_rule {rule!r}")
    grid = [float(x) for x in manifest.get("action_space.alpha_swept_in_calibration")]
    star = float(manifest.get("action_space.alpha_star"))
    passing = manifest.get("action_space.alpha_calibration.passing_alphas", None)
    if passing is None:
        raise IntegrityError(
            "G15_ALPHA_GRID",
            "no sealed alpha_calibration.passing_alphas: without the set of "
            "magnitudes that actually passed coherence, this gate can only "
            "enforce adjacency, not the selection rule it claims to enforce")
    P = sorted(float(x) for x in passing)
    if not P:
        raise IntegrityError("G15_ALPHA_GRID", "passing_alphas is empty")
    F = sorted(float(x) for x in
               (manifest.get("action_space.alpha_calibration.failing_alphas", None) or []))
    if set(P) & set(F):
        raise IntegrityError(
            "G15_ALPHA_GRID", f"passing and failing alphas overlap: {sorted(set(P) & set(F))}")
    if sorted(set(P) | set(F)) != sorted(grid):
        raise IntegrityError(
            "G15_ALPHA_GRID",
            f"passing {P} + failing {F} do not partition the swept grid {sorted(grid)}; "
            f"undeclared magnitudes {sorted(set(grid) - set(P) - set(F))} leave the "
            f"selection rule unverifiable",
            detail={"undeclared": sorted(set(grid) - set(P) - set(F))})
    reasons = manifest.get("action_space.alpha_calibration.failure_reasons", None) or {}
    missing_r = [x for x in F if str(x) not in reasons]
    if missing_r:
        raise IntegrityError(
            "G15_ALPHA_GRID", f"failing alphas without a recorded reason: {missing_r}")
    stray = [x for x in P if str(x) in reasons]
    if stray:
        raise IntegrityError(
            "G15_ALPHA_GRID", f"passing alphas carrying a failure reason: {stray}")
    if rule == "largest_passing" and star != max(P):
        raise IntegrityError(
            "G15_ALPHA_GRID",
            f"rule 'largest_passing' but alpha_star {star} != max(passing) {max(P)}",
            detail={"alpha_star": star, "max_passing": max(P), "passing": P})
    if rule == "largest_passing_minus_one_grid_step":
        below = [x for x in P if x < max(P)]
        if not below or star != max(below):
            raise IntegrityError(
                "G15_ALPHA_GRID",
                f"rule 'largest_passing_minus_one_grid_step' but alpha_star "
                f"{star} is not the next passing magnitude below max {max(P)}")
    sec = manifest.get("action_space.alpha_secondary", None)
    below_star = [x for x in P if x < star]
    if sec is not None and below_star and float(sec) != max(below_star):
        raise IntegrityError(
            "G15_ALPHA_GRID",
            f"alpha_secondary {sec} != the largest passing magnitude below "
            f"alpha_star, which is {max(below_star)}")
    if star not in grid:
        raise IntegrityError(
            "G15_ALPHA_GRID", f"alpha_star {star} is not in the swept grid {grid}")
    arms = set(manifest.get("bank.arms_confirmatory"))
    if sec is None:
        if "structured_secondary" in arms:
            raise IntegrityError(
                "G15_ALPHA_GRID",
                "structured_secondary is a confirmatory arm but alpha_secondary "
                "is null")
        return
    sec = float(sec)
    if sec not in P:
        raise IntegrityError(
            "G15_ALPHA_GRID",
            f"alpha_secondary {sec} is not in passing_alphas {P}")
    if len(P) == 1:
        raise IntegrityError(
            "G15_ALPHA_GRID",
            "only one magnitude passed coherence, so alpha_secondary must be null "
            "and the secondary arm omitted")
    if not sec < star:
        raise IntegrityError(
            "G15_ALPHA_GRID",
            f"rule {rule!r} requires alpha_secondary ({sec}) < alpha_star "
            f"({star}); under 'largest_passing' the secondary is the next "
            f"SMALLER passing grid value")

    if ("structured_secondary" in arms) != any(
            r["arm"] == "structured_secondary" for r in records):
        raise IntegrityError(
            "G15_ALPHA_GRID",
            "structured_secondary must be present in the records iff it is a "
            "sealed confirmatory arm")


def gate_G9_arm_completeness(records, required_arms) -> None:
    """Every task carries exactly the sealed arm set. A missing arm is a defect,
    not a smaller analysis."""
    want = set(required_arms)
    per: dict[str, set[str]] = {}
    for r in records:
        per.setdefault(r["task_id"], set()).add(r["arm"])
    for task, got in sorted(per.items()):
        if got != want:
            raise IntegrityError(
                "G9_ARM_COMPLETENESS",
                f"task {task}: arms {sorted(got)} != sealed {sorted(want)}",
                detail={"task_id": task, "missing": sorted(want - got),
                        "extra": sorted(got - want)},
            )


def gate_G10_alpha_discipline(records, manifest: Manifest) -> None:
    for idx, r in enumerate(records):
        want = manifest.arm_alpha(r["arm"])
        if not math.isclose(float(r["alpha"]), want, rel_tol=0.0, abs_tol=1e-12):
            raise IntegrityError(
                "G10_ALPHA_DISCIPLINE",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): alpha={r['alpha']} "
                f"but the sealed magnitude for this arm is {want}",
                detail={"row": idx, "arm": r["arm"], "alpha": r["alpha"],
                        "sealed": want},
            )


def gate_G11_stratum_consistency(records, expected_counts: dict,
                                 confirmatory=None, selected_settings=None) -> None:
    """
    One stratum per task, sealed counts, AND — when the sealed task manifest is
    supplied — the sealed per-task assignment. Aggregate counts alone permit two
    tasks to swap strata after sealing, which moves both the difficulty design
    and the Latin-square permutation.
    """
    per: dict[str, set[str]] = {}
    per_setting: dict[str, set[str]] = {}
    for r in records:
        per.setdefault(r["task_id"], set()).add(r["stratum"])
        per_setting.setdefault(r["task_id"], set()).add(r["generator_setting_id"])
    if confirmatory is not None:
        sealed = {t["task_id"]: t["stratum"] for t in confirmatory}
        sealed_setting = {t["task_id"]: t["generator_setting_id"] for t in confirmatory}
        if selected_settings is not None:
            for t in confirmatory:
                allowed = set(selected_settings.get(t["stratum"], []))
                if t["generator_setting_id"] not in allowed:
                    raise IntegrityError(
                        "G11_STRATUM_CONSISTENCY",
                        f"task {t['task_id']}: generator setting "
                        f"{t['generator_setting_id']!r} is not among calibration-"
                        f"selected settings {sorted(allowed)} for stratum "
                        f"{t['stratum']!r}",
                        detail={"task_id": t["task_id"],
                                "setting": t["generator_setting_id"],
                                "allowed": sorted(allowed)})
        for task, settings in sorted(per_setting.items()):
            if len(settings) != 1 or next(iter(settings)) != sealed_setting.get(task):
                raise IntegrityError(
                    "G11_STRATUM_CONSISTENCY",
                    f"task {task}: record generator settings {sorted(settings)} != "
                    f"sealed {sealed_setting.get(task)!r}",
                    detail={"task_id": task, "actual_settings": sorted(settings),
                            "sealed_setting": sealed_setting.get(task)})
        for task, s in sorted(per.items()):
            if len(s) == 1 and next(iter(s)) != sealed.get(task):
                raise IntegrityError(
                    "G11_STRATUM_CONSISTENCY",
                    f"task {task}: stratum {next(iter(s))!r} but the sealed task "
                    f"manifest assigns {sealed.get(task)!r}",
                    detail={"task_id": task, "actual": next(iter(s)),
                            "sealed": sealed.get(task)})
    for task, s in sorted(per.items()):
        if len(s) != 1:
            raise IntegrityError(
                "G11_STRATUM_CONSISTENCY",
                f"task {task}: strata {sorted(s)} — one stratum per task, every arm",
                detail={"task_id": task, "strata": sorted(s)},
            )
    counts = {k: 0 for k in STRATA}
    for s in per.values():
        counts[next(iter(s))] += 1
    want = {k: int(v) for k, v in expected_counts.items()}
    if counts != want:
        raise IntegrityError(
            "G11_STRATUM_CONSISTENCY",
            f"stratum counts {counts} != sealed {want}",
            detail={"actual": counts, "sealed": want},
        )


def gate_G12_exact_G(records, G_sealed) -> None:
    G = len({r["task_id"] for r in records})
    if G != int(G_sealed):
        raise IntegrityError(
            "G12_EXACT_G",
            f"{G} tasks present but sealed G is {G_sealed}; resume the missing "
            f"sealed shards — do not analyse a completed subset, since failure to "
            f"finish may correlate with length, malformedness or difficulty",
            detail={"actual": G, "sealed": int(G_sealed)},
        )


def transport_summary(records, arm: str, dead_rms: float | None = None,
                      manifest=None) -> dict:
    """
    Per-signed-action downstream residual change. This is what lets a null be
    decomposed into (a) the delta never propagated, (b) it propagated but changed
    no outcome, (c) it changed outcomes unhelpfully. Without it all three read as
    'no diversity', and a direction outside the writable span looks identical to
    a direction that transports fine but does not help.
    """
    if dead_rms is None:
        dead_rms = (float(manifest.get("transport.dead_rms_threshold"))
                    if manifest is not None else 1e-6)
    per: dict[str, list[float]] = {}
    for r in records:
        if r["arm"] != arm or r.get("direction_id") is None:
            continue
        k = f"d{r['direction_id']}{'+' if r['sign'] == 1 else '-'}"
        per.setdefault(k, []).append(float(r[TRANSPORT_FIELD]))
    out = {}
    for k, v in sorted(per.items()):
        a = np.array(v)
        out[k] = {"n": int(a.size), "median_rms": float(np.median(a)),
                  "q25": float(np.percentile(a, 25)),
                  "q75": float(np.percentile(a, 75)),
                  "dead_rate": float(np.mean(a <= dead_rms))}
    return out




def axis_transport_profile(records, arm: str = "structured", dead_rms: float | None = None,
                           manifest=None) -> dict:
    """Aggregate transport diagnostics across signs for each declared axis.

    This is diagnostic only.  It prevents a bank-level null from being read as
    evidence that the full action family is barren when one or more injected
    directions never produce a measurable downstream state change.  A signed
    action is ``DEAD`` only when every observed row is at or below the frozen
    transport threshold; an axis is ``DEAD_BOTH_SIGNS`` only when both of its
    signed actions are dead.
    """
    signed = transport_summary(records, arm, dead_rms=dead_rms, manifest=manifest)
    axes: dict[str, dict] = {}
    live_signed = 0
    for d in range(1, N_DIRECTIONS + 1):
        plus = signed.get(f"d{d}+")
        minus = signed.get(f"d{d}-")
        entries = [x for x in (plus, minus) if x is not None]
        if not entries:
            axes[f"d{d}"] = {
                "status": "NO_OBSERVATIONS",
                "n": 0,
                "dead_rate": None,
                "median_rms": None,
                "signed": {"plus": plus, "minus": minus},
            }
            continue
        sign_dead = []
        for item in (plus, minus):
            is_dead = item is not None and math.isclose(
                float(item["dead_rate"]), 1.0, rel_tol=0.0, abs_tol=1e-12
            )
            sign_dead.append(is_dead)
            if item is not None and not is_dead:
                live_signed += 1
        vals = [
            float(r[TRANSPORT_FIELD])
            for r in records
            if r["arm"] == arm and r.get("direction_id") == d
        ]
        if plus is not None and minus is not None and all(sign_dead):
            status = "DEAD_BOTH_SIGNS"
        elif any(sign_dead):
            status = "PARTIAL_SIGN_TRANSPORT"
        else:
            status = "TRANSPORTS_BOTH_SIGNS"
        a = np.asarray(vals, dtype=np.float64)
        threshold = (float(manifest.get("transport.dead_rms_threshold"))
                     if dead_rms is None and manifest is not None
                     else (1e-6 if dead_rms is None else float(dead_rms)))
        axes[f"d{d}"] = {
            "status": status,
            "n": int(a.size),
            "median_rms": float(np.median(a)),
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)),
            "dead_rate": float(np.mean(a <= threshold)),
            "signed": {"plus": plus, "minus": minus},
        }
    live_axes = sum(v["status"] not in ("DEAD_BOTH_SIGNS", "NO_OBSERVATIONS")
                    for v in axes.values())
    return {
        "dead_rms_threshold": (float(manifest.get("transport.dead_rms_threshold"))
                               if dead_rms is None and manifest is not None
                               else (1e-6 if dead_rms is None else float(dead_rms))),
        "observed_transporting_signed_actions": int(live_signed),
        "observed_transporting_axes": int(live_axes),
        "declared_signed_actions": 2 * N_DIRECTIONS,
        "declared_axes": N_DIRECTIONS,
        "axes": axes,
        "interpretation": (
            "Diagnostic only: dead axes narrow the scope of a null to the actions "
            "that measurably transported. They do not establish that the full "
            "writable action space is barren."
        ),
    }

def gate_G22_text_recomputation(records, task_manifest) -> None:
    """
    Recompute well_formed, verifier_correct and R from the stored generated
    text with the SEALED parser and verifier, and reject stored metadata that
    disagrees.  Also binds the text and token list to their recorded digests.

    Without this gate the analysis trusts scoring booleans assembled upstream;
    with it, forging a reward requires forging text that genuinely parses and
    genuinely verifies against the sealed task content — which G7 binds by
    content hash to the sealed manifest.
    """
    tasks: dict[str, dict] = {}
    for t in task_manifest:
        for key in ("options", "symbolic"):
            if key not in t:
                raise IntegrityError(
                    "G22_TEXT_RECOMPUTATION",
                    f"task manifest row {t.get('task_id')!r} lacks {key!r}; the "
                    f"sealed manifest must carry full task content so scoring "
                    f"can be recomputed, not trusted")
        tasks[t["task_id"]] = t
    for idx, r in enumerate(records):
        text = r["generated_text"]
        if text_sha256(text) != r["generated_text_sha256"]:
            raise IntegrityError(
                "G22_TEXT_RECOMPUTATION",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): generated_text "
                f"does not hash to generated_text_sha256",
                detail={"row": idx})
        if token_ids_sha256(r["generated_token_ids"]) != r["generated_token_ids_sha256"]:
            raise IntegrityError(
                "G22_TEXT_RECOMPUTATION",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): token ids do "
                f"not hash to generated_token_ids_sha256",
                detail={"row": idx})
        task = tasks.get(r["task_id"])
        if task is None:
            raise IntegrityError(
                "G22_TEXT_RECOMPUTATION",
                f"row {idx}: task {r['task_id']} absent from the sealed manifest")
        parsed = parse_final_answer(text)
        if "parsed_answer" in r and r.get("parsed_answer") != parsed:
            raise IntegrityError(
                "G22_TEXT_RECOMPUTATION",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): stored "
                f"parsed_answer={r.get('parsed_answer')!r} but the sealed parser "
                f"returns {parsed!r}",
                detail={"row": idx})
        want_wf = parsed is not None
        if bool(r["well_formed"]) != want_wf:
            raise IntegrityError(
                "G22_TEXT_RECOMPUTATION",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): stored "
                f"well_formed={r['well_formed']} but the sealed parser derives "
                f"{want_wf}",
                detail={"row": idx})
        want_vc = verify_letter(task, parsed)
        if bool(r["verifier_correct"]) != want_vc:
            raise IntegrityError(
                "G22_TEXT_RECOMPUTATION",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): stored "
                f"verifier_correct={r['verifier_correct']} but the sealed "
                f"verifier derives {want_vc}",
                detail={"row": idx})


def gate_G24_length_bounds(records, manifest: "Manifest") -> None:
    """No row may exceed the sealed decoding budget."""
    cap = int(manifest.get("decoding.max_new_tokens"))
    for idx, r in enumerate(records):
        if r["n_generated_tokens"] > cap:
            raise IntegrityError(
                "G24_LENGTH_BOUNDS",
                f"row {idx} (task {r['task_id']} arm {r['arm']}): "
                f"{r['n_generated_tokens']} generated tokens exceed the sealed "
                f"decoding.max_new_tokens = {cap}",
                detail={"row": idx, "n": r["n_generated_tokens"], "cap": cap})


def gate_G23_pool_membership(confirmatory, candidate_pool) -> None:
    """
    Every confirmatory task must be a member of the sealed candidate pool with
    identical content hash and generator setting.  The pool file itself is
    hash-bound by cohorts.confirmatory_candidate_pool_sha256, so a task outside
    the original outcome-blind population cannot enter the cohort.
    """
    pool: dict[str, dict] = {}
    for row in candidate_pool:
        tid = row.get("task_id")
        if not isinstance(tid, str) or not tid:
            raise IntegrityError("G23_POOL_MEMBERSHIP",
                                 "candidate pool row lacks task_id")
        if tid in pool:
            raise IntegrityError("G23_POOL_MEMBERSHIP",
                                 f"candidate pool duplicates task_id {tid}")
        pool[tid] = row
    for t in confirmatory:
        src = pool.get(t["task_id"])
        if src is None:
            raise IntegrityError(
                "G23_POOL_MEMBERSHIP",
                f"confirmatory task {t['task_id']} is not in the sealed "
                f"candidate pool",
                detail={"task_id": t["task_id"]})
        if src.get("task_content_sha256") != t["task_content_sha256"]:
            raise IntegrityError(
                "G23_POOL_MEMBERSHIP",
                f"confirmatory task {t['task_id']}: content hash differs from "
                f"the sealed pool entry",
                detail={"task_id": t["task_id"]})
        if src.get("generator_setting_id") != t["generator_setting_id"]:
            raise IntegrityError(
                "G23_POOL_MEMBERSHIP",
                f"confirmatory task {t['task_id']}: generator setting differs "
                f"from the sealed pool entry",
                detail={"task_id": t["task_id"]})


def gate_G21_record_run_binding(records, provenance_path: str,
                                pregeneration_seal_path: str) -> None:
    """Bind every generated row to the exact committed runtime and preseal."""
    want_prov = sha256_file(provenance_path)
    want_pregen = sha256_file(pregeneration_seal_path)
    for idx, r in enumerate(records):
        if r["run_provenance_sha256"] != want_prov:
            raise IntegrityError(
                "G21_RECORD_RUN_BINDING",
                f"row {idx}: run_provenance_sha256 does not match committed RUN_PROVENANCE.json",
                detail={"row": idx, "sealed": want_prov,
                        "actual": r["run_provenance_sha256"]})
        if r["pregeneration_seal_sha256"] != want_pregen:
            raise IntegrityError(
                "G21_RECORD_RUN_BINDING",
                f"row {idx}: pregeneration_seal_sha256 does not match PREGENERATION_SEAL.json",
                detail={"row": idx, "sealed": want_pregen,
                        "actual": r["pregeneration_seal_sha256"]})


def validate(records, manifest: Manifest, confirmatory, calibration,
             seed_matrix: dict | None = None) -> None:
    gate_G1_schema(records)
    gate_G17_manifest_consistency(manifest)
    gate_G15_alpha_grid(manifest, records)
    gate_G2_bank_size(records)
    gate_G3_stream_uniqueness(records)
    gate_G5_pair_completeness(records)
    gate_G4_index_matched_seeds(records)
    gate_G6_malformed_convention(records)
    gate_G8_bank_geometry(records)
    gate_G9_arm_completeness(records, manifest.get("bank.arms_confirmatory"))
    gate_G10_alpha_discipline(records, manifest)
    gate_G7_task_set(records, confirmatory, calibration)
    gate_G11_stratum_consistency(
        records, manifest.get("difficulty_strata.expected_counts"), confirmatory,
        manifest.get("difficulty_strata.generator_settings_per_stratum")
    )
    gate_G12_exact_G(records, manifest.get("power.G_confirm"))
    gate_G13_stream_assignment(records, confirmatory, manifest)
    gate_G24_length_bounds(records, manifest)
    gate_G22_text_recomputation(records, confirmatory)
    gate_G17_manifest_consistency(manifest)
    if seed_matrix is not None:
        gate_G16_seed_matrix_binding(records, seed_matrix, confirmatory, manifest)


# ==========================================================================
# endpoints
# ==========================================================================


def _R(r: dict) -> int:
    return int(bool(r["well_formed"]) and bool(r["verifier_correct"]))


def first_divergence(records, arm: str) -> dict[tuple[str, int], int | None]:
    """
    COMPUTED from generated_token_ids against the index-paired baseline. Any
    upstream-supplied divergence field is ignored. None = identical sequences.
    """
    base = {(r["task_id"], r["stream_index"]): r["generated_token_ids"]
            for r in records if r["arm"] == "baseline"}
    out: dict[tuple[str, int], int | None] = {}
    for r in records:
        if r["arm"] != arm or arm == "baseline":
            continue
        a, b = r["generated_token_ids"], base[(r["task_id"], r["stream_index"])]
        d = next((t for t, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if d is None and len(a) != len(b):
            d = min(len(a), len(b))
        out[(r["task_id"], r["stream_index"])] = d
    return out


def oracle_per_task(records, arm: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        if r["arm"] == arm:
            out[r["task_id"]] = max(out.get(r["task_id"], 0), _R(r))
    return out


def paired_arrays(records, int_arm: str, base_arm: str = "baseline"):
    yi, yb = oracle_per_task(records, int_arm), oracle_per_task(records, base_arm)
    if set(yi) != set(yb):
        raise IntegrityError(
            "E1_UNEQUAL_TASK_SETS",
            f"{int_arm!r} covers {len(yi)} tasks, {base_arm!r} covers {len(yb)}; "
            f"refusing to intersect — that would silently change G",
            detail={"only_int": sorted(set(yi) - set(yb))[:10],
                    "only_base": sorted(set(yb) - set(yi))[:10]},
        )
    tasks = sorted(yi)
    return (tasks,
            np.array([yi[t] for t in tasks], dtype=np.int64),
            np.array([yb[t] for t in tasks], dtype=np.int64))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Bin(n, p), exact via log-pmf."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    lf = _logfact(n)
    return float(np.exp(_log_binom_pmf(n, p, np.arange(0, k + 1), lf)).sum())


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05,
                          tol: float = 1e-9) -> float:
    """
    Exact one-sided upper (1 - alpha) confidence bound on a binomial proportion:
    the p solving P(X <= k | n, p) = alpha. Bisection, no scipy.
    """
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def bootstrap(y_int, y_base, n_draws=10_000, seed=20260728) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = (y_int - y_base).astype(np.float64)
    idx = rng.integers(0, d.size, size=(n_draws, d.size))
    return d[idx].mean(axis=1)


@dataclass
class EndpointResult:
    label: str
    arm: str
    alpha: float
    G: int
    n11: int
    n10: int
    n01: int
    n00: int
    n_disc: int
    q_disc: float
    H_reach: float
    p_exact: float
    ci_lo: float
    ci_hi: float
    ub_one_sided_95: float
    lb_one_sided_95: float
    delta_target: float
    verdict: str
    verdict_reason: str
    ci_method: str = "95% CI: paired task bootstrap, 10000 draws, percentile; one-sided bounds: exact Clopper-Pearson on the discordant cell"

    def to_dict(self):
        return asdict(self)


def _verdict(H, p, ci_lo, ci_hi, ub, delta_target, alpha_level):
    """
    Four states. A nonsignificant test is NOT a negative oracle: powering to
    detect +delta with 80% probability does not make failure to reject evidence
    that the effect is below delta. Only the equivalence arm can say that.
    """
    # The power calculation targets this exact directional McNemar criterion.
    # The bootstrap interval is descriptive and does not add an unpowered second
    # rejection condition.
    if H > 0 and p < alpha_level:
        return "POSITIVE_HEADROOM", (
            f"H={H:+.4f}, directional effect with exact two-sided McNemar "
            f"p={p:.4g} < {alpha_level}; bootstrap interval reported descriptively"
        )
    if H < 0 and p < alpha_level:
        return "HARMFUL", (
            f"H={H:+.4f}, directional harm with exact two-sided McNemar "
            f"p={p:.4g} < {alpha_level}; bootstrap interval reported descriptively"
        )
    if ub < delta_target:
        return f"PRACTICALLY_NULL_AT_{delta_target:g}", (
            f"one-sided 95% upper bound {ub:+.4f} lies below the predeclared "
            f"practical threshold {delta_target:+.4f}: headroom of at least "
            f"{delta_target:+.4f} is excluded"
        )
    return "INCONCLUSIVE", (
        f"one-sided 95% upper bound {ub:+.4f} >= {delta_target:+.4f}; neither a "
        f"positive effect nor practical equivalence is established. Failure to "
        f"reject is not evidence of no headroom."
    )


def endpoint(
    records, int_arm: str, manifest: Manifest, label: str = "",
    alpha_level: float = 0.05, bootstrap_seed: int = 20260728,
    n_draws: int = 10_000,
) -> EndpointResult:
    tasks, yi, yb = paired_arrays(records, int_arm)
    n11 = int(np.sum((yi == 1) & (yb == 1)))
    n10 = int(np.sum((yi == 1) & (yb == 0)))
    n01 = int(np.sum((yi == 0) & (yb == 1)))
    n00 = int(np.sum((yi == 0) & (yb == 0)))
    G = len(tasks)
    draws = bootstrap(yi, yb, n_draws, bootstrap_seed)
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    # The equivalence claim must NOT rest on the percentile bootstrap: when every
    # paired difference is identical (n_disc = 0) the bootstrap is degenerate and
    # reports zero width regardless of G. H_reach = p10 - p01 <= p10, so an exact
    # Clopper-Pearson upper bound on p10 is a valid, conservative, never-degenerate
    # one-sided bound. Symmetrically for the lower bound.
    ub = clopper_pearson_upper(n10, G, 0.05)
    lb = -clopper_pearson_upper(n01, G, 0.05)
    H, p = float(np.mean(yi - yb)), mcnemar_exact_p(n10, n01)
    dt = float(manifest.get("power.delta_target"))
    v, why = _verdict(H, p, lo, hi, ub, dt, alpha_level)
    return EndpointResult(
        label=label or int_arm, arm=int_arm, alpha=manifest.arm_alpha(int_arm),
        G=G, n11=n11, n10=n10, n01=n01, n00=n00, n_disc=n10 + n01,
        q_disc=(n10 + n01) / G, H_reach=H, p_exact=p, ci_lo=lo, ci_hi=hi,
        ub_one_sided_95=ub, lb_one_sided_95=lb, delta_target=dt,
        verdict=v, verdict_reason=why,
    )


def selection_headroom(records, arm: str) -> float:
    per: dict[str, list[int]] = {}
    for r in records:
        if r["arm"] == arm:
            per.setdefault(r["task_id"], []).append(_R(r))
    return float(np.mean([max(v) - sum(v) / len(v) for v in per.values()])) if per else 0.0


def outcome_diversity(records, arm: str) -> float:
    per: dict[str, set[int]] = {}
    for r in records:
        if r["arm"] == arm:
            per.setdefault(r["task_id"], set()).add(_R(r))
    return float(np.mean([len(v) > 1 for v in per.values()])) if per else 0.0


def coupling_survival(records, arm: str, ts=(0, 1, 8, 32, 128)) -> dict:
    """
    S(t) = P(T_div > t), computed from token ids. Diagnostic only: no row is
    ever filtered on T_div, and no endpoint conditions on it.
    """
    div = first_divergence(records, arm)
    if not div:
        return {"n": 0}
    a = np.array([math.inf if v is None else float(v) for v in div.values()])
    fin = a[np.isfinite(a)]
    return {
        "n": int(a.size),
        "p_no_divergence": float(np.mean(~np.isfinite(a))),
        "survival": {int(t): float(np.mean(a > t)) for t in ts},
        "median_T_div": float(np.median(fin)) if fin.size else None,
        "q25_T_div": float(np.percentile(fin, 25)) if fin.size else None,
        "q75_T_div": float(np.percentile(fin, 75)) if fin.size else None,
    }


# ==========================================================================
# zero-alpha: mechanical parity first, endpoint second
# ==========================================================================


@dataclass
class ZeroAlphaGate:
    parity: dict
    result: EndpointResult
    tau: float
    verdict: str
    reasons: list[str] = field(default_factory=list)


def zero_alpha_gate(records, manifest: Manifest,
                    bootstrap_seed: int = 20260728) -> ZeroAlphaGate:
    """
    A bank-level difference of zero can coexist with massive but symmetric
    divergence, so mechanical row parity is checked first and the endpoint-level
    tau is only a backstop. H_0 is NEVER subtracted from the primary.
    """
    tau = float(manifest.get("zero_alpha_null.tau"))
    need = bool(manifest.get("zero_alpha_null.token_identity_required"))
    min_ident = float(manifest.get("zero_alpha_null.min_token_identity_rate"))

    base = {(r["task_id"], r["stream_index"]): r
            for r in records if r["arm"] == "baseline"}
    n = ident = len_eq = term_eq = r_eq = dead = wf_eq = vc_eq = 0
    for r in records:
        if r["arm"] != "zero_alpha":
            continue
        b = base[(r["task_id"], r["stream_index"])]
        n += 1
        same_tok = r["generated_token_ids"] == b["generated_token_ids"]
        ident += same_tok
        wf_eq += r["well_formed"] == b["well_formed"]
        vc_eq += r["verifier_correct"] == b["verifier_correct"]
        len_eq += r["n_generated_tokens"] == b["n_generated_tokens"]
        term_eq += r["hit_max_tokens"] == b["hit_max_tokens"]
        r_eq += _R(r) == _R(b)
        dead += float(r.get(TRANSPORT_FIELD) or 0.0) == 0.0
    parity = {
        "n": n,
        "token_identity_rate": ident / n if n else 0.0,
        "length_equality_rate": len_eq / n if n else 0.0,
        "termination_equality_rate": term_eq / n if n else 0.0,
        "derived_R_identity_rate": r_eq / n if n else 0.0,
        "well_formed_identity_rate": wf_eq / n if n else 0.0,
        "verifier_correct_identity_rate": vc_eq / n if n else 0.0,
        "zero_transport_rate": dead / n if n else 0.0,
        "required_min_token_identity_rate": min_ident if need else None,
    }

    res = endpoint(records, "zero_alpha", manifest, label="zero_alpha_null",
                   bootstrap_seed=bootstrap_seed)
    reasons: list[str] = []
    if need and parity["token_identity_rate"] < min_ident:
        reasons.append(
            f"rowwise token identity {parity['token_identity_rate']:.4f} < required "
            f"{min_ident:.4f} on a path documented as bit-exact"
        )
    if need:
        for k in ("length_equality_rate", "termination_equality_rate",
                  "well_formed_identity_rate", "verifier_correct_identity_rate"):
            if parity[k] < 1.0:
                reasons.append(
                    f"{k} = {parity[k]:.4f} < 1.0 on a documented bit-exact path; "
                    f"identical tokens must imply identical deterministic "
                    f"downstream metadata, or a parser/stopping-rule/verifier "
                    f"defect can hide behind the same final binary reward")
    if need and parity["zero_transport_rate"] < 1.0:
        reasons.append(
            f"zero-alpha downstream_delta_rms is nonzero on "
            f"{(1 - parity['zero_transport_rate']) * 100:.1f}% of rows")
    if need and parity["derived_R_identity_rate"] < 1.0:
        reasons.append(
            f"derived-R identity {parity['derived_R_identity_rate']:.4f} < 1.0"
        )
    if abs(res.H_reach) > tau:
        reasons.append(f"|H_0| = {abs(res.H_reach):.4f} > tau = {tau:.4f}")
    if res.p_exact < 0.05:
        reasons.append(
            f"discordance asymmetric: n10={res.n10} n01={res.n01} p={res.p_exact:.4g}"
        )
    if res.ci_lo > 0 or res.ci_hi < 0:
        reasons.append(
            f"bootstrap interval [{res.ci_lo:+.4f}, {res.ci_hi:+.4f}] excludes zero"
        )
    return ZeroAlphaGate(parity, res, tau,
                         "HALT_FOR_PIPELINE_REPAIR" if reasons else "PROCEED", reasons)


# ==========================================================================
# append-only seal
# ==========================================================================


class Seal:
    """Single-write, append-only, atomic, hash-chained."""

    SECONDARY = (
        "secondary_magnitude", "stratum_hard_medium", "stratum_medium",
        "random_basis", "selection_headroom", "well_formedness",
        "wellformed_only_descriptive", "per_direction",
    )

    def __init__(self, path: str, manifest_sha256: str,
                 pregeneration_seal_sha256: str | None = None):
        if os.path.exists(path):
            raise SealViolation(
                f"seal already exists at {path}: use Seal.resume() to append a "
                f"declared secondary, or a fresh path for a rerun"
            )
        self.path = path
        self.manifest_sha256 = manifest_sha256
        self.pregeneration_seal_sha256 = (
            pregeneration_seal_sha256 or manifest_sha256
        )
        self.entries: list[dict] = []
        self._primary_open = False

    @classmethod
    def resume(cls, path: str, manifest_sha256: str,
               pregeneration_seal_sha256: str) -> "Seal":
        """Verify and reopen an existing result chain for declared secondaries."""
        if not os.path.exists(path):
            raise SealViolation(f"cannot resume missing seal {path}")
        with open(path) as fh:
            payload = json.load(fh)
        if payload.get("manifest_sha256") != manifest_sha256:
            raise SealViolation("result seal manifest hash disagrees")
        if payload.get("pregeneration_seal_sha256") != pregeneration_seal_sha256:
            raise SealViolation("result seal pre-generation hash disagrees")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise SealViolation("result seal entries are malformed")
        parent = pregeneration_seal_sha256
        primary_count = 0
        for i, entry in enumerate(entries):
            if entry.get("parent") != parent:
                raise SealViolation(f"result seal chain parent mismatch at entry {i}")
            want = cls._h({k: entry.get(k) for k in ("name", "parent", "payload")})
            if entry.get("hash") != want:
                raise SealViolation(f"result seal hash mismatch at entry {i}")
            parent = want
            primary_count += entry.get("name") == "primary"
        if payload.get("chain_head") != (parent if entries else None):
            raise SealViolation("result seal chain_head disagrees")
        if primary_count > 1:
            raise SealViolation("result seal contains multiple primary entries")
        obj = cls.__new__(cls)
        obj.path = path
        obj.manifest_sha256 = manifest_sha256
        obj.pregeneration_seal_sha256 = pregeneration_seal_sha256
        obj.entries = entries
        obj._primary_open = primary_count == 1
        return obj

    @staticmethod
    def _h(obj) -> str:
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _write(self) -> None:
        payload = {
            "manifest_sha256": self.manifest_sha256,
            "pregeneration_seal_sha256": self.pregeneration_seal_sha256,
            "entries": self.entries,
            "chain_head": self.entries[-1]["hash"] if self.entries else None,
        }
        d = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".seal.tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _append(self, name: str, payload: Any) -> str:
        parent = (self.entries[-1]["hash"] if self.entries
                  else self.pregeneration_seal_sha256)
        entry = {"name": name,
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "parent": parent, "payload": payload}
        entry["hash"] = self._h({k: entry[k] for k in ("name", "parent", "payload")})
        self.entries.append(entry)
        self._write()
        return entry["hash"]

    def open_halt(self, gate: ZeroAlphaGate) -> str:
        """A halted run still leaves an immutable, hashed artifact."""
        return self._append("zero_alpha_halt", asdict(gate))

    def open_primary(self, result: EndpointResult) -> str:
        if self._primary_open:
            raise SealViolation("primary already opened; the seal is append-only")
        self._primary_open = True
        return self._append("primary", result.to_dict())

    def open_secondary(self, name: str, payload: Any) -> str:
        if name not in self.SECONDARY:
            raise SealViolation(f"{name!r} is not a declared secondary endpoint")
        if not self._primary_open:
            raise SealViolation(
                f"secondary endpoint {name!r} requested before the primary was "
                f"opened — sealed opening order violated"
            )
        return self._append(name, payload)


# ==========================================================================
# confirmatory pipeline
# ==========================================================================


def load_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load_task_manifest(path: str) -> list[dict]:
    """
    Sealed task manifest, JSONL. Task IDs alone are not enough: the per-task
    stratum and its rank within that stratum are load-bearing design variables
    (the rank drives the Latin square), so they are sealed, not derived from
    whatever the records happen to say.

    Required per line: task_id, stratum, generator_setting_id,
    sealed_rank_within_stratum, task_content_sha256, options, symbolic.
    The full task content (options, symbolic) is required so that gate G22 can
    recompute scoring from generated text rather than trust stored booleans.
    """
    out = []
    with open(path) as fh:
        for i, ln in enumerate(fh):
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                raise IntegrityError(
                    "G7_TASK_SET",
                    f"{os.path.basename(path)} line {i + 1} is not JSON; the "
                    f"sealed task manifest must be JSONL carrying stratum and "
                    f"rank, not a bare list of task IDs")
            for k in ("task_id", "stratum", "generator_setting_id",
                      "sealed_rank_within_stratum", "task_content_sha256",
                      "options", "symbolic"):
                if k not in row:
                    raise IntegrityError(
                        "G7_TASK_SET",
                        f"{os.path.basename(path)} line {i + 1}: missing {k!r}")
            if row["stratum"] not in STRATA:
                raise IntegrityError(
                    "G7_TASK_SET",
                    f"line {i + 1}: unknown stratum {row['stratum']!r}")
            out.append(row)
    ids = [r["task_id"] for r in out]
    if len(set(ids)) != len(ids):
        raise IntegrityError("G7_TASK_SET",
                             f"{os.path.basename(path)}: duplicate task_id")
    by_s: dict[str, list[int]] = {}
    for r in out:
        rk = r["sealed_rank_within_stratum"]
        if isinstance(rk, bool) or not isinstance(rk, int) or rk < 0:
            raise IntegrityError(
                "G7_TASK_SET",
                f"{r['task_id']}: sealed_rank_within_stratum must be a "
                f"non-negative int, got {rk!r}")
        by_s.setdefault(r["stratum"], []).append(rk)
    for st, rks in by_s.items():
        if sorted(rks) != list(range(len(rks))):
            dup = sorted({x for x in rks if rks.count(x) > 1})
            raise IntegrityError(
                "G7_TASK_SET",
                f"{os.path.basename(path)} stratum {st!r}: sealed ranks are not "
                f"a permutation of 0..{len(rks) - 1}"
                + (f"; duplicated {dup}" if dup else "")
                + ". The Latin square can be perfectly obeyed and still fail to "
                  "balance actions across streams.",
                detail={"stratum": st, "duplicated": dup})
    ch = [r["task_content_sha256"] for r in out]
    if len(set(ch)) != len(ch):
        raise IntegrityError(
            "G7_TASK_SET",
            f"{os.path.basename(path)}: duplicate task_content_sha256 within the "
            f"cohort; two task IDs share identical content")
    return out


def load_calibration_task_manifest(path: str) -> list[dict]:
    """
    Sealed CALIBRATION task manifest, JSONL.  The real calibration cohort is
    frozen BEFORE difficulty banding exists, so its rows carry the outcome-blind
    placeholder stratum ``CALIBRATION_SETTING`` and per-setting ranks — not the
    confirmatory stratum/rank schema.  The confirmatory pipeline uses this
    manifest only for task-set and content disjointness, so only identity
    fields are required here.  (v2.0 routed this file through
    ``load_task_manifest`` and would have rejected the real sealed cohort at
    confirmatory analysis time; the fixtures fabricated calibration manifests
    in confirmatory format and never exercised the real schema.)
    """
    out = []
    with open(path) as fh:
        for i, ln in enumerate(fh):
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                raise IntegrityError(
                    "G7_TASK_SET",
                    f"{os.path.basename(path)} line {i + 1} is not JSON")
            for k in ("task_id", "task_content_sha256", "generator_setting_id"):
                if k not in row or not isinstance(row[k], str) or not row[k]:
                    raise IntegrityError(
                        "G7_TASK_SET",
                        f"{os.path.basename(path)} line {i + 1}: missing or "
                        f"invalid {k!r}")
            out.append(row)
    if not out:
        raise IntegrityError("G7_TASK_SET",
                             f"{os.path.basename(path)} is empty")
    ids = [r["task_id"] for r in out]
    ch = [r["task_content_sha256"] for r in out]
    if len(set(ids)) != len(ids) or len(set(ch)) != len(ch):
        raise IntegrityError(
            "G7_TASK_SET",
            f"{os.path.basename(path)}: duplicate task_id or content hash")
    return out


def derive_stream_seed(master_seed: int, task_id: str, stream_index: int) -> int:
    """Deterministic uint64 stream seed; no post-hoc seed-matrix discretion."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or not (0 <= master_seed < 2**64):
        raise IntegrityError("G16_SEED_MATRIX_BINDING",
                             "cohorts.master_seed must be an integer in [0, 2^64)")
    if not isinstance(task_id, str) or not task_id:
        raise IntegrityError("G16_SEED_MATRIX_BINDING", "task_id must be a non-empty string")
    if isinstance(stream_index, bool) or not isinstance(stream_index, int) or not (0 <= stream_index < BANK_SIZE):
        raise IntegrityError("G16_SEED_MATRIX_BINDING", "stream_index outside 0..7")
    tid = task_id.encode("utf-8")
    payload = (SEED_DOMAIN + master_seed.to_bytes(8, "big")
               + len(tid).to_bytes(4, "big") + tid
               + stream_index.to_bytes(1, "big"))
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def validate_seed_matrix(seed_matrix: dict, confirmatory, manifest: Manifest) -> None:
    """Validate task/stream coverage, uniqueness, and exact deterministic derivation."""
    if manifest.get("cohorts.seed_hash_function") != "sha256":
        raise IntegrityError("G16_SEED_MATRIX_BINDING", "seed_hash_function must be sha256")
    if manifest.get("cohorts.seed_derivation_scheme") != SEED_DERIVATION_SCHEME:
        raise IntegrityError(
            "G16_SEED_MATRIX_BINDING",
            f"unsupported seed derivation scheme {manifest.get('cohorts.seed_derivation_scheme')!r}")
    if not isinstance(seed_matrix, dict):
        raise IntegrityError("G16_SEED_MATRIX_BINDING", "seed matrix must be a JSON object")
    task_ids = {t["task_id"] for t in confirmatory}
    if set(seed_matrix) != task_ids:
        raise IntegrityError(
            "G16_SEED_MATRIX_BINDING",
            "seed-matrix task set does not equal the sealed confirmatory task set",
            detail={"missing": sorted(task_ids - set(seed_matrix)),
                    "extra": sorted(set(seed_matrix) - task_ids)})
    master = manifest.get("cohorts.master_seed")
    for task_id in sorted(task_ids):
        entry = seed_matrix[task_id]
        if not isinstance(entry, dict):
            raise IntegrityError("G16_SEED_MATRIX_BINDING",
                                 f"task {task_id}: seed entry must be an object")
        want_keys = {str(i) for i in range(BANK_SIZE)}
        if set(entry) != want_keys:
            raise IntegrityError(
                "G16_SEED_MATRIX_BINDING",
                f"task {task_id}: streams must be exactly 0..7",
                detail={"missing": sorted(want_keys - set(entry)),
                        "extra": sorted(set(entry) - want_keys)})
        vals = []
        for i in range(BANK_SIZE):
            val = entry[str(i)]
            if isinstance(val, bool) or not isinstance(val, int):
                raise IntegrityError("G16_SEED_MATRIX_BINDING",
                                     f"task {task_id} stream {i}: seed must be int")
            want = derive_stream_seed(master, task_id, i)
            if val != want:
                raise IntegrityError(
                    "G16_SEED_MATRIX_BINDING",
                    f"task {task_id} stream {i}: sealed seed {val} != deterministic {want}",
                    detail={"task_id": task_id, "stream": i,
                            "sealed": val, "deterministic": want})
            vals.append(val)
        if len(set(vals)) != BANK_SIZE:
            raise IntegrityError("G16_SEED_MATRIX_BINDING",
                                 f"task {task_id}: deterministic stream seeds are not unique")


def gate_G16_seed_matrix_binding(records, seed_matrix: dict, confirmatory, manifest: Manifest) -> None:
    """
    G4 proves the two arms used the SAME seeds. It does not prove they used the
    SEALED seeds: a global shift preserves CRN pairing and passes everything.
    """
    validate_seed_matrix(seed_matrix, confirmatory, manifest)
    for r in records:
        t, i = r["task_id"], str(r["stream_index"])
        if t not in seed_matrix:
            raise IntegrityError(
                "G16_SEED_MATRIX_BINDING", f"task {t} absent from the sealed seed matrix")
        entry = seed_matrix[t]
        if i not in entry:
            raise IntegrityError(
                "G16_SEED_MATRIX_BINDING",
                f"task {t}: sealed seed matrix has no stream {i}; it must carry "
                f"explicit per-stream seeds, not an empty placeholder")
        if int(entry[i]) != int(r["seed"]):
            raise IntegrityError(
                "G16_SEED_MATRIX_BINDING",
                f"task {t} arm {r['arm']} stream {i}: seed {r['seed']} != sealed "
                f"{entry[i]} — the banks are CRN-paired but not to the sealed streams",
                detail={"task_id": t, "stream": i, "actual": r["seed"],
                        "sealed": int(entry[i])})


def gate_G17_manifest_consistency(manifest: Manifest) -> None:
    """Cross-check stored quantities that can silently disagree."""
    a = manifest.get("action_space")
    dup = [("structured_axes_sha256", "artifact_hashes.structured_axis_tensor"),
           ("random_axes_sha256", "artifact_hashes.random_axis_tensor")]
    for k, other in dup:
        if k in a:
            raise IntegrityError(
                "G17_MANIFEST_CONSISTENCY",
                f"action_space.{k} duplicates {other}; keep one source of truth")
    if "structured_axes_gram" in a and "gram_matrix" in a.get("structured_axes", {}):
        raise IntegrityError(
            "G17_MANIFEST_CONSISTENCY",
            "action_space.structured_axes_gram duplicates "
            "action_space.structured_axes.gram_matrix")

    verdict = manifest.get("power.calibration_verdict")
    proceed = {"PROCEED_POWERED", "PROCEED_BUDGET_CONSTRAINED"}
    if verdict not in proceed:
        raise IntegrityError(
            "G17_CALIBRATION_HALT",
            f"calibration verdict {verdict!r} does not authorize confirmation",
            detail={"verdict": verdict},
        )

    if bool(manifest.get("code.repository_clean_required")):
        empty_diff = hashlib.sha256(b"").hexdigest()
        if manifest.get("code.repository_diff_sha256") != empty_diff:
            raise IntegrityError(
                "G17_MANIFEST_CONSISTENCY",
                "repository_clean_required is true but repository_diff_sha256 "
                "is not SHA256(empty diff)")

    q = float(manifest.get("power.q_disc_upper_confidence_bound"))
    gp = int(manifest.get("power.G_power"))
    gm = int(manifest.get("power.G_max"))
    gpool = int(manifest.get("power.G_pool"))
    gc = int(manifest.get("power.G_confirm"))
    dt = float(manifest.get("power.delta_target"))
    tp = float(manifest.get("power.target_power"))
    al = float(manifest.get("power.alpha_level"))
    if not (0 < q <= 1):
        raise IntegrityError("G17_MANIFEST_CONSISTENCY", f"q_disc upper bound {q} outside (0,1]")
    if q < dt:
        raise IntegrityError(
            "G17_CALIBRATION_HALT",
            f"q_disc upper bound {q:.6f} is below target headroom {dt:.6f}; "
            "|H_reach| <= q_disc, so the target is excluded before confirmation",
        )
    if gpool < 1:
        raise IntegrityError(
            "G17_CALIBRATION_HALT",
            f"G_pool {gpool} < 1: the banded candidate pool supplies no "
            f"confirmatory tasks")
    want_gp = required_G(q, dt, tp, al)
    if gp != want_gp:
        raise IntegrityError(
            "G17_MANIFEST_CONSISTENCY",
            f"G_power {gp} != required_G(q_upper={q}, delta={dt}) = {want_gp}",
            detail={"sealed": gp, "recomputed": want_gp})
    want_gc = min(gp, gm, gpool)
    if gc != want_gc:
        raise IntegrityError(
            "G17_MANIFEST_CONSISTENCY",
            f"G_confirm {gc} != min(G_power {gp}, G_max {gm}, G_pool {gpool}) "
            f"= {want_gc}")
    expected_verdict = ("PROCEED_POWERED" if gp <= min(gm, gpool)
                        else "PROCEED_BUDGET_CONSTRAINED")
    if verdict != expected_verdict:
        raise IntegrityError(
            "G17_MANIFEST_CONSISTENCY",
            f"calibration verdict {verdict} inconsistent with G_power={gp}, "
            f"G_max={gm}, G_pool={gpool}; expected {expected_verdict}")


def run_primary(
    records_path: str,
    manifest_path: str,
    confirmatory_task_manifest_path: str,
    calibration_task_manifest_path: str,
    calibration_precommit_path: str,
    seed_matrix_path: str,
    seal_path: str,
    provenance_path: str,
    pregeneration_seal_path: str,
    calibration_records_path: str,
    calibration_metadata_path: str,
    calibration_results_path: str,
    axis_package_path: str,
    candidate_pool_path: str,
) -> dict:
    """
    Sealed order:
      1. load complete manifest
      2. verify external pre-generation commitment
      3. verify exact resolved runtime configuration
      4. deterministically recompute calibration and bind every frozen output
      5. verify artifacts and record-level gates
      6. zero-alpha parity/null
      7. open the primary exactly once
    """
    man = Manifest(manifest_path)
    gate_G17_manifest_consistency(man)
    verify_pregeneration_seal(
        pregeneration_seal_path, man, provenance_path,
        confirmatory_task_manifest_path, calibration_task_manifest_path,
        calibration_precommit_path, seed_matrix_path, calibration_records_path,
        calibration_metadata_path,
        calibration_results_path, axis_package_path, candidate_pool_path,
    )
    man.verify_artifacts({
        "cohorts.confirmatory_task_manifest_sha256": confirmatory_task_manifest_path,
        "cohorts.calibration_task_manifest_sha256": calibration_task_manifest_path,
        "cohorts.confirmatory_candidate_pool_sha256": candidate_pool_path,
        "cohorts.seed_matrix_sha256": seed_matrix_path,
        "code.analysis_module_sha256": os.path.abspath(__file__),
        "code.calibration_module_sha256": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "calibration_analysis.py"),
        "code.orchestrator_module_sha256": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "run_o1_v2_orchestrator.py"),
        "code.transport_module_sha256": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "o1_transport_v2.py"),
        "code.axis_verifier_sha256": os.path.join(axis_package_path, "verify_axis_artifact.py"),
        "calibration.precommit_sha256": calibration_precommit_path,
        "calibration.raw_records_sha256": calibration_records_path,
        "calibration.metadata_sha256": calibration_metadata_path,
        "calibration.results_sha256": calibration_results_path,
    })
    prov = gate_G14_provenance(man, provenance_path)
    from verify_axis_artifact import verify as verify_axis_artifact
    axis_report = verify_axis_artifact(axis_package_path)
    if axis_report.get("verdict") != "SEALABLE":
        raise IntegrityError("G20_AXIS_ARTIFACT", "axis package did not verify as SEALABLE")

    # Imported lazily to avoid a module-import cycle: calibration_analysis uses
    # exact power and CP helpers from this module.
    from calibration_analysis import verify_calibration_binding
    cal_result = verify_calibration_binding(
        calibration_records_path, calibration_metadata_path,
        calibration_results_path, manifest_path, calibration_task_manifest_path,
        calibration_precommit_path, candidate_pool_path,
    )
    if cal_result["verdict"] not in ("PROCEED_POWERED", "PROCEED_BUDGET_CONSTRAINED"):
        raise IntegrityError(
            "G19_CALIBRATION_BINDING",
            f"calibration result {cal_result['verdict']!r} does not authorize confirmation")

    conf = load_task_manifest(confirmatory_task_manifest_path)
    cal = load_calibration_task_manifest(calibration_task_manifest_path)
    pool = load_jsonl(candidate_pool_path)
    gate_G23_pool_membership(conf, pool)
    with open(seed_matrix_path) as fh:
        seed_matrix = json.load(fh)
    records = load_jsonl(records_path)
    gate_G21_record_run_binding(records, provenance_path, pregeneration_seal_path)
    validate(records, man, conf, cal, seed_matrix)

    pregen_hash = sha256_file(pregeneration_seal_path)
    seal = Seal(seal_path, man.sha256, pregen_hash)
    gate = zero_alpha_gate(records, man)
    if gate.verdict != "PROCEED":
        return {
            "manifest_sha256": man.sha256,
            "pregeneration_seal_sha256": sha256_file(pregeneration_seal_path),
            "records_sha256": sha256_file(records_path),
            "zero_alpha": asdict(gate), "primary": None,
            "verdict": gate.verdict, "halt_sha256": seal.open_halt(gate),
        }

    res = endpoint(records, "structured", man, label="H_reach_primary")
    return {
        "manifest_sha256": man.sha256,
        "pregeneration_seal_sha256": sha256_file(pregeneration_seal_path),
        "records_sha256": sha256_file(records_path),
        "zero_alpha": asdict(gate), "primary": res.to_dict(),
        "primary_sha256": seal.open_primary(res),
        "provenance_verified": True,
        "calibration_results_sha256": sha256_file(calibration_results_path),
        "axis_verification": axis_report,
        "coupling": coupling_survival(records, "structured"),
        "outcome_diversity": outcome_diversity(records, "structured"),
        "transport": transport_summary(records, "structured", manifest=man),
        "axis_transport_profile": axis_transport_profile(
            records, "structured", manifest=man),
        "verdict": res.verdict,
    }
