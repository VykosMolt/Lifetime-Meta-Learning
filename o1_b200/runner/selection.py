"""Deterministic implementation of the frozen backend-selection policy.

Pure function over (equivalence report, benchmark results); no I/O, no model
access, no O1 outcome anywhere in its inputs.  The JSON policy
(policies/B200_BACKEND_SELECTION_POLICY.json) is the human-readable frozen
form; this module is its executable form and the tests hold them together.
"""
from __future__ import annotations

SEMANTIC_COMPLEXITY = {
    "REFERENCE_SERIAL": 0,
    "B200_REPLICA": 1,
    "B200_BATCHED": 2,           # without compaction
    "B200_BATCHED_COMPACTION": 3,
}

REQUIRED_GATES = (
    "structural_pass", "parser_verifier_pass", "action_seed_mapping_exact",
    "intervention_pass", "transport_pass", "resume_pass",
    "no_missing_or_duplicate_rows", "no_oom", "free_hbm_fraction_ok",
    "no_unvalidated_optimization", "throughput_stable",
    "scientific_config_unchanged",
)

MIN_FREE_HBM_FRACTION = 0.15
MAX_THROUGHPUT_SPREAD = 0.25


class SelectionError(RuntimeError):
    pass


def is_eligible(candidate: dict) -> tuple[bool, list[str]]:
    """candidate carries gates plus measured fields."""
    failures = []
    for gate in REQUIRED_GATES:
        if candidate.get(gate) is not True:
            failures.append(gate)
    free = candidate.get("steady_state_free_hbm_fraction")
    if free is None or free < MIN_FREE_HBM_FRACTION:
        if "free_hbm_fraction_ok" not in failures:
            failures.append("free_hbm_fraction_ok")
    spread = candidate.get("throughput_stability_spread")
    if spread is None or spread > MAX_THROUGHPUT_SPREAD:
        if "throughput_stable" not in failures:
            failures.append("throughput_stable")
    return (not failures, failures)


def _complexity(candidate: dict) -> int:
    backend = candidate["backend"]
    if backend == "B200_BATCHED" and candidate.get("compaction"):
        return SEMANTIC_COMPLEXITY["B200_BATCHED_COMPACTION"]
    return SEMANTIC_COMPLEXITY[backend]


def select_backend(candidates: list[dict]) -> dict:
    """Frozen rule: highest verified completed rows/hour among eligible.

    Tie-break: lower peak HBM; lower semantic complexity; smaller
    worker/batch count; lexicographic config_id.
    """
    judged = []
    for c in candidates:
        ok, failures = is_eligible(c)
        judged.append({**c, "eligible": ok, "gate_failures": failures})
    eligible = [c for c in judged if c["eligible"]]
    if not eligible:
        raise SelectionError(
            "no eligible backend configuration; REFERENCE_SERIAL must be "
            "re-benchmarked or the session aborted")

    def sort_key(c):
        return (
            -float(c["completed_rows_per_hour"]),
            float(c.get("peak_hbm_reserved_bytes", float("inf"))),
            _complexity(c),
            int(c.get("workers", 1)) + int(c.get("batch", 1)),
            int(c.get("workers", 1)),
            str(c["config_id"]),
        )

    winner = sorted(eligible, key=sort_key)[0]
    return {"selected": winner, "eligible": eligible, "all_judged": judged}
