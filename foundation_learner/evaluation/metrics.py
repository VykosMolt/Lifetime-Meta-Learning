"""The 14 frozen metrics of contract section 9 (+3 Amendment 13 additions).

PURE functions over records.

Every function here takes ``flb200.episode_record.v1`` dicts (or the small
summary dicts produced by ``interference``/``poison_eval``/``remap_eval``) and
returns numbers.  No model, no I/O, no randomness.

MACRO CONVENTION (frozen, applied everywhere)
---------------------------------------------
Aggregation is always FAMILY-LEVEL FIRST: episodes are averaged within a
family, then families are averaged with equal weight.  A family contributing
ten times as many episodes as another therefore cannot move an aggregate;
``tests/hostile/test_hostile_metric_family_domination.py`` pins this.

Curve conventions
-----------------
``R_k`` is the mean verifier success over the attempts carried by interaction
index ``k`` in one episode (so ``R_5`` averages the three QUERY items and
``R_6`` the two TRANSFER items, contract section 6).  A family curve is the
episode-mean of ``R_k``; the macro curve is the family-mean of family curves.
Because every episode of ``EPISODE_STRUCTURE_V0`` shares the index grid
``0..6``, the OLS slope of the macro curve equals the mean of the per-family
OLS slopes; :func:`improvement_slope` returns the macro-curve slope and
:func:`per_family_slopes` exposes the family values, and
:func:`slope_conventions_agree` checks the identity on real inputs rather than
assuming it.

Amendment 13 additions (metrics 15-17, frozen before any run)
-------------------------------------------------------------
15 ``answer_line_rate`` — the fraction of MODEL_ATTEMPT generations that
   contained a parseable ``ANSWER:`` line, macro-aggregated and also reported
   per family and per interaction index.  A macro-AULC cell whose rate is below
   ``ANSWER_LINE_RATE_FLAG_THRESHOLD`` is FLAGGED ``FORMAT_NONCOMPLIANT``
   alongside the number; the number is never replaced or suppressed.
16 ``aulc_post_feedback_fresh`` — macro-AULC restricted to the fresh indices
   {4, 5, 6}, where an answer-flip heuristic cannot pay.
17 ``flip_attributable_success_rate`` — the share of revision successes on
   binary-answer families explained by "emit the other label after INCORRECT".

Amendment 15 addition (metric 18)
---------------------------------
18 ``constant_answer_baseline`` — the AULC a CONSTANT-ANSWER policy would score
   on each family, computed with this module's own aggregation so it is
   directly comparable to ``per_family_aulc``.  Open-answer families report
   ``None`` (a constant answer is not a policy there) together with the number
   of distinct answers, so the omission is visible.  Pure reporting: no gate,
   threshold or promotion rule consumes it.

Records that carry no scoreable curve (``status = ONLINE_BUDGET_EXCEEDED``) are
excluded from every aggregate and COUNTED by :func:`excluded_records`, which
``summarize`` always reports.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

METRICS_SCHEMA = "flb200.metrics_v0.v1"

#: contract section 9, metric 7
DEFAULT_THRESHOLD = 0.5
#: contract section 6
DEFAULT_INDICES: tuple[int, ...] = tuple(range(7))
#: contract section 9, metric 8 (related-task transfer is the index-4 attempt)
RELATED_INDEX = 4

#: Interaction indices carrying FRESH items that no earlier item's feedback can
#: be answer-flipped into (Amendment 13, metric 15).  Index 4 is the related
#: task, 5 the three queries, 6 the two transfers: all are new instances, so a
#: "flip the label the feedback just called wrong" heuristic cannot pay there,
#: while indices 1 and 3 are REVISIONS of an item whose certified verdict the
#: model has already seen.
POST_FEEDBACK_FRESH_INDICES: tuple[int, ...] = (4, 5, 6)
#: revision indices (contract section 6): the attempts a flip heuristic reaches
REVISION_INDICES: tuple[int, ...] = (1, 3)
#: FL2 imitates attempt0 behaviour directly, so a delta against FL2 that
#: includes index 0 mixes an imitation handicap into the learning claim
#: (Amendment 13, metric 17).
INDICES_EXCLUDING_ZERO: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

#: Families whose answer is one of exactly TWO labels, i.e. where "the other
#: label" is uniquely defined and an answer-flip heuristic is available.  Pinned
#: here rather than inferred from records (a record set need not contain both
#: labels); ``tests/test_evaluation_metrics.py`` checks the list against the
#: real generators, so it cannot drift from the ecology.
BINARY_ANSWER_FAMILIES: tuple[str, ...] = (
    "boolean_rule", "constraint_rules", "grammar_classification",
    "graph_edge_semantics")

#: An arm/family/index cell whose ``answer_line_rate`` falls below this is
#: FLAGGED in the report (never replaced, never dropped): its macro-AULC is
#: dominated by generations that never emitted a parseable ANSWER line, so the
#: number measures format compliance more than competence.
ANSWER_LINE_RATE_FLAG_THRESHOLD = 0.5
FORMAT_NONCOMPLIANT_FLAG = "FORMAT_NONCOMPLIANT"

#: episode-record ``status`` values (``evaluation.learning_curve``)
STATUS_COMPLETE = "COMPLETE"
STATUS_ONLINE_BUDGET_EXCEEDED = "ONLINE_BUDGET_EXCEEDED"
#: A ratio whose denominator is only floating-point noise (e.g. a "gain" of
#: 1e-17 produced by averaging identical numbers) carries no information; such
#: a ratio is reported as undefined rather than as a spurious 1.0.
MIN_DENOMINATOR = 1e-9


# --------------------------------------------------------------------------
# record accessors
# --------------------------------------------------------------------------
def record_family(record: Mapping[str, Any]) -> str:
    fid = record.get("family_id")
    if fid is None:
        raise KeyError("episode record has no family_id")
    return str(fid)


def record_R(record: Mapping[str, Any]) -> dict[int, float]:
    raw = record.get("R") or {}
    return {int(k): float(v) for k, v in raw.items()}


def record_aulc(record: Mapping[str, Any],
                indices: Sequence[int] | None = None) -> float | None:
    """Area under the learning curve of ONE episode = mean over indices of R_k."""
    R = record_R(record)
    keys = [k for k in (indices if indices is not None else sorted(R)) if k in R]
    if not keys:
        return None
    return float(sum(R[k] for k in keys) / len(keys))


def record_status(record: Mapping[str, Any]) -> str:
    """``status`` of an episode record (records written before Amendment 13
    carry none and are treated as COMPLETE)."""
    return str(record.get("status") or STATUS_COMPLETE)


def record_is_scoreable(record: Mapping[str, Any]) -> bool:
    """True when the record carries a COMPLETE curve that metrics may consume."""
    return record_status(record) == STATUS_COMPLETE and bool(record_R(record))


def excluded_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Explicit accounting of records that carry no scoreable curve.

    Contract-relevant because an episode may now be recorded and EXCLUDED
    (``ONLINE_BUDGET_EXCEEDED``) instead of aborting a batch: the count is
    reported, never inferred from a missing row.
    """
    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    excluded_ids: list[str] = []
    for r in records:
        if record_is_scoreable(r):
            continue
        status = record_status(r)
        if status == STATUS_COMPLETE:
            status = "COMPLETE_BUT_EMPTY_R"
        by_status[status] = by_status.get(status, 0) + 1
        fid = str(r.get("family_id", ""))
        by_family[fid] = by_family.get(fid, 0) + 1
        excluded_ids.append(str(r.get("episode_id", "")))
    n_excluded = len(excluded_ids)
    return {
        "n_records": len(records),
        "n_scoreable": len(records) - n_excluded,
        "n_excluded": n_excluded,
        "excluded_fraction": (float(n_excluded / len(records))
                              if records else None),
        "by_status": by_status,
        "by_family": by_family,
        "excluded_episode_ids": excluded_ids,
    }


def group_by_family(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        out.setdefault(record_family(r), []).append(r)
    return out


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def macro_mean(records: Sequence[Mapping[str, Any]],
               value_fn: Callable[[Mapping[str, Any]], float | None]) -> float | None:
    """Family means first, then the unweighted mean over families."""
    per_family = []
    for _fid, rs in sorted(group_by_family(records).items()):
        fam = _mean(value_fn(r) for r in rs)
        if fam is not None:
            per_family.append(fam)
    return _mean(per_family)


def per_family_mean(records: Sequence[Mapping[str, Any]],
                    value_fn: Callable[[Mapping[str, Any]], float | None]
                    ) -> dict[str, float]:
    out: dict[str, float] = {}
    for fid, rs in sorted(group_by_family(records).items()):
        fam = _mean(value_fn(r) for r in rs)
        if fam is not None:
            out[fid] = fam
    return out


# --------------------------------------------------------------------------
# 1-3: AULC and deltas
# --------------------------------------------------------------------------
def macro_aulc(records: Sequence[Mapping[str, Any]],
               indices: Sequence[int] | None = None) -> float | None:
    """METRIC 1 — macro-AULC (mean over families of mean success over 0..K)."""
    return macro_mean(records, lambda r: record_aulc(r, indices))


def per_family_aulc(records: Sequence[Mapping[str, Any]],
                    indices: Sequence[int] | None = None) -> dict[str, float]:
    return per_family_mean(records, lambda r: record_aulc(r, indices))


def delta_aulc(treatment: Sequence[Mapping[str, Any]],
               baseline: Sequence[Mapping[str, Any]],
               indices: Sequence[int] | None = None) -> float | None:
    """METRICS 2/3 — macro-AULC(treatment) - macro-AULC(baseline)."""
    a = macro_aulc(treatment, indices)
    b = macro_aulc(baseline, indices)
    if a is None or b is None:
        return None
    return float(a - b)


# --------------------------------------------------------------------------
# 4-6: curve shape
# --------------------------------------------------------------------------
def per_family_curve(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for fid, rs in sorted(group_by_family(records).items()):
        curve: dict[int, float] = {}
        indices = sorted({k for r in rs for k in record_R(r)})
        for k in indices:
            vals = [record_R(r)[k] for r in rs if k in record_R(r)]
            if vals:
                curve[k] = float(sum(vals) / len(vals))
        out[fid] = curve
    return out


def macro_curve(records: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    fam = per_family_curve(records)
    indices = sorted({k for c in fam.values() for k in c})
    out: dict[int, float] = {}
    for k in indices:
        vals = [c[k] for c in fam.values() if k in c]
        if vals:
            out[k] = float(sum(vals) / len(vals))
    return out


def r_at(records: Sequence[Mapping[str, Any]], k: int) -> float | None:
    """Macro ``R_k``."""
    return macro_curve(records).get(int(k))


def r_0(records: Sequence[Mapping[str, Any]]) -> float | None:
    """METRIC 4 — macro R_0."""
    return r_at(records, 0)


def r_K(records: Sequence[Mapping[str, Any]], K: int | None = None) -> float | None:
    """METRIC 5 — macro R_K (K = the largest observed interaction index)."""
    curve = macro_curve(records)
    if not curve:
        return None
    key = max(curve) if K is None else int(K)
    return curve.get(key)


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Ordinary least squares slope of ``ys`` on ``xs`` (numpy only)."""
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    if x.size < 2 or y.size != x.size:
        return None
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom <= 0.0:
        return None
    return float((xc * (y - y.mean())).sum() / denom)


def ols_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """Return ``(slope, intercept)`` of the OLS fit."""
    slope = ols_slope(xs, ys)
    if slope is None:
        return None
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    return slope, float(y.mean() - slope * x.mean())


def per_family_slopes(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for fid, curve in per_family_curve(records).items():
        if len(curve) >= 2:
            s = ols_slope(sorted(curve), [curve[k] for k in sorted(curve)])
            if s is not None:
                out[fid] = s
    return out


def improvement_slope(records: Sequence[Mapping[str, Any]]) -> float | None:
    """METRIC 6 — OLS slope of the MACRO learning curve over interaction index."""
    curve = macro_curve(records)
    if len(curve) < 2:
        return None
    ks = sorted(curve)
    return ols_slope(ks, [curve[k] for k in ks])


def slope_conventions_agree(records: Sequence[Mapping[str, Any]],
                            tol: float = 1e-9) -> bool:
    """True when macro-curve slope == mean of per-family slopes (equal grids)."""
    macro = improvement_slope(records)
    fam = per_family_slopes(records)
    if macro is None or not fam:
        return False
    return abs(macro - float(np.mean(list(fam.values())))) <= tol


# --------------------------------------------------------------------------
# 7: interactions to threshold
# --------------------------------------------------------------------------
def interactions_to_threshold(records: Sequence[Mapping[str, Any]],
                              threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """METRIC 7 — first interaction index whose family success >= threshold.

    ``None`` where the family never reaches the threshold ("where
    identifiable"); the macro summary averages only the identifiable families
    and reports how many were identifiable, never silently dropping the rest.
    """
    per_family: dict[str, int | None] = {}
    for fid, curve in per_family_curve(records).items():
        hit = None
        for k in sorted(curve):
            if curve[k] >= float(threshold):
                hit = int(k)
                break
        per_family[fid] = hit
    identifiable = [v for v in per_family.values() if v is not None]
    return {
        "threshold": float(threshold),
        "per_family": per_family,
        "macro_mean": (float(np.mean(identifiable)) if identifiable else None),
        "n_identifiable": len(identifiable),
        "n_families": len(per_family),
    }


# --------------------------------------------------------------------------
# 8-9: transfer
# --------------------------------------------------------------------------
def related_task_transfer(records: Sequence[Mapping[str, Any]]) -> float | None:
    """METRIC 8 — macro R_4 (fresh instance of the same latent rule)."""
    return r_at(records, RELATED_INDEX)


def whole_family_transfer(records: Sequence[Mapping[str, Any]],
                          trained_family_ids: Iterable[str]) -> float | None:
    """METRIC 9 — macro-AULC restricted to families whose GENERATOR was unseen."""
    trained = {str(f) for f in trained_family_ids}
    unseen = [r for r in records if record_family(r) not in trained]
    return macro_aulc(unseen)


# --------------------------------------------------------------------------
# 10: context-reset persistence
# --------------------------------------------------------------------------
def _post_reset_mean(record: Mapping[str, Any], reset_from_index: int) -> float | None:
    R = record_R(record)
    keys = [k for k in sorted(R) if k >= int(reset_from_index)]
    if not keys:
        return None
    return float(sum(R[k] for k in keys) / len(keys))


def context_reset_persistence(reset_records: Sequence[Mapping[str, Any]],
                              history_records: Sequence[Mapping[str, Any]],
                              reset_from_index: int = 4) -> dict[str, Any]:
    """METRIC 10 — retained fraction of the in-context gain.

    Per family:  ``(post_reset - R_0) / (post_history - R_0)`` where ``R_0`` is
    the pre-learning baseline measured on the matched history arm and ``post_*``
    is the mean success over interaction indices >= ``reset_from_index``.  The
    ratio is ``None`` when the history arm shows no in-context gain (a
    denominator <= 0 makes "retained fraction" undefined; it is reported as
    such rather than clipped).
    """
    hist_by_family = group_by_family(history_records)
    reset_by_family = group_by_family(reset_records)
    per_family: dict[str, Any] = {}
    for fid in sorted(set(hist_by_family) | set(reset_by_family)):
        hist = hist_by_family.get(fid, [])
        rst = reset_by_family.get(fid, [])
        base = _mean(record_R(r).get(0) for r in hist)
        post_h = _mean(_post_reset_mean(r, reset_from_index) for r in hist)
        post_r = _mean(_post_reset_mean(r, reset_from_index) for r in rst)
        gain_h = None if (base is None or post_h is None) else post_h - base
        gain_r = None if (base is None or post_r is None) else post_r - base
        ratio = None
        if gain_h is not None and gain_r is not None and gain_h > MIN_DENOMINATOR:
            ratio = float(gain_r / gain_h)
        per_family[fid] = {
            "baseline_R0": base,
            "post_history": post_h,
            "post_reset": post_r,
            "gain_history": gain_h,
            "gain_reset": gain_r,
            "retained_fraction": ratio,
        }
    ratios = [v["retained_fraction"] for v in per_family.values()
              if v["retained_fraction"] is not None]
    return {
        "reset_from_index": int(reset_from_index),
        "per_family": per_family,
        "macro_retained_fraction": (float(np.mean(ratios)) if ratios else None),
        "n_families_defined": len(ratios),
        "n_families": len(per_family),
        "macro_gain_history": _mean(v["gain_history"] for v in per_family.values()),
        "macro_gain_reset": _mean(v["gain_reset"] for v in per_family.values()),
    }


# --------------------------------------------------------------------------
# 11: A -> B -> A retention / interference
# --------------------------------------------------------------------------
def retention_interference(chain_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """METRIC 11 — family-balanced retention ratio / interference cost.

    Input: the per-chain dicts produced by ``interference.run_interference_eval``
    (keys ``family_a``, ``aulc_a1``, ``aulc_a2``, ``aulc_b``,
    ``recovery_interactions``).
    """
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for c in chain_summaries:
        by_family.setdefault(str(c.get("family_a")), []).append(c)
    per_family: dict[str, Any] = {}
    for fid, cs in sorted(by_family.items()):
        a1 = _mean(c.get("aulc_a1") for c in cs)
        a2 = _mean(c.get("aulc_a2") for c in cs)
        ratio = None
        if a1 is not None and a2 is not None and a1 > MIN_DENOMINATOR:
            ratio = float(a2 / a1)
        per_family[fid] = {
            "n_chains": len(cs),
            "aulc_a1": a1,
            "aulc_a2": a2,
            "aulc_b": _mean(c.get("aulc_b") for c in cs),
            "retention_ratio": ratio,
            "interference_cost": (None if (a1 is None or a2 is None) else float(a1 - a2)),
            "recovery_interactions": _mean(
                c.get("recovery_interactions") for c in cs),
        }
    ratios = [v["retention_ratio"] for v in per_family.values()
              if v["retention_ratio"] is not None]
    costs = [v["interference_cost"] for v in per_family.values()
             if v["interference_cost"] is not None]
    return {
        "per_family": per_family,
        "macro_retention_ratio": (float(np.mean(ratios)) if ratios else None),
        "macro_interference_cost": (float(np.mean(costs)) if costs else None),
        "n_families": len(per_family),
        "n_chains": len(chain_summaries),
    }


# --------------------------------------------------------------------------
# 12-13: poison / remap robustness
# --------------------------------------------------------------------------
#: The condition ids are the DATA LAYER's frozen strings
#: (``ecology.poison.POISON_CONDITIONS``), imported when available so a
#: consumer can never drift from the producer.  Contract section 9 spells the
#: untouched condition "correct-informative" and metric 12 calls the same thing
#: "clean"; the data layer's id is ``clean`` and both spellings alias onto it.
def _producer_conditions() -> tuple[str, ...]:
    try:
        from foundation_learner.ecology.poison import POISON_CONDITIONS as _P  # type: ignore
        return tuple(str(c) for c in _P)
    except Exception:
        return ("clean", "correct-redundant", "irrelevant",
                "partially-misleading", "corrupted")


CLEAN_CONDITION = "clean"
POISON_CONDITIONS: tuple[str, ...] = _producer_conditions()
POISON_CONDITION_ALIASES: dict[str, str] = {
    "correct_informative": "clean",
    "correct-informative": "clean",
    "correct_redundant": "correct-redundant",
    "partially_misleading": "partially-misleading",
}


def canonical_poison_condition(name: str) -> str:
    key = str(name).strip()
    key = POISON_CONDITION_ALIASES.get(key, key)
    if key not in POISON_CONDITIONS:
        raise ValueError(
            f"unknown poison condition {name!r}; frozen set {POISON_CONDITIONS}")
    return key


def poison_robustness(records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]]
                      ) -> dict[str, Any]:
    """METRIC 12 — macro-AULC gap of each poison condition vs the clean one."""
    aulc = {canonical_poison_condition(c): macro_aulc(rs)
            for c, rs in records_by_condition.items()}
    clean = aulc.get(CLEAN_CONDITION)
    gaps = {c: (None if (clean is None or v is None) else float(clean - v))
            for c, v in aulc.items() if c != CLEAN_CONDITION}
    return {
        "clean_condition": CLEAN_CONDITION,
        "macro_aulc": aulc,
        "gap_vs_clean": gaps,
        "corrupted_gap": gaps.get("corrupted"),
    }


def remap_robustness(canonical_records: Sequence[Mapping[str, Any]],
                     remapped_records_by_variant: Mapping[str, Sequence[Mapping[str, Any]]]
                     ) -> dict[str, Any]:
    """METRIC 13 — macro-AULC gap between canonical and remapped surfaces."""
    base = macro_aulc(canonical_records)
    per_variant = {str(k): macro_aulc(v) for k, v in remapped_records_by_variant.items()}
    gaps = {k: (None if (base is None or v is None) else float(base - v))
            for k, v in per_variant.items()}
    defined = [g for g in gaps.values() if g is not None]
    return {
        "canonical_macro_aulc": base,
        "remapped_macro_aulc": per_variant,
        "gap_vs_canonical": gaps,
        "macro_gap": (float(np.mean(defined)) if defined else None),
    }


# --------------------------------------------------------------------------
# 14: value-head ranking / calibration / regret
# --------------------------------------------------------------------------
def value_ranking_metrics(predictions: Sequence[float], targets: Sequence[float],
                          groups: Sequence[Any] | None = None) -> dict[str, Any]:
    """METRIC 14 — delegates to ``analysis.stats`` (numpy-only rank metrics)."""
    from foundation_learner.analysis.stats import value_head_metrics
    return value_head_metrics(predictions, targets, groups)


# --------------------------------------------------------------------------
# 15-17: Amendment 13 additions (format compliance, flip-immune AULC, flip rate)
#
# These are ADDITIONS, frozen before any run.  They change no frozen value and
# replace no metric: 1-14 keep their definitions, and every headline number is
# still reported.  They exist because three failure modes can make a metric 1
# number mean something other than "learned from feedback":
#   * the base model may never emit a parseable ANSWER line (metric 15);
#   * on binary-answer families "flip the label after INCORRECT" earns credit
#     at the REVISION indices without any rule inference (metrics 16, 17);
#   * FL2 is trained to imitate attempt0, so a delta against FL2 that includes
#     index 0 mixes an imitation handicap into the learning claim (metric 17's
#     index-restricted companion, wired in ``analysis/report.py``).
# --------------------------------------------------------------------------
def record_attempts(record: Mapping[str, Any],
                    indices: Sequence[int] | None = None
                    ) -> list[Mapping[str, Any]]:
    """MODEL_ATTEMPT rows of one episode record, optionally index-filtered."""
    out = []
    keep = None if indices is None else {int(k) for k in indices}
    for a in record.get("attempts") or []:
        if str(a.get("role", "MODEL_ATTEMPT")).upper() != "MODEL_ATTEMPT":
            continue
        idx = a.get("interaction_index")
        if keep is not None and (idx is None or int(idx) not in keep):
            continue
        out.append(a)
    return out


def attempt_has_answer_line(attempt: Mapping[str, Any]) -> bool:
    """True when the generation contained a parseable ``ANSWER:`` line.

    ``parsed`` is the payload of the LAST well-formed answer line as returned
    by ``episodes.parse.parse_answer`` (``None`` when there is none), which is
    exactly the strict grammar the verifier scores against, so no separate
    re-parse can disagree with the verifier.
    """
    if "parsed" not in attempt:
        raise KeyError(
            "attempt record has no 'parsed' field; answer_line_rate needs the "
            "walker's parse result (evaluation/learning_curve.py writes it)")
    return attempt.get("parsed") is not None


def record_answer_line_rate(record: Mapping[str, Any],
                            indices: Sequence[int] | None = None) -> float | None:
    """Fraction of THIS episode's attempts that emitted a parseable answer line."""
    attempts = record_attempts(record, indices)
    if not attempts:
        return None
    return float(sum(1 for a in attempts if attempt_has_answer_line(a))
                 / len(attempts))


def answer_line_rate(records: Sequence[Mapping[str, Any]],
                     indices: Sequence[int] | None = None) -> float | None:
    """METRIC 15 — macro answer-line rate (family means first, then families)."""
    return macro_mean(records, lambda r: record_answer_line_rate(r, indices))


def per_family_answer_line_rate(records: Sequence[Mapping[str, Any]],
                                indices: Sequence[int] | None = None
                                ) -> dict[str, float]:
    return per_family_mean(records, lambda r: record_answer_line_rate(r, indices))


def answer_line_rate_by_index(records: Sequence[Mapping[str, Any]]
                              ) -> dict[int, float]:
    """Macro answer-line rate at each interaction index."""
    indices = sorted({int(a["interaction_index"])
                      for r in records for a in record_attempts(r)
                      if a.get("interaction_index") is not None})
    out: dict[int, float] = {}
    for k in indices:
        value = answer_line_rate(records, [k])
        if value is not None:
            out[k] = value
    return out


def finish_reason_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Diagnostic breakdown of WHY each generation stopped.

    ``max_new_tokens`` dominating is the signature of a model whose answer never
    arrives inside the frozen 64-token decode budget.
    """
    counts: dict[str, int] = {}
    for r in records:
        for a in record_attempts(r):
            reason = str(a.get("finish_reason", "unrecorded"))
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def format_compliance(records: Sequence[Mapping[str, Any]],
                      threshold: float = ANSWER_LINE_RATE_FLAG_THRESHOLD
                      ) -> dict[str, Any]:
    """METRIC 15 bundle: macro / per-family / per-index rates plus the flags.

    The flag NEVER replaces or suppresses a number; it is reported next to it.
    """
    macro = answer_line_rate(records)
    per_family = per_family_answer_line_rate(records)
    per_index = answer_line_rate_by_index(records)
    return {
        "threshold": float(threshold),
        "macro_answer_line_rate": macro,
        "per_family_answer_line_rate": per_family,
        "answer_line_rate_by_index": {str(k): v for k, v in per_index.items()},
        "finish_reasons": finish_reason_counts(records),
        "flag": (FORMAT_NONCOMPLIANT_FLAG
                 if (macro is not None and macro < float(threshold)) else None),
        "flagged_families": sorted(
            fid for fid, v in per_family.items() if v < float(threshold)),
        "flagged_indices": sorted(
            int(k) for k, v in per_index.items() if v < float(threshold)),
    }


def format_flag(answer_line_rate_value: float | None,
                threshold: float = ANSWER_LINE_RATE_FLAG_THRESHOLD) -> str | None:
    """``FORMAT_NONCOMPLIANT`` when the rate is below the frozen threshold."""
    if answer_line_rate_value is None:
        return None
    return (FORMAT_NONCOMPLIANT_FLAG
            if float(answer_line_rate_value) < float(threshold) else None)


#: A family whose scored items carry more than this many distinct canonical
#: answers is an OPEN-answer family: "always answer X" is not a strategy there,
#: and its constant-answer floor is reported as undefined rather than as 0.
MAX_LABEL_ANSWERS = 4


def constant_answer_baseline(records: Sequence[Mapping[str, Any]],
                             max_label_answers: int = MAX_LABEL_ANSWERS
                             ) -> dict[str, Any]:
    """METRIC 18 — the AULC a CONSTANT-ANSWER policy would score, per family.

    On a label-answer family, "always answer the majority label" needs no
    reading, no rule inference and no feedback, so a family AULC below (or at)
    this floor is evidence of nothing.  The floor is computed with the metric's
    OWN aggregation — per candidate constant answer, the mean over interaction
    indices of the fraction of that index's attempts whose canonical answer is
    that constant, averaged over episodes — and the best constant is taken, so
    it is exactly comparable to :func:`per_family_aulc`.

    Open-answer families (more than ``max_label_answers`` distinct canonical
    answers) report ``None``: the number of distinct answers is reported instead
    so the omission is visible rather than silent.
    """
    per_family: dict[str, Any] = {}
    for fid, rows in sorted(group_by_family(records).items()):
        rows = [r for r in rows if record_is_scoreable(r)]
        answers: set[str] = set()
        for r in rows:
            for a in record_attempts(r):
                if a.get("canonical_answer") is not None:
                    answers.add(str(a["canonical_answer"]))
        if not answers:
            per_family[fid] = {"floor": None, "n_distinct_answers": 0,
                               "reason": "no canonical answers in the records"}
            continue
        if len(answers) > int(max_label_answers):
            per_family[fid] = {
                "floor": None, "n_distinct_answers": len(answers),
                "reason": "open-answer family; a constant answer is not a policy"}
            continue
        best_answer, best_score = None, None
        for constant in sorted(answers):
            episode_scores = []
            for r in rows:
                by_index: dict[int, list[float]] = {}
                for a in record_attempts(r):
                    idx = a.get("interaction_index")
                    if idx is None or a.get("canonical_answer") is None:
                        continue
                    by_index.setdefault(int(idx), []).append(
                        float(str(a["canonical_answer"]) == constant))
                if by_index:
                    episode_scores.append(
                        sum(sum(v) / len(v) for v in by_index.values())
                        / len(by_index))
            score = _mean(episode_scores)
            if score is not None and (best_score is None or score > best_score):
                best_answer, best_score = constant, score
        per_family[fid] = {
            "floor": best_score,
            "best_constant_answer": best_answer,
            "n_distinct_answers": len(answers),
            "answers": sorted(answers),
        }
    defined = [v["floor"] for v in per_family.values() if v["floor"] is not None]
    return {
        "max_label_answers": int(max_label_answers),
        "per_family": per_family,
        "macro_floor_label_families": (float(np.mean(defined)) if defined
                                       else None),
        "n_families_defined": len(defined),
        "n_families": len(per_family),
    }


def aulc_above_constant_answer_floor(records: Sequence[Mapping[str, Any]]
                                     ) -> dict[str, Any]:
    """Per-family macro-AULC MINUS its constant-answer floor (``None`` where the
    floor is undefined).  Pure reporting: no threshold consumes it."""
    aulc = per_family_aulc(records)
    baseline = constant_answer_baseline(records)["per_family"]
    out: dict[str, Any] = {}
    for fid, value in aulc.items():
        floor = (baseline.get(fid) or {}).get("floor")
        out[fid] = {
            "macro_aulc": value,
            "constant_answer_floor": floor,
            "above_floor": (None if floor is None else float(value - floor)),
        }
    return out


def aulc_post_feedback_fresh(records: Sequence[Mapping[str, Any]]) -> float | None:
    """METRIC 16 — macro-AULC restricted to the FRESH indices {4, 5, 6}.

    An answer-flip heuristic cannot earn credit here: every attempt at these
    indices is on an item the model has had no feedback about.
    """
    return macro_aulc(records, POST_FEEDBACK_FRESH_INDICES)


def per_family_aulc_post_feedback_fresh(records: Sequence[Mapping[str, Any]]
                                        ) -> dict[str, float]:
    return per_family_aulc(records, POST_FEEDBACK_FRESH_INDICES)


def _previous_attempt(attempts: Sequence[Mapping[str, Any]],
                      attempt: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The immediately preceding attempt on the SAME item, if any."""
    item = attempt.get("item_id")
    number = attempt.get("attempt_number")
    if item is None or number is None:
        return None
    for other in attempts:
        if (other.get("item_id") == item
                and other.get("attempt_number") == int(number) - 1):
            return other
    return None


def flip_attributable_success_rate(
        records: Sequence[Mapping[str, Any]],
        binary_family_ids: Sequence[str] | None = None,
        indices: Sequence[int] = REVISION_INDICES) -> dict[str, Any]:
    """METRIC 17 — how much revision success is "just flip the other label".

    Over BINARY-answer families only, of the successful attempts at the revision
    indices (1, 3), the fraction that (a) followed an INCORRECT verdict on the
    same item and (b) changed the emitted label.  On a two-label family that
    combination is exactly the deterministic "answer the other one" heuristic;
    it is the ONLY thing needed to succeed there, so a high rate means the
    revision credit carries no evidence of rule inference.

    Computed from the transcripts already in the records (per-attempt
    ``item_id``, ``attempt_number``, ``parsed``, ``correct``); no re-simulation.
    """
    families = tuple(BINARY_ANSWER_FAMILIES if binary_family_ids is None
                     else binary_family_ids)
    per_family: dict[str, Any] = {}
    for fid in families:
        rows = [r for r in records if record_family(r) == fid]
        if not rows:
            continue
        n_success = 0
        n_flip = 0
        n_attempts = 0
        for record in rows:
            attempts = list(record.get("attempts") or [])
            for a in record_attempts(record, indices):
                n_attempts += 1
                if not bool(a.get("correct")):
                    continue
                n_success += 1
                prev = _previous_attempt(attempts, a)
                if prev is None:
                    continue
                changed = a.get("parsed") != prev.get("parsed")
                if (not bool(prev.get("correct"))) and changed:
                    n_flip += 1
        per_family[fid] = {
            "n_episodes": len(rows),
            "n_revision_attempts": n_attempts,
            "n_successes": n_success,
            "n_flip_attributable": n_flip,
            "rate": (float(n_flip / n_success) if n_success else None),
        }
    rates = [v["rate"] for v in per_family.values() if v["rate"] is not None]
    total_success = sum(v["n_successes"] for v in per_family.values())
    total_flip = sum(v["n_flip_attributable"] for v in per_family.values())
    return {
        "indices": [int(k) for k in indices],
        "binary_families": list(families),
        "per_family": per_family,
        "macro_rate": (float(np.mean(rates)) if rates else None),
        "pooled_rate": (float(total_flip / total_success)
                        if total_success else None),
        "n_successes": total_success,
        "n_flip_attributable": total_flip,
        "n_families_present": len(per_family),
    }


# --------------------------------------------------------------------------
# frozen registry
# --------------------------------------------------------------------------
METRICS_V0: tuple[dict[str, str], ...] = (
    {"id": "1", "name": "macro_aulc", "fn": "macro_aulc"},
    {"id": "2", "name": "delta_aulc_vs_fl1", "fn": "delta_aulc"},
    {"id": "3", "name": "delta_aulc_vs_fl2", "fn": "delta_aulc"},
    {"id": "4", "name": "r_0", "fn": "r_0"},
    {"id": "5", "name": "r_K", "fn": "r_K"},
    {"id": "6", "name": "improvement_slope", "fn": "improvement_slope"},
    {"id": "7", "name": "interactions_to_threshold", "fn": "interactions_to_threshold"},
    {"id": "8", "name": "related_task_transfer", "fn": "related_task_transfer"},
    {"id": "9", "name": "whole_family_transfer", "fn": "whole_family_transfer"},
    {"id": "10", "name": "context_reset_persistence", "fn": "context_reset_persistence"},
    {"id": "11", "name": "retention_interference", "fn": "retention_interference"},
    {"id": "12", "name": "poison_robustness", "fn": "poison_robustness"},
    {"id": "13", "name": "remap_robustness", "fn": "remap_robustness"},
    {"id": "14", "name": "value_ranking_metrics", "fn": "value_ranking_metrics"},
    # Amendment 13 additions (frozen before any run; nothing above changes)
    {"id": "15", "name": "answer_line_rate", "fn": "answer_line_rate"},
    {"id": "16", "name": "aulc_post_feedback_fresh",
     "fn": "aulc_post_feedback_fresh"},
    {"id": "17", "name": "flip_attributable_success_rate",
     "fn": "flip_attributable_success_rate"},
    # Amendment 15 addition
    {"id": "18", "name": "constant_answer_baseline",
     "fn": "constant_answer_baseline"},
)


def metric_function(name: str) -> Callable[..., Any]:
    for entry in METRICS_V0:
        if entry["name"] == name:
            return globals()[entry["fn"]]
    raise KeyError(f"unknown frozen metric {name!r}")


def summarize(records: Sequence[Mapping[str, Any]],
              trained_family_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Convenience bundle of the record-only metrics (1, 4, 5, 6, 7, 8, 9)."""
    out: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "n_episodes": len(records),
        "n_families": len(group_by_family(records)),
        "excluded_records": excluded_records(records),
        "macro_aulc": macro_aulc(records),
        "answer_line_rate": answer_line_rate(records),
        "format_compliance": format_compliance(records),
        "format_flag": format_flag(answer_line_rate(records)),
        "aulc_post_feedback_fresh": aulc_post_feedback_fresh(records),
        "per_family_aulc_post_feedback_fresh":
            per_family_aulc_post_feedback_fresh(records),
        "flip_attributable_success_rate": flip_attributable_success_rate(records),
        "constant_answer_baseline": constant_answer_baseline(records),
        "aulc_above_constant_answer_floor":
            aulc_above_constant_answer_floor(records),
        "per_family_aulc": per_family_aulc(records),
        "macro_curve": {str(k): v for k, v in macro_curve(records).items()},
        "r_0": r_0(records),
        "r_K": r_K(records),
        "improvement_slope": improvement_slope(records),
        "per_family_slopes": per_family_slopes(records),
        "interactions_to_threshold": interactions_to_threshold(records),
        "related_task_transfer": related_task_transfer(records),
    }
    if trained_family_ids is not None:
        out["whole_family_transfer"] = whole_family_transfer(records, trained_family_ids)
    return out


__all__ = [
    "METRICS_SCHEMA",
    "METRICS_V0",
    "DEFAULT_THRESHOLD",
    "DEFAULT_INDICES",
    "POST_FEEDBACK_FRESH_INDICES",
    "REVISION_INDICES",
    "INDICES_EXCLUDING_ZERO",
    "BINARY_ANSWER_FAMILIES",
    "ANSWER_LINE_RATE_FLAG_THRESHOLD",
    "FORMAT_NONCOMPLIANT_FLAG",
    "STATUS_COMPLETE",
    "STATUS_ONLINE_BUDGET_EXCEEDED",
    "record_status",
    "record_is_scoreable",
    "excluded_records",
    "record_attempts",
    "attempt_has_answer_line",
    "record_answer_line_rate",
    "answer_line_rate",
    "per_family_answer_line_rate",
    "answer_line_rate_by_index",
    "finish_reason_counts",
    "format_compliance",
    "format_flag",
    "aulc_post_feedback_fresh",
    "per_family_aulc_post_feedback_fresh",
    "flip_attributable_success_rate",
    "MAX_LABEL_ANSWERS",
    "constant_answer_baseline",
    "aulc_above_constant_answer_floor",
    "CLEAN_CONDITION",
    "POISON_CONDITIONS",
    "canonical_poison_condition",
    "record_family",
    "record_R",
    "record_aulc",
    "group_by_family",
    "macro_mean",
    "per_family_mean",
    "macro_aulc",
    "per_family_aulc",
    "delta_aulc",
    "per_family_curve",
    "macro_curve",
    "r_at",
    "r_0",
    "r_K",
    "ols_slope",
    "ols_fit",
    "per_family_slopes",
    "improvement_slope",
    "slope_conventions_agree",
    "interactions_to_threshold",
    "related_task_transfer",
    "whole_family_transfer",
    "context_reset_persistence",
    "retention_interference",
    "poison_robustness",
    "remap_robustness",
    "value_ranking_metrics",
    "metric_function",
    "summarize",
]
