"""Clustered bootstrap and rank statistics (contract section 14). numpy only.

Clustered bootstrap
-------------------
Episodes within a family share a generator, and (in this design) sometimes a
latent rule family's difficulty profile; treating them as independent would
understate uncertainty.  The frozen procedure is therefore:

1. resample FAMILIES with replacement (``F`` draws from ``F`` families);
2. for each drawn family, resample ITS episodes with replacement
   (``n_f`` draws from that family's ``n_f`` episodes);
3. the replicate statistic is the macro mean: mean over the ``F`` drawn family
   positions of the mean over that position's drawn episodes;
4. 10,000 replicates by default; percentile interval.

PAIRED comparisons (arm A vs arm B) use the SAME resample indices: the index
matrices are drawn ONCE per comparison (:class:`ResamplePlan`) and applied to
both arms, so the bootstrap distribution is over the difference and the
comparison is not inflated by independent resampling noise.  A plan is a
concrete array, not a re-seeded stream, which is what makes "identical indices
across arms" checkable rather than merely intended.

Everything is deterministic given ``seed``; no global RNG, no wall clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

BOOTSTRAP_SCHEMA = "flb200.clustered_bootstrap.v1"
VALUE_METRICS_SCHEMA = "flb200.value_head_metrics.v1"
ARM_COMPARISON_SCHEMA = "flb200.arm_comparison.v1"

DEFAULT_REPLICATES = 10_000
DEFAULT_ALPHA = 0.05
#: refuse to materialize an index plan bigger than this many entries (~200 MB
#: as float32); the caller lowers ``n_replicates`` instead of silently
#: switching to a different (unverifiable) resampling scheme.
MAX_PLAN_ENTRIES = 50_000_000


class ClusteredSampleError(ValueError):
    """Malformed or mismatched clustered sample."""


# --------------------------------------------------------------------------
# clustered sample
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusteredSample:
    """Per-episode values grouped by family, with stable episode identity."""

    family_ids: tuple[str, ...]
    values: tuple[np.ndarray, ...]
    episode_ids: tuple[tuple[str, ...], ...]

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(int(v.size) for v in self.values)

    @property
    def n_episodes(self) -> int:
        return int(sum(self.sizes))

    def family_means(self) -> dict[str, float]:
        return {fid: float(v.mean()) for fid, v in zip(self.family_ids, self.values)
                if v.size}

    def macro_mean(self) -> float | None:
        means = [float(v.mean()) for v in self.values if v.size]
        return float(np.mean(means)) if means else None

    def pooled_values(self) -> np.ndarray:
        return np.concatenate(self.values) if self.values else np.zeros(0)


def clustered_sample_from_records(
    records: Sequence[Mapping[str, Any]],
    value_fn: Callable[[Mapping[str, Any]], float | None] | None = None,
) -> ClusteredSample:
    """Group episode records into a :class:`ClusteredSample` (families sorted)."""
    from foundation_learner.evaluation import metrics as _metrics

    value_fn = value_fn or _metrics.record_aulc
    grouped: dict[str, list[tuple[str, float]]] = {}
    for rec in records:
        value = value_fn(rec)
        if value is None:
            continue
        grouped.setdefault(_metrics.record_family(rec), []).append(
            (str(rec.get("episode_id", "")), float(value)))
    families = tuple(sorted(grouped))
    values = tuple(np.asarray([v for _eid, v in grouped[f]], dtype=np.float64)
                   for f in families)
    ids = tuple(tuple(eid for eid, _v in grouped[f]) for f in families)
    return ClusteredSample(families, values, ids)


# --------------------------------------------------------------------------
# resample plan
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ResamplePlan:
    """Materialized resample indices, drawn ONCE and reused across arms."""

    n_replicates: int
    seed: int
    family_ids: tuple[str, ...]
    sizes: tuple[int, ...]
    family_index: np.ndarray   # int32 [reps, F]
    unit_uniform: np.ndarray   # float32 [reps, F, max_n]

    def digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.family_index).tobytes())
        h.update(np.ascontiguousarray(self.unit_uniform).tobytes())
        return h.hexdigest()

    def check_compatible(self, sample: ClusteredSample) -> None:
        if tuple(sample.family_ids) != tuple(self.family_ids):
            raise ClusteredSampleError(
                f"plan families {self.family_ids} != sample families {sample.family_ids}")
        if tuple(sample.sizes) != tuple(self.sizes):
            raise ClusteredSampleError(
                f"plan sizes {self.sizes} != sample sizes {sample.sizes}")

    def macro_replicates(self, sample: ClusteredSample) -> np.ndarray:
        """Bootstrap distribution of the macro mean for ``sample``."""
        self.check_compatible(sample)
        reps, F = self.family_index.shape
        position_means = np.zeros((reps, F), dtype=np.float64)
        for f in range(F):
            mask = self.family_index == f
            if not mask.any():
                continue
            n = int(self.sizes[f])
            u = self.unit_uniform[mask][:, :n]
            idx = np.minimum((u.astype(np.float64) * n).astype(np.int64), n - 1)
            position_means[mask] = sample.values[f][idx].mean(axis=1)
        return position_means.mean(axis=1)


def make_resample_plan(sample: ClusteredSample, *, n_replicates: int = DEFAULT_REPLICATES,
                       seed: int = 0,
                       max_plan_entries: int = MAX_PLAN_ENTRIES) -> ResamplePlan:
    F = len(sample.family_ids)
    if F == 0:
        raise ClusteredSampleError("cannot bootstrap an empty sample")
    if any(n == 0 for n in sample.sizes):
        raise ClusteredSampleError("every family must contribute at least one episode")
    max_n = max(sample.sizes)
    entries = int(n_replicates) * F * int(max_n)
    if entries > int(max_plan_entries):
        raise ClusteredSampleError(
            f"resample plan would need {entries} entries (> {max_plan_entries}); "
            "reduce n_replicates rather than changing the resampling scheme")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    family_index = rng.integers(0, F, size=(int(n_replicates), F), dtype=np.int64
                                ).astype(np.int32)
    unit_uniform = rng.random(size=(int(n_replicates), F, int(max_n)),
                              dtype=np.float32)
    return ResamplePlan(int(n_replicates), int(seed), tuple(sample.family_ids),
                        tuple(sample.sizes), family_index, unit_uniform)


def percentile_ci(draws: np.ndarray, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    lo = float(np.percentile(draws, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def clustered_bootstrap_ci(sample: ClusteredSample, *,
                           n_replicates: int = DEFAULT_REPLICATES,
                           seed: int = 0, alpha: float = DEFAULT_ALPHA,
                           plan: ResamplePlan | None = None) -> dict[str, Any]:
    """Macro mean with a family-clustered percentile interval."""
    plan = plan or make_resample_plan(sample, n_replicates=n_replicates, seed=seed)
    draws = plan.macro_replicates(sample)
    lo, hi = percentile_ci(draws, alpha)
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "estimate": sample.macro_mean(),
        "ci_low": lo,
        "ci_high": hi,
        "alpha": float(alpha),
        "se": float(draws.std(ddof=1)) if draws.size > 1 else None,
        "n_families": len(sample.family_ids),
        "n_episodes": sample.n_episodes,
        "n_replicates": int(plan.n_replicates),
        "seed": int(plan.seed),
        "plan_digest": plan.digest(),
        "family_means": sample.family_means(),
        "clustered": True,
    }


def naive_bootstrap_ci(sample: ClusteredSample, *,
                       n_replicates: int = DEFAULT_REPLICATES,
                       seed: int = 0, alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """DIAGNOSTIC comparator only: i.i.d. episode resampling, family structure ignored.

    Reported alongside the clustered interval to show how much the clustering
    matters; it is never the interval a result is claimed with.
    """
    pooled = sample.pooled_values()
    if pooled.size == 0:
        raise ClusteredSampleError("cannot bootstrap an empty sample")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    idx = rng.integers(0, pooled.size, size=(int(n_replicates), pooled.size))
    draws = pooled[idx].mean(axis=1)
    lo, hi = percentile_ci(draws, alpha)
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "estimate": float(pooled.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "alpha": float(alpha),
        "se": float(draws.std(ddof=1)) if draws.size > 1 else None,
        "n_episodes": int(pooled.size),
        "n_replicates": int(n_replicates),
        "seed": int(seed),
        "clustered": False,
    }


def paired_clustered_bootstrap(sample_a: ClusteredSample, sample_b: ClusteredSample, *,
                               n_replicates: int = DEFAULT_REPLICATES,
                               seed: int = 0, alpha: float = DEFAULT_ALPHA,
                               plan: ResamplePlan | None = None,
                               require_matched_episodes: bool = True) -> dict[str, Any]:
    """Paired macro difference A - B using ONE shared resample plan."""
    if tuple(sample_a.family_ids) != tuple(sample_b.family_ids):
        raise ClusteredSampleError("paired arms must cover the same families")
    if tuple(sample_a.sizes) != tuple(sample_b.sizes):
        raise ClusteredSampleError("paired arms must have identical family sizes")
    if require_matched_episodes and sample_a.episode_ids != sample_b.episode_ids:
        raise ClusteredSampleError(
            "paired arms must present episodes in the same order (matched episode ids); "
            "pass require_matched_episodes=False only for deliberately unmatched arms")
    plan = plan or make_resample_plan(sample_a, n_replicates=n_replicates, seed=seed)
    draws = plan.macro_replicates(sample_a) - plan.macro_replicates(sample_b)
    lo, hi = percentile_ci(draws, alpha)
    est_a, est_b = sample_a.macro_mean(), sample_b.macro_mean()
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "paired": True,
        "estimate": (None if (est_a is None or est_b is None) else float(est_a - est_b)),
        "estimate_a": est_a,
        "estimate_b": est_b,
        "ci_low": lo,
        "ci_high": hi,
        "alpha": float(alpha),
        "se": float(draws.std(ddof=1)) if draws.size > 1 else None,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "n_families": len(sample_a.family_ids),
        "n_episodes": sample_a.n_episodes,
        "n_replicates": int(plan.n_replicates),
        "seed": int(plan.seed),
        "plan_digest": plan.digest(),
        "clustered": True,
    }


# --------------------------------------------------------------------------
# curves, retention, robustness
# --------------------------------------------------------------------------
def learning_curve_with_ci(records: Sequence[Mapping[str, Any]], *,
                           n_replicates: int = DEFAULT_REPLICATES,
                           seed: int = 0, alpha: float = DEFAULT_ALPHA
                           ) -> dict[str, Any]:
    """Macro ``R_k`` with a clustered interval at every interaction index."""
    from foundation_learner.evaluation import metrics as _metrics

    indices = sorted({k for r in records for k in _metrics.record_R(r)})
    out: dict[str, Any] = {"indices": indices, "points": {}}
    for k in indices:
        sample = clustered_sample_from_records(
            records, value_fn=lambda r, _k=k: _metrics.record_R(r).get(_k))
        if not sample.family_ids:
            continue
        out["points"][str(k)] = clustered_bootstrap_ci(
            sample, n_replicates=n_replicates, seed=seed + int(k), alpha=alpha)
    return out


def arm_comparison(records_a: Sequence[Mapping[str, Any]],
                   records_b: Sequence[Mapping[str, Any]], *,
                   label_a: str = "A", label_b: str = "B",
                   n_replicates: int = DEFAULT_REPLICATES,
                   seed: int = 0, alpha: float = DEFAULT_ALPHA,
                   require_matched_episodes: bool = True,
                   indices: Sequence[int] | None = None) -> dict[str, Any]:
    """Paired macro-AULC difference between two arms with a clustered CI.

    ``indices`` restricts the per-episode AULC to those interaction indices, so
    the SAME paired machinery produces the Amendment 13 companions of the
    headline delta ({4, 5, 6} only; everything except index 0).  ``None`` is the
    frozen full-curve delta of metrics 2/3.
    """
    from foundation_learner.evaluation import metrics as _metrics

    value_fn = (None if indices is None
                else (lambda r, _i=tuple(indices): _metrics.record_aulc(r, _i)))
    sample_a = clustered_sample_from_records(records_a, value_fn)
    sample_b = clustered_sample_from_records(records_b, value_fn)
    result = paired_clustered_bootstrap(
        sample_a, sample_b, n_replicates=n_replicates, seed=seed, alpha=alpha,
        require_matched_episodes=require_matched_episodes)
    result.update({"schema": ARM_COMPARISON_SCHEMA,
                   "label_a": label_a, "label_b": label_b,
                   "indices": (None if indices is None
                               else [int(k) for k in indices])})
    return result


def retention_summary(chain_summaries: Sequence[Mapping[str, Any]], *,
                      n_replicates: int = DEFAULT_REPLICATES,
                      seed: int = 0, alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """Family-clustered interval for the A->B->A retention ratio."""
    grouped: dict[str, list[tuple[str, float]]] = {}
    for c in chain_summaries:
        ratio = c.get("retention_ratio")
        if ratio is None:
            continue
        grouped.setdefault(str(c.get("family_a")), []).append(
            (str(c.get("chain_id")), float(ratio)))
    families = tuple(sorted(grouped))
    if not families:
        return {"schema": BOOTSTRAP_SCHEMA, "estimate": None, "n_families": 0}
    sample = ClusteredSample(
        families,
        tuple(np.asarray([v for _i, v in grouped[f]], dtype=np.float64) for f in families),
        tuple(tuple(i for i, _v in grouped[f]) for f in families))
    return clustered_bootstrap_ci(sample, n_replicates=n_replicates, seed=seed, alpha=alpha)


def robustness_summary(records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
                       reference: str, *,
                       n_replicates: int = DEFAULT_REPLICATES,
                       seed: int = 0, alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """Paired macro-AULC gaps of each condition against ``reference``.

    Used for both poison (reference = the clean condition) and remap
    (reference = the canonical surface).  Pairing requires the conditions to
    cover the same families with the same episode counts; where they do not,
    the unpaired clustered interval of each arm is reported instead and the
    pairing status is recorded honestly.
    """
    ref_records = records_by_condition.get(reference)
    if ref_records is None:
        raise ClusteredSampleError(f"reference condition {reference!r} absent")
    ref_sample = clustered_sample_from_records(ref_records)
    out: dict[str, Any] = {
        "reference": reference,
        "reference_ci": clustered_bootstrap_ci(
            ref_sample, n_replicates=n_replicates, seed=seed, alpha=alpha),
        "conditions": {},
    }
    for cond, recs in sorted(records_by_condition.items()):
        if cond == reference:
            continue
        sample = clustered_sample_from_records(recs)
        entry: dict[str, Any] = {"ci": clustered_bootstrap_ci(
            sample, n_replicates=n_replicates, seed=seed, alpha=alpha)}
        if (sample.family_ids == ref_sample.family_ids
                and sample.sizes == ref_sample.sizes):
            entry["paired_gap"] = paired_clustered_bootstrap(
                ref_sample, sample, n_replicates=n_replicates, seed=seed,
                alpha=alpha, require_matched_episodes=False)
            entry["paired"] = True
        else:
            entry["paired"] = False
            entry["paired_reason"] = "condition arms do not share family sizes"
        out["conditions"][cond] = entry
    return out


# --------------------------------------------------------------------------
# value-head rank metrics (numpy only; no scipy anywhere)
# --------------------------------------------------------------------------
def rankdata_average(values: Sequence[float]) -> np.ndarray:
    """Ranks 1..n with ties receiving their average rank (scipy-free)."""
    a = np.asarray(list(values), dtype=np.float64)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    if a.size < 2 or a.size != b.size:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 0.0:
        return None
    return float((a * b).sum() / denom)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation implemented on numpy ranks (no scipy)."""
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(rankdata_average(x), rankdata_average(y))


def pairwise_ranking_accuracy(predictions: Sequence[float], targets: Sequence[float],
                              groups: Sequence[Any] | None = None) -> dict[str, Any]:
    """Fraction of comparable pairs ordered correctly (ties in prediction = 0.5).

    With ``groups`` (episode ids), only WITHIN-group pairs are compared, which
    is the quantity contract section 7 FL4 trains and section 10 gates on.
    """
    pred = np.asarray(list(predictions), dtype=np.float64)
    targ = np.asarray(list(targets), dtype=np.float64)
    if pred.size != targ.size:
        raise ValueError("predictions and targets must align")
    keys = list(groups) if groups is not None else [0] * pred.size
    buckets: dict[Any, list[int]] = {}
    for i, g in enumerate(keys):
        buckets.setdefault(g, []).append(i)
    score = 0.0
    n_pairs = 0
    for idx in buckets.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if targ[i] == targ[j]:
                    continue
                n_pairs += 1
                dp = pred[i] - pred[j]
                dt = targ[i] - targ[j]
                if dp == 0.0:
                    score += 0.5
                elif (dp > 0) == (dt > 0):
                    score += 1.0
    return {
        "accuracy": (float(score / n_pairs) if n_pairs else None),
        "n_pairs": n_pairs,
        "n_groups": len(buckets),
    }


def calibration_fit(predictions: Sequence[float], targets: Sequence[float]
                    ) -> dict[str, Any]:
    """OLS of target on prediction: slope 1 / intercept 0 = calibrated."""
    from foundation_learner.evaluation.metrics import ols_fit

    fit = ols_fit(predictions, targets)
    if fit is None:
        return {"slope": None, "intercept": None, "r": pearson(predictions, targets)}
    slope, intercept = fit
    return {"slope": slope, "intercept": intercept,
            "r": pearson(predictions, targets)}


def top1_regret(predictions: Sequence[float], targets: Sequence[float],
                groups: Sequence[Any] | None = None) -> dict[str, Any]:
    """Mean per-group regret of selecting the argmax-prediction item.

    Also reports the random-selection baseline (mean target) and the oracle
    (zero by construction), which is how contract section 7 defines the FL4
    DEV comparison.
    """
    pred = np.asarray(list(predictions), dtype=np.float64)
    targ = np.asarray(list(targets), dtype=np.float64)
    keys = list(groups) if groups is not None else [0] * pred.size
    buckets: dict[Any, list[int]] = {}
    for i, g in enumerate(keys):
        buckets.setdefault(g, []).append(i)
    regrets: list[float] = []
    random_regrets: list[float] = []
    for idx in buckets.values():
        if len(idx) < 2:
            continue
        t = targ[idx]
        p = pred[idx]
        best = float(t.max())
        chosen = int(np.argmax(p))
        regrets.append(best - float(t[chosen]))
        random_regrets.append(best - float(t.mean()))
    return {
        "top1_regret": (float(np.mean(regrets)) if regrets else None),
        "top1_regret_random": (float(np.mean(random_regrets)) if random_regrets else None),
        "top1_regret_oracle": 0.0 if regrets else None,
        "n_groups": len(regrets),
    }


def value_head_metrics(predictions: Sequence[float], targets: Sequence[float],
                       groups: Sequence[Any] | None = None) -> dict[str, Any]:
    """METRIC 14 — Spearman, pairwise ranking accuracy, calibration, top-1 regret."""
    out: dict[str, Any] = {
        "schema": VALUE_METRICS_SCHEMA,
        "n": len(list(predictions)),
        "spearman": spearman(predictions, targets),
        "pearson": pearson(predictions, targets),
        "calibration": calibration_fit(predictions, targets),
    }
    out.update({"pairwise": pairwise_ranking_accuracy(predictions, targets, groups)})
    out.update({"regret": top1_regret(predictions, targets, groups)})
    return out


# --------------------------------------------------------------------------
# convenience
# --------------------------------------------------------------------------
def summarize_arm(records: Sequence[Mapping[str, Any]], *, label: str = "arm",
                  n_replicates: int = DEFAULT_REPLICATES, seed: int = 0,
                  alpha: float = DEFAULT_ALPHA,
                  trained_family_ids: Iterable[str] | None = None) -> dict[str, Any]:
    from foundation_learner.evaluation import metrics as _metrics

    sample = clustered_sample_from_records(records)
    fresh_sample = clustered_sample_from_records(
        records,
        lambda r: _metrics.record_aulc(r, _metrics.POST_FEEDBACK_FRESH_INDICES))
    metrics = _metrics.summarize(records, trained_family_ids)
    out = {
        "label": label,
        "metrics": metrics,
        "macro_aulc_ci": (clustered_bootstrap_ci(
            sample, n_replicates=n_replicates, seed=seed, alpha=alpha)
            if sample.family_ids else None),
        # Amendment 13: the flip-immune companion of the headline AULC, with
        # the same clustered interval, reported ALONGSIDE it.
        "aulc_post_feedback_fresh_ci": (clustered_bootstrap_ci(
            fresh_sample, n_replicates=n_replicates, seed=seed, alpha=alpha)
            if fresh_sample.family_ids else None),
        "answer_line_rate": metrics.get("answer_line_rate"),
        "format_flag": metrics.get("format_flag"),
        "excluded_records": metrics.get("excluded_records"),
        "learning_curve": learning_curve_with_ci(
            records, n_replicates=n_replicates, seed=seed, alpha=alpha),
    }
    return out


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "VALUE_METRICS_SCHEMA",
    "ARM_COMPARISON_SCHEMA",
    "DEFAULT_REPLICATES",
    "DEFAULT_ALPHA",
    "MAX_PLAN_ENTRIES",
    "ClusteredSampleError",
    "ClusteredSample",
    "ResamplePlan",
    "clustered_sample_from_records",
    "make_resample_plan",
    "percentile_ci",
    "clustered_bootstrap_ci",
    "naive_bootstrap_ci",
    "paired_clustered_bootstrap",
    "learning_curve_with_ci",
    "arm_comparison",
    "retention_summary",
    "robustness_summary",
    "rankdata_average",
    "pearson",
    "spearman",
    "pairwise_ranking_accuracy",
    "calibration_fit",
    "top1_regret",
    "value_head_metrics",
    "summarize_arm",
]
