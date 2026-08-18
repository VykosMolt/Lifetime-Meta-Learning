"""A -> B -> A interference chains (contract sections 9 and 16).

A chain is three ONLINE episodes run in order with ONE persistent mechanism
(one callbacks object shared across the three episodes of that chain, so a
mechanism can carry whatever it is designed to carry; per-episode resetting is
the mechanism's own decision inside ``on_episode_start``):

    A1 : learn family A
    B  : learn family B (the interfering task)
    A2 : return to family A

Reported per chain and then FAMILY-BALANCED (aggregated per family A, then
across families with equal weight):

* retention ratio      ``AULC(A2) / AULC(A1)``
* interference cost     ``AULC(A1) - AULC(A2)``
* recovery interactions the first interaction index of A2 whose success
                        reaches A1's terminal success level, or ``None``

Chains are pre-generated (contract section 16: 200 chains per ordered DEV
family pair); this module consumes chain specs, never invents them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .learning_curve import (
    LearningCurveConfig,
    annotate_records,
    event_attr,
    run_episodes,
)
from . import metrics as _metrics

INTERFERENCE_REPORT_SCHEMA = "flb200.interference_report.v1"
CHAIN_SPEC_SCHEMA = "flb200.interference_chain.v1"

CHAIN_POSITIONS: tuple[str, ...] = ("a1", "b", "a2")


class ChainSpecError(ValueError):
    """A chain spec is malformed or unbalanced."""


@dataclass(frozen=True)
class ChainSpec:
    chain_id: str
    family_a: str
    family_b: str
    episode_a1: Any
    episode_b: Any
    episode_a2: Any

    def episodes(self) -> dict[str, Any]:
        return {"a1": self.episode_a1, "b": self.episode_b, "a2": self.episode_a2}


def chain_spec_from_dict(spec: Mapping[str, Any],
                         episode_resolver: Callable[[str], Any] | None = None) -> ChainSpec:
    """Build a :class:`ChainSpec` from a pre-generated dict.

    Episodes may be embedded objects (``episode_a1``) or ids
    (``episode_a1_id``) resolved through ``episode_resolver``.
    """
    def _resolve(key: str) -> Any:
        if key in spec and spec[key] is not None:
            return spec[key]
        id_key = f"{key}_id"
        if id_key in spec and episode_resolver is not None:
            return episode_resolver(str(spec[id_key]))
        raise ChainSpecError(f"chain spec lacks {key!r} (and no resolvable {id_key!r})")

    ep_a1 = _resolve("episode_a1")
    ep_b = _resolve("episode_b")
    ep_a2 = _resolve("episode_a2")
    family_a = str(spec.get("family_a") or event_attr(ep_a1, "family_id", ""))
    family_b = str(spec.get("family_b") or event_attr(ep_b, "family_id", ""))
    if not family_a or not family_b:
        raise ChainSpecError("chain spec has no resolvable family ids")
    if str(event_attr(ep_a1, "family_id", family_a)) != family_a:
        raise ChainSpecError("episode_a1 family does not match family_a")
    if str(event_attr(ep_a2, "family_id", family_a)) != family_a:
        raise ChainSpecError("episode_a2 family does not match family_a")
    if str(event_attr(ep_b, "family_id", family_b)) != family_b:
        raise ChainSpecError("episode_b family does not match family_b")
    if family_a == family_b:
        raise ChainSpecError("A -> B -> A requires two DIFFERENT families")
    return ChainSpec(
        chain_id=str(spec.get("chain_id") or f"{family_a}->{family_b}"),
        family_a=family_a, family_b=family_b,
        episode_a1=ep_a1, episode_b=ep_b, episode_a2=ep_a2)


def check_pair_balance(chains: Sequence[ChainSpec], *, require_balanced: bool = True
                       ) -> dict[str, int]:
    """Count chains per ordered family pair; optionally require equal counts."""
    counts: dict[str, int] = {}
    for c in chains:
        counts[f"{c.family_a}->{c.family_b}"] = counts.get(f"{c.family_a}->{c.family_b}", 0) + 1
    if require_balanced and counts and len(set(counts.values())) != 1:
        raise ChainSpecError(
            f"family-balanced reporting requires equal chains per ordered pair; got {counts}")
    return counts


def recovery_interactions(a1_record: Mapping[str, Any],
                          a2_record: Mapping[str, Any]) -> int | None:
    """First A2 index whose success reaches A1's terminal success level."""
    r1 = _metrics.record_R(a1_record)
    r2 = _metrics.record_R(a2_record)
    if not r1 or not r2:
        return None
    target = r1[max(r1)]
    for k in sorted(r2):
        if r2[k] >= target:
            return int(k)
    return None


def run_interference_eval(
    bundle: Any,
    chains: Sequence[ChainSpec],
    env_factory: Callable[[Any], Any],
    *,
    cfg: LearningCurveConfig | None = None,
    callbacks_factory: Callable[[ChainSpec], Any] | None = None,
    render_module: Any = None,
    parser: Callable[[str], str | None] | None = None,
    require_balanced: bool = True,
) -> dict[str, Any]:
    """Run every chain; A1 for all chains, then B for all, then A2 for all.

    Running position-by-position keeps the ORDER within each chain exact while
    still batching across chains.  Each chain owns ONE callbacks object, so
    mechanism state follows its own chain and never crosses chains.
    """
    cfg = cfg or LearningCurveConfig()
    pair_counts = check_pair_balance(chains, require_balanced=require_balanced)
    if len({c.chain_id for c in chains}) != len(chains):
        raise ChainSpecError("chain ids must be unique within one evaluation")
    chain_callbacks = [callbacks_factory(c) if callbacks_factory else None
                       for c in chains]
    live = [cb for cb in chain_callbacks if cb is not None]
    if len({id(cb) for cb in live}) != len(live):
        raise ChainSpecError(
            "callbacks_factory must return a DISTINCT object per chain; a shared "
            "object would carry adaptation state across chains")

    records: dict[str, list[dict[str, Any]]] = {}
    for position in CHAIN_POSITIONS:
        episodes = [c.episodes()[position] for c in chains]
        if len({id(ep) for ep in episodes}) != len(episodes):
            raise ChainSpecError(
                f"two chains share the same {position} episode object")
        cbs = {id(ep): cb for ep, cb in zip(episodes, chain_callbacks)}
        recs = run_episodes(
            bundle, episodes, env_factory, cfg=cfg,
            callbacks_factory=((lambda ep: cbs[id(ep)]) if callbacks_factory else None),
            render_module=render_module, parser=parser)
        records[position] = annotate_records(recs, chain_position=position)

    chain_summaries: list[dict[str, Any]] = []
    for i, c in enumerate(chains):
        a1, b, a2 = records["a1"][i], records["b"][i], records["a2"][i]
        aulc_a1 = _metrics.record_aulc(a1)
        aulc_a2 = _metrics.record_aulc(a2)
        ratio = None
        if (aulc_a1 is not None and aulc_a2 is not None
                and aulc_a1 > _metrics.MIN_DENOMINATOR):
            ratio = float(aulc_a2 / aulc_a1)
        chain_summaries.append({
            "schema": CHAIN_SPEC_SCHEMA,
            "chain_id": c.chain_id,
            "family_a": c.family_a,
            "family_b": c.family_b,
            "aulc_a1": aulc_a1,
            "aulc_b": _metrics.record_aulc(b),
            "aulc_a2": aulc_a2,
            "retention_ratio": ratio,
            "interference_cost": (None if (aulc_a1 is None or aulc_a2 is None)
                                  else float(aulc_a1 - aulc_a2)),
            "recovery_interactions": recovery_interactions(a1, a2),
            "episode_ids": {"a1": a1.get("episode_id"), "b": b.get("episode_id"),
                            "a2": a2.get("episode_id")},
        })

    summary = _metrics.retention_interference(chain_summaries)
    return {
        "schema": INTERFERENCE_REPORT_SCHEMA,
        "n_chains": len(chains),
        "pair_counts": pair_counts,
        "balanced": require_balanced,
        "summary": summary,
        "chains": chain_summaries,
        "records": records,
    }


__all__ = [
    "INTERFERENCE_REPORT_SCHEMA",
    "CHAIN_SPEC_SCHEMA",
    "CHAIN_POSITIONS",
    "ChainSpec",
    "ChainSpecError",
    "chain_spec_from_dict",
    "check_pair_balance",
    "recovery_interactions",
    "run_interference_eval",
]
