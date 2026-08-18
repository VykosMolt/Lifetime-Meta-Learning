"""Surface-remap evaluation (contract sections 4 and 9).

The latent rule is fixed; only the SURFACE changes.  The model learns from the
support phase in the canonical surface and is then queried in a remapped
surface (two remapped variants per canonical episode, per the frozen campaign
design), so any drop measures surface brittleness rather than rule ignorance.

Mechanically: the canonical episode supplies every event up to the evaluation
phase, and the remapped episode supplies the evaluation phase (interaction
indices >= ``eval_from_index``, default 5 = QUERY/TRANSFER).  The verifier
operates in the remapped surface, because ``surface_remap`` returns an
instance carrying its own ``answer_canonical`` (contract section 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .learning_curve import (
    EpisodeEnvironmentError,
    LearningCurveConfig,
    annotate_records,
    condition_get,
    episode_events,
    event_attr,
    replace_fields,
    role_name,
    run_episodes,
)
from . import metrics as _metrics

REMAP_REPORT_SCHEMA = "flb200.remap_report.v1"

#: contract section 6: QUERY items carry interaction index 5
DEFAULT_EVAL_FROM_INDEX = 5


class RemapSpliceError(ValueError):
    """Canonical and remapped episodes cannot be spliced."""


@dataclass(frozen=True)
class RemapPair:
    canonical: Any
    variants: dict[str, Any]  # remap_id -> remapped episode


def _first_eval_event_index(ep: Any, eval_from_index: int) -> int:
    for i, ev in enumerate(episode_events(ep)):
        idx = event_attr(ev, "interaction_index")
        if idx is not None and int(idx) >= int(eval_from_index):
            return i
        if role_name(ev) == "QUERY_TASK" and int(eval_from_index) <= 5:
            return i
    raise RemapSpliceError(
        f"episode has no event at interaction index >= {eval_from_index}")


def splice_remapped_episode(canonical: Any, remapped: Any,
                            eval_from_index: int = DEFAULT_EVAL_FROM_INDEX) -> Any:
    """Canonical learning phase + remapped evaluation phase, as one episode."""
    c_cut = _first_eval_event_index(canonical, eval_from_index)
    r_cut = _first_eval_event_index(remapped, eval_from_index)
    events = list(episode_events(canonical))[:c_cut] + list(episode_events(remapped))[r_cut:]
    return replace_fields(canonical, events=events)


class SplicedEnvironment:
    """Serve items from the canonical env, falling through to the remapped env."""

    def __init__(self, canonical_env: Any, remapped_env: Any):
        self.canonical = canonical_env
        self.remapped = remapped_env

    def _pick(self, item_id: str) -> Any:
        for env in (self.canonical, self.remapped):
            known = getattr(env, "instances", None)
            if known is not None and item_id in known:
                return env
        raise EpisodeEnvironmentError(
            f"item {item_id!r} is served by neither the canonical nor the remapped "
            "environment of this remap pair")

    def verify(self, item_id: str, answer_text: str) -> Any:
        return self._pick(item_id).verify(item_id, answer_text)

    def feedback_text(self, item_id: str, answer_text: str, outcome, **kwargs: Any) -> str:
        return self._pick(item_id).feedback_text(item_id, answer_text, outcome, **kwargs)


def evaluation_item_ids(ep: Any, eval_from_index: int = DEFAULT_EVAL_FROM_INDEX
                        ) -> set[str]:
    """Item ids answered at interaction index >= ``eval_from_index``."""
    out: set[str] = set()
    for ev in episode_events(ep):
        idx = event_attr(ev, "interaction_index")
        if (role_name(ev) == "MODEL_ATTEMPT" and idx is not None
                and int(idx) >= int(eval_from_index)):
            out.add(str(event_attr(ev, "item_id", "")))
    return out


def build_remap_episode(canonical: Any, remapped_items: Mapping[str, Any],
                        eval_from_index: int = DEFAULT_EVAL_FROM_INDEX) -> Any:
    """Canonical learning phase, remapped SURFACE for the evaluation items.

    ``remapped_items`` is the data layer's ``item_id -> remapped instance``
    table (``remaps/<split>/<family>.jsonl``).  Only the evaluation items are
    resurfaced: the support/related phase stays in the canonical surface, which
    is what makes the comparison "learned here, asked there".
    """
    targets = evaluation_item_ids(canonical, eval_from_index)
    if not targets:
        raise RemapSpliceError(
            f"episode has no items at interaction index >= {eval_from_index}")
    instances = {str(k): _as_instance(v) for k, v in remapped_items.items()}
    missing = [t for t in sorted(targets) if t not in instances]
    if missing:
        raise RemapSpliceError(f"remap table lacks evaluation items {missing}")

    events = []
    for ev in episode_events(canonical):
        item_id = str(event_attr(ev, "item_id", ""))
        if item_id in targets and role_name(ev) in TASK_ROLE_NAMES:
            events.append(replace_fields(
                ev, text=str(getattr(instances[item_id], "prompt_text"))))
        elif item_id in targets and role_name(ev) == "REVEAL":
            events.append(replace_fields(
                ev, text=str(getattr(instances[item_id], "answer_canonical"))))
        else:
            events.append(ev)

    new_ep = replace_fields(canonical, events=events)
    items = event_attr(new_ep, "items")
    if isinstance(items, dict):
        merged = dict(items)
        for item_id in targets:
            record = dict(merged.get(item_id, {}))
            record["instance"] = instances[item_id].to_dict()
            merged[item_id] = record
        new_ep = replace_fields(new_ep, items=merged)
    return new_ep


def _as_instance(value: Any) -> Any:
    if isinstance(value, dict):
        from foundation_learner.ecology.base import TaskInstance  # type: ignore
        payload = value.get("instance") if "instance" in value else value
        return TaskInstance.from_dict(payload)
    return value


TASK_ROLE_NAMES = frozenset({"TASK", "RELATED_TASK", "QUERY_TASK", "TRANSFER_TASK"})


def remap_id_of(ep: Any) -> str:
    value = condition_get(ep, "remap_id")
    if value is None:
        raise RemapSpliceError("remapped episode carries no remap_id")
    return str(value)


def run_remap_eval(
    bundle: Any,
    pairs: Sequence[RemapPair],
    env_factory: Callable[[Any], Any],
    *,
    cfg: LearningCurveConfig | None = None,
    eval_from_index: int = DEFAULT_EVAL_FROM_INDEX,
    callbacks_factory: Callable[[Any], Any] | None = None,
    render_module: Any = None,
    parser: Callable[[str], str | None] | None = None,
    run_canonical_control: bool = True,
) -> dict[str, Any]:
    """Learn canonical, evaluate under each remapped variant."""
    cfg = cfg or LearningCurveConfig()
    variant_ids = sorted({v for p in pairs for v in p.variants})

    canonical_records: list[dict[str, Any]] = []
    if run_canonical_control:
        canonical_records = annotate_records(
            run_episodes(bundle, [p.canonical for p in pairs], env_factory, cfg=cfg,
                         callbacks_factory=callbacks_factory,
                         render_module=render_module, parser=parser),
            remap_id="canonical")

    records_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in variant_ids:
        spliced: list[Any] = []
        envs: dict[int, Any] = {}
        for p in pairs:
            if variant not in p.variants:
                continue
            payload = p.variants[variant]
            if hasattr(payload, "events") or (isinstance(payload, dict)
                                              and "events" in payload):
                ep = splice_remapped_episode(p.canonical, payload, eval_from_index)
                env = SplicedEnvironment(env_factory(p.canonical), env_factory(payload))
            else:
                ep = build_remap_episode(p.canonical, payload, eval_from_index)
                env = env_factory(ep)
            spliced.append(ep)
            envs[id(ep)] = env
        if not spliced:
            continue
        recs = run_episodes(bundle, spliced, lambda ep: envs[id(ep)], cfg=cfg,
                            callbacks_factory=callbacks_factory,
                            render_module=render_module, parser=parser)
        records_by_variant[variant] = annotate_records(recs, remap_id=variant)

    robustness = _metrics.remap_robustness(canonical_records, records_by_variant)
    return {
        "schema": REMAP_REPORT_SCHEMA,
        "eval_from_index": int(eval_from_index),
        "variants": variant_ids,
        "n_pairs": len(pairs),
        "robustness": robustness,
        "canonical_records": canonical_records,
        "records_by_variant": records_by_variant,
    }


def pairs_from_mapping(canonical_episodes: Sequence[Any],
                       remapped_by_episode: Mapping[str, Sequence[Any]]) -> list[RemapPair]:
    """Build :class:`RemapPair`s from ``episode_id -> [remapped episodes]``."""
    out = []
    for ep in canonical_episodes:
        variants = {remap_id_of(v): v
                    for v in remapped_by_episode.get(str(event_attr(ep, "episode_id")), [])}
        out.append(RemapPair(canonical=ep, variants=variants))
    return out


def pairs_from_remap_records(canonical_episodes: Sequence[Any],
                             remap_records: Sequence[Mapping[str, Any]]
                             ) -> list[RemapPair]:
    """Build pairs from the data layer's ``remaps/<split>/<family>.jsonl`` rows.

    Each row is ``{episode_id, remap_id, items: {item_id: instance}}``; the
    variant payload is therefore an item table, and the evaluation phase of the
    canonical episode is resurfaced with it.
    """
    by_episode: dict[str, dict[str, Any]] = {}
    for row in remap_records:
        by_episode.setdefault(str(row["episode_id"]), {})[str(row["remap_id"])] = \
            dict(row["items"])
    return [RemapPair(canonical=ep,
                      variants=by_episode.get(str(event_attr(ep, "episode_id")), {}))
            for ep in canonical_episodes]


__all__ = [
    "REMAP_REPORT_SCHEMA",
    "DEFAULT_EVAL_FROM_INDEX",
    "RemapSpliceError",
    "RemapPair",
    "SplicedEnvironment",
    "splice_remapped_episode",
    "build_remap_episode",
    "evaluation_item_ids",
    "remap_id_of",
    "pairs_from_mapping",
    "pairs_from_remap_records",
    "run_remap_eval",
]
