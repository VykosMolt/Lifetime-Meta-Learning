"""FL8 — consolidation of selected fast deltas into a slow bank (contract §7).

Frozen sequence
===============
    fast-adapt on family A episodes -> fast-adapt on family B -> CONSOLIDATE:
    merge the SELECTED per-episode fast deltas (selection = value gate;
    comparisons: none / indiscriminate all / value-gated) into a separate slow
    LoRA bank with scale eta = 0.5 -> clear fast state -> test A, B and
    unrelated families.

"Only counts as consolidation because it consumes the selected learning
history / fast deltas" (contract §7): this module never trains anything.  It
merges deltas that the FL7 inner loop actually produced, using W2's
``training.lora.merge_scaled_``.

Where the "separate slow bank" lives
------------------------------------
``training/lora.py`` implements a merge as an appended FROZEN low-rank
component: ``D_layer = (alpha/r) B A + sum_i s_i B_i A_i``.  The first term is
the TRAINABLE fast delta (zeroed by ``zero_lora_``, clipped by
``clip_lora_frobenius_``); the ``s_i, A_i, B_i`` terms are the SLOW BANK.  They
are separate objects with separate semantics — non-trainable buffers that
``zero_lora_`` explicitly does not touch — so "merge into a separate slow bank,
then clear the fast state" is exactly ``merge_scaled_`` followed by
``zero_lora_``.  :func:`slow_bank_report` reports the bank on its own.

A recorded per-episode delta is carried by :class:`FastDelta`, which exposes
precisely the three attributes ``merge_scaled_`` reads from a source handle
(``names()``, ``layers[name].lora_A``, ``.lora_B``, ``.scaling``).  It is a
storage carrier for a REAL recorded delta, not a stand-in for the merge: the
merge arithmetic is W2's function, unmodified.  (Restoring a delta with
``load_lora_state_`` instead would CLEAR the destination's merged components —
i.e. wipe the slow bank — which is why the delta is carried separately.)

Selection rule for ``value_gated`` (frozen here; recorded as an amendment)
--------------------------------------------------------------------------
Contract §7 says "selection = value gate" without saying how per-item gate
decisions become a per-episode selection.  Frozen rule: an episode's delta is
merged iff the value gate ADMITTED AT LEAST ONE update in that episode
(``n_admitted > 0``).  A delta produced by zero admitted updates is exactly the
zero delta, so merging it would be a no-op anyway; the rule simply makes the
bookkeeping explicit.

Sealed refusal
--------------
Every entry point rejects a SEALED_TEST family (checked both against the
recorded ``split`` and against ``ecology.split.split_of``), because no model
modification may ever be justified from sealed material.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

from foundation_learner.ecology.base import domain_sha256
from foundation_learner.ecology.split import split_of
from foundation_learner.training.lora import (
    LoraHandles,
    lora_frobenius_norm,
    merge_scaled_,
    zero_lora_,
)
from foundation_learner.training.peft_modes import FL8_CONSOLIDATION_SCALE

__all__ = [
    "CONSOLIDATION_MODES",
    "FL8_CONSOLIDATION_SCALE",
    "ConsolidationError",
    "SealedInputRefused",
    "assert_not_sealed",
    "FastDelta",
    "capture_fast_delta",
    "Consolidator",
    "slow_bank_report",
    "consolidation_eval_plan",
    "run_fl8_stage",
]

#: contract §7 comparisons
CONSOLIDATION_MODES: tuple[str, ...] = ("none", "indiscriminate", "value_gated")

SEALED_SPLIT = "SEALED_TEST"


class ConsolidationError(RuntimeError):
    """Raised on malformed FL8 inputs."""


class SealedInputRefused(ConsolidationError):
    """A SEALED_TEST family reached a consolidation entry point."""


def assert_not_sealed(family_id: str, split: str | None = None) -> str:
    """Refuse sealed material anywhere in the consolidation path."""
    fid = str(family_id)
    computed = None
    try:
        computed = split_of(fid)
    except KeyError:
        computed = None
    if split is not None and str(split) == SEALED_SPLIT:
        raise SealedInputRefused(
            f"REFUSED: consolidation input carries split {split!r} "
            f"(family {fid!r}); no model modification may be justified from "
            "sealed material (contract §12)")
    if computed == SEALED_SPLIT:
        raise SealedInputRefused(
            f"REFUSED: family {fid!r} is in {SEALED_SPLIT}; consolidation "
            "never consumes sealed families (contract §12)")
    if computed is None and split is None:
        raise ConsolidationError(
            f"unknown family {fid!r} and no declared split; refusing to guess "
            "whether it is sealed")
    return computed or str(split)


# ---------------------------------------------------------------------------
# recorded per-episode fast delta
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _DeltaLayer:
    lora_A: torch.Tensor
    lora_B: torch.Tensor
    scaling: float


# eq=False: these records carry tensors, and dataclass equality on tensors
# returns a TENSOR, which would make `x in list` ambiguous.  Identity is the
# correct notion here (one record per episode).
@dataclass(eq=False)
class FastDelta:
    """One episode's recorded fast delta (a merge SOURCE for ``merge_scaled_``)."""

    episode_id: str
    family_id: str
    split: str
    layers: dict[str, _DeltaLayer]
    n_admitted: int = 0
    n_declined: int = 0
    frobenius_norm: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.layers)

    @property
    def admitted(self) -> bool:
        """Frozen value-gate selection rule (see the module docstring)."""
        return self.n_admitted > 0

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "family_id": self.family_id,
            "split": self.split,
            "n_layers": len(self.layers),
            "n_admitted": self.n_admitted,
            "n_declined": self.n_declined,
            "frobenius_norm": self.frobenius_norm,
            "admitted": self.admitted,
        }


def capture_fast_delta(handles: LoraHandles, *, episode_id: str,
                       family_id: str, split: str,
                       n_admitted: int = 0, n_declined: int = 0,
                       meta: dict[str, Any] | None = None) -> FastDelta:
    """Snapshot the TRAINABLE delta of ``handles`` (merged bank excluded)."""
    assert_not_sealed(family_id, split)
    layers = {
        name: _DeltaLayer(
            lora_A=handles.layers[name].lora_A.detach().cpu().clone(),
            lora_B=handles.layers[name].lora_B.detach().cpu().clone(),
            scaling=float(handles.layers[name].scaling))
        for name in handles.names()
    }
    return FastDelta(
        episode_id=str(episode_id), family_id=str(family_id), split=str(split),
        layers=layers, n_admitted=int(n_admitted), n_declined=int(n_declined),
        frobenius_norm=lora_frobenius_norm(handles), meta=dict(meta or {}))


# ---------------------------------------------------------------------------
# the consolidator
# ---------------------------------------------------------------------------
class Consolidator:
    """Merge selected per-episode deltas into the slow bank (eta = 0.5)."""

    def __init__(self, handles: LoraHandles, *, mode: str = "value_gated",
                 scale: float = FL8_CONSOLIDATION_SCALE) -> None:
        if mode not in CONSOLIDATION_MODES:
            raise ConsolidationError(
                f"unknown consolidation mode {mode!r}; frozen: "
                f"{list(CONSOLIDATION_MODES)}")
        self.handles = handles
        self.mode = mode
        self.scale = float(scale)
        self.deltas: list[FastDelta] = []
        self.merges: list[dict[str, Any]] = []

    # -- accumulation -----------------------------------------------------
    def record(self, delta: FastDelta) -> FastDelta:
        assert_not_sealed(delta.family_id, delta.split)
        if delta.names() != self.handles.names():
            raise ConsolidationError(
                "recorded delta targets different modules than the live "
                "adapter; a partial merge would corrupt the consolidation claim")
        self.deltas.append(delta)
        return delta

    def record_many(self, deltas: Iterable[FastDelta]) -> list[FastDelta]:
        return [self.record(d) for d in deltas]

    def selected(self) -> list[FastDelta]:
        if self.mode == "none":
            return []
        if self.mode == "indiscriminate":
            return list(self.deltas)
        return [d for d in self.deltas if d.admitted]

    # -- the merge --------------------------------------------------------
    def consolidate(self, *, clear_fast_state: bool = True) -> dict[str, Any]:
        """Merge the selected deltas, then clear the fast state.

        Deterministic: deltas are merged in the order they were recorded, with
        the frozen scale, using ``training.lora.merge_scaled_``.
        """
        for delta in self.deltas:
            assert_not_sealed(delta.family_id, delta.split)
        selected = self.selected()
        selected_ids = {id(d) for d in selected}
        merged_records = []
        for delta in selected:
            n_layers = merge_scaled_(self.handles, delta, self.scale)
            merged_records.append({**delta.to_record(), "merged_layers": n_layers})
        fast_norm_before = lora_frobenius_norm(self.handles)
        if clear_fast_state:
            zero_lora_(self.handles)
        report = {
            "schema": "flb200.fl8_consolidation.v1",
            "mode": self.mode,
            "scale": self.scale,
            "n_recorded": len(self.deltas),
            "n_selected": len(selected),
            "selected_episode_ids": [d.episode_id for d in selected],
            "declined_episode_ids": [d.episode_id for d in self.deltas
                                     if id(d) not in selected_ids],
            "merged": merged_records,
            "fast_delta_norm_before_clear": fast_norm_before,
            "fast_delta_norm_after_clear": lora_frobenius_norm(self.handles),
            "fast_state_cleared": bool(clear_fast_state),
            "slow_bank": slow_bank_report(self.handles),
        }
        report["report_hash"] = domain_sha256("FL_V0_FL8_CONSOLIDATION", report)
        self.merges.append(report)
        return report

    def config(self) -> dict[str, Any]:
        return {"mode": self.mode, "scale": self.scale,
                "modules": self.handles.names()}

    def config_hash(self) -> str:
        return domain_sha256("FL_V0_FL8_CONSOLIDATOR", self.config())


def slow_bank_report(handles: LoraHandles) -> dict[str, Any]:
    """The frozen merged components only (the slow bank), never the fast delta."""
    per_layer = []
    total_sq = 0.0
    n_components = 0
    for name in handles.names():
        layer = handles.layers[name]
        components = layer.merged_components()
        n_components += len(components)
        norm_sq = 0.0
        for scale, a, b in components:
            dense = float(scale) * (b.detach().double() @ a.detach().double())
            norm_sq += float(torch.sum(dense * dense).item())
        total_sq += norm_sq
        per_layer.append({"module": name, "n_components": len(components),
                          "frobenius_norm": float(norm_sq ** 0.5)})
    return {
        "n_modules": len(per_layer),
        "n_components": n_components,
        "frobenius_norm": float(total_sq ** 0.5),
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# the A -> B -> A + unrelated evaluation plan
# ---------------------------------------------------------------------------
def consolidation_eval_plan(
    *,
    family_a: str,
    family_b: str,
    episodes_a1: Sequence[Any],
    episodes_b: Sequence[Any],
    episodes_a2: Sequence[Any],
    unrelated_episodes: Sequence[Any] = (),
    chain_id_prefix: str = "fl8",
) -> dict[str, Any]:
    """Build the FL8 evaluation plan for W4's evaluators.

    Returns ``chains`` (``evaluation.interference.ChainSpec`` objects, ready for
    ``run_interference_eval``) covering A -> B -> A, plus ``unrelated``
    episodes for a plain ``evaluation.learning_curve.run_episodes`` sweep
    (contract §7: "test A, B, and unrelated families").

    Deterministic: chains are emitted in the order of the supplied episode
    lists and their ids are derived from that order, never from a clock.
    Sealed families are refused.
    """
    from foundation_learner.evaluation.interference import ChainSpec

    assert_not_sealed(family_a)
    assert_not_sealed(family_b)
    if family_a == family_b:
        raise ConsolidationError("A -> B -> A requires two DIFFERENT families")
    n = min(len(episodes_a1), len(episodes_b), len(episodes_a2))
    if n == 0:
        raise ConsolidationError(
            "consolidation_eval_plan needs at least one A1, B and A2 episode")
    chains = []
    for i in range(n):
        for episode, expected in ((episodes_a1[i], family_a),
                                  (episodes_b[i], family_b),
                                  (episodes_a2[i], family_a)):
            fid = str(getattr(episode, "family_id", expected))
            if fid != expected:
                raise ConsolidationError(
                    f"chain {i}: episode family {fid!r} != expected {expected!r}")
            assert_not_sealed(fid, getattr(episode, "split", None))
        chains.append(ChainSpec(
            chain_id=f"{chain_id_prefix}:{family_a}->{family_b}:{i}",
            family_a=family_a, family_b=family_b,
            episode_a1=episodes_a1[i], episode_b=episodes_b[i],
            episode_a2=episodes_a2[i]))
    unrelated = []
    for episode in unrelated_episodes:
        fid = str(getattr(episode, "family_id", ""))
        assert_not_sealed(fid, getattr(episode, "split", None))
        if fid in (family_a, family_b):
            raise ConsolidationError(
                f"episode of family {fid!r} is not an UNRELATED family")
        unrelated.append(episode)
    plan = {
        "schema": "flb200.fl8_eval_plan.v1",
        "family_a": family_a,
        "family_b": family_b,
        "n_chains": len(chains),
        "chain_ids": [c.chain_id for c in chains],
        "unrelated_families": sorted({str(getattr(e, "family_id", ""))
                                      for e in unrelated}),
        "n_unrelated_episodes": len(unrelated),
    }
    plan["plan_hash"] = domain_sha256("FL_V0_FL8_EVAL_PLAN", plan)
    return {"plan": plan, "chains": chains, "unrelated": unrelated}


# ---------------------------------------------------------------------------
# campaign stage adapter (contract §7 FL8)
# ---------------------------------------------------------------------------
def run_fl8_stage(ctx: Any, stage: Any) -> dict:
    """FL8 rung entry point for ``campaign/stage_definitions.py``.

    The frozen §7 sequence, once per consolidation mode, each mode on its OWN
    fresh bundle:

        fast-adapt on family A episodes -> fast-adapt on family B ->
        consolidate the SELECTED deltas into the slow bank (eta = 0.5) ->
        clear the fast state -> test A, B and unrelated families.

    "Test A, B and unrelated" is executed as the pre-generated A->B->A chains
    through W4's ``interference.run_interference_eval`` (retention ratio,
    interference cost, recovery interactions) plus a plain DEVELOPMENT sweep on
    an UNRELATED family.  The three modes {none, indiscriminate, value_gated}
    are the frozen comparison; ``value_gated`` needs the FL4 head and falls
    back to recording that the gate was unavailable rather than selecting
    deltas by some other rule.
    """
    from foundation_learner.evaluation.interference import run_interference_eval
    from foundation_learner.mechanisms import stage_support as _s
    from foundation_learner.mechanisms.fast_adapter import FastAdapter

    dev_eps = _s.dev_episodes(ctx, stage)
    if not dev_eps:
        return _s.finish_stage(ctx, stage, _s.mechanism_payload(
            stage, _s.STATUS_SKIPPED_INPUT,
            reason="no DEVELOPMENT episodes are available"))
    all_dev = _s.load_split_episodes(ctx, "DEVELOPMENT")
    n_chains = int(ctx.extra.get("fl8_chains", 1 if ctx.rehearsal
                                 else ctx.eval_cap(stage)))
    chains = _s.load_chain_specs(ctx, all_dev, limit=n_chains)
    if not chains:
        return _s.finish_stage(ctx, stage, _s.mechanism_payload(
            stage, _s.STATUS_SKIPPED_INPUT,
            reason="no pre-generated A->B->A chains resolve against the shards"))

    family_a, family_b = chains[0].family_a, chains[0].family_b
    n_unrelated = int(ctx.extra.get("fl8_unrelated_episodes",
                                    1 if ctx.rehearsal else ctx.eval_cap(stage)))
    unrelated = [ep for ep in all_dev
                 if ep.family_id not in (family_a, family_b)][:n_unrelated]
    plan = consolidation_eval_plan(
        family_a=family_a, family_b=family_b,
        episodes_a1=[c.episode_a1 for c in chains],
        episodes_b=[c.episode_b for c in chains],
        episodes_a2=[c.episode_a2 for c in chains],
        unrelated_episodes=unrelated, chain_id_prefix=stage.stage_id.lower())

    # the episodes the fast adapter learns from (REVEAL-supervised)
    adapt_a = [_s.reveal_variant(c.episode_a1) for c in chains]
    adapt_b = [_s.reveal_variant(c.episode_b) for c in chains]

    head = _s.value_head_from_ctx(ctx)
    gate = None
    if head is not None:
        from foundation_learner.mechanisms.value_gating import ValueGate

        gate = ValueGate(head)

    declared = ctx.extra.get("fl8_modes")
    modes = tuple(declared) if declared else CONSOLIDATION_MODES
    unknown = [m for m in modes if m not in CONSOLIDATION_MODES]
    if unknown:
        raise _s.stage_error(f"unknown consolidation modes {unknown}")

    results: dict[str, Any] = {}
    for mode in modes:
        bundle = ctx.bundle_factory()                   # FRESH load per mode
        use_gate = gate if mode == "value_gated" else None
        adapter = FastAdapter(bundle, seed=int(ctx.root_seed), gate=use_gate)
        consolidator = Consolidator(adapter.handles, mode=mode)
        adapted: list[dict] = []
        for family, episodes in ((family_a, adapt_a), (family_b, adapt_b)):
            for episode in episodes:
                adapter.reset_episode()
                admitted = declined = 0
                reveals = _reveal_items(episode)
                for item_id in sorted(reveals, key=lambda k: reveals[k]):
                    summary = None
                    if use_gate is not None:
                        summary = _item_summary(bundle, episode, item_id)
                    record = adapter.adapt_item(episode, item_id,
                                                hidden_summary=summary)
                    admitted += int(record.applied)
                    declined += int(not record.applied)
                delta = capture_fast_delta(
                    adapter.handles, episode_id=episode.episode_id,
                    family_id=episode.family_id, split=str(episode.split),
                    n_admitted=admitted, n_declined=declined,
                    meta={"phase": "A" if family == family_a else "B"})
                consolidator.record(delta)
                adapted.append(delta.to_record())
        report = consolidator.consolidate()
        # the inner loop trains IN the model, and the evaluation that follows
        # is a greedy decode: force the evaluation state first (Amendment 16).
        # `dev_records` does this for the unrelated probes; the interference
        # walk calls the evaluator directly, so it needs it here.
        _s.prepare_for_evaluation(bundle)
        interference = run_interference_eval(
            bundle, plan["chains"], _s.env_factory,
            cfg=_lc_config(ctx, f"FL8_{mode}"), require_balanced=False)
        unrelated_records = (
            _s.dev_records(bundle, plan["unrelated"], ctx=ctx,
                           arm_tag=f"FL8_{mode}_unrelated")
            if plan["unrelated"] else [])
        results[mode] = {
            "consolidation": report,
            "deltas": adapted,
            "gate_available": bool(use_gate is not None),
            "retention_interference": interference["summary"],
            "chains": interference["chains"],
            "unrelated_macro_aulc": _s.macro_aulc(unrelated_records),
            "unrelated_dev": (_s.dev_metrics_dict(unrelated_records,
                                                  f"FL8_{mode}_unrelated")
                              if unrelated_records else None),
            "slow_bank": report["slow_bank"],
            "fast_state_cleared": report["fast_state_cleared"],
        }

    payload = _s.mechanism_payload(
        stage, _s.STATUS_COMPLETE,
        modes=list(modes), scale=FL8_CONSOLIDATION_SCALE,
        family_a=family_a, family_b=family_b,
        n_chains=len(chains), n_unrelated_episodes=len(plan["unrelated"]),
        eval_plan=plan["plan"],
        gate_available=bool(gate is not None),
        results=results,
        sequence=("fast-adapt A -> fast-adapt B -> consolidate selected deltas "
                  "(eta = 0.5) -> clear fast state -> test A, B, unrelated"),
    )
    return _s.finish_stage(ctx, stage, payload)


def _reveal_items(episode: Any) -> dict[str, int]:
    from foundation_learner.mechanisms.fast_adapter import revealed_support_items

    return revealed_support_items(episode)


def _item_summary(bundle: Any, episode: Any, item_id: str):
    """Hidden summary of the item's REVEAL event, for the gate.

    The gate consumes ONLY this tensor (contract §7 FL6); it is the same
    ``reduce="last"`` summary the FL4 head was trained on.
    """
    from foundation_learner.mechanisms.value_head import head_input_hidden

    reveals = _reveal_items(episode)
    return head_input_hidden(bundle, episode, int(reveals[item_id]))


def _lc_config(ctx: Any, arm_tag: str):
    from foundation_learner.evaluation.learning_curve import LearningCurveConfig
    from foundation_learner.mechanisms import stage_support as _s

    return LearningCurveConfig(arm_tag=arm_tag,
                               generation=_s.generation_config(ctx))
