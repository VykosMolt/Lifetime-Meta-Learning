"""FL6 — value-gated fast adaptation (contract §7).

Frozen rule
===========
    Apply the FROZEN FL4 head to item ``j``; incorporate the FL5 state update /
    the FL7 inner update IFF ``v_j > 0``, else skip.  Threshold 0 is FROZEN
    before sealed evaluation.

The gate's ONLY input is the head prediction
=============================================
:meth:`ValueGate.decide` takes exactly one positional argument — the item's
hidden summary — and returns a :class:`GateDecision`.  It has no parameter for
a verifier result, a poison flag, a certified/uncertified channel tag, an
interaction outcome or any future observation, and it holds no reference to an
episode, an environment or a verifier.  Passing such information is a
``TypeError``, and the hostile fixture
``tests/hostile/test_hostile_value_gate_ground_truth.py`` asserts exactly that.

The head is frozen at construction (``ValueHead.freeze_()`` + ``eval()``), and
:meth:`decide` runs under ``torch.no_grad``: gating can never train the head,
and the head's parameters cannot drift while it is being used as a gate.

``UnconditionalGate`` is the contract's comparison arm ("unconditional vs
gated"): the identical call site, always incorporating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from foundation_learner.ecology.base import domain_sha256
from foundation_learner.mechanisms.value_head import ValueHead

__all__ = [
    "FL6_GATE_THRESHOLD",
    "GateDecision",
    "ValueGate",
    "UnconditionalGate",
    "GateError",
    "REHEARSAL_POISON_CONDITIONS",
    "run_fl6_stage",
]

#: contract §7: "incorporate iff v_j > 0"; frozen before sealed evaluation.
FL6_GATE_THRESHOLD = 0.0


class GateError(RuntimeError):
    """Raised on malformed gate configuration."""


@dataclass(frozen=True)
class GateDecision:
    """What a gate returns: the decision and the value it was based on."""

    incorporate: bool
    #: the head prediction; ``None`` for the unconditional arm, which computes
    #: no value at all (NaN is never used: it is not canonical-JSON safe).
    value: float | None
    threshold: float | None
    gate: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "incorporate": bool(self.incorporate),
            "value": None if self.value is None else float(self.value),
            "threshold": None if self.threshold is None else float(self.threshold),
            "gate": self.gate,
        }


class ValueGate:
    """Frozen-FL4-head gate: incorporate iff ``head(summary) > threshold``."""

    name = "value_gated"

    def __init__(self, head: ValueHead,
                 threshold: float = FL6_GATE_THRESHOLD) -> None:
        if not isinstance(head, ValueHead):
            raise GateError(
                f"the FL6 gate needs an FL4 ValueHead, got {type(head)!r}")
        self.head = head.freeze_()
        self.threshold = float(threshold)
        if any(p.requires_grad for p in self.head.parameters()):
            raise GateError("the FL6 gate requires a FROZEN head")

    def decide(self, hidden_summary: torch.Tensor) -> GateDecision:
        """The ONLY public entry point; it accepts the item summary and nothing else."""
        with torch.no_grad():
            value = float(self.head(hidden_summary).reshape(()).item())
        return GateDecision(incorporate=value > self.threshold, value=value,
                            threshold=self.threshold, gate=self.name)

    def config(self) -> dict[str, Any]:
        return {
            "gate": self.name,
            "threshold": self.threshold,
            "head_config": self.head.config(),
            "head_config_hash": self.head.config_hash(),
        }

    def config_hash(self) -> str:
        return domain_sha256("FL_V0_FL6_GATE", self.config())


class UnconditionalGate:
    """The comparison arm: always incorporate (contract §7 "unconditional")."""

    name = "unconditional"

    def __init__(self) -> None:
        self.threshold = None

    def decide(self, hidden_summary: torch.Tensor) -> GateDecision:
        return GateDecision(incorporate=True, value=None,
                            threshold=self.threshold, gate=self.name)

    def config(self) -> dict[str, Any]:
        return {"gate": self.name, "threshold": self.threshold}

    def config_hash(self) -> str:
        return domain_sha256("FL_V0_FL6_GATE", self.config())


# ---------------------------------------------------------------------------
# campaign stage adapter (contract §7 FL6, §9 poison conditions)
# ---------------------------------------------------------------------------
#: §7 FL6 compares "unconditional vs gated under poison-laden episodes".  A
#: dress rehearsal walks the two ENDS of the frozen §9 condition list (the
#: untouched condition and the corrupted one) to keep the mechanics walk short;
#: a real run uses every frozen condition present in the data.
REHEARSAL_POISON_CONDITIONS: tuple[str, ...] = ("clean", "corrupted")


def run_fl6_stage(ctx: Any, stage: Any) -> dict:
    """FL6 rung entry point for ``campaign/stage_definitions.py``.

    For every frozen poison condition the DEVELOPMENT episodes are evaluated
    twice with the SAME fast-state mechanism: once with the UNCONDITIONAL gate
    and once with the FROZEN FL4 head as the gate (``incorporate iff v > 0``).
    Recorded per condition and per variant: macro-AULC, the number of state
    updates admitted and declined, and the admission rate — i.e. exactly the
    "blind incorporation vs gating behaviour" contrast of §9.

    The gate needs the FL4 head produced earlier in this ladder.  If FL4 did
    not run in this session the rung records a SKIP; it never fabricates a
    head (an untrained head would be a random gate presented as a mechanism).
    """
    from foundation_learner.ecology.poison import POISON_CONDITIONS
    from foundation_learner.evaluation.poison_eval import episodes_from_poison_record
    from foundation_learner.mechanisms import stage_support as _s
    from foundation_learner.mechanisms.fast_state import (FastStateCallbacks,
                                                          FastStateModule)

    head = _s.value_head_from_ctx(ctx)
    if head is None:
        return _s.finish_stage(ctx, stage, _s.mechanism_payload(
            stage, _s.STATUS_SKIPPED_INPUT,
            reason=("no FL4 value head is available in this session; the FL6 "
                    "gate is the FROZEN FL4 head and is never fabricated")))

    dev_eps = _s.dev_episodes(ctx, stage)
    poison_records = _s.load_poison_records(ctx, "DEVELOPMENT")
    if not dev_eps:
        return _s.finish_stage(ctx, stage, _s.mechanism_payload(
            stage, _s.STATUS_SKIPPED_INPUT,
            reason="no DEVELOPMENT episodes are available"))

    declared = ctx.extra.get("fl6_conditions")
    conditions = tuple(declared) if declared else (
        REHEARSAL_POISON_CONDITIONS if ctx.rehearsal else POISON_CONDITIONS)
    unknown = [c for c in conditions if c not in POISON_CONDITIONS]
    if unknown:
        raise _s.stage_error(f"unknown poison conditions {unknown}")

    bundle = ctx.bundle_factory()                       # FRESH load (§7)
    module = FastStateModule(
        int(bundle.model.config.hidden_size),
        embedding_matrix=bundle.model.get_input_embeddings().weight.detach(),
        seed=int(ctx.root_seed))
    gates = {"unconditional": UnconditionalGate(), "value_gated": ValueGate(head)}

    by_variant: dict[str, dict] = {}
    records_by_variant: dict[str, dict[str, list]] = {}
    for variant, gate in gates.items():
        per_condition: dict[str, Any] = {}
        records_by_variant[variant] = {}
        for condition in conditions:
            episodes = []
            for episode in dev_eps:
                record = poison_records.get(episode.episode_id)
                if record is None:
                    continue
                variants = episodes_from_poison_record(episode, record)
                if condition in variants:
                    episodes.append(variants[condition])
            if not episodes:
                per_condition[condition] = {"n_episodes": 0,
                                            "reason": "no variant in the shards"}
                continue
            seen: list[FastStateCallbacks] = []

            def factory(_ep, _gate=gate, _seen=seen):
                cb = FastStateCallbacks(module, gate=_gate)
                _seen.append(cb)
                return cb

            recs = _s.dev_records(bundle, episodes, ctx=ctx,
                                  arm_tag=f"FL6_{variant}_{condition}",
                                  callbacks_factory=factory)
            recs = [dict(r, poison_condition=condition) for r in recs]
            records_by_variant[variant][condition] = recs
            applied = sum(cb.updates_applied for cb in seen)
            skipped = sum(cb.updates_skipped for cb in seen)
            total = applied + skipped
            per_condition[condition] = {
                "n_episodes": len(episodes),
                "macro_aulc": _s.macro_aulc(recs),
                "dev": _s.dev_metrics_dict(recs, f"FL6_{variant}_{condition}"),
                "updates_admitted": applied,
                "updates_declined": skipped,
                "admission_rate": (applied / total) if total else None,
                "gate_values": [v for cb in seen for v in cb.gate_values],
            }
        by_variant[variant] = per_condition

    from foundation_learner.evaluation import metrics as _metrics

    robustness = {
        variant: _metrics.poison_robustness(recs)
        for variant, recs in records_by_variant.items() if recs}
    payload = _s.mechanism_payload(
        stage, _s.STATUS_COMPLETE,
        gate_threshold=FL6_GATE_THRESHOLD,
        conditions=list(conditions),
        variants=by_variant,
        poison_robustness=robustness,
        head_config_hash=head.config_hash(),
        gate_config={name: gate.config() for name, gate in gates.items()},
        n_dev_episodes=len(dev_eps),
        comparison=("unconditional vs value-gated incorporation under the "
                    "frozen §9 poison conditions"),
    )
    return _s.finish_stage(ctx, stage, payload)
