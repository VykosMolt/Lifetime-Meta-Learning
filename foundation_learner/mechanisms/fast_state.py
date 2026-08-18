"""FL5 — persistent fast state (contract §7, §9, §23).

Frozen mechanism
================
::

    s in R^1024, s = 0 at episode start
    u_j = W_in . mean(final-layer, final-loop hidden states over item-j tokens)
    s  <- GRUCell(u_j, s)                         # on FEEDBACK / HINT / REVEAL
    P   = reshape(W_p . LayerNorm(s), (8, hidden))  # injected via inputs_embeds
    ||P_i|| <= rho = 2 * median base input-embedding row L2 norm

Contract §23 signature::

    FastStateModule(torch.nn.Module):
        reset_state(batch) -> s0
        update(s, h_item)  -> s'
        prefix_embeds(s)   -> Tensor[8, hidden]  (per-vector norm clamped)

Implementation decisions, all documented here
---------------------------------------------
* ``W_in`` and ``W_p`` are BIAS-FREE ``nn.Linear`` layers, matching the
  contract's equations (``u_j = W_in . mean(...)``) and its parameter estimate
  (``2048*1024 + 3*(1024^2 + 1024^2) + 1024*8*2048 ~ 25M``, which counts no
  biases).  ``GRUCell`` keeps its standard biases (they are part of the audited
  GRU equations).
* ``hidden_size`` is a constructor parameter, not the constant 2048, so the
  same module runs on the tiny mechanics model (hidden 64).  For the frozen
  checkpoint it is 2048, i.e. the contract's numbers exactly.
* ``rho`` is computed ONCE from a provided input-embedding matrix
  (:func:`rho_from_embeddings` = 2 x median row L2 norm), stored in the module
  as a non-persistent-free buffer, and recorded in :meth:`config` /
  :meth:`config_hash`.  A module may alternatively be built with an explicit
  ``rho`` (needed to reload a checkpointed module); exactly one of the two must
  be supplied, so a rho can never be silently defaulted.
* The clamp is ``P_i * min(1, rho / ||P_i||)``, which is differentiable
  everywhere the norm is non-zero and is EXACT at the boundary (a unit test
  pins ``||P_i|| == rho`` after clamping an over-long vector).
* Detach control is explicit on :meth:`update` (``detach_state`` /
  ``detach_input``).  Nothing detaches implicitly, so BPTT within an episode
  works by default and any truncation is a visible caller decision.

Nothing in this module uses the global RNG or the wall clock; the parameter
initialisation is driven by an explicit ``torch.Generator`` seeded through
``ecology.base.derive_seed``.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from torch import nn

from foundation_learner.ecology.base import derive_seed, domain_sha256
from foundation_learner.evaluation.learning_curve import NullCallbacks

__all__ = [
    "FL5_STATE_DIM",
    "FL5_PREFIX_K",
    "FL5_RHO_MEDIAN_FACTOR",
    "FastStateError",
    "rho_from_embeddings",
    "FastStateModule",
    "FastStateCallbacks",
    "FastStateS0Callbacks",
    "FastStateOffCallbacks",
    "FL5_STATE_NORM_LIMIT",
    "ARM_FAST_STATE_ON_S0",
    "run_fl5_stage",
]

#: EVAL-ONLY control arm (Amendment 13): the trained FAST_STATE_ON module with
#: ``s`` pinned to 0.  It is never trained separately.
ARM_FAST_STATE_ON_S0 = "FAST_STATE_ON_S0"

#: contract §7
FL5_STATE_DIM = 1024
FL5_PREFIX_K = 8
FL5_RHO_MEDIAN_FACTOR = 2.0

#: mechanism-specific fast-state norm bound handed to ``training.stability``
#: (Amendment 6 leaves the value to FL5; ||s|| is bounded by the GRU's tanh
#: gate to sqrt(1024) ~ 32, so 64.0 is a genuine explosion detector).
FL5_STATE_NORM_LIMIT = 64.0


class FastStateError(RuntimeError):
    """Raised on malformed FL5 configuration or usage."""


def rho_from_embeddings(embedding_matrix: torch.Tensor) -> float:
    """``rho = 2 * median_i ||E_i||_2`` over the input-embedding rows."""
    if embedding_matrix.dim() != 2:
        raise FastStateError(
            f"embedding matrix must be 2-D, got {tuple(embedding_matrix.shape)}")
    norms = torch.linalg.vector_norm(embedding_matrix.detach().float(), dim=1)
    return float(FL5_RHO_MEDIAN_FACTOR * torch.median(norms).item())


def _init_linear_(layer: nn.Linear, gen: torch.Generator) -> None:
    fan_in = int(layer.weight.shape[1])
    bound = math.sqrt(6.0 / ((1.0 + 5.0) * fan_in))  # kaiming_uniform_(a=sqrt(5))
    with torch.no_grad():
        layer.weight.copy_(torch.empty(layer.weight.shape, dtype=torch.float32)
                           .uniform_(-bound, bound, generator=gen))
        if layer.bias is not None:
            b = 1.0 / math.sqrt(fan_in)
            layer.bias.copy_(torch.empty(layer.bias.shape, dtype=torch.float32)
                             .uniform_(-b, b, generator=gen))


def _init_gru_(cell: nn.GRUCell, gen: torch.Generator) -> None:
    """``nn.GRUCell.reset_parameters`` driven by an explicit generator."""
    stdv = 1.0 / math.sqrt(cell.hidden_size)
    with torch.no_grad():
        for param in (cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh):
            if param is None:
                continue
            param.copy_(torch.empty(param.shape, dtype=torch.float32)
                        .uniform_(-stdv, stdv, generator=gen))


class FastStateModule(nn.Module):
    """Contract §23 ``FastStateModule``."""

    def __init__(self, hidden_size: int, *,
                 embedding_matrix: torch.Tensor | None = None,
                 rho: float | None = None,
                 state_dim: int = FL5_STATE_DIM,
                 prefix_k: int = FL5_PREFIX_K,
                 seed: int = 0,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if (embedding_matrix is None) == (rho is None):
            raise FastStateError(
                "supply EXACTLY ONE of embedding_matrix (rho is then computed "
                "as 2 x median row norm and recorded) or an explicit rho "
                "(checkpoint reload); rho is never defaulted")
        self.hidden_size = int(hidden_size)
        self.state_dim = int(state_dim)
        self.prefix_k = int(prefix_k)
        self.seed = int(seed)
        self.W_in = nn.Linear(self.hidden_size, self.state_dim, bias=False)
        self.gru = nn.GRUCell(self.state_dim, self.state_dim)
        self.ln = nn.LayerNorm(self.state_dim)
        self.W_p = nn.Linear(self.state_dim, self.prefix_k * self.hidden_size,
                             bias=False)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(derive_seed(self.seed, "FL5_FAST_STATE_INIT",
                                    self.hidden_size, self.state_dim,
                                    self.prefix_k))
        _init_linear_(self.W_in, gen)
        _init_gru_(self.gru, gen)
        _init_linear_(self.W_p, gen)
        if rho is None:
            rho_value = rho_from_embeddings(embedding_matrix)
            rho_source = "2x_median_input_embedding_row_norm"
        else:
            rho_value = float(rho)
            rho_source = "explicit"
        if not math.isfinite(rho_value) or rho_value <= 0.0:
            raise FastStateError(f"rho must be finite and positive, got {rho_value}")
        self.rho_source = rho_source
        self.register_buffer("rho", torch.tensor(rho_value, dtype=torch.float32))
        self.to(dtype=dtype)

    # -- state -----------------------------------------------------------
    def reset_state(self, batch: int = 1) -> torch.Tensor:
        """Contract §7: ``s = 0`` at episode start (exact zeros)."""
        if batch < 1:
            raise FastStateError(f"batch must be >= 1, got {batch}")
        return torch.zeros(int(batch), self.state_dim,
                           dtype=self.W_in.weight.dtype,
                           device=self.W_in.weight.device)

    def update(self, s: torch.Tensor, h_item: torch.Tensor, *,
               detach_state: bool = False,
               detach_input: bool = False) -> torch.Tensor:
        """``s' = GRUCell(W_in . h_item, s)``.

        ``h_item`` is the MEAN final-loop hidden state over item-j tokens
        (``[hidden]`` or ``[batch, hidden]``).  Both detach switches default to
        False, i.e. gradients flow through the state across segments (BPTT
        within the episode).
        """
        state = self._as_batched(s, self.state_dim, "state")
        item = self._as_batched(h_item, self.hidden_size, "item summary")
        item = item.to(dtype=self.W_in.weight.dtype, device=self.W_in.weight.device)
        state = state.to(dtype=self.W_in.weight.dtype, device=self.W_in.weight.device)
        if detach_state:
            state = state.detach()
        if detach_input:
            item = item.detach()
        if state.shape[0] != item.shape[0]:
            if state.shape[0] == 1:
                state = state.expand(item.shape[0], -1)
            elif item.shape[0] == 1:
                item = item.expand(state.shape[0], -1)
            else:
                raise FastStateError(
                    f"batch mismatch: state {tuple(state.shape)} vs item "
                    f"{tuple(item.shape)}")
        u = self.W_in(item)
        return self.gru(u, state)

    def prefix_embeds(self, s: torch.Tensor) -> torch.Tensor:
        """``[k, hidden]`` (or ``[batch, k, hidden]``) prefix, norm-clamped."""
        state = self._as_batched(s, self.state_dim, "state")
        squeeze = s.dim() == 1
        state = state.to(dtype=self.W_p.weight.dtype, device=self.W_p.weight.device)
        flat = self.W_p(self.ln(state))
        prefix = flat.view(state.shape[0], self.prefix_k, self.hidden_size)
        norms = torch.linalg.vector_norm(prefix, dim=-1, keepdim=True)
        scale = torch.clamp(self.rho.to(prefix.dtype) / norms.clamp_min(1e-12),
                            max=1.0)
        prefix = prefix * scale
        return prefix[0] if squeeze else prefix

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _as_batched(t: torch.Tensor, width: int, what: str) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise FastStateError(f"{what} must be a tensor, got {type(t)!r}")
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() != 2 or t.shape[-1] != width:
            raise FastStateError(
                f"{what} must be [{width}] or [batch, {width}], got "
                f"{tuple(t.shape)}")
        return t

    @staticmethod
    def state_norm(s: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(s.detach().float()).item())

    def config(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "state_dim": self.state_dim,
            "prefix_k": self.prefix_k,
            "seed": self.seed,
            "rho": float(self.rho.item()),
            "rho_source": self.rho_source,
            "w_in_bias": False,
            "w_p_bias": False,
            "layernorm": "before W_p",
        }

    def config_hash(self) -> str:
        return domain_sha256("FL_V0_FL5_FAST_STATE", self.config())

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# eval-side adapter (W4 callback protocol)
# ---------------------------------------------------------------------------
class FastStateCallbacks(NullCallbacks):
    """FL5 adapter for ``evaluation.learning_curve``'s callback protocol.

    Contract behaviour:

    * ``on_episode_start`` RESETS ``s`` to zero — unless ``carry_across_episodes``
      is set, which is the interference-chain mode (``evaluation/interference``
      hands ONE callbacks object to the three episodes of a chain, so the
      mechanism itself decides what persists).  The default therefore makes
      cross-episode leakage impossible.
    * ``on_feedback`` updates ``s`` from the ``mean`` hidden summary of the
      feedback item — the same ``u_j`` the training path uses.  The summary is
      requested with ``include_prefix=False`` so the update is a pure function
      of the textual context (no ``s -> prefix -> s`` recursion).
    * ``prefix_provider`` returns ``[k, hidden]``; the walker prepends it to
      that episode's row through ``inputs_embeds``.
    * A ``gate`` (FL6) may veto an update; the gate sees ONLY the head
      prediction on the item summary (see ``value_gating``).
    * Nothing here reads the poison flag, the verifier verdict, or any hidden
      label; the protocol does not carry them.
    """

    def __init__(self, module: FastStateModule, *,
                 carry_across_episodes: bool = False,
                 gate: Any = None,
                 update_reduce: str = "mean",
                 gate_reduce: str = "last") -> None:
        self.module = module
        self.carry_across_episodes = bool(carry_across_episodes)
        self.gate = gate
        self.update_reduce = str(update_reduce)
        self.gate_reduce = str(gate_reduce)
        self.s = module.reset_state(1)
        self.episodes_seen = 0
        self.updates_applied = 0
        self.updates_skipped = 0
        self.gate_values: list[float] = []
        self.state_norms: list[float] = []
        self.last_context: Any = None

    # -- protocol ---------------------------------------------------------
    def on_episode_start(self, ctx: Any) -> None:
        self.last_context = ctx
        if not (self.carry_across_episodes and self.episodes_seen > 0):
            self.s = self.module.reset_state(1)
        self.episodes_seen += 1

    def prefix_provider(self) -> torch.Tensor | None:
        with torch.no_grad():
            return self.module.prefix_embeds(self.s[0]).detach()

    def on_attempt(self, attempt: Any) -> None:
        return None

    def on_feedback(self, event: Any, hidden_summary_fn: Any) -> None:
        summary = hidden_summary_fn(reduce=self.update_reduce,
                                    include_prefix=False)
        if self.gate is not None:
            gate_summary = (summary if self.gate_reduce == self.update_reduce
                            else hidden_summary_fn(reduce=self.gate_reduce,
                                                   include_prefix=False))
            decision = self.gate.decide(gate_summary)
            self.gate_values.append(decision.value)
            if not decision.incorporate:
                self.updates_skipped += 1
                return
        with torch.no_grad():
            self.s = self.module.update(self.s, summary)
        self.updates_applied += 1
        self.state_norms.append(self.module.state_norm(self.s))

    def on_episode_end(self, ctx: Any) -> None:
        return None

    # -- reporting --------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "schema": "flb200.fast_state_callbacks.v1",
            "episodes_seen": self.episodes_seen,
            "updates_applied": self.updates_applied,
            "updates_skipped": self.updates_skipped,
            "carry_across_episodes": self.carry_across_episodes,
            "gated": self.gate is not None,
            "state_norms": list(self.state_norms),
            "final_state_norm": FastStateModule.state_norm(self.s),
            "module_config_hash": self.module.config_hash(),
        }


class FastStateS0Callbacks(FastStateCallbacks):
    """``FAST_STATE_ON_S0``: the TRAINED module with ``s`` pinned to 0.

    Why this arm exists (Amendment 13).  ``FAST_STATE_ON`` minus
    ``FAST_STATE_OFF`` is not "the state carried the learning": OFF is an
    impossible-task control (segmentation removes the history and OFF has no
    state), ON carries ~25M extra trained parameters, and ``prefix_embeds(0)``
    is a LEARNED STATIC PREFIX — ordinary prefix tuning, which needs no
    within-episode learning at all.

    This arm runs the same trained module and the same 8-vector injection but
    NEVER updates ``s``: every prompt of every episode sees exactly
    ``prefix_embeds(0)``.  Therefore

        ON  - ON_S0  = the contribution of the STATE UPDATES,
        ON_S0 - OFF  = the static prefix plus the extra parameters.

    It is EVAL-ONLY: no third training run exists, and none is implied.  The
    hidden summary is deliberately not requested (the state is pinned, so the
    forward would be paid for nothing), which also keeps this arm's decode
    numerically identical to ON's except for the prefix content.
    """

    ARM_ID = "FAST_STATE_ON_S0"

    def __init__(self, module: FastStateModule, **kwargs: Any) -> None:
        kwargs.pop("carry_across_episodes", None)
        kwargs.pop("gate", None)
        super().__init__(module, carry_across_episodes=False, gate=None,
                         **kwargs)
        self._zero = self.module.reset_state(1)

    def on_episode_start(self, ctx: Any) -> None:
        self.last_context = ctx
        self.s = self._zero
        self.episodes_seen += 1

    def on_feedback(self, event: Any, hidden_summary_fn: Any) -> None:
        # The update is deliberately NOT applied; it is counted as skipped so
        # the record still shows how many updates this arm declined.
        self.updates_skipped += 1
        return None

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out.update({
            "arm_id": self.ARM_ID,
            "state_pinned_to_zero": True,
            "eval_only_control": True,
            "isolates": "learned static prefix (prefix-tuning confound)",
        })
        return out


class FastStateOffCallbacks(NullCallbacks):
    """FAST_STATE_OFF eval arm: the identical pipeline with NO injection.

    It is a real object (not ``None``) so that the walker records the same
    callback plumbing for both arms; it simply never returns a prefix and never
    updates a state.  Being a distinct subclass keeps the two arms
    distinguishable in the episode record's ``callbacks_class`` field.
    """

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "flb200.fast_state_callbacks.v1",
            "injection": "DISABLED",
            "updates_applied": 0,
            "updates_skipped": 0,
        }


# ---------------------------------------------------------------------------
# campaign stage adapter (contract §7 FL5, §9 context reset, §11 stage table)
# ---------------------------------------------------------------------------
def run_fl5_stage(ctx: Any, stage: Any) -> dict:
    """FL5 rung entry point for ``campaign/stage_definitions.py``.

    Both arms of the frozen §7 contrast run here, each from its OWN fresh
    bundle:

    * ``FAST_STATE_ON``  — segmented FL3 training with prefix injection and
      state updates, then the DEVELOPMENT learning curve in BOTH the
      ``history`` and the ``context_reset`` condition with the trained state;
    * ``FAST_STATE_OFF`` — the identical pipeline, data order and seed with
      injection disabled (no prefix, no state usage);
    * ``FAST_STATE_ON_S0`` — EVAL-ONLY control (Amendment 13): the TRAINED ON
      module evaluated on the same episodes with ``s`` pinned to 0, i.e. the
      learned static prefix without any state update.  No third training run
      exists.  ``ON - ON_S0`` is the state-update contribution; ``ON_S0 - OFF``
      is the static prefix plus the extra ~25M parameters.

    Every payload and every per-arm record carries ``comparable_to_fl3: false``
    with the reason: FL5's segmented forward, its impossible-task OFF control,
    its extra parameters and its static-prefix channel all put it off the FL3
    core-comparison scale.

    Promotion interface: the paired family-clustered bootstrap of retained
    post-reset improvement (ON minus OFF, §14 conventions) is published to
    ``ctx.extra["persistence_evidence"]`` as ``{"point", "ci"}``, which is
    exactly what ``stage_definitions.fl8_entry`` feeds to
    ``promotion.promote_fl8`` ("context-reset persistence > 0 with CI
    excluding 0 on DEV").  ``status`` and the per-arm macro-AULCs are the
    rung's own result.
    """
    from foundation_learner.mechanisms import stage_support as _s
    from foundation_learner.mechanisms.fl5_training import (
        ARM_FAST_STATE_OFF, ARM_FAST_STATE_ON, FL5_COMPARABLE_TO_FL3,
        FL5_NOT_COMPARABLE_REASON, FL5TrainConfig, run_fast_state_arm,
        segment_episode)

    comparability = {"comparable_to_fl3": FL5_COMPARABLE_TO_FL3,
                     "comparable_to_fl3_reason": FL5_NOT_COMPARABLE_REASON}

    n_train = _s.stage_episode_count(ctx, stage, "fl5_train_episodes")
    train_eps = _s.train_episodes(ctx, n_train)
    dev_eps = _s.dev_episodes(ctx, stage)
    if not train_eps or not dev_eps:
        return _s.finish_stage(ctx, stage, _s.mechanism_payload(
            stage, _s.STATUS_SKIPPED_INPUT,
            reason="no TRAIN or DEVELOPMENT episodes are available",
            **comparability))
    updates = _s.stage_updates(ctx, "fl5_updates")
    learning_rate = float(ctx.extra.get("fl5_learning_rate",
                                        ctx.learning_rate or 1e-4))
    gate = None
    if ctx.extra.get("fl5_gated"):          # FL6's gated FL5 variant
        from foundation_learner.mechanisms.value_gating import ValueGate

        head = _s.value_head_from_ctx(ctx)
        gate = None if head is None else ValueGate(head)

    arms: dict[str, Any] = {}
    records: dict[str, dict[str, list]] = {}
    for arm_id in (ARM_FAST_STATE_ON, ARM_FAST_STATE_OFF):
        bundle = ctx.bundle_factory()                    # FRESH load per arm
        trainable = _s.trainable_params(ctx, bundle.model)
        module = FastStateModule(
            int(bundle.model.config.hidden_size),
            embedding_matrix=bundle.model.get_input_embeddings().weight.detach(),
            seed=int(ctx.root_seed))
        segments = [segment_episode(ep, bundle.tokenizer) for ep in train_eps]
        cfg = FL5TrainConfig(arm_id=arm_id, learning_rate=learning_rate,
                             updates=updates,
                             episodes_per_batch=int(ctx.extra.get(
                                 "fl5_episodes_per_batch", 1)),
                             seed=int(ctx.root_seed), stage=stage.stage_id)
        result = run_fast_state_arm(
            cfg, bundle, module, segments,
            os.path.join(ctx.stage_dir(stage.stage_id), arm_id.lower()),
            backbone_parameters=trainable.parameters)

        def factory(_ep, _module=module, _arm=arm_id):
            if _arm == ARM_FAST_STATE_ON:
                return FastStateCallbacks(_module, gate=gate)
            return FastStateOffCallbacks()

        records[arm_id] = {
            mode: _s.dev_records(bundle, dev_eps, ctx=ctx,
                                 arm_tag=f"{arm_id}_{mode}",
                                 callbacks_factory=factory, context_mode=mode)
            for mode in ("history", "context_reset")}
        arms[arm_id] = {
            "arm_result": result.to_dict(),
            "trainable": trainable.to_dict(),
            "fast_state_config": module.config(),
            "fast_state_config_hash": module.config_hash(),
            "dev": {mode: _s.dev_metrics_dict(recs, f"{arm_id}_{mode}")
                    for mode, recs in records[arm_id].items()},
            "prefix_used": {mode: [bool(r.get("prefix_used")) for r in recs]
                            for mode, recs in records[arm_id].items()},
            **comparability,
        }

        if arm_id == ARM_FAST_STATE_ON:
            # EVAL-ONLY control on the SAME trained bundle and the SAME trained
            # module: s pinned to 0, so the injection is the learned static
            # prefix and nothing else.  No third training run.
            def s0_factory(_ep, _module=module):
                return FastStateS0Callbacks(_module)

            control = ARM_FAST_STATE_ON_S0
            records[control] = {
                mode: _s.dev_records(bundle, dev_eps, ctx=ctx,
                                     arm_tag=f"{control}_{mode}",
                                     callbacks_factory=s0_factory,
                                     context_mode=mode)
                for mode in ("history", "context_reset")}
            arms[control] = {
                "eval_only": True,
                "control_of": ARM_FAST_STATE_ON,
                "trained_by": ARM_FAST_STATE_ON,
                "state_pinned_to_zero": True,
                "isolates": ("learned static prefix (prefix-tuning confound); "
                             "ON - ON_S0 is the state-update contribution, "
                             "ON_S0 - OFF is the static prefix plus the extra "
                             "parameters"),
                "arm_result": None,
                "trainable": trainable.to_dict(),
                "fast_state_config": module.config(),
                "fast_state_config_hash": module.config_hash(),
                "dev": {mode: _s.dev_metrics_dict(recs, f"{control}_{mode}")
                        for mode, recs in records[control].items()},
                "prefix_used": {mode: [bool(r.get("prefix_used")) for r in recs]
                                for mode, recs in records[control].items()},
                **comparability,
            }

    from foundation_learner.evaluation import metrics as _metrics

    persistence = _s.paired_persistence_evidence(
        records[ARM_FAST_STATE_ON]["context_reset"],
        records[ARM_FAST_STATE_OFF]["context_reset"],
        label="FL5", seed=int(ctx.root_seed))
    _s.publish_persistence_evidence(ctx, persistence, key="fl5_persistence_evidence")
    retained = {
        arm: _metrics.context_reset_persistence(recs["context_reset"],
                                                recs["history"])
        for arm, recs in records.items()}

    state_update_contribution = _s.paired_persistence_evidence(
        records[ARM_FAST_STATE_ON]["context_reset"],
        records[ARM_FAST_STATE_ON_S0]["context_reset"],
        label="FL5_ON_minus_ON_S0", seed=int(ctx.root_seed))
    static_prefix_contribution = _s.paired_persistence_evidence(
        records[ARM_FAST_STATE_ON_S0]["context_reset"],
        records[ARM_FAST_STATE_OFF]["context_reset"],
        label="FL5_ON_S0_minus_OFF", seed=int(ctx.root_seed))

    payload = _s.mechanism_payload(
        stage, _s.STATUS_COMPLETE,
        arms=arms,
        updates=updates, learning_rate=learning_rate,
        n_train_episodes=len(train_eps), n_dev_episodes=len(dev_eps),
        gated=bool(gate is not None),
        macro_aulc={arm: {mode: _s.macro_aulc(recs)
                          for mode, recs in by_mode.items()}
                    for arm, by_mode in records.items()},
        context_reset_persistence=retained,
        persistence_evidence=persistence,
        control_arm=ARM_FAST_STATE_ON_S0,
        state_update_contribution=state_update_contribution,
        static_prefix_contribution=static_prefix_contribution,
        **comparability,
    )
    return _s.finish_stage(ctx, stage, payload)
