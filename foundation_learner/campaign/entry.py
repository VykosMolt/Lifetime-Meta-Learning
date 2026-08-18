"""The PRODUCTION campaign entry (contract §13, §22).

``session_supervisor.main()`` used to construct a supervisor with
``context_factory=None`` and ``ladder_runner=None``, so the pod entry
(``deploy/fl_b200_entry.sh``) reached ``RUN_FL_LADDER`` and refused: the only
code that ever built a :class:`~campaign.stage_definitions.StageContext` lived
in the dress rehearsal and in the tests (review finding R-C1).  This module is
that missing wiring, and it is the ONLY place where a stage context is built.

What it does, in order:

1. :func:`configure_determinism` — contract §22's determinism knobs
   (``torch.use_deterministic_algorithms(True)``, cuDNN deterministic, cuDNN
   benchmark off).  They are applied at CAMPAIGN START, for real runs and
   rehearsals alike, and the observed flags are recorded so the claim is
   checkable rather than asserted (review finding R-M9).
2. :func:`build_stage_context` — the real context: a FRESH-load bundle factory
   bound to the session's checkpoint, the pre-generation root, the isolation
   guard, the throughput table, the scope/seed/token budget, and the caps.
3. :func:`run_ladder` — the scheduler over the frozen §11 stage table.

**The real and the rehearsal path are the same code.**  The ONLY difference is
the bundle SOURCE: a real session loads the frozen 2.6 B backbone with
``verify_tree_hash=True`` (§13 reloads a PRISTINE checkpoint after O1 closes);
a session explicitly marked ``rehearsal`` may set ``fl_tiny_model: true`` to
build the tiny NONSCIENTIFIC model instead, which every artefact then says.
Nothing else branches on ``rehearsal`` here, so walking ``main()`` with a
rehearsal config exercises the production wiring itself.
"""
from __future__ import annotations

import os
from typing import Any

from ..training.arms import ROOT_SEED_PRIMARY
from .affordability import PEFT_MODE, SCOPES
from .scheduler import Scheduler
from .stage_definitions import StageContext, StageError

__all__ = [
    "EntryError",
    "DETERMINISM_SCHEMA",
    "configure_determinism",
    "build_bundle_factory",
    "build_stage_context",
    "run_ladder",
    "attach_to_supervisor",
]

DETERMINISM_SCHEMA = "flb200.determinism_record.v1"

#: Caps below the frozen per-stage evaluation maxima reduce the amount of
#: evidence collected, so only an explicitly labelled rehearsal may set them.
_REHEARSAL_ONLY_KEYS = ("fl_eval_episode_cap", "fl_train_example_cap")


class EntryError(RuntimeError):
    """The campaign entry refused to build a runnable context."""


# --------------------------------------------------------------------------
# determinism (contract §22)
# --------------------------------------------------------------------------

def configure_determinism(*, label: str = "FL_CAMPAIGN") -> dict:
    """Apply and RECORD the §22 determinism configuration.

    Called once at campaign start.  The record carries the flags as torch
    actually reports them afterwards, so a journal entry claiming determinism
    can be checked against the runtime instead of being taken on trust.
    """
    from ..training import model_loading

    model_loading.set_deterministic_eval()
    import torch

    record = {
        "schema": DETERMINISM_SCHEMA,
        "label": str(label),
        "called": "foundation_learner.training.model_loading.set_deterministic_eval",
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_version": torch.__version__,
    }
    if not record["deterministic_algorithms"]:      # pragma: no cover - torch
        raise EntryError(
            "torch.use_deterministic_algorithms(True) did not take effect; the "
            "campaign refuses to run evaluation paths nondeterministically "
            "(contract §22)")
    return record


# --------------------------------------------------------------------------
# the bundle factory (the ONLY thing a rehearsal changes)
# --------------------------------------------------------------------------

def build_bundle_factory(payload: dict, *, rehearsal: bool):
    """Return ``(factory, description)`` for a FRESH bundle per arm (§7)."""
    checkpoint_dir = str(payload["checkpoint_dir"])
    device = str(payload.get("fl_device") or ("cpu" if rehearsal else "cuda"))
    tiny = bool(payload.get("fl_tiny_model", False))
    seed = int(payload.get("fl_root_seed", ROOT_SEED_PRIMARY))

    if tiny and not rehearsal:
        raise EntryError(
            "fl_tiny_model is set on a session that is NOT marked rehearsal; "
            "the tiny model is a mechanics stand-in and may never carry a "
            "scientific result (contract §18/§20)")

    if tiny:
        from ..training.tiny_model import build_tiny_model

        def factory():
            return build_tiny_model(seed=seed, device=device)

        description = {
            "source": "TINY_NONSCIENTIFIC_MODEL",
            "device": device,
            "seed": seed,
            "warning": ("DRESS REHEARSAL: hidden 64 / 2 layers / 2 ut steps, "
                        "random init. Mechanics only; never a scientific "
                        "result."),
        }
        return factory, description

    from ..training.model_loading import load_frozen_backbone

    def factory():
        # contract §13: FL starts from a FRESH reload of the PRISTINE
        # checkpoint, tree-hash verified, never from an O1-mutated state
        return load_frozen_backbone(checkpoint_dir, device,
                                    verify_tree_hash=True)

    description = {
        "source": "FROZEN_BACKBONE",
        "checkpoint_dir": checkpoint_dir,
        "device": device,
        "verify_tree_hash": True,
        "recipe": "training.model_loading.load_frozen_backbone (contract §22)",
    }
    return factory, description


# --------------------------------------------------------------------------
# the stage context
# --------------------------------------------------------------------------

def build_stage_context(supervisor: Any) -> StageContext:
    """Build the real :class:`StageContext` from the session configuration."""
    payload = supervisor.payload
    rehearsal = bool(supervisor.rehearsal)

    for key in _REHEARSAL_ONLY_KEYS:
        if payload.get(key) is not None and not rehearsal:
            raise EntryError(
                f"{key} shortens the frozen per-stage evaluation maxima; only "
                "an explicitly labelled dress rehearsal may set it (contract "
                "§9, Amendment 10 item 5)")

    scope = str(payload.get("fl_scope") or PEFT_MODE)
    if scope not in SCOPES:
        raise EntryError(f"fl_scope={scope!r} is not one of {list(SCOPES)}")

    pregen_root = str(payload["pregen_root"])
    resolved_pregen = supervisor.guard.guard(pregen_root, "read")
    if not os.path.isdir(resolved_pregen):
        raise EntryError(
            f"pregen_root {pregen_root!r} does not exist; the B200 never "
            "generates ordinary data (contract §16) — stage the pre-generated "
            "shards before the session")

    factory, bundle_description = build_bundle_factory(payload,
                                                       rehearsal=rehearsal)
    extra = dict(payload.get("fl_extra") or {})
    extra.setdefault("bundle_source", bundle_description)
    ctx = StageContext(
        out_dir=os.path.join(supervisor.out_dir, "ladder"),
        pregen_root=resolved_pregen,
        bundle_factory=factory,
        guard=supervisor.guard,
        scope=scope,
        root_seed=int(payload.get("fl_root_seed", ROOT_SEED_PRIMARY)),
        max_tokens_per_batch=int(payload.get("fl_max_tokens_per_batch", 2048)),
        eval_episode_cap=payload.get("fl_eval_episode_cap"),
        train_example_cap=payload.get("fl_train_example_cap"),
        rehearsal=rehearsal,
        label=supervisor.label(),
        checkpoint_dir=str(payload["checkpoint_dir"]),
        device=str(bundle_description["device"]),
        extra=extra,
    )
    if getattr(supervisor, "durability", None) is not None:
        ctx.extra["durability_mirror"] = supervisor.durability
    ctx.extra.setdefault("stage_states", {})
    ctx.extra.setdefault("campaign_hashes", {
        "checkpoint_tree_sha256": payload.get("checkpoint_tree_sha256"),
        "pregen_root_basename": os.path.basename(resolved_pregen),
    })
    return ctx


def run_ladder(scheduler: Scheduler, ctx: StageContext) -> dict:
    """Run the frozen §11 ladder (the supervisor's ``ladder_runner``)."""
    return scheduler.run_ladder(ctx)


def attach_to_supervisor(supervisor: Any) -> dict:
    """Wire the production context factory + ladder runner onto a supervisor.

    Idempotent and non-destructive: an explicitly injected factory or runner
    (the dress rehearsal, the unit tests) is left alone, so the injection point
    stays testable while the DEFAULT is the real thing.
    """
    record = configure_determinism(label=supervisor.label())
    if supervisor.context_factory is None:
        supervisor.context_factory = build_stage_context
    if supervisor.ladder_runner is None:
        supervisor.ladder_runner = run_ladder
    supervisor.journal("CAMPAIGN_ENTRY_WIRED", "START_SESSION", {
        "determinism": record,
        "context_factory": getattr(supervisor.context_factory, "__name__",
                                   str(supervisor.context_factory)),
        "ladder_runner": getattr(supervisor.ladder_runner, "__name__",
                                 str(supervisor.ladder_runner)),
    })
    supervisor.determinism = record
    return record
