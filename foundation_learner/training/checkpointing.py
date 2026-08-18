"""Atomic, bitwise-resumable checkpoints (contract §15).

Every checkpoint records (§15): base checkpoint tree hash, training config
hash, optimizer state, trained module / full-model state, stage, step count,
token count, RNG states (python + numpy + torch + cuda), family-split hash,
training-shard hashes, code commit, environment digest.

On-disk layout for tag ``T`` under ``out_dir``::

    ckpt_T.pt              torch payload (tensors + the manifest dict)
    ckpt_T.manifest.json   flb200.checkpoint_manifest.v1 + payload_sha256

Both files are written tmp + fsync + rename (``os.replace``).  The sidecar is
written LAST and carries the payload's SHA-256, so a crash between the two
writes leaves a checkpoint that :func:`load_checkpoint` refuses (missing or
mismatched sidecar) instead of a silently truncated resume.

Retention tags (§15): ``initial``, ``best_dev``, ``final``, ``pre_promotion``,
plus cadence snapshots ``step{n}``.  Cadence itself (600 s / 200 steps) is
enforced by the trainer (§11).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from typing import Any, Mapping, Sequence

import torch

from foundation_learner.training.model_loading import (
    domain_sha256,
    environment_digest,
    sha256_file,
)

CHECKPOINT_MANIFEST_SCHEMA = "flb200.checkpoint_manifest.v1"

TAG_INITIAL = "initial"
TAG_BEST_DEV = "best_dev"
TAG_FINAL = "final"
TAG_PRE_PROMOTION = "pre_promotion"
RETAINED_TAGS = (TAG_INITIAL, TAG_BEST_DEV, TAG_FINAL, TAG_PRE_PROMOTION)


class CheckpointError(RuntimeError):
    """Raised on malformed checkpoint requests."""


class CorruptCheckpointError(CheckpointError):
    """Raised when a checkpoint fails hash verification (contract §17)."""


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(directory)


def atomic_write_json(path: str, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_torch_save(path: str, obj: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(directory)


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------


def capture_rng_state(extra_generators: Mapping[str, Any] | None = None) -> dict:
    """Capture every RNG stream needed for a bitwise resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy_legacy"] = np.random.get_state()
    except Exception:  # pragma: no cover - numpy is a declared dependency
        state["numpy_legacy"] = None
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    named: dict[str, Any] = {}
    for name, generator in (extra_generators or {}).items():
        if hasattr(generator, "bit_generator"):  # numpy Generator
            named[name] = {"kind": "numpy_generator", "state": generator.bit_generator.state}
        elif hasattr(generator, "get_state"):  # torch.Generator
            named[name] = {"kind": "torch_generator", "state": generator.get_state()}
        else:
            raise CheckpointError(f"unsupported generator type for {name!r}")
    state["named"] = named
    return state


def restore_rng_state(
    state: Mapping[str, Any], extra_generators: Mapping[str, Any] | None = None
) -> None:
    """Exact inverse of :func:`capture_rng_state`."""
    random.setstate(_as_python_state(state["python"]))
    torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if state.get("numpy_legacy") is not None:
        import numpy as np

        np.random.set_state(state["numpy_legacy"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [_as_byte_tensor(s) for s in state["torch_cuda"]]
        )
    named = state.get("named") or {}
    for name, generator in (extra_generators or {}).items():
        if name not in named:
            raise CheckpointError(f"checkpoint has no RNG state for generator {name!r}")
        entry = named[name]
        if entry["kind"] == "numpy_generator":
            generator.bit_generator.state = entry["state"]
        else:
            generator.set_state(_as_byte_tensor(entry["state"]))


def _as_byte_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.clone().to(torch.uint8).cpu()
    return torch.tensor(bytearray(value), dtype=torch.uint8)


def _as_python_state(value: Any) -> tuple:
    # torch.save/load round-trips tuples faithfully; JSON round-trips would not.
    if isinstance(value, tuple):
        return (value[0], tuple(value[1]), value[2])
    raise CheckpointError("python RNG state has an unexpected shape")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def code_commit(repo_hint: str | None = None) -> dict[str, Any]:
    """Read the current git commit for provenance (never fabricated)."""
    start = repo_hint or os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(
            ["git", "-C", start, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", start, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", start, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        return {"commit": commit, "branch": branch, "dirty": bool(dirty)}
    except Exception as exc:  # noqa: BLE001 - provenance failure must be visible
        return {"commit": None, "branch": None, "dirty": None, "error": repr(exc)}


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def checkpoint_paths(out_dir: str, tag: str) -> tuple[str, str]:
    return (
        os.path.join(out_dir, f"ckpt_{tag}.pt"),
        os.path.join(out_dir, f"ckpt_{tag}.manifest.json"),
    )


def save_checkpoint(
    out_dir: str,
    tag: str,
    *,
    trained_state: Mapping[str, torch.Tensor],
    trained_scope: str,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    tokens: int = 0,
    loss_tokens: int = 0,
    arm_config: Mapping[str, Any] | None = None,
    arm_config_hash: str | None = None,
    base_identity: Mapping[str, Any] | None = None,
    family_split_hash: str | None = None,
    shard_hashes: Mapping[str, str] | Sequence[str] | None = None,
    stage: str = "CORE",
    extra_generators: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one atomic checkpoint; return its manifest."""
    if not tag or "/" in tag:
        raise CheckpointError(f"invalid checkpoint tag {tag!r}")
    os.makedirs(out_dir, exist_ok=True)
    payload_path, manifest_path = checkpoint_paths(out_dir, tag)

    manifest: dict[str, Any] = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "tag": tag,
        "stage": stage,
        "step": int(step),
        "tokens": int(tokens),
        "loss_tokens": int(loss_tokens),
        "trained_scope": trained_scope,
        "trained_tensors": sorted(trained_state),
        "arm_config": dict(arm_config) if arm_config else None,
        "arm_config_hash": arm_config_hash,
        "base_identity": dict(base_identity) if base_identity else None,
        "base_checkpoint_tree_sha256": (
            (base_identity or {}).get("checkpoint_tree_sha256")
        ),
        "base_identity_hash": (base_identity or {}).get("identity_hash"),
        "family_split_hash": family_split_hash,
        "shard_hashes": (
            dict(shard_hashes)
            if isinstance(shard_hashes, Mapping)
            else (list(shard_hashes) if shard_hashes is not None else None)
        ),
        "code_commit": code_commit(),
        "environment": environment_digest(),
    }
    if extra:
        manifest["extra"] = dict(extra)
    manifest["manifest_hash"] = domain_sha256(
        "FL_V0_CHECKPOINT_MANIFEST",
        json.loads(json.dumps(manifest, sort_keys=True, default=str)),
    )

    payload = {
        "manifest": manifest,
        "trained_state": {k: v.detach().to("cpu").clone() for k, v in trained_state.items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": capture_rng_state(extra_generators),
    }
    atomic_torch_save(payload_path, payload)
    sidecar = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "payload_file": os.path.basename(payload_path),
        "payload_sha256": sha256_file(payload_path),
        **manifest,
    }
    atomic_write_json(manifest_path, json.loads(json.dumps(sidecar, default=str)))
    return sidecar


def load_checkpoint(out_dir: str, tag: str, *, verify: bool = True) -> dict[str, Any]:
    """Load and (by default) hash-verify a checkpoint."""
    payload_path, manifest_path = checkpoint_paths(out_dir, tag)
    if not os.path.isfile(payload_path):
        raise CheckpointError(f"no checkpoint payload at {payload_path}")
    if not os.path.isfile(manifest_path):
        raise CorruptCheckpointError(
            f"checkpoint {payload_path} has no manifest sidecar; the write did "
            "not complete and the checkpoint is not trustworthy"
        )
    with open(manifest_path, encoding="utf-8") as fh:
        sidecar = json.load(fh)
    if verify:
        got = sha256_file(payload_path)
        if got != sidecar.get("payload_sha256"):
            raise CorruptCheckpointError(
                f"checkpoint payload hash mismatch for {payload_path}: manifest "
                f"{sidecar.get('payload_sha256')} != actual {got}"
            )
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    payload["sidecar"] = sidecar
    return payload


def restore_training_state(
    payload: Mapping[str, Any],
    *,
    load_trained_state,
    optimizer=None,
    scheduler=None,
    extra_generators: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore parameters, optimizer, scheduler and every RNG stream."""
    load_trained_state(payload["trained_state"])
    if optimizer is not None:
        if payload.get("optimizer_state") is None:
            raise CheckpointError("checkpoint carries no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if payload.get("scheduler_state") is None:
            raise CheckpointError("checkpoint carries no scheduler state")
        scheduler.load_state_dict(payload["scheduler_state"])
    restore_rng_state(payload["rng_state"], extra_generators)
    return dict(payload["manifest"])


def list_checkpoints(out_dir: str) -> list[str]:
    if not os.path.isdir(out_dir):
        return []
    tags = []
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("ckpt_") and name.endswith(".pt"):
            tags.append(name[len("ckpt_") : -len(".pt")])
    return tags


def prune_checkpoints(out_dir: str, keep_tags: Sequence[str] = RETAINED_TAGS) -> list[str]:
    """Delete cadence snapshots that are not in ``keep_tags`` (§15 anti-spam).

    The retained tags are never touched.  Returns the removed tags.
    """
    removed = []
    keep = set(keep_tags)
    for tag in list_checkpoints(out_dir):
        if tag in keep:
            continue
        payload_path, manifest_path = checkpoint_paths(out_dir, tag)
        for path in (payload_path, manifest_path):
            if os.path.isfile(path):
                os.remove(path)
        removed.append(tag)
    return removed
