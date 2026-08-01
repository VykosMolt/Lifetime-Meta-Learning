#!/usr/bin/env python3
"""Capture real adapter-induced L3_24 updates and reconstruct both d4 PCs."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT = Path("/home/moloch/ouro_project").resolve()
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
RUN = PROJECT / "o1_runs/O1_REAL_001"
MODEL = PROJECT / "models/ouro_rltt_local"
CAUSAL_CHECKPOINT = (
    PROJECT
    / "artifacts/reports/probes/bg_causal_intervention_adapter_2026-05-18/"
    "adapter_checkpoints/best_adapter.pt"
)
SEQUENCE_CHECKPOINT = (
    PROJECT
    / "artifacts/reports/probes/bg_sequence_level_adapter_2026-05-18/"
    "sequence_adapter_checkpoints/best_sequence_adapter.pt"
)
REFERENCE = RUN / "axis_source_bundle_v1/D4_REFERENCE_SET.jsonl"
REFERENCE_MANIFEST = (
    RUN / "axis_source_bundle_v1/D4_REFERENCE_SET_MANIFEST.json"
)
SEQUENCE_HOOK_CODE = PROJECT / "src/evaluator/bg_sequence_adapter.py"
CAUSAL_ADAPTER_CODE = PROJECT / "src/evaluator/bg_causal_adapter.py"
D_MODEL = 2048
CAPTURE_LOOP = 3
CAPTURE_LAYER = 24
CAUSAL_ALPHA = 0.01
SEQUENCE_ALPHA = 0.02
INTERVENTION_MODE = "multi_loop_decayed"
SEED = 20260518
MIN_COMPONENT_COSINE = 0.90

EXPECTED = {
    CAUSAL_CHECKPOINT: "f0831a9abf846cdf702d1211da346e2de3d32fe2ce11275247bad8400af215f4",
    SEQUENCE_CHECKPOINT: "75140bf3af7dfba4ee6851a500693b019f884e25dad0a2112f493dc99630597c",
    REFERENCE: "d145955fdd15a7c413fae26276d4a61a0d4eab05f5241b73365982e4ed49c75b",
    SEQUENCE_HOOK_CODE: "dfe73d69092cf02e2404358387239249a9e0bb0cce004cb1da8c9b80cbff4db1",
    CAUSAL_ADAPTER_CODE: "1c232c81290fec647153c6e98a77ab20980672a55039513af8ec8e7797d872c0",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def model_tree_hash() -> str:
    rows = []
    for path in sorted(child for child in MODEL.rglob("*") if child.is_file()):
        relative = path.relative_to(PROJECT).as_posix()
        rows.append(f"{relative}\0{sha256_file(path)}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def verify_sources() -> dict[str, str]:
    observed = {str(path): sha256_file(path) for path in EXPECTED}
    for path, expected in EXPECTED.items():
        if observed[str(path)] != expected:
            raise RuntimeError(
                f"frozen d4 source mismatch for {path}: "
                f"expected {expected}, got {observed[str(path)]}"
            )
    tree = model_tree_hash()
    if tree != "7dbbe4ef91de369bd712bad587c1af4f7c5dc42deff5377d4d70086c4c08a802":
        raise RuntimeError(f"Ouro-RLTT checkpoint tree hash mismatch: {tree}")
    observed[str(MODEL)] = tree
    return observed


def fixed_settings() -> dict[str, Any]:
    return {
        "schema_version": "o1.d4_capture_precommit.v1.1",
        "model_path": str(MODEL),
        "model_tree_sha256": "7dbbe4ef91de369bd712bad587c1af4f7c5dc42deff5377d4d70086c4c08a802",
        "model_dtype": "torch.bfloat16",
        "attn_implementation": "eager",
        "device": "cuda:0",
        "required_gpu_name_fragment": "5070 Ti",
        "total_ut_steps": 4,
        "early_exit_threshold": 1.0,
        "capture_locus": "L3_24",
        "capture_loop_1indexed": CAPTURE_LOOP,
        "capture_physical_layer": CAPTURE_LAYER,
        "capture_tensor": "post-block output of model.model.layers[23]",
        "capture_position": "final non-padding prompt token",
        "trajectory_phase": "prompt_only",
        "pooling": "none",
        "intervention_target_layer": 36,
        "intervention_mode": INTERVENTION_MODE,
        "intervention_loop_scales": {
            "1": 0.25,
            "2": 0.50,
            "3": 0.75,
            "4": 1.0,
        },
        "causal_adapter_alpha": CAUSAL_ALPHA,
        "sequence_adapter_alpha": SEQUENCE_ALPHA,
        "alpha_reason": "each checkpoint's saved best validation alpha; PC scale is normalized after capture",
        "reference_jsonl": str(REFERENCE),
        "reference_jsonl_sha256": EXPECTED[REFERENCE],
        "reference_selection_uses_o1_outcomes": False,
        "pc_centering": "subtract reference-set mean induced update per adapter",
        "pc_sign_rule": "positive dot with that adapter's mean induced update",
        "component_normalization": "unit RMS",
        "minimum_absolute_component_cosine": MIN_COMPONENT_COSINE,
        "random_seed": SEED,
        "transformers_required": "4.54.1",
    }


def prepare_precommit(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite d4 capture precommit: {path}")
    observed = verify_sources()
    settings = fixed_settings()
    settings["created_at"] = dt.datetime.now().astimezone().isoformat()
    settings["source_hashes"] = observed
    settings["capture_script_sha256"] = sha256_file(Path(__file__).resolve())
    settings["settings_sha256"] = canonical_json_hash(fixed_settings())
    path.write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(settings, indent=2, sort_keys=True))


def read_reference() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in REFERENCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 8:
        raise ValueError(f"d4 reference set must contain 8 tasks, got {len(rows)}")
    return rows


def set_determinism() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def force_four_loops(model: Any) -> None:
    inner = getattr(model, "model", None)
    for object_ in (model, inner):
        if object_ is None:
            continue
        if hasattr(object_, "total_ut_steps"):
            object_.total_ut_steps = 4
        config = getattr(object_, "config", None)
        if config is not None and hasattr(config, "total_ut_steps"):
            config.total_ut_steps = 4
        if config is not None and hasattr(config, "early_exit_threshold"):
            config.early_exit_threshold = 1.0


class LocusCapture:
    def __init__(self, model: Any, position: int) -> None:
        self.model = model
        self.position = int(position)
        self.handle: Any | None = None
        self.call_count = 0
        self.loop_source = ""
        self.rows: list[torch.Tensor] = []

    def _hook(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
        output: Any,
    ) -> Any:
        self.call_count += 1
        if kwargs and "current_ut" in kwargs:
            value = kwargs["current_ut"]
            if isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().item())
            loop = int(value) + 1
            self.loop_source = "current_ut"
        else:
            loop = ((self.call_count - 1) % 4) + 1
            self.loop_source = "validated_call_counter"
        if loop == CAPTURE_LOOP:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                raise RuntimeError("L3_24 capture output is not a [batch,seq,hidden] tensor")
            if tensor.shape[0] != 1 or tensor.shape[-1] != D_MODEL:
                raise RuntimeError(f"unexpected L3_24 tensor shape {tuple(tensor.shape)}")
            pos = max(0, min(self.position, int(tensor.shape[1]) - 1))
            self.rows.append(tensor[0, pos].detach().float().cpu().clone())
        return output

    def __enter__(self) -> "LocusCapture":
        module = self.model.model.layers[CAPTURE_LAYER - 1]
        try:
            self.handle = module.register_forward_hook(
                lambda mod, args, kwargs, out: self._hook(
                    mod, args, kwargs, out
                ),
                with_kwargs=True,
            )
        except TypeError:
            self.handle = module.register_forward_hook(
                lambda mod, args, out: self._hook(mod, args, None, out)
            )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def vector(self) -> torch.Tensor:
        if len(self.rows) != 1:
            raise RuntimeError(
                f"expected exactly one L3_24 capture, got {len(self.rows)} "
                f"over {self.call_count} layer calls"
            )
        return self.rows[0]


def run_capture(
    model: Any,
    encoded: dict[str, torch.Tensor],
    position: int,
    *,
    adapter: torch.nn.Module | None,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if adapter is None:
        hook_context: Any = contextlib.nullcontext(None)
    else:
        from src.evaluator.bg_sequence_adapter import sequence_adapter_hook

        hook_context = sequence_adapter_hook(
            model,
            adapter,
            alpha=float(alpha),
            intervention_mode=INTERVENTION_MODE,
            position=int(position),
            max_rms_fraction=0.02,
            max_alpha=0.02,
        )
    with LocusCapture(model, position) as capture:
        with hook_context as active:
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            diagnostics = active.diagnostics() if active is not None else {}
        vector = capture.vector()
    diagnostics["capture_layer_calls"] = capture.call_count
    diagnostics["capture_loop_index_source"] = capture.loop_source
    return vector, diagnostics


def unit_rms(vector: np.ndarray) -> np.ndarray:
    row = np.asarray(vector, dtype=np.float64).reshape(-1)
    if row.shape != (D_MODEL,) or not np.isfinite(row).all():
        raise ValueError(f"invalid d4 PC geometry {row.shape}")
    rms = float(np.sqrt(np.mean(row * row)))
    if rms <= 1e-12:
        raise ValueError(f"d4 PC has zero/invalid RMS {rms}")
    return np.ascontiguousarray(row / rms, dtype="<f8")


def leading_update_pc(updates: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(updates, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != D_MODEL:
        raise ValueError(f"invalid update matrix shape {matrix.shape}")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    pc = vh[0]
    mean_update = matrix.mean(axis=0)
    if float(pc @ mean_update) < 0:
        pc = -pc
    axis = unit_rms(pc)
    denom = float(np.sum(singular * singular))
    return axis, {
        "update_matrix_shape": list(matrix.shape),
        "update_matrix_finite": bool(np.isfinite(matrix).all()),
        "update_rms_min": float(
            np.min(np.sqrt(np.mean(matrix * matrix, axis=1)))
        ),
        "update_rms_max": float(
            np.max(np.sqrt(np.mean(matrix * matrix, axis=1)))
        ),
        "leading_singular_values": [float(value) for value in singular[:5]],
        "leading_explained_variance": (
            float(singular[0] ** 2 / denom) if denom > 0 else None
        ),
        "centering": "reference-set mean induced update subtracted",
        "sign_rule": "positive dot with uncentered mean induced update",
    }


def load_adapter(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    from src.evaluator.bg_causal_adapter import LowRankDeltaAdapter

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    rank = int(checkpoint.get("adapter_rank", 32))
    adapter = LowRankDeltaAdapter(rank=rank)
    missing, unexpected = adapter.load_state_dict(
        checkpoint["adapter_state_dict"], strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"adapter state mismatch: missing={missing}, unexpected={unexpected}")
    adapter.to(device).eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter, checkpoint


def save_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def capture(precommit: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite d4 capture output: {output}")
    if not precommit.is_file():
        raise FileNotFoundError(f"d4 capture precommit missing: {precommit}")
    committed = json.loads(precommit.read_text(encoding="utf-8"))
    if committed.get("settings_sha256") != canonical_json_hash(fixed_settings()):
        raise RuntimeError("d4 capture settings differ from precommit")
    if committed.get("capture_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("d4 capture script differs from precommit")
    sources = verify_sources()
    if str(__import__("transformers").__version__) != "4.54.1":
        raise RuntimeError("transformers must remain exactly 4.54.1")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("d4 capture requires visible CUDA device 0")
    gpu_name = torch.cuda.get_device_name(0)
    if "5070 Ti" not in gpu_name:
        raise RuntimeError(f"d4 capture requires the real 5070 Ti, got {gpu_name!r}")
    set_determinism()
    output.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL),
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda:0")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    force_four_loops(model)
    causal, causal_meta = load_adapter(CAUSAL_CHECKPOINT, device)
    sequence, sequence_meta = load_adapter(SEQUENCE_CHECKPOINT, device)
    reference = read_reference()

    baseline_rows: list[np.ndarray] = []
    causal_rows: list[np.ndarray] = []
    sequence_rows: list[np.ndarray] = []
    task_diagnostics: list[dict[str, Any]] = []
    zero_alpha_checked = False
    for index, task in enumerate(reference):
        encoded = tokenizer(
            task["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=1536,
            padding=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        position = int(encoded["attention_mask"][0].sum().item()) - 1
        baseline, base_diag = run_capture(
            model, encoded, position, adapter=None, alpha=0.0
        )
        if not zero_alpha_checked:
            zero, zero_diag = run_capture(
                model, encoded, position, adapter=causal, alpha=0.0
            )
            zero_max_abs = float(torch.max(torch.abs(zero - baseline)).item())
            if zero_max_abs != 0.0:
                raise RuntimeError(
                    f"zero-alpha d4 capture is not bit exact: {zero_max_abs}"
                )
            zero_alpha_checked = True
        else:
            zero_diag = {}
            zero_max_abs = None
        causal_state, causal_diag = run_capture(
            model, encoded, position, adapter=causal, alpha=CAUSAL_ALPHA
        )
        sequence_state, sequence_diag = run_capture(
            model, encoded, position, adapter=sequence, alpha=SEQUENCE_ALPHA
        )
        baseline_rows.append(baseline.numpy())
        causal_rows.append((causal_state - baseline).numpy())
        sequence_rows.append((sequence_state - baseline).numpy())
        task_diagnostics.append(
            {
                "reference_rank": index,
                "task_id": task["task_id"],
                "prompt_tokens": int(encoded["input_ids"].shape[1]),
                "capture_position": position,
                "baseline_capture": base_diag,
                "zero_alpha_capture": zero_diag,
                "zero_alpha_max_abs": zero_max_abs,
                "causal_capture": causal_diag,
                "sequence_capture": sequence_diag,
            }
        )
        print(
            f"[{index + 1}/{len(reference)}] {task['task_id']} "
            f"tokens={encoded['input_ids'].shape[1]}",
            flush=True,
        )
        del encoded, baseline, causal_state, sequence_state
        torch.cuda.empty_cache()

    baseline_matrix = np.stack(baseline_rows).astype("<f4")
    causal_updates = np.stack(causal_rows).astype("<f4")
    sequence_updates = np.stack(sequence_rows).astype("<f4")
    component1, component1_meta = leading_update_pc(causal_updates)
    component2, component2_meta = leading_update_pc(sequence_updates)
    cosine = float(np.dot(component1, component2) / D_MODEL)
    abs_cosine = abs(cosine)
    saved = {
        "baseline_states": save_npy(
            output / "baseline_l3_24.npy", baseline_matrix
        ),
        "causal_induced_updates": save_npy(
            output / "d4_adapter1_induced_updates_l3_24.npy", causal_updates
        ),
        "sequence_induced_updates": save_npy(
            output / "d4_adapter2_induced_updates_l3_24.npy", sequence_updates
        ),
        "component1": save_npy(
            output / "d4_adapter1_l3_24.npy", component1
        ),
        "component2": save_npy(
            output / "d4_adapter2_l3_24.npy", component2
        ),
    }
    report = {
        "schema_version": "o1.d4_induced_update_capture.v1.1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "scientific_scope": "real pre-O1 adapter capture; no O1 calibration or confirmatory outcomes",
        "status": (
            "RECONSTRUCTED"
            if abs_cosine >= MIN_COMPONENT_COSINE
            else "BLOCKED_COMPONENT_COSINE_BELOW_0.90"
        ),
        "source_hashes": sources,
        "precommit_path": str(precommit),
        "precommit_sha256": sha256_file(precommit),
        "settings": fixed_settings(),
        "runtime": {
            "gpu_name": gpu_name,
            "gpu_uuid": torch.cuda.get_device_properties(0).uuid,
            "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "model_parameter_device": str(next(model.parameters()).device),
            "attn_implementation": str(model.config._attn_implementation),
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "adapter_checkpoint_metadata": {
            "causal": {
                "epoch": causal_meta.get("epoch"),
                "saved_best_alpha": causal_meta.get("epoch_log", {})
                .get("best_val", {})
                .get("alpha"),
                "target_layer": causal_meta.get("target_layer"),
                "intervention_mode": causal_meta.get("intervention_mode"),
            },
            "sequence": {
                "update": sequence_meta.get("update"),
                "saved_best_alpha": sequence_meta.get("best_val_alpha"),
                "target_layer": sequence_meta.get("target_layer"),
                "intervention_mode": sequence_meta.get("intervention_mode"),
                "init_source": sequence_meta.get("init_source"),
            },
            "independence_caveat": "different objectives and optimization runs; sequence checkpoint records initialization from the causal checkpoint",
        },
        "component1": component1_meta,
        "component2": component2_meta,
        "signed_component_cosine": cosine,
        "absolute_component_cosine": abs_cosine,
        "minimum_required_cosine": MIN_COMPONENT_COSINE,
        "task_diagnostics": task_diagnostics,
        "saved_artifacts": saved,
    }
    report_path = output / "D4_CAPTURE_REPORT.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if abs_cosine >= MIN_COMPONENT_COSINE else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", required=True)
    parser.add_argument("--output")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    precommit = Path(args.precommit).resolve()
    if args.prepare:
        if args.output:
            raise ValueError("--output is not used with --prepare")
        prepare_precommit(precommit)
        return 0
    if not args.output:
        raise ValueError("--output is required for capture")
    return capture(precommit, Path(args.output).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
