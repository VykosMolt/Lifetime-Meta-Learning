#!/usr/bin/env python3
"""Sealed Ouro-RLTT branch generator for O1 v2.

This module is deliberately generation-only: it never chooses tasks, axes,
alpha, strata, or arms.  It consumes sealed rows and writes branch records.
Physical layer 24 is ``model.model.layers[23]``.  The edit is one-shot at the
last unpadded prompt token during prefill, zero-based loop 2.  Transport is
captured at the normalized L4_47 loop boundary before sampling.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

# Strict set, not setdefault: a pre-set environment value must not silently
# override the sealed deterministic configuration (v2.1 hardening).
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from o1_answer_parser_v2 import has_catastrophic_repetition, parse_final_answer
from o1_prompt_template_v2 import render_prompt
from o1_truth_table_verifier_v2 import verify_letter

LOOP_INDEX = 2
PHYSICAL_LAYER = 24
MODULE_INDEX = 23
HIDDEN = 2048
LOOPS = 4
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 448


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).mean(dim=-1, keepdim=True).sqrt()


class OneShotResidualEdit:
    def __init__(self, direction: np.ndarray | None, signed_alpha: float):
        self.direction = direction
        self.signed_alpha = float(signed_alpha)
        self.applied = 0
        self.injected_rms = 0.0

    def __call__(self, _module, _args, kwargs, output):
        current_ut = kwargs.get("current_ut")
        if current_ut is None or int(current_ut) != LOOP_INDEX:
            return output
        self.applied += 1
        # The zero-alpha path is an identity return, not clone-plus-zero.
        # DOCUMENTED SCOPE (v2.1): zero-alpha is therefore a registered-hook /
        # zero-delta sham. It proves hook registration and pipeline determinism
        # (bitwise parity with baseline is enforced downstream) but does NOT
        # exercise the alpha != 0 clone/cast/add arithmetic below; only real
        # intervention arms execute that branch.
        if self.direction is None or self.signed_alpha == 0.0:
            return output
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[-1] != HIDDEN:
            raise RuntimeError(f"unexpected L3_24 output shape {tuple(hidden.shape)}")
        direction = torch.as_tensor(
            self.direction, device=hidden.device, dtype=hidden.dtype,
        )
        if not torch.isfinite(direction).all():
            raise RuntimeError("non-finite direction")
        target = hidden[:, -1, :]
        delta = self.signed_alpha * _rms(target).to(hidden.dtype) * direction
        changed = hidden.clone()
        changed[:, -1, :] = target + delta
        # injected_rms is the INTENDED |s*alpha*r*d| RMS computed in float32
        # from the bfloat16 delta product. The realized in-graph update
        # (target+delta at bf16) deviates elementwise by bf16 rounding
        # (O(0.3%)). The intended value is the frozen transport denominator;
        # this asymmetry is documented, symmetric across intervention arms,
        # and small against the factor-2 alpha grid (v2.1 documentation).
        self.injected_rms = float(_rms(delta).item())
        if isinstance(output, tuple):
            return (changed, *output[1:])
        if isinstance(output, list):
            return [changed, *output[1:]]
        return changed


def _sample_top_p(logits: torch.Tensor, seed: int) -> int:
    scores = logits.float() / TEMPERATURE
    sorted_scores, sorted_idx = torch.sort(scores, descending=True)
    probs = torch.softmax(sorted_scores, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative - probs >= TOP_P
    probs = probs.masked_fill(remove, 0.0)
    probs = probs / probs.sum()
    generator = torch.Generator(device=probs.device)
    generator.manual_seed(int(seed))
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sorted_idx[sampled].item())


def _load_model(checkpoint: Path):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if transformers.__version__ != "4.54.1":
        raise RuntimeError(
            f"transformers must equal 4.54.1, got {transformers.__version__}"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tok = AutoTokenizer.from_pretrained(
        str(checkpoint), trust_remote_code=True, local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager",
    ).to("cuda:0")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.total_ut_steps = LOOPS
    model.config.early_exit_threshold = 1.0
    model.model.total_ut_steps = LOOPS
    return model, tok


def generate_branch(model, tokenizer, task: dict, seed: int,
                    direction: np.ndarray | None, signed_alpha: float) -> dict:
    prompt = render_prompt(task)
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1536,
        padding=False,
    )
    input_ids = enc["input_ids"].to("cuda:0")
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to("cuda:0")
    hook = OneShotResidualEdit(direction, signed_alpha)
    handle = model.model.layers[MODULE_INDEX].register_forward_hook(
        hook, with_kwargs=True,
    )
    try:
        with torch.inference_mode():
            out = model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=True, logits_to_keep=1,
                return_per_loop_hidden_states=True,
            )
    finally:
        handle.remove()
    if hook.applied != 1:
        raise RuntimeError(f"expected one L3_24 edit visit, got {hook.applied}")
    boundary = out.per_loop_hidden_states[3][:, -1, :].detach().float().cpu()
    logits = out.logits[0, -1]
    cache = out.past_key_values
    generated: list[int] = []
    text = ""
    finish_reason = "max_new_tokens"
    logits_finite = bool(torch.isfinite(logits).all().item())
    for step in range(MAX_NEW_TOKENS):
        if not logits_finite:
            finish_reason = "nonfinite_logits"
            break
        token = _sample_top_p(logits, int(seed) + step)
        generated.append(token)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if parse_final_answer(text) is not None:
            finish_reason = "parseable_final_answer"
            break
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            finish_reason = "eos_without_commitment"
            break
        next_id = torch.tensor([[token]], device="cuda:0", dtype=torch.long)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device="cuda:0", dtype=attention_mask.dtype)],
            dim=1,
        )
        with torch.inference_mode():
            out = model(
                input_ids=next_id, attention_mask=attention_mask,
                past_key_values=cache, use_cache=True, logits_to_keep=1,
            )
        cache = out.past_key_values
        logits = out.logits[0, -1]
        logits_finite = logits_finite and bool(torch.isfinite(logits).all().item())
    parsed = parse_final_answer(text)
    return {
        "generated_token_ids": generated,
        "generated_text": text,
        "well_formed": parsed is not None,
        "verifier_correct": verify_letter(task, parsed),
        "parsed_answer": parsed,
        "finish_reason": finish_reason,
        "logits_finite": logits_finite,
        "catastrophic_repetition": has_catastrophic_repetition(generated),
        "injected_rms": hook.injected_rms,
        "_prefill_l4_47": boundary.numpy(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task-json", required=True)
    p.add_argument("--axis-npy")
    p.add_argument("--axis-row", type=int)
    p.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if (a.axis_npy is None) != (a.axis_row is None):
        raise ValueError("--axis-npy and --axis-row must be given together")
    task = json.loads(Path(a.task_json).read_text(encoding="utf-8"))
    direction = None
    if a.axis_npy:
        axes = np.load(a.axis_npy, allow_pickle=False)
        direction = np.asarray(axes[a.axis_row], dtype=np.float64)
        if direction.shape != (HIDDEN,) or not np.isfinite(direction).all():
            raise ValueError("invalid axis row")
        if not math.isclose(float(np.sqrt(np.mean(direction * direction))), 1.0,
                            rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("axis row is not unit RMS")
    model, tokenizer = _load_model(Path(a.checkpoint))
    result = generate_branch(
        model, tokenizer, task, a.seed, direction, a.sign * a.alpha,
    )
    result.pop("_prefill_l4_47")
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

