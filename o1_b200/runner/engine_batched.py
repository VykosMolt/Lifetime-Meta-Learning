"""Batched generation engine (B200_BATCHED core).

Reproduces the SEALED single-row generation semantics per batch member:

  * tokenizer call identical to the sealed generator (truncation, max_length
    1536, no per-row padding);
  * LEFT padding to the batch maximum, with correct attention masks — the
    final non-padding prompt position of every row is sequence position -1,
    exactly the position the sealed hook edits and the sealed transport reads;
  * one-shot per-row intervention during prefill (intervention.py);
  * per-row transport-boundary capture from per_loop_hidden_states[3][:, -1];
  * per-row sampling via the SEALED ``_sample_top_p`` with the sealed
    per-step generator seed (seed + step) — no shared RNG state exists, so
    batch composition cannot remap any row's stream;
  * per-row stopping with the sealed order of checks: nonfinite logits ->
    first parseable FINAL ANSWER -> EOS -> max_new_tokens;
  * per-row independent token histories, cache state, transport state, and
    record identity.

The correctness baseline is NON-COMPACTING: finished rows keep their batch
slot (their outputs are frozen and ignored) so active rows never change slot.
Dynamic compaction exists behind ``compaction=True`` (default OFF) and moves
row identity, cache slices, token history, mask, stopping and intervention
state together; it stays scientifically ineligible until it passes the future
B200 equivalence gates.
"""
from __future__ import annotations

import numpy as np
import torch

from . import sealed_import
from .intervention import BatchedInterventionError, BatchedOneShotResidualEdit, register

sealed_import.ensure_sealed_path()

from run_o1_v2_generation import (  # noqa: E402 (sealed)
    HIDDEN, MAX_NEW_TOKENS, _sample_top_p,
)
from o1_answer_parser_v2 import has_catastrophic_repetition, parse_final_answer  # noqa: E402


class BatchedEngineError(RuntimeError):
    pass


def _left_pad(token_lists: list[list[int]], pad_id: int, device) -> tuple:
    T = max(len(t) for t in token_lists)
    B = len(token_lists)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.long)
    for b, toks in enumerate(token_lists):
        L = len(toks)
        if L == 0:
            raise BatchedEngineError(f"row {b}: empty prompt")
        ids[b, T - L:] = torch.tensor(toks, dtype=torch.long)
        mask[b, T - L:] = 1
    return ids.to(device), mask.to(device)


def generate_batch(model, tokenizer, tasks: list[dict], seeds: list[int],
                   directions: list, signed_alphas: list[float],
                   *, compaction: bool = False,
                   max_new_tokens: int = MAX_NEW_TOKENS) -> list[dict]:
    """Run one batch of independent rows; returns per-row sealed-shape results.

    tasks/seeds/directions/signed_alphas are parallel per-row lists.
    Result dicts carry the same keys as the sealed ``generate_branch`` plus
    ``realized_delta_rms``.
    """
    from o1_prompt_template_v2 import render_prompt  # sealed

    B = len(tasks)
    if not (len(seeds) == len(directions) == len(signed_alphas) == B):
        raise BatchedEngineError("per-row argument lists must align")
    device = getattr(model, "device", torch.device("cuda:0"))

    prompt_token_lists = []
    for task in tasks:
        enc = tokenizer(render_prompt(task), return_tensors="pt",
                        truncation=True, max_length=1536, padding=False)
        prompt_token_lists.append([int(t) for t in enc["input_ids"][0]])
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    input_ids, attention_mask = _left_pad(prompt_token_lists, pad_id, device)

    hook = BatchedOneShotResidualEdit(directions, signed_alphas, attention_mask)
    handle = register(model, hook)
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=True, logits_to_keep=1,
                        return_per_loop_hidden_states=True)
    finally:
        handle.remove()
    if hook.applied != 1:
        raise BatchedEngineError(
            f"expected one L3_24 edit visit, got {hook.applied}")
    boundary = out.per_loop_hidden_states[3][:, -1, :].detach().float().cpu()
    if boundary.shape != (B, HIDDEN):
        raise BatchedEngineError(f"boundary shape {tuple(boundary.shape)}")
    cache = out.past_key_values
    logits = out.logits[:, -1, :]  # [B, V]

    # per-row trajectory state
    slot_of_row = list(range(B))          # row -> current batch slot (or None)
    generated: list[list[int]] = [[] for _ in range(B)]
    text = [""] * B
    finish = ["max_new_tokens"] * B
    finite = [bool(torch.isfinite(logits[slot_of_row[r]]).all().item())
              for r in range(B)]
    active = [True] * B

    step = 0
    while step < max_new_tokens and any(active):
        next_tokens: dict[int, int] = {}
        for r in range(B):
            if not active[r]:
                continue
            slot = slot_of_row[r]
            if not finite[r]:
                finish[r] = "nonfinite_logits"
                active[r] = False
                continue
            token = _sample_top_p(logits[slot], int(seeds[r]) + step)
            generated[r].append(token)
            text[r] = tokenizer.decode(generated[r], skip_special_tokens=True)
            if parse_final_answer(text[r]) is not None:
                finish[r] = "parseable_final_answer"
                active[r] = False
                continue
            if (tokenizer.eos_token_id is not None
                    and token == tokenizer.eos_token_id):
                finish[r] = "eos_without_commitment"
                active[r] = False
                continue
            next_tokens[r] = token
        step += 1
        if step >= max_new_tokens or not any(active):
            break

        if compaction:
            keep_rows = [r for r in range(B) if active[r]]
            keep_slots = [slot_of_row[r] for r in keep_rows]
            cache = _select_cache(cache, keep_slots)
            attention_mask = attention_mask[
                torch.tensor(keep_slots, dtype=torch.long,
                             device=attention_mask.device)]
            for new_slot, r in enumerate(keep_rows):
                slot_of_row[r] = new_slot
            for r in range(B):
                if not active[r]:
                    slot_of_row[r] = None
            feed_rows = keep_rows
        else:
            feed_rows = list(range(B))

        feed = []
        for r in feed_rows:
            if active[r]:
                feed.append(next_tokens[r])
            else:
                feed.append(pad_id)  # frozen slot; output ignored
        next_ids = torch.tensor([[t] for t in feed], dtype=torch.long,
                                device=device)
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device)], dim=1)
        with torch.inference_mode():
            out = model(input_ids=next_ids, attention_mask=attention_mask,
                        past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
        for r in feed_rows if compaction else range(B):
            if active[r]:
                slot = slot_of_row[r]
                finite[r] = finite[r] and bool(
                    torch.isfinite(logits[slot]).all().item())

    results = []
    for r in range(B):
        parsed = parse_final_answer(text[r])
        from o1_truth_table_verifier_v2 import verify_letter  # sealed
        results.append({
            "generated_token_ids": list(generated[r]),
            "generated_text": text[r],
            "well_formed": parsed is not None,
            "verifier_correct": verify_letter(tasks[r], parsed),
            "parsed_answer": parsed,
            "finish_reason": finish[r],
            "logits_finite": finite[r],
            "catastrophic_repetition": has_catastrophic_repetition(generated[r]),
            "injected_rms": hook.injected_rms[r],
            "realized_delta_rms": hook.realized_delta_rms[r],
            "_prefill_l4_47": boundary[r].numpy(),
            "_prompt_token_ids": prompt_token_lists[r],
        })
    return results


def _select_cache(cache, keep_slots: list[int]):
    """Move cache slices with their rows (compaction support)."""
    if hasattr(cache, "clone_select"):
        return cache.clone_select(keep_slots)
    if hasattr(cache, "batch_select_indices"):
        cache.batch_select_indices(torch.tensor(keep_slots, dtype=torch.long))
        return cache
    raise BatchedEngineError(
        "cache object supports no slot selection; compaction must stay OFF")
