"""Manual left-padded greedy decode loop (contract section 22).

``model.generate()`` is NEVER used anywhere in this campaign.  The frozen
recipe implemented here is the O1 engine pattern:

* each prompt is tokenized independently (``add_special_tokens=False``: the FL
  protocol is plain text with fixed markers and no chat template, section 6);
* rows are LEFT padded to the batch maximum with
  ``pad_id = tokenizer.eos_token_id`` (or ``0`` when the tokenizer declares
  none), so the final non-padding position of every row is sequence position
  ``-1``;
* an explicit ``attention_mask`` is maintained across every decode step and
  grown by one column per step;
* ``position_ids`` are derived from the mask as ``cumsum(mask) - 1`` clamped at
  zero, so a row's rotary positions are identical to the positions it would
  receive when decoded alone.  (RoPE is relative, so the un-corrected "global"
  convention is also mathematically equivalent; the corrected convention is
  used because it is bit-closer to the unbatched reference and therefore makes
  the equivalence gate below a tighter instrument.)
* ``use_cache=True`` with the model's own ``UniversalTransformerCache``;
* greedy = ``argmax`` over the final-position logits at every step;
* stopping per row, in the frozen order: non-finite logits -> parseable
  ``ANSWER:`` line -> EOS -> ``max_new_tokens`` (default 64);
* finished rows KEEP their batch slot (non-compacting): active rows never
  change slot, so batch composition cannot remap a row's numerics.

Optional ``prefix_embeds`` (needed by the FL5/FL7 context-reset evaluations)
are prepended through ``inputs_embeds`` with matching attention-mask ones.
Rows may carry prefixes of different lengths; the combined
``[prefix | prompt]`` sequence is what gets left-padded.

Equivalence gate
----------------
``run_equivalence_gate(bundle)`` runs at campaign start and asserts that
batched greedy decoding equals unbatched greedy decoding token-for-token on
mixed-length prompts, with and without prefix embeddings.  On failure,
``generation_policy_from_gate`` returns a config that forces batch size 1;
evaluators consume that config, so correctness is preserved and only
throughput is lost.  The gate result is a record, never a silent fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import torch

from . import text_sha256

GENERATION_RECORD_SCHEMA = "flb200.generation_record.v1"
EQUIVALENCE_GATE_SCHEMA = "flb200.equivalence_gate.v1"

MAX_NEW_TOKENS_DEFAULT = 64

#: Contract section 4 grammar, used when ``episodes/parse.py`` is unavailable.
ANSWER_LINE_RE = re.compile(r"^ANSWER:\s*(\S.*?)\s*$", re.MULTILINE)

#: Streaming stop rule: an ``ANSWER:`` line that has been TERMINATED by a
#: newline.  Requiring the terminator is what keeps multi-token payloads (sets,
#: sequences, formulas) from being truncated mid-answer, which an
#: end-of-text-tolerant rule would do after the first payload token.
_COMPLETE_ANSWER_LINE_RE = re.compile(r"^ANSWER:[^\S\n]*(\S[^\n]*?)[^\S\n]*\n", re.MULTILINE)

_PARSER_CANDIDATE_NAMES = ("parse_answer", "parse_final_answer", "parse_answer_line")


class GenerationError(RuntimeError):
    """Raised for malformed generation requests (never swallowed)."""


# --------------------------------------------------------------------------
# answer parsing
# --------------------------------------------------------------------------
def parse_answer_local(text: str) -> str | None:
    """Contract section 4 grammar: the LAST line matching ``^ANSWER:\\s*(\\S.*?)\\s*$``."""
    last = None
    for m in ANSWER_LINE_RE.finditer(str(text or "")):
        last = m
    return None if last is None else last.group(1)


def resolve_answer_parser() -> tuple[Callable[[str], str | None], str]:
    """Return ``(parser, source)``; prefers the real ``episodes/parse.py``."""
    try:
        from foundation_learner.episodes import parse as _parse  # type: ignore
    except Exception:
        return parse_answer_local, "evaluation.generation.parse_answer_local"
    for name in _PARSER_CANDIDATE_NAMES:
        fn = getattr(_parse, name, None)
        if callable(fn):
            return fn, f"episodes.parse.{name}"
    return parse_answer_local, "evaluation.generation.parse_answer_local"


def complete_answer_line_end(text: str) -> int | None:
    """Char offset just past the first newline-terminated ``ANSWER:`` line."""
    m = _COMPLETE_ANSWER_LINE_RE.search(str(text or ""))
    return None if m is None else m.end()


# --------------------------------------------------------------------------
# configuration / records
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationConfig:
    """Frozen generation settings.

    ``batch_size=None`` means "one batch containing every pending prompt".
    ``force_unbatched=True`` (set by the equivalence-gate fallback) pins the
    batch size to 1 regardless of ``batch_size``.
    """

    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT
    batch_size: int | None = None
    force_unbatched: bool = False
    stop_on_answer: bool = True

    def effective_batch_size(self, n_pending: int) -> int:
        if self.force_unbatched:
            return 1
        if self.batch_size is None:
            return max(1, int(n_pending))
        return max(1, min(int(self.batch_size), int(n_pending)))


DEFAULT_GENERATION_CONFIG = GenerationConfig()


@dataclass(frozen=True)
class GenerationRecord:
    """Per-row generation transcript (contract: "generations, token ids, hashes")."""

    text: str
    token_ids: tuple[int, ...]
    finish_reason: str
    parsed_answer: str | None
    prompt_text_sha256: str
    prompt_token_count: int
    prefix_len: int
    logits_finite: bool
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GENERATION_RECORD_SCHEMA,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "finish_reason": self.finish_reason,
            "parsed_answer": self.parsed_answer,
            "prompt_text_sha256": self.prompt_text_sha256,
            "prompt_token_count": self.prompt_token_count,
            "prefix_len": self.prefix_len,
            "logits_finite": self.logits_finite,
            "batch_size": self.batch_size,
            "generated_text_sha256": text_sha256(self.text),
        }


# --------------------------------------------------------------------------
# bundle / tensor helpers
# --------------------------------------------------------------------------
def _bundle_parts(bundle: Any) -> tuple[Any, Any, torch.device]:
    model = getattr(bundle, "model", None)
    tokenizer = getattr(bundle, "tokenizer", None)
    if model is None or tokenizer is None:
        raise GenerationError(
            "bundle must expose .model and .tokenizer (contract section 23)")
    device = getattr(bundle, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - degenerate model
            device = torch.device("cpu")
    return model, tokenizer, torch.device(device)


def _pad_id(tokenizer: Any) -> int:
    eos = getattr(tokenizer, "eos_token_id", None)
    return int(eos) if eos is not None else 0


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    enc = tokenizer(str(prompt), add_special_tokens=False)
    ids = [int(t) for t in enc["input_ids"]]
    if not ids:
        raise GenerationError("empty prompt after tokenization")
    return ids


def normalize_prefix_embeds(
    prefix_embeds: Any,
    n_rows: int,
    hidden_size: int,
) -> list[torch.Tensor | None]:
    """Accept ``None`` / ``[k,H]`` shared / ``[B,k,H]`` / per-row sequence."""
    if prefix_embeds is None:
        return [None] * n_rows
    if isinstance(prefix_embeds, torch.Tensor):
        if prefix_embeds.dim() == 2:
            if prefix_embeds.shape[1] != hidden_size:
                raise GenerationError(
                    f"prefix_embeds hidden size {prefix_embeds.shape[1]} != {hidden_size}")
            return [prefix_embeds] * n_rows
        if prefix_embeds.dim() == 3:
            if prefix_embeds.shape[0] != n_rows:
                raise GenerationError(
                    f"prefix_embeds batch {prefix_embeds.shape[0]} != {n_rows} rows")
            if prefix_embeds.shape[2] != hidden_size:
                raise GenerationError(
                    f"prefix_embeds hidden size {prefix_embeds.shape[2]} != {hidden_size}")
            return [prefix_embeds[i] for i in range(n_rows)]
        raise GenerationError(f"prefix_embeds must be 2-D or 3-D, got {prefix_embeds.dim()}-D")
    if isinstance(prefix_embeds, (list, tuple)):
        if len(prefix_embeds) != n_rows:
            raise GenerationError(
                f"prefix_embeds sequence length {len(prefix_embeds)} != {n_rows} rows")
        out: list[torch.Tensor | None] = []
        for p in prefix_embeds:
            if p is None:
                out.append(None)
                continue
            if not isinstance(p, torch.Tensor) or p.dim() != 2 or p.shape[1] != hidden_size:
                raise GenerationError("per-row prefix_embeds must be [k, hidden_size] tensors")
            out.append(p)
        return out
    raise GenerationError(f"unsupported prefix_embeds type {type(prefix_embeds)!r}")


def _positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask.cumsum(-1) - 1).clamp(min=0)


# --------------------------------------------------------------------------
# core decode loop
# --------------------------------------------------------------------------
def assert_decodable(model: Any) -> None:
    """Refuse to decode a model that cannot be decoded correctly.

    THE hazard (Amendment 16): transformers'
    ``GradientCheckpointingLayer.__call__`` does
    ``if self.gradient_checkpointing and self.training`` and then DROPS
    ``use_cache`` and ``past_key_values``.  This manual greedy loop carries its
    own KV cache, so a model left in train mode with layer-level checkpointing
    enabled silently recomputes from scratch and emits DIFFERENT text than the
    same weights in eval mode.  Every trained arm is evaluated right after
    training, so this is not a hypothetical.

    Making the decode entry point REFUSE is what turns "remember to call
    model.eval()" into a structural guarantee: the callers restore eval mode
    (``training.trainer.run_training_arm``,
    ``campaign.stage_definitions.prepare_bundle_for_evaluation``,
    ``mechanisms.stage_support.dev_records``), and this assert makes a missed
    call impossible to ignore rather than invisible in the numbers.
    """
    if getattr(model, "training", False):
        checkpointed = sorted({type(m).__name__ for m in model.modules()
                               if getattr(m, "gradient_checkpointing", False)})
        raise GenerationError(
            "REFUSED: greedy decoding was asked to run on a model in TRAIN "
            "mode. Evaluation requires eval mode: with layer-level gradient "
            "checkpointing active in train mode, transformers drops "
            "use_cache/past_key_values and this decoder's KV cache is silently "
            "discarded, producing different text than the same weights in eval "
            "mode. Call training.model_loading.set_evaluation_mode(model) "
            f"first. (checkpointed modules: {checkpointed or 'none'})")
    still_on = sorted({type(m).__name__ for m in model.modules()
                       if getattr(m, "gradient_checkpointing", False)})
    if still_on:
        raise GenerationError(
            "REFUSED: greedy decoding was asked to run with layer-level "
            f"gradient checkpointing still enabled on {still_on}. It is "
            "incompatible with the KV cache this decoder maintains; call "
            "training.model_loading.set_evaluation_mode(model) first.")


def _decode_group(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: Sequence[str],
    prefixes: Sequence[torch.Tensor | None],
    cfg: GenerationConfig,
    parser: Callable[[str], str | None],
) -> list[GenerationRecord]:
    assert_decodable(model)
    n = len(prompts)
    pad_id = _pad_id(tokenizer)
    token_lists = [encode_prompt(tokenizer, p) for p in prompts]
    prefix_lens = [0 if p is None else int(p.shape[0]) for p in prefixes]
    total_lens = [prefix_lens[i] + len(token_lists[i]) for i in range(n)]
    width = max(total_lens)

    embed = model.get_input_embeddings()
    emb_weight = embed.weight
    use_embeds = any(p is not None for p in prefixes)

    attention_mask = torch.zeros((n, width), dtype=torch.long, device=device)
    for i in range(n):
        attention_mask[i, width - total_lens[i]:] = 1

    if use_embeds:
        rows = []
        for i in range(n):
            ids = torch.tensor(token_lists[i], dtype=torch.long, device=device)
            tok_emb = embed(ids)
            parts = []
            pad_n = width - total_lens[i]
            if pad_n:
                pad_ids = torch.full((pad_n,), pad_id, dtype=torch.long, device=device)
                parts.append(embed(pad_ids))
            if prefixes[i] is not None:
                parts.append(prefixes[i].to(device=device, dtype=emb_weight.dtype))
            parts.append(tok_emb)
            rows.append(torch.cat(parts, dim=0))
        model_inputs: dict[str, Any] = {"inputs_embeds": torch.stack(rows, dim=0)}
    else:
        ids = torch.full((n, width), pad_id, dtype=torch.long, device=device)
        for i in range(n):
            ids[i, width - total_lens[i]:] = torch.tensor(
                token_lists[i], dtype=torch.long, device=device)
        model_inputs = {"input_ids": ids}

    with torch.inference_mode():
        out = model(
            attention_mask=attention_mask,
            position_ids=_positions_from_mask(attention_mask),
            use_cache=True,
            logits_to_keep=1,
            **model_inputs,
        )
    cache = out.past_key_values
    logits = out.logits[:, -1, :]

    generated: list[list[int]] = [[] for _ in range(n)]
    texts = [""] * n
    finish = ["max_new_tokens"] * n
    parsed: list[str | None] = [None] * n
    finite = [bool(torch.isfinite(logits[i]).all().item()) for i in range(n)]
    active = [True] * n
    eos_id = getattr(tokenizer, "eos_token_id", None)

    step = 0
    while step < int(cfg.max_new_tokens) and any(active):
        next_tokens = [pad_id] * n
        for i in range(n):
            if not active[i]:
                continue
            if not finite[i]:
                finish[i] = "nonfinite_logits"
                active[i] = False
                continue
            token = int(torch.argmax(logits[i]).item())
            generated[i].append(token)
            texts[i] = tokenizer.decode(generated[i], skip_special_tokens=True)
            next_tokens[i] = token
            if cfg.stop_on_answer:
                end = complete_answer_line_end(texts[i])
                if end is not None and parser(texts[i][:end]) is not None:
                    finish[i] = "answer_line"
                    active[i] = False
                    continue
            if eos_id is not None and token == int(eos_id):
                finish[i] = "eos"
                active[i] = False
                continue
        step += 1
        if step >= int(cfg.max_new_tokens) or not any(active):
            break

        next_ids = torch.tensor([[t] for t in next_tokens], dtype=torch.long, device=device)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((n, 1), dtype=torch.long, device=device)], dim=1)
        with torch.inference_mode():
            out = model(
                input_ids=next_ids,
                attention_mask=attention_mask,
                position_ids=_positions_from_mask(attention_mask)[:, -1:],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
        for i in range(n):
            if active[i]:
                finite[i] = finite[i] and bool(torch.isfinite(logits[i]).all().item())

    records = []
    for i in range(n):
        parsed[i] = parser(texts[i])
        records.append(GenerationRecord(
            text=texts[i],
            token_ids=tuple(generated[i]),
            finish_reason=finish[i],
            parsed_answer=parsed[i],
            prompt_text_sha256=text_sha256(prompts[i]),
            prompt_token_count=len(token_lists[i]),
            prefix_len=prefix_lens[i],
            logits_finite=finite[i],
            batch_size=n,
        ))
    return records


def greedy_generate_detailed(
    bundle: Any,
    prompts: Sequence[str],
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT,
    prefix_embeds: Any = None,
    *,
    config: GenerationConfig | None = None,
    parser: Callable[[str], str | None] | None = None,
) -> list[GenerationRecord]:
    """Greedy decode ``prompts``; returns full per-row generation records."""
    prompts = list(prompts)
    if not prompts:
        return []
    model, tokenizer, device = _bundle_parts(bundle)
    cfg = config or GenerationConfig(max_new_tokens=max_new_tokens)
    if config is not None and max_new_tokens != MAX_NEW_TOKENS_DEFAULT:
        cfg = replace(cfg, max_new_tokens=max_new_tokens)
    if parser is None:
        parser = resolve_answer_parser()[0]

    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    prefixes = normalize_prefix_embeds(prefix_embeds, len(prompts), hidden_size)

    bs = cfg.effective_batch_size(len(prompts))
    records: list[GenerationRecord] = []
    for start in range(0, len(prompts), bs):
        chunk = prompts[start:start + bs]
        chunk_prefixes = prefixes[start:start + bs]
        records.extend(_decode_group(
            model, tokenizer, device, chunk, chunk_prefixes, cfg, parser))
    return records


def greedy_generate(
    bundle: Any,
    prompts: Sequence[str],
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT,
    prefix_embeds: Any = None,
    *,
    config: GenerationConfig | None = None,
    parser: Callable[[str], str | None] | None = None,
) -> list[str]:
    """Contract section 23 signature: greedy decode, returning texts only."""
    return [r.text for r in greedy_generate_detailed(
        bundle, prompts, max_new_tokens, prefix_embeds,
        config=config, parser=parser)]


# --------------------------------------------------------------------------
# equivalence gate
# --------------------------------------------------------------------------
GATE_PROMPTS: tuple[str, ...] = (
    "TASK: apply the hidden rule to the sequence a b c\nATTEMPT:",
    "TASK: x\nATTEMPT:",
    "TASK: this is a deliberately much longer support problem statement with "
    "several additional clauses so that the batch contains mixed prompt "
    "lengths and left padding is actually exercised\nATTEMPT:",
    "TASK: medium length statement about a hidden rule\nFEEDBACK: INCORRECT\n"
    "ATTEMPT:",
)

GATE_PREFIX_SEED = 20260809
GATE_PREFIX_LEN = 8


def build_gate_prefix(bundle: Any, k: int = GATE_PREFIX_LEN,
                      seed: int = GATE_PREFIX_SEED) -> torch.Tensor:
    """Deterministic [k, hidden] probe prefix (explicit generator, never global RNG)."""
    model, _, device = _bundle_parts(bundle)
    weight = model.get_input_embeddings().weight
    hidden = int(weight.shape[1])
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    scale = float(weight.detach().float().norm(dim=1).median().item()) / max(1.0, hidden ** 0.5)
    raw = torch.empty((k, hidden), dtype=torch.float32)
    raw.normal_(0.0, 1.0, generator=gen)
    return (raw * scale).to(device=device, dtype=weight.dtype)


def run_equivalence_gate(
    bundle: Any,
    prompts: Sequence[str] | None = None,
    *,
    max_new_tokens: int = 16,
    include_prefix_case: bool = True,
) -> dict[str, Any]:
    """Assert batched greedy == unbatched greedy, token for token.

    Returns a ``flb200.equivalence_gate.v1`` record.  ``passed=False`` never
    raises here: the caller converts it into a batch-size-1 generation policy
    via :func:`generation_policy_from_gate`, so science continues at reduced
    throughput.
    """
    probe = list(prompts) if prompts else list(GATE_PROMPTS)
    cases: list[dict[str, Any]] = []
    variants: list[tuple[str, Any]] = [("no_prefix", None)]
    if include_prefix_case:
        prefix = build_gate_prefix(bundle)
        variants.append(("shared_prefix", prefix))
        variants.append(("per_row_prefix", [prefix[: 1 + (i % GATE_PREFIX_LEN)]
                                            for i in range(len(probe))]))

    for name, pfx in variants:
        cfg_batched = GenerationConfig(max_new_tokens=max_new_tokens)
        cfg_single = GenerationConfig(max_new_tokens=max_new_tokens, force_unbatched=True)
        batched = greedy_generate_detailed(bundle, probe, prefix_embeds=pfx, config=cfg_batched)
        single = greedy_generate_detailed(bundle, probe, prefix_embeds=pfx, config=cfg_single)
        rows = []
        for i, (b, s) in enumerate(zip(batched, single)):
            first_div = None
            for j, (tb, ts) in enumerate(zip(b.token_ids, s.token_ids)):
                if tb != ts:
                    first_div = j
                    break
            if first_div is None and len(b.token_ids) != len(s.token_ids):
                first_div = min(len(b.token_ids), len(s.token_ids))
            rows.append({
                "row": i,
                "equal": b.token_ids == s.token_ids,
                "first_divergence_step": first_div,
                "batched_tokens": len(b.token_ids),
                "unbatched_tokens": len(s.token_ids),
                "batched_finish": b.finish_reason,
                "unbatched_finish": s.finish_reason,
            })
        cases.append({
            "variant": name,
            "n_rows": len(probe),
            "passed": all(r["equal"] for r in rows),
            "rows": rows,
        })

    identity = getattr(bundle, "identity", None)
    if callable(identity):
        try:
            identity = identity()
        except Exception:
            identity = None
    return {
        "schema": EQUIVALENCE_GATE_SCHEMA,
        "passed": all(c["passed"] for c in cases),
        "max_new_tokens": int(max_new_tokens),
        "n_prompts": len(probe),
        "prompt_sha256": [text_sha256(p) for p in probe],
        "cases": cases,
        "model_identity": identity if isinstance(identity, dict) else None,
    }


def generation_policy_from_gate(
    gate: dict[str, Any],
    base: GenerationConfig | None = None,
) -> GenerationConfig:
    """Frozen fallback rule: gate failure -> unbatched (batch size 1) decoding."""
    cfg = base or DEFAULT_GENERATION_CONFIG
    if bool(gate.get("passed")):
        return cfg
    return replace(cfg, force_unbatched=True, batch_size=1)


__all__ = [
    "GENERATION_RECORD_SCHEMA",
    "EQUIVALENCE_GATE_SCHEMA",
    "MAX_NEW_TOKENS_DEFAULT",
    "ANSWER_LINE_RE",
    "GenerationError",
    "GenerationConfig",
    "DEFAULT_GENERATION_CONFIG",
    "GenerationRecord",
    "parse_answer_local",
    "resolve_answer_parser",
    "complete_answer_line_end",
    "encode_prompt",
    "normalize_prefix_embeds",
    "assert_decodable",
    "greedy_generate",
    "greedy_generate_detailed",
    "build_gate_prefix",
    "run_equivalence_gate",
    "generation_policy_from_gate",
    "GATE_PROMPTS",
]
