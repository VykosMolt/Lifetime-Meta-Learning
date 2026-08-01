"""Deterministic synthetic Ouro-like runtime for hardware-independent testing.

This is NOT a substitute for B200 validation.  Its purpose is to catch
software defects (indexing, batching, RNG, resume, records) locally.

Design goals, all load-bearing for the tests:

  * Same call surface as the real model, so the SEALED single-row generator
    ``run_o1_v2_generation.generate_branch`` drives it unmodified:
    ``tokenizer(prompt, return_tensors=..., truncation=..., max_length=...,
    padding=...)``, ``model(input_ids, attention_mask, past_key_values,
    use_cache, logits_to_keep, return_per_loop_hidden_states)``,
    ``model.model.layers[23]`` visited once per loop with a ``current_ut``
    kwarg, ``out.per_loop_hidden_states[3]``, ``out.logits``,
    ``out.past_key_values``, EOS id, char-exact ``decode``.
  * BITWISE batch invariance: every tensor op is elementwise (mul/add/
    sin/cos/tanh/gather/scatter).  There are no cross-position or
    cross-hidden reductions and no parallel scans, so a row's hidden states,
    logits, and sampled tokens are bit-identical at batch sizes 1..64, under
    any scheduling order, and beside neighbors of any completion length.
    (The causal state mixing is an explicit sequential loop over positions.)
  * Genuine recurrent per-row cache (per-loop causal states + script
    progress), so batch-slot identity swaps, stale caches after compaction,
    and cross-row mixing are DETECTABLE defects, not no-ops.
  * Scripted behavior classes derived deterministically from the prompt
    tokens: valid A/B/C commitments (bare and angle form), malformed EOS,
    prefix-parser attacks (multi-char " About"-style tokens exist precisely so
    the attack does not commit at a bare letter), catastrophic repetition,
    max-token runs, and stochastic rows whose trajectory genuinely depends on
    the sealed per-step sampling generator and on the (intervened) hidden
    state.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch
from torch import nn

HIDDEN = 2048
VOCAB = 512
N_LAYERS = 24
N_LOOPS = 4
EOS_ID = 2
CHAR_BASE = 3  # ids 3..97 map to printable ASCII 32..126

# Multi-char tokens (ids 200+): exist so that prefix-attack scripts such as
# "FINAL ANSWER: About" never pass through a state whose decoded text ends at
# a bare parseable letter (mirrors the sealed-tokenizer property).
MULTI_TOKENS = {
    200: " About", 201: " Based", 202: " Correct", 203: " Because",
    204: "FINAL ANSWER", 205: " Answer",
}
_MULTI_BY_TEXT = sorted(MULTI_TOKENS.items(), key=lambda kv: -len(kv[1]))

BEHAVIOR_CLASSES = (
    "COMMIT_A", "COMMIT_B_ANGLE", "COMMIT_C_PERIOD", "MALFORMED_EOS",
    "PREFIX_ATTACK", "REPEAT_MAX", "NOSTOP_MAX", "STOCH_COMMIT", "STOCH_EOS",
)


class SyntheticTokenizer:
    eos_token_id = EOS_ID

    def encode_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for tid, tok in _MULTI_BY_TEXT:
                if text.startswith(tok, i):
                    ids.append(tid)
                    i += len(tok)
                    break
            else:
                ch = text[i]
                o = ord(ch)
                ids.append(CHAR_BASE + (o - 32) if 32 <= o <= 126 else
                           CHAR_BASE + (ord("?") - 32))
                if ch == "\n":
                    ids[-1] = 98  # newline token
                i += 1
        return ids

    def __call__(self, text: str, return_tensors: str = "pt",
                 truncation: bool = False, max_length: int | None = None,
                 padding=False):
        ids = self.encode_ids(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out = []
        for t in ids:
            t = int(t)
            if t in MULTI_TOKENS:
                out.append(MULTI_TOKENS[t])
            elif t == 98:
                out.append("\n")
            elif CHAR_BASE <= t < CHAR_BASE + 95:
                out.append(chr(32 + t - CHAR_BASE))
            elif skip_special_tokens:
                continue
            else:
                out.append(f"<{t}>")
        return "".join(out)


def _prompt_digest(prompt_ids: list[int]) -> bytes:
    payload = b"o1b200.synth.v1|" + b",".join(
        str(int(t)).encode() for t in prompt_ids)
    return hashlib.sha256(payload).digest()


def behavior_of_prompt(prompt_ids: list[int], tok: SyntheticTokenizer) -> dict:
    """Deterministic behavior class + script for a prompt (pure function)."""
    d = _prompt_digest(prompt_ids)
    cls = BEHAVIOR_CLASSES[d[0] % len(BEHAVIOR_CLASSES)]
    n_stoch = 5 + (d[1] % 36)
    scripts = {
        "COMMIT_A": "Reasoning first.\nFINAL ANSWER: A\n",
        "COMMIT_B_ANGLE": "Consider it.\nFINAL ANSWER: <B>\n",
        "COMMIT_C_PERIOD": "Done.\nFINAL ANSWER: C.\n",
        "MALFORMED_EOS": "It could be either of them honestly",
        "PREFIX_ATTACK": "FINAL ANSWER: About right\nFINAL ANSWER: Based on it\n",
    }
    out = {"class": cls, "n_stoch": n_stoch}
    if cls in scripts:
        out["script"] = tok.encode_ids(scripts[cls])
        out["then_eos"] = cls in ("MALFORMED_EOS", "PREFIX_ATTACK")
    elif cls == "REPEAT_MAX":
        cycle = tok.encode_ids("loop it ")  # 8 chars -> 8 tokens
        assert len(cycle) == 8
        out["script"] = tok.encode_ids("prefix ") + cycle * 64
        out["then_eos"] = False
    elif cls == "NOSTOP_MAX":
        out["script"] = None  # pseudo-junk from step hash, never stops
        out["then_eos"] = False
    elif cls == "STOCH_COMMIT":
        out["script"] = tok.encode_ids("\nFINAL ANSWER: B\n")  # after n_stoch
        out["then_eos"] = False
    elif cls == "STOCH_EOS":
        out["script"] = None
        out["then_eos"] = True
    return out


class _SynthLayer(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index
        # deterministic per-layer gain, close to 1 so 96 visits stay stable
        self.gain = 1.0 + ((index * 37 + 11) % 23 - 11) * 1e-4

    def forward(self, hidden: torch.Tensor, current_ut: int | None = None):
        return hidden * self.gain


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(_SynthLayer(i) for i in range(N_LAYERS))


class SynthCache:
    """Per-row recurrent state: per-loop causal sums + script progress."""

    def __init__(self, states: list[torch.Tensor], rows: list[dict]):
        self.states = states          # N_LOOPS tensors of shape [B, HIDDEN]
        self.rows = rows              # per-row dicts: behavior, gen_count, prompt_len

    def clone_select(self, keep: list[int]) -> "SynthCache":
        """Select batch slots (used by tests and optional compaction)."""
        idx = torch.tensor(keep, dtype=torch.long,
                           device=self.states[0].device)
        return SynthCache([s.index_select(0, idx).clone() for s in self.states],
                          [dict(self.rows[i]) for i in keep])


class SyntheticOuroModel(nn.Module):
    """Deterministic looped model with the sealed-generator call surface."""

    def __init__(self, device: str = "cpu", seed_tag: int = 0):
        super().__init__()
        self.model = _Inner()
        self.config = SimpleNamespace(total_ut_steps=N_LOOPS,
                                      early_exit_threshold=1.0)
        self._device = torch.device(device)
        self.seed_tag = int(seed_tag)
        self.tokenizer = SyntheticTokenizer()
        arange_h = torch.arange(HIDDEN, dtype=torch.float32)
        self._dscale = (0.1 * (arange_h + 1.0) / 64.0)
        self._pscale = (0.05 * ((arange_h % 17) + 1.0))
        self._gather_idx = ((torch.arange(VOCAB) * 97) % HIDDEN).long()
        self._logit_w = 0.5 + 0.01 * torch.sin(
            0.3 * (torch.arange(VOCAB, dtype=torch.float32) + 1.0))
        self.to(self._device)
        for t in ("_dscale", "_pscale", "_gather_idx", "_logit_w"):
            setattr(self, t, getattr(self, t).to(self._device))
        self.eval()

    @property
    def device(self):
        return self._device

    def to(self, device):  # noqa: D401 - keep nn.Module behavior + buffers
        out = super().to(device)
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        return out

    # ---------------- deterministic elementwise pieces ----------------

    def _embed(self, input_ids: torch.Tensor, positions: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        # [B,T] -> [B,T,H]; strictly elementwise (broadcasted sin/cos)
        tok = input_ids.to(torch.float32).unsqueeze(-1)
        pos = positions.to(torch.float32).unsqueeze(-1)
        h = 0.5 * torch.sin((tok + 1.0 + self.seed_tag) * self._dscale)
        h = h + 0.01 * torch.cos((pos + 1.0) * self._pscale)
        return h * mask.to(torch.float32).unsqueeze(-1)

    def _mix_sequential(self, h: torch.Tensor, mask: torch.Tensor,
                        state: torch.Tensor) -> torch.Tensor:
        """Causal state mixing, position by position (NO parallel scan).

        state is updated IN PLACE ([B,H]); returns mixed hidden [B,T,H].
        Padded positions contribute exactly 0 and receive no mixing.
        """
        T = h.shape[1]
        outs = []
        for t in range(T):
            m = mask[:, t].to(torch.float32).unsqueeze(-1)      # [B,1]
            state += h[:, t, :] * m
            outs.append(h[:, t, :] + 0.01 * torch.tanh(state) * m)
        return torch.stack(outs, dim=1)

    def _norm(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(h)  # elementwise stand-in for the boundary norm

    def _logits_from_hidden(self, h_last: torch.Tensor,
                            rows: list[dict]) -> torch.Tensor:
        """Per-row logits: elementwise gather + script/stochastic shaping."""
        B = h_last.shape[0]
        base = h_last.index_select(-1, self._gather_idx) * self._logit_w  # [B,V]
        logits = base.clone()
        for b in range(B):
            row = rows[b]
            beh = row["behavior"]
            step = row["gen_count"]
            cls = beh["class"]
            if cls == "NOSTOP_MAX":
                tok = CHAR_BASE + (ord("a") - 32) + (
                    _step_hash(row["prompt_key"], step) % 26)
                logits[b, tok] += 60.0
            elif cls == "STOCH_EOS":
                if step >= beh["n_stoch"]:
                    logits[b, EOS_ID] += 60.0
                else:
                    logits[b] = self._stochastic_row(logits[b])
            elif cls == "STOCH_COMMIT":
                if step >= beh["n_stoch"]:
                    k = step - beh["n_stoch"]
                    script = beh["script"]
                    tok = script[k] if k < len(script) else EOS_ID
                    logits[b, tok] += 60.0
                else:
                    logits[b] = self._stochastic_row(logits[b])
            else:
                script = beh["script"]
                if step < len(script):
                    logits[b, script[step]] += 60.0
                else:
                    logits[b, EOS_ID] += 60.0
        return logits

    def _stochastic_row(self, row_logits: torch.Tensor) -> torch.Tensor:
        """Nearly flat over lowercase chars; genuinely seed- and h-dependent."""
        out = torch.full_like(row_logits, -30.0)
        lo = CHAR_BASE + (ord("a") - 32)
        out[lo:lo + 26] = 2.0 + 0.8 * torch.tanh(row_logits[lo:lo + 26])
        return out

    # ---------------- forward ----------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                past_key_values: SynthCache | None = None,
                use_cache: bool = True, logits_to_keep: int = 1,
                return_per_loop_hidden_states: bool = False,
                position_ids: torch.Tensor | None = None):
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        B, T = input_ids.shape
        if past_key_values is None:
            # prefill
            states = [torch.zeros(B, HIDDEN, dtype=torch.float32,
                                  device=self._device) for _ in range(N_LOOPS)]
            rows = []
            for b in range(B):
                m = attention_mask[b].bool()
                prompt_ids = [int(t) for t in input_ids[b][m]]
                rows.append({
                    "behavior": behavior_of_prompt(prompt_ids, self.tokenizer),
                    "prompt_key": _prompt_digest(prompt_ids),
                    "gen_count": 0,
                    "prompt_len": len(prompt_ids),
                })
            cache = SynthCache(states, rows)
            if position_ids is None:
                position_ids = attention_mask.long().cumsum(dim=1) - 1
                position_ids = position_ids.clamp(min=0)
            h = self._embed(input_ids, position_ids, attention_mask)
            per_loop = []
            for u in range(N_LOOPS):
                for layer in self.model.layers:
                    h = layer(h, current_ut=u)
                h = self._mix_sequential(h, attention_mask, cache.states[u])
                per_loop.append(self._norm(h))
            logits = self._logits_from_hidden(h[:, -1, :], cache.rows)
        else:
            # single-token decode
            cache = past_key_values
            if T != 1:
                raise RuntimeError("decode expects one token per step")
            pos = (attention_mask.long().sum(dim=1, keepdim=True) - 1).clamp(min=0)
            m1 = torch.ones(B, 1, dtype=attention_mask.dtype,
                            device=self._device)
            h = self._embed(input_ids, pos, m1)
            per_loop = None
            for u in range(N_LOOPS):
                for layer in self.model.layers:
                    h = layer(h, current_ut=u)
                h = self._mix_sequential(h, m1, cache.states[u])
            for b in range(B):
                cache.rows[b]["gen_count"] += 1
            logits = self._logits_from_hidden(h[:, -1, :], cache.rows)
        out = SimpleNamespace(
            logits=logits.unsqueeze(1),          # [B, 1, V]
            past_key_values=cache if use_cache else None,
        )
        if return_per_loop_hidden_states:
            out.per_loop_hidden_states = per_loop
        return out


def _step_hash(prompt_key: bytes, step: int) -> int:
    return int.from_bytes(
        hashlib.sha256(prompt_key + step.to_bytes(4, "big")).digest()[:4], "big")


def load_synthetic(model_artifact: dict) -> tuple[SyntheticOuroModel, SyntheticTokenizer]:
    """model_artifact: {"kind": "synthetic", "device": ..., "seed_tag": int}"""
    model = SyntheticOuroModel(device=model_artifact.get("device", "cpu"),
                               seed_tag=int(model_artifact.get("seed_tag", 0)))
    return model, model.tokenizer
