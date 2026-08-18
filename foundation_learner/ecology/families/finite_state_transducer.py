"""``finite_state_transducer`` — hidden Mealy machine transduction.

Mechanism: a hidden Mealy machine with 3-5 states, an input alphabet of size
3-4 and an output alphabet of size 2-4.  Each ``(state, input symbol)`` pair
carries a next state and one output symbol; the machine starts in state 0 and
emits one output symbol per input symbol.  Answer: the output string.

Difficulty 0/1/2 -> input length 5/7/9.

TRANSFER SHIFT: much longer inputs (length 14), which exercise state cycles the
short training inputs rarely reach.  The latent machine is unchanged.

Structured feedback: the length of the longest correct output prefix, plus a
length band.  The output string itself is never stated.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, make_rng)
from ..surface_remap import PromptSpec, syms

IN_POOL = ("m", "n", "o", "p")
OUT_POOL = ("u", "v", "w", "x")
SYMBOL_POOL = IN_POOL + OUT_POOL
_LENGTHS = {0: 5, 1: 7, 2: 9}
_TRANSFER_LENGTH = 14


class FiniteStateTransducerFamily(TaskFamily):
    family_id = "finite_state_transducer"
    canon_mode = "string"
    symbol_pool = SYMBOL_POOL
    symbol_blocks = (IN_POOL, OUT_POOL)

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        n_states = int(rng.integers(3, 6))
        n_in = int(rng.integers(3, 5))
        n_out = int(rng.integers(2, 5))
        delta = [[[int(rng.integers(0, n_states)), int(rng.integers(0, n_out))]
                  for _ in range(n_in)] for _ in range(n_states)]
        return Rule(self.family_id, {"n_states": n_states, "n_in": n_in,
                                     "n_out": n_out, "delta": delta})

    def _run(self, rule: Rule, seq: list[int], start: int = 0) -> list[int]:
        state = start
        out = []
        for symbol in seq:
            state, emitted = rule.params["delta"][state][symbol]
            out.append(emitted)
        return out

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        n_in = rule.params["n_in"]
        n_out = rule.params["n_out"]
        length = _TRANSFER_LENGTH if kind == KIND_TRANSFER else _LENGTHS[difficulty]
        seq = [int(v) for v in rng.integers(0, n_in, size=length)]
        out = self._run(rule, seq)
        clauses = [
            "A hidden machine reads an input string and writes an output string,"
            " one output symbol per input symbol.",
            f"Input alphabet: {syms(IN_POOL[:n_in])}",
            f"Output alphabet: {syms(OUT_POOL[:n_out])}",
            f"Input string: {syms((IN_POOL[s] for s in seq), sep='')}",
        ]
        spec = PromptSpec(
            clauses=tuple(clauses),
            question="What does the hidden machine write for this input?",
            answer_format=("Reply with a final line: ANSWER: "
                           "<the output string>"),
            alphabet=tuple(IN_POOL[:n_in] + OUT_POOL[:n_out]),
            labels=(),
            answer=syms((OUT_POOL[s] for s in out), sep=""),
        )
        return Item(spec=spec, data={"seq": seq, "out": out})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        guess = attempt or ""
        prefix = 0
        for a, b in zip(truth, guess):
            if a != b:
                break
            prefix += 1
        band = 0 if len(guess) == len(truth) else (
            1 if len(guess) < len(truth) else 2)
        return Feedback("HINT-PREFIX", (
            HintField("correct_prefix", "CNT", prefix, len(truth) + 1),
            HintField("length", "LEN", band, 3, ("EQ", "SHORT", "LONG")),
        ))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        from ...episodes.parse import canonicalize

        item, plan = self._item_and_plan(rule, instance)
        seq = item.data["seq"]
        for start in range(1, rule.params["n_states"]):
            out = self._run(rule, seq, start=start)
            text = canonicalize("".join(plan.map_symbol(OUT_POOL[s]) for s in out),
                                self.canon_mode)
            if text is not None and text != instance.answer_canonical:
                return text
        out = self._run(rule, seq[1:])
        return canonicalize("".join(plan.map_symbol(OUT_POOL[s]) for s in out),
                            self.canon_mode) or instance.answer_canonical


FAMILY = FiniteStateTransducerFamily()
