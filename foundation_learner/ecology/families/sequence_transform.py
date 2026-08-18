"""``sequence_transform`` — hidden composition of sequence operators.

Mechanism: a hidden composition of 2-3 operators drawn from ``reverse``,
``rotate_k``, ``swap_adjacent_pairs``, ``drop_every_k``, ``duplicate_first`` and
``map_symbol`` (a hidden bijection of the five-symbol alphabet), applied left to
right to a symbol sequence.  Answer: the transformed sequence, space separated.

Difficulty 0/1/2 -> input length 6/8/10.

TRANSFER SHIFT: longer inputs (length 16, outside the training length range),
which also lengthens the effect of the length-changing operators
(``drop_every_k``, ``duplicate_first``).  The latent operator composition is
unchanged.

Structured feedback: whether the attempt has the right length, and the first
position at which it diverges from the target.  The target sequence itself is
never stated.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, letters, make_rng)
from ..surface_remap import PromptSpec, syms

SYMBOL_POOL = ("a", "b", "c", "d", "e", "f", "g", "h")
N_SYMBOLS = 5
OPS = ("reverse", "rotate", "swap_pairs", "drop_every", "duplicate_first",
       "map_symbol")
_LENGTHS = {0: 6, 1: 8, 2: 10}
_TRANSFER_LENGTH = 16


def _apply(op: dict, seq: list[int]) -> list[int]:
    name = op["op"]
    if name == "reverse":
        return list(reversed(seq))
    if name == "rotate":
        k = op["k"] % max(1, len(seq))
        return seq[k:] + seq[:k]
    if name == "swap_pairs":
        out = list(seq)
        for i in range(0, len(out) - 1, 2):
            out[i], out[i + 1] = out[i + 1], out[i]
        return out
    if name == "drop_every":
        k = op["k"]
        return [s for i, s in enumerate(seq) if (i + 1) % k != 0] or list(seq[:1])
    if name == "duplicate_first":
        return ([seq[0]] + seq) if seq else seq
    return [op["sigma"][s] for s in seq]


class SequenceTransformFamily(TaskFamily):
    family_id = "sequence_transform"
    canon_mode = "token_list"
    symbol_pool = SYMBOL_POOL

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        n_ops = int(rng.integers(2, 4))
        chosen = [OPS[int(i)] for i in
                  rng.choice(len(OPS), size=n_ops, replace=False)]
        ops = []
        for name in chosen:
            if name == "rotate":
                ops.append({"op": name, "k": int(rng.integers(1, 4))})
            elif name == "drop_every":
                ops.append({"op": name, "k": int(rng.integers(2, 4))})
            elif name == "map_symbol":
                ops.append({"op": name,
                            "sigma": [int(i) for i in rng.permutation(N_SYMBOLS)]})
            else:
                ops.append({"op": name})
        return Rule(self.family_id, {"ops": ops})

    def _run(self, rule: Rule, seq: list[int]) -> list[int]:
        out = list(seq)
        for op in rule.params["ops"]:
            out = _apply(op, out)
        return out

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        length = _TRANSFER_LENGTH if kind == KIND_TRANSFER else _LENGTHS[difficulty]
        seq = [int(v) for v in rng.integers(0, N_SYMBOLS, size=length)]
        out = self._run(rule, seq)
        names = SYMBOL_POOL[:N_SYMBOLS]
        clauses = [
            "A hidden transformation rewrites sequences of symbols.",
            f"Alphabet: {syms(names)}",
            f"Input sequence: {syms(names[s] for s in seq)}",
        ]
        spec = PromptSpec(
            clauses=tuple(clauses),
            question="Apply the hidden transformation to the input sequence.",
            answer_format=("Reply with a final line: ANSWER: "
                           "<the output symbols separated by spaces>"),
            alphabet=tuple(names),
            labels=(),
            answer=syms(names[s] for s in out),
        )
        return Item(spec=spec, data={"seq": seq, "out": out})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        truth_tokens = truth.split()
        attempt_tokens = attempt.split() if attempt else []
        n = len(truth_tokens)
        names = tuple(letters(i) for i in range(n)) + ("TAIL", "ABSENT")
        position = None
        for i in range(min(n, len(attempt_tokens))):
            if truth_tokens[i] != attempt_tokens[i]:
                position = i
                break
        if position is None:
            if len(attempt_tokens) == n:
                position = n + 1
            elif len(attempt_tokens) > n:
                position = n
            else:
                position = len(attempt_tokens)
        band = 0 if len(attempt_tokens) == n else (
            1 if len(attempt_tokens) < n else 2)
        return Feedback("HINT-ALIGN", (
            HintField("length", "LEN", band, 3, ("EQ", "SHORT", "LONG")),
            HintField("first", "POS", position, n + 2, names),
        ))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        from ...episodes.parse import canonicalize

        item, plan = self._item_and_plan(rule, instance)
        names = [plan.map_symbol(s) for s in SYMBOL_POOL[:N_SYMBOLS]]
        ops = rule.params["ops"]
        candidates = []
        if len(ops) > 1:
            candidates.append(Rule(self.family_id, {"ops": ops[:-1]}))
            candidates.append(Rule(self.family_id, {"ops": list(reversed(ops))}))
        candidates.append(Rule(self.family_id, {"ops": ops + [{"op": "reverse"}]}))
        for variant in candidates:
            out = self._run(variant, item.data["seq"])
            text = canonicalize(" ".join(names[s] for s in out), self.canon_mode)
            if text is not None and text != instance.answer_canonical:
                return text
        return " ".join(reversed(instance.answer_canonical.split()))


FAMILY = SequenceTransformFamily()
