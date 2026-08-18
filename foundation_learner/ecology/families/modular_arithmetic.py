"""``modular_arithmetic`` — hidden polynomial residue algebra.

Mechanism: a hidden ``f(x) = (a*x^2 + b*x + c) mod m`` with degree in {1,2} and
``m`` in {7,11,13,17,23}.  The task is to compute ``f(x)`` for a displayed ``x``.
Answer: the residue as a plain integer.

Difficulty 0/1/2 -> ``x`` drawn below ``m`` / below 100 / below 10000.

TRANSFER SHIFT: composed application.  Transfer instances ask for ``f(f(x))``
with ``x`` drawn from a wider range than any training instance, so the shift is
both distributional and compositional.  The latent coefficients are unchanged.

Structured feedback: the circular direction of the attempt relative to the truth
modulo ``m`` plus a coarse distance band.  Neither the residue nor its digits
appear: like every family here, the hint is rendered from opaque letter codes.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, make_rng)
from ..surface_remap import PromptSpec, sym

MODULI = (7, 11, 13, 17, 23)
VAR_POOL = ("n", "k", "j", "z", "y", "t")
_XMAX = {0: None, 1: 100, 2: 10000}
_TRANSFER_XMAX = 100000


class ModularArithmeticFamily(TaskFamily):
    family_id = "modular_arithmetic"
    canon_mode = "int"
    symbol_pool = VAR_POOL

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        m = int(MODULI[int(rng.integers(0, len(MODULI)))])
        degree = int(rng.integers(1, 3))
        a = int(rng.integers(1, m)) if degree == 2 else 0
        b = int(rng.integers(1, m))
        c = int(rng.integers(0, m))
        return Rule(self.family_id, {"m": m, "degree": degree,
                                     "a": a, "b": b, "c": c})

    # -- semantics ----------------------------------------------------------

    @staticmethod
    def _f(rule: Rule, x: int) -> int:
        p = rule.params
        return (p["a"] * x * x + p["b"] * x + p["c"]) % p["m"]

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        m = rule.params["m"]
        transfer = kind == KIND_TRANSFER
        hi = _TRANSFER_XMAX if transfer else (_XMAX[difficulty] or m)
        x = int(rng.integers(0, hi))
        var = VAR_POOL[0]

        if transfer:
            value = self._f(rule, self._f(rule, x))
            clauses = [
                "A hidden integer function maps a whole number to a residue.",
                f"Let {sym(var)} = {x}.",
                "Apply the hidden function twice: first to the number, then to "
                "the result.",
            ]
            question = f"What is the value after applying the function twice to {sym(var)}?"
        else:
            value = self._f(rule, x)
            clauses = [
                "A hidden integer function maps a whole number to a residue.",
                f"Let {sym(var)} = {x}.",
            ]
            question = f"What is the value of the hidden function at {sym(var)}?"

        spec = PromptSpec(
            clauses=tuple(clauses),
            question=question,
            answer_format="Reply with a final line: ANSWER: <a whole number>",
            alphabet=(var,),
            labels=(),
            answer=str(value),
        )
        return Item(spec=spec, data={"x": x, "value": value, "m": m,
                                     "transfer": transfer})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        m = item.data["m"]
        true_value = int(truth)
        try:
            guess = int(attempt) % m if attempt is not None else None
        except ValueError:
            guess = None
        if guess is None:
            direction, band = 3, 3
        else:
            delta = (guess - true_value) % m
            if delta == 0:
                direction, band = 2, 0
            else:
                direction = 0 if delta <= m // 2 else 1
                distance = min(delta, m - delta)
                band = 1 if distance <= max(1, m // 4) else (
                    2 if distance <= m // 2 else 3)
        return Feedback("HINT-RESIDUE", (
            HintField("direction", "DIR", direction, 4,
                      ("AHEAD", "BEHIND", "EQUAL", "UNREAD")),
            HintField("band", "BND", band, 4, ("ZERO", "NEAR", "MID", "FAR")),
        ))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        m = rule.params["m"]
        truth = int(instance.answer_canonical)
        offset = 1 + int(rng.integers(0, m - 1))
        return str((truth + offset) % m)


FAMILY = ModularArithmeticFamily()
