"""``boolean_rule`` — truth-table evaluation of a hidden bounded DNF.

Mechanism: a hidden Boolean function over k in {3,4,5} variables written as a
bounded DNF (2-4 terms, term width <= 3).  The task is to evaluate it on a
displayed assignment.  Answer: ``0``/``1``.

Difficulty (0/1/2) adds 0/1/2 displayed DISTRACTOR variables that the hidden
rule ignores, so the learner must also identify which variables matter.

TRANSFER SHIFT: composed application.  A transfer instance shows the widest
assignment (k + 4 variables) and asks for the rule value AFTER flipping a named
subset of the variables, i.e. ``f(flip_S(a))`` rather than ``f(a)``.  The latent
rule is unchanged; the input distribution and the required composition are not.

Structured feedback: the display index of one variable whose flip changes the
output for the queried (post-flip) assignment, or ``ABSENT`` when no single
flip changes it.  The output itself is never stated.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, letters, make_rng)
from ..surface_remap import PromptSpec, lab, sym

VAR_POOL = tuple(f"x{i}" for i in range(12))


class BooleanRuleFamily(TaskFamily):
    family_id = "boolean_rule"
    canon_mode = "label"
    symbol_pool = VAR_POOL
    # NOTE: answer labels must never be a single letter.  A one-letter label
    # collides with the opaque letter codes used by every structured hint
    # (``VAR_F`` would then contain the answer token ``F``); the hostile
    # fixture tests/hostile/test_hostile_feedback_answer_leak.py found exactly
    # that with an earlier ("F","T") variant.
    label_variants = (("LO", "HI"), ("FALSE", "TRUE"), ("DOWN", "UP"))

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        k = int(rng.integers(3, 6))
        n_terms = int(rng.integers(2, 5))
        terms = []
        for _ in range(n_terms):
            width = int(rng.integers(1, min(3, k) + 1))
            varsel = sorted(int(v) for v in rng.choice(k, size=width, replace=False))
            terms.append([[v, int(rng.integers(0, 2))] for v in varsel])
        return Rule(self.family_id, {"k": k, "terms": terms})

    # -- semantics ----------------------------------------------------------

    @staticmethod
    def _evaluate(rule: Rule, assignment: list[int]) -> int:
        for term in rule.params["terms"]:
            if all(assignment[v] == (0 if neg else 1) for v, neg in term):
                return 1
        return 0

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        k = rule.params["k"]
        transfer = kind == KIND_TRANSFER
        n_shown = k + (4 if transfer else difficulty)
        names = [VAR_POOL[i] for i in range(n_shown)]
        shown = [int(b) for b in rng.integers(0, 2, size=n_shown)]

        flips: list[int] = []
        if transfer:
            n_flips = int(rng.integers(1, 3))
            flips = sorted(int(i) for i in
                           rng.choice(n_shown, size=n_flips, replace=False))
        effective = list(shown)
        for i in flips:
            effective[i] = 1 - effective[i]
        value = self._evaluate(rule, effective[:k])

        assign = ", ".join(f"{sym(names[i])}={shown[i]}" for i in range(n_shown))
        clauses = [
            "A hidden Boolean rule maps an assignment of variables to a value.",
            f"Assignment: {assign}",
        ]
        if transfer:
            flipped = ", ".join(sym(names[i]) for i in flips)
            clauses.append(
                f"Before applying the hidden rule, flip these variables: {flipped}.")
            question = ("What value does the hidden rule give after the listed "
                        "variables are flipped?")
        else:
            question = "What value does the hidden rule give for this assignment?"

        spec = PromptSpec(
            clauses=tuple(clauses),
            question=question,
            answer_format=f"Reply with a final line: ANSWER: {lab('0')} or {lab('1')}",
            alphabet=tuple(names),
            labels=("0", "1"),
            answer=lab(str(value)),
        )
        return Item(spec=spec, data={"names": names, "shown": shown,
                                     "effective": effective, "flips": flips,
                                     "value": value, "n_shown": n_shown})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        k = rule.params["k"]
        effective = item.data["effective"]
        n_shown = item.data["n_shown"]
        base = self._evaluate(rule, effective[:k])
        pivots = []
        for i in range(n_shown):
            probe = list(effective)
            probe[i] = 1 - probe[i]
            if self._evaluate(rule, probe[:k]) != base:
                pivots.append(i)
        names = tuple(letters(i) for i in range(n_shown)) + ("ABSENT",)
        value = int(rng.choice(pivots)) if pivots else n_shown
        return Feedback("HINT-PIVOT", (
            HintField("pivot", "VAR", value, n_shown + 1, names),))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        return self._flip_label(rule, instance)


FAMILY = BooleanRuleFamily()
