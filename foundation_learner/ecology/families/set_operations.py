"""``set_operations`` — hidden set-algebra expression.

Mechanism: a hidden expression of depth <= 3 over union, intersection and
difference, referring to 2-4 named base sets.  Each instance displays fresh
contents for those base sets; the task is to list the elements of the hidden
expression's value.  Answer: the element list (canonicalised as a sorted,
de-duplicated set; ``EMPTY`` when the value is empty).

Difficulty 0/1/2 -> universe size 5/7/9.

TRANSFER SHIFT: a larger universe (12 elements), denser base sets and one extra
DECOY base set that the hidden expression never mentions, so the surface of the
problem grows in a direction training instances never show.  The latent
expression is unchanged.

Structured feedback: the cardinality of the symmetric difference between the
attempt's set and the target set, plus a size band.  The target set is never
stated.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, make_rng)
from ..surface_remap import PromptSpec, sym, syms

SET_NAMES = ("sa", "sb", "sc", "sd", "se")
ELEMENTS = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l")
SYMBOL_POOL = SET_NAMES + ELEMENTS
OPS = ("union", "inter", "minus")
_UNIVERSE = {0: 5, 1: 7, 2: 9}
_TRANSFER_UNIVERSE = 12


def _build_expr(rng: np.random.Generator, n_leaves: int, n_sets: int):
    if n_leaves == 1:
        return ("set", int(rng.integers(0, n_sets)))
    left = int(rng.integers(1, n_leaves))
    op = OPS[int(rng.integers(0, len(OPS)))]
    return (op, _build_expr(rng, left, n_sets),
            _build_expr(rng, n_leaves - left, n_sets))


def _eval(node, sets: list[set]) -> set:
    kind = node[0]
    if kind == "set":
        return set(sets[node[1]])
    left = _eval(node[1], sets)
    right = _eval(node[2], sets)
    if kind == "union":
        return left | right
    if kind == "inter":
        return left & right
    return left - right


class SetOperationsFamily(TaskFamily):
    family_id = "set_operations"
    canon_mode = "set"
    symbol_pool = SYMBOL_POOL
    symbol_blocks = (SET_NAMES, ELEMENTS)

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        n_sets = int(rng.integers(2, 5))
        n_leaves = int(rng.integers(2, 5))
        expr = _build_expr(rng, n_leaves, n_sets)
        return Rule(self.family_id, {"n_sets": n_sets,
                                     "expr": _to_json(expr)})

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        n_sets = rule.params["n_sets"]
        transfer = kind == KIND_TRANSFER
        universe = _TRANSFER_UNIVERSE if transfer else _UNIVERSE[difficulty]
        expr = _from_json(rule.params["expr"])

        shown = n_sets + 1 if transfer else n_sets
        shown = min(shown, len(SET_NAMES))
        sets: list[set] = []
        for i in range(shown):
            size = int(rng.integers(1, max(2, universe - 1)))
            picks = rng.choice(universe, size=size, replace=False)
            sets.append({ELEMENTS[int(p)] for p in picks})

        value = sorted(_eval(expr, sets))
        clauses = ["A hidden set expression combines the named sets below."]
        for i in range(shown):
            contents = syms(sorted(sets[i])) if sets[i] else "(nothing)"
            clauses.append(f"Set {sym(SET_NAMES[i])} contains: {contents}")
        spec = PromptSpec(
            clauses=tuple(clauses),
            question="Which elements does the hidden set expression contain?",
            answer_format=("Reply with a final line: ANSWER: "
                           "<the elements separated by spaces, or EMPTY>"),
            alphabet=tuple(SET_NAMES[:shown]) + tuple(ELEMENTS[:universe]),
            labels=(),
            answer=syms(value) if value else "EMPTY",
        )
        return Item(spec=spec, data={"sets": [sorted(s) for s in sets],
                                     "value": value, "universe": universe,
                                     "shown": shown})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        truth_set = set() if truth == "EMPTY" else set(truth.split())
        if attempt is None:
            guess: set = set()
        elif attempt == "EMPTY":
            guess = set()
        else:
            guess = set(attempt.split())
        universe = item.data["universe"]
        distance = min(len(truth_set ^ guess), universe)
        band = 0 if len(guess) == len(truth_set) else (
            1 if len(guess) < len(truth_set) else 2)
        return Feedback("HINT-SYMDIFF", (
            HintField("distance", "CRD", distance, universe + 1),
            HintField("size", "SZ", band, 3, ("EQ", "SMALL", "LARGE")),
        ))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        from ...episodes.parse import canonicalize

        item, plan = self._item_and_plan(rule, instance)
        sets = [{plan.map_symbol(e) for e in s} for s in item.data["sets"]]
        expr = _from_json(rule.params["expr"])
        for swap in (("union", "inter"), ("inter", "minus"), ("minus", "union")):
            variant = _swap_ops(expr, swap[0], swap[1])
            value = sorted(_eval(variant, sets))
            text = canonicalize(" ".join(value) if value else "EMPTY",
                                self.canon_mode)
            if text is not None and text != instance.answer_canonical:
                return text
        return "EMPTY" if instance.answer_canonical != "EMPTY" else \
            plan.map_symbol(ELEMENTS[0])


def _to_json(node):
    if node[0] == "set":
        return {"op": "set", "index": node[1]}
    return {"op": node[0], "left": _to_json(node[1]), "right": _to_json(node[2])}


def _from_json(node):
    if node["op"] == "set":
        return ("set", node["index"])
    return (node["op"], _from_json(node["left"]), _from_json(node["right"]))


def _swap_ops(node, a: str, b: str):
    if node[0] == "set":
        return node
    op = node[0]
    if op == a:
        op = b
    elif op == b:
        op = a
    return (op, _swap_ops(node[1], a, b), _swap_ops(node[2], a, b))


FAMILY = SetOperationsFamily()
