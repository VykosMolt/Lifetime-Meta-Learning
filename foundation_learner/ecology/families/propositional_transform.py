"""``propositional_transform`` — tree rewriting on propositional formulas.

Mechanism: a hidden syntactic transformation ``T`` on formula trees over
``~``, ``&``, ``|``.  ``T`` is parameterised by (frozen order of application):

1. ``push_negation`` — De Morgan: push every ``~`` down to the leaves;
2. ``swap_ops`` — exchange ``&`` and ``|`` at every internal node;
3. ``neg_exempt`` — a hidden subset of variables whose literal negation is
   dropped (``~x`` becomes ``x``);
4. ``rename`` — a hidden permutation of the variable names;
5. ``sort_children`` — order the children of every binary node by their
   rendered string.

``T`` is a string-to-string rewrite; it is not required to preserve truth.
The task is to output the transformed formula in the canonical fully
parenthesised rendering.  Answer canonicalisation: all whitespace removed,
lowercased.

Difficulty 0/1/2 -> 3/4/5 leaves.

TRANSFER SHIFT: longer, deeper inputs (7 leaves, more nesting than any training
instance) — the same latent ``T`` applied to a strictly larger formula
distribution.

Structured feedback: the pre-order token position of the canonical rendering at
which the attempt first diverges from the target (``TAIL`` when the attempt only
runs on past the end, ``ABSENT`` when it matches), plus a length band.  The
target string is never stated.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, letters, make_rng)
from ..surface_remap import PromptSpec, sym

VAR_POOL = ("p", "q", "r", "s", "t", "u", "v", "w")
_LEAVES = {0: 3, 1: 4, 2: 5}
_TRANSFER_LEAVES = 7


def _render(node, name_of) -> str:
    kind = node[0]
    if kind == "var":
        return name_of(node[1])
    if kind == "not":
        return "~" + _render(node[1], name_of)
    return "(" + _render(node[1], name_of) + ("&" if kind == "and" else "|") \
        + _render(node[2], name_of) + ")"


def _tokens(text: str) -> list[str]:
    """Pre-order token stream of a canonical rendering."""
    out: list[str] = []
    buf = ""
    for ch in text:
        if ch.isalnum():
            buf += ch
            continue
        if buf:
            out.append(buf)
            buf = ""
        out.append(ch)
    if buf:
        out.append(buf)
    return out


def _push(node, negated: bool):
    kind = node[0]
    if kind == "var":
        return ("not", node) if negated else node
    if kind == "not":
        return _push(node[1], not negated)
    if negated:
        flipped = "or" if kind == "and" else "and"
        return (flipped, _push(node[1], True), _push(node[2], True))
    return (kind, _push(node[1], False), _push(node[2], False))


def _swap(node):
    kind = node[0]
    if kind == "var":
        return node
    if kind == "not":
        return ("not", _swap(node[1]))
    return ("or" if kind == "and" else "and", _swap(node[1]), _swap(node[2]))


def _drop_exempt(node, exempt: set[int]):
    kind = node[0]
    if kind == "var":
        return node
    if kind == "not":
        inner = _drop_exempt(node[1], exempt)
        if inner[0] == "var" and inner[1] in exempt:
            return inner
        return ("not", inner)
    return (kind, _drop_exempt(node[1], exempt), _drop_exempt(node[2], exempt))


def _rename(node, perm):
    kind = node[0]
    if kind == "var":
        return ("var", perm[node[1]])
    if kind == "not":
        return ("not", _rename(node[1], perm))
    return (kind, _rename(node[1], perm), _rename(node[2], perm))


def _sort_children(node, name_of):
    kind = node[0]
    if kind == "var":
        return node
    if kind == "not":
        return ("not", _sort_children(node[1], name_of))
    a = _sort_children(node[1], name_of)
    b = _sort_children(node[2], name_of)
    if _render(b, name_of) < _render(a, name_of):
        a, b = b, a
    return (kind, a, b)


def _random_formula(rng: np.random.Generator, n_leaves: int, n_vars: int):
    if n_leaves == 1:
        node = ("var", int(rng.integers(0, n_vars)))
        if rng.random() < 0.35:
            node = ("not", node)
        return node
    left = int(rng.integers(1, n_leaves))
    op = "and" if int(rng.integers(0, 2)) == 0 else "or"
    node = (op, _random_formula(rng, left, n_vars),
            _random_formula(rng, n_leaves - left, n_vars))
    if rng.random() < 0.2:
        node = ("not", node)
    return node


class PropositionalTransformFamily(TaskFamily):
    family_id = "propositional_transform"
    canon_mode = "formula"
    symbol_pool = VAR_POOL

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        n_vars = int(rng.integers(3, 6))
        perm = [int(i) for i in rng.permutation(n_vars)]
        exempt = sorted(int(v) for v in range(n_vars)
                        if int(rng.integers(0, 2)) == 1)
        return Rule(self.family_id, {
            "n_vars": n_vars,
            "push_negation": int(rng.integers(0, 2)),
            "swap_ops": int(rng.integers(0, 2)),
            "neg_exempt": exempt,
            "rename": perm,
            "sort_children": int(rng.integers(0, 2)),
        })

    # -- semantics ----------------------------------------------------------

    def _transform(self, rule: Rule, node, name_of):
        params = rule.params
        out = _push(node, False) if params["push_negation"] else node
        if params["swap_ops"]:
            out = _swap(out)
        out = _drop_exempt(out, set(params["neg_exempt"]))
        out = _rename(out, params["rename"])
        if params["sort_children"]:
            out = _sort_children(out, name_of)
        return out

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        n_vars = rule.params["n_vars"]
        n_leaves = _TRANSFER_LEAVES if kind == KIND_TRANSFER else _LEAVES[difficulty]
        tree = _random_formula(rng, n_leaves, n_vars)

        plain = (lambda i: VAR_POOL[i])
        marked = (lambda i: sym(VAR_POOL[i]))
        target = self._transform(rule, tree, plain)

        clauses = [
            "A hidden rewrite rule turns one propositional formula into another.",
            "Formulas use ~ for not, & for and, | for or, fully parenthesised.",
            f"Input formula: {_render(tree, marked)}",
        ]
        spec = PromptSpec(
            clauses=tuple(clauses),
            question="Apply the hidden rewrite rule to the input formula.",
            answer_format=("Reply with a final line: ANSWER: "
                           "<the rewritten formula>"),
            alphabet=tuple(VAR_POOL[:n_vars]),
            labels=(),
            answer=_render(target, marked),
        )
        return Item(spec=spec, data={"tree": tree, "target": target,
                                     "n_leaves": n_leaves})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        truth_tokens = _tokens(truth)
        attempt_tokens = _tokens(attempt) if attempt else []
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
        length_band = 0 if len(attempt_tokens) == n else (
            1 if len(attempt_tokens) < n else 2)
        return Feedback("HINT-DIVERGE", (
            HintField("first", "POS", position, n + 2, names),
            HintField("length", "LEN", length_band, 3, ("EQ", "SHORT", "LONG")),
        ))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        from ...episodes.parse import canonicalize

        item, plan = self._item_and_plan(rule, instance)
        name_of = (lambda i: plan.map_symbol(VAR_POOL[i]))
        for key in ("swap_ops", "push_negation", "sort_children"):
            variant = Rule(self.family_id, dict(rule.params))
            variant.params[key] = 1 - int(rule.params[key])
            text = canonicalize(_render(self._transform(variant, item.data["tree"],
                                                        name_of), name_of),
                                self.canon_mode)
            if text is not None and text != instance.answer_canonical:
                return text
        mangled = instance.answer_canonical.replace("&", "|", 1)
        if mangled == instance.answer_canonical:
            mangled = instance.answer_canonical.replace("|", "&", 1)
        if mangled == instance.answer_canonical:
            mangled = "~" + instance.answer_canonical
        return mangled


FAMILY = PropositionalTransformFamily()
