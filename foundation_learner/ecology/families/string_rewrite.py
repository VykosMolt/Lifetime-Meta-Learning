"""``string_rewrite`` — hidden leftmost, non-overlapping rewrite system.

Mechanism: 2-3 hidden productions ``pattern -> replacement`` over a small
alphabet, applied leftmost and non-overlapping for a fixed number of passes.
One pass scans the string from the left; at each position the productions are
tried in rule order and the first match is rewritten, after which the scan
resumes past the replacement.  Replacements never exceed their pattern in
length, so the rewrite is length non-increasing and always terminates.
Answer: the rewritten string.

Difficulty 0/1/2 -> input length 6/9/12.

TRANSFER SHIFT: much longer inputs (length 20) give the hidden productions far
more chances to interact and to chain across a pass than any training instance
does.  The latent production set and pass count are unchanged.

Structured feedback: how many aligned positions of the attempt match the target
on their common prefix, plus a length band.  The target string is never stated.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, make_rng)
from ..surface_remap import PromptSpec, syms

SYMBOL_POOL = ("a", "b", "c", "d", "e", "f")
N_SYMBOLS = 3
_LENGTHS = {0: 6, 1: 9, 2: 12}
_TRANSFER_LENGTH = 20


def _one_pass(text: str, productions: list[list[str]]) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        for pattern, replacement in productions:
            if text.startswith(pattern, i):
                out.append(replacement)
                i += len(pattern)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


class StringRewriteFamily(TaskFamily):
    family_id = "string_rewrite"
    canon_mode = "string"
    symbol_pool = SYMBOL_POOL

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        alphabet = SYMBOL_POOL[:N_SYMBOLS]
        patterns: list[str] = []
        productions: list[list[str]] = []
        for _ in range(int(rng.integers(2, 4))):
            for _attempt in range(32):
                plen = int(rng.integers(1, 3))
                pattern = "".join(alphabet[int(i)] for i in
                                  rng.integers(0, N_SYMBOLS, size=plen))
                if pattern not in patterns:
                    break
            else:  # pragma: no cover - alphabet is far larger than 3 draws
                raise RuntimeError("string_rewrite: pattern space exhausted")
            patterns.append(pattern)
            rlen = int(rng.integers(1, len(pattern) + 1))
            replacement = "".join(alphabet[int(i)] for i in
                                  rng.integers(0, N_SYMBOLS, size=rlen))
            productions.append([pattern, replacement])
        return Rule(self.family_id, {"productions": productions,
                                     "passes": int(rng.integers(1, 3))})

    def _run(self, productions, passes: int, text: str) -> str:
        out = text
        for _ in range(passes):
            out = _one_pass(out, productions)
        return out

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        length = _TRANSFER_LENGTH if kind == KIND_TRANSFER else _LENGTHS[difficulty]
        alphabet = SYMBOL_POOL[:N_SYMBOLS]
        text = "".join(alphabet[int(i)] for i in
                       rng.integers(0, N_SYMBOLS, size=length))
        out = self._run(rule.params["productions"], rule.params["passes"], text)
        clauses = [
            "A hidden set of rewrite rules is applied to strings.",
            f"Alphabet: {syms(alphabet)}",
            f"Input string: {syms(text, sep='')}",
        ]
        spec = PromptSpec(
            clauses=tuple(clauses),
            question="Apply the hidden rewrite rules to the input string.",
            answer_format=("Reply with a final line: ANSWER: "
                           "<the rewritten string>"),
            alphabet=tuple(alphabet),
            labels=(),
            answer=syms(out, sep=""),
        )
        return Item(spec=spec, data={"text": text, "out": out})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        guess = attempt or ""
        matched = sum(1 for a, b in zip(truth, guess) if a == b)
        band = 0 if len(guess) == len(truth) else (
            1 if len(guess) < len(truth) else 2)
        return Feedback("HINT-MATCH", (
            HintField("aligned", "CNT", matched, len(truth) + 1),
            HintField("length", "LEN", band, 3, ("EQ", "SHORT", "LONG")),
        ))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        from ...episodes.parse import canonicalize

        item, plan = self._item_and_plan(rule, instance)
        prods = rule.params["productions"]
        passes = rule.params["passes"]
        variants = [(prods[:1], passes), (prods, passes + 1),
                    (list(reversed(prods)), passes)]
        for productions, n_passes in variants:
            out = self._run(productions, n_passes, item.data["text"])
            text = canonicalize("".join(plan.map_symbol(ch) for ch in out),
                                self.canon_mode)
            if text is not None and text != instance.answer_canonical:
                return text
        return instance.answer_canonical[::-1]


FAMILY = StringRewriteFamily()
