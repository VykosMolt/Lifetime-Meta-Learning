"""``permutation_composition`` — hidden group action on positions.

Mechanism: a hidden permutation ``sigma`` in ``S_n`` (n in {5,6,7}) acting on
the positions of a displayed list of symbols: position ``i`` of the output takes
the symbol at position ``sigma[i]`` of the input.  Answer: the rearranged list.

Difficulty 0/1/2 -> 0/1/2 repeated symbols in the input list.  Repeats make the
action harder to read off a single example because several positions look alike.

TRANSFER SHIFT: composed application.  Transfer instances ask for
``sigma o sigma`` (contract §4: "apply sigma or sigma o sigma") on a list with
the maximum number of repeats, i.e. the square of the latent group element
rather than the element itself.

Structured feedback: how many of the target's fixed points (positions whose
symbol is unchanged) the attempt also got right — a count only.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, make_rng)
from ..surface_remap import PromptSpec, syms

SYMBOL_POOL = ("al", "be", "ga", "de", "ep", "ze", "et", "th", "io", "ka")
_REPEATS = {0: 0, 1: 1, 2: 2}
_TRANSFER_REPEATS = 2


def _apply(sigma: list[int], values: list[str]) -> list[str]:
    return [values[sigma[i]] for i in range(len(sigma))]


class PermutationCompositionFamily(TaskFamily):
    family_id = "permutation_composition"
    canon_mode = "token_list"
    symbol_pool = SYMBOL_POOL

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        n = int(rng.integers(5, 8))
        for _ in range(64):
            sigma = [int(i) for i in rng.permutation(n)]
            if any(sigma[i] != i for i in range(n)):
                break
        return Rule(self.family_id, {"n": n, "sigma": sigma})

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        n = rule.params["n"]
        sigma = rule.params["sigma"]
        transfer = kind == KIND_TRANSFER
        repeats = _TRANSFER_REPEATS if transfer else _REPEATS[difficulty]

        pick = [int(i) for i in rng.choice(len(SYMBOL_POOL), size=n, replace=False)]
        values = [SYMBOL_POOL[i] for i in pick]
        for _ in range(repeats):
            src, dst = (int(v) for v in rng.choice(n, size=2, replace=False))
            values[dst] = values[src]

        out = _apply(sigma, values)
        if transfer:
            out = _apply(sigma, out)

        clauses = [
            "A hidden rearrangement always moves list positions the same way.",
            f"Input list: {syms(values)}",
        ]
        if transfer:
            clauses.append("Apply the hidden rearrangement twice in a row.")
            question = "What is the list after two applications?"
        else:
            question = "What is the list after the hidden rearrangement?"

        spec = PromptSpec(
            clauses=tuple(clauses),
            question=question,
            answer_format=("Reply with a final line: ANSWER: "
                           "<the rearranged list, separated by spaces>"),
            alphabet=tuple(values),
            labels=(),
            answer=syms(out),
        )
        return Item(spec=spec, data={"values": values, "out": out,
                                     "transfer": transfer, "n": n})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        source = plan.map_seq(item.data["values"])
        truth_tokens = truth.split()
        guess = attempt.split() if attempt else []
        n = len(truth_tokens)
        hit = 0
        for i in range(n):
            if truth_tokens[i] != source[i]:
                continue
            if i < len(guess) and guess[i] == source[i]:
                hit += 1
        return Feedback("HINT-FIXED", (
            HintField("kept", "CNT", hit, n + 1),))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        from ...episodes.parse import canonicalize

        item, plan = self._item_and_plan(rule, instance)
        values = plan.map_seq(item.data["values"])
        sigma = rule.params["sigma"]
        candidates = [_apply(sigma, values)]
        if item.data["transfer"]:
            candidates.insert(0, _apply(sigma, _apply(sigma, _apply(sigma, values))))
        inverse = [0] * len(sigma)
        for i, target in enumerate(sigma):
            inverse[target] = i
        candidates.append(_apply(inverse, values))
        for candidate in candidates:
            text = canonicalize(" ".join(candidate), self.canon_mode)
            if text is not None and text != instance.answer_canonical:
                return text
        return " ".join(reversed(instance.answer_canonical.split()))


FAMILY = PermutationCompositionFamily()
