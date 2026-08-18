"""``dsl_execution`` — execution of a hidden register program.

Mechanism: a hidden program of 3-5 instructions in a tiny register machine
``{INC r, DEC r, ADD r s, SWAP r s, JNZ r offset}`` over 2-3 registers, run from
the displayed initial register state under a bounded step budget.  Register
values saturate at +-999 so that ``ADD`` inside a ``JNZ`` loop cannot produce
unbounded literals; execution stops at the step budget.  Answer: the final
register state, e.g. ``r0=3 r1=0``.

Difficulty 0/1/2 -> initial register values drawn from 0..2 / 0..5 / 0..9.

TRANSFER SHIFT: much larger initial values (0..20) with a proportionally larger
step budget, so ``JNZ`` loops run far longer than in any training instance.  The
latent program is unchanged.

Structured feedback: a per-register OK/BAD flag saying WHICH registers are
wrong, never their values.  Registers are addressed by their opaque slot index
in the displayed order (``slot_one`` is the first register shown), which keeps
the hint free of any answer token.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, make_rng)
from ..surface_remap import PromptSpec, sym

REGISTER_POOL = ("r0", "r1", "r2", "q0", "q1", "q2", "w0", "w1", "w2")
INSTRUCTIONS = ("INC", "DEC", "ADD", "SWAP", "JNZ")
SLOT_NAMES = ("slot_one", "slot_two", "slot_three")
_INIT_MAX = {0: 3, 1: 6, 2: 10}
_TRANSFER_INIT_MAX = 21
_STEP_BUDGET = 64
_TRANSFER_STEP_BUDGET = 256
#: register values saturate here, so ADD loops cannot produce huge literals
_CLAMP = 999


class DslExecutionFamily(TaskFamily):
    family_id = "dsl_execution"
    canon_mode = "assignment"
    symbol_pool = REGISTER_POOL

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        n_regs = int(rng.integers(2, 4))
        n_instr = int(rng.integers(3, 6))
        program = []
        for index in range(n_instr):
            op = INSTRUCTIONS[int(rng.integers(0, len(INSTRUCTIONS)))]
            if op in ("INC", "DEC"):
                program.append({"op": op, "r": int(rng.integers(0, n_regs))})
            elif op in ("ADD", "SWAP"):
                r, s = (int(x) for x in rng.choice(n_regs, size=2, replace=False))
                program.append({"op": op, "r": r, "s": s})
            else:
                offset = int(rng.integers(-2, 3))
                if offset == 0:
                    offset = -1
                program.append({"op": op, "r": int(rng.integers(0, n_regs)),
                                "offset": offset})
        return Rule(self.family_id, {"n_regs": n_regs, "program": program})

    def _run(self, rule: Rule, initial: list[int], budget: int) -> list[int]:
        regs = list(initial)
        program = rule.params["program"]
        pc = 0
        steps = 0
        while 0 <= pc < len(program) and steps < budget:
            instr = program[pc]
            op = instr["op"]
            if op == "INC":
                regs[instr["r"]] += 1
            elif op == "DEC":
                regs[instr["r"]] -= 1
            elif op == "ADD":
                regs[instr["r"]] += regs[instr["s"]]
            elif op == "SWAP":
                regs[instr["r"]], regs[instr["s"]] = \
                    regs[instr["s"]], regs[instr["r"]]
            elif op == "JNZ" and regs[instr["r"]] != 0:
                pc += instr["offset"]
                steps += 1
                continue
            regs = [max(-_CLAMP, min(_CLAMP, value)) for value in regs]
            pc += 1
            steps += 1
        return regs

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        n_regs = rule.params["n_regs"]
        transfer = kind == KIND_TRANSFER
        hi = _TRANSFER_INIT_MAX if transfer else _INIT_MAX[difficulty]
        budget = _TRANSFER_STEP_BUDGET if transfer else _STEP_BUDGET
        initial = [int(v) for v in rng.integers(0, hi, size=n_regs)]
        final = self._run(rule, initial, budget)

        names = [REGISTER_POOL[i] for i in range(n_regs)]
        state = " ".join(f"{sym(names[i])}={initial[i]}" for i in range(n_regs))
        clauses = [
            "A hidden program runs on a small register machine and then halts.",
            f"Initial registers: {state}",
            "The registers are listed in a fixed order.",
        ]
        if transfer:
            clauses.append("The program may loop for a long time on these values.")
        spec = PromptSpec(
            clauses=tuple(clauses),
            question="What are the register values after the hidden program halts?",
            answer_format=("Reply with a final line: ANSWER: "
                           "<name=value pairs separated by spaces>"),
            alphabet=tuple(names),
            labels=(),
            answer=" ".join(f"{sym(names[i])}={final[i]}" for i in range(n_regs)),
        )
        return Item(spec=spec, data={"initial": initial, "final": final,
                                     "n_regs": n_regs, "budget": budget})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        truth_map = dict(part.split("=") for part in truth.split())
        guess_map = {}
        if attempt:
            for part in attempt.split():
                if "=" in part:
                    key, value = part.split("=", 1)
                    guess_map[key] = value
        order = sorted(truth_map)
        fields = []
        for index, name in enumerate(order):
            ok = guess_map.get(name) == truth_map[name]
            fields.append(HintField(SLOT_NAMES[index], "FLG", 0 if ok else 1,
                                    2, ("OK", "BAD")))
        return Feedback("HINT-REG", tuple(fields))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        from ...episodes.parse import canonicalize

        item, plan = self._item_and_plan(rule, instance)
        names = [plan.map_symbol(REGISTER_POOL[i])
                 for i in range(item.data["n_regs"])]
        final = list(item.data["final"])
        index = int(rng.integers(0, len(final)))
        final[index] += 1 if int(rng.integers(0, 2)) == 0 else -1
        text = canonicalize(" ".join(f"{names[i]}={final[i]}"
                                     for i in range(len(final))),
                            self.canon_mode)
        return text if text is not None else instance.answer_canonical


FAMILY = DslExecutionFamily()
