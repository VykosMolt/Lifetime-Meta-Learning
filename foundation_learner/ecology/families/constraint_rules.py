"""``constraint_rules`` — satisfaction checking against a hidden constraint set.

Mechanism: a hidden constraint template over 3-5 named slots.  Every ordered
slot pair (i, j) carries one of ``LT`` (slot_i < slot_j), ``GT``, ``NEQ`` or no
constraint, and each slot may additionally carry a hidden forbidden value.  The
task is to decide whether a displayed full assignment satisfies ALL hidden
constraints.  Answer: ``SAT``/``UNSAT``.

Difficulty 0/1/2 -> assignment values drawn from 1..4 / 1..6 / 1..8.

TRANSFER SHIFT: the assignment is presented over EXTRA slots that the hidden
constraint set never mentions, drawn from a wider value range, so the surface
assignment is longer and noisier than any training instance.  The latent
constraint set is unchanged.

Structured feedback (generator 1.1.0): a CANDIDATE-CONSTRAINT PROBE.
====================================================================
Generator 1.0.0 reported "the number of violated constraints" (§4.11).  That
count DECODES the answer: a count of zero is satisfaction and any positive
count is violation, so every branch of the hint determined the label with
certainty (measured P = 1.000 on all seven branches with n >= 30 over 2160
sampled items).  The hint never contained the answer TOKEN, but a reader could
read the answer off it without ever inferring the hidden constraint set, which
is the capability the task is supposed to measure.

Generator 1.1.0 reports instead:

* the identity of ONE CANDIDATE constraint, as a stable opaque positional code
  into :data:`CANDIDATE_POOL` — every pair among the first three displayed
  slots with one of ``LT`` / ``GT`` / ``NEQ``, and every "slot must not equal
  v" candidate over those slots and the forbidden-value range.  The candidate
  NEED NOT be one of the hidden rule's active constraints;
* whether the DISPLAYED assignment satisfies or violates that candidate.

Because an inactive candidate can be violated while the assignment is SAT under
the hidden rule (and an active one can hold while another is broken), no branch
of this hint implies the label.

The probe is selected from DISPLAYED data only (the assignment values and the
number of displayed slots), never from the hidden constraint set and never from
the answer, so the selection channel carries no satisfaction information either.
It is a pure function of the instance, so both attempts on one item see the same
probe.

WHY IT IS STILL INFORMATIVE.  The probe identity is a STABLE POSITIONAL code:
slots are named by their displayed position and the relation by a fixed code, so
the same code always means the same candidate constraint.  Across feedback
rounds a learner accumulates (candidate, holds/broken, certified verdict)
triples over many items and eliminates candidate constraint sets that cannot
explain the observed verdicts — which is the rule identification this family
exists to test.  Recorded honestly in Amendment 14: the probe STATUS is
computable from the displayed assignment, so the hint's value is that it points
at a hypothesis and pre-computes it, not that it discloses hidden state.  The
alternative — also reporting whether the candidate is ACTIVE in the hidden rule
— was rejected because "active AND broken" implies UNSAT with certainty, i.e. it
would reintroduce exactly the defect being repaired.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, MAX_HINT_OPTIONS,
                    Rule, TaskFamily, TaskInstance, canonical_json,
                    derive_seed, letters, make_rng)
from ..surface_remap import PromptSpec, lab, sym

SLOT_POOL = ("sl", "sm", "sn", "so", "sp", "sq", "sr", "ss", "st")
COMPARATORS = ("none", "LT", "GT", "NEQ")
_VALUE_MAX = {0: 5, 1: 7, 2: 9}
_TRANSFER_VALUE_MAX = 12
_TRANSFER_EXTRA_SLOTS = 3

#: relations a candidate PAIR constraint may carry (the rule's own comparators)
PROBE_RELATIONS = ("LT", "GT", "NEQ")
#: the rule draws forbidden values from 1..5, so candidate SLOT constraints
#: ("slot i must not equal v") range over exactly that set
PROBE_FORBIDDEN_MAX = 5
#: Status codes.  They may NOT collide with any answer label of this family
#: (``SAT``/``UNSAT``, ``VALID``/``INVALID``, ``HOLDS``/``FAILS``,
#: ``PASS``/``REJECT``) under any surface, or the frozen answer-leak invariant
#: would fire on a legitimate hint — the ("F","T") lesson of Amendment 7.4.
PROBE_STATUS = ("MET", "UNMET")


#: The probe ranges over the first three displayed slots ONLY, and therefore
#: over a pool that is IDENTICAL for every instance of the family.  This is not
#: cosmetic: a pool whose SIZE depends on the displayed slot count makes the
#: opaque code disclose that count, and the displayed slot count is itself
#: strongly predictive of the label (a rule over five slots carries more
#: constraints and is almost always UNSAT).  Measured with a slot-count
#: dependent pool, 13 of 55 code branches predicted UNSAT with certainty for
#: exactly that reason.  Every rule has at least three slots (``sample_rule``
#: draws ``n_slots`` in 3..5) and every instance displays at least ``n_slots``,
#: so these three candidates are always applicable and the code is a globally
#: stable positional identifier.
MAX_PROBE_SLOTS = 3


def _build_candidate_pool() -> tuple[tuple, ...]:
    """The FROZEN candidate-constraint pool a hint may probe (Amendment 14).

    Enumeration order (fixed and instance-independent, so an opaque index is a
    stable positional code a learner can decode and accumulate evidence
    against):

    1. every ``("PAIR", i, j, relation)`` with
       ``0 <= i < j < MAX_PROBE_SLOTS`` and ``relation`` in
       :data:`PROBE_RELATIONS`, ordered by ``(i, j, relation)``;
    2. every ``("SLOT", i, v)`` with ``0 <= i < MAX_PROBE_SLOTS`` and
       ``1 <= v <= PROBE_FORBIDDEN_MAX``, ordered by ``(i, v)``.

    A candidate need NOT be an active constraint of the hidden rule; that is
    exactly why the probe's status cannot imply the label.
    """
    pool: list[tuple] = []
    for i in range(MAX_PROBE_SLOTS):
        for j in range(i + 1, MAX_PROBE_SLOTS):
            for relation in PROBE_RELATIONS:
                pool.append(("PAIR", i, j, relation))
    for i in range(MAX_PROBE_SLOTS):
        for value in range(1, PROBE_FORBIDDEN_MAX + 1):
            pool.append(("SLOT", i, value))
    if not pool or len(pool) > MAX_HINT_OPTIONS:
        raise ValueError(
            f"constraint_rules candidate pool has {len(pool)} entries, outside "
            f"(0, {MAX_HINT_OPTIONS}]; the opaque hint code must stay inside "
            "the frozen cap")
    return tuple(pool)


#: frozen, instance-independent candidate pool (see :func:`_build_candidate_pool`)
CANDIDATE_POOL: tuple[tuple, ...] = _build_candidate_pool()

# --------------------------------------------------------------------------
# label balance (generator 1.2.0, Amendment 15)
# --------------------------------------------------------------------------
#: Target share of SAT items.  Generator 1.1.0 produced 10.7 % SAT, so a
#: constant "always answer UNSAT" policy scored 0.893 on this family's R_k and
#: every learning curve had to be read against that floor.
LABEL_TARGET_RATE = 0.5
#: Deterministic rejection budget per item.  512 draws make the target label
#: reachable for every rule that survives the rule-level condition below; the
#: fallback is the FIRST draw, so an item is always well defined.
LABEL_DRAW_ATTEMPTS = 512
#: Rule-level condition.  14 % of 1.1.0 rules were outright UNSATISFIABLE at
#: the frozen campaign difficulty and a further 14 % were satisfiable only with
#: probability < 0.005, so NO per-item rejection budget can balance them.
#: Rules whose estimated satisfiability rate is below this floor are redrawn.
#: The floor is deliberately the mildest one that works: it keeps ~72 % of the
#: 1.1.0 rule space (the band [0.2, 0.8] would have kept only 19 %).
RULE_MIN_SAT_RATE = 0.005
#: Monte-Carlo draws used to estimate a rule's satisfiability rate.  Seeded
#: from the RULE ITSELF, so the estimate — and therefore the rule — stays a
#: pure function with no dependence on the caller's generator state.
RULE_SAT_PROBE_DRAWS = 256
#: Bounded deterministic redraw budget; exhausting it is a hard error rather
#: than a silent acceptance of an unbalanceable rule.
RULE_BALANCE_ATTEMPTS = 64
#: The difficulty at which the rule condition is evaluated: the frozen campaign
#: difficulty (contract section 4, "Default campaign difficulty: level 1").
RULE_REFERENCE_DIFFICULTY = 1


def candidate_holds(candidate: tuple, values: list[int]) -> bool:
    """Does the DISPLAYED assignment satisfy this candidate constraint?"""
    if candidate[0] == "PAIR":
        _kind, i, j, relation = candidate
        a, b = values[i], values[j]
        if relation == "LT":
            return a < b
        if relation == "GT":
            return a > b
        return a != b
    _kind, i, value = candidate
    return values[i] != value


class ConstraintRulesFamily(TaskFamily):
    family_id = "constraint_rules"
    #: 1.0.0 -> 1.1.0: candidate-constraint probe replaces the violation
    #: count, which decoded the label (Amendment 14).
    #: 1.1.0 -> 1.2.0: label-balanced item sampler (Amendment 15).
    generator_version = "1.2.0"
    canon_mode = "label"
    symbol_pool = SLOT_POOL
    label_variants = (("VALID", "INVALID"), ("HOLDS", "FAILS"),
                      ("PASS", "REJECT"))

    def _draw_rule(self, rng: np.random.Generator) -> Rule:
        """One unconditioned draw from the frozen contract section 4.11 rule
        space (this is generator 1.0.0's ``sample_rule``, unchanged)."""
        n_slots = int(rng.integers(3, 6))
        pairs = []
        for i in range(n_slots):
            for j in range(i + 1, n_slots):
                comparator = COMPARATORS[int(rng.integers(0, len(COMPARATORS)))]
                if comparator != "none":
                    pairs.append([i, j, comparator])
        forbidden = []
        for i in range(n_slots):
            if int(rng.integers(0, 3)) == 0:
                forbidden.append([i, int(rng.integers(1, 6))])
        if not pairs and not forbidden:
            pairs.append([0, 1, "LT"])
        return Rule(self.family_id, {"n_slots": n_slots, "pairs": pairs,
                                     "forbidden": forbidden})

    def satisfiability_rate(self, rule: Rule) -> float:
        """Estimated P(a random assignment satisfies this rule) at the frozen
        campaign difficulty.

        A PURE function of the rule: the Monte-Carlo generator is derived from
        the rule spec, never from the caller's stream, so the same rule always
        gets the same estimate no matter who asks or when.
        """
        n_slots = int(rule.params["n_slots"])
        hi = _VALUE_MAX[RULE_REFERENCE_DIFFICULTY]
        probe = make_rng(derive_seed(0, self.family_id, "SAT_PROBE",
                                     canonical_json(rule.params)))
        draws = probe.integers(1, hi, size=(RULE_SAT_PROBE_DRAWS, n_slots))
        hits = sum(1 for row in draws
                   if self._violations(rule, [int(v) for v in row]) == 0)
        return hits / RULE_SAT_PROBE_DRAWS

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        """Draw a latent rule that CAN produce both labels (Amendment 15).

        14 % of unconditioned rules are unsatisfiable at the frozen campaign
        difficulty and a further 14 % are satisfiable with probability < 0.005,
        so for those no item sampler can produce a SAT item at all.  Such rules
        are redrawn from the SAME frozen rule space until
        :meth:`satisfiability_rate` clears :data:`RULE_MIN_SAT_RATE`.  This is
        the mildest condition that makes per-item balancing possible: it keeps
        about 72 % of the rule space and touches only the degenerate tail.
        Exhausting the deterministic redraw budget is a hard error, never a
        silent acceptance.
        """
        for _attempt in range(RULE_BALANCE_ATTEMPTS):
            rule = self._draw_rule(rng)
            if self.satisfiability_rate(rule) >= RULE_MIN_SAT_RATE:
                return rule
        raise ValueError(
            f"{self.family_id}: no rule with satisfiability rate >= "
            f"{RULE_MIN_SAT_RATE} was drawn in {RULE_BALANCE_ATTEMPTS} "
            "deterministic attempts")

    @staticmethod
    def _violations(rule: Rule, values: list[int]) -> int:
        count = 0
        for i, j, comparator in rule.params["pairs"]:
            a, b = values[i], values[j]
            if comparator == "LT" and not a < b:
                count += 1
            elif comparator == "GT" and not a > b:
                count += 1
            elif comparator == "NEQ" and not a != b:
                count += 1
        for i, bad in rule.params["forbidden"]:
            if values[i] == bad:
                count += 1
        return count

    def _n_constraints(self, rule: Rule) -> int:
        """Size of the hidden constraint set.

        Kept as the family's own description of its latent rule (manifests and
        analyses may report it).  It is deliberately NOT used by the hint any
        more: generator 1.0.0 reported the violated COUNT against this total,
        and that count decoded the answer (Amendment 14).
        """
        return len(rule.params["pairs"]) + len(rule.params["forbidden"])

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        n_slots = rule.params["n_slots"]
        transfer = kind == KIND_TRANSFER
        hi = _TRANSFER_VALUE_MAX if transfer else _VALUE_MAX[difficulty]
        shown = min(n_slots + (_TRANSFER_EXTRA_SLOTS if transfer else 0),
                    len(SLOT_POOL))
        # LABEL BALANCE (Amendment 15).  The item draws its own fair target
        # label from its own generator and then REJECTION-SAMPLES assignments
        # until one carries that label.  Assignments are still drawn uniformly
        # from the value grid, so a balanced item has no distributional
        # signature beyond the one every conditional sampler has (a SAT item's
        # assignment is a uniform draw from the satisfying set); nothing about
        # the ORDER or SHAPE of the values encodes the label.  The instance
        # stays a pure function of (rule, seed, difficulty, surface_map_id):
        # the target, every candidate and the fallback all come from ``rng``.
        want_sat = int(rng.integers(0, 2)) == 0
        values: list[int] | None = None
        for _attempt in range(LABEL_DRAW_ATTEMPTS):
            candidate = [int(v) for v in rng.integers(1, hi, size=shown)]
            if values is None:
                values = candidate          # fallback: the first draw
            if (self._violations(rule, candidate[:n_slots]) == 0) is want_sat:
                values = candidate
                break
        assert values is not None           # LABEL_DRAW_ATTEMPTS >= 1
        violations = self._violations(rule, values[:n_slots])
        answer = "SAT" if violations == 0 else "UNSAT"

        assignment = ", ".join(f"{sym(SLOT_POOL[i])}={values[i]}"
                               for i in range(shown))
        clauses = [
            "A hidden set of constraints restricts how the slots may be filled.",
            f"Assignment: {assignment}",
        ]
        if transfer:
            clauses.append("Some slots shown here may not be constrained at all.")
        spec = PromptSpec(
            clauses=tuple(clauses),
            question="Does this assignment satisfy every hidden constraint?",
            answer_format=(f"Reply with a final line: ANSWER: {lab('SAT')} "
                           f"or {lab('UNSAT')}"),
            alphabet=tuple(SLOT_POOL[:shown]),
            labels=("SAT", "UNSAT"),
            answer=lab(answer),
        )
        return Item(spec=spec, data={"values": values, "shown": shown,
                                     "violations": violations})

    # -- feedback -----------------------------------------------------------

    def _probe_index(self, item: Item) -> int:
        """Which candidate constraint this item's hint reports on.

        Seeded from DISPLAYED data only (the assignment values and the number of
        displayed slots).  The hidden constraint set and the answer are absent
        from the seed, so the selection channel cannot carry the satisfaction
        bit.  Pure function of the instance, so both attempts on one item see
        the same probe.  The RANGE is the instance-independent
        :data:`CANDIDATE_POOL`, so the code itself discloses nothing either.
        """
        values = [int(v) for v in item.data["values"]]
        shown = int(item.data["shown"])
        rng = make_rng(derive_seed(0, self.family_id, "PROBE", values, shown))
        return int(rng.integers(0, len(CANDIDATE_POOL)))

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        values = [int(v) for v in item.data["values"]]
        pool = CANDIDATE_POOL
        index = self._probe_index(item)
        names = tuple(letters(i) for i in range(len(pool)))
        # ONE opaque positional code identifies the candidate, so the
        # deterministic corruption can only move it to ANOTHER WELL-FORMED
        # candidate (and/or flip the status).  A multi-field encoding would let
        # corruption emit an impossible record (left == right under LT), and a
        # malformed hint would disclose that the hint is poisoned - contract
        # section 6 requires truthful and poisoned hints to render identically.
        return Feedback("HINT-PROBE", (
            HintField("probe", "CND", index, len(pool), names),
            HintField("status", "STA",
                      0 if candidate_holds(pool[index], values) else 1,
                      len(PROBE_STATUS), PROBE_STATUS),
        ))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        return self._flip_label(rule, instance)


FAMILY = ConstraintRulesFamily()
