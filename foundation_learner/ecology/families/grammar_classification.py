"""``grammar_classification`` — membership in a hidden regular language.

Mechanism: the hidden language is a Boolean combination of two atomic
predicates over the alphabet {a, b, c}.  There are eight predicate SCHEMAS,
each parameterised (so the atomic-predicate pool has roughly sixty members):

``count_mod``, ``ends_with``, ``starts_with``, ``contains_bigram``,
``length_mod``, ``no_occurrence``, ``first_before`` and ``count_at_least``.

Each atom may be negated and the two atoms are joined by a hidden ``AND`` or
``OR``.  Every predicate is regular, so the hidden language is regular.  The
task is to classify a string.  Answer: ``IN``/``OUT``.

Difficulty 0/1/2 -> string length 5/8/11.

TRANSFER SHIFT: much longer strings (length 18) where the modular-count and
bigram predicates behave very differently from the short training strings.  The
latent language is unchanged.

Structured feedback (generator 1.1.0): a POOL-PREDICATE PROBE.
==============================================================
Generator 1.0.0 reported WHICH of the two hidden atomic predicates the string
fails (``A`` / ``B`` / ``BOTH`` / ``ABSENT``, §4.12).  Two of those branches
DECODE the answer: failing NEITHER conjunct implies membership under both
admissible connectives, and failing BOTH implies non-membership under both
(measured P = 1.000 at n = 541 and n = 576 over 2160 sampled items).  A reader
could therefore read the label off the hint without inferring the hidden
language at all.

Generator 1.1.0 reports instead:

* the opaque index of ONE CANDIDATE predicate drawn from :data:`PREDICATE_POOL`
  — a FIXED, rule-independent enumeration of the family's eight predicate
  schemas over the displayed alphabet POSITIONS.  The probe need not be one of
  the hidden rule's two conjuncts;
* whether the string passes that candidate predicate.

A pool predicate outside the rule can fail while the string is IN (and pass
while it is OUT), so no branch of this hint implies the label.  The probe is
selected from the DISPLAYED string only — never from the rule, never from the
answer — so the selection channel carries no membership information either.

WHY IT IS STILL INFORMATIVE.  :data:`PREDICATE_POOL` is frozen and ordered, and
it is indexed over alphabet POSITIONS rather than literal symbols, so one code
always denotes the same predicate under every surface remap.  Across feedback
rounds a learner accumulates (candidate predicate, pass/fail, certified
verdict) triples and eliminates the candidate languages that cannot explain the
verdicts — the rule identification this family exists to test.  Recorded
honestly in Amendment 14: the probe's pass/fail is computable from the
displayed string, so the hint's value is that it points at a hypothesis from the
family's own pool and pre-computes it, not that it discloses hidden state.  The
alternative — also reporting whether the candidate is one of the rule's
conjuncts — was rejected because "is a conjunct AND fails" implies OUT under
``AND`` with certainty, i.e. it would reintroduce the defect being repaired.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, MAX_HINT_OPTIONS,
                    Rule, TaskFamily, TaskInstance, derive_seed, letters,
                    make_rng)
from ..surface_remap import PromptSpec, lab, syms

SYMBOL_POOL = ("a", "b", "c", "d", "e", "f")
N_SYMBOLS = 3
SCHEMAS = ("count_mod", "ends_with", "starts_with", "contains_bigram",
           "length_mod", "no_occurrence", "first_before", "count_at_least")
_LENGTHS = {0: 5, 1: 8, 2: 11}
_TRANSFER_LENGTH = 18

#: Probe status codes.  They may not collide with any answer label of this
#: family (``IN``/``OUT``, ``ACCEPT``/``REJECT``, ``MEMBER``/``NONMEMBER``,
#: ``GREEN``/``AMBER``) under any surface.
PROBE_STATUS = ("PASS", "FAIL")


def _build_predicate_pool() -> tuple[dict, ...]:
    """The FROZEN candidate-predicate pool the hint probes (Amendment 14).

    One deterministic enumeration of all eight schemas over the alphabet
    POSITIONS 0..N_SYMBOLS-1 (positions, not literal symbols, so a probe code
    denotes the same predicate under every surface remap).  The order is fixed
    by this function, so the opaque index is a stable positional code that a
    learner can accumulate evidence against across feedback rounds.
    """
    pool: list[dict] = []
    for s in range(N_SYMBOLS):
        for m in (2, 3):
            for r in range(m):
                pool.append({"schema": "count_mod", "sym": s, "m": m, "r": r})
    for schema in ("ends_with", "starts_with", "no_occurrence"):
        for s in range(N_SYMBOLS):
            pool.append({"schema": schema, "sym": s})
    for x in range(N_SYMBOLS):
        for y in range(N_SYMBOLS):
            pool.append({"schema": "contains_bigram", "x": x, "y": y})
    for m in (2, 3):
        for r in range(m):
            pool.append({"schema": "length_mod", "m": m, "r": r})
    for x in range(N_SYMBOLS):
        for y in range(N_SYMBOLS):
            if x != y:
                pool.append({"schema": "first_before", "x": x, "y": y})
    for s in range(N_SYMBOLS):
        for t in (1, 2):
            pool.append({"schema": "count_at_least", "sym": s, "t": t})
    return tuple(pool)


#: frozen, rule-independent candidate pool (see :func:`_build_predicate_pool`)
PREDICATE_POOL: tuple[dict, ...] = _build_predicate_pool()
assert len(PREDICATE_POOL) <= MAX_HINT_OPTIONS, (
    "the probe index must stay inside the frozen opaque-code cap")


def _pool_atom(entry: dict) -> dict:
    """Bind a positional pool entry to the concrete alphabet symbols."""
    alphabet = SYMBOL_POOL[:N_SYMBOLS]
    atom = dict(entry)
    for key in ("sym", "x", "y"):
        if key in atom:
            atom[key] = alphabet[int(atom[key])]
    return atom


def _sample_atom(rng: np.random.Generator) -> dict:
    schema = SCHEMAS[int(rng.integers(0, len(SCHEMAS)))]
    alphabet = SYMBOL_POOL[:N_SYMBOLS]
    if schema == "count_mod":
        m = int(rng.integers(2, 4))
        return {"schema": schema, "sym": alphabet[int(rng.integers(0, N_SYMBOLS))],
                "m": m, "r": int(rng.integers(0, m))}
    if schema in ("ends_with", "starts_with", "no_occurrence"):
        return {"schema": schema,
                "sym": alphabet[int(rng.integers(0, N_SYMBOLS))]}
    if schema == "contains_bigram":
        return {"schema": schema,
                "x": alphabet[int(rng.integers(0, N_SYMBOLS))],
                "y": alphabet[int(rng.integers(0, N_SYMBOLS))]}
    if schema == "length_mod":
        m = int(rng.integers(2, 4))
        return {"schema": schema, "m": m, "r": int(rng.integers(0, m))}
    if schema == "first_before":
        x, y = (int(i) for i in rng.choice(N_SYMBOLS, size=2, replace=False))
        return {"schema": schema, "x": alphabet[x], "y": alphabet[y]}
    return {"schema": schema,
            "sym": alphabet[int(rng.integers(0, N_SYMBOLS))],
            "t": int(rng.integers(1, 3))}


def _holds(atom: dict, text: str) -> bool:
    schema = atom["schema"]
    if schema == "count_mod":
        return text.count(atom["sym"]) % atom["m"] == atom["r"]
    if schema == "ends_with":
        return text.endswith(atom["sym"])
    if schema == "starts_with":
        return text.startswith(atom["sym"])
    if schema == "contains_bigram":
        return (atom["x"] + atom["y"]) in text
    if schema == "length_mod":
        return len(text) % atom["m"] == atom["r"]
    if schema == "no_occurrence":
        return atom["sym"] not in text
    if schema == "first_before":
        xi = text.find(atom["x"])
        yi = text.find(atom["y"])
        return xi != -1 and yi != -1 and xi < yi
    return text.count(atom["sym"]) >= atom["t"]


class GrammarClassificationFamily(TaskFamily):
    family_id = "grammar_classification"
    #: 1.0.0 -> 1.1.0: pool-predicate probe replaces the failing-conjunct
    #: report, two of whose branches decoded the label (Amendment 14).
    generator_version = "1.1.0"
    canon_mode = "label"
    symbol_pool = SYMBOL_POOL
    label_variants = (("ACCEPT", "REJECT"), ("MEMBER", "NONMEMBER"),
                      ("GREEN", "AMBER"))

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        atoms = [_sample_atom(rng), _sample_atom(rng)]
        negate = [int(rng.integers(0, 2)), int(rng.integers(0, 2))]
        connective = "AND" if int(rng.integers(0, 2)) == 0 else "OR"
        return Rule(self.family_id, {"atoms": atoms, "negate": negate,
                                     "connective": connective})

    def _atom_values(self, rule: Rule, text: str) -> list[bool]:
        return [bool(_holds(atom, text)) != bool(neg)
                for atom, neg in zip(rule.params["atoms"], rule.params["negate"])]

    def _member(self, rule: Rule, text: str) -> bool:
        left, right = self._atom_values(rule, text)
        return (left and right) if rule.params["connective"] == "AND" \
            else (left or right)

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        length = _TRANSFER_LENGTH if kind == KIND_TRANSFER else _LENGTHS[difficulty]
        alphabet = SYMBOL_POOL[:N_SYMBOLS]
        text = "".join(alphabet[int(i)] for i in
                       rng.integers(0, N_SYMBOLS, size=length))
        member = self._member(rule, text)
        clauses = [
            "A hidden language accepts some strings over the alphabet below.",
            f"Alphabet: {syms(alphabet)}",
            f"String: {syms(text, sep='')}",
        ]
        spec = PromptSpec(
            clauses=tuple(clauses),
            question="Is this string in the hidden language?",
            answer_format=(f"Reply with a final line: ANSWER: {lab('IN')} "
                           f"or {lab('OUT')}"),
            alphabet=tuple(alphabet),
            labels=("IN", "OUT"),
            answer=lab("IN" if member else "OUT"),
        )
        return Item(spec=spec, data={"text": text, "member": member})

    # -- feedback -----------------------------------------------------------

    def _probe_index(self, item: Item) -> int:
        """Which pool predicate this item's hint reports on.

        Seeded from the DISPLAYED string only: the hidden rule and the
        membership bit are absent from the seed, so the selection channel
        cannot carry the answer.  Pure function of the instance, so both
        attempts on one item see the same probe.
        """
        rng = make_rng(derive_seed(0, self.family_id, "PROBE",
                                   item.data["text"]))
        return int(rng.integers(0, len(PREDICATE_POOL)))

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        index = self._probe_index(item)
        holds = _holds(_pool_atom(PREDICATE_POOL[index]), item.data["text"])
        names = tuple(letters(i) for i in range(len(PREDICATE_POOL)))
        return Feedback("HINT-PROBE", (
            HintField("probe", "PRD", index, len(PREDICATE_POOL), names),
            HintField("status", "STA", 0 if holds else 1, len(PROBE_STATUS),
                      PROBE_STATUS),
        ))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        return self._flip_label(rule, instance)


FAMILY = GrammarClassificationFamily()
