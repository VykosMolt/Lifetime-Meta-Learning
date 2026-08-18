"""HOSTILE: the structured hint must not DECODE the answer.

Attack (Amendment 13, finding R-M3).  ``ecology/base.answer_leak`` checks that a
hint does not CONTAIN the answer string.  It says nothing about a hint that
lets a reader INFER the answer with certainty from a feature the prompt already
shows.  Generator 1.0.0 of ``graph_edge_semantics`` drew the hinted node from
the queried PATH when one existed and from the SOURCE otherwise, so

    "the hinted node is not the source"  ->  YES   (measured P = 1.000, n = 382)
    "the hinted node is the target"      ->  YES   (measured P = 1.000, n = 291)

i.e. the SELECTION channel — which node the hint names, not what it says about
it — carried the reachability bit outright, and a learner could read the answer
without ever inferring the hidden edge semantics.

What this fixture measures
==========================
For every family, over >= 2000 sampled items, it computes
``P(correct answer | branch)`` for every discrete branch of every hint-derived
feature, and classifies each feature into one of three channels:

``content``     the value a hint field renders (its opaque code).  This is the
                informative payload contract section 4 defines.
``selection``   WHICH displayed entity the hint names, expressed relative to
                the entities the QUESTION names.  A reader can compute these
                from the prompt alone, so a deterministic selection branch is a
                free decode of the answer.  This channel must NEVER be
                deterministic, for any family.
``diagnostic``  features that require the HIDDEN state to evaluate (e.g. "the
                hinted node lies on the true path").  A reader cannot compute
                them without having already solved the task, so they are not a
                decode channel; they are measured and RECORDED, never asserted.

Assertions (NO allow-list, Amendment 14)
========================================
1. NO branch of ANY model-computable feature (``content`` or ``selection``) of
   ANY of the twelve families is deterministic.  There is no exemption list:
   ``graph_edge_semantics`` (1.1.0), ``constraint_rules`` (1.1.0) and
   ``grammar_classification`` (1.1.0) were REPAIRED rather than excused.
2. Non-vacuity, proved by construction for all three repairs: the identical
   measurement is re-run against a frozen inline reimplementation of each
   family's OLD hint rule, and the violation is REQUIRED to appear.
3. Per-family measured decodability stays under the pinned ceilings in
   :data:`MAX_PURITY_CEILING` (regression detector for the fixed generators).

Probabilistic evidence is the legitimate content of feedback and is recorded,
not forbidden: ``boolean_rule``'s ``pivot=ABSENT`` branch predicts the answer at
P ~ 0.89 against a 0.63 base rate, which is exactly what an informative hint is
supposed to do.  Only DETERMINISTIC decode branches are defects.

Why 6480 items per family
=========================
A branch that happens to contain no counterexample is not evidence of a decode
channel when the family's base rate is lopsided: at ``constraint_rules``' 0.893
base rate, a 32-item branch is all-UNSAT by chance about 2 % of the time, and
with dozens of branches per family such accidents are expected.  The sampling
plan is therefore sized so that EVERY branch that clears ``MIN_BRANCH_N``
carries enough items for accidental purity to be negligible (the smallest
branch observed across the twelve families holds 99 items, at which
``0.893 ** 99 ~ 1e-5``), and the report records the smallest branch size so the
detection power is auditable rather than assumed.  The criterion itself is
never relaxed: a perfect branch fails the fixture, and its chance probability
under the family base rate is recorded next to it for the reviewer.

The measured table is written to ``HINT_LEAK_REPORT_PATH`` for review.
"""
from __future__ import annotations

import collections
import json
import os
import tempfile

import pytest

from foundation_learner.ecology.base import (KIND_QUERY, KIND_RELATED,
                                             KIND_SUPPORT, KIND_TRANSFER,
                                             Feedback, HintField,
                                             derive_seed, encode_seed, letters,
                                             make_rng)
from foundation_learner.ecology.families import all_families, get_family
from foundation_learner.episodes.parse import format_answer
from foundation_learner.tests import _w4_support as S

pytestmark = pytest.mark.skipif(
    not S.integration_available(), reason="ecology/episodes packages absent")

#: sampling plan: 360 rules x 18 items = 6480 items per family (see the module
#: docstring for why this is larger than the 2000-item minimum)
N_RULES = 360
ITEMS_PER_RULE = 18
#: a branch below this count carries no evidence either way
MIN_BRANCH_N = 30
SAMPLE_SEED = 20260809

CHANNEL_CONTENT = "content"
CHANNEL_SELECTION = "selection"
CHANNEL_DIAGNOSTIC = "diagnostic"

HINT_LEAK_REPORT_PATH = os.path.join(
    tempfile.gettempdir(), "fl_v0_hint_selection_leak", "hint_leak_report.json")

#: There is NO allow-list.  Amendment 14 removed the last two entries by
#: REPAIRING the families rather than exempting them; this constant is kept
#: (empty, and asserted empty) so that re-introducing an exemption is a visible
#: edit to a named object rather than a quiet change of a comparison.
DETERMINISTIC_CONTENT_BY_DESIGN: frozenset[tuple[str, str]] = frozenset()

#: Which hint fields IDENTIFY the candidate/entity a hint reports on, per
#: family.  These are the ``selection`` channel: the choice must be made
#: answer-independently, so a deterministic branch here is a free decode.
SELECTION_FIELDS: dict[str, frozenset[str]] = {
    "graph_edge_semantics": frozenset({"at"}),
    "constraint_rules": frozenset({"probe"}),
    "grammar_classification": frozenset({"probe"}),
}

#: Measured ceilings on the strongest MODEL-COMPUTABLE branch (content +
#: selection), with margin.  Measured on the current generators with the
#: sampling plan above; a regression that makes a hint more decodable trips
#: these.  No family is exempt and none sits at 1.0.
MAX_PURITY_CEILING: dict[str, float] = {
    "boolean_rule": 0.95,               # measured 0.8861 (pivot=ABSENT)
    "propositional_transform": 0.20,    # measured 0.0476
    "modular_arithmetic": 0.20,         # measured 0.0843
    "sequence_transform": 0.20,         # measured 0.0198
    "string_rewrite": 0.35,             # measured 0.2024
    "finite_state_transducer": 0.35,    # measured 0.1553
    "permutation_composition": 0.20,    # measured 0.0027
    "set_operations": 0.70,             # measured 0.5828
    "graph_edge_semantics": 0.85,       # measured 0.7405 (1.0.0: 1.0000)
    "dsl_execution": 0.20,              # measured 0.0225
    "constraint_rules": 0.65,           # measured 0.5734 (1.0.0: 1.0000)
    "grammar_classification": 0.70,     # measured 0.6045 (1.0.0: 1.0000)
}


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------
def _features(family_id: str, item, record: Feedback
              ) -> list[tuple[str, str, str]]:
    """``(channel, feature, branch)`` triples derived from ONE hint."""
    selection = SELECTION_FIELDS.get(family_id, frozenset())
    out: list[tuple[str, str, str]] = [
        (CHANNEL_SELECTION if f.name in selection else CHANNEL_CONTENT,
         f.name, f.code())
        for f in record.fields]
    if family_id == "graph_edge_semantics":
        # The hint names a DISPLAYED node; the question names the source and the
        # target.  Both relations are computable from the prompt alone.
        at = next(f for f in record.fields if f.name == "at")
        out.append((CHANNEL_SELECTION, "at_is_source",
                    str(at.value == item.data["source"])))
        out.append((CHANNEL_SELECTION, "at_is_target",
                    str(at.value == item.data["target"])))
        # NOT model-computable: membership of the true path needs the hidden
        # edge semantics, i.e. the answer itself.
        path = item.data["path"]
        out.append((CHANNEL_DIAGNOSTIC, "at_in_path",
                    str(path is not None and at.value in path)))
    return out


def _sample_family(family, feedback_fn=None) -> dict:
    """Measure P(answer | branch) over the sampling plan for one family."""
    counts: dict[tuple[str, str, str], collections.Counter] = (
        collections.defaultdict(collections.Counter))
    answers: collections.Counter = collections.Counter()
    n_items = 0
    for r in range(N_RULES):
        rule = family.sample_rule(
            make_rng(derive_seed(SAMPLE_SEED, family.family_id, "rule", r)))
        for i in range(ITEMS_PER_RULE):
            kind = (KIND_SUPPORT, KIND_RELATED, KIND_QUERY)[i % 3]
            raw = int(make_rng(derive_seed(7, family.family_id, r, i)
                               ).integers(0, 1 << 40))
            instance = family.instance(rule, encode_seed(raw, kind), 1)
            # PROBE: the hint that follows a plausible INCORRECT attempt, i.e.
            # the case where feedback is supposed to do work.
            attempt = format_answer(
                family.wrong_answer(rule, instance, make_rng(1)))
            record = (feedback_fn(rule, instance, attempt) if feedback_fn
                      else family.feedback_record(rule, instance, attempt,
                                                  make_rng(2)))
            item, _plan = family._item_and_plan(rule, instance)
            n_items += 1
            answers[instance.answer_canonical] += 1
            for triple in _features(family.family_id, item, record):
                counts[triple][instance.answer_canonical] += 1

    branches = []
    for (channel, feature, branch), dist in sorted(counts.items()):
        n = int(sum(dist.values()))
        if n < MIN_BRANCH_N:
            continue
        top_answer, top_n = dist.most_common(1)[0]
        purity = top_n / n
        base = answers[top_answer] / n_items
        branches.append({
            "channel": channel, "feature": feature, "branch": branch,
            "n": n, "answer": top_answer, "purity": purity, "base_rate": base,
            "lift": purity - base, "deterministic": purity >= 1.0,
            # Diagnostic ONLY (never an excuse): how likely a branch of this
            # size would be pure under the family's base rate.  A perfect
            # branch fails the fixture whatever this says.
            "chance_of_purity_under_base_rate": (base ** n if purity >= 1.0
                                                 else None),
        })
    top_answer, top_n = answers.most_common(1)[0]
    return {
        "family_id": family.family_id,
        "generator_version": family.generator_version,
        "n_items": n_items,
        "n_distinct_answers": len(answers),
        "base_top_answer": top_answer,
        "base_rate_top": top_n / n_items,
        "min_branch_n": MIN_BRANCH_N,
        "branches": branches,
    }


def _model_computable(report: dict) -> list[dict]:
    return [b for b in report["branches"]
            if b["channel"] in (CHANNEL_CONTENT, CHANNEL_SELECTION)]


@pytest.fixture(scope="module")
def measurements() -> dict:
    reports = {f.family_id: _sample_family(f) for f in all_families()}
    os.makedirs(os.path.dirname(HINT_LEAK_REPORT_PATH), exist_ok=True)
    payload = {
        "schema": "flb200.hint_selection_leak.v1",
        "sampling": {"n_rules": N_RULES, "items_per_rule": ITEMS_PER_RULE,
                     "seed": SAMPLE_SEED, "min_branch_n": MIN_BRANCH_N,
                     "probe": "hint after a plausible INCORRECT attempt"},
        "deterministic_content_by_design": sorted(
            list(pair) for pair in DETERMINISTIC_CONTENT_BY_DESIGN),
        "allow_list_is_empty": not DETERMINISTIC_CONTENT_BY_DESIGN,
        "max_purity_ceiling": MAX_PURITY_CEILING,
        "smallest_measured_branch": min(
            (b["n"] for r in reports.values() for b in r["branches"]),
            default=None),
        "families": {
            fid: {
                "generator_version": r["generator_version"],
                "n_items": r["n_items"],
                "n_distinct_answers": r["n_distinct_answers"],
                "base_top_answer": r["base_top_answer"],
                "base_rate_top": r["base_rate_top"],
                "max_purity_model_computable": max(
                    (b["purity"] for b in _model_computable(r)), default=0.0),
                "max_lift_model_computable": max(
                    (b["lift"] for b in _model_computable(r)), default=0.0),
                "max_purity_selection": max(
                    (b["purity"] for b in r["branches"]
                     if b["channel"] == CHANNEL_SELECTION), default=None),
                "smallest_branch": min((b["n"] for b in r["branches"]),
                                       default=None),
                "deterministic_branches": [
                    b for b in r["branches"] if b["deterministic"]],
                "notable_branches": [
                    b for b in r["branches"] if b["lift"] > 0.15],
            }
            for fid, r in sorted(reports.items())},
    }
    with open(HINT_LEAK_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return reports


# --------------------------------------------------------------------------
# assertions
# --------------------------------------------------------------------------
def test_every_family_was_sampled_at_the_declared_scale(measurements):
    assert set(measurements) == {f.family_id for f in all_families()}
    for report in measurements.values():
        assert report["n_items"] >= 2000
        assert report["branches"], "no branch reached the minimum count"
    assert os.path.isfile(HINT_LEAK_REPORT_PATH)


def test_the_sample_has_the_power_to_call_a_pure_branch_structural(measurements):
    """Detection power is asserted, not assumed: every measured branch must be
    large enough that accidental purity under the family's own base rate is
    negligible, so a perfect branch really does mean a decode channel."""
    for fid, report in sorted(measurements.items()):
        # A branch is accidentally pure only if all of its items happen to
        # carry ONE answer, so the worst case is the MOST COMMON answer's rate
        # (``base_rate_top`` is that maximum by construction).
        base = report["base_rate_top"]
        smallest = min(b["n"] for b in report["branches"])
        assert smallest >= MIN_BRANCH_N
        assert base ** smallest < 1e-4, (
            f"{fid}: smallest branch has {smallest} items at base rate {base}, "
            "so a purely accidental perfect branch would not be surprising; "
            "raise N_RULES rather than relaxing the criterion")


def test_no_branch_of_any_family_decodes_the_answer(measurements):
    """Amendment 14: NO allow-list.  Neither WHICH candidate a hint reports on
    nor WHAT it says about it may imply the label with certainty, in any of the
    twelve families."""
    offenders = []
    for fid, report in sorted(measurements.items()):
        for b in report["branches"]:
            if b["channel"] == CHANNEL_DIAGNOSTIC:
                continue  # not computable without the hidden state
            if b["deterministic"]:
                offenders.append((fid, b["channel"], b["feature"], b["branch"],
                                  b["n"], b["answer"],
                                  b["chance_of_purity_under_base_rate"]))
    assert offenders == [], (
        "a hint branch decodes the answer with certainty: " f"{offenders}")


def test_the_allow_list_is_empty(measurements):
    """A future exemption must be an explicit, reviewable edit."""
    assert DETERMINISTIC_CONTENT_BY_DESIGN == frozenset()
    observed = {(fid, b["feature"])
                for fid, report in measurements.items()
                for b in report["branches"]
                if b["channel"] != CHANNEL_DIAGNOSTIC and b["deterministic"]}
    assert observed == set(), (
        f"answer-determining hint features are no longer permitted: {observed}")


def test_measured_decodability_stays_under_the_pinned_ceilings(measurements):
    assert set(MAX_PURITY_CEILING) == set(measurements)
    for fid, report in sorted(measurements.items()):
        worst = max((b["purity"] for b in _model_computable(report)),
                    default=0.0)
        assert worst <= MAX_PURITY_CEILING[fid] + 1e-9, (
            f"{fid}: strongest model-computable hint branch predicts the answer "
            f"at P = {worst:.4f}, above the pinned ceiling "
            f"{MAX_PURITY_CEILING[fid]}")


def test_informative_probabilistic_evidence_is_recorded_not_forbidden(measurements):
    """``boolean_rule``'s ABSENT pivot is strong evidence and must stay legal:
    the fixture forbids CERTAINTY, not information."""
    report = measurements["boolean_rule"]
    absent = [b for b in report["branches"]
              if b["feature"] == "pivot" and b["branch"] == "ABSENT"]
    assert len(absent) == 1
    branch = absent[0]
    assert branch["deterministic"] is False
    assert branch["purity"] > branch["base_rate"] + 0.15, (
        "the hint has stopped being informative")
    assert branch["purity"] < 1.0


def test_graph_hint_selection_is_answer_independent(measurements):
    """The repaired family: selection branches sit at the base rate."""
    report = measurements["graph_edge_semantics"]
    assert report["generator_version"] == "1.1.0"
    base = report["base_rate_top"]
    selection = [b for b in report["branches"]
                 if b["channel"] == CHANNEL_SELECTION]
    assert selection, "the selection channel must actually be measured"
    for b in selection:
        assert b["answer"] == report["base_top_answer"], (
            f"selection branch {b['feature']}={b['branch']} prefers "
            f"{b['answer']} against a {report['base_top_answer']} base rate")
        assert abs(b["purity"] - base) < 0.05, (
            f"selection branch {b['feature']}={b['branch']} deviates from the "
            f"base rate: {b['purity']:.4f} vs {base:.4f}")


def test_the_hint_still_reports_the_true_out_degree(measurements):
    """The repair may not have bought its independence with an empty hint."""
    from foundation_learner.ecology.families.graph_edge_semantics import (
        _adjacency)

    family = get_family("graph_edge_semantics")
    checked = 0
    for r in range(6):
        rule = family.sample_rule(
            make_rng(derive_seed(SAMPLE_SEED, family.family_id, "rule", r)))
        for i in range(6):
            raw = int(make_rng(derive_seed(7, family.family_id, r, i)
                               ).integers(0, 1 << 40))
            instance = family.instance(rule, encode_seed(raw, KIND_SUPPORT), 1)
            attempt = format_answer(
                family.wrong_answer(rule, instance, make_rng(1)))
            record = family.feedback_record(rule, instance, attempt, make_rng(2))
            item, _plan = family._item_and_plan(rule, instance)
            at = next(f for f in record.fields if f.name == "at")
            outdeg = next(f for f in record.fields if f.name == "outdeg")
            adjacency = _adjacency(item.data["edges"],
                                   rule.params["semantics"],
                                   item.data["n_nodes"])
            assert outdeg.value == len(adjacency[at.value])
            assert 0 <= at.value < item.data["n_nodes"]
            checked += 1
    assert checked == 36


def test_the_hinted_node_is_a_pure_function_of_the_instance():
    """Two different attempts on the same item must get the same hinted node:
    an attempt-dependent selection would reintroduce a side channel."""
    family = get_family("graph_edge_semantics")
    rule = family.sample_rule(
        make_rng(derive_seed(SAMPLE_SEED, family.family_id, "rule", 0)))
    for i in range(8):
        raw = int(make_rng(derive_seed(7, family.family_id, 0, i)
                           ).integers(0, 1 << 40))
        instance = family.instance(rule, encode_seed(raw, KIND_SUPPORT), 1)
        wrong = family.wrong_answer(rule, instance, make_rng(1))
        nodes = {
            next(f for f in family.feedback_record(
                rule, instance, text, make_rng(2)).fields
                 if f.name == "at").value
            for text in (format_answer(wrong),
                         format_answer(instance.answer_canonical),
                         "no answer line at all")}
        assert len(nodes) == 1


def _probe_field(record: Feedback, name: str):
    return next(f for f in record.fields if f.name == name)


def _record_for(family, rule, raw: int, kind: int = KIND_SUPPORT,
                difficulty: int = 1):
    instance = family.instance(rule, encode_seed(raw, kind), difficulty)
    attempt = format_answer(family.wrong_answer(rule, instance, make_rng(1)))
    return instance, family.feedback_record(rule, instance, attempt, make_rng(2))


def test_constraint_probe_is_stable_and_reports_the_true_status():
    """The repaired hint must still be TRUE and positionally decodable."""
    from foundation_learner.ecology.families.constraint_rules import (
        CANDIDATE_POOL, candidate_holds)

    family = get_family("constraint_rules")
    checked = 0
    for r in range(8):
        rule = family.sample_rule(
            make_rng(derive_seed(SAMPLE_SEED, family.family_id, "rule", r)))
        for i in range(8):
            raw = int(make_rng(derive_seed(7, family.family_id, r, i)
                               ).integers(0, 1 << 40))
            instance, record = _record_for(family, rule, raw)
            item, _plan = family._item_and_plan(rule, instance)
            probe = _probe_field(record, "probe")
            status = _probe_field(record, "status")
            # the pool is INSTANCE-INDEPENDENT, so the code means the same
            # candidate everywhere (that is what makes it accumulable)
            assert probe.n_options == len(CANDIDATE_POOL)
            candidate = CANDIDATE_POOL[probe.value]
            values = [int(v) for v in item.data["values"]]
            expected = candidate_holds(candidate, values)
            assert (status.value == 0) is expected, (
                f"{candidate} on {values} reported {status.code()}")
            # and it is a pure function of the instance, not of the attempt
            others = {
                _probe_field(family.feedback_record(rule, instance, text,
                                                    make_rng(2)),
                             "probe").value
                for text in (format_answer(instance.answer_canonical),
                             "no answer line at all")}
            assert others == {probe.value}
            checked += 1
    assert checked == 64


def test_grammar_probe_is_stable_and_reports_the_true_status():
    from foundation_learner.ecology.families.grammar_classification import (
        PREDICATE_POOL, _holds, _pool_atom)

    family = get_family("grammar_classification")
    checked = 0
    for r in range(8):
        rule = family.sample_rule(
            make_rng(derive_seed(SAMPLE_SEED, family.family_id, "rule", r)))
        for i in range(8):
            raw = int(make_rng(derive_seed(7, family.family_id, r, i)
                               ).integers(0, 1 << 40))
            instance, record = _record_for(family, rule, raw)
            item, _plan = family._item_and_plan(rule, instance)
            probe = _probe_field(record, "probe")
            status = _probe_field(record, "status")
            assert probe.n_options == len(PREDICATE_POOL)
            expected = _holds(_pool_atom(PREDICATE_POOL[probe.value]),
                              item.data["text"])
            assert (status.value == 0) is expected
            others = {
                _probe_field(family.feedback_record(rule, instance, text,
                                                    make_rng(2)),
                             "probe").value
                for text in (format_answer(instance.answer_canonical),
                             "no answer line at all")}
            assert others == {probe.value}
            checked += 1
    assert checked == 64


def test_probe_pools_do_not_depend_on_the_hidden_rule_or_the_instance():
    """A pool whose SIZE varies with an instance covariate makes the opaque code
    disclose that covariate; for constraint_rules the displayed slot count is
    itself strongly predictive of the label, which is how a slot-count-dependent
    pool produced thirteen perfect branches during development."""
    from foundation_learner.ecology.families.constraint_rules import (
        CANDIDATE_POOL, MAX_PROBE_SLOTS)
    from foundation_learner.ecology.families.grammar_classification import (
        PREDICATE_POOL)

    assert len(CANDIDATE_POOL) == 24 and MAX_PROBE_SLOTS == 3
    assert len(PREDICATE_POOL) == 50
    for family_id, size in (("constraint_rules", len(CANDIDATE_POOL)),
                            ("grammar_classification", len(PREDICATE_POOL))):
        family = get_family(family_id)
        widths = set()
        for r in range(4):
            rule = family.sample_rule(
                make_rng(derive_seed(SAMPLE_SEED, family_id, "rule", r)))
            for kind in (KIND_SUPPORT, KIND_RELATED, KIND_QUERY,
                         KIND_TRANSFER):
                for difficulty in family.difficulty_levels():
                    raw = int(make_rng(derive_seed(3, family_id, r, kind,
                                                   difficulty)
                                       ).integers(0, 1 << 40))
                    _inst, record = _record_for(family, rule, raw, kind,
                                                difficulty)
                    widths.add(_probe_field(record, "probe").n_options)
        assert widths == {size}, (
            f"{family_id}: the probe code range varies with the instance "
            f"({sorted(widths)}), so the code discloses an instance covariate")


def _legacy_deterministic(family_id: str, feedback_fn) -> list[dict]:
    """Run the SAME measurement against an OLD hint rule and return the
    model-computable branches it decodes the answer from."""
    legacy = _sample_family(get_family(family_id), feedback_fn=feedback_fn)
    return [b for b in legacy["branches"]
            if b["channel"] != CHANNEL_DIAGNOSTIC and b["deterministic"]]


def test_the_fixture_fails_on_the_graph_1_0_0_selection_rule():
    """By construction: the old graph SELECTION rule must be caught."""
    from foundation_learner.ecology.families.graph_edge_semantics import (
        _adjacency)

    family = get_family("graph_edge_semantics")

    def legacy_feedback_record(rule, instance, attempt_text):
        """Generator 1.0.0's ``_feedback`` body, verbatim in behaviour."""
        item, _plan = family._item_and_plan(rule, instance)
        n_nodes = item.data["n_nodes"]
        adjacency = _adjacency(item.data["edges"], rule.params["semantics"],
                               n_nodes)
        path = item.data["path"]
        rng = make_rng(derive_seed(0, "FL_V0_FEEDBACK", family.family_id,
                                   instance.instance_id, attempt_text))
        node = int(rng.choice(path)) if path else item.data["source"]
        names = tuple(letters(i) for i in range(n_nodes))
        return Feedback("HINT-DEGREE", (
            HintField("at", "PT", node, n_nodes, names),
            HintField("outdeg", "DEG", len(adjacency[node]), n_nodes + 1)))

    deterministic = _legacy_deterministic("graph_edge_semantics",
                                          legacy_feedback_record)
    assert deterministic, (
        "the fixture no longer detects the graph 1.0.0 selection leak; it "
        "would pass vacuously")
    assert {"at_is_source", "at_is_target"} & {b["feature"]
                                               for b in deterministic}
    assert max(b["n"] for b in deterministic) >= MIN_BRANCH_N


def test_the_fixture_fails_on_the_constraint_rules_1_0_0_content():
    """By construction: the old violation COUNT must be caught."""
    family = get_family("constraint_rules")

    def legacy_feedback_record(rule, instance, _attempt_text):
        """Generator 1.0.0's ``_feedback`` body: the violated count."""
        item, _plan = family._item_and_plan(rule, instance)
        total = family._n_constraints(rule)
        return Feedback("HINT-VIOL", (
            HintField("violated", "CNT", item.data["violations"], total + 1),))

    deterministic = _legacy_deterministic("constraint_rules",
                                          legacy_feedback_record)
    assert deterministic, (
        "the fixture no longer detects the constraint_rules 1.0.0 count leak")
    assert {b["feature"] for b in deterministic} == {"violated"}
    # a count of zero IS satisfaction, so the zero branch decodes SAT
    assert any(b["answer"] == "SAT" for b in deterministic)
    assert max(b["n"] for b in deterministic) >= MIN_BRANCH_N


def test_the_fixture_fails_on_the_grammar_classification_1_0_0_content():
    """By construction: the old failing-conjunct report must be caught."""
    family = get_family("grammar_classification")

    def legacy_feedback_record(rule, instance, _attempt_text):
        """Generator 1.0.0's ``_feedback`` body: which conjunct fails."""
        item, _plan = family._item_and_plan(rule, instance)
        left, right = family._atom_values(rule, item.data["text"])
        if not left and not right:
            value = 2
        elif not left:
            value = 0
        elif not right:
            value = 1
        else:
            value = 3
        return Feedback("HINT-PRED", (
            HintField("fails", "PRD", value, 4, ("A", "B", "BOTH", "ABSENT")),))

    deterministic = _legacy_deterministic("grammar_classification",
                                          legacy_feedback_record)
    assert deterministic, (
        "the fixture no longer detects the grammar_classification 1.0.0 leak")
    assert {b["feature"] for b in deterministic} == {"fails"}
    assert {b["branch"] for b in deterministic} >= {"BOTH", "ABSENT"}
    assert max(b["n"] for b in deterministic) >= MIN_BRANCH_N


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
