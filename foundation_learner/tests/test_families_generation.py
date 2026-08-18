"""Generator determinism, identity and difficulty/transfer structure (§4)."""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import (KIND_NAMES, KIND_SUPPORT,
                                             canonical_json, encode_seed,
                                             decode_seed, derive_seed,
                                             domain_sha256, make_rng)
from foundation_learner.ecology.families import (FAMILY_REGISTRY, all_families,
                                                 family_module_paths,
                                                 get_family)
from foundation_learner.ecology.split import FAMILY_IDS

FAMILIES = all_families()
IDS = [f.family_id for f in FAMILIES]


def _rule_and_instances(family, seed=20260809, difficulty=1):
    rng = make_rng(derive_seed(seed, family.family_id, "fixture"))
    rule = family.sample_rule(rng)
    return rule, {
        "support": family.sample_instance(rule, rng, difficulty),
        "related": family.related_instance(rule, rng, difficulty),
        "query": family.query_instance(rule, rng, difficulty),
        "transfer": family.transfer_instance(rule, rng, difficulty),
    }


#: Frozen per-family generator versions.  Amendment 13 bumped
#: ``graph_edge_semantics`` to 1.1.0 (answer-independent hint-node selection);
#: Amendment 14 bumped ``constraint_rules`` (candidate-constraint probe) and
#: ``grammar_classification`` (pool-predicate probe) for the same reason — a
#: hint that decoded the label — and Amendment 15 bumped ``constraint_rules``
#: again to 1.2.0 for its label-balanced item sampler.  Every other family is
#: untouched at 1.0.0.
#: Pinned so that a generator can never change without the version, the
#: manifests and the data moving with it.
EXPECTED_GENERATOR_VERSIONS = {
    "boolean_rule": "1.0.0",
    "propositional_transform": "1.0.0",
    "modular_arithmetic": "1.0.0",
    "sequence_transform": "1.0.0",
    "string_rewrite": "1.0.0",
    "finite_state_transducer": "1.0.0",
    "permutation_composition": "1.0.0",
    "set_operations": "1.0.0",
    "graph_edge_semantics": "1.1.0",
    "dsl_execution": "1.0.0",
    "constraint_rules": "1.2.0",
    "grammar_classification": "1.1.0",
}


def test_generator_versions_are_pinned():
    assert {f.family_id: f.generator_version
            for f in FAMILIES} == EXPECTED_GENERATOR_VERSIONS


#: Amendment 15: the label-balanced family, its target and the honest tolerance
#: for a finite sample.  ``boolean_rule``'s 0.633 imbalance is RECORDED AND
#: ACCEPTED unchanged (a hidden Boolean function is not expected to be balanced
#: over assignments), so it carries a wider bound that pins the status quo
#: rather than a target.
LABEL_BALANCE_BOUNDS = {
    "constraint_rules": (0.35, 0.65),
    "graph_edge_semantics": (0.20, 0.80),
    "grammar_classification": (0.35, 0.65),
    "boolean_rule": (0.20, 0.80),
}


@pytest.mark.parametrize("family_id", sorted(LABEL_BALANCE_BOUNDS))
def test_label_answer_families_are_not_degenerate(family_id):
    """A family whose items are almost all one label makes a constant-answer
    policy score that rate, so its learning curve says nothing.  Measured over
    >= 2000 items spanning every instance kind."""
    from foundation_learner.ecology.base import (KIND_QUERY, KIND_RELATED,
                                                 KIND_SUPPORT, KIND_TRANSFER,
                                                 encode_seed)

    family = get_family(family_id)
    counts: dict[str, int] = {}
    kinds = (KIND_SUPPORT, KIND_RELATED, KIND_QUERY, KIND_TRANSFER)
    for r in range(180):
        rule = family.sample_rule(
            make_rng(derive_seed(20260809, family_id, "balance", r)))
        for i in range(12):
            raw = int(make_rng(derive_seed(7, family_id, r, i)
                               ).integers(0, 1 << 40))
            instance = family.instance(rule, encode_seed(raw, kinds[i % 4]), 1)
            counts[instance.answer_canonical] = \
                counts.get(instance.answer_canonical, 0) + 1
    total = sum(counts.values())
    assert total >= 2000
    assert len(counts) == 2, f"{family_id} is not a two-label family: {counts}"
    lo, hi = LABEL_BALANCE_BOUNDS[family_id]
    rate = min(counts.values()) / total
    assert lo <= rate <= hi, (
        f"{family_id}: minority-label share {rate:.4f} outside [{lo}, {hi}]; "
        f"a constant-answer policy would score {1 - rate:.4f}")


def test_constraint_rules_balance_is_a_pure_deterministic_function():
    """The balanced sampler may not smuggle in state: the same identity must
    rebuild the same instance, and the rule condition must be reproducible."""
    from foundation_learner.ecology.base import KIND_SUPPORT, encode_seed
    from foundation_learner.ecology.families import constraint_rules as CR

    family = get_family("constraint_rules")
    rule_a = family.sample_rule(make_rng(derive_seed(1, "balance")))
    rule_b = family.sample_rule(make_rng(derive_seed(1, "balance")))
    assert canonical_json(rule_a.params) == canonical_json(rule_b.params)
    assert family.satisfiability_rate(rule_a) == \
        family.satisfiability_rate(rule_b) >= CR.RULE_MIN_SAT_RATE
    for raw in (11, 2222, 333333):
        first = family.instance(rule_a, encode_seed(raw, KIND_SUPPORT), 1)
        second = family.instance(rule_a, encode_seed(raw, KIND_SUPPORT), 1)
        assert canonical_json(first.to_dict()) == canonical_json(second.to_dict())


def test_every_sampled_constraint_rule_can_produce_both_labels():
    from foundation_learner.ecology.families import constraint_rules as CR

    family = get_family("constraint_rules")
    for r in range(60):
        rule = family.sample_rule(
            make_rng(derive_seed(5, "constraint_rules", "reachable", r)))
        assert family.satisfiability_rate(rule) >= CR.RULE_MIN_SAT_RATE


def test_registry_matches_the_frozen_family_ids():
    assert tuple(FAMILY_REGISTRY) == FAMILY_IDS
    assert len(FAMILIES) == 12
    assert set(family_module_paths()) == set(FAMILY_IDS)
    with pytest.raises(KeyError):
        get_family("no_such_family")


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_at_least_three_difficulty_levels(family):
    levels = family.difficulty_levels()
    assert len(levels) >= 3
    assert levels == sorted(set(levels))
    with pytest.raises(ValueError):
        family.instance(family.sample_rule(make_rng(1)), 0, max(levels) + 1)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_generation_is_bitwise_deterministic(family):
    first = _rule_and_instances(family)
    second = _rule_and_instances(family)
    assert canonical_json(first[0].params) == canonical_json(second[0].params)
    for kind in first[1]:
        assert canonical_json(first[1][kind].to_dict()) == \
            canonical_json(second[1][kind].to_dict())


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_instance_is_a_pure_function_of_its_identity(family):
    rule, instances = _rule_and_instances(family)
    for instance in instances.values():
        rebuilt = family.instance(rule, instance.seed, instance.difficulty,
                                  instance.surface_map_id)
        assert rebuilt == instance


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_identity_fields_follow_the_frozen_scheme(family):
    rule, instances = _rule_and_instances(family)
    assert family.rule_id(rule) == domain_sha256("FL_V0_RULE",
                                                 family.rule_spec(rule))
    for kind, instance in instances.items():
        assert instance.family_id == family.family_id
        assert instance.generator_version == family.generator_version
        assert instance.kind == kind
        assert instance.surface_map_id == "canonical"
        assert instance.instance_id == domain_sha256("FL_V0_INSTANCE", {
            "family_id": instance.family_id, "rule_id": instance.rule_id,
            "seed": instance.seed, "difficulty": instance.difficulty,
            "surface_map_id": instance.surface_map_id,
            "prompt_text": instance.prompt_text,
            "answer_canonical": instance.answer_canonical})
        assert KIND_NAMES[decode_seed(instance.seed)[1]] == kind
    ids = {i.instance_id for i in instances.values()}
    assert len(ids) == len(instances)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_rules_are_fresh_and_rule_ids_carry_the_family(family):
    rng = make_rng(derive_seed(4242, family.family_id))
    rules = [family.sample_rule(rng) for _ in range(40)]
    ids = {family.rule_id(r) for r in rules}
    assert len(ids) >= 30, "rule space is too small for a 2000-episode pool"
    for rule in rules:
        assert rule.family_id == family.family_id
        assert family.rule_spec(rule)["family_id"] == family.family_id


def test_rule_ids_never_collide_across_families():
    seen: dict[str, str] = {}
    for family in FAMILIES:
        rng = make_rng(derive_seed(99, family.family_id))
        for _ in range(50):
            rule_id = family.rule_id(family.sample_rule(rng))
            assert seen.setdefault(rule_id, family.family_id) == family.family_id


def test_instance_ids_never_collide_across_families():
    seen: dict[str, str] = {}
    for family in FAMILIES:
        rng = make_rng(derive_seed(31337, family.family_id))
        rule = family.sample_rule(rng)
        for _ in range(30):
            instance = family.sample_instance(rule, rng, 1)
            owner = seen.setdefault(instance.instance_id, family.family_id)
            assert owner == family.family_id


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_transfer_is_a_shifted_distribution(family):
    """Transfer prompts must be visibly outside the training distribution."""
    rng = make_rng(derive_seed(77, family.family_id))
    rule = family.sample_rule(rng)
    support = [family.sample_instance(rule, rng, 1) for _ in range(12)]
    transfer = [family.transfer_instance(rule, rng, 1) for _ in range(12)]
    support_len = max(len(i.prompt_text) for i in support)
    transfer_len = min(len(i.prompt_text) for i in transfer)
    support_texts = {i.prompt_text for i in support}
    assert not support_texts & {i.prompt_text for i in transfer}
    assert transfer_len > 0.9 * support_len
    assert max(len(i.prompt_text) for i in transfer) > support_len


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_difficulty_changes_the_instance_distribution(family):
    """Every level must actually enter generation, not just be accepted.

    Length is not the axis for every family (repeats, value ranges, distractor
    variables), so the check is on the generated instances themselves: with the
    SAME instance seed, each difficulty level must produce a different task.
    """
    rng = make_rng(derive_seed(5, family.family_id))
    rule = family.sample_rule(rng)
    seeds = [encode_seed(int(rng.integers(0, 1 << 40)), KIND_SUPPORT)
             for _ in range(8)]
    by_level = {}
    for level in family.difficulty_levels():
        by_level[level] = {family.instance(rule, s, level).prompt_text
                           for s in seeds}
    levels = family.difficulty_levels()
    for i, left in enumerate(levels):
        for right in levels[i + 1:]:
            assert by_level[left] != by_level[right], \
                f"difficulty {left} and {right} generate identical tasks"


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_symbol_alphabet_stays_inside_the_declared_pool(family):
    rule, instances = _rule_and_instances(family)
    pool = set(family.symbol_pool)
    for instance in instances.values():
        raw, kind = decode_seed(instance.seed)
        spec = family._item(rule, raw, kind, instance.difficulty).spec
        assert set(spec.alphabet) <= pool
        blocks = family.blocks()
        flat = [s for block in blocks for s in block]
        assert len(flat) == len(set(flat)), "symbol blocks must be disjoint"
        assert set(flat) == pool
