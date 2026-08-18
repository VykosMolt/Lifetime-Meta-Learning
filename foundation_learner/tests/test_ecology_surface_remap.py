"""Surface remapping: determinism, bijectivity, semantic preservation (§4)."""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import derive_seed, make_rng
from foundation_learner.ecology.families import all_families
from foundation_learner.ecology.surface_remap import (DISTRACTOR_POOL,
                                                      N_FORMAT_VARIANTS,
                                                      build_plan,
                                                      remap_ids_for)
from foundation_learner.episodes.parse import canonicalize, format_answer

FAMILIES = all_families()
IDS = [f.family_id for f in FAMILIES]
REMAPS = ("remap-a-0", "remap-a-1", "remap-b-0")


def _fixture(family, seed=20260809):
    rng = make_rng(derive_seed(seed, family.family_id, "remap"))
    rule = family.sample_rule(rng)
    instances = [family.sample_instance(rule, rng, 1) for _ in range(3)]
    instances.append(family.transfer_instance(rule, rng, 1))
    return rule, instances


def _plan(family, remap_id, rule, instance):
    item, plan = family._item_and_plan(rule, instance)
    return item, plan


def test_remap_ids_are_derived_from_the_episode():
    ids = remap_ids_for("a" * 64, 2)
    assert ids == [f"remap-{'a' * 16}-0", f"remap-{'a' * 16}-1"]
    assert len(set(ids)) == 2


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_canonical_surface_is_the_identity(family):
    rule, instances = _fixture(family)
    for instance in instances:
        item, plan = _plan(family, "canonical", rule, instance)
        assert plan.identity
        assert plan.symbol_map == {} and plan.label_map == {}
        assert plan.distractors == ()
        assert plan.clause_order(5) == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError):
        family.surface_remap(rule, instances[0], "canonical")


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_plan_is_deterministic_and_shared_across_an_episode(family):
    rule, instances = _fixture(family)
    for remap_id in REMAPS:
        plans = [build_plan(family.family_id, remap_id,
                            family._item_and_plan(rule, i)[0].spec,
                            family.blocks(), family.label_variants)
                 for i in instances]
        first = plans[0]
        for plan in plans[1:]:
            assert plan.symbol_map == first.symbol_map
            assert plan.label_map == first.label_map
            assert plan.format_variant == first.format_variant
            assert plan.distractors == first.distractors
        assert 0 <= first.format_variant < N_FORMAT_VARIANTS
        assert all(d in DISTRACTOR_POOL for d in first.distractors)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_symbol_map_is_bijective_within_each_block(family):
    rule, instances = _fixture(family)
    for remap_id in REMAPS:
        _, plan = family._item_and_plan(
            rule, family.surface_remap(rule, instances[0], remap_id))
        assert set(plan.symbol_map) == set(family.symbol_pool)
        assert set(plan.symbol_map.values()) == set(family.symbol_pool)
        for block in family.blocks():
            images = {plan.symbol_map[s] for s in block}
            assert images == set(block), "renaming crossed a symbol namespace"


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_remap_preserves_the_latent_rule_and_verifies_in_its_surface(family):
    rule, instances = _fixture(family)
    for instance in instances:
        for remap_id in REMAPS:
            remapped = family.surface_remap(rule, instance, remap_id)
            assert remapped.rule_id == instance.rule_id
            assert remapped.seed == instance.seed
            assert remapped.difficulty == instance.difficulty
            assert remapped.surface_map_id == remap_id
            assert remapped.instance_id != instance.instance_id
            assert family.verify(
                rule, remapped,
                format_answer(remapped.answer_canonical)).correct
            assert remapped == family.surface_remap(rule, instance, remap_id)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_inverse_map_recovers_the_canonical_answer(family):
    """The latent rule is invariant under the inverse of the surface map."""
    rule, instances = _fixture(family)
    for instance in instances:
        for remap_id in REMAPS:
            remapped = family.surface_remap(rule, instance, remap_id)
            _, plan = family._item_and_plan(rule, remapped)
            inverse_symbol = {v: k for k, v in plan.symbol_map.items()}
            inverse_label = {v: k for k, v in plan.label_map.items()}
            payload = remapped.answer_canonical
            if inverse_label:
                payload = inverse_label.get(payload, payload)
            elif family.canon_mode in ("token_list", "set"):
                payload = " ".join(inverse_symbol.get(t, t)
                                   for t in payload.split())
            elif family.canon_mode in ("string", "formula"):
                payload = "".join(inverse_symbol.get(c, c) for c in payload)
            elif family.canon_mode == "assignment":
                pairs = [p.split("=", 1) for p in payload.split()]
                payload = " ".join(
                    f"{inverse_symbol.get(name, name)}={value}"
                    for name, value in pairs)
            recovered = canonicalize(payload, family.canon_mode)
            assert recovered == instance.answer_canonical


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_remap_actually_changes_the_surface(family):
    rule, instances = _fixture(family)
    surfaces = {instances[0].prompt_text}
    for remap_id in REMAPS:
        remapped = family.surface_remap(rule, instances[0], remap_id)
        surfaces.add(remapped.prompt_text)
    assert len(surfaces) >= 3, "remapping must produce distinct surfaces"


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_label_families_carry_a_legend(family):
    if not family.label_variants:
        pytest.skip("no answer labels to remap")
    rule, instances = _fixture(family)
    found = False
    for remap_id in REMAPS + ("remap-c-0", "remap-d-0"):
        remapped = family.surface_remap(rule, instances[0], remap_id)
        _, plan = family._item_and_plan(rule, remapped)
        if plan.label_map:
            found = True
            assert "Answer legend:" in remapped.prompt_text
            for old, new in plan.label_map.items():
                assert new in remapped.prompt_text
    assert found, "label remapping never fired for this family"
