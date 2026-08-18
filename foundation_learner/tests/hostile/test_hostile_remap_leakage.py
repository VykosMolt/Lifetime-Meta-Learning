"""HOSTILE: try to defeat or exploit the surface remapping.

Attacks, all of which must fail:

1. answer carry-over — answer a remapped item with the CANONICAL answer and
   hope the verifier accepts it (that would make the remap evaluation trivial);
2. surface leakage — look for the canonical answer, the canonical prompt or the
   latent rule spec inside a remapped prompt;
3. rule drift — hope the remap quietly changes the latent rule, so that the
   "same rule, new surface" claim is false;
4. surface collision — hope two different remap ids produce the same surface,
   or that a remap silently degenerates into the identity;
5. marker smuggling — hope an unresolved remap marker survives into a prompt.
"""
from __future__ import annotations

import pytest

from foundation_learner.ecology.base import canonical_json, derive_seed, make_rng
from foundation_learner.ecology.families import all_families
from foundation_learner.ecology.surface_remap import LAB_L, LAB_R, SYM_L, SYM_R
from foundation_learner.episodes.parse import format_answer

FAMILIES = all_families()
IDS = [f.family_id for f in FAMILIES]
REMAPS = ("remap-hostile-a", "remap-hostile-b", "remap-hostile-c",
          "remap-hostile-d")


def _fixture(family):
    rng = make_rng(derive_seed(20260809, "hostile-remap", family.family_id))
    rule = family.sample_rule(rng)
    instances = [family.sample_instance(rule, rng, 1) for _ in range(4)]
    instances.append(family.transfer_instance(rule, rng, 1))
    return rule, instances


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_canonical_answer_does_not_solve_a_remapped_item(family):
    """Whenever the surface actually renames the answer, carry-over must fail."""
    rule, instances = _fixture(family)
    attacks = 0
    for instance in instances:
        for remap_id in REMAPS:
            remapped = family.surface_remap(rule, instance, remap_id)
            if remapped.answer_canonical == instance.answer_canonical:
                continue  # this surface does not touch the answer vocabulary
            attacks += 1
            assert not family.verify(
                rule, remapped,
                format_answer(instance.answer_canonical)).correct
    if family.canon_mode not in ("int",):
        assert attacks > 0, "no surface ever renamed this family's answer"


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_remapped_prompt_leaks_neither_the_answer_nor_the_rule(family):
    rule, instances = _fixture(family)
    rule_json = canonical_json(family.rule_spec(rule))
    for instance in instances:
        for remap_id in REMAPS:
            remapped = family.surface_remap(rule, instance, remap_id)
            assert rule_json not in remapped.prompt_text
            assert instance.prompt_text != remapped.prompt_text
            # a remapped prompt must not simply carry the old surface along
            assert instance.prompt_text not in remapped.prompt_text


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_remap_never_changes_the_latent_rule(family):
    rule, instances = _fixture(family)
    for instance in instances:
        for remap_id in REMAPS:
            remapped = family.surface_remap(rule, instance, remap_id)
            assert remapped.rule_id == instance.rule_id
            assert remapped.rule_id == family.rule_id(rule)
            assert remapped.seed == instance.seed
            assert remapped.difficulty == instance.difficulty
            assert remapped.family_id == instance.family_id


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_distinct_remap_ids_give_distinct_surfaces(family):
    rule, instances = _fixture(family)
    prompts = {instances[0].prompt_text}
    for remap_id in REMAPS:
        prompts.add(family.surface_remap(rule, instances[0],
                                         remap_id).prompt_text)
    assert len(prompts) == len(REMAPS) + 1


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_no_remap_marker_survives_into_a_prompt(family):
    rule, instances = _fixture(family)
    for instance in instances:
        for remap_id in ("canonical",) + REMAPS:
            item = instance if remap_id == "canonical" else \
                family.surface_remap(rule, instance, remap_id)
            for marker in (SYM_L, SYM_R, LAB_L, LAB_R):
                assert marker not in item.prompt_text
                assert marker not in item.answer_canonical


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_a_remapped_item_still_verifies_and_still_rejects_wrong_answers(family):
    rule, instances = _fixture(family)
    rng = make_rng(derive_seed(9, family.family_id))
    for instance in instances:
        for remap_id in REMAPS:
            remapped = family.surface_remap(rule, instance, remap_id)
            assert family.verify(
                rule, remapped,
                format_answer(remapped.answer_canonical)).correct
            wrong = family.wrong_answer(rule, remapped, rng)
            assert not family.verify(rule, remapped,
                                     format_answer(wrong)).correct
