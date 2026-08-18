"""HOSTILE: try to make two different tasks share one identity.

Attacks, all of which must fail:

1. seed reuse — build instances with the same raw seed under different rules,
   kinds, difficulties and surfaces and hope one identity is reused;
2. content collision — construct two instances with the same prompt and answer
   under different rules and hope the ids collide;
3. ledger bypass — hand the generation ledger a genuine collision and hope it
   is silently accepted;
4. kind smuggling — hope the kind bits of a seed can be forged so a transfer
   item is served as a support item under the same identity.
"""
from __future__ import annotations

import pytest

from foundation_learner.data import generate_shards as gen
from foundation_learner.ecology.base import (KIND_NAMES, KIND_SUPPORT,
                                             KIND_TRANSFER, decode_seed,
                                             derive_seed, encode_seed,
                                             make_instance_id, make_rng)
from foundation_learner.ecology.families import all_families
from foundation_learner.ecology.split import split_of

FAMILIES = all_families()
IDS = [f.family_id for f in FAMILIES]


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_seed_reuse_across_rules_kinds_and_surfaces_never_collides(family):
    rng = make_rng(derive_seed(1, "collide", family.family_id))
    rules = [family.sample_rule(rng) for _ in range(3)]
    raw_seeds = [int(rng.integers(0, 1 << 40)) for _ in range(4)]
    seen: dict[str, tuple] = {}
    for rule_index, rule in enumerate(rules):
        for raw in raw_seeds:
            for kind in KIND_NAMES:
                for difficulty in family.difficulty_levels():
                    for surface in ("canonical", "remap-collide-0"):
                        instance = family.instance(
                            rule, encode_seed(raw, kind), difficulty, surface)
                        key = (rule_index, raw, kind, difficulty, surface)
                        prior = seen.setdefault(instance.instance_id, key)
                        assert prior == key, (prior, key)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_identity_covers_every_field_that_can_differ(family):
    rng = make_rng(derive_seed(2, "fields", family.family_id))
    rule = family.sample_rule(rng)
    instance = family.sample_instance(rule, rng, 1)
    base = dict(family_id=instance.family_id, rule_id=instance.rule_id,
                seed=instance.seed, difficulty=instance.difficulty,
                surface_map_id=instance.surface_map_id,
                prompt_text=instance.prompt_text,
                answer_canonical=instance.answer_canonical)
    assert make_instance_id(**base) == instance.instance_id
    for field, value in (("family_id", "other_family"),
                         ("rule_id", "0" * 64),
                         ("seed", instance.seed + 4),
                         ("difficulty", instance.difficulty + 1),
                         ("surface_map_id", "remap-x-9"),
                         ("prompt_text", instance.prompt_text + " "),
                         ("answer_canonical", instance.answer_canonical + "z")):
        forged = dict(base)
        forged[field] = value
        assert make_instance_id(**forged) != instance.instance_id


def test_the_generation_ledger_refuses_a_real_collision():
    ledger = gen._UniquenessLedger()
    ledger.claim_instance("id-1", "TRAIN", "dsl_execution", "key-a")
    ledger.claim_instance("id-1", "TRAIN", "dsl_execution", "key-a")  # idempotent
    with pytest.raises(ValueError):
        ledger.claim_instance("id-1", "DEVELOPMENT", "dsl_execution", "key-a")
    with pytest.raises(ValueError):
        ledger.claim_instance("id-1", "TRAIN", "set_operations", "key-a")
    with pytest.raises(ValueError):
        ledger.claim_instance("id-1", "TRAIN", "dsl_execution", "key-b")

    assert ledger.claim_rule("rule-1", "TRAIN", "dsl_execution", 0)
    assert ledger.claim_rule("rule-1", "TRAIN", "dsl_execution", 0)
    # a within-family repeat is a resample signal, not a crash
    assert not ledger.claim_rule("rule-1", "TRAIN", "dsl_execution", 1)
    # a cross-family repeat is impossible and must be loud
    with pytest.raises(ValueError):
        ledger.claim_rule("rule-1", "TRAIN", "set_operations", 0)


def test_rule_redraw_on_collision_is_deterministic_and_bounded():
    family = FAMILIES[0]
    ledger = gen._UniquenessLedger()
    first = gen._draw_unique_rule(family, ledger, 20260809,
                                  split_of(family.family_id), 0)
    fresh = gen._UniquenessLedger()
    again = gen._draw_unique_rule(family, fresh, 20260809,
                                  split_of(family.family_id), 0)
    assert first[1] == again[1] and first[2] == again[2]
    # forcing every candidate to look used must fail loudly, never silently
    class _AlwaysUsed(gen._UniquenessLedger):
        def claim_rule(self, *args, **kwargs):
            return False

    with pytest.raises(ValueError):
        gen._draw_unique_rule(family, _AlwaysUsed(), 20260809, "TRAIN", 0)


@pytest.mark.parametrize("family", FAMILIES, ids=IDS)
def test_kind_bits_cannot_be_forged_without_changing_the_identity(family):
    rng = make_rng(derive_seed(3, "kind", family.family_id))
    rule = family.sample_rule(rng)
    raw = int(rng.integers(0, 1 << 40))
    support = family.instance(rule, encode_seed(raw, KIND_SUPPORT), 1)
    transfer = family.instance(rule, encode_seed(raw, KIND_TRANSFER), 1)
    assert support.instance_id != transfer.instance_id
    assert decode_seed(support.seed)[1] == KIND_SUPPORT
    assert decode_seed(transfer.seed)[1] == KIND_TRANSFER
    assert support.kind == "support" and transfer.kind == "transfer"
    assert support.prompt_text != transfer.prompt_text
