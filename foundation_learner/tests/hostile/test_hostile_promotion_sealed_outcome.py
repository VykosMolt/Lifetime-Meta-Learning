"""HOSTILE: promote a stage from SEALED_TEST outcomes.

Contract §10/§12: SEALED_TEST is NEVER used for promotion.  The attack tries
every route into a promotion decision — a sealed split tag, a sealed family id
smuggled into an otherwise development-tagged object, an untagged record, a
raw dict that pretends to be metrics, and the FL8 persistence evidence.
"""
from __future__ import annotations

import pytest

from foundation_learner.campaign import promotion as promo
from foundation_learner.ecology.split import compute_split

SEALED = tuple(compute_split()["SEALED_TEST"])
DEV = tuple(compute_split()["DEVELOPMENT"])


def sealed_records(n=300, value=1.0):
    return [{"split": "SEALED_TEST", "family_id": SEALED[i % len(SEALED)],
             "R": {"0": value, "1": value, "2": value}} for i in range(n)]


def test_sealed_records_cannot_become_promotion_evidence():
    with pytest.raises(promo.SealedEvidenceRefused):
        promo.DevMetrics.from_records(sealed_records(), stage="FL3")


def test_a_sealed_split_tag_is_refused_even_with_development_families():
    with pytest.raises(promo.SealedEvidenceRefused):
        promo.DevMetrics(stage="FL3", macro_aulc=0.99, slope=1.0,
                         family_ids=DEV, split="SEALED_TEST")


def test_a_sealed_family_is_refused_even_with_a_development_tag():
    with pytest.raises(promo.SealedEvidenceRefused):
        promo.DevMetrics(stage="FL3", macro_aulc=0.99, slope=1.0,
                         family_ids=DEV + SEALED[:1])


def test_mixed_records_are_refused_wholesale():
    mixed = [{"split": "DEVELOPMENT", "family_id": DEV[0], "R": {"0": 0.1}}]
    mixed += sealed_records(1)
    with pytest.raises(promo.SealedEvidenceRefused):
        promo.DevMetrics.from_records(mixed, stage="FL3")


def test_untagged_records_are_refused_rather_than_assumed_development():
    with pytest.raises(promo.PromotionError):
        promo.DevMetrics.from_records([{"family_id": DEV[0], "R": {"0": 0.1}}],
                                      stage="FL3")


def test_the_fl3_gate_cannot_be_called_with_raw_dicts():
    fake = {"macro_aulc": 1.0, "slope": 1.0, "split": "SEALED_TEST",
            "stable": True}
    with pytest.raises((AttributeError, TypeError, promo.PromotionError)):
        promo.promote_fl3_to_extensions(fake, fake)


def test_fl8_refuses_sealed_persistence_evidence():
    with pytest.raises(promo.SealedEvidenceRefused):
        promo.promote_fl8(persistence_point=0.9, persistence_ci=[0.5, 1.0],
                          split="SEALED_TEST")


def test_assert_no_sealed_is_applied_to_every_record_not_just_the_first():
    records = [{"split": "DEVELOPMENT", "family_id": DEV[0]}] * 50
    records += [{"split": "SEALED_TEST", "family_id": SEALED[0]}]
    with pytest.raises(promo.SealedEvidenceRefused):
        promo.assert_no_sealed(records)


def test_the_promotion_module_never_reads_a_shard_or_a_key():
    source = open(promo.__file__, encoding="utf-8").read()
    for forbidden in ("read_shard", "unseal_bytes", "sealed_key", "open("):
        assert forbidden not in source, forbidden
