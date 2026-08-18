"""Frozen pool sizes and generation constants (contract §7, §16).

TRAIN 2,000 episodes/family (6 families -> 12,000), DEVELOPMENT 300/family,
SEALED_TEST 300/family.  Surface-remap variants exist ONLY for DEVELOPMENT and
SEALED_TEST (2 per episode): the remap evaluation asks whether a learner
carries a latent rule across surfaces, so remapped surfaces must not be
training material.  Poison schedules cover all five conditions per episode.
FL4 ablation PLANS are event-subset specifications, never duplicated text.
200 A->B->A chains per ORDERED DEVELOPMENT family pair.

``--tiny`` mode divides every pool by 100 (floor, minimum 2) for tests and the
dress rehearsal; it changes no seed derivation, only how many episodes are
drawn, so a tiny run is a strict prefix-like subsample of the same stream.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ROOT_SEED", "SECOND_SEED", "DEFAULT_DIFFICULTY", "MAX_SEQ_LEN",
           "EVAL_ONLINE_MAX_SEQ_LEN", "EVAL_MAX_NEW_TOKENS",
           "PoolSizes", "FULL_POOLS", "tiny_pools", "SPLIT_POOL_KEYS",
           "REMAP_SPLITS", "N_REMAP_VARIANTS", "N_CHAINS_PER_PAIR",
           "TINY_DIVISOR", "FL1_ITEMS_PER_EPISODE"]

#: primary root seed (contract §7); 20260810 is the predeclared second seed
ROOT_SEED = 20260809
SECOND_SEED = 20260810

#: frozen campaign difficulty for every arm (level 1 = middle)
DEFAULT_DIFFICULTY = 1

#: contract §7 sequence budget; episodes are SIZED to fit, never truncated
MAX_SEQ_LEN = 2048

#: Frozen EVAL-TIME online context allowance (Amendment 13).  The ONLINE
#: evaluator replaces every scripted attempt line with up to
#: ``EVAL_MAX_NEW_TOKENS`` real generated tokens, so the online context grows
#: past the 2048 rendering budget; the same constant lives in
#: ``evaluation.learning_curve.EVAL_ONLINE_MAX_SEQ_LEN`` (a test pins the two
#: together) and is duplicated here because nothing under ``data/`` may import
#: the torch-dependent evaluation package.
EVAL_ONLINE_MAX_SEQ_LEN = 4096

#: contract §6: frozen online decode budget per MODEL_ATTEMPT slot
EVAL_MAX_NEW_TOKENS = 64

TINY_DIVISOR = 100

#: splits that receive surface-remapped variants
REMAP_SPLITS = ("DEVELOPMENT", "SEALED_TEST")
N_REMAP_VARIANTS = 2
N_CHAINS_PER_PAIR = 200

#: how many isolated (TASK -> ANSWER) items FL1 draws from one episode
FL1_ITEMS_PER_EPISODE = 3


@dataclass(frozen=True)
class PoolSizes:
    train: int
    development: int
    sealed_test: int
    chains_per_pair: int
    remap_variants: int

    def for_split(self, split: str) -> int:
        return {"TRAIN": self.train, "DEVELOPMENT": self.development,
                "SEALED_TEST": self.sealed_test}[split]


FULL_POOLS = PoolSizes(train=2000, development=300, sealed_test=300,
                       chains_per_pair=N_CHAINS_PER_PAIR,
                       remap_variants=N_REMAP_VARIANTS)

SPLIT_POOL_KEYS = ("TRAIN", "DEVELOPMENT", "SEALED_TEST")


def tiny_pools() -> PoolSizes:
    """Pools divided by 100 (minimum 2) for tests and rehearsal."""
    def shrink(n: int) -> int:
        return max(2, n // TINY_DIVISOR)

    return PoolSizes(train=shrink(FULL_POOLS.train),
                     development=shrink(FULL_POOLS.development),
                     sealed_test=shrink(FULL_POOLS.sealed_test),
                     chains_per_pair=shrink(FULL_POOLS.chains_per_pair),
                     remap_variants=FULL_POOLS.remap_variants)
