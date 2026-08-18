"""Shared support for the W3 (FL4-FL8 mechanisms) test modules.

OWNERSHIP: W3.  Not a test module (its name does not match ``test_*``); it
defines no test.  It builds REAL objects only — the real tiny Ouro model
(``training.tiny_model``), real generator families (``ecology.families``) and
real episodes (``episodes.assemble``).  Nothing here is a stand-in for a
producer module: if a producer is missing, the tests fail loudly.
"""
from __future__ import annotations

import os
import sys
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from foundation_learner.ecology.base import make_rng, derive_seed  # noqa: E402
from foundation_learner.ecology.families import get_family  # noqa: E402
from foundation_learner.ecology.split import split_of  # noqa: E402
from foundation_learner.episodes.assemble import assemble_episode  # noqa: E402

#: a TRAIN family with short prompts (Amendment 2 assignment)
TRAIN_FAMILY = "modular_arithmetic"
#: a DEVELOPMENT family
DEV_FAMILY = "boolean_rule"
#: a SEALED_TEST family (used ONLY to prove that guards refuse it)
SEALED_FAMILY = "constraint_rules"

TINY_SEED = 4242
ROOT_SEED = 20260809

_BUNDLE_CACHE: dict[tuple[int, str], Any] = {}


def tiny_bundle(seed: int = TINY_SEED, device: str = "cpu"):
    """The real tiny Ouro model (contract §18); cached per (seed, device)."""
    from foundation_learner.training.tiny_model import build_tiny_model

    key = (int(seed), str(device))
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE[key] = build_tiny_model(seed, device)
    return _BUNDLE_CACHE[key]


def fresh_tiny_bundle(seed: int = TINY_SEED, device: str = "cpu"):
    """An UNCACHED tiny model (needed whenever a test mutates the model)."""
    from foundation_learner.training.tiny_model import build_tiny_model

    return build_tiny_model(seed, device)


def real_episode(family_id: str = TRAIN_FAMILY, *, index: int = 0,
                 reveal: bool = False, structured: bool = True,
                 condition: str = "clean", mode: str = "scripted",
                 difficulty: int = 1):
    """A real assembled episode of ``family_id`` (its true split is used)."""
    family = get_family(family_id)
    split = split_of(family_id)
    rule_rng = make_rng(derive_seed(ROOT_SEED, "W3_TEST_RULE", family_id, index))
    rule = family.sample_rule(rule_rng)
    episode_seed = derive_seed(ROOT_SEED, "W3_TEST_EPISODE", family_id, index)
    return assemble_episode(
        family, rule, split=split, difficulty=difficulty, root_seed=ROOT_SEED,
        episode_seed=episode_seed, mode=mode, condition=condition,
        structured=structured, reveal=reveal)


def real_environment(episode, family_id: str | None = None):
    """The exact-verifier environment W4's walker consumes for ``episode``."""
    from foundation_learner.evaluation.learning_curve import environment_from_episode

    return environment_from_episode(
        episode, get_family(family_id or episode.family_id))


def feedback_event_indices(episode) -> list[int]:
    from foundation_learner.episodes.schema import Role

    return [i for i, e in enumerate(episode.events)
            if e.role in (Role.FEEDBACK, Role.HINT, Role.REVEAL)]


def embedding_matrix(bundle):
    return bundle.model.get_input_embeddings().weight.detach()
