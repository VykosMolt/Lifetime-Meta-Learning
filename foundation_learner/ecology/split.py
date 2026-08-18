"""The frozen public family split (contract §5, Amendment 2).

For each of the twelve frozen ``family_id`` strings::

    h(fid) = sha256(b"FOUNDATION_LEARNER_V0_FAMILY_SPLIT\\0" + fid.encode())

Sort ascending by hex digest; positions 0-5 -> TRAIN, 6-8 -> DEVELOPMENT,
9-11 -> SEALED_TEST.  No manual placement, no re-rolls, no post-hoc
adjustment.  The rule was frozen before it was computed; the resulting
assignment is pinned by ``tests/test_split.py`` and recorded in Amendment 2.
"""
from __future__ import annotations

import hashlib

__all__ = ["SPLIT_DOMAIN", "FAMILY_IDS", "SPLITS", "N_TRAIN", "N_DEV",
           "N_SEALED", "family_hash", "ordered_families", "compute_split",
           "split_of", "families_in", "EXPECTED_SPLIT"]

SPLIT_DOMAIN = "FOUNDATION_LEARNER_V0_FAMILY_SPLIT"

#: The twelve frozen family ids, in contract §4 order.
FAMILY_IDS: tuple[str, ...] = (
    "boolean_rule",
    "propositional_transform",
    "modular_arithmetic",
    "sequence_transform",
    "string_rewrite",
    "finite_state_transducer",
    "permutation_composition",
    "set_operations",
    "graph_edge_semantics",
    "dsl_execution",
    "constraint_rules",
    "grammar_classification",
)

SPLITS: tuple[str, ...] = ("TRAIN", "DEVELOPMENT", "SEALED_TEST")
N_TRAIN, N_DEV, N_SEALED = 6, 3, 3

#: Amendment 2: the one mechanical execution of the frozen rule.
EXPECTED_SPLIT: dict[str, tuple[str, ...]] = {
    "TRAIN": ("grammar_classification", "set_operations", "dsl_execution",
              "string_rewrite", "finite_state_transducer", "modular_arithmetic"),
    "DEVELOPMENT": ("sequence_transform", "graph_edge_semantics", "boolean_rule"),
    "SEALED_TEST": ("constraint_rules", "permutation_composition",
                    "propositional_transform"),
}


def family_hash(family_id: str) -> str:
    """``sha256(domain + b"\\0" + family_id)`` hex digest."""
    return hashlib.sha256(SPLIT_DOMAIN.encode("utf-8") + b"\0"
                          + family_id.encode("utf-8")).hexdigest()


def ordered_families() -> list[tuple[str, str]]:
    """``(family_id, hash)`` pairs sorted ascending by hex digest."""
    return sorted(((fid, family_hash(fid)) for fid in FAMILY_IDS),
                  key=lambda p: p[1])


def compute_split() -> dict[str, list[str]]:
    """Apply the frozen rule; returns split name -> ordered family ids."""
    order = [fid for fid, _ in ordered_families()]
    return {
        "TRAIN": order[:N_TRAIN],
        "DEVELOPMENT": order[N_TRAIN:N_TRAIN + N_DEV],
        "SEALED_TEST": order[N_TRAIN + N_DEV:],
    }


def split_of(family_id: str) -> str:
    """The split a family belongs to (exactly one)."""
    for name, fids in compute_split().items():
        if family_id in fids:
            return name
    raise KeyError(f"unknown family_id {family_id!r}")


def families_in(split: str) -> list[str]:
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}")
    return list(compute_split()[split])
