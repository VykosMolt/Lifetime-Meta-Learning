"""The twelve frozen generator families (contract §4).

Each module exposes a module-level ``FAMILY`` singleton implementing
``ecology.base.TaskFamily``.  The registry order is the contract §4 order; the
train/dev/sealed assignment is computed independently in ``ecology.split``.
"""
from __future__ import annotations

from ..base import TaskFamily
from ..split import FAMILY_IDS
from .boolean_rule import FAMILY as BOOLEAN_RULE
from .constraint_rules import FAMILY as CONSTRAINT_RULES
from .dsl_execution import FAMILY as DSL_EXECUTION
from .finite_state_transducer import FAMILY as FINITE_STATE_TRANSDUCER
from .grammar_classification import FAMILY as GRAMMAR_CLASSIFICATION
from .graph_edge_semantics import FAMILY as GRAPH_EDGE_SEMANTICS
from .modular_arithmetic import FAMILY as MODULAR_ARITHMETIC
from .permutation_composition import FAMILY as PERMUTATION_COMPOSITION
from .propositional_transform import FAMILY as PROPOSITIONAL_TRANSFORM
from .sequence_transform import FAMILY as SEQUENCE_TRANSFORM
from .set_operations import FAMILY as SET_OPERATIONS
from .string_rewrite import FAMILY as STRING_REWRITE

__all__ = ["FAMILY_REGISTRY", "get_family", "all_families", "family_module_paths"]

_FAMILIES: tuple[TaskFamily, ...] = (
    BOOLEAN_RULE, PROPOSITIONAL_TRANSFORM, MODULAR_ARITHMETIC,
    SEQUENCE_TRANSFORM, STRING_REWRITE, FINITE_STATE_TRANSDUCER,
    PERMUTATION_COMPOSITION, SET_OPERATIONS, GRAPH_EDGE_SEMANTICS,
    DSL_EXECUTION, CONSTRAINT_RULES, GRAMMAR_CLASSIFICATION,
)

FAMILY_REGISTRY: dict[str, TaskFamily] = {f.family_id: f for f in _FAMILIES}

if tuple(FAMILY_REGISTRY) != FAMILY_IDS:
    raise RuntimeError(
        "family registry disagrees with the frozen family id list: "
        f"{tuple(FAMILY_REGISTRY)} != {FAMILY_IDS}")


def get_family(family_id: str) -> TaskFamily:
    try:
        return FAMILY_REGISTRY[family_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown family_id {family_id!r}") from exc


def all_families() -> list[TaskFamily]:
    return list(_FAMILIES)


def family_module_paths() -> dict[str, str]:
    """``family_id -> absolute path`` of the generator source file."""
    import importlib
    import os

    out = {}
    for fid in FAMILY_IDS:
        module = importlib.import_module(f"{__name__}.{fid}")
        out[fid] = os.path.abspath(module.__file__)
    return out
