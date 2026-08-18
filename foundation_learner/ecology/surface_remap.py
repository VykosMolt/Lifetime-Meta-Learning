"""Semantics-preserving surface remapping (contract §4).

A family never writes a prompt string directly.  It emits a
:class:`PromptSpec`: self-contained description clauses, a question, an answer
format line, and the canonical answer — with every symbol wrapped in ``sym()``
and every answer label wrapped in ``lab()`` markers.  :func:`render_prompt`
then resolves those markers under a :class:`RemapPlan` and lays the prompt out.

A plan is a deterministic function of ``(family_id, remap_id)`` only, so all
instances of one episode share one surface:

* **symbol map** — a permutation of the family's whole ``symbol_pool``
  (bijective; restricted to whatever symbols an instance happens to use);
* **label map** — a bijection from the family's canonical answer labels onto an
  alternative vocabulary, always accompanied by an explicit legend clause, so
  the semantics of the answer is preserved;
* **format variant** — clause bullets / separators / question prefix;
* **distractors** — irrelevant sentences from a fixed pool;
* **clause order** — a permutation of the (self-contained) clauses.

The latent rule is untouched.  The remapped instance carries its own
``answer_canonical`` in the remapped surface and verifies there.

``remap_id == "canonical"`` yields the identity plan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .base import CANONICAL_SURFACE, derive_seed, make_rng

__all__ = ["PromptSpec", "RemapPlan", "sym", "lab", "syms", "build_plan",
           "render_prompt", "DISTRACTOR_POOL", "N_FORMAT_VARIANTS",
           "remap_ids_for"]

SYM_L, SYM_R = "\x01", "\x02"
LAB_L, LAB_R = "\x03", "\x04"
_SYM_RE = re.compile(SYM_L + r"([^" + SYM_R + r"]*)" + SYM_R)
_LAB_RE = re.compile(LAB_L + r"([^" + LAB_R + r"]*)" + LAB_R)

N_FORMAT_VARIANTS = 3

#: Irrelevant, digit-free, label-free filler sentences.
DISTRACTOR_POOL: tuple[str, ...] = (
    "Note: this transcript was archived for later review.",
    "Note: the ambient log level for this session is verbose.",
    "Note: the session identifier is recorded elsewhere.",
    "Note: a printable copy of this page is not available.",
    "Note: the display font may differ between terminals.",
    "Note: the operator on duty rotates every shift.",
    "Note: the room temperature is within the usual range.",
    "Note: this page intentionally carries no other content.",
    "Note: line wrapping depends on the reading device.",
    "Note: the file was transferred without compression.",
    "Note: comments in the margin are not part of the task.",
    "Note: the header colour scheme is configurable.",
)


def sym(symbol: str) -> str:
    """Wrap a remappable symbol."""
    return SYM_L + symbol + SYM_R


def lab(label: str) -> str:
    """Wrap a remappable answer label."""
    return LAB_L + label + LAB_R


def syms(sequence, sep: str = " ") -> str:
    """Wrap and join a sequence of symbols."""
    return sep.join(sym(s) for s in sequence)


@dataclass(frozen=True)
class PromptSpec:
    """Surface-independent description of one task."""

    clauses: tuple[str, ...]
    question: str
    answer_format: str
    alphabet: tuple[str, ...]
    labels: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class RemapPlan:
    """Deterministic surface plan for ``(family_id, remap_id)``."""

    family_id: str
    remap_id: str
    identity: bool
    symbol_map: dict
    label_map: dict
    format_variant: int
    distractors: tuple[str, ...]
    seed: int

    def map_symbol(self, symbol: str) -> str:
        return self.symbol_map.get(symbol, symbol)

    def map_seq(self, sequence) -> list:
        return [self.map_symbol(s) for s in sequence]

    def map_label(self, label: str) -> str:
        return self.label_map.get(label, label)

    def clause_order(self, n: int) -> tuple[int, ...]:
        if self.identity:
            return tuple(range(n))
        rng = make_rng(derive_seed(self.seed, "clause_order", n))
        return tuple(int(i) for i in rng.permutation(n))


def remap_ids_for(episode_id: str, count: int) -> list[str]:
    """The canonical remap ids attached to one episode."""
    return [f"remap-{episode_id[:16]}-{i}" for i in range(count)]


def build_plan(family_id: str, remap_id: str, spec: PromptSpec,
               symbol_blocks, label_variants) -> RemapPlan:
    """Build the (pure, uncached) plan for one surface id.

    ``symbol_blocks`` is a tuple of disjoint symbol groups; the renaming is a
    permutation WITHIN each block, so distinct namespaces (set names vs set
    elements, input alphabet vs output alphabet) never trade places.
    """
    if remap_id == CANONICAL_SURFACE:
        return RemapPlan(family_id, remap_id, True, {}, {}, 0, (), 0)

    seed = derive_seed(0, "FL_V0_REMAP", family_id, remap_id)
    rng = make_rng(seed)

    symbol_map: dict = {}
    for block in symbol_blocks:
        block = tuple(block)
        if not block:
            continue
        perm = rng.permutation(len(block))
        for i, j in enumerate(perm):
            symbol_map[block[i]] = block[int(j)]

    label_map: dict = {}
    if spec.labels and label_variants:
        variants = [v for v in label_variants if len(v) == len(spec.labels)]
        if variants:
            variant = variants[int(rng.integers(0, len(variants)))]
            order = rng.permutation(len(variant))
            label_map = {spec.labels[i]: variant[int(order[i])]
                         for i in range(len(spec.labels))}
            if all(k == v for k, v in label_map.items()):
                label_map = {}

    fmt = int(rng.integers(0, N_FORMAT_VARIANTS))
    n_distractors = int(rng.integers(0, 3))
    if n_distractors:
        picks = rng.choice(len(DISTRACTOR_POOL), size=n_distractors, replace=False)
        distractors = tuple(DISTRACTOR_POOL[int(i)] for i in sorted(picks))
    else:
        distractors = ()

    return RemapPlan(family_id, remap_id, False, symbol_map, label_map, fmt,
                     distractors, seed)


def _resolve(text: str, plan: RemapPlan) -> str:
    text = _SYM_RE.sub(lambda m: plan.map_symbol(m.group(1)), text)
    return _LAB_RE.sub(lambda m: plan.map_label(m.group(1)), text)


def _legend(spec: PromptSpec, plan: RemapPlan) -> str | None:
    if not plan.label_map:
        return None
    pairs = ", ".join(f"{plan.map_label(l)} in place of {l}"
                      for l in spec.labels)
    return f"Answer legend: write {pairs}."


def render_prompt(spec: PromptSpec, plan: RemapPlan,
                  canon_mode: str) -> tuple[str, str]:
    """Render ``(prompt_text, answer_canonical)`` in the plan's surface."""
    from ..episodes.parse import canonicalize

    clauses = [_resolve(c, plan) for c in spec.clauses]
    clauses.extend(plan.distractors)
    order = plan.clause_order(len(clauses))
    clauses = [clauses[i] for i in order]
    legend = _legend(spec, plan)
    if legend is not None:
        clauses.append(legend)

    question = _resolve(spec.question, plan)
    answer_format = _resolve(spec.answer_format, plan)

    if plan.format_variant == 0:
        body = "\n".join(clauses)
        qline = question
    elif plan.format_variant == 1:
        body = "\n".join("- " + c for c in clauses)
        qline = "Q: " + question
    else:
        body = " ".join(clauses)
        qline = "Question: " + question
    prompt = f"{body}\n{qline}\n{answer_format}"

    answer = canonicalize(_resolve(spec.answer, plan), canon_mode)
    if answer is None:
        raise ValueError(
            f"{plan.family_id}: canonical answer failed to canonicalise in "
            f"mode {canon_mode!r}: {_resolve(spec.answer, plan)!r}")
    if SYM_L in prompt or LAB_L in prompt:
        raise ValueError(f"{plan.family_id}: unresolved surface markers")
    return prompt, answer
