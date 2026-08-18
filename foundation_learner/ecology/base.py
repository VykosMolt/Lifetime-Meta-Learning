"""Hashing, seeds, sealing, and the frozen ``TaskFamily`` interface.

Implements contract §3 (determinism/hash conventions), §4 (task-family
interface, ``TaskInstance``/``VerifierResult``) and Amendment 1 (sealed-shard
key derivation).  Pure stdlib + numpy: nothing under ``ecology/``,
``episodes/`` or ``data/`` may import torch or transformers except the lazily
imported tokenizer-length assertion in the pre-generation path.

Randomness is always an explicit ``numpy.random.Generator(PCG64(seed))``
derived through :func:`derive_seed`; the global RNG and wall-clock are never
used in a scientific path.

INSTANCE KIND ENCODING.  ``TaskInstance`` has exactly the nine contract fields,
so the *kind* of an instance (support / related / query / transfer) cannot be
stored as a separate field.  It is carried in the two low bits of ``seed``
(:func:`encode_seed` / :func:`decode_seed`).  Every instance is therefore a
pure function of ``(rule, seed, difficulty, surface_map_id)``, which lets
``verify`` and ``structured_feedback`` reconstruct the full latent state of an
item from the instance alone — no hidden side channel and no extra schema.

FEEDBACK LEAK INVARIANT.  Structured feedback is rendered exclusively from
opaque letter codes (``TAG_<CODE>``): it contains no digits and no answer
payload token.  :func:`answer_leak` is the machine check; every family's
feedback is validated against it on every call (contract §4, hostile-tested).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "canonical_json", "sha256_bytes", "sha256_canonical", "domain_sha256",
    "sha256_file", "sha256_tree", "derive_seed", "make_rng",
    "sealed_key", "seal_bytes", "unseal_bytes",
    "KIND_SUPPORT", "KIND_RELATED", "KIND_QUERY", "KIND_TRANSFER",
    "KIND_NAMES", "encode_seed", "decode_seed",
    "Rule", "TaskInstance", "VerifierResult", "TaskFamily", "Item",
    "HintField", "Feedback", "letters", "answer_leak", "AnswerLeakError",
    "MAX_HINT_OPTIONS",
    "CANONICAL_SURFACE", "GENERATOR_VERSION",
]

GENERATOR_VERSION = "1.0.0"
CANONICAL_SURFACE = "canonical"


# --------------------------------------------------------------------------
# hashing (contract §3; identical conventions to the O1 programme)
# --------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Canonical JSON text: sorted keys, minimal separators, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_canonical(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def domain_sha256(domain: str, obj: Any) -> str:
    """Domain-tagged canonical digest; distinct domains cannot collide."""
    payload = domain.encode("utf-8") + b"\0" + canonical_json(obj).encode("utf-8")
    return sha256_bytes(payload)


def sha256_file(path: str) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path: str) -> str:
    """Directory-tree digest (O1 ``deploy/verify_artifacts.py`` convention)."""
    if os.path.isfile(path):
        return sha256_file(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"sha256_tree: path does not exist: {path}")
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(bytes.fromhex(sha256_file(full)))
            h.update(b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------

def derive_seed(root_seed: int, *tags: Any) -> int:
    """Substream seed (contract §3).

    ``derive_seed(root, *tags) =
    int(domain_sha256("FL_V0_SEED", [root, *tags])[:16], 16) % 2**63``
    """
    payload = [int(root_seed), *list(tags)]
    return int(domain_sha256("FL_V0_SEED", payload)[:16], 16) % (2 ** 63)


def make_rng(seed: int) -> np.random.Generator:
    """The only RNG constructor used anywhere in a scientific path."""
    return np.random.Generator(np.random.PCG64(int(seed)))


# --------------------------------------------------------------------------
# sealed-shard cipher (Amendment 1)
# --------------------------------------------------------------------------

def sealed_key(split_manifest_sha256_hex: str) -> bytes:
    """``K_seal = sha256(b"FL_V0_SEALED_KEY\\0" + split_manifest_sha256_hex)``."""
    if not re.fullmatch(r"[0-9a-f]{64}", split_manifest_sha256_hex):
        raise ValueError("split_manifest_sha256_hex must be a 64-char lowercase hex digest")
    return hashlib.sha256(b"FL_V0_SEALED_KEY\0"
                          + split_manifest_sha256_hex.encode("utf-8")).digest()


def _keystream(key: bytes, n: int) -> bytes:
    """SHA-256 counter keystream: block i = sha256(key + i.to_bytes(8,'big'))."""
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(key + i.to_bytes(8, "big")).digest()
        i += 1
    return bytes(out[:n])


def seal_bytes(data: bytes, key: bytes) -> bytes:
    """XOR ``data`` with the SHA-256 counter keystream of ``key``."""
    ks = _keystream(key, len(data))
    return bytes(a ^ b for a, b in zip(data, ks))


def unseal_bytes(data: bytes, key: bytes) -> bytes:
    """Inverse of :func:`seal_bytes` (the cipher is an involution).

    Pure function.  The ONLY caller permitted to derive ``key`` and invoke this
    on sealed shards is ``campaign/sealed_gate.py``, after the development
    decisions are frozen and the single-use opening ledger entry is written.
    This package ships no decipher convenience CLI.
    """
    return seal_bytes(data, key)


# --------------------------------------------------------------------------
# instance kinds
# --------------------------------------------------------------------------

KIND_SUPPORT = 0
KIND_RELATED = 1
KIND_QUERY = 2
KIND_TRANSFER = 3
KIND_NAMES = {KIND_SUPPORT: "support", KIND_RELATED: "related",
              KIND_QUERY: "query", KIND_TRANSFER: "transfer"}


def encode_seed(raw_seed: int, kind: int) -> int:
    """Fold the instance kind into the two low bits of the instance seed."""
    if kind not in KIND_NAMES:
        raise ValueError(f"unknown instance kind {kind!r}")
    if int(raw_seed) < 0:
        raise ValueError("raw_seed must be non-negative")
    return (int(raw_seed) << 2) | int(kind)


def decode_seed(seed: int) -> tuple[int, int]:
    """Inverse of :func:`encode_seed`: ``(raw_seed, kind)``."""
    return int(seed) >> 2, int(seed) & 3


# --------------------------------------------------------------------------
# answer-leak invariant
# --------------------------------------------------------------------------

_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


class AnswerLeakError(AssertionError):
    """Raised when feedback text would expose a pending item's answer."""


def answer_leak(text: str, answer_canonical: str) -> bool:
    """True if ``text`` exposes ``answer_canonical``.

    Two checks:

    * substring — for answers of length >= 3 or containing whitespace (the
      structured answers: sequences, formulas, sets, register states, and the
      multi-character labels), the payload must not appear anywhere;
    * token — no alphanumeric token of the payload may appear as a standalone
      alphanumeric token of ``text``.  This is the check that protects short
      labels such as ``0``/``1``/``T``/``NO`` from being revealed while still
      permitting them to occur inside unrelated words.
    """
    ans = answer_canonical.strip()
    if not ans:
        return False
    if len(ans) >= 3 or any(c.isspace() for c in ans):
        if ans in text:
            return True
    text_tokens = {t for t in _TOKEN_SPLIT.split(text) if t}
    ans_tokens = {t for t in _TOKEN_SPLIT.split(ans) if t}
    return bool(ans_tokens & text_tokens)


# --------------------------------------------------------------------------
# structured feedback records
# --------------------------------------------------------------------------

_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Hard cap on how many values one opaque hint field may take.  It bounds the
#: set of letter codes a hint can ever render to ``{A..Z} u {AA..DX}``, which is
#: what makes the "no answer label is also a hint code" precondition of the
#: leak invariant checkable (tests/test_families_feedback.py).
MAX_HINT_OPTIONS = 128


def letters(n: int) -> str:
    """Digit-free bijective base-26 code for a non-negative integer."""
    n = int(n)
    if n < 0:
        raise ValueError("letters() requires n >= 0")
    out = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        out = _ALPHA[r] + out
    return out


@dataclass(frozen=True)
class HintField:
    """One opaque field of a structured hint.

    ``name`` is the lowercase field label, ``tag`` an uppercase namespace, and
    ``value`` an index in ``[0, n_options)``.  When ``names`` is given the value
    renders as ``TAG_<names[value]>``, otherwise as ``TAG_<letters(value)>``.
    Rendered feedback is digit-free by construction, which is what makes the
    leak invariant checkable.
    """

    name: str
    tag: str
    value: int
    n_options: int
    names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_options < 1:
            raise ValueError(f"{self.name}: n_options must be >= 1")
        if self.n_options > MAX_HINT_OPTIONS:
            raise ValueError(
                f"{self.name}: n_options {self.n_options} exceeds the frozen "
                f"cap {MAX_HINT_OPTIONS}; opaque codes must stay short enough "
                f"that no answer label can coincide with one")
        if not 0 <= self.value < self.n_options:
            raise ValueError(
                f"{self.name}: value {self.value} outside [0,{self.n_options})")
        if self.names is not None and len(self.names) != self.n_options:
            raise ValueError(f"{self.name}: names length != n_options")
        if not re.fullmatch(r"[a-z_]+", self.name):
            raise ValueError(f"bad hint field name {self.name!r}")
        if not re.fullmatch(r"[A-Z]+", self.tag):
            raise ValueError(f"bad hint field tag {self.tag!r}")

    def code(self) -> str:
        return self.names[self.value] if self.names else letters(self.value)

    def render(self) -> str:
        return f"{self.name}={self.tag}_{self.code()}"

    def with_value(self, value: int) -> "HintField":
        return HintField(self.name, self.tag, int(value), self.n_options, self.names)


@dataclass(frozen=True)
class Feedback:
    """A structured-feedback record (rendered to text by :meth:`render`)."""

    code: str
    fields: tuple[HintField, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"HINT-[A-Z]+", self.code):
            raise ValueError(f"bad feedback code {self.code!r}")
        if not self.fields:
            raise ValueError("a Feedback needs at least one field")
        if not any(f.n_options >= 2 for f in self.fields):
            raise ValueError(
                "a Feedback needs at least one field with n_options >= 2 so "
                "that deterministic corruption is always well defined")

    def render(self) -> str:
        text = self.code + ": " + "; ".join(f.render() for f in self.fields)
        if any(c.isdigit() for c in text):
            raise ValueError(f"structured feedback must be digit-free: {text!r}")
        return text

    def with_fields(self, fields: Sequence[HintField]) -> "Feedback":
        return Feedback(self.code, tuple(fields))


# --------------------------------------------------------------------------
# rules, instances, verifier results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """A latent generator rule: a family id plus a JSON-serialisable payload."""

    family_id: str
    params: dict

    def spec(self, generator_version: str = GENERATOR_VERSION) -> dict:
        return {"family_id": self.family_id,
                "generator_version": generator_version,
                "params": self.params}


@dataclass(frozen=True)
class TaskInstance:
    """Frozen contract §4 instance record (exactly these nine fields)."""

    family_id: str
    generator_version: str
    rule_id: str
    seed: int
    difficulty: int
    surface_map_id: str
    prompt_text: str
    answer_canonical: str
    instance_id: str

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "generator_version": self.generator_version,
            "rule_id": self.rule_id,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "surface_map_id": self.surface_map_id,
            "prompt_text": self.prompt_text,
            "answer_canonical": self.answer_canonical,
            "instance_id": self.instance_id,
        }

    @staticmethod
    def from_dict(d: dict) -> "TaskInstance":
        return TaskInstance(**d)

    @property
    def kind(self) -> str:
        return KIND_NAMES[decode_seed(self.seed)[1]]


def make_instance_id(family_id: str, rule_id: str, seed: int, difficulty: int,
                     surface_map_id: str, prompt_text: str,
                     answer_canonical: str) -> str:
    return domain_sha256("FL_V0_INSTANCE", {
        "family_id": family_id, "rule_id": rule_id, "seed": int(seed),
        "difficulty": int(difficulty), "surface_map_id": surface_map_id,
        "prompt_text": prompt_text, "answer_canonical": answer_canonical})


@dataclass(frozen=True)
class VerifierResult:
    correct: bool
    canonical_answer: str
    parsed: str | None

    def to_dict(self) -> dict:
        return {"correct": self.correct, "canonical_answer": self.canonical_answer,
                "parsed": self.parsed}


@dataclass
class Item:
    """A family's internal view of one generated task.

    ``spec`` is the surface-independent prompt specification; ``data`` holds
    whatever latent state the family's feedback functions need (inputs, truth,
    sizes).  Items are pure functions of ``(rule, seed, difficulty)``.
    """

    spec: Any  # surface_remap.PromptSpec (late-bound to avoid a cycle)
    data: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# the frozen task-family interface (contract §4)
# --------------------------------------------------------------------------

class TaskFamily(ABC):
    """Frozen contract §4 interface.

    Subclasses supply ``sample_rule``, ``_item``, ``_feedback`` and
    ``plausible_error``; the generic machinery here supplies instance
    construction, exact verification, surface remapping and feedback rendering
    so that all twelve families share one auditable code path.

    ``plausible_error`` is an interface extension required by contract §6
    ("a family-specific plausible-error sampler"); it lives on the family
    because only the family knows what a plausible wrong answer looks like.
    """

    family_id: str = ""
    generator_version: str = GENERATOR_VERSION
    canon_mode: str = "label"
    #: superset of every symbol any instance of this family may use
    symbol_pool: tuple[str, ...] = ()
    #: disjoint symbol namespaces; renaming permutes WITHIN each block
    symbol_blocks: tuple[tuple[str, ...], ...] = ()
    #: alternative label vocabularies for surface remapping (all same length)
    label_variants: tuple[tuple[str, ...], ...] = ()
    #: how many distinct difficulty levels (contract requires >= 3)
    n_difficulty_levels: int = 3

    # -- required family hooks ---------------------------------------------

    @abstractmethod
    def sample_rule(self, rng: np.random.Generator) -> Rule:
        """Draw a fresh latent rule."""

    @abstractmethod
    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        """Build the surface-independent item for ``(rule, seed, difficulty)``."""

    @abstractmethod
    def _feedback(self, rule: Rule, item: Item, plan: Any, attempt: str | None,
                  truth: str, rng: np.random.Generator) -> Feedback:
        """Structured feedback record; never encodes the pending answer."""

    @abstractmethod
    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        """A plausible WRONG canonical payload in the instance's surface."""

    # -- generic machinery --------------------------------------------------

    def difficulty_levels(self) -> list[int]:
        return list(range(self.n_difficulty_levels))

    def blocks(self) -> tuple[tuple[str, ...], ...]:
        """Symbol namespaces used by the surface remapper."""
        return self.symbol_blocks or ((self.symbol_pool,) if self.symbol_pool else ())

    def rule_spec(self, rule: Rule) -> dict:
        return rule.spec(self.generator_version)

    def rule_id(self, rule: Rule) -> str:
        return domain_sha256("FL_V0_RULE", self.rule_spec(rule))

    def _check_difficulty(self, difficulty: int) -> int:
        if int(difficulty) not in self.difficulty_levels():
            raise ValueError(f"{self.family_id}: bad difficulty {difficulty}")
        return int(difficulty)

    def instance(self, rule: Rule, seed: int, difficulty: int,
                 surface_map_id: str = CANONICAL_SURFACE) -> TaskInstance:
        """Build the instance identified by ``(rule, seed, difficulty, surface)``."""
        from .surface_remap import build_plan, render_prompt  # local: cycle-free

        difficulty = self._check_difficulty(difficulty)
        raw, kind = decode_seed(seed)
        item = self._item(rule, raw, kind, difficulty)
        plan = build_plan(self.family_id, surface_map_id, item.spec,
                          self.blocks(), self.label_variants)
        prompt_text, answer_canonical = render_prompt(item.spec, plan,
                                                      self.canon_mode)
        rid = self.rule_id(rule)
        return TaskInstance(
            family_id=self.family_id,
            generator_version=self.generator_version,
            rule_id=rid,
            seed=int(seed),
            difficulty=difficulty,
            surface_map_id=surface_map_id,
            prompt_text=prompt_text,
            answer_canonical=answer_canonical,
            instance_id=make_instance_id(self.family_id, rid, seed, difficulty,
                                         surface_map_id, prompt_text,
                                         answer_canonical),
        )

    def _draw(self, rule: Rule, rng: np.random.Generator, difficulty: int,
              kind: int) -> TaskInstance:
        raw = int(rng.integers(0, 1 << 44))
        return self.instance(rule, encode_seed(raw, kind), difficulty)

    def sample_instance(self, rule: Rule, rng: np.random.Generator,
                        difficulty: int) -> TaskInstance:
        return self._draw(rule, rng, difficulty, KIND_SUPPORT)

    def related_instance(self, rule: Rule, rng: np.random.Generator,
                         difficulty: int) -> TaskInstance:
        return self._draw(rule, rng, difficulty, KIND_RELATED)

    def query_instance(self, rule: Rule, rng: np.random.Generator,
                       difficulty: int) -> TaskInstance:
        return self._draw(rule, rng, difficulty, KIND_QUERY)

    def transfer_instance(self, rule: Rule, rng: np.random.Generator,
                          difficulty: int) -> TaskInstance:
        return self._draw(rule, rng, difficulty, KIND_TRANSFER)

    def surface_remap(self, rule: Rule, instance: TaskInstance,
                      remap_id: str) -> TaskInstance:
        """Semantics-preserving resurfacing; latent rule unchanged."""
        if remap_id == CANONICAL_SURFACE:
            raise ValueError("remap_id 'canonical' is the identity surface")
        return self.instance(rule, instance.seed, instance.difficulty, remap_id)

    def _item_and_plan(self, rule: Rule, instance: TaskInstance):
        """Reconstruct an instance's latent item and its surface plan."""
        from .surface_remap import build_plan

        raw, kind = decode_seed(instance.seed)
        item = self._item(rule, raw, kind, instance.difficulty)
        plan = build_plan(self.family_id, instance.surface_map_id, item.spec,
                          self.blocks(), self.label_variants)
        return item, plan

    def _flip_label(self, rule: Rule, instance: TaskInstance) -> str:
        """The other answer label of a label family, in the instance surface."""
        from ..episodes.parse import canonicalize

        item, plan = self._item_and_plan(rule, instance)
        options = []
        for raw_label in item.spec.labels:
            canon = canonicalize(plan.map_label(raw_label), self.canon_mode)
            if canon is not None and canon != instance.answer_canonical:
                options.append(canon)
        if not options:
            raise ValueError(f"{self.family_id}: no alternative label available")
        return options[0]

    # -- verification -------------------------------------------------------

    def verify(self, rule: Rule, instance: TaskInstance,
               answer_text: str) -> VerifierResult:
        """EXACT verification: strict grammar parse then exact match."""
        from ..episodes.parse import canonicalize, parse_answer

        payload = parse_answer(answer_text)
        parsed = canonicalize(payload, self.canon_mode) if payload is not None else None
        return VerifierResult(
            correct=parsed is not None and parsed == instance.answer_canonical,
            canonical_answer=instance.answer_canonical,
            parsed=parsed)

    def _forced_wrong(self, answer_canonical: str) -> str:
        """A canonical payload guaranteed to differ from ``answer_canonical``.

        Used only when a family's plausible-error sampler happens to land on
        the truth (palindromic outputs, degenerate rules).  Deliberately less
        plausible than the family sampler; the attempt policy never relies on
        it for realism, only for the never-accidentally-correct guarantee.
        """
        mode = self.canon_mode
        if mode == "int":
            return str(int(answer_canonical) + 1)
        if mode == "label":
            return answer_canonical + "X"
        if mode in ("string", "formula"):
            return answer_canonical + "z"
        if mode == "token_list":
            tokens = answer_canonical.split()
            return " ".join(tokens + [tokens[0]])
        if mode == "set":
            if answer_canonical == "EMPTY":
                return "zz"
            return " ".join(sorted(set(answer_canonical.split()) | {"zz"}))
        if mode == "assignment":
            return answer_canonical + " zz=1"
        raise ValueError(f"no forced-wrong rule for canon mode {mode!r}")

    def wrong_answer(self, rule: Rule, instance: TaskInstance,
                     rng: np.random.Generator) -> str:
        """A canonical payload that is plausible AND guaranteed incorrect."""
        candidate = self.plausible_error(rule, instance, rng)
        if candidate == instance.answer_canonical:
            candidate = self._forced_wrong(instance.answer_canonical)
        if candidate == instance.answer_canonical:
            raise AssertionError(
                f"{self.family_id}: could not construct a wrong answer for "
                f"{instance.instance_id[:16]}")
        return candidate

    # -- feedback -----------------------------------------------------------

    def feedback_record(self, rule: Rule, instance: TaskInstance,
                        answer_text: str,
                        rng: np.random.Generator) -> Feedback:
        """Structured feedback as a record (poison operates on records).

        Feedback content is a DETERMINISTIC function of
        ``(rule, instance, attempt)``: any family tie-break draws from a seed
        derived from the instance, not from the caller's generator.  ``rng`` is
        accepted for interface compatibility (contract §4) and deliberately not
        consumed, because a stochastic hint would break the guarantee that a
        corrupted hint differs from the true hint for the same item.
        """
        from ..episodes.parse import canonicalize, parse_answer

        item, plan = self._item_and_plan(rule, instance)
        payload = parse_answer(answer_text)
        attempt = canonicalize(payload, self.canon_mode) if payload is not None else None
        local = make_rng(derive_seed(0, "FL_V0_FEEDBACK", self.family_id,
                                     instance.instance_id, str(attempt)))
        fb = self._feedback(rule, item, plan, attempt,
                            instance.answer_canonical, local)
        text = fb.render()
        if answer_leak(text, instance.answer_canonical):
            raise AnswerLeakError(
                f"{self.family_id}: structured feedback would reveal the "
                f"pending answer: {text!r} vs {instance.answer_canonical!r}")
        return fb

    def structured_feedback(self, rule: Rule, instance: TaskInstance,
                            answer_text: str,
                            rng: np.random.Generator) -> str:
        """Informative feedback that never reveals the pending answer."""
        return self.feedback_record(rule, instance, answer_text, rng).render()

    def corrupted_feedback(self, rule: Rule, instance: TaskInstance,
                           answer_text: str,
                           rng: np.random.Generator) -> str:
        """Deterministically corrupted feedback, guaranteed != the true one."""
        from .poison import corrupt_feedback

        record = self.feedback_record(rule, instance, answer_text, rng)
        local = make_rng(derive_seed(0, "FL_V0_CORRUPT", self.family_id,
                                     instance.instance_id, answer_text))
        bad = corrupt_feedback(record, local, severity="full")
        text = bad.render()
        if text == record.render():
            raise AssertionError(f"{self.family_id}: corruption was a no-op")
        if answer_leak(text, instance.answer_canonical):
            raise AnswerLeakError(
                f"{self.family_id}: corrupted feedback would reveal the answer")
        return text
