"""Episode schema (contract §6): roles, events, ``EPISODE_STRUCTURE_V0``.

Nothing is flattened into undifferentiated next-token data.  Every event
carries ``{role, text, item_id, interaction_index, certified, poison, reveal}``
plus a rendering ``slot`` label (``P1``, ``Q2``, ...) that the renderer needs
for the ``REVEALED ANSWER (P#)`` marker.  ``poison`` is an ENVIRONMENT-ONLY
flag: it is recorded in the shard record and never rendered into the text the
model sees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from ..ecology.base import domain_sha256

__all__ = ["Role", "Event", "EpisodeStructure", "EPISODE_STRUCTURE_V0",
           "Episode", "make_episode_id", "make_episode_key", "TASK_ROLES",
           "ATTEMPT_ROLE", "FEEDBACK_ROLES"]


class Role(str, Enum):
    TASK = "TASK"
    SUPPORT_OBS = "SUPPORT_OBS"
    MODEL_ATTEMPT = "MODEL_ATTEMPT"
    FEEDBACK = "FEEDBACK"
    HINT = "HINT"
    REVEAL = "REVEAL"
    ADAPT_EVENT = "ADAPT_EVENT"
    RELATED_TASK = "RELATED_TASK"
    QUERY_TASK = "QUERY_TASK"
    TRANSFER_TASK = "TRANSFER_TASK"


TASK_ROLES = (Role.TASK, Role.RELATED_TASK, Role.QUERY_TASK, Role.TRANSFER_TASK)
ATTEMPT_ROLE = Role.MODEL_ATTEMPT
#: the events an FL4 ablation may remove and a fast state may consume
FEEDBACK_ROLES = (Role.FEEDBACK, Role.HINT, Role.REVEAL)


@dataclass(frozen=True)
class Event:
    """One episode event.

    ``slot`` is the rendering label (``P1``, ``Q2``, ...) the renderer needs for
    the ``REVEALED ANSWER (P#)`` marker.  ``meta`` is the contract §23
    environment-only side table; it is never rendered and never shown to the
    model.
    """

    role: Role
    text: str
    item_id: str
    interaction_index: int | None
    certified: bool
    poison: bool
    reveal: bool
    slot: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["role"] = self.role.value
        return out

    @staticmethod
    def from_dict(d: dict) -> "Event":
        d = dict(d)
        d["role"] = Role(d["role"])
        d.setdefault("meta", {})
        return Event(**d)


@dataclass(frozen=True)
class EpisodeStructure:
    """Frozen interaction layout.  Configurable in code, frozen for v0."""

    name: str
    n_support: int
    attempts_per_support: int
    n_related: int
    n_queries: int
    n_transfers: int
    k_max: int
    related_index: int
    query_index: int
    transfer_index: int

    def attempt_indices(self) -> list[int]:
        return list(range(self.k_max + 1))

    def revision_indices(self) -> list[int]:
        """Interaction indices of the second attempt on each support item."""
        return [i * self.attempts_per_support + 1 for i in range(self.n_support)]

    def attempt0_indices(self) -> list[int]:
        return [i * self.attempts_per_support for i in range(self.n_support)]


EPISODE_STRUCTURE_V0 = EpisodeStructure(
    name="EPISODE_STRUCTURE_V0",
    n_support=2,
    attempts_per_support=2,
    n_related=1,
    n_queries=3,
    n_transfers=2,
    k_max=6,
    related_index=4,
    query_index=5,
    transfer_index=6,
)


def make_episode_id(family_id: str, split: str, rule_id: str, seed: int,
                    difficulty: int, mode: str, structure: str,
                    structured: bool, reveal: bool, arm_tag: str = "") -> str:
    """Identity of an episode.

    The POISON CONDITION is deliberately NOT part of the identity: the poison
    conditions of contract §9 are alternative HINT contents over one and the
    same episode (same rule, same items, same scripted attempts), which is what
    makes the poison comparison controlled.
    """
    return domain_sha256("FL_V0_EPISODE", {
        "family_id": family_id, "split": split, "rule_id": rule_id,
        "seed": int(seed), "difficulty": int(difficulty),
        "mode": mode, "structure": structure, "structured": bool(structured),
        "reveal": bool(reveal), "arm_tag": arm_tag})


def make_episode_key(family_id: str, split: str, rule_id: str, seed: int,
                     difficulty: int, structure: str, structured: bool,
                     reveal: bool, arm_tag: str = "") -> str:
    """Mode-INDEPENDENT identity used to derive items and attempt draws.

    Two episodes that differ only in attempt-policy mode (FL2's ``successful``
    vs FL3's ``scripted``) therefore contain the SAME latent rule, the SAME
    items and the SAME uniform draws; they differ only in which attempts the
    policy is allowed to get wrong.  That makes the FL2/FL3 data contrast
    controlled instead of confounded with a different item sample.
    """
    return domain_sha256("FL_V0_EPISODE_KEY", {
        "family_id": family_id, "split": split, "rule_id": rule_id,
        "seed": int(seed), "difficulty": int(difficulty),
        "structure": structure, "structured": bool(structured),
        "reveal": bool(reveal), "arm_tag": arm_tag})


@dataclass
class Episode:
    episode_id: str
    family_id: str
    split: str
    rule_id: str
    rule_spec: dict
    seed: int
    difficulty: int
    condition: str
    mode: str
    structure: str
    structured: bool
    reveal: bool
    events: list[Event]
    items: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "family_id": self.family_id,
            "split": self.split,
            "rule_id": self.rule_id,
            "rule_spec": self.rule_spec,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "condition": self.condition,
            "mode": self.mode,
            "structure": self.structure,
            "structured": self.structured,
            "reveal": self.reveal,
            "events": [e.to_dict() for e in self.events],
            "items": self.items,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Episode":
        d = dict(d)
        d["events"] = [Event.from_dict(e) for e in d["events"]]
        return Episode(**d)

    def event_indices(self, *roles: Role) -> list[int]:
        wanted = set(roles)
        return [i for i, e in enumerate(self.events) if e.role in wanted]

    def instance_ids(self) -> list[str]:
        return sorted(self.items)
