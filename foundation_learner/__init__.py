"""Foundation Learner B200 V0 — persistent-adaptation pilot package.

Pilot 0 research question: can Ouro-RLTT be trained over complete learning
histories so that it becomes better at LEARNING a previously unseen task
family from attempts and feedback?

Nothing in this package may be described as recursive self-improvement.
See ``docs/IMPLEMENTATION_CONTRACT.md`` (frozen) and
``docs/CONTRACT_AMENDMENTS.md``.
"""
from __future__ import annotations

import os

__all__ = ["VERSION", "STATUS", "PACKAGE_ROOT"]

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


def _read(name: str) -> str:
    with open(os.path.join(PACKAGE_ROOT, name), encoding="utf-8") as fh:
        return fh.read().strip()


VERSION = _read("VERSION")
STATUS = _read("STATUS")
