"""Strict O1 v2 terminal commitment parser."""
from __future__ import annotations

import re

FINAL = re.compile(r"(?:^|\n)\s*FINAL ANSWER\s*:\s*<?([A-C])>?\s*[.\s]*", re.I)


def parse_final_answer(text: str) -> str | None:
    found = FINAL.findall(str(text or ""))
    return found[0].upper() if found else None


def has_catastrophic_repetition(token_ids: list[int]) -> bool:
    """Frozen mechanical flag: final 32 tokens are four repeats of 8 tokens."""
    return len(token_ids) >= 32 and token_ids[-32:-24] * 4 == token_ids[-32:]

