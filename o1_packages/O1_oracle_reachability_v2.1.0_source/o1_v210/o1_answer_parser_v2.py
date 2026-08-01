"""Strict O1 v2 terminal commitment parser.

Grammar v2.1 — the v2.0 pattern accepted any A/B/C-initial word after the
marker ("FINAL ANSWER: Correct" parsed as C), converting degenerate text into
1/3-chance oracle lottery tickets inside a max-of-8 endpoint.  v2.1 accepts
only a complete answer line:

    FINAL ANSWER: <letter-form> [.]  <end-of-line or end-of-text>

where <letter-form> is a bare letter A/B/C or the literal angle-bracket form
<A>/<B>/<C> shown in the prompt template.  At most one trailing period and
horizontal whitespace may follow; any other character on the answer line
rejects the match.  Marker and letter remain case-insensitive exactly as in
v2.0; matching remains first-match, which implements the sealed stopping rule
"stop after the first parseable FINAL ANSWER commitment".

The end-of-text alternative is required for streaming use: the generation loop
parses after every sampled token, so a commitment is recognized the moment the
line is complete even before a newline is emitted.  Residual hazard: a stop
could in principle occur when the decoded text ends exactly at a bare letter
that the model would have continued into a longer word.  With the sealed
tokenizer, " About"/" Based"/" Correct" are single tokens distinct from
" A"/" B"/" C", so the ambiguous prefix state does not arise at a token
boundary; the hazard is documented rather than load-bearing.

The trailing run is an ATOMIC group.  The equivalent backtracking form
``[^\\S\\n]*\\.?[^\\S\\n]*`` has two adjacent quantifiers over the same class and
exhibits quadratic backtracking on a long horizontal-whitespace run followed
by a non-EOL character — a red-team finding: because the parser runs inside
the sealed scoring gates, a single oversized ``generated_text`` could stall
analysis.  The atomic group accepts exactly the same language (optional
spaces, at most one period, optional spaces) with no backtracking, and the
record schema additionally caps ``generated_text`` length.
"""
from __future__ import annotations

import re

FINAL = re.compile(
    r"(?:^|\n)[^\S\n]*FINAL ANSWER[^\S\n]*:[^\S\n]*"
    r"(?:<([A-C])>|([A-C]))"
    r"(?>[^\S\n]*(?:\.[^\S\n]*)?)(?=\n|$)",
    re.I,
)


def parse_final_answer(text: str) -> str | None:
    m = FINAL.search(str(text or ""))
    if m is None:
        return None
    return (m.group(1) or m.group(2)).upper()


def has_catastrophic_repetition(token_ids: list[int]) -> bool:
    """Frozen mechanical flag: final 32 tokens are four repeats of 8 tokens."""
    return len(token_ids) >= 32 and token_ids[-32:-24] * 4 == token_ids[-32:]
