"""Adversarial fixtures for the strict v2.1 commitment parser.

Every case that defeated the v2.0 grammar (A/B/C-initial words, dangling
letters inside prose, cross-line letters) is a regression fixture here.  The
accepted grammar is deliberately narrow: a complete answer line only, bare or
angle-bracketed letter, at most one trailing period.

Run:  python test_parser_fixtures.py
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from o1_answer_parser_v2 import has_catastrophic_repetition, parse_final_answer

PASSED = 0
FAILED: list[str] = []


def check(name: str, text, want) -> None:
    global PASSED
    got = parse_final_answer(text)
    if got == want:
        PASSED += 1
    else:
        FAILED.append(f"{name}: parse({text!r}) = {got!r}, want {want!r}")


def check_rep(name: str, toks, want: bool) -> None:
    global PASSED
    got = has_catastrophic_repetition(toks)
    if got == want:
        PASSED += 1
    else:
        FAILED.append(f"{name}: repetition({toks!r}) = {got!r}, want {want!r}")


# ---- accepted complete answer forms -------------------------------------
check("bare_A", "FINAL ANSWER: A", "A")
check("bare_B", "FINAL ANSWER: B", "B")
check("bare_C", "FINAL ANSWER: C", "C")
check("after_reasoning", "Some reasoning first.\nFINAL ANSWER: A", "A")
check("trailing_period", "FINAL ANSWER: A.", "A")
check("trailing_period_newline", "FINAL ANSWER: A.\n", "A")
check("angle_form", "FINAL ANSWER: <B>", "B")
check("angle_form_period", "FINAL ANSWER: <B>.", "B")
check("no_space_after_colon", "FINAL ANSWER:A", "A")
check("space_before_colon", "FINAL ANSWER : C", "C")
check("lowercase_marker_letter", "final answer: b", "B")
check("lowercase_angle", "FINAL ANSWER: <c>", "C")
check("leading_indent", "  FINAL ANSWER: A", "A")
check("trailing_spaces", "FINAL ANSWER: A   ", "A")
check("period_then_spaces", "FINAL ANSWER: A. ", "A")
check("complete_line_then_more_lines", "FINAL ANSWER: A.\nWait, actually no.", "A")

# ---- v2.0 exploit regressions: A/B/C-initial words must NOT parse -------
check("exploit_About", "FINAL ANSWER: About", None)
check("exploit_About_prose", "FINAL ANSWER: About half of the models", None)
check("exploit_Based", "FINAL ANSWER: Based on the rules", None)
check("exploit_Correct", "FINAL ANSWER: Correct", None)
check("exploit_Bbecause", "FINAL ANSWER: Bbecause", None)
check("exploit_Categorical", "FINAL ANSWER: Categorical", None)
check("exploit_lower_about", "final answer: about", None)

# ---- trailing prose on the answer line rejects the line -----------------
check("letter_then_prose", "FINAL ANSWER: B or maybe C", None)
check("letter_then_word", "FINAL ANSWER: A story", None)
check("two_letters", "FINAL ANSWER: A B", None)
check("letter_bang", "FINAL ANSWER: A!", None)
check("double_period", "FINAL ANSWER: A..", None)
check("paren_form_not_protocol", "FINAL ANSWER: (A)", None)
check("bare_then_gt", "FINAL ANSWER: A>", None)
check("truncated_angle", "FINAL ANSWER: <A", None)
check("comma_after_letter", "FINAL ANSWER: A,", None)

# ---- missing/truncated/malformed markers and answers --------------------
check("no_letter", "FINAL ANSWER:", None)
check("no_colon", "FINAL ANSWER", None)
check("no_colon_letter", "FINAL ANSWER A", None)
check("out_of_range_D", "FINAL ANSWER: D", None)
check("glued_marker", "FINALANSWER: A", None)
check("truncated_marker", "FINAL ANSW", None)
check("midline_marker", "the FINAL ANSWER: A story", None)
check("empty", "", None)
check("none_input", None, None)
check("letter_on_next_line", "FINAL ANSWER: \nA", None)
check("only_whitespace_line", "FINAL ANSWER:   \nmore text", None)

# ---- repeated markers: first parseable line wins (sealed stopping rule) --
check("repeat_first_wins", "FINAL ANSWER: A\nFINAL ANSWER: B", "A")
check("invalid_then_valid", "FINAL ANSWER: B or maybe C\nFINAL ANSWER: <C>.", "C")
check("prose_marker_then_valid",
      "I think FINAL ANSWER: B inline\nFINAL ANSWER: B", "B")

# ---- streaming lock-in: no prefix of an exploit string ever parses ------
stream = ""
for piece in ["Reasoning.", "\n", "FINAL", " ANSWER", ":", " About", " half"]:
    stream += piece
    check(f"stream_no_lockin_at_{len(stream)}", stream, None)
stream = ""
seen = None
for piece in ["FINAL", " ANSWER", ":", " A"]:
    stream += piece
    seen = parse_final_answer(stream)
check("stream_clean_commit_locks", stream, "A")

# ---- catastrophic repetition flag is unchanged --------------------------
check_rep("rep_true", list(range(8)) * 4, True)
check_rep("rep_false_short", list(range(8)) * 3, False)
check_rep("rep_false_mixed", list(range(31)) + [7], False)
check_rep("rep_true_prefixed", [99] * 5 + [1, 2, 3, 4, 5, 6, 7, 8] * 4, True)

TOTAL = PASSED + len(FAILED)
if FAILED:
    for f in FAILED:
        print("FAIL ", f)
    print(f"{PASSED}/{TOTAL} passed")
    raise SystemExit(1)
print(f"parser adversarial fixtures: {PASSED}/{TOTAL} passed")
