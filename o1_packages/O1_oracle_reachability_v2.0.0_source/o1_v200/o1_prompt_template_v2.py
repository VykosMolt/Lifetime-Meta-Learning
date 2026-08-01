"""Frozen model-visible O1 v2 prompt template."""
from __future__ import annotations


def render_prompt(task: dict) -> str:
    options = task["options"]
    option_text = "\n".join(
        f"{chr(65 + idx)}. {text}" for idx, text in enumerate(options)
    )
    return (
        "Answer the multiple-choice question. Give concise reasoning if useful, "
        "then end with exactly one line: FINAL ANSWER: <letter>.\n\n"
        f"Question:\n{task['task_prompt']}\n\nOptions:\n{option_text}"
    )

