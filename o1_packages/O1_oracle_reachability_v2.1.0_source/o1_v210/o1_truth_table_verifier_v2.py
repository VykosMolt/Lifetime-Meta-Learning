"""Independent deterministic verifier for frozen O1 v2 tasks."""
from __future__ import annotations

import itertools


def verified_label(task: dict) -> str:
    sym = task["symbolic"]
    n = int(sym["n_vars"])
    facts = [(int(v), bool(p)) for v, p in sym["facts"]]
    rules = [
        ([(int(v), bool(p)) for v, p in body], (int(h[0]), bool(h[1])))
        for body, h in sym["rules"]
    ]
    qv, qp = int(sym["query"][0]), bool(sym["query"][1])

    def satisfies(a: dict[int, bool]) -> bool:
        if any(a[v] != p for v, p in facts):
            return False
        for body, (hv, hp) in rules:
            if all(a[v] == p for v, p in body) and a[hv] != hp:
                return False
        return True

    models = []
    for bits in itertools.product([False, True], repeat=n):
        a = dict(zip(range(n), bits))
        if satisfies(a):
            models.append(a)
    if not models:
        return "Unknown"
    if all(a[qv] == qp for a in models):
        return "True"
    if all(a[qv] != qp for a in models):
        return "False"
    return "Unknown"


def verify_letter(task: dict, letter: str | None) -> bool:
    if letter is None:
        return False
    idx = ord(letter.upper()) - ord("A")
    return 0 <= idx < len(task["options"]) and task["options"][idx] == verified_label(task)

