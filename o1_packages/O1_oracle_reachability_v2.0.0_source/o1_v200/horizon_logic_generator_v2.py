#!/usr/bin/env python3
"""Frozen Horizon Logic generator for the O1 v2 calibration/candidate cohorts.

This is a self-contained, byte-bound extraction of the propositional generator
used by branch_training_logic_expansion_v1.  It generates symbolic tasks first,
checks them by exhaustive truth table, and only then renders the model-visible
prompt.  Cohort namespaces are disjoint and no model outcome is an input.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from typing import Any

REVISION = "O1_HORIZON_LOGIC_V2_PROPOSITIONAL_20260729"
OPTIONS = ["True", "False", "Unknown"]
PROPS = [
    "it is raining", "the alarm is on", "the door is locked",
    "the light is green", "the engine starts", "the battery is charged",
    "the gate is open", "the signal is sent", "the tank is full",
    "the valve is closed", "the fan is running", "the pump is active",
]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _eval_kb(assignment: dict[int, bool], facts: list, rules: list) -> bool:
    if any(assignment[v] != pol for v, pol in facts):
        return False
    return all(
        not all(assignment[v] == pol for v, pol in body)
        or assignment[head[0]] == head[1]
        for body, head in rules
    )


def truth_table_label(symbolic: dict[str, Any]) -> str:
    n_vars = int(symbolic["n_vars"])
    facts = [(int(v), bool(p)) for v, p in symbolic["facts"]]
    rules = [
        ([(int(v), bool(p)) for v, p in body], (int(head[0]), bool(head[1])))
        for body, head in symbolic["rules"]
    ]
    qv, qp = int(symbolic["query"][0]), bool(symbolic["query"][1])
    models = [
        dict(zip(range(n_vars), bits))
        for bits in itertools.product([False, True], repeat=n_vars)
        if _eval_kb(dict(zip(range(n_vars), bits)), facts, rules)
    ]
    if not models:
        return "Unknown"
    if all(row[qv] == qp for row in models):
        return "True"
    if all(row[qv] != qp for row in models):
        return "False"
    return "Unknown"


def _candidate(seed: int) -> dict[str, Any] | None:
    rng = random.Random(int(seed))
    n_vars = rng.randint(3, 5)
    props = rng.sample(PROPS, n_vars)
    facts = [(v, rng.random() < 0.7)
             for v in rng.sample(range(n_vars), rng.randint(1, 2))]
    rules = []
    requested_rules = rng.randint(2, 4)
    for _ in range(requested_rules):
        body_vars = rng.sample(range(n_vars), rng.randint(1, 2))
        body = [(v, rng.random() < 0.8) for v in body_vars]
        head_var = rng.randrange(n_vars)
        if head_var in body_vars:
            continue
        rules.append((body, (head_var, rng.random() < 0.85)))
    if not rules:
        return None
    query = (rng.randrange(n_vars), True)
    symbolic = {
        "n_vars": n_vars,
        "facts": facts,
        "rules": rules,
        "query": query,
    }
    label = truth_table_label(symbolic)
    # The historical source rejected inconsistent KBs; Unknown is retained only
    # when at least one satisfying model exists.
    assignments = (
        dict(zip(range(n_vars), bits))
        for bits in itertools.product([False, True], repeat=n_vars)
    )
    if not any(_eval_kb(a, facts, rules) for a in assignments):
        return None

    def lit(v: int, polarity: bool) -> str:
        return props[v] if polarity else f"it is not the case that {props[v]}"

    facts_text = "; ".join(lit(v, p) for v, p in facts)
    rules_text = " ".join(
        f"If {' and '.join(lit(v, p) for v, p in body)}, "
        f"then {lit(head[0], head[1])}."
        for body, head in rules
    )
    task_prompt = (
        f"Facts: {facts_text}.\nRules: {rules_text}\n"
        "Question: Based only on the facts and rules, is the following "
        "statement True, False, or Unknown?\n"
        f"Statement: {props[query[0]]}."
    )
    return {
        "generator_revision": REVISION,
        "category": "synthetic_propositional",
        "generator_setting_id": f"rules_{len(rules)}",
        "proof_depth": len(rules),
        "rule_count": len(rules),
        "task_prompt": task_prompt,
        "options": OPTIONS,
        "answer_key": OPTIONS.index(label),
        "gold_answer": label,
        "verifier": {"type": "exhaustive_truth_table", "n_vars": n_vars},
        "symbolic": symbolic,
    }


def _finalize(task: dict[str, Any], namespace: str, ordinal: int) -> dict[str, Any]:
    """Attach canonical cohort identity to an accepted symbolic task."""
    content = {
        "generator_revision": task["generator_revision"],
        "category": task["category"],
        "generator_setting_id": task["generator_setting_id"],
        "proof_depth": task["proof_depth"],
        "rule_count": task["rule_count"],
        "task_prompt": task["task_prompt"],
        "options": task["options"],
        "symbolic": task["symbolic"],
    }
    task["cohort_namespace"] = namespace
    task["ordinal_within_setting"] = int(ordinal)
    task["task_content_sha256"] = sha256_hex(content)
    task["task_id"] = "o1v2-" + task["task_content_sha256"]
    if truth_table_label(task["symbolic"]) != task["gold_answer"]:
        raise AssertionError("generator/verifier disagreement")
    return task


def generate_tasks(namespace: str, setting_id: str, count: int) -> list[dict[str, Any]]:
    """Return the first ``count`` accepted tasks in one linear deterministic scan."""
    if namespace not in {"calibration", "confirmatory_candidate"}:
        raise ValueError("unsealed cohort namespace")
    if setting_id not in {"rules_2", "rules_3", "rules_4"}:
        raise ValueError("unsealed generator setting")
    target_rules = int(setting_id.rsplit("_", 1)[1])
    accepted: list[dict[str, Any]] = []
    nonce = -1
    while len(accepted) < int(count):
        nonce += 1
        digest = hashlib.sha256(
            f"{REVISION}\0{namespace}\0{setting_id}\0{nonce}".encode("utf-8")
        ).digest()
        task = _candidate(int.from_bytes(digest[:8], "big"))
        if task is None or task["rule_count"] != target_rules:
            continue
        accepted.append(_finalize(task, namespace, len(accepted)))
    return accepted


def generate_task(namespace: str, setting_id: str, ordinal: int) -> dict[str, Any]:
    """Return the ordinal-th accepted task for a sealed namespace/setting."""
    return generate_tasks(namespace, setting_id, int(ordinal) + 1)[-1]
