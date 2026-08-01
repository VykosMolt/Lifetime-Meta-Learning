"""Non-O1 validation corpus: tasks for backend equivalence and benchmarking.

HARD GUARANTEES (mechanically verified, see DISJOINTNESS_REPORT.json and
tests/test_corpus.py):

  * zero task-ID overlap and zero content-hash overlap with the 96-task O1
    calibration manifest and the 2,400-task confirmatory candidate pool;
  * distinct ID namespace (``b200val-``) and distinct generator_setting_id;
  * outcomes generated from this corpus may NEVER be used for O1 scientific
    decisions — records carry ``binding_key = validation_corpus_sha256`` so
    the sealed analysis (which requires the calibration precommit binding)
    refuses them structurally.

The corpus exercises: short/long/mixed prompts; behavior classes producing
valid A/B/C outputs (bare and angle forms), malformed outputs, repeated
markers / prefix-parser attacks, EOS termination, maximum-token termination,
catastrophic repetition, and variable-length stochastic completions.  Class
steering works by salting the prompt until the synthetic runtime's
deterministic behavior function lands on the desired class.
"""
from __future__ import annotations

import json
import os

from . import sealed_import
from .identity import canonical_json, domain_sha256, sha256_file
from .persistence import atomic_write_text

sealed_import.ensure_sealed_path()

from o1_prompt_template_v2 import render_prompt  # noqa: E402 (sealed)

TASK_CONTENT_DOMAIN = "o1b200.validation_task.v1"
GENERATOR_SETTING_ID = "b200_validation_v1"
MASTER_SEED = 314159265358979

# (class, options order, long_prompt) — options order controls whether the
# scripted letter is verifier-correct, so R=0 and R=1 both occur.
_TASK_PLAN = [
    ("COMMIT_A", ["True", "False", "Unknown"], False),
    ("COMMIT_A", ["Unknown", "True", "False"], False),
    ("COMMIT_B_ANGLE", ["True", "False", "Unknown"], False),
    ("COMMIT_C_PERIOD", ["True", "False", "Unknown"], True),
    ("MALFORMED_EOS", ["True", "False", "Unknown"], False),
    ("PREFIX_ATTACK", ["True", "False", "Unknown"], False),
    ("REPEAT_MAX", ["True", "False", "Unknown"], False),
    ("NOSTOP_MAX", ["True", "False", "Unknown"], True),
    ("STOCH_COMMIT", ["True", "False", "Unknown"], False),
    ("STOCH_COMMIT", ["False", "Unknown", "True"], True),
    ("STOCH_EOS", ["True", "False", "Unknown"], False),
    ("STOCH_EOS", ["Unknown", "False", "True"], False),
]

_SYMBOLIC = {
    "n_vars": 2,
    "facts": [[0, True]],
    "rules": [[[[0, True]], [1, True]]],
    "query": [1, True],
}  # verified_label == "True"

_LONG_FILLER = (
    " The next paragraph restates the premises in verbose prose purely to "
    "lengthen the prompt for padding and truncation testing." * 6)


class CorpusError(RuntimeError):
    pass


def _task_content_sha256(task: dict) -> str:
    body = {k: v for k, v in task.items() if k != "task_content_sha256"}
    return domain_sha256(TASK_CONTENT_DOMAIN, body)


def _steer_salt(base_prompt: str, options: list[str], want_class: str,
                task_id: str) -> tuple[str, int]:
    from .synthetic_runtime import SyntheticTokenizer, behavior_of_prompt
    tok = SyntheticTokenizer()
    for salt in range(4096):
        prompt = f"{base_prompt} [probe {salt}]"
        task = {"task_prompt": prompt, "options": options}
        ids = tok.encode_ids(render_prompt(task))
        if behavior_of_prompt(ids, tok)["class"] == want_class:
            return prompt, salt
    raise CorpusError(f"could not steer {task_id} to class {want_class}")


def build_tasks() -> list[dict]:
    tasks = []
    for i, (cls, options, long_prompt) in enumerate(_TASK_PLAN):
        base = (f"Suppose fact f0 holds and rule f0 -> f1 holds. "
                f"Is the query f1 True, False, or Unknown? (case {i})")
        if long_prompt:
            base += _LONG_FILLER
        task_id = f"b200val-{i:03d}-{cls.lower()}"
        prompt, salt = _steer_salt(base, options, cls, task_id)
        task = {
            "task_id": task_id,
            "generator_setting_id": GENERATOR_SETTING_ID,
            "cohort_namespace": "b200_validation",
            "behavior_class": cls,
            "steer_salt": salt,
            "task_prompt": prompt,
            "options": options,
            "symbolic": _SYMBOLIC,
        }
        task["task_content_sha256"] = _task_content_sha256(task)
        tasks.append(task)
    ids = [t["task_id"] for t in tasks]
    hashes = [t["task_content_sha256"] for t in tasks]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise CorpusError("duplicate identity inside the corpus")
    return tasks


def disjointness_report(tasks: list[dict], o1_manifests: list[str]) -> dict:
    """Mechanical zero-overlap verification against the O1 populations."""
    corpus_ids = {t["task_id"] for t in tasks}
    corpus_hashes = {t["task_content_sha256"] for t in tasks}
    report = {"corpus_tasks": len(tasks), "checked_manifests": [], "verdict": None}
    clean = True
    for path in o1_manifests:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip()]
        ids = {r["task_id"] for r in rows}
        hashes = {r["task_content_sha256"] for r in rows}
        id_overlap = sorted(corpus_ids & ids)
        hash_overlap = sorted(corpus_hashes & hashes)
        report["checked_manifests"].append({
            "path": path,
            "sha256": sha256_file(path),
            "n_tasks": len(rows),
            "task_id_overlap": id_overlap,
            "content_hash_overlap": hash_overlap,
        })
        if id_overlap or hash_overlap:
            clean = False
    report["verdict"] = "DISJOINT" if clean else "OVERLAP_FOUND"
    return report


def write_corpus(out_dir: str, o1_manifests: list[str]) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    tasks = build_tasks()
    report = disjointness_report(tasks, o1_manifests)
    if report["verdict"] != "DISJOINT":
        raise CorpusError(f"corpus overlaps O1 populations: {report}")
    tasks_path = os.path.join(out_dir, "corpus_tasks.jsonl")
    atomic_write_text(tasks_path,
                      "".join(canonical_json(t) + "\n" for t in tasks))
    tasks_sha = sha256_file(tasks_path)
    config = {
        "corpus_version": "b200_validation_corpus_v1",
        "purpose": ("backend equivalence / benchmarking ONLY; outcomes may "
                    "never inform any O1 scientific decision"),
        "master_seed": MASTER_SEED,
        "arms": ["baseline", "zero_alpha", "structured", "random"],
        "alpha_structured": 0.04,
        "alpha_random": 0.04,
        "alpha_grid": [0.005, 0.01, 0.02, 0.04, 0.08],
        "tasks_sha256": tasks_sha,
        "n_tasks": len(tasks),
    }
    config["binding_sha256"] = domain_sha256("o1b200.validation_binding.v1",
                                             config)
    atomic_write_text(os.path.join(out_dir, "CORPUS_CONFIG.json"),
                      json.dumps(config, indent=2, sort_keys=True) + "\n")
    atomic_write_text(os.path.join(out_dir, "DISJOINTNESS_REPORT.json"),
                      json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"tasks_path": tasks_path, "tasks_sha256": tasks_sha,
            "config": config, "disjointness": report["verdict"]}


def load_corpus(corpus_dir: str) -> tuple[list[dict], dict]:
    with open(os.path.join(corpus_dir, "CORPUS_CONFIG.json"),
              encoding="utf-8") as fh:
        config = json.load(fh)
    tasks_path = os.path.join(corpus_dir, "corpus_tasks.jsonl")
    got = sha256_file(tasks_path)
    if got != config["tasks_sha256"]:
        raise CorpusError("corpus task file does not match its sealed hash")
    binding = dict(config)
    binding.pop("binding_sha256")
    if domain_sha256("o1b200.validation_binding.v1", binding) != config["binding_sha256"]:
        raise CorpusError("corpus config binding hash mismatch")
    with open(tasks_path, encoding="utf-8") as fh:
        tasks = [json.loads(ln) for ln in fh if ln.strip()]
    for t in tasks:
        if _task_content_sha256(t) != t["task_content_sha256"]:
            raise CorpusError(f"task {t['task_id']} content hash mismatch")
    return tasks, config
