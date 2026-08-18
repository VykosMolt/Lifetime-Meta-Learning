#!/usr/bin/env python3
"""LOCAL_NONSCIENTIFIC_OURO_SMOKE_TEST (contract §18).

Loads the REAL frozen 2.6 B checkpoint on the LOCAL GPU and proves that the
campaign's own code paths speak to it: load + tree-hash verify, one forward,
one greedy generation through ``evaluation/generation.py`` on a NON-EVALUATION
synthetic prompt, and one ``PEFT_MODE`` optimizer step through the real W2
trainer on a throwaway shard, with finiteness assertions and a parse
round-trip.

**This is not a training arm and not a scientific result.**  The prompt is a
throwaway TRAIN-family instance drawn from a SCRATCH seed that is disjoint from
every campaign seed (``SCRATCH_SEED``, tagged ``NONSCIENTIFIC_SMOKE``), so
nothing it touches can ever enter an evaluation.  ``SMOKE_REPORT.json`` is
marked ``NONSCIENTIFIC``.

A hard 15-minute wall clock is enforced two ways: every step checks the
deadline, and an independent daemon watchdog aborts the process (writing a
TIMEOUT report first) if a single operation blocks past it.

``--tiny`` runs the identical sequence on the tiny nonscientific model with no
GPU and no checkpoint weights; that mode exists so the unit suite can exercise
these steps, and it says so in its report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

SMOKE_LABEL = "LOCAL_NONSCIENTIFIC_OURO_SMOKE_TEST"
REPORT_SCHEMA = "flb200.smoke_report.v1"
REPORT_NAME = "SMOKE_REPORT.json"
TIME_BUDGET_SECONDS = 15 * 60

#: A scratch seed that is NOT the campaign root seed 20260809, NOT the second
#: predeclared seed 20260810, and never used by the pre-generation.  Everything
#: this script builds is disjoint from every campaign artefact.
SCRATCH_SEED = 990001
SCRATCH_TAG = "NONSCIENTIFIC_SMOKE"

DEFAULT_OUT = os.path.join(_PKG_PARENT, "foundation_learner", "reports")


class SmokeTimeout(RuntimeError):
    """The 15-minute budget was exceeded."""


class Watchdog:
    """Deadline enforcement: cooperative checks plus a hard daemon abort."""

    def __init__(self, budget_seconds: float, report_path: str,
                 label: str = SMOKE_LABEL):
        self.budget = float(budget_seconds)
        self.report_path = report_path
        self.label = label
        self.t0 = time.monotonic()
        self.timings: list[dict] = []
        self._timer: threading.Timer | None = None

    def arm(self) -> None:
        self._timer = threading.Timer(self.budget, self._hard_abort)
        self._timer.daemon = True
        self._timer.start()

    def disarm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def check(self, step: str) -> float:
        used = self.elapsed()
        self.timings.append({"step": step, "elapsed_seconds": round(used, 3)})
        if used > self.budget:
            raise SmokeTimeout(
                f"{self.label}: {used:.0f}s exceeds the {self.budget:.0f}s budget "
                f"at step {step!r}")
        return used

    def _hard_abort(self) -> None:  # pragma: no cover - timing path
        payload = {
            "schema": REPORT_SCHEMA, "label": self.label,
            "classification": "NONSCIENTIFIC",
            "verdict": "TIMEOUT",
            "elapsed_seconds": round(self.elapsed(), 3),
            "budget_seconds": self.budget,
            "timings": self.timings,
            "note": "watchdog aborted the process; no result is claimed",
        }
        try:
            with open(self.report_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
        finally:
            os._exit(124)


# --------------------------------------------------------------------------
# throwaway (non-evaluation) material
# --------------------------------------------------------------------------

def scratch_instance(*, seed: int = SCRATCH_SEED):
    """A throwaway TRAIN-family instance from a scratch seed.

    TRAIN family + scratch seed = an item that cannot appear in any DEVELOPMENT
    or SEALED_TEST evaluation, and whose rule is not one of the pre-generated
    campaign rules.
    """
    from foundation_learner.ecology.base import derive_seed, make_rng
    from foundation_learner.ecology.families import get_family
    from foundation_learner.ecology.split import compute_split

    family_id = compute_split()["TRAIN"][0]
    family = get_family(family_id)
    rule = family.sample_rule(make_rng(derive_seed(seed, SCRATCH_TAG, "rule")))
    rng = make_rng(derive_seed(seed, SCRATCH_TAG, "instance"))
    instance = family.sample_instance(rule, rng, 1)
    return family, rule, instance


def scratch_episode(*, seed: int = SCRATCH_SEED):
    """A throwaway episode used as the one-step training shard."""
    from foundation_learner.ecology.base import derive_seed
    from foundation_learner.episodes.assemble import assemble_episode

    family, rule, _ = scratch_instance(seed=seed)
    episode = assemble_episode(
        family, rule, split="TRAIN", difficulty=1, root_seed=seed,
        episode_seed=derive_seed(seed, SCRATCH_TAG, "episode"),
        mode="scripted", condition="clean", structured=True, reveal=False)
    return family, rule, episode


# --------------------------------------------------------------------------
# the smoke sequence
# --------------------------------------------------------------------------

def run_smoke(*, tiny: bool, out_dir: str, device: str | None = None,
              budget_seconds: float = TIME_BUDGET_SECONDS,
              checkpoint_dir: str | None = None, log=print) -> dict:
    import torch

    from foundation_learner.episodes.parse import format_answer, parse_answer
    from foundation_learner.training import model_loading as ml

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, REPORT_NAME)
    watchdog = Watchdog(budget_seconds, report_path)
    watchdog.arm()
    checks: dict[str, object] = {}
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "label": SMOKE_LABEL,
        "classification": "NONSCIENTIFIC",
        "scientific_result": None,
        "mode": "TINY_SELF_TEST" if tiny else "REAL_CHECKPOINT",
        "budget_seconds": budget_seconds,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scratch_seed": SCRATCH_SEED,
        "scratch_tag": SCRATCH_TAG,
    }
    try:
        # 1) the environment binding is not negotiable
        version = ml.assert_transformers_version()
        checks["transformers_exactly_4_54_1"] = version == "4.54.1"
        watchdog.check("assert_transformers_version")

        # 2) load
        if tiny:
            from foundation_learner.training.tiny_model import build_tiny_model

            device = device or "cpu"
            bundle = build_tiny_model(seed=SCRATCH_SEED, device=device)
            checks["tree_hash_verified"] = None
        else:
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "cpu":
                raise RuntimeError(
                    "the real-checkpoint smoke test requires a local GPU; "
                    "refusing to claim a GPU result from a CPU run")
            bundle = ml.load_frozen_backbone(
                checkpoint_dir or ml.FROZEN_CHECKPOINT_DIR, device=device,
                verify_tree_hash=True)
            checks["tree_hash_verified"] = True
        ml.set_evaluation_mode(bundle.model)
        report["device"] = device
        report["identity"] = dict(bundle.identity)
        log(f"[{SMOKE_LABEL}] loaded ({report['mode']}) on {device}")
        watchdog.check("load")

        # 3) one forward
        family, rule, instance = scratch_instance()
        report["scratch_family_id"] = family.family_id
        prompt = f"TASK: {instance.prompt_text}\nATTEMPT: "
        encoded = bundle.tokenizer(prompt, return_tensors="pt",
                                   add_special_tokens=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            out = bundle.model(**encoded)
        logits = out.logits
        checks["forward_logits_finite"] = bool(torch.isfinite(logits).all().item())
        checks["forward_shape_ok"] = list(logits.shape[:2]) == \
            list(encoded["input_ids"].shape)
        report["forward_tokens"] = int(encoded["input_ids"].shape[1])
        watchdog.check("forward")

        # 4) one greedy generation through the real evaluation path
        from foundation_learner.evaluation.generation import (GenerationConfig,
                                                              greedy_generate)

        texts = greedy_generate(bundle, [prompt], 64,
                                config=GenerationConfig(max_new_tokens=64,
                                                        batch_size=1))
        report["generation_preview"] = texts[0][:200]
        checks["generation_returned_text"] = isinstance(texts[0], str)
        watchdog.check("generation")

        # 5) parse round-trip (the §4 grammar, not the model's competence)
        rendered = format_answer(instance.answer_canonical)
        checks["parse_round_trip"] = parse_answer(rendered) == \
            instance.answer_canonical
        verdict = family.verify(rule, instance, rendered)
        checks["verifier_accepts_the_true_answer"] = bool(verdict.correct)
        watchdog.check("parse_round_trip")

        # 6) one PEFT_MODE optimizer step on a throwaway shard
        from foundation_learner.training.arms import STAGE_SMOKE, make_arm_config
        from foundation_learner.training.tokenization import iter_examples
        from foundation_learner.training.trainer import run_training_arm

        _, _, episode = scratch_episode()
        examples = list(iter_examples([episode], bundle.tokenizer, "FL3"))
        cfg = make_arm_config("FL3", learning_rate=1e-4, updates=1,
                              max_tokens_per_batch=2048, peft_mode="PEFT_MODE",
                              seed=SCRATCH_SEED, stage=STAGE_SMOKE)
        result = run_training_arm(cfg, bundle, examples,
                                  os.path.join(out_dir, "smoke_arm"),
                                  allow_retry=False)
        loss = result.final_loss
        grad_norm = result.grad_norm_history[-1] if result.grad_norm_history else None
        checks["loss_finite"] = bool(loss is not None and math.isfinite(loss))
        checks["grad_norm_finite"] = bool(
            grad_norm is not None and math.isfinite(grad_norm))
        checks["optimizer_step_taken"] = int(result.ledger.get(
            "optimizer_updates", 0)) == 1
        report["training"] = {"final_loss": loss, "grad_norm": grad_norm,
                              "steps": result.steps,
                              "ledger": result.ledger}
        watchdog.check("peft_optimizer_step")

        if device.startswith("cuda"):
            report["memory"] = {
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        report["verdict"] = "PASS" if all(
            v is not False for v in checks.values()) else "FAIL"
    except BaseException as exc:  # noqa: BLE001 - recorded, never swallowed
        report["verdict"] = "FAIL"
        report["error"] = repr(exc)
        raise
    finally:
        watchdog.disarm()
        report["checks"] = checks
        report["timings"] = watchdog.timings
        report["elapsed_seconds"] = round(watchdog.elapsed(), 3)
        report["scope_note"] = (
            "NONSCIENTIFIC: a mechanics check on a throwaway TRAIN-family "
            "instance from a scratch seed. It is not a training arm, produces "
            "no scientific result, and says nothing about the research "
            "question.")
        tmp = report_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, report_path)
        log(f"[{SMOKE_LABEL}] report -> {report_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=SMOKE_LABEL)
    parser.add_argument("--tiny", action="store_true",
                        help="tiny nonscientific model, no GPU, no weights "
                             "(self-test of these steps)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--budget-seconds", type=float,
                        default=TIME_BUDGET_SECONDS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out
    if args.tiny:
        out = os.path.join(out, "local_runs", "smoke_tiny")
    report = run_smoke(tiny=args.tiny, out_dir=out, device=args.device,
                       budget_seconds=args.budget_seconds,
                       checkpoint_dir=args.checkpoint)
    print(json.dumps({"verdict": report["verdict"], "checks": report["checks"],
                      "elapsed_seconds": report["elapsed_seconds"]},
                     indent=2, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
