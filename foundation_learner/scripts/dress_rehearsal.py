#!/usr/bin/env python3
"""Offline miniature campaign through the REAL supervisor and scheduler (§18).

This is a DRESS REHEARSAL.  It is not a scientific result and never becomes
one.  What is real here:

* the real :mod:`campaign.session_supervisor` state machine, every transition;
* the real :mod:`campaign.scheduler`, its admission arithmetic, its journal and
  the frozen 1200 s transfer reserve;
* the real ``BENCH`` harness (W2 trainer on a non-evaluation TRAIN shard);
* the real FL0 evaluator, the real trainers for FL1/FL2/FL3, the real
  development grid and selection, and the real ``DEV_DECISIONS_FROZEN.json``;
* the real sealed gate: a genuine single opening of genuinely enciphered
  SEALED_TEST shards, a genuine append-only ledger, and genuinely immutable
  (0444) results;
* the real result verifier and the real deterministic transfer archive.

What is miniature: the tiny nonscientific Ouro model (hidden 64 / 2 layers /
2 ut steps, random init), the ``--tiny`` pools, an explicitly labelled
miniature update ladder, and a stub O1 command.  The FL4–FL8 mechanism rungs
run only when ``foundation_learner.mechanisms.*`` is importable; otherwise the
rehearsal records ``SKIPPED_MECHANISMS`` and says so in the report.

The authorized budget is nominal (3600 s) rather than "seconds-scale" for one
honest reason: the frozen final-transfer reserve of 1200 s is NOT scaled down
for a rehearsal, so any budget below it would refuse every stage and prove
nothing.  The wall-clock cost of the rehearsal itself is well under 10 minutes
on CPU.

Usage::

    python -m foundation_learner.scripts.dress_rehearsal
    python -m foundation_learner.scripts.dress_rehearsal --out /tmp/rehearsal
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:  # allow direct execution from a checkout
    sys.path.insert(0, _PKG_PARENT)

from foundation_learner.campaign import o1_isolation  # noqa: E402
from foundation_learner.campaign.affordability import (  # noqa: E402
    FULL_MODEL_MODE, PEFT_MODE)
from foundation_learner.campaign.scheduler import (  # noqa: E402
    STATE_COMPLETE, Scheduler)
from foundation_learner.campaign.session_supervisor import (  # noqa: E402
    REHEARSAL_LABEL, STATES, SessionConfig, SessionSupervisor)
from foundation_learner.campaign.stage_definitions import (  # noqa: E402
    STAGE_TABLE, StageContext, module_available)

REPORT_SCHEMA = "flb200.rehearsal_report.v1"
REPORT_NAME = "REHEARSAL_REPORT.json"

DEFAULT_OUT = os.path.join(_PKG_PARENT, "foundation_learner", "reports",
                           "local_runs", "dress_rehearsal")
TINY_PREGEN = os.path.join(_PKG_PARENT, "artifacts_fl", "pregen_tiny")

#: miniature, explicitly labelled; the frozen ladder is (600, 1200, 2400, 4800)
REHEARSAL_UPDATE_LADDER = (2,)
REHEARSAL_AUTHORIZED_SECONDS = 3600.0
#: Contract §18 fixes no rehearsal wall-clock number; this is the rehearsal's
#: own ADVISORY bound on itself (Amendment 16).  It was 600 s when the
#: miniature ladder was BENCH, FL0, the grid, three core arms and the sealed
#: opening.  Amendment 12 adds four stages (CORE_MATCHING and the three
#: unconditional §9 diagnostics) and makes every evaluation set
#: family-balanced, which triples the episode walks of the existing stages.
#: Measured breakdown of the added work on an idle 24-core CPU: REMAP_DIAG
#: 218 s, POISON_DIAG 188 s, INTERFERENCE_DIAG 33 s, CORE_MATCHING 0.002 s.
#:
#: It is ADVISORY because it is a property of the HOST, not of the package: the
#: same rehearsal measured 673 s and 813 s on the same machine depending on
#: what else was running, so a hard wall-clock gate turns another process into
#: a failed validation.  The measured time is always reported, and the bound is
#: scaled by a measured single-core reference so a slow or loaded host is
#: judged against itself.  The fifteen SUBSTANTIVE checks stay hard.
REHEARSAL_WALL_CLOCK_LIMIT = 1200.0
#: Seconds the reference micro-benchmark takes on the machine this bound was
#: measured on; the advisory budget is scaled by (measured / reference).
REHEARSAL_HOST_REFERENCE_SECONDS = 0.35
ADVISORY_CHECKS: tuple[str, ...] = ("within_wall_clock_limit",)


def measure_host_speed(repeats: int = 3) -> float:
    """A tiny deterministic CPU benchmark: seconds for a fixed matmul chain.

    Used ONLY to scale the advisory wall-clock budget to the host actually
    running the rehearsal.  No scientific path reads it.
    """
    import torch

    gen = torch.Generator().manual_seed(20260809)
    a = torch.randn(512, 512, generator=gen)
    b = torch.randn(512, 512, generator=gen)
    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.monotonic()
        for _ in range(20):
            a = torch.mm(a, b) / 32.0
        best = min(best, time.monotonic() - t0)
    return float(best)


# --------------------------------------------------------------------------
# fixtures: a fake O1 root, a stub O1 command, a stand-in checkpoint tree
# --------------------------------------------------------------------------

_STUB_O1 = r'''
import hashlib, json, os, sys
root = sys.argv[1]
os.makedirs(root, exist_ok=True)
# The stub stands in for the OPAQUE O1 pod entry.  It writes exactly what the
# supervisor is configured to expect: completion markers and a hash manifest.
records = os.path.join(root, "o1_records.jsonl")
with open(records, "w", encoding="utf-8") as fh:
    fh.write('{"stub": "DRESS_REHEARSAL", "row": 0}\n')
marker = os.path.join(root, "O1_COMPLETE")
with open(marker, "w", encoding="utf-8") as fh:
    fh.write("DRESS_REHEARSAL\n")
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()
manifest = {"schema": "o1b200.transfer_manifest.v1", "artifacts": {
    "records": {"path": "o1_records.jsonl", "kind": "file",
                "sha256": sha(records), "transfer_out_of_band": False},
    "marker": {"path": "O1_COMPLETE", "kind": "file",
               "sha256": sha(marker), "transfer_out_of_band": False}}}
with open(os.path.join(root, "TRANSFER_MANIFEST.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
print("DRESS_REHEARSAL stub O1 entry complete")
'''


def _write_stub_checkpoint(path: str) -> str:
    """A tiny stand-in 'checkpoint tree' whose real tree hash is verified."""
    from foundation_learner.ecology.base import sha256_tree

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
        fh.write('{"model_type": "ouro", "DRESS_REHEARSAL": true}\n')
    with open(os.path.join(path, "README"), "w", encoding="utf-8") as fh:
        fh.write("DRESS_REHEARSAL stand-in; NOT the frozen backbone\n")
    return sha256_tree(path)


def _prepare_pregen(work_dir: str, log) -> str:
    """Reuse ``artifacts_fl/pregen_tiny`` when complete, else generate it."""
    manifest = os.path.join(TINY_PREGEN, "PREGEN_MANIFEST.json")
    if os.path.isfile(manifest):
        log(f"[rehearsal] reusing tiny pregeneration at {TINY_PREGEN}")
        return TINY_PREGEN
    out = os.path.join(work_dir, "pregen_tiny")
    log(f"[rehearsal] generating tiny pools into {out}")
    from foundation_learner.data.generate_shards import generate_all
    from foundation_learner.data.pools import tiny_pools

    generate_all(out, pools=tiny_pools(), log=None)
    return out


# --------------------------------------------------------------------------
# the rehearsal
# --------------------------------------------------------------------------

def build_session_config(*, work_dir: str, o1_root: str, checkpoint_dir: str,
                         checkpoint_tree_sha256: str, pregen_root: str,
                         fl_out: str) -> dict:
    stub = os.path.join(work_dir, "stub_o1_entry.py")
    with open(stub, "w", encoding="utf-8") as fh:
        fh.write(_STUB_O1)
    transfer_dir = os.path.join(work_dir, "o1_transferred")
    return {
        "schema": "flb200.session_config.v1",
        "session_id": "DRESS_REHEARSAL_0001",
        "label": REHEARSAL_LABEL,
        "rehearsal": True,
        "o1_entry_command": [sys.executable, stub, o1_root],
        "o1_completion_markers": [os.path.join(o1_root, "O1_COMPLETE")],
        "o1_hash_manifests": [os.path.join(o1_root, "TRANSFER_MANIFEST.json")],
        "o1_transfer_command": [
            sys.executable, "-c",
            "import os,shutil,sys; shutil.copytree(sys.argv[1], sys.argv[2], "
            "dirs_exist_ok=True); print('DRESS_REHEARSAL o1 transfer done')",
            o1_root, transfer_dir],
        "o1_close_command": [sys.executable, "-c",
                             "print('DRESS_REHEARSAL o1 process closed')"],
        "o1_roots": [o1_root],
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "pregen_root": pregen_root,
        "fl_out_dir": fl_out,
        "session_authorized_seconds": REHEARSAL_AUTHORIZED_SECONDS,
        "terminate_command": [sys.executable, "-c",
                              "print('DRESS_REHEARSAL accelerator terminated')"],
        "o1_entry_command_note": (
            "REAL SESSIONS: this is the operator-bound UNRESOLVED field; the "
            "sealed O1 pod entrypoint is a refusing stub (contract §22)."),
    }


def _context_factory(pregen_root: str, guard, log):
    from foundation_learner.training.tiny_model import build_tiny_model

    def factory(supervisor: SessionSupervisor) -> StageContext:
        def bundle_factory():
            # a FRESH tiny bundle per arm, mirroring the frozen "fresh load of
            # the identical checkpoint per arm" rule
            bundle = build_tiny_model(seed=20260809, device="cpu")
            return bundle

        return StageContext(
            out_dir=os.path.join(supervisor.out_dir, "ladder"),
            pregen_root=pregen_root,
            bundle_factory=bundle_factory,
            guard=guard,
            scope=PEFT_MODE,
            max_tokens_per_batch=2048,
            eval_episode_cap=1,
            train_example_cap=2,
            rehearsal=True,
            label=REHEARSAL_LABEL,
            extra={
                "update_ladder": list(REHEARSAL_UPDATE_LADDER),
                "bench_scopes": [PEFT_MODE, FULL_MODEL_MODE],
                "bench_probe_updates": 2,
                "bench_probe_episodes": 1,
                "bench_declared_budget_seconds": 900.0,
                "run_equivalence_gate": False,
                # frozen at 64 for the campaign; shortened here ONLY to keep
                # the offline mechanics walk under its wall-clock limit
                "max_new_tokens": 8,
                "campaign_hashes": {
                    "pregen_root_basename": os.path.basename(pregen_root),
                    "rehearsal": True,
                },
                "stage_states": {"REHEARSAL": True},
            },
        )

    return factory


def run_rehearsal(out_dir: str, *, keep: bool = True, log=print) -> dict:
    t0 = time.monotonic()
    work_dir = os.path.abspath(out_dir)
    if os.path.exists(work_dir) and not keep:
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    session_dir = os.path.join(work_dir, "session")
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)      # a rehearsal always starts clean
    os.makedirs(session_dir, exist_ok=True)

    o1_root = os.path.join(work_dir, "fake_o1_root")
    os.makedirs(o1_root, exist_ok=True)
    checkpoint_dir = os.path.join(work_dir, "stub_checkpoint")
    tree_sha = _write_stub_checkpoint(checkpoint_dir)
    pregen_root = _prepare_pregen(work_dir, log)

    guard = o1_isolation.IsolationGuard(label="FL_DRESS_REHEARSAL")
    config_payload = build_session_config(
        work_dir=work_dir, o1_root=o1_root, checkpoint_dir=checkpoint_dir,
        checkpoint_tree_sha256=tree_sha, pregen_root=pregen_root,
        fl_out=session_dir)
    config_path = os.path.join(work_dir, "REHEARSAL_SESSION_CONFIG.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config_payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    supervisor = SessionSupervisor(
        config=SessionConfig.load(config_path, guard=guard),
        out_dir=session_dir, guard=guard,
        context_factory=_context_factory(pregen_root, guard, log))
    # the REAL campaign entry: it applies and records the §22 determinism
    # configuration for rehearsals exactly as for a real session, and leaves
    # the rehearsal's explicitly injected context factory alone
    from foundation_learner.campaign import entry as campaign_entry

    determinism = campaign_entry.attach_to_supervisor(supervisor)

    stage_times: dict[str, float] = {}

    def ladder_runner(scheduler: Scheduler, ctx: StageContext) -> dict:
        summary = scheduler.run_ladder(ctx)
        for outcome in scheduler.outcomes:
            stage_times[outcome.stage_id] = round(outcome.seconds, 3)
        return summary

    supervisor.ladder_runner = ladder_runner

    log(f"[rehearsal] running the real supervisor over {len(STATES)} states")
    status = supervisor.run(resume=False)
    elapsed = time.monotonic() - t0

    # advisory wall-clock budget, scaled to THIS host (Amendment 16, N3)
    reference = measure_host_speed()
    host_factor = max(1.0, reference / REHEARSAL_HOST_REFERENCE_SECONDS)
    advisory_budget = REHEARSAL_WALL_CLOCK_LIMIT * host_factor
    host_speed = {
        "reference_seconds": REHEARSAL_HOST_REFERENCE_SECONDS,
        "measured_seconds": round(reference, 4),
        "factor": round(host_factor, 3),
        "note": ("the wall-clock budget is scaled by a measured single-core "
                 "benchmark so a loaded or slow host is judged against itself; "
                 "it is ADVISORY and never fails the rehearsal"),
    }

    ladder_summary = supervisor.state_results.get("RUN_FL_LADDER") or {}
    states_walked = list(status["states_completed"])
    missing_states = [s for s in STATES if s not in states_walked]
    stage_states = ladder_summary.get("states", {})
    mechanisms_present = {
        stage.stage_id: all(module_available(m) for m in stage.requires_modules)
        for stage in STAGE_TABLE if stage.requires_modules}

    checks = {
        "all_supervisor_states_walked": not missing_states,
        "fl_started_only_after_o1_transfer": (
            states_walked.index("TRANSFER_O1_RECORDS")
            < states_walked.index("RUN_FL_LADDER")
            if "RUN_FL_LADDER" in states_walked else False),
        "bench_completed": stage_states.get("BENCH") == STATE_COMPLETE,
        "fl0_completed": stage_states.get("FL0") == STATE_COMPLETE,
        "dev_grid_completed": stage_states.get("DEV_GRID") == STATE_COMPLETE,
        "core_arms_completed": all(
            stage_states.get(a) == STATE_COMPLETE for a in ("FL1", "FL2", "FL3")),
        "sealed_eval_completed": stage_states.get("SEALED_EVAL") == STATE_COMPLETE,
        "dev_decisions_frozen_exists": os.path.isfile(
            os.path.join(session_dir, "ladder", "DEV_DECISIONS_FROZEN.json")),
        "sealed_ledger_exists": os.path.isfile(
            os.path.join(session_dir, "ladder", "SEALED_OPENING_LEDGER.jsonl")),
        "transfer_archive_built": os.path.isfile(
            os.path.join(session_dir, "FL_ARTIFACTS.zip")),
        "result_manifest_verified": bool(
            (supervisor.state_results.get("CHECKPOINT_AND_VERIFY") or {})
            .get("verification", {}).get("ok")),
        "within_wall_clock_limit": elapsed <= advisory_budget,
        "outcome_complete": status["outcome"] == "COMPLETE",
        "determinism_configured": bool(
            determinism.get("deterministic_algorithms")),
        "core_matching_completed":
            stage_states.get("CORE_MATCHING") == STATE_COMPLETE,
        "diagnostics_completed": all(
            stage_states.get(sid) == STATE_COMPLETE
            for sid in ("REMAP_DIAG", "INTERFERENCE_DIAG", "POISON_DIAG")),
    }
    mechanism_states = {sid: stage_states.get(sid)
                        for sid in sorted(mechanisms_present)}
    importable = all(mechanisms_present.values()) and bool(mechanisms_present)
    ran = [sid for sid, state in mechanism_states.items()
           if state == STATE_COMPLETE]
    gated = [sid for sid, state in mechanism_states.items()
             if state == "SKIPPED_ENTRY_CONDITION"]
    mechanism_note = (
        f"mechanisms/ importable={importable}; rungs completed={ran}; "
        f"rungs skipped by a PROMOTION entry condition={gated}. "
        "A rung skipped by its entry condition is a promotion decision, not a "
        "missing module: on the tiny randomly initialised model the FL3 "
        "extension gate (DEV macro-AULC >= FL1 + 0.02 with a positive slope) "
        "cannot pass, which is the correct outcome. The SKIPPED_MECHANISMS "
        "branch (module absent) is covered by the unit suite.")

    report = {
        "schema": REPORT_SCHEMA,
        "label": REHEARSAL_LABEL,
        "scientific_result": None,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 3),
        "wall_clock_limit_seconds": REHEARSAL_WALL_CLOCK_LIMIT,
        "wall_clock_advisory_budget_seconds": round(advisory_budget, 1),
        "host_speed": host_speed,
        "advisory_checks": list(ADVISORY_CHECKS),
        "out_dir": session_dir,
        "supervisor": {
            "outcome": status["outcome"],
            "states_expected": list(STATES),
            "states_walked": states_walked,
            "states_missing": missing_states,
            "failed_state": status["failed_state"],
            "failure": status["failure"],
        },
        "ladder": {
            "states": stage_states,
            "core_plan": ladder_summary.get("core_plan"),
            "stage_seconds": stage_times,
            "journal_records": ladder_summary.get("journal_records"),
        },
        "determinism": determinism,
        "mechanisms_available": mechanisms_present,
        "mechanism_states": mechanism_states,
        "mechanism_note": mechanism_note,
        "checks": checks,
        "verdict": ("PASS" if all(v for k, v in checks.items()
                                  if k not in ADVISORY_CHECKS) else "FAIL"),
        "advisory_warnings": [k for k in ADVISORY_CHECKS if not checks.get(k, True)],
        "miniature": {
            "update_ladder": list(REHEARSAL_UPDATE_LADDER),
            "frozen_ladder": [600, 1200, 2400, 4800],
            "model": "TINY_NONSCIENTIFIC (hidden 64 / 2 layers / 2 ut steps)",
            "max_new_tokens": 8,
            "frozen_max_new_tokens": 64,
            "note": ("every step count here is a REHEARSAL override, stamped "
                     "REHEARSAL_LADDER_OVERRIDE in the core plan and "
                     "STAGE_SMOKE in every arm configuration"),
        },
        "scope_note": ("Offline mechanics validation only: tiny model, tiny "
                       "pools, stub O1 command, CPU. No B200 hardware claim "
                       "and no scientific claim is made or implied."),
    }
    report_path = os.path.join(work_dir, REPORT_NAME)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    report["report_path"] = report_path
    log(f"[rehearsal] {report['verdict']} in {elapsed:.1f}s -> {report_path}")
    for name, ok in sorted(checks.items()):
        if ok:
            continue
        if name in ADVISORY_CHECKS:
            log(f"[rehearsal]   ADVISORY (not a failure): {name} "
                f"({elapsed:.0f}s > {advisory_budget:.0f}s on this host)")
        else:
            log(f"[rehearsal]   FAILED CHECK: {name}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline miniature Foundation Learner campaign (§18)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"rehearsal working directory (default {DEFAULT_OUT})")
    parser.add_argument("--fresh", action="store_true",
                        help="delete the working directory first")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print
    report = run_rehearsal(args.out, keep=not args.fresh, log=log)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
