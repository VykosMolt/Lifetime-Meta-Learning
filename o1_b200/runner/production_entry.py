#!/usr/bin/env python3
"""Production on-pod entry: the REAL zero-touch session composition.

This replaces the historical refusing stub: it composes the sealed state
machine with REAL components — hardware gate, real Ouro-RLTT artifact,
real non-O1 equivalence and benchmark, frozen backend selection, the
replacement precommit, external (off-pod) commit verification, and the
sealed v2.1 calibration ORCHESTRATOR (never a re-implementation) — with
continuous off-pod durability so eviction of the interruptible pod loses
at most the un-mirrored tail of work.

Every scientific decision stays where it always was: sealed package
semantics, frozen selection policy, frozen benchmark order, frozen budget
policy.  This module only wires validated parts together on the pod.

Configuration arrives via the frozen env-name contract (pod_request):

  O1_B200_OUT                 output dir (default /outputs)
  O1_B200_ARTIFACT_SOURCE     artifact staging source (verify_artifacts)
  O1_B200_RESULT_DESTINATION  durable store destination (hf://… or path)
  O1_ACQUIRED_PROFILE         B300 | B200 (the profile actually rented)
  O1_SESSION_AUTHORIZED_SECONDS   hard runtime limit from the live quote
  O1_HOURLY_RATE_USD          accepted all-in hourly rate (report only)

Fails closed on any absent/unresolved required value.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time

from .budget import BudgetWatchdog, affordability_gate
from .compare_o1_backends import compare_rows, run_backend
from .durability import CheckpointDurability, store_for_destination
from .env_report import collect_pod_report, validate_b200_report
from .persistence import atomic_write_text
from .precommit_template import finalize, load_template, resolve
from .provider_adapter import LocalProviderAdapter
from .runbuild import O1_MANIFEST_PATHS
from .selection import select_backend
from .state_machine import ZeroTouchStateMachine
from .validation_corpus import disjointness_report, load_corpus
from . import sealed_import

ROOT = sealed_import.WORKTREE_ROOT
CHECKPOINT_DIR = os.environ.get("O1_CHECKPOINT_DIR",
                                "/artifacts/ouro_rltt_local")
O1_ROWS_TOTAL = 4608
RECORDS_SYNC_EVERY_ROWS = 25
RECORDS_SYNC_POLL_SECONDS = 15.0


class ProductionEntryError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v or v.startswith("UNRESOLVED"):
        raise ProductionEntryError(
            f"required session env {name} is absent/unresolved; the "
            f"authorized launcher must inject it")
    return v


def build_production_handlers(out_dir: str, provider: LocalProviderAdapter,
                              clock) -> dict:
    profile_key = _require_env("O1_ACQUIRED_PROFILE")
    authorized_seconds = int(float(_require_env(
        "O1_SESSION_AUTHORIZED_SECONDS")))
    result_destination = _require_env("O1_B200_RESULT_DESTINATION")
    corpus_dir = os.path.join(ROOT, "o1_b200", "corpus")
    artifact = {"kind": "ouro_rltt", "checkpoint": CHECKPOINT_DIR,
                "device": "cuda"}
    store = store_for_destination(result_destination)
    records_mirror = CheckpointDurability(
        store, remote_prefix="durable_o1_records")

    def precheck(ctx):
        ctx["instance_ref"] = provider.start_instance()
        ctx["hourly_rate"] = float(os.environ.get("O1_HOURLY_RATE_USD", "0"))
        ctx["runtime_limit"] = authorized_seconds
        ctx["profile"] = profile_key
        return {"instance": ctx["instance_ref"], "profile": profile_key,
                "runtime_limit_seconds": authorized_seconds}

    def artifact_verify(ctx):
        # O1_B200_ARTIFACT_SOURCE is deployment identity (it is part of the
        # authorization hash), so it must actually be CONSUMED — a declared
        # identity field with no reader reads as protection that is not
        # there.  When the operator stages artifacts out of band it names
        # the manifest that describes them; otherwise the image-baked
        # manifest is used.
        staged = os.environ.get("O1_B200_ARTIFACT_SOURCE", "").strip()
        # the POD manifest: container paths, and the only one whose
        # entries can resolve inside the image
        baked = os.path.join(ROOT, "o1_b200", "deploy",
                             "POD_TRANSFER_MANIFEST.json")
        default_manifest = baked
        if staged:
            candidate = (staged if staged.endswith(".json")
                         else os.path.join(staged, "TRANSFER_MANIFEST.json"))
            if not os.path.exists(candidate):
                raise ProductionEntryError(
                    f"O1_B200_ARTIFACT_SOURCE={staged!r} names no readable "
                    f"transfer manifest ({candidate}); refusing to fall "
                    f"back silently to the image-baked manifest")
            default_manifest = candidate
        manifest = os.environ.get("O1_B200_TRANSFER_MANIFEST",
                                  default_manifest)
        ctx["artifact_manifest"] = manifest
        proc = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "o1_b200", "deploy", "verify_artifacts.py"),
             "--manifest", manifest],
            capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise ProductionEntryError(
                f"artifact verification failed:\n{proc.stdout[-1500:]}"
                f"{proc.stderr[-1500:]}")
        tasks, config = load_corpus(corpus_dir)
        rep = disjointness_report(tasks, O1_MANIFEST_PATHS)
        if rep["verdict"] != "DISJOINT":
            raise ProductionEntryError("corpus not disjoint from O1 pools")
        ctx["corpus_config"] = config
        # load_corpus verifies the corpus against its sealed hash and the
        # disjointness report is DISJOINT, so the scientific inputs are
        # provably the frozen ones (fed to the selection gates)
        ctx["corpus_config_verified"] = True
        return {"artifacts_verified": True, "disjointness": rep["verdict"]}

    def environment_verify(ctx):
        from o1_b200.deploy.hardware_gate import (
            exercise_workloads, gather_facts, validate_facts,
        )
        gate = validate_facts(profile_key, gather_facts())
        if not gate["accepted"]:
            raise ProductionEntryError(
                "hardware gate refused this machine: "
                + "; ".join(gate["problems"]))
        gate["workload_exercises"] = exercise_workloads(CHECKPOINT_DIR,
                                                        out_dir)
        if not gate["workload_exercises"]["accepted"]:
            bad = {k: v for k, v in
                   gate["workload_exercises"]["workloads"].items()
                   if not v.get("ok")}
            raise ProductionEntryError(
                f"representative workload exercises failed: {sorted(bad)}")
        atomic_write_text(
            os.path.join(out_dir, "HARDWARE_GATE_REPORT.json"),
            json.dumps(gate, indent=2, sort_keys=True, default=str) + "\n")
        env_report = collect_pod_report(
            container_image_digest=os.environ.get("O1_IMAGE_DIGEST",
                                                  "UNKNOWN"))
        validated = validate_b200_report(env_report)
        atomic_write_text(
            os.path.join(out_dir, "ENVIRONMENT_REPORT.resolved.json"),
            json.dumps({"report": env_report, "validation": validated},
                       indent=2, sort_keys=True, default=str) + "\n")
        ctx["environment_report"] = validated
        ctx["environment_report_raw"] = env_report
        ctx["hardware_gate"] = gate
        return {"hardware_gate": "ACCEPTED", "profile": profile_key}

    def non_o1_equivalence(ctx):
        """Measure structural equivalence PER BENCHMARK CONFIGURATION.

        Selection ranks by throughput, so it prefers the deepest batch that
        ran — exactly the configuration whose batched numerics are least
        like the serial reference.  Borrowing one backend's verdict for
        every batch size of that backend would let a configuration be
        selected on an equivalence result never measured for it, so each
        config_id carries its own verdict and any config without one is
        ineligible (fail closed in _derive_gates).
        """
        from .benchmark_o1_b200 import load_benchmark_order
        stages = load_benchmark_order()["staged_candidates"]
        ref = run_backend("REFERENCE_SERIAL", corpus_dir,
                          os.path.join(out_dir, "eq_ref"), artifact)
        comp = {"REFERENCE_SERIAL_w1_b1": {
            "eligible_structurally": True, "scientific_core_identical": True,
            "is_reference": True}}
        clean_so_far = True
        for entry in stages:
            cid = entry["config_id"]
            if cid in comp:
                continue
            conditional = bool(entry.get("conditional"))
            if conditional and not clean_so_far:
                # frozen stop rule: once a stage is not clean, larger
                # configurations are not attempted
                comp[cid] = {"eligible_structurally": False,
                             "skipped": "prior stage not clean"}
                continue
            try:
                cand = run_backend(
                    entry["backend"], corpus_dir,
                    os.path.join(out_dir, f"eq_{cid}"), artifact,
                    worker_count=int(entry.get("workers", 1)),
                    batch_size=int(entry.get("batch", 1)))
            except MemoryError as exc:
                clean_so_far = False
                if not conditional:
                    raise
                # a capacity-conditional stage that OOMs is SKIPPED (and
                # therefore ineligible), exactly as in the benchmark — it
                # must not abort the whole paid session
                comp[cid] = {"eligible_structurally": False,
                             "skipped": f"OOM: {exc!r}"[:200]}
                continue
            except Exception as exc:  # noqa: BLE001
                clean_so_far = False
                if not conditional:
                    raise
                comp[cid] = {"eligible_structurally": False,
                             "skipped": f"integrity failure: {exc!r}"[:200]}
                continue
            comp[cid] = compare_rows(ref["rows"], cand["rows"])
            if not comp[cid].get("eligible_structurally"):
                clean_so_far = False
        mandatory = [e["config_id"] for e in stages
                     if not e.get("conditional")]
        failed = [cid for cid in mandatory
                  if not comp.get(cid, {}).get("eligible_structurally")]
        if failed:
            raise ProductionEntryError(
                f"structural equivalence gate failed for {failed}")
        ctx["equivalence"] = comp
        # the reference run defines the expected row count every benchmark
        # configuration must reproduce exactly
        ctx["corpus_row_count"] = len(ref["rows"])
        atomic_write_text(
            os.path.join(out_dir, "EQUIVALENCE_REPORT.real.json"),
            json.dumps(comp, indent=2, sort_keys=True, default=str) + "\n")
        return {cid: v.get("eligible_structurally")
                for cid, v in sorted(comp.items())}

    def non_o1_benchmark(ctx):
        from .benchmark_o1_b200 import run_benchmarks
        rep = run_benchmarks(
            corpus_dir, os.path.join(out_dir, "benchmark"),
            mode="real-hardware", artifact=artifact,
            remaining_authorized_seconds=ctx["runtime_limit"] - clock())
        ctx["benchmark_raw"] = [r for r in rep["results"]
                                if not r.get("skipped")]
        atomic_write_text(
            os.path.join(out_dir, "BENCHMARK_REPORT.real.json"),
            json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n")
        return {"benchmarked": len(ctx["benchmark_raw"])}

    def _derive_gates(entry: dict, ctx) -> dict:
        """Mechanical gate derivation from real per-config measurements.

        Every gate is derived from a measurement OF THIS CONFIGURATION.  A
        config with no equivalence verdict of its own is ineligible; the
        row-count and environment gates are checked against the corpus size
        and the recorded environment rather than asserted.
        """
        import torch
        cid = entry.get("config_id")
        eq = (ctx.get("equivalence") or {}).get(cid)
        measured = eq is not None
        structural = bool(measured and eq.get("eligible_structurally"))
        core_identical = bool(measured and (eq.get("is_reference")
                                            or eq.get("scientific_core_identical")))
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = (entry.get("gpu") or {}).get("hbm_reserved_bytes", 0)
        free_frac = 1.0 - (reserved / total if total else 1.0)
        expected_rows = int(ctx.get("corpus_row_count") or 0)
        rows_ok = (expected_rows > 0
                   and int(entry.get("n_rows", -1)) == expected_rows)
        spread = entry.get("throughput_stability_spread")
        env = ctx.get("environment_report_raw") or {}
        no_unvalidated_opt = (env.get("compile_state") in ("OFF", "off", False)
                              and env.get("cuda_graph_state") in
                              ("OFF", "off", False)
                              and env.get("attention_backend") == "eager")
        return {
            **entry,
            "equivalence_measured_for_this_config": measured,
            "completed_rows_per_hour":
                float(entry.get("completed_rows_per_second", 0.0)) * 3600.0,
            "peak_hbm_reserved_bytes": reserved,
            "steady_state_free_hbm_fraction": free_frac,
            "structural_pass": structural,
            "parser_verifier_pass": (measured
                                     and entry.get("integrity_failures", 1) == 0),
            "action_seed_mapping_exact": core_identical,
            "intervention_pass": core_identical,
            "transport_pass": core_identical,
            "resume_pass": entry.get(
                "resume_remaining_after_completion", -1) == 0,
            "no_missing_or_duplicate_rows": rows_ok,
            "no_oom": entry.get("oom_count", 1) == 0,
            "free_hbm_fraction_ok": free_frac >= 0.15,
            "no_unvalidated_optimization": no_unvalidated_opt,
            "throughput_stable": (spread is not None and spread <= 0.25),
            "scientific_config_unchanged": bool(
                ctx.get("corpus_config_verified")),
        }

    def backend_select(ctx):
        candidates = [_derive_gates(e, ctx) for e in ctx["benchmark_raw"]
                      if "config_id" in e and not e.get("oom_count")
                      and not e.get("integrity_failures")
                      and not e.get("oom")
                      and not e.get("integrity_failure")]
        ctx["benchmark"] = candidates
        out = select_backend(candidates)
        ctx["selected_backend"] = out["selected"]
        atomic_write_text(
            os.path.join(out_dir, "BACKEND_SELECTION.json"),
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
        return {"selected": out["selected"]["config_id"]}

    def precommit_build(ctx):
        import torch
        template = load_template()
        gate_sha = __import__("hashlib").sha256(json.dumps(
            ctx["hardware_gate"], sort_keys=True, default=str
        ).encode()).hexdigest()
        resolved = resolve(template, {
            "provider": "runpod",
            "instance_id": os.environ.get("RUNPOD_POD_ID", "POD"),
            "gpu_uuid": torch.cuda.get_device_properties(0).uuid.__str__(),
            "driver_runtime_report_sha256": gate_sha,
            "container_image_digest_as_deployed": os.environ.get(
                "O1_IMAGE_DIGEST", "UNKNOWN"),
            "selected_eligible_backend": ctx["selected_backend"]["backend"],
            "selected_worker_batch_configuration":
                ctx["selected_backend"]["config_id"],
            "measured_benchmark_throughput_rows_per_hour":
                ctx["selected_backend"]["completed_rows_per_hour"],
            "actual_hourly_rate_usd": ctx["hourly_rate"],
            "computed_hard_runtime_seconds": ctx["runtime_limit"],
            "environment_digest_sha256": ctx["environment_report"].get(
                "report_sha256", "0" * 64),
            "final_backend_benchmark_report_sha256": gate_sha,
        }, mock=False)
        ctx["finalized_precommit"] = finalize(resolved)
        path = os.path.join(out_dir, "CALIBRATION_PRECOMMIT.deployed.json")
        atomic_write_text(path, json.dumps(
            ctx["finalized_precommit"], indent=2, sort_keys=True) + "\n")
        ctx["precommit_path"] = path
        ctx["calibration_binding_sha256"] = \
            ctx["finalized_precommit"]["document_sha256"]
        return {"finalized": True,
                "document_sha256": ctx["calibration_binding_sha256"]}

    def external_commit_verify(ctx):
        # Off-pod commitment: push the finalized hardware precommit, re-fetch
        # it independently, and verify byte identity.  The pod deliberately
        # has no git credentials; the durable store is the external witness.
        #
        # The key is per-acquisition: this document records THIS pod's
        # hardware facts, and a fixed key would silently replace an earlier
        # pod's commitment after partial results exist, destroying the
        # chain of pre-registrations across an evicted session.
        from .identity import sha256_file
        digest = sha256_file(ctx["precommit_path"])
        key = f"commitments/hardware/{digest[:16]}.json"
        existing = set(store.list_prefix("commitments/hardware"))
        pushed = store.push_file(ctx["precommit_path"], key)
        back = os.path.join(out_dir, "precommit_refetch.json")
        store.fetch_file(key, back)
        if sha256_file(back) != digest:
            raise ProductionEntryError(
                "external commitment verification failed: re-fetched "
                "precommit differs")
        os.remove(back)
        return {"remote_verified": True, "sha256": pushed["sha256"],
                "key": key, "prior_commitments": len(existing)}

    def affordability(ctx):
        # The sealed v2.1 orchestrator takes NO backend/worker/batch
        # parameter — it builds its own single-row serial backend. So the
        # calibration runs at the REFERENCE_SERIAL rate no matter which
        # configuration the frozen policy selected, and projecting from the
        # selected (fastest, batched) rate would under-estimate the time by
        # the whole batching speed-up and pass a gate that cannot hold.
        eq_ref = "REFERENCE_SERIAL_w1_b1"
        serial = next((c for c in ctx.get("benchmark", [])
                       if c.get("config_id") == eq_ref), None)
        if serial is None:
            raise ProductionEntryError(
                f"no measured {eq_ref} throughput; the affordability gate "
                f"cannot be projected from an unmeasured rate")
        rows_per_hour = float(serial["completed_rows_per_hour"])
        ctx["affordability_rate_basis"] = {
            "config_id": eq_ref, "rows_per_hour": rows_per_hour,
            "why": ("the sealed orchestrator generates serially; the "
                    "selected backend governs the non-O1 benchmark and the "
                    "precommit record, not calibration throughput")}
        done = _existing_row_count(os.path.join(out_dir, "o1_records.jsonl"))
        projected = (O1_ROWS_TOTAL - done) / max(rows_per_hour, 1e-9) * 3600
        return affordability_gate(
            projected_calibration_seconds=projected,
            verification_transfer_reserve_seconds=1200,
            termination_reserve_seconds=300,
            remaining_authorized_runtime_seconds=(
                ctx["runtime_limit"] - clock()))

    def _build_replacement_manifest(path: str) -> str:
        """Replacement freeze manifest for the deployed runtime.

        Exactly two fields differ from the sealed
        FREEZE_MANIFEST.precalibration.json: ``code.torch`` (which the
        sealed orchestrator runtime-asserts against the live interpreter,
        so it MUST carry the deployed build) and ``code.
        runtime_version_note`` (a provenance string recording that
        substitution).  Every other field — including every artifact hash,
        design constant, seed policy and deterministic flag — is copied
        verbatim, and the diff is asserted below so the claim cannot rot.
        The file is re-serialized, so it is not byte-identical; identity is
        established field-by-field, not by bytes.
        """
        import torch
        run_root = os.path.join(ROOT, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN")
        with open(os.path.join(
                run_root, "FREEZE_MANIFEST.precalibration.json"),
                encoding="utf-8") as fh:
            manifest = json.load(fh)
        sealed_torch = manifest["code"]["torch"]
        manifest["code"]["torch"] = torch.__version__
        manifest["code"]["runtime_version_note"] = (
            manifest["code"].get("runtime_version_note", "") +
            f" | INFRASTRUCTURE MIGRATION: code.torch replaced "
            f"{sealed_torch!r} -> {torch.__version__!r} for the B300/cu130 "
            f"accelerator image; all other fields verbatim from the sealed "
            f"manifest")
        # prove the claim rather than assert it: nothing outside those two
        # fields may differ from the sealed manifest
        with open(os.path.join(
                run_root, "FREEZE_MANIFEST.precalibration.json"),
                encoding="utf-8") as fh:
            original = json.load(fh)
        changed = _diff_paths(original, manifest)
        allowed = {"code.torch", "code.runtime_version_note"}
        if set(changed) - allowed:
            raise ProductionEntryError(
                f"replacement manifest changed fields outside the permitted "
                f"infrastructure substitution: {sorted(set(changed) - allowed)}")
        atomic_write_text(path, json.dumps(manifest, indent=2,
                                           sort_keys=True) + "\n")
        return path

    def _build_pod_artifact_map(path: str) -> str:
        run_root = os.path.join(ROOT, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN")
        sealed = sealed_import.SEALED_DIR
        atomic_write_text(path, json.dumps({
            "artifact_hashes.parser":
                os.path.join(sealed, "o1_answer_parser_v2.py"),
            "artifact_hashes.prompt_template":
                os.path.join(sealed, "o1_prompt_template_v2.py"),
            "artifact_hashes.random_axis_tensor":
                os.path.join(run_root, "AXIS_PACKAGE_V2",
                             "random_axes_l3_24.npy"),
            "artifact_hashes.structured_axis_tensor":
                os.path.join(run_root, "AXIS_PACKAGE_V2", "axes_l3_24.npy"),
            "artifact_hashes.tokenizer":
                os.path.join(run_root, "TOKENIZER_BINDING.json"),
            "artifact_hashes.verifier_implementation":
                os.path.join(sealed, "o1_truth_table_verifier_v2.py"),
            "code.generation_module_sha256":
                os.path.join(sealed, "run_o1_v2_generation.py"),
            "code.orchestrator_module_sha256":
                os.path.join(sealed, "run_o1_v2_orchestrator.py"),
            "code.transport_module_sha256":
                os.path.join(sealed, "o1_transport_v2.py"),
            "model.checkpoint_sha256": CHECKPOINT_DIR,
        }, indent=2, sort_keys=True) + "\n")
        return path

    def calibration(ctx):
        # 1. restore any durable records from an earlier evicted pod, so the
        #    sealed orchestrator resumes at the next missing canonical row
        records_path = os.path.join(out_dir, "o1_records.jsonl")
        progress_path = os.path.join(out_dir, "o1_progress.json")
        prior = records_mirror.restore_latest(os.path.join(
            out_dir, "durable_restore"))
        if prior is not None and not os.path.exists(records_path):
            shutil.copyfile(prior["archive"], records_path)
        # restore the wall-clock progress sidecar as well: without it every
        # pod restarts the accumulator and the final metadata under-reports
        # elapsed session time (provenance, not correctness)
        if not os.path.exists(progress_path):
            try:
                if "durable_progress/PROGRESS.json" in store.list_prefix(
                        "durable_progress"):
                    store.fetch_file("durable_progress/PROGRESS.json",
                                     progress_path)
            except Exception as exc:  # noqa: BLE001 - provenance only
                _log_event(out_dir, "PROGRESS_RESTORE_FAILED",
                           error=str(exc)[:200])
        # The sealed orchestrator re-generates baseline stream 0 on resume
        # and demands BITWISE-identical tokens against the stored row.  A
        # different accelerator architecture (sm_103 vs sm_100) can change
        # those bits, which would poison the durable dataset permanently,
        # so the profile that produced the existing rows is binding.
        profile_key_path = "durable_o1_records/PROFILE.json"
        if _existing_row_count(records_path) > 0:
            try:
                import tempfile as _tf
                if profile_key_path in store.list_prefix("durable_o1_records"):
                    with _tf.TemporaryDirectory() as _t:
                        _p = os.path.join(_t, "PROFILE.json")
                        store.fetch_file(profile_key_path, _p)
                        with open(_p, encoding="utf-8") as fh:
                            bound = json.load(fh).get("profile")
                    if bound and bound != profile_key:
                        raise ProductionEntryError(
                            f"durable rows were generated on {bound} but this "
                            f"pod is {profile_key}; the sealed resume replays "
                            f"baselines bitwise, so a different accelerator "
                            f"architecture would corrupt the dataset. "
                            f"Reacquire {bound}, or start a new session "
                            f"prefix for a clean run.")
            except ProductionEntryError:
                raise
            except Exception as exc:  # noqa: BLE001 - absent marker is fine
                _log_event(out_dir, "PROFILE_BINDING_UNREADABLE",
                           error=str(exc)[:200])
        else:
            marker = os.path.join(out_dir, "PROFILE.json")
            atomic_write_text(marker, json.dumps(
                {"profile": profile_key,
                 "why": "binds the accelerator architecture that produced the "
                        "first rows; the sealed resume replays baselines "
                        "bitwise"}, indent=2, sort_keys=True) + "\n")
            try:
                store.push_file(marker, profile_key_path)
            except Exception as exc:  # noqa: BLE001
                _log_event(out_dir, "PROFILE_BINDING_PUBLISH_FAILED",
                           error=str(exc)[:200])
        run_root = os.path.join(ROOT, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN")
        # 2. replacement manifest (deployed torch) + pod artifact map +
        #    regenerated sealed-format precommit bound to them
        manifest_path = _build_replacement_manifest(
            os.path.join(out_dir, "FREEZE_MANIFEST.deployed.json"))
        artifact_map_path = _build_pod_artifact_map(
            os.path.join(out_dir, "RUNTIME_ARTIFACT_PATHS.pod.json"))
        sealed_precommit_path = os.path.join(
            out_dir, "CALIBRATION_PRECOMMIT.sealed_format.json")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": sealed_import.SEALED_DIR}
        # The sealed record sink binds every row to the precommit's sha256
        # and refuses to mix bindings, so the precommit must be minted ONCE
        # for the whole session and reused by every later pod — regenerating
        # it after an eviction would both break resume and overwrite the
        # session's external pre-registration after partial results exist.
        durable_key = "commitments/CALIBRATION_PRECOMMIT.sealed.json"
        reused = False
        try:
            if durable_key in store.list_prefix("commitments"):
                store.fetch_file(durable_key, sealed_precommit_path)
                reused = True
        except Exception as exc:  # noqa: BLE001 - absent witness = fresh mint
            _log_event(out_dir, "COMMITMENT_FETCH_FAILED", error=str(exc)[:200])
        if not reused:
            regen = subprocess.run(
                [sys.executable,
                 os.path.join(sealed_import.SEALED_DIR,
                              "calibration_precommit.py"),
                 "--output", sealed_precommit_path,
                 "--manifest-design", manifest_path,
                 "--artifact-paths", artifact_map_path,
                 "--calibration-task-manifest",
                 os.path.join(run_root, "COHORTS", "calibration_tasks.jsonl"),
                 # ARM the sealed "no precommit once records exist" guard
                 # rather than bypassing it by omitting the flag
                 "--records", records_path],
                capture_output=True, text=True, timeout=1800, env=env,
                cwd=sealed_import.SEALED_DIR)
            if regen.returncode != 0 or not os.path.exists(
                    sealed_precommit_path):
                raise ProductionEntryError(
                    f"sealed precommit regeneration failed:\n"
                    f"{regen.stdout[-1000:]}{regen.stderr[-1000:]}")
            # publish the one-time external witness (write-once: a later pod
            # reuses it, and the branch above never re-mints over it)
            store.push_file(sealed_precommit_path, durable_key)
        _log_event(out_dir, "SEALED_PRECOMMIT_BOUND", reused=reused)
        # 3. run the sealed v2.1 orchestrator (never re-implemented) as a
        #    subprocess; mirror the records file periodically for durability
        cmd = [
            sys.executable,
            os.path.join(sealed_import.SEALED_DIR,
                         "run_o1_v2_orchestrator.py"),
            "calibration",
            "--manifest-design", manifest_path,
            "--artifact-paths", artifact_map_path,
            "--precommit", sealed_precommit_path,
            "--calibration-task-manifest",
            os.path.join(run_root, "COHORTS", "calibration_tasks.jsonl"),
            "--axis-package", os.path.join(run_root, "AXIS_PACKAGE_V2"),
            "--checkpoint", CHECKPOINT_DIR,
            "--output", records_path,
            "--metadata-output", os.path.join(out_dir, "o1_metadata.json"),
            "--progress", progress_path,
            "--boundary-cache-dir", os.path.join(out_dir, "boundary_cache"),
        ]
        # binary stdout + explicit decoding: text=True decodes strict UTF-8,
        # so one stray byte from a CUDA/NCCL/driver message would raise
        # UnicodeDecodeError inside the drain and strand the child on a full
        # pipe — the exact hang the drain exists to prevent.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                env=env, cwd=sealed_import.SEALED_DIR)
        last_mirrored = _existing_row_count(records_path)
        log_path = os.path.join(out_dir, "o1_orchestrator.log")
        # Drain the child's output on a THREAD: readline() blocks, so
        # driving the mirror cadence from the same loop meant a quiet
        # stretch of the orchestrator stalled mirroring indefinitely and
        # silently widened the eviction-loss window far past the nominal
        # row interval.  The mirror now ticks on wall time regardless.
        drain_state = {"error": None}

        def _drain():
            """Consume the child's stdout UNCONDITIONALLY.

            If logging fails (ENOSPC on the container disk, permissions),
            the thread must keep draining anyway: a dead drain leaves the
            child blocked on a full pipe forever while the poll loop spins,
            burning the entire remaining allocation with nothing bounding
            it but the off-pod watchdog.
            """
            log = None
            try:
                log = open(log_path, "ab")
            except Exception as exc:  # noqa: BLE001
                drain_state["error"] = repr(exc)[:200]
            try:
                for line in proc.stdout:          # bytes; never decoded
                    if log is not None:
                        try:
                            log.write(line)
                            log.flush()
                        except Exception as exc:  # noqa: BLE001
                            drain_state["error"] = repr(exc)[:200]
                            try:
                                log.close()
                            except Exception:  # noqa: BLE001
                                pass
                            log = None      # keep consuming, stop logging
            except Exception as exc:  # noqa: BLE001
                drain_state["error"] = repr(exc)[:200]
            finally:
                if log is not None:
                    try:
                        log.close()
                    except Exception:  # noqa: BLE001
                        pass
        drain = threading.Thread(target=_drain, daemon=True)
        drain.start()
        while proc.poll() is None:
            time.sleep(RECORDS_SYNC_POLL_SECONDS)
            # hard wall-clock bound tied to the authorized runtime: the
            # orchestrator can never outlive the session's paid allowance
            if clock() > ctx["runtime_limit"]:
                proc.kill()
                proc.wait(timeout=120)
                rows_at_kill = _existing_row_count(records_path)
                records_mirror.sync_checkpoint(
                    records_path, {"rows": rows_at_kill,
                                   "progress": "runtime_limit_reached"})
                raise ProductionEntryError(
                    f"authorized runtime ({ctx['runtime_limit']}s) reached "
                    f"with {rows_at_kill} rows committed; orchestrator "
                    f"killed and records mirrored")
            rows_now = _existing_row_count(records_path)
            if rows_now - last_mirrored >= RECORDS_SYNC_EVERY_ROWS:
                records_mirror.sync_checkpoint(
                    records_path, {"rows": rows_now, "progress": "partial"})
                last_mirrored = rows_now
                if os.path.exists(progress_path):
                    try:
                        store.push_file(progress_path,
                                        "durable_progress/PROGRESS.json")
                    except Exception as exc:  # noqa: BLE001 - provenance only
                        _log_event(out_dir, "PROGRESS_SYNC_FAILED",
                                   error=str(exc)[:200])
        drain.join(timeout=60)
        if drain_state["error"]:
            _log_event(out_dir, "ORCHESTRATOR_LOG_CAPTURE_DEGRADED",
                       error=drain_state["error"])
        rows_final = _existing_row_count(records_path)
        records_mirror.sync_checkpoint(records_path,
                                       {"rows": rows_final,
                                        "progress": "final"
                                        if proc.returncode == 0
                                        else "interrupted"})
        if proc.returncode != 0:
            raise ProductionEntryError(
                f"sealed orchestrator exited {proc.returncode}; records "
                f"mirrored at {rows_final} rows (see o1_orchestrator.log)")
        ctx["calibration_records_path"] = records_path
        return {"rows": rows_final}

    def record_verify(ctx):
        # the sealed orchestrator already verifies each row (parser,
        # truth-table verifier, token/text binding); here we recount and
        # hash the final artifact for the transfer manifest
        from .identity import sha256_file
        path = ctx["calibration_records_path"]
        n = _existing_row_count(path)
        if n != O1_ROWS_TOTAL:
            raise ProductionEntryError(
                f"records carry {n} rows, expected {O1_ROWS_TOTAL}")
        ctx["records_sha256"] = sha256_file(path)
        return {"verified_rows": n, "records_sha256": ctx["records_sha256"]}

    def result_transfer(ctx):
        from .identity import sha256_file
        archive = os.path.join(out_dir, "o1_results.tar.gz")
        import tarfile
        with tarfile.open(archive, "w:gz") as tar:
            for name in ("o1_records.jsonl", "o1_metadata.json",
                         "HARDWARE_GATE_REPORT.json",
                         "ENVIRONMENT_REPORT.resolved.json",
                         "BENCHMARK_REPORT.real.json",
                         "BACKEND_SELECTION.json",
                         "CALIBRATION_PRECOMMIT.deployed.json"):
                p = os.path.join(out_dir, name)
                if os.path.exists(p):
                    tar.add(p, arcname=name)
        digest = sha256_file(archive)
        store.push_file(archive, "results/o1_results.tar.gz")
        back = os.path.join(out_dir, "transfer_verify.tar.gz")
        store.fetch_file("results/o1_results.tar.gz", back)
        if sha256_file(back) != digest:
            raise ProductionEntryError("result transfer verification failed")
        os.remove(back)
        # publish the digest sidecar LAST: the off-pod driver cross-checks
        # its download against it, so a stale archive from an earlier
        # attempt can never be reported as this session's result
        sidecar = os.path.join(out_dir, "o1_results.tar.gz.sha256")
        atomic_write_text(sidecar, json.dumps(
            {"archive": "o1_results.tar.gz", "sha256": digest,
             "rows": _existing_row_count(
                 os.path.join(out_dir, "o1_records.jsonl"))},
            indent=2, sort_keys=True) + "\n")
        store.push_file(sidecar, "results/o1_results.tar.gz.sha256")
        _log_event(out_dir, "RESULTS_PUBLISHED", sha256=digest)
        return {"transferred": True, "sha256": digest}

    return {
        "PRECHECK": precheck,
        "ARTIFACT_VERIFY": artifact_verify,
        "ENVIRONMENT_VERIFY": environment_verify,
        "NON_O1_EQUIVALENCE": non_o1_equivalence,
        "NON_O1_BENCHMARK": non_o1_benchmark,
        "BACKEND_SELECT": backend_select,
        "PRECOMMIT_BUILD": precommit_build,
        "EXTERNAL_COMMIT_VERIFY": external_commit_verify,
        "CALIBRATION_AFFORDABILITY_CHECK": affordability,
        "CALIBRATION": calibration,
        "RECORD_VERIFY": record_verify,
        "RESULT_TRANSFER": result_transfer,
    }


def _log_event(out_dir: str, event: str, **fields) -> None:
    """Append one machine-readable pod-side event (durability, commitments)."""
    from ..provider.runpod.redaction import redact
    line = json.dumps({"event": event,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
                       **fields}, sort_keys=True, default=str)
    with open(os.path.join(out_dir, "production_entry_events.jsonl"), "a",
              encoding="utf-8") as fh:
        fh.write(redact(line) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _diff_paths(a, b, prefix: str = "") -> list[str]:
    """Dotted paths where two nested JSON documents differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            p = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.append(p)
            else:
                out.extend(_diff_paths(a[key], b[key], p))
        return out
    return [] if a == b else [prefix or "<root>"]


def _existing_row_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def main() -> int:
    out_dir = os.environ.get("O1_B200_OUT", "/outputs")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.monotonic()
    clock = lambda: time.monotonic() - t0  # noqa: E731
    provider = LocalProviderAdapter(out_dir)
    handlers = build_production_handlers(out_dir, provider, clock)
    watchdog = BudgetWatchdog(
        int(float(_require_env("O1_SESSION_AUTHORIZED_SECONDS"))),
        clock=clock)
    machine = ZeroTouchStateMachine(provider, out_dir, handlers,
                                    watchdog=watchdog, clock=clock)
    status = machine.run()
    # The off-pod driver greps these markers.  Both are written literally
    # (not assembled) so the marker the driver looks for and the marker the
    # pod emits cannot drift apart: a deterministic abort mistaken for an
    # eviction costs a full reacquisition cycle.
    if status["outcome"] == "COMPLETE":
        print("ZERO_TOUCH_COMPLETE")
    else:
        print("ZERO_TOUCH_ABORTED_AT_" + str(
            status.get("failed_state") or status["outcome"]))
    return 0 if status["outcome"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
