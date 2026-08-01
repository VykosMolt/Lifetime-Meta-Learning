"""Full mocked end-to-end dress rehearsal of the zero-touch state machine.

Runs every state with the MockProviderAdapter, the synthetic runtime, and the
non-O1 validation corpus.  The CALIBRATION state executes a small
calibration-SHAPED run over validation tasks (sweep enumeration) — it never
touches an O1 task, and the built-in guards would refuse if it tried.

Produces reports/DRESS_REHEARSAL_REPORT.json.  Also used by the failure
injection suite, which re-runs the machine with a fault at every state.
"""
from __future__ import annotations

import json
import os
import shutil

from .backend_interface import RuntimeConfig
from .backends import BACKENDS
from .budget import BudgetWatchdog, affordability_gate, compute_runtime_limit_seconds
from .compare_o1_backends import compare_rows, run_backend
from .env_report import load_template as load_env_template, unresolved_fields
from .persistence import atomic_write_text
from .precommit_template import finalize, load_template, resolve
from .provider_adapter import MockProviderAdapter
from .runbuild import O1_MANIFEST_PATHS, build_validation_bundle
from .selection import select_backend
from .state_machine import ZeroTouchStateMachine
from .validation_corpus import disjointness_report, load_corpus


class MockClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt: float):
        self.t += dt


def build_handlers(corpus_dir: str, work_dir: str, provider: MockProviderAdapter,
                   clock: MockClock, *, subset: list[str],
                   fail_at: str | None = None,
                   fail_exc: Exception | None = None) -> dict:
    artifact = {"kind": "synthetic", "device": "cpu", "seed_tag": 0}

    def maybe_fail(state):
        if state == fail_at:
            raise fail_exc or RuntimeError(f"injected failure at {state}")

    def precheck(ctx):
        maybe_fail("PRECHECK")
        quote = provider.quote_instance()
        provider.validate_single_gpu()
        ctx["hourly_rate"] = quote["hourly_rate_usd"]
        ctx["runtime_limit"] = compute_runtime_limit_seconds(
            40.00, quote["hourly_rate_usd"])
        ctx["instance_ref"] = provider.start_instance()
        ctx["dress_rehearsal"] = True
        return {"instance": ctx["instance_ref"],
                "runtime_limit_seconds": ctx["runtime_limit"]}

    def artifact_verify(ctx):
        maybe_fail("ARTIFACT_VERIFY")
        tasks, config = load_corpus(corpus_dir)
        rep = disjointness_report(tasks, O1_MANIFEST_PATHS)
        if rep["verdict"] != "DISJOINT":
            raise RuntimeError("corpus not disjoint from O1 populations")
        provider.upload_artifacts(ctx["instance_ref"],
                                  {"corpus": corpus_dir})
        ctx["corpus_config"] = config
        with open(O1_MANIFEST_PATHS[0], encoding="utf-8") as fh:
            ctx["o1_task_ids"] = {json.loads(ln)["task_id"]
                                  for ln in fh if ln.strip()}
        return {"disjointness": rep["verdict"]}

    def environment_verify(ctx):
        maybe_fail("ENVIRONMENT_VERIFY")
        template = load_env_template()
        missing = unresolved_fields(template)
        # in the rehearsal we PROVE the template refuses while unresolved
        if not missing:
            raise RuntimeError("environment template unexpectedly resolved")
        return {"template_unresolved_fields": len(missing),
                "local_stack_validated": True}

    def non_o1_equivalence(ctx):
        maybe_fail("NON_O1_EQUIVALENCE")
        ref = run_backend("REFERENCE_SERIAL", corpus_dir,
                          os.path.join(work_dir, "eq_ref"), artifact,
                          task_subset=subset)
        comp = {}
        for backend_id, w, b in (("B200_REPLICA", 2, 1),
                                 ("B200_BATCHED", 1, 8)):
            cand = run_backend(backend_id, corpus_dir,
                               os.path.join(work_dir, f"eq_{backend_id}"),
                               artifact, worker_count=w, batch_size=b,
                               task_subset=subset)
            comp[backend_id] = compare_rows(ref["rows"], cand["rows"])
        if not all(c["eligible_structurally"] for c in comp.values()):
            raise RuntimeError("structural equivalence gate failed")
        ctx["equivalence"] = {k: {"structural": v["eligible_structurally"],
                                  "core_identical": v["scientific_core_identical"]}
                              for k, v in comp.items()}
        return ctx["equivalence"]

    def non_o1_benchmark(ctx):
        maybe_fail("NON_O1_BENCHMARK")
        clock.advance(600.0)  # mocked benchmark wall time
        ctx["benchmark"] = [
            {"config_id": "REFERENCE_SERIAL_w1_b1", "backend": "REFERENCE_SERIAL",
             "workers": 1, "batch": 1, "completed_rows_per_hour": 1200.0,
             "peak_hbm_reserved_bytes": 4 * 10**10,
             "steady_state_free_hbm_fraction": 0.75,
             "throughput_stability_spread": 0.04},
            {"config_id": "B200_REPLICA_w4_b1", "backend": "B200_REPLICA",
             "workers": 4, "batch": 1, "completed_rows_per_hour": 4100.0,
             "peak_hbm_reserved_bytes": 1.2 * 10**11,
             "steady_state_free_hbm_fraction": 0.35,
             "throughput_stability_spread": 0.08},
            {"config_id": "B200_BATCHED_w1_b8", "backend": "B200_BATCHED",
             "workers": 1, "batch": 8, "completed_rows_per_hour": 5200.0,
             "peak_hbm_reserved_bytes": 6 * 10**10,
             "steady_state_free_hbm_fraction": 0.6,
             "throughput_stability_spread": 0.06}]
        return {"benchmarked": len(ctx["benchmark"])}

    def backend_select(ctx):
        maybe_fail("BACKEND_SELECT")
        gates = {g: True for g in (
            "structural_pass", "parser_verifier_pass",
            "action_seed_mapping_exact", "intervention_pass", "transport_pass",
            "resume_pass", "no_missing_or_duplicate_rows", "no_oom",
            "free_hbm_fraction_ok", "no_unvalidated_optimization",
            "throughput_stable", "scientific_config_unchanged")}
        out = select_backend([{**c, **gates} for c in ctx["benchmark"]])
        ctx["selected_backend"] = out["selected"]
        return {"selected": out["selected"]["config_id"]}

    def precommit_build(ctx):
        maybe_fail("PRECOMMIT_BUILD")
        template = load_template()
        # PROOF: finalization refuses while unresolved
        try:
            finalize(template)
        except Exception:
            pass
        else:
            raise RuntimeError("template finalized while unresolved!")
        resolved = resolve(template, {
            "provider": "MOCK", "instance_id": ctx["instance_ref"],
            "gpu_uuid": "MOCK-UUID", "driver_runtime_report_sha256": "0" * 64,
            "container_image_digest_as_deployed": "sha256:" + "0" * 64,
            "selected_eligible_backend": ctx["selected_backend"]["backend"],
            "selected_worker_batch_configuration":
                ctx["selected_backend"]["config_id"],
            "measured_benchmark_throughput_rows_per_hour":
                ctx["selected_backend"]["completed_rows_per_hour"],
            "actual_hourly_rate_usd": ctx["hourly_rate"],
            "computed_hard_runtime_seconds": ctx["runtime_limit"],
            "environment_digest_sha256": "1" * 64,
            "final_backend_benchmark_report_sha256": "2" * 64,
        }, mock=True)
        ctx["finalized_precommit"] = finalize(resolved)
        return {"finalized": True, "mock": True,
                "document_sha256": ctx["finalized_precommit"]["document_sha256"]}

    def external_commit_verify(ctx):
        maybe_fail("EXTERNAL_COMMIT_VERIFY")
        # mocked push + independent re-fetch/hash verification
        digest = ctx["finalized_precommit"]["document_sha256"]
        remote_copy = json.loads(json.dumps(ctx["finalized_precommit"]))
        if remote_copy["document_sha256"] != digest:
            raise RuntimeError("remote verification failed")
        return {"remote_verified": True}

    def affordability(ctx):
        maybe_fail("CALIBRATION_AFFORDABILITY_CHECK")
        rows_per_hour = ctx["selected_backend"]["completed_rows_per_hour"]
        projected = 4608 / rows_per_hour * 3600
        return affordability_gate(
            projected_calibration_seconds=projected,
            verification_transfer_reserve_seconds=1200,
            termination_reserve_seconds=300,
            remaining_authorized_runtime_seconds=(
                ctx["runtime_limit"] - clock()))

    def calibration(ctx):
        maybe_fail("CALIBRATION")
        # calibration-SHAPED run on validation tasks only (never O1)
        bundle = build_validation_bundle(corpus_dir, artifact, shape="sweep",
                                         task_subset=subset[:1])
        ctx["calibration_binding_sha256"] = bundle.binding_sha256
        ctx["calibration_task_ids"] = list(bundle.tasks_by_id)
        run_dir = os.path.join(work_dir, "rehearsal_calibration")
        shutil.rmtree(run_dir, ignore_errors=True)
        cfg = RuntimeConfig(backend_id="REFERENCE_SERIAL", run_dir=run_dir,
                            model_artifact=artifact, device="cpu")
        backend = BACKENDS["REFERENCE_SERIAL"]()
        backend.initialize(cfg, bundle)
        backend.load_model(artifact)
        backend.execute_rows(bundle.specs)
        fin = backend.finalize_records()
        ctx["calibration_records"] = fin
        ctx["calibration_bundle"] = bundle
        ctx["calibration_cfg"] = cfg
        return {"rows": fin["n_rows"], "records_sha256": fin["records_sha256"]}

    def record_verify(ctx):
        maybe_fail("RECORD_VERIFY")
        from .backends import make_verifier
        verify = make_verifier(ctx["calibration_bundle"])
        n = 0
        with open(ctx["calibration_records"]["records"],
                  encoding="utf-8") as fh:
            for ln in fh:
                verify(json.loads(ln))
                n += 1
        return {"verified_rows": n}

    def result_transfer(ctx):
        maybe_fail("RESULT_TRANSFER")
        dest = os.path.join(work_dir, "downloaded")
        provider.download_results(ctx["instance_ref"], dest)
        shutil.copy2(ctx["calibration_records"]["records"],
                     os.path.join(dest, "records.jsonl"))
        from .identity import sha256_file
        got = sha256_file(os.path.join(dest, "records.jsonl"))
        if got != ctx["calibration_records"]["records_sha256"]:
            raise RuntimeError("transfer hash mismatch")
        return {"transferred": True, "records_sha256": got}

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


def run_rehearsal(corpus_dir: str, out_dir: str, *, subset: list[str],
                  fail_at: str | None = None,
                  provider: MockProviderAdapter | None = None) -> dict:
    clock = MockClock()
    provider = provider or MockProviderAdapter()
    handlers = build_handlers(corpus_dir, out_dir, provider, clock,
                              subset=subset, fail_at=fail_at)
    wd = BudgetWatchdog(compute_runtime_limit_seconds(40.0, 2.99), clock=clock)
    machine = ZeroTouchStateMachine(provider, out_dir, handlers,
                                    watchdog=wd, clock=clock)
    status = machine.run()
    status["provider_calls"] = list(provider.calls)
    return status


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--subset", nargs="*", default=[
        "b200val-000-commit_a", "b200val-004-malformed_eos",
        "b200val-008-stoch_commit"])
    a = p.parse_args()
    status = run_rehearsal(a.corpus, a.out, subset=a.subset)
    atomic_write_text(os.path.join(a.out, "DRESS_REHEARSAL_REPORT.json"),
                      json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outcome": status["outcome"],
                      "states_run": status["states_run"]}, indent=2))
    return 0 if status["outcome"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
