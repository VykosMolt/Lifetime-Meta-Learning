"""Corpus disjointness, O1-insertion attacks, benchmark/selection determinism."""
from __future__ import annotations

import json
import os

from _h import CORPUS_DIR, Runner, fresh_dir

from o1_b200.runner.benchmark_o1_b200 import BenchmarkError, load_benchmark_order, run_benchmarks
from o1_b200.runner.runbuild import O1_MANIFEST_PATHS, build_validation_bundle
from o1_b200.runner.selection import SelectionError, select_backend
from o1_b200.runner.validation_corpus import (
    CorpusError, disjointness_report, load_corpus, write_corpus,
)

ARTIFACT = {"kind": "synthetic", "device": "cpu", "seed_tag": 0}


def _candidate(cid, backend, rows_per_hour, **over):
    c = {"config_id": cid, "backend": backend, "workers": over.pop("workers", 1),
         "batch": over.pop("batch", 1), "compaction": over.pop("compaction", False),
         "completed_rows_per_hour": rows_per_hour,
         "peak_hbm_reserved_bytes": over.pop("hbm", 10**11),
         "steady_state_free_hbm_fraction": over.pop("free", 0.4),
         "throughput_stability_spread": over.pop("spread", 0.05)}
    for g in ("structural_pass", "parser_verifier_pass",
              "action_seed_mapping_exact", "intervention_pass",
              "transport_pass", "resume_pass", "no_missing_or_duplicate_rows",
              "no_oom", "free_hbm_fraction_ok", "no_unvalidated_optimization",
              "throughput_stable", "scientific_config_unchanged"):
        c[g] = over.pop(g, True)
    c.update(over)
    return c


def run() -> Runner:
    r = Runner("corpus_policies")

    def corpus_disjoint():
        tasks, config = load_corpus(CORPUS_DIR)
        rep = disjointness_report(tasks, O1_MANIFEST_PATHS)
        assert rep["verdict"] == "DISJOINT"
        assert sum(m["n_tasks"] for m in rep["checked_manifests"]) == 96 + 2400
    r.check("mechanical zero-overlap with the 96-task calibration manifest "
            "and the 2400-task confirmatory candidate pool", corpus_disjoint)

    def o1_task_insertion_refused():
        tasks, _ = load_corpus(CORPUS_DIR)
        with open(O1_MANIFEST_PATHS[0], encoding="utf-8") as fh:
            o1_task = json.loads(fh.readline())
        poisoned = tasks + [o1_task]
        rep = disjointness_report(poisoned, O1_MANIFEST_PATHS)
        assert rep["verdict"] == "OVERLAP_FOUND"
    r.check("HOSTILE O1 task inserted into the validation corpus is detected",
            o1_task_insertion_refused)

    def out_of_population_task_refused():
        try:
            build_validation_bundle(CORPUS_DIR, ARTIFACT,
                                    task_subset=["not-a-corpus-task"])
        except ValueError:
            return
        raise AssertionError("out-of-population validation task accepted")
    r.check("HOSTILE out-of-population validation task is refused",
            out_of_population_task_refused)

    def tampered_corpus_refused():
        import shutil
        d = fresh_dir("tampered_corpus")
        for name in ("corpus_tasks.jsonl", "CORPUS_CONFIG.json",
                     "DISJOINTNESS_REPORT.json"):
            shutil.copy2(os.path.join(CORPUS_DIR, name), os.path.join(d, name))
        with open(os.path.join(d, "corpus_tasks.jsonl"), "a") as fh:
            fh.write("\n")
        try:
            load_corpus(d)
        except CorpusError:
            return
        raise AssertionError("tampered corpus accepted")
    r.check("HOSTILE tampered corpus file is refused by its sealed hash",
            tampered_corpus_refused)

    def benchmark_order_frozen_and_deterministic():
        order = load_benchmark_order()
        ids = [e["config_id"] for e in order["staged_candidates"]]
        assert ids == ["REFERENCE_SERIAL_w1_b1", "B200_REPLICA_w2_b1",
                       "B200_REPLICA_w4_b1", "B200_REPLICA_w8_b1",
                       "B200_BATCHED_w1_b4", "B200_BATCHED_w1_b8",
                       "B200_BATCHED_w1_b16"]
        assert [e["stage"] for e in order["staged_candidates"]] == \
            list(range(1, 8))
    r.check("benchmark policy: staged order is frozen and deterministic",
            benchmark_order_frozen_and_deterministic)

    def benchmark_refuses_non_local_mode():
        try:
            run_benchmarks(CORPUS_DIR, fresh_dir("bench_refuse"), mode="b200")
        except BenchmarkError:
            return
        raise AssertionError("non-local benchmark mode accepted")
    r.check("benchmark harness refuses any mode but local-synthetic",
            benchmark_refuses_non_local_mode)

    def selection_deterministic_and_gated():
        cands = [
            _candidate("B200_BATCHED_w1_b16", "B200_BATCHED", 9000, batch=16,
                       structural_pass=False),      # fastest but ineligible
            _candidate("B200_BATCHED_w1_b8", "B200_BATCHED", 7000, batch=8),
            _candidate("B200_REPLICA_w4_b1", "B200_REPLICA", 7000, workers=4,
                       hbm=9 * 10**10),
            _candidate("REFERENCE_SERIAL_w1_b1", "REFERENCE_SERIAL", 1000),
        ]
        out = select_backend(cands)
        assert out["selected"]["config_id"] == "B200_REPLICA_w4_b1", \
            "tie-break failed (lower HBM should win at equal rows/hour)"
        again = select_backend(list(reversed(cands)))
        assert again["selected"]["config_id"] == out["selected"]["config_id"]
        ineligible = [c for c in out["all_judged"]
                      if c["config_id"] == "B200_BATCHED_w1_b16"][0]
        assert ineligible["eligible"] is False
    r.check("HOSTILE ineligible-but-fastest backend can never win; selection "
            "is deterministic under input reordering",
            selection_deterministic_and_gated)

    def selection_no_eligible_raises():
        try:
            select_backend([_candidate("x", "B200_BATCHED", 1.0, no_oom=False)])
        except SelectionError:
            return
        raise AssertionError("empty eligible set did not raise")
    r.check("selection with no eligible configuration refuses",
            selection_no_eligible_raises)

    def local_benchmark_dress_rehearsal():
        report = run_benchmarks(
            CORPUS_DIR, fresh_dir("bench_local"), mode="local-synthetic",
            task_subset=["b200val-000-commit_a", "b200val-004-malformed_eos"],
            max_stages=3)
        assert report["mode"] == "LOCAL_SYNTHETIC_DRESS_REHEARSAL"
        assert len(report["results"]) == 3
        for res in report["results"]:
            assert res["n_rows"] == 64
            assert res["integrity_failures"] == 0
            assert res["oom_count"] == 0
            assert res["resume_remaining_after_completion"] == 0
    r.check("benchmark harness dress rehearsal (serial + replica w2/w4) "
            "collects metrics on the corpus", local_benchmark_dress_rehearsal)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
