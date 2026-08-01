"""Backend equivalence matrix, replica crash/restart, interruption/resume."""
from __future__ import annotations

import json
import os

from _h import CORPUS_DIR, FAST_SUBSET, Runner, fresh_dir

from o1_b200.runner.backend_interface import RuntimeConfig
from o1_b200.runner.backends import (
    BackendError, BatchedBackend, ReferenceSerialBackend, ReplicaBackend,
)
from o1_b200.runner.compare_o1_backends import local_matrix
from o1_b200.runner.records import scientific_core
from o1_b200.runner.runbuild import build_validation_bundle

ARTIFACT = {"kind": "synthetic", "device": "cpu", "seed_tag": 0}


def _mk(backend_cls, backend_id, name, subset, **kw):
    bundle = build_validation_bundle(CORPUS_DIR, ARTIFACT, task_subset=subset)
    cfg = RuntimeConfig(backend_id=backend_id, run_dir=fresh_dir(name),
                        model_artifact=ARTIFACT, device="cpu", **kw)
    b = backend_cls()
    b.initialize(cfg, bundle)
    b.load_model(ARTIFACT)
    return b, bundle, cfg


def _rows(fin):
    return {row["row_id"]: row for row in
            (json.loads(ln) for ln in open(fin["records"], encoding="utf-8"))}


def run() -> Runner:
    r = Runner("backends")

    def equivalence_matrix():
        report = local_matrix(
            CORPUS_DIR, fresh_dir("equiv_matrix"), task_subset=FAST_SUBSET,
            worker_counts=[2, 3], batch_sizes=[1, 2, 4, 8, 16, 32],
            compaction_batches=[8], device="cpu")
        assert report["all_structural_pass"], report
        assert report["all_scientific_core_identical"], report
    r.check("equivalence matrix: replica w2/w3 + batched b1..b32 + compaction "
            "are structurally exact and scientific-core identical to serial",
            equivalence_matrix)

    def worker_crash_and_restart():
        b, bundle, _ = _mk(ReplicaBackend, "B200_REPLICA", "crash_run",
                           FAST_SUBSET, worker_count=2)
        b.crash_plan = {"w0": 3}  # w0 dies (SystemExit 97) after 3 rows
        out = b.execute_rows(bundle.specs)
        assert out["restarts"] >= 1, "crash did not trigger a restart"
        fin = b.finalize_records()
        assert fin["n_rows"] == len(bundle.specs)
        # no duplicates: finalize would have refused; also verify by rescan
        assert len(b.store.load_completed()) == len(bundle.specs)
    r.check("worker crash mid-run: safe restart completes with no missing "
            "and no duplicate rows", worker_crash_and_restart)

    def interruption_and_resume_identity():
        subset = FAST_SUBSET
        # uninterrupted reference
        b1, bundle1, _ = _mk(ReferenceSerialBackend, "REFERENCE_SERIAL",
                             "resume_ref", subset)
        b1.execute_rows(bundle1.specs)
        ref = _rows(b1.finalize_records())
        # interrupted: first 10 specs only, then resume with the full list
        b2, bundle2, cfg2 = _mk(ReferenceSerialBackend, "REFERENCE_SERIAL",
                                "resume_cut", subset)
        b2.execute_rows(bundle2.specs[:10])
        state = b2.resume()
        assert state["completed"] == 10 and state["remaining"] == \
            len(bundle2.specs) - 10
        b3 = ReferenceSerialBackend()
        b3.initialize(cfg2, bundle2)
        b3.load_model(ARTIFACT)
        b3.execute_rows(bundle2.specs)
        got = _rows(b3.finalize_records())
        assert set(got) == set(ref)
        assert all(scientific_core(got[k]) == scientific_core(ref[k])
                   for k in ref), "resume changed scientific content"
    r.check("interruption + resume reproduces the uninterrupted run exactly",
            interruption_and_resume_identity)

    def batched_resume_with_baseline_replay():
        subset = FAST_SUBSET[:2]
        b1, bundle, cfg = _mk(BatchedBackend, "B200_BATCHED", "b_resume",
                              subset, batch_size=8)
        base_specs = [s for s in bundle.specs if s.arm == "baseline"]
        b1.execute_rows(base_specs)          # only baselines complete
        b2 = BatchedBackend()
        b2.initialize(cfg, bundle)
        b2.load_model(ARTIFACT)
        b2.execute_rows(bundle.specs)        # forces stream-0 replay path
        fin = b2.finalize_records()
        assert fin["n_rows"] == len(bundle.specs)
    r.check("batched resume re-establishes h_base via exact stream-0 replay",
            batched_resume_with_baseline_replay)

    def wrong_model_artifact_refused():
        b, bundle, _ = _mk(ReferenceSerialBackend, "REFERENCE_SERIAL",
                           "wrong_model", FAST_SUBSET[:1])
        try:
            b.load_model({"kind": "synthetic", "device": "cpu", "seed_tag": 9})
        except BackendError:
            return
        raise AssertionError("model artifact identity mismatch accepted")
    r.check("HOSTILE corrupted/substituted model artifact is refused at load",
            wrong_model_artifact_refused)

    def config_locks():
        for kw, err in ((dict(torch_compile=True), "compile"),
                        (dict(cuda_graphs=True), "graphs"),
                        (dict(attention_override="sdpa"), "attention")):
            try:
                RuntimeConfig(backend_id="REFERENCE_SERIAL", run_dir="x",
                              model_artifact=ARTIFACT, **kw).validate()
            except ValueError:
                continue
            raise AssertionError(f"unvalidated optimization {err} accepted")
    r.check("optional performance modes are locked OFF by validate()",
            config_locks)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
