"""Persistence: atomic commits, quarantine, duplicate rejection, resume identity."""
from __future__ import annotations

import json
import os

from _h import CORPUS_DIR, Runner, fresh_dir

from o1_b200.runner.backend_interface import RuntimeConfig
from o1_b200.runner.backends import ReferenceSerialBackend, make_verifier
from o1_b200.runner.persistence import (
    PersistenceError, ResumeIdentityError, RowStore,
)
from o1_b200.runner.records import record_hash
from o1_b200.runner.runbuild import build_validation_bundle

ARTIFACT = {"kind": "synthetic", "device": "cpu", "seed_tag": 0}


def _completed_run(name, subset):
    bundle = build_validation_bundle(CORPUS_DIR, ARTIFACT, task_subset=subset)
    run_dir = fresh_dir(name)
    cfg = RuntimeConfig(backend_id="REFERENCE_SERIAL", run_dir=run_dir,
                        model_artifact=ARTIFACT, device="cpu")
    b = ReferenceSerialBackend()
    b.initialize(cfg, bundle)
    b.load_model(ARTIFACT)
    b.execute_rows(bundle.specs)
    return bundle, cfg, b, run_dir


def run() -> Runner:
    r = Runner("persistence")
    bundle, cfg, backend, run_dir = _completed_run(
        "persist_run", ["b200val-000-commit_a"])
    identity = bundle.resume_identity(cfg, backend.env_sha)
    rows_dir = os.path.join(run_dir, "rows")
    q_dir = os.path.join(run_dir, "quarantine")
    a_row = sorted(os.listdir(rows_dir))[0]

    def reopen():
        return RowStore(run_dir, identity, make_verifier(bundle))

    r.check("reopening a completed store verifies every row",
            lambda: (lambda d: [
                None if len(d) == len(bundle.specs)
                else (_ for _ in ()).throw(AssertionError(len(d)))
            ])(reopen().load_completed()))

    def partial_write_quarantined():
        tmp = os.path.join(rows_dir, "deadbeef.json.tmp")
        with open(tmp, "w") as fh:
            fh.write('{"row_id": "dead')
        done = reopen().load_completed()
        assert not os.path.exists(tmp)
        assert any(n.startswith("deadbeef") for n in os.listdir(q_dir))
        assert len(done) == len(bundle.specs)
    r.check("a partially written row can never appear valid (quarantined)",
            partial_write_quarantined)

    def corrupt_json_quarantined():
        path = os.path.join(rows_dir, a_row)
        original = open(path, encoding="utf-8").read()
        with open(path, "w") as fh:
            fh.write(original[: len(original) // 2])
        done = reopen().load_completed()
        assert len(done) == len(bundle.specs) - 1
        # restore for later checks
        with open(path, "w") as fh:
            fh.write(original)
    r.check("HOSTILE corrupt record is quarantined, not accepted",
            corrupt_json_quarantined)

    def stale_hash_quarantined():
        path = os.path.join(rows_dir, a_row)
        original = open(path, encoding="utf-8").read()
        rec = json.loads(original)
        rec["generated_text"] = rec["generated_text"] + " tampered"
        with open(path, "w") as fh:
            json.dump(rec, fh)
        done = reopen().load_completed()
        assert rec["row_id"] not in done, "stale-hash record accepted!"
        with open(path, "w") as fh:
            fh.write(original)
    r.check("HOSTILE stale record (edited content) is rejected on resume",
            stale_hash_quarantined)

    def duplicate_conflict_refused():
        store = reopen()
        rec = json.loads(open(os.path.join(rows_dir, a_row)).read())
        store.commit_row(rec)  # identical re-commit is a no-op
        rec2 = dict(rec)
        rec2["worker_id"] = "attacker"
        rec2["record_hash"] = record_hash(rec2)
        try:
            store.commit_row(rec2)
        except PersistenceError:
            return
        raise AssertionError("duplicate row with different content accepted")
    r.check("HOSTILE duplicate row with different content is refused",
            duplicate_conflict_refused)

    def foreign_binding_refused():
        store = reopen()
        rec = json.loads(open(os.path.join(rows_dir, a_row)).read())
        rec["binding_sha256"] = "f" * 64
        rec["row_id"] = "f" * 64
        rec["record_hash"] = record_hash(rec)
        try:
            store.commit_row(rec)
        except PersistenceError:
            return
        raise AssertionError("foreign-binding row accepted")
    r.check("HOSTILE row from another sealed run is refused at commit",
            foreign_binding_refused)

    for field, value in [
            ("backend_id", "B200_BATCHED"),
            ("batch_size", 64),
            ("worker_count", 4),
            ("runtime_environment_sha256", "0" * 64),
            ("binding_sha256", "0" * 64),
            ("seal_sha256", "0" * 64),
            ("task_manifest_sha256", "0" * 64),
            ("seed_matrix_sha256", "0" * 64),
            ("source_commit", "0" * 40),
            ("package_zip_sha256", "0" * 64),
            ("model_artifact_sha256", "0" * 64),
            ("tokenizer_sha256", "0" * 64),
            ("axis_package_sha256", "0" * 64),
            ("backend_config_sha256", "0" * 64),
            ("row_spec_manifest_sha256", "0" * 64)]:
        def _attempt(field=field, value=value):
            changed = dict(identity)
            changed[field] = value
            RowStore(run_dir, changed, make_verifier(bundle))
        r.expect_raises(f"resume refuses changed {field}",
                        ResumeIdentityError, _attempt)

    def missing_row_detected():
        store = reopen()
        path = os.path.join(rows_dir, a_row)
        original = open(path).read()
        os.remove(path)
        try:
            store.finalize(bundle.specs)
        except PersistenceError:
            with open(path, "w") as fh:
                fh.write(original)
            return
        raise AssertionError("finalize accepted a missing row")
    r.check("missing-row detection: finalize refuses gaps",
            missing_row_detected)

    def canonical_merge_deterministic():
        store = reopen()
        one = store.finalize(bundle.specs)
        two = reopen().finalize(bundle.specs)
        assert one["records_sha256"] == two["records_sha256"]
        rows = [json.loads(ln) for ln in open(one["records"])]
        assert [x["exec_index"] for x in rows] == list(range(len(rows)))
    r.check("canonical merge is deterministic and exec_index-ordered",
            canonical_merge_deterministic)

    def completed_rows_never_regenerated():
        b2 = ReferenceSerialBackend()
        b2.initialize(cfg, bundle)
        b2.load_model(ARTIFACT)
        out = b2.execute_rows(bundle.specs)
        assert out["n_generated"] == 0, "completed rows were regenerated"
    r.check("a completed, verified row is not regenerated after resume",
            completed_rows_never_regenerated)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
