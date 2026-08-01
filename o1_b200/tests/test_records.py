"""Canonical record schema: construction, recomputation, and forgery attacks."""
from __future__ import annotations

import json

from _h import CORPUS_DIR, Runner, fresh_dir

from o1_b200.runner.backend_interface import RuntimeConfig
from o1_b200.runner.backends import ReferenceSerialBackend
from o1_b200.runner.records import (
    ALL_FIELDS, RecordError, record_hash, verify_record,
)
from o1_b200.runner.runbuild import build_validation_bundle

ARTIFACT = {"kind": "synthetic", "device": "cpu", "seed_tag": 0}


def _one_run(subset):
    bundle = build_validation_bundle(CORPUS_DIR, ARTIFACT, task_subset=subset)
    run_dir = fresh_dir("records_run")
    cfg = RuntimeConfig(backend_id="REFERENCE_SERIAL", run_dir=run_dir,
                        model_artifact=ARTIFACT, device="cpu")
    b = ReferenceSerialBackend()
    b.initialize(cfg, bundle)
    b.load_model(ARTIFACT)
    b.execute_rows(bundle.specs)
    fin = b.finalize_records()
    rows = [json.loads(ln) for ln in open(fin["records"], encoding="utf-8")]
    return bundle, rows


def run() -> Runner:
    r = Runner("records")
    bundle, rows = _one_run(["b200val-000-commit_a"])
    specs_by_id = {s.row_id: s for s in bundle.specs}
    good = next(row for row in rows if row["arm"] == "structured")
    task = bundle.tasks_by_id[good["task_id"]]

    def _verify(rec):
        verify_record(rec, task=task, spec=specs_by_id.get(rec["row_id"]),
                      expected_artifact_ids=bundle.artifact_ids)

    r.check("schema: every record carries the full canonical field set",
            lambda: [([k for k in ALL_FIELDS if k not in row] and
                      (_ for _ in ()).throw(AssertionError(row)))
                     or None for row in rows])
    r.check("a genuine record passes full recomputation",
            lambda: _verify(dict(good)))

    def forge(field, value, rehash=False):
        rec = json.loads(json.dumps(good))
        rec[field] = value
        if rehash:
            rec["record_hash"] = record_hash(rec)
        return rec

    r.expect_raises("HOSTILE forged well_formed is caught", RecordError,
                    lambda: _verify(forge("well_formed",
                                          not good["well_formed"], True)))
    r.expect_raises("HOSTILE forged verifier_correct is caught", RecordError,
                    lambda: _verify(forge("verifier_correct",
                                          not good["verifier_correct"], True)))
    r.expect_raises("HOSTILE forged R is caught", RecordError,
                    lambda: _verify(forge("R", 1 - good["R"], True)))
    r.expect_raises("HOSTILE changed text with stale hash is caught",
                    RecordError,
                    lambda: _verify(forge("generated_text",
                                          good["generated_text"] + "!", True)))
    r.expect_raises("HOSTILE changed tokens with stale hash is caught",
                    RecordError,
                    lambda: _verify(forge("generated_token_ids",
                                          good["generated_token_ids"] + [5],
                                          True)))
    r.expect_raises("HOSTILE record edited without rehash is caught",
                    RecordError,
                    lambda: _verify(forge("generated_text",
                                          good["generated_text"] + "!")))
    r.expect_raises("HOSTILE changed alpha is caught", RecordError,
                    lambda: _verify(forge("alpha", 0.32, True)))
    r.expect_raises("HOSTILE changed sign is caught", RecordError,
                    lambda: _verify(forge("sign", -good["sign"], True)))
    r.expect_raises("HOSTILE changed direction/axis is caught", RecordError,
                    lambda: _verify(forge("direction_id",
                                          1 + (good["direction_id"] % 4), True)))
    r.expect_raises("HOSTILE changed stream is caught", RecordError,
                    lambda: _verify(forge("stream_index",
                                          (good["stream_index"] + 1) % 8, True)))
    r.expect_raises("HOSTILE changed seed is caught", RecordError,
                    lambda: _verify(forge("seed", good["seed"] + 1, True)))
    r.expect_raises("HOSTILE changed task-content hash is caught", RecordError,
                    lambda: _verify(forge("task_content_sha256", "0" * 64, True)))
    r.expect_raises("HOSTILE changed package identity is caught", RecordError,
                    lambda: _verify(forge("package_zip_sha256", "0" * 64, True)))
    r.expect_raises("HOSTILE changed precommit/binding identity is caught",
                    RecordError,
                    lambda: _verify(forge("binding_sha256", "0" * 64, True)))
    r.expect_raises("HOSTILE forged repetition flag is caught", RecordError,
                    lambda: _verify(forge("catastrophic_repetition", True, True)))
    r.expect_raises("HOSTILE baseline transport must be null", RecordError,
                    lambda: _verify(forge("downstream_delta_rms", None, True))
                    if good["arm"] != "baseline" else
                    (_ for _ in ()).throw(RecordError("n/a")))

    def stored_booleans_never_override():
        # verify_record recomputes from generated_text; flipping the stored
        # boolean AND the hash still fails because recomputation wins
        rec = forge("well_formed", not good["well_formed"], rehash=True)
        assert record_hash(rec) == rec["record_hash"]  # attacker was thorough
        try:
            _verify(rec)
        except RecordError:
            return
        raise AssertionError("stored boolean overrode recomputation")
    r.check("stored booleans can never override recomputed values",
            stored_booleans_never_override)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
