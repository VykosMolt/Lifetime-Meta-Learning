"""Row-specific RNG: sealed mapping identity, purity, and stream invariance."""
from __future__ import annotations

import torch

from _h import CORPUS_DIR, Runner

from o1_b200.runner import row_rng
from o1_b200.runner.identity import derive_stream_seed
from o1_b200.runner.runbuild import build_validation_bundle
from o1_b200.runner.row_specs import validation_rowspecs


def run() -> Runner:
    r = Runner("row_rng")
    artifact = {"kind": "synthetic", "device": "cpu", "seed_tag": 0}
    bundle = build_validation_bundle(CORPUS_DIR, artifact)
    specs = bundle.specs

    def seeds_match_sealed_derivation():
        from o1_b200.runner.validation_corpus import load_corpus
        _, config = load_corpus(CORPUS_DIR)
        for s in specs:
            row_rng.verify_spec_seed(s, config["master_seed"])
    r.check("every RowSpec seed equals the sealed CRN derivation",
            seeds_match_sealed_derivation)

    def crn_pairing_preserved():
        by_ts = {}
        for s in specs:
            by_ts.setdefault((s.task_id, s.stream_index), set()).add(s.seed)
        assert all(len(v) == 1 for v in by_ts.values()), \
            "same (task, stream) must share one seed across arms (CRN)"
    r.check("CRN pairing: same (task, stream) shares one seed across arms",
            crn_pairing_preserved)

    def step_seed_is_pure_function():
        s = specs[0]
        a = [row_rng.step_generator_seed(s.seed, t) for t in range(16)]
        b = [row_rng.step_generator_seed(s.seed, t) for t in range(16)]
        assert a == b and a[1] == s.seed + 1
    r.check("per-step generator seed is a pure function of (seed, step)",
            step_seed_is_pure_function)

    def audit_record_recomputable():
        s = specs[3]
        one = row_rng.rng_audit_record(s)
        two = row_rng.rng_audit_record(s)
        assert one == two
        assert one["identity"]["task_id"] == s.task_id
        assert one["identity"]["arm"] == s.arm
        assert one["identity"]["stream_index"] == s.stream_index
    r.check("RNG audit record binds full identity and is recomputable",
            audit_record_recomputable)

    def sealed_sampler_deterministic_and_seed_sensitive():
        logits = torch.linspace(-2.0, 2.0, 512)
        t1 = row_rng.sample_token(logits, 12345, 0)
        t2 = row_rng.sample_token(logits, 12345, 0)
        assert t1 == t2, "same (logits, seed, step) must sample identically"
        draws = {row_rng.sample_token(logits, 12345, st) for st in range(64)}
        assert len(draws) > 1, "different steps must be able to differ"
    r.check("sealed sampler: deterministic per (seed, step), varies over steps",
            sealed_sampler_deterministic_and_seed_sensitive)

    def global_rng_cannot_leak():
        logits = torch.linspace(-1.0, 3.0, 512)
        torch.manual_seed(111)
        a = [row_rng.sample_token(logits, 999, t) for t in range(32)]
        torch.manual_seed(222)
        torch.rand(1000)  # consume global RNG heavily
        b = [row_rng.sample_token(logits, 999, t) for t in range(32)]
        assert a == b, "sampling must be independent of global RNG state"
    r.check("HOSTILE global-RNG leakage: global state cannot move any stream",
            global_rng_cannot_leak)

    def neighbor_completion_cannot_remap():
        # the mapping row->stream is a pure function of sealed identity;
        # regenerating the spec list in any task order yields identical seeds
        from o1_b200.runner.validation_corpus import load_corpus
        tasks, config = load_corpus(CORPUS_DIR)
        cfg = {"master_seed": config["master_seed"],
               "binding_sha256": config["binding_sha256"],
               "arms": config["arms"],
               "alpha_structured": config["alpha_structured"],
               "alpha_random": config["alpha_random"]}
        fwd = validation_rowspecs(tasks, cfg)
        rev = validation_rowspecs(list(reversed(tasks)), cfg)
        fwd_map = {s.row_id: (s.seed, s.stream_index) for s in fwd}
        for s in rev:
            # row_id binds task order-independent identity; note the Latin
            # square rank DOES depend on manifest order by design, so
            # identical row identities must keep identical seeds/streams
            if s.row_id in fwd_map:
                assert fwd_map[s.row_id] == (s.seed, s.stream_index)
    r.check("scheduling/enumeration order cannot remap seed or stream of a row",
            neighbor_completion_cannot_remap)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
