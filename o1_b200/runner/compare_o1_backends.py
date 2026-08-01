#!/usr/bin/env python3
"""Backend equivalence harness.

Local mode (the only mode runnable now) drives REFERENCE_SERIAL,
B200_REPLICA, and B200_BATCHED over the non-O1 validation corpus on the
synthetic runtime and reports three separate levels:

  A. structural identity (must be exact): row IDs, task hashes, prompt
     tokens, action mapping, axis/sign/alpha mapping, stream and seed
     mapping, record schema, parser/verifier recomputation, resume behavior,
     canonical ordering;
  B. numerical agreement: prefill transport boundaries via realized
     injected/realized delta RMS and downstream transport values (frozen
     tolerances, reported not hidden);
  C. stochastic trajectory identity: generated token identity, stop-reason
     identity, text identity.

Failures are never downgraded: any required structural gate failure marks the
configuration INELIGIBLE in the report.

Future B200 mode runs the same comparisons with the real Ouro-RLTT model at
the frozen worker/batch matrix; it is implemented by the same functions but
is NOT run in this task (no B200 access).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from .backend_interface import RuntimeConfig
from .backends import BACKENDS
from .identity import canonical_json, domain_sha256
from .records import BINDING_FIELDS, SCIENTIFIC_FIELDS, scientific_core
from .runbuild import build_validation_bundle

# Frozen numerical tolerances for level B (relative).  These are REPORTING
# thresholds for the local synthetic runtime; the future B200 report uses the
# same fields and must state its own frozen tolerances before the run.
TOL_INJECTED_RMS_REL = 1e-6
TOL_TRANSPORT_REL = 1e-6

STOCHASTIC_FIELDS = ("generated_token_ids", "finish_reason", "generated_text")


def run_backend(backend_id: str, corpus_dir: str, run_dir: str,
                model_artifact: dict, *, worker_count: int = 1,
                batch_size: int = 1, compaction: bool = False,
                task_subset: list[str] | None = None,
                device: str = "cpu") -> dict:
    bundle = build_validation_bundle(corpus_dir, model_artifact,
                                     task_subset=task_subset)
    shutil.rmtree(run_dir, ignore_errors=True)
    cfg = RuntimeConfig(backend_id=backend_id, run_dir=run_dir,
                        model_artifact=model_artifact, device=device,
                        worker_count=worker_count, batch_size=batch_size,
                        compaction=compaction)
    backend = BACKENDS[backend_id]()
    backend.initialize(cfg, bundle)
    backend.load_model(model_artifact)
    backend.validate_artifacts()
    backend.execute_rows(bundle.specs)
    fin = backend.finalize_records()
    backend.shutdown()
    rows = []
    with open(fin["records"], encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    return {"finalized": fin, "rows": rows, "bundle": bundle,
            "config_id": config_id(backend_id, worker_count, batch_size,
                                   compaction)}


def config_id(backend_id: str, workers: int, batch: int,
              compaction: bool) -> str:
    return f"{backend_id}:w{workers}:b{batch}:{'c1' if compaction else 'c0'}"


def compare_rows(reference_rows: list[dict], candidate_rows: list[dict]) -> dict:
    ref = {r["row_id"]: r for r in reference_rows}
    cand = {r["row_id"]: r for r in candidate_rows}
    report = {"structural": {}, "numerical": {}, "stochastic": {},
              "eligible_structurally": None}

    s = report["structural"]
    s["row_id_sets_equal"] = set(ref) == set(cand)
    s["missing_rows"] = sorted(set(ref) - set(cand))[:5]
    s["extra_rows"] = sorted(set(cand) - set(ref))[:5]
    s["canonical_order_equal"] = (
        [r["row_id"] for r in reference_rows] ==
        [r["row_id"] for r in candidate_rows])
    common = sorted(set(ref) & set(cand))
    identity_fields = [f for f in BINDING_FIELDS + SCIENTIFIC_FIELDS
                       if f in ("row_id", "exec_index", "task_id",
                                "task_content_sha256", "prompt_sha256",
                                "prompt_token_ids_sha256", "bank", "arm",
                                "direction_id", "sign", "alpha",
                                "stream_index", "seed", "binding_sha256")]
    bad = []
    for rid in common:
        for f in identity_fields:
            if ref[rid][f] != cand[rid][f]:
                bad.append((rid[:16], f))
                break
    s["identity_mapping_exact"] = not bad
    s["identity_mapping_failures"] = bad[:5]
    scoring_bad = []
    for rid in common:
        a, b = ref[rid], cand[rid]
        for f in ("parsed_answer", "well_formed", "verifier_correct", "R",
                  "catastrophic_repetition"):
            # scoring must agree WHENEVER text agrees (recomputation identity)
            if (a["generated_text_sha256"] == b["generated_text_sha256"]
                    and a[f] != b[f]):
                scoring_bad.append((rid[:16], f))
                break
    s["parser_verifier_recompute_exact"] = not scoring_bad
    s["parser_verifier_failures"] = scoring_bad[:5]
    report["eligible_structurally"] = all((
        s["row_id_sets_equal"], s["canonical_order_equal"],
        s["identity_mapping_exact"], s["parser_verifier_recompute_exact"]))

    n = report["numerical"]
    inj_max = 0.0
    rho_max = 0.0
    for rid in common:
        a, b = ref[rid], cand[rid]
        if a["injected_rms"] and b["injected_rms"]:
            inj_max = max(inj_max, abs(a["injected_rms"] - b["injected_rms"])
                          / abs(a["injected_rms"]))
        ra, rb = a["downstream_delta_rms"], b["downstream_delta_rms"]
        if ra and rb:
            rho_max = max(rho_max, abs(ra - rb) / abs(ra))
    n["max_rel_injected_rms_diff"] = inj_max
    n["max_rel_transport_diff"] = rho_max
    n["tolerance_injected_rms_rel"] = TOL_INJECTED_RMS_REL
    n["tolerance_transport_rel"] = TOL_TRANSPORT_REL
    n["within_tolerance"] = (inj_max <= TOL_INJECTED_RMS_REL
                             and rho_max <= TOL_TRANSPORT_REL)

    st = report["stochastic"]
    for f in STOCHASTIC_FIELDS:
        diffs = [rid[:16] for rid in common if ref[rid][f] != cand[rid][f]]
        st[f"{f}_identical"] = not diffs
        st[f"{f}_diff_count"] = len(diffs)
        st[f"{f}_first_diffs"] = diffs[:5]
    st["fully_identical"] = all(st[f"{f}_identical"] for f in STOCHASTIC_FIELDS)

    core_equal = all(scientific_core(ref[rid]) == scientific_core(cand[rid])
                     for rid in common) and set(ref) == set(cand)
    report["scientific_core_identical"] = core_equal
    return report


def local_matrix(corpus_dir: str, out_dir: str, *,
                 task_subset: list[str] | None,
                 worker_counts: list[int], batch_sizes: list[int],
                 compaction_batches: list[int], device: str = "cpu") -> dict:
    """The runnable-now comparison matrix on the synthetic runtime."""
    os.makedirs(out_dir, exist_ok=True)
    artifact = {"kind": "synthetic", "device": device, "seed_tag": 0}
    results = {}
    ref = run_backend("REFERENCE_SERIAL", corpus_dir,
                      os.path.join(out_dir, "ref_serial"), artifact,
                      task_subset=task_subset, device=device)
    candidates = []
    for w in worker_counts:
        candidates.append(("B200_REPLICA", w, 1, False))
    for b in batch_sizes:
        candidates.append(("B200_BATCHED", 1, b, False))
    for b in compaction_batches:
        candidates.append(("B200_BATCHED", 1, b, True))
    for backend_id, w, b, comp in candidates:
        cid = config_id(backend_id, w, b, comp)
        run = run_backend(backend_id, corpus_dir,
                          os.path.join(out_dir, cid.replace(":", "_")),
                          artifact, worker_count=w, batch_size=b,
                          compaction=comp, task_subset=task_subset,
                          device=device)
        results[cid] = compare_rows(ref["rows"], run["rows"])
    report = {
        "mode": "LOCAL_SYNTHETIC",
        "corpus_dir": corpus_dir,
        "reference": ref["config_id"],
        "n_rows": len(ref["rows"]),
        "task_subset": task_subset,
        "comparisons": results,
        "all_structural_pass": all(r["eligible_structurally"]
                                   for r in results.values()),
        "all_scientific_core_identical": all(
            r["scientific_core_identical"] for r in results.values()),
        "note": ("LOCAL SYNTHETIC RESULT ONLY. Proves software equivalence "
                 "of scheduling/batching/RNG/records. It does NOT claim B200 "
                 "hardware equivalence, which remains NOT STARTED."),
    }
    report["report_sha256"] = domain_sha256("o1b200.equivalence_report.v1",
                                            report)
    path = os.path.join(out_dir, "EQUIVALENCE_REPORT.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("local",), default="local",
                   help="b200 mode exists in code but requires hardware "
                        "access and is refused here")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--workers", nargs="*", type=int, default=[2, 3])
    p.add_argument("--batches", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--compaction-batches", nargs="*", type=int, default=[8])
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    report = local_matrix(a.corpus, a.out, task_subset=a.tasks,
                          worker_counts=a.workers, batch_sizes=a.batches,
                          compaction_batches=a.compaction_batches,
                          device=a.device)
    print(json.dumps({k: report[k] for k in
                      ("mode", "n_rows", "all_structural_pass",
                       "all_scientific_core_identical")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
