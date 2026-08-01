"""Hostile fixtures for the sealed v2.1 orchestrator and transport module.

A deterministic fake backend stands in for the model so the ENTIRE sealed
bridge — arm dispatch, Latin square, seeds, record assembly, transport,
resume — runs end to end and its outputs are fed straight into the sealed
calibration and confirmatory analyses.  Every mutation the audit called out
must be rejected at a specific gate.

Run:  python test_orchestrator_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import test_o1_fixtures as base
from calibration_analysis import compute_calibration
from o1_analysis import IntegrityError, sha256_file, token_ids_sha256
from o1_transport_v2 import TransportError, downstream_delta_rms
from run_o1_v2_orchestrator import (
    OrchestrationError,
    orchestrate_calibration,
    orchestrate_confirmatory,
)

HIDDEN = 2048
PASSED = 0
FAILED: list[str] = []


def check(name, fn):
    global PASSED
    try:
        fn()
        PASSED += 1
        print(f"PASS  {name}")
    except Exception as exc:                                       # noqa: BLE001
        FAILED.append(name)
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")


def expect(exc_type, code, fn):
    try:
        fn()
    except exc_type as e:
        if code is not None and isinstance(e, IntegrityError):
            assert e.code == code, f"expected {code}, got {e.code}: {e}"
        return e
    raise AssertionError(f"expected {exc_type.__name__}[{code}]")


class FakeBackend:
    """Deterministic parser/verifier-consistent stand-in for the model.

    The prefill boundary depends only on (task, direction, alpha) — mirroring
    the sampling-free real prefill — while tokens/text depend on the seed.
    """

    def __init__(self, zero_alpha_break: bool = False,
                 baseline_boundary_noise: bool = False,
                 lie_about_well_formed: bool = False):
        self.zero_alpha_break = zero_alpha_break
        self.baseline_boundary_noise = baseline_boundary_noise
        self.lie_about_well_formed = lie_about_well_formed
        self.calls = 0
        self._identity_calls: dict[str, int] = {}

    @staticmethod
    def _h(*parts) -> int:
        return int.from_bytes(hashlib.sha256(
            "\0".join(str(p) for p in parts).encode()).digest()[:8], "big")

    def generate(self, task, seed, direction, signed_alpha):
        self.calls += 1
        tid = task["task_id"]
        dkey = "none" if direction is None else hashlib.sha256(
            np.ascontiguousarray(direction).tobytes()).hexdigest()[:12]
        rng = np.random.default_rng(self._h("boundary", tid) % (2**32))
        boundary = rng.standard_normal(HIDDEN).astype(np.float32) * 0.12
        injected = 0.0
        if direction is not None and signed_alpha != 0.0:
            injected = abs(float(signed_alpha)) * 0.12
            boundary = boundary + np.float32(signed_alpha) * 0.1 * np.asarray(
                direction, dtype=np.float32)
        if direction is None and signed_alpha == 0.0:
            n = self._identity_calls.get(tid, 0) + 1
            self._identity_calls[tid] = n
            # Late-onset nondeterminism: zero-alpha IS the baseline
            # computation, so the only physically possible identity break is a
            # pipeline that stops being deterministic after the baseline bank.
            if self.zero_alpha_break and n > 8:
                boundary = boundary + 1e-3
        if self.baseline_boundary_noise and direction is None:
            boundary = boundary + np.float32(seed % 97) * 1e-6
        h = self._h("outcome", tid, seed, dkey, f"{signed_alpha:.6f}")
        wf = h % 11 != 0
        vc = wf and (h // 11) % 3 == 0
        text = ("FINAL ANSWER: A" if vc else "FINAL ANSWER: B") if wf else \
            "still thinking about it"
        # Tokens are the code points of the text, so this fake tokenizer is a
        # genuine bijection and the sealed token/text binding check is
        # meaningfully exercised.
        toks = [ord(c) for c in text]
        out = {
            "generated_token_ids": toks,
            "generated_text": text,
            "well_formed": wf,
            "verifier_correct": vc,
            "parsed_answer": ("A" if vc else "B") if wf else None,
            "finish_reason": "parseable_final_answer" if wf else "max_new_tokens",
            "logits_finite": True,
            "catastrophic_repetition": False,
            "injected_rms": injected,
            "_prefill_l4_47": boundary.reshape(1, HIDDEN),
        }
        if self.lie_about_well_formed and wf:
            out["well_formed"] = False
        return out

    @staticmethod
    def decode(token_ids) -> str:
        return "".join(chr(int(t)) for t in token_ids)


def _run_dir(tag):
    r = base.write_run(base.make_clean(G=8), tag=tag)
    # the fixture's pre-binding design manifest also verifies the precommit
    return r


def _cal_args(r, tag):
    out = os.path.join(r["dir"], f"orch_{tag}_records.jsonl")
    meta = os.path.join(r["dir"], f"orch_{tag}_metadata.json")
    return dict(
        manifest_design_path=r["manifest"],
        artifact_paths_path=r["artifact_paths"],
        precommit_path=r["cal_precommit"],
        task_manifest_path=r["cal"],
        axis_package_path=r["axis_package"],
        checkpoint="UNUSED_BY_FAKE_BACKEND",
        output_path=out,
        metadata_output_path=meta,
    )


def t00_calibration_end_to_end_through_sealed_analysis():
    r = _run_dir("o00")
    args = _cal_args(r, "o00")
    result = orchestrate_calibration(**args, backend=FakeBackend())
    assert result["n_rows"] == 40 * 8 * 6, result["n_rows"]
    # The fake backend generates instantly; the sealed throughput plausibility
    # ceiling (a red-team fix) would correctly reject that. Substitute a
    # hardware-plausible elapsed for the analysis step, keeping the sealed
    # precommit binding intact.
    meta = json.load(open(args["metadata_output_path"]))
    meta["elapsed_wall_seconds"] = result["n_rows"] / 124.0 * 3600.0
    json.dump(meta, open(args["metadata_output_path"], "w"))
    out = compute_calibration(
        args["output_path"], args["metadata_output_path"], r["manifest"],
        r["cal"], r["cal_precommit"], r["pool"])
    assert out["verdict"] in (
        "PROCEED_POWERED", "PROCEED_BUDGET_CONSTRAINED",
        "TARGET_HEADROOM_EXCLUDED_BY_DISCORDANCE_BOUND",
        "POWER_TARGET_UNREACHABLE", "NO_COHERENT_ALPHA",
        "NO_DIFFICULTY_SETTINGS")
    # deterministic transport recorded on every structured row
    rows = [json.loads(x) for x in open(args["output_path"])]
    st = [x for x in rows if x["arm"] == "structured"]
    assert st and all(x["downstream_delta_rms"] > 0 for x in st)
    assert all(x["downstream_delta_rms"] is None
               for x in rows if x["arm"] == "baseline")


def t01_resume_reproduces_identical_records():
    ra = _run_dir("o01a")
    args_a = _cal_args(ra, "o01a")
    orchestrate_calibration(**args_a, backend=FakeBackend())

    rb = _run_dir("o01b")
    args_b = _cal_args(rb, "o01b")

    class Dying(FakeBackend):
        def generate(self, *a, **kw):
            if self.calls >= 100:
                raise RuntimeError("simulated crash")
            return super().generate(*a, **kw)

    try:
        orchestrate_calibration(**args_b, backend=Dying())
    except RuntimeError:
        pass
    assert not os.path.exists(args_b["metadata_output_path"])
    orchestrate_calibration(**args_b, backend=FakeBackend())
    # Same fixture geometry in a fresh dir: record CONTENT must be identical
    # apart from the run-binding hash, which differs per fixture precommit.
    rows_a = [json.loads(x) for x in open(args_a["output_path"])]
    rows_b = [json.loads(x) for x in open(args_b["output_path"])]
    for row in rows_a + rows_b:
        row.pop("calibration_precommit_sha256")
        row.pop("seed")            # seeds derive from per-fixture task ids
    assert len(rows_a) == len(rows_b) == 40 * 8 * 6
    assert rows_a == rows_b
    assert os.path.exists(args_b["metadata_output_path"])


def t02_foreign_precommit_rows_refused_on_resume():
    r = _run_dir("o02")
    args = _cal_args(r, "o02")
    row = {"task_id": "x", "arm": "baseline", "alpha": 0.0, "stream_index": 0,
           "calibration_precommit_sha256": "0" * 64}
    with open(args["output_path"], "w") as fh:
        fh.write(json.dumps(row) + "\n")
    expect(OrchestrationError, "O6_RESUME_BINDING",
           lambda: orchestrate_calibration(**args, backend=FakeBackend()))


def t03_tampered_runtime_artifact_refused():
    r = _run_dir("o03")
    args = _cal_args(r, "o03")
    paths = json.load(open(r["artifact_paths"]))
    with open(paths["artifact_hashes.parser"], "ab") as fh:
        fh.write(b"tamper")
    expect(OrchestrationError, "O1_ARTIFACT_MAP",
           lambda: orchestrate_calibration(**args, backend=FakeBackend()))


def t04_backend_scoring_lie_caught():
    r = _run_dir("o04")
    args = _cal_args(r, "o04")
    expect(OrchestrationError, "O5_SCORING",
           lambda: orchestrate_calibration(
               **args, backend=FakeBackend(lie_about_well_formed=True)))


def t05_baseline_boundary_nondeterminism_caught():
    r = _run_dir("o05")
    args = _cal_args(r, "o05")
    expect(OrchestrationError, "O4_BOUNDARY",
           lambda: orchestrate_calibration(
               **args, backend=FakeBackend(baseline_boundary_noise=True)))


def _conf_args(r, tag):
    out = os.path.join(r["dir"], f"conf_{tag}_records.jsonl")
    return dict(
        manifest_path=r["manifest"],
        artifact_paths_path=r["artifact_paths"],
        task_manifest_path=r["conf"],
        seed_matrix_path=r["seeds"],
        provenance_path=r["prov"],
        pregeneration_seal_path=r["pregen"],
        axis_package_path=r["axis_package"],
        checkpoint="UNUSED_BY_FAKE_BACKEND",
        output_path=out,
    )


def t06_confirmatory_end_to_end_through_run_primary():
    from o1_analysis import run_primary
    r = _run_dir("o06")
    args = _conf_args(r, "o06")
    result = orchestrate_confirmatory(**args, backend=FakeBackend())
    assert result["n_rows"] == 8 * 8 * 3, result["n_rows"]
    out = run_primary(
        args["output_path"], r["manifest"], r["conf"], r["cal"],
        r["cal_precommit"], r["seeds"],
        os.path.join(r["dir"], "ORCH_RESULT_SEAL.json"), r["prov"],
        r["pregen"], r["cal_records"], r["cal_metadata"], r["cal_results"],
        r["axis_package"], r["pool"])
    assert out["zero_alpha"]["verdict"] == "PROCEED", out["zero_alpha"]
    assert out["zero_alpha"]["parity"]["token_identity_rate"] == 1.0
    assert out["zero_alpha"]["parity"]["zero_transport_rate"] == 1.0
    assert out["primary"] is not None
    print(f"      orchestrated confirmatory verdict: {out['verdict']}")


def t07_zero_alpha_identity_break_halts_at_transport():
    r = _run_dir("o07")
    args = _conf_args(r, "o07")
    try:
        orchestrate_confirmatory(**args, backend=FakeBackend(zero_alpha_break=True))
    except TransportError as e:
        assert "zero-alpha" in str(e)
        return
    raise AssertionError("expected TransportError on broken zero-alpha identity")


def t08_transport_module_contract():
    a = np.zeros(HIDDEN, dtype=np.float32)
    b = np.ones(HIDDEN, dtype=np.float32)
    rho = downstream_delta_rms(b, a, 0.5)
    assert abs(rho - 2.0) < 1e-12
    expect(TransportError, None, lambda: downstream_delta_rms(b, a, 0.0))
    expect(TransportError, None, lambda: downstream_delta_rms(b, a, float("nan")))
    expect(TransportError, None,
           lambda: downstream_delta_rms(b, a, 0.0, is_zero_alpha=True))
    assert downstream_delta_rms(a, a, 0.0, is_zero_alpha=True) == 0.0
    bad = a.copy(); bad[7] = 1e-6
    expect(TransportError, None,
           lambda: downstream_delta_rms(bad, a, 0.0, is_zero_alpha=True))
    nb = b.copy(); nb[0] = np.nan
    expect(TransportError, None, lambda: downstream_delta_rms(nb, a, 0.5))


def t09_duplicate_row_refused():
    r = _run_dir("o09")
    args = _cal_args(r, "o09")
    orchestrate_calibration(**args, backend=FakeBackend())
    rows = [json.loads(x) for x in open(args["output_path"])]
    with open(args["output_path"], "a") as fh:
        fh.write(json.dumps(rows[0]) + "\n")
    os.remove(args["metadata_output_path"])
    expect(OrchestrationError, None,
           lambda: orchestrate_calibration(**args, backend=FakeBackend()))


def t10_metadata_never_written_before_completion():
    r = _run_dir("o10")
    args = _cal_args(r, "o10")

    class DieEarly(FakeBackend):
        def generate(self, *a, **kw):
            if self.calls >= 30:
                raise RuntimeError("die")
            return super().generate(*a, **kw)

    try:
        orchestrate_calibration(**args, backend=DieEarly())
    except RuntimeError:
        pass
    assert not os.path.exists(args["metadata_output_path"])
    assert os.path.exists(args["output_path"])


def t11_forged_tokens_with_clean_text_rejected():
    """Red-team #1 regression: token ids are a coherence input (repetition flag
    -> alpha_star), so a forged token list carrying byte-identical text must be
    rejected by the sealed token/text binding at finalization."""
    r = _run_dir("o11")
    args = _cal_args(r, "o11")
    orchestrate_calibration(**args, backend=FakeBackend())
    rows = [json.loads(x) for x in open(args["output_path"])]
    hit = 0
    for row in rows:
        if row["arm"] == "structured" and float(row["alpha"]) == 0.08:
            toks = [42] * 32                       # genuinely repetitive
            row["generated_token_ids"] = toks
            row["n_generated_tokens"] = len(toks)
            row["generated_token_ids_sha256"] = token_ids_sha256(toks)
            row["catastrophic_repetition"] = True
            hit += 1
    assert hit > 0
    with open(args["output_path"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.remove(args["metadata_output_path"])
    e = expect(OrchestrationError, "O8_TOKEN_TEXT_BINDING",
               lambda: orchestrate_calibration(**args, backend=FakeBackend()))
    assert "do not decode" in str(e)


def t12_poisoned_boundary_cache_rejected():
    """Red-team #5 regression: the boundary cache is derived, never a trust
    root; a poisoned vector must not become h_base for forged transport."""
    r = _run_dir("o12")
    args = _cal_args(r, "o12")

    class StopAfterBaselines(FakeBackend):
        def generate(self, task, seed, direction, signed_alpha):
            if direction is not None:
                raise RuntimeError("stop after the baseline bank")
            return super().generate(task, seed, direction, signed_alpha)

    try:
        orchestrate_calibration(**args, backend=StopAfterBaselines())
    except RuntimeError:
        pass
    cache_dir = args["output_path"] + ".boundaries"
    names = sorted(os.listdir(cache_dir))
    assert names, "no cached boundary to poison"
    poisoned = np.full(HIDDEN, 7.0, dtype=np.float32)
    np.save(os.path.join(cache_dir, names[0]), poisoned)
    e = expect(OrchestrationError, "O4_BOUNDARY",
               lambda: orchestrate_calibration(**args, backend=FakeBackend()))
    assert "cache has been altered" in str(e)


def t13_forged_stored_scoring_on_resume_rejected():
    """Red-team #4 regression: resumed rows are re-scored, not trusted."""
    r = _run_dir("o13")
    args = _cal_args(r, "o13")
    orchestrate_calibration(**args, backend=FakeBackend())
    rows = [json.loads(x) for x in open(args["output_path"])]
    for row in rows:
        if row["arm"] == "baseline" and not row["verifier_correct"]:
            row["verifier_correct"] = True         # free reward, text untouched
            break
    with open(args["output_path"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.remove(args["metadata_output_path"])
    expect(OrchestrationError, "O9_RESUME_SCORING",
           lambda: orchestrate_calibration(**args, backend=FakeBackend()))


def t14_parser_is_linear_on_adversarial_whitespace():
    """Red-team #2 regression: the atomic tail must not backtrack quadratically
    inside the sealed scoring gates."""
    import time
    from o1_answer_parser_v2 import parse_final_answer
    timings = []
    for n in (4000, 64000):
        s = "FINAL ANSWER: A" + " " * n + "x"
        t0 = time.perf_counter()
        assert parse_final_answer(s) is None
        timings.append(time.perf_counter() - t0)
    # 16x input must not cost anywhere near 256x time (quadratic signature)
    assert timings[1] < timings[0] * 40, timings
    assert timings[1] < 0.5, timings


def main() -> int:
    for name in sorted(k for k in globals() if k.startswith("t") and k[1:3].isdigit()):
        check(name, globals()[name])
    total = PASSED + len(FAILED)
    print(f"\n{'=' * 70}\norchestrator fixtures: {PASSED}/{total} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
