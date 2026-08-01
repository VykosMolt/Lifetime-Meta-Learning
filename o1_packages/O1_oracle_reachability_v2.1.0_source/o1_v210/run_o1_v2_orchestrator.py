#!/usr/bin/env python3
"""Sealed O1 v2.1 generation orchestrator.

v2.0 sealed the single-branch generator and the analysis but left everything
between them — arm dispatch, Latin-square stream assignment, seed lookup,
record assembly, transport computation — to unwritten operator code.  v2.1
seals that bridge: this module is the ONLY way an analysis-valid JSONL row is
produced.  No operator-side ad hoc Python is required or permitted.

The orchestrator owns, per generated row:

  * arm dispatch (baseline / structured / zero_alpha / random /
    structured_secondary) exactly as sealed;
  * action identity: direction_id, sign, alpha, axis-row selection from the
    verified structured or random tensor;
  * the cyclic Latin-square action-to-stream map stream(a,g)=(a+r_g) mod 8
    (calibration: r_g = row index of the task in the sealed calibration
    manifest, mod 8 — all calibration tasks share the outcome-blind
    placeholder stratum, so sealed manifest order is the rank;
    confirmatory: r_g = sealed_rank_within_stratum mod 8, gate G13);
  * deterministic seed lookup (sha256_domain_uint64_be_v1; confirmatory seeds
    additionally verified against the sealed seed matrix);
  * task metadata, task_content_sha256, run/seal binding hashes;
  * exact model invocation through the sealed generator;
  * generated token IDs, generated text, and both digests;
  * token count, stopping status, hit_max_tokens;
  * parser and verifier invocation (well_formed, verifier_correct, R,
    parsed_answer) — the same sealed modules the analysis gates re-run;
  * first-divergence inputs (the token IDs; divergence itself is computed in
    analysis, never trusted from here);
  * the frozen downstream transport measurement via o1_transport_v2 (null on
    baseline; asserted-bitwise 0.0 on zero_alpha; RMS ratio on interventions);
  * runtime provenance binding and environment version asserts.

Resume: the records file is append-only.  Completed rows are keyed by
(task_id, arm, alpha, stream_index); a resumed session verifies the binding
hash of every existing row before continuing and only generates missing rows.
Baseline prefill boundary states are cached per task (prefill is
sampling-free; the orchestrator asserts bitwise equality across baseline
streams before trusting the cache).

Wall-clock accounting: a sidecar progress file accumulates generation wall
seconds across sessions (model load included — confirmatory capacity planning
pays the same cost).  CALIBRATION_METADATA.json is written only at 100%
completion, with exactly the two sealed keys.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
from typing import Any, Callable

import numpy as np

from build_run_provenance import REQUIRED_ARTIFACT_KEYS
from calibration_analysis import verify_calibration_precommit
from o1_answer_parser_v2 import has_catastrophic_repetition, parse_final_answer
from o1_transport_v2 import downstream_delta_rms
from o1_truth_table_verifier_v2 import verify_letter
from o1_analysis import (
    BANK_SIZE,
    IntegrityError,
    Manifest,
    N_DIRECTIONS,
    action_index,
    derive_stream_seed,
    gate_G14_provenance,
    gate_G17_manifest_consistency,
    load_jsonl,
    sha256_file,
    sha256_tree,
    text_sha256,
    token_ids_sha256,
)

HIDDEN = 2048
SIGNED_ACTIONS = tuple(
    (d, s) for d in range(1, N_DIRECTIONS + 1) for s in (1, -1)
)  # action index a = 2*(d-1) + (0 if s==+1 else 1), matching action_index()


class OrchestrationError(IntegrityError):
    pass


# ==========================================================================
# runtime verification
# ==========================================================================


def verify_artifact_paths(manifest_raw: dict, artifact_paths_path: str) -> dict:
    """Hash-verify every runtime artifact against the sealed manifest values."""
    with open(artifact_paths_path) as fh:
        paths = json.load(fh)
    missing = [k for k in REQUIRED_ARTIFACT_KEYS if k not in paths]
    extra = sorted(set(paths) - set(REQUIRED_ARTIFACT_KEYS))
    if missing or extra:
        raise OrchestrationError(
            "O1_ARTIFACT_MAP",
            f"artifact map mismatch: missing={missing}, extra={extra}")
    out = {}
    for key in REQUIRED_ARTIFACT_KEYS:
        path = paths[key]
        if not isinstance(path, str) or not os.path.exists(path):
            raise OrchestrationError(
                "O1_ARTIFACT_MAP", f"{key}: missing artifact {path!r}")
        got = sha256_tree(path)
        node: Any = manifest_raw
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise OrchestrationError(
                    "O1_ARTIFACT_MAP", f"manifest missing {key!r}")
            node = node[part]
        if got != node:
            raise OrchestrationError(
                "O1_ARTIFACT_MAP",
                f"{key}: artifact hashes {got[:16]}... but sealed value is "
                f"{str(node)[:16]}...")
        out[key] = path
    return out


def assert_runtime_versions(manifest_raw: dict) -> None:
    """Exact-version asserts for the pinned generation stack (v2.1 hardening)."""
    import numpy as _np
    import torch as _torch
    import transformers as _transformers

    code = manifest_raw.get("code", {})
    for name, mod_version in (("transformers", _transformers.__version__),
                              ("torch", _torch.__version__),
                              ("numpy", _np.__version__)):
        want = code.get(name)
        if not isinstance(want, str) or want == "FREEZE_SLOT":
            raise OrchestrationError(
                "O2_RUNTIME_VERSIONS", f"manifest code.{name} is not sealed")
        if mod_version != want:
            raise OrchestrationError(
                "O2_RUNTIME_VERSIONS",
                f"{name} must equal {want}, got {mod_version}")
    flags = manifest_raw.get("model", {}).get("deterministic_flags", {})
    want_cublas = flags.get("cublas_workspace_config")
    got_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if want_cublas is not None and got_cublas != want_cublas:
        raise OrchestrationError(
            "O2_RUNTIME_VERSIONS",
            f"CUBLAS_WORKSPACE_CONFIG={got_cublas!r} but the sealed flag is "
            f"{want_cublas!r}")


def load_axis_tensor(path: str, expected_sha256: str, label: str) -> np.ndarray:
    if sha256_file(path) != expected_sha256:
        raise OrchestrationError(
            "O3_AXIS_TENSOR", f"{label} tensor hash disagrees with the seal")
    axes = np.load(path, allow_pickle=False)
    if axes.shape != (N_DIRECTIONS, HIDDEN) or not np.isfinite(axes).all():
        raise OrchestrationError(
            "O3_AXIS_TENSOR", f"{label} tensor is not finite [4,{HIDDEN}]")
    for i in range(N_DIRECTIONS):
        rms = float(np.sqrt(np.mean(axes[i] * axes[i])))
        if not math.isclose(rms, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise OrchestrationError(
                "O3_AXIS_TENSOR", f"{label} row {i} is not unit RMS ({rms})")
    return np.asarray(axes, dtype=np.float64)


# ==========================================================================
# generation backend
# ==========================================================================


class RealBackend:
    """The sealed model-invocation backend (lazy torch import)."""

    def __init__(self, checkpoint: str, manifest_raw: dict):
        assert_runtime_versions(manifest_raw)
        import run_o1_v2_generation as gen
        self._gen = gen
        self.model, self.tokenizer = gen._load_model(pathlib.Path(checkpoint))

    def generate(self, task: dict, seed: int, direction, signed_alpha: float) -> dict:
        return self._gen.generate_branch(
            self.model, self.tokenizer, task, int(seed),
            None if direction is None else np.asarray(direction, dtype=np.float64),
            float(signed_alpha),
        )

    def decode(self, token_ids) -> str:
        """Sealed tokenizer decode, identical to the generation-time call."""
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)


def verify_token_text_binding(rows, decode) -> int:
    """Re-decode every row's token ids and require equality with its text.

    The analysis layer is stdlib+numpy and cannot re-decode, so the token list
    is bound there only to its own digest.  A red-team attack exploited that
    gap: token ids feed the mechanical repetition flag, which feeds the
    coherence criteria, which select ``alpha_star`` — so forged tokens with
    byte-identical text could move the sealed intervention magnitude for the
    entire confirmatory run.  This check runs wherever the sealed tokenizer is
    available (generation finalization and the standalone verifier) and closes
    the loop back onto the scored text.
    """
    n = 0
    for i, row in enumerate(rows):
        toks = [int(t) for t in row["generated_token_ids"]]
        want = row["generated_text"]
        got = decode(toks)
        if got != want:
            raise OrchestrationError(
                "O8_TOKEN_TEXT_BINDING",
                f"row {i} (task {row.get('task_id')} arm {row.get('arm')}): "
                f"generated_token_ids do not decode to generated_text",
                detail={"row": i, "decoded_prefix": got[:80],
                        "stored_prefix": want[:80]})
        if text_sha256(want) != row["generated_text_sha256"]:
            raise OrchestrationError(
                "O8_TOKEN_TEXT_BINDING", f"row {i}: text digest disagrees")
        if token_ids_sha256(toks) != row["generated_token_ids_sha256"]:
            raise OrchestrationError(
                "O8_TOKEN_TEXT_BINDING", f"row {i}: token digest disagrees")
        n += 1
    return n


def verify_stored_scoring(rows, tasks_by_id: dict) -> int:
    """Recompute parser/verifier outcomes for stored rows (resume hardening)."""
    n = 0
    for i, row in enumerate(rows):
        task = tasks_by_id.get(row.get("task_id"))
        if task is None:
            raise OrchestrationError(
                "O9_RESUME_SCORING",
                f"row {i}: task {row.get('task_id')!r} is not in the sealed manifest")
        parsed = parse_final_answer(row["generated_text"])
        if bool(row["well_formed"]) != (parsed is not None):
            raise OrchestrationError(
                "O9_RESUME_SCORING",
                f"row {i}: stored well_formed disagrees with the sealed parser")
        if bool(row["verifier_correct"]) != verify_letter(task, parsed):
            raise OrchestrationError(
                "O9_RESUME_SCORING",
                f"row {i}: stored verifier_correct disagrees with the sealed verifier")
        if "parsed_answer" in row and row.get("parsed_answer") != parsed:
            raise OrchestrationError(
                "O9_RESUME_SCORING",
                f"row {i}: stored parsed_answer disagrees with the sealed parser")
        n += 1
    return n


# ==========================================================================
# shared row machinery
# ==========================================================================


def _boundary_of(branch: dict) -> np.ndarray:
    b = np.asarray(branch["_prefill_l4_47"])
    if b.ndim == 2 and b.shape[0] == 1:
        b = b[0]
    if b.shape != (HIDDEN,):
        raise OrchestrationError(
            "O4_BOUNDARY", f"prefill boundary shape {b.shape} != ({HIDDEN},)")
    return b


def _score_fields(task: dict, branch: dict) -> dict:
    text = branch["generated_text"]
    toks = [int(t) for t in branch["generated_token_ids"]]
    parsed = parse_final_answer(text)
    if branch.get("well_formed") != (parsed is not None):
        raise OrchestrationError(
            "O5_SCORING",
            "generator well_formed disagrees with the sealed parser")
    well_formed = parsed is not None
    verifier_correct = verify_letter(task, parsed)
    if bool(branch.get("verifier_correct")) != verifier_correct:
        raise OrchestrationError(
            "O5_SCORING",
            "generator verifier_correct disagrees with the sealed verifier")
    return {
        "generated_token_ids": toks,
        "generated_text": text,
        "generated_text_sha256": text_sha256(text),
        "generated_token_ids_sha256": token_ids_sha256(toks),
        "n_generated_tokens": len(toks),
        "hit_max_tokens": branch["finish_reason"] == "max_new_tokens",
        "finish_reason": branch["finish_reason"],
        "parsed_answer": parsed,
        "well_formed": well_formed,
        "verifier_correct": verifier_correct,
        "logits_finite": bool(branch["logits_finite"]),
        # Recomputed here from the token ids rather than copied: the flag is a
        # coherence input that can move alpha selection.
        "catastrophic_repetition": has_catastrophic_repetition(toks),
    }


def _row_key(row: dict) -> tuple:
    return (row["task_id"], row["arm"], f"{float(row['alpha']):.12g}",
            int(row["stream_index"]))


class _RecordSink:
    """Append-only JSONL with per-write fsync and resume indexing."""

    def __init__(self, path: str, binding_key: str, binding_value: str):
        self.path = path
        self.done: set[tuple] = set()
        if os.path.exists(path):
            for i, row in enumerate(load_jsonl(path)):
                if row.get(binding_key) != binding_value:
                    raise OrchestrationError(
                        "O6_RESUME_BINDING",
                        f"existing row {i} carries a different {binding_key}; "
                        f"records from another sealed run must not be mixed")
                key = _row_key(row)
                if key in self.done:
                    raise OrchestrationError(
                        "O6_RESUME_BINDING",
                        f"existing records contain a duplicate row {key}; the "
                        f"file has been tampered with or corrupted")
                self.done.add(key)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, row: dict) -> None:
        key = _row_key(row)
        if key in self.done:
            raise OrchestrationError(
                "O6_RESUME_BINDING", f"duplicate row {key} refused")
        self._fh.write(json.dumps(row, sort_keys=True) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.done.add(key)

    def close(self) -> None:
        self._fh.close()


class _Progress:
    """Cross-session wall-clock accumulator (sidecar, not a sealed artifact)."""

    def __init__(self, path: str):
        self.path = path
        self.prior = 0.0
        if os.path.exists(path):
            with open(path) as fh:
                self.prior = float(json.load(fh).get("elapsed_wall_seconds", 0.0))
        self.t0 = time.monotonic()

    def elapsed(self) -> float:
        return self.prior + (time.monotonic() - self.t0)

    def flush(self, extra: dict | None = None) -> None:
        payload = {"elapsed_wall_seconds": self.elapsed()}
        if extra:
            payload.update(extra)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, self.path)


class _BoundaryCache:
    """Per-task baseline L4_47 boundary cache (derived, deterministic)."""

    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)

    def _p(self, task_id: str) -> str:
        return os.path.join(self.dir, f"{task_id}.npy")

    def get(self, task_id: str) -> np.ndarray | None:
        p = self._p(task_id)
        if not os.path.exists(p):
            return None
        arr = np.load(p, allow_pickle=False)
        if arr.shape != (HIDDEN,):
            raise OrchestrationError(
                "O4_BOUNDARY", f"cached boundary for {task_id} has shape {arr.shape}")
        return arr

    def put(self, task_id: str, arr: np.ndarray) -> None:
        p = self._p(task_id)
        if not os.path.exists(p):
            tmp = p[:-len(".npy")] + ".tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, p)


def _run_baseline_bank(backend, task: dict, seeds: dict[int, int],
                       sink: _RecordSink, cache: _BoundaryCache,
                       base_row: Callable[[int, int, dict], dict],
                       replay_check_rows: dict[int, dict]) -> np.ndarray:
    """Generate missing baseline streams; return the VERIFIED h_base.

    The on-disk boundary cache is a derived convenience, never a trust root: a
    red-team attack poisoned it to forge every transport measurement for a
    task.  h_base is therefore only ever accepted from a boundary computed in
    this session; when the baseline bank already exists the stream-0 baseline
    is replayed and required to reproduce the stored tokens bitwise, and any
    cached vector must match that replay exactly.
    """
    task_id = task["task_id"]
    cached = cache.get(task_id)
    h_base = None
    for i in range(BANK_SIZE):
        key = (task_id, "baseline", "0", i)
        if key in sink.done:
            continue
        branch = backend.generate(task, seeds[i], None, 0.0)
        boundary = _boundary_of(branch)
        if h_base is None:
            h_base = boundary
        elif not np.array_equal(boundary, h_base):
            raise OrchestrationError(
                "O4_BOUNDARY",
                f"task {task_id}: baseline prefill boundary differs across "
                f"streams — prefill must be sampling-free and deterministic")
        sink.write(base_row(i, seeds[i], branch))
    if h_base is None:
        # Resume path: every baseline row pre-exists. Replay stream 0 and
        # require exact token reproduction before trusting the boundary.
        branch = backend.generate(task, seeds[0], None, 0.0)
        stored = replay_check_rows.get(0)
        if stored is None or list(stored["generated_token_ids"]) != list(
                branch["generated_token_ids"]):
            raise OrchestrationError(
                "O4_BOUNDARY",
                f"task {task_id}: baseline replay does not reproduce the stored "
                f"row; the boundary cannot be re-established safely")
        h_base = _boundary_of(branch)
    if cached is not None and not np.array_equal(cached, h_base):
        raise OrchestrationError(
            "O4_BOUNDARY",
            f"task {task_id}: cached baseline boundary disagrees with the "
            f"boundary recomputed this session; the cache has been altered")
    if cached is None:
        cache.put(task_id, h_base)
    return h_base


# ==========================================================================
# calibration orchestration
# ==========================================================================


def orchestrate_calibration(
    manifest_design_path: str,
    artifact_paths_path: str,
    precommit_path: str,
    task_manifest_path: str,
    axis_package_path: str,
    checkpoint: str,
    output_path: str,
    metadata_output_path: str,
    progress_path: str | None = None,
    boundary_cache_dir: str | None = None,
    backend=None,
    flush_every_rows: int = 8,
) -> dict:
    with open(manifest_design_path) as fh:
        manifest_raw = json.load(fh)
    verify_calibration_precommit(precommit_path, manifest_raw, task_manifest_path)
    precommit_sha = sha256_file(precommit_path)
    paths = verify_artifact_paths(manifest_raw, artifact_paths_path)

    from verify_axis_artifact import verify as verify_axis
    axis_report = verify_axis(axis_package_path)
    if axis_report.get("verdict") != "SEALABLE":
        raise OrchestrationError(
            "O0_PREFLIGHT", "axis package did not verify as SEALABLE")

    axes = load_axis_tensor(
        paths["artifact_hashes.structured_axis_tensor"],
        manifest_raw["artifact_hashes"]["structured_axis_tensor"], "structured")

    tasks = load_jsonl(task_manifest_path)
    if not tasks:
        raise OrchestrationError("O0_PREFLIGHT", "calibration task manifest is empty")
    grid = [float(x) for x in
            manifest_raw["action_space"]["alpha_swept_in_calibration"]]
    master = int(manifest_raw["cohorts"]["master_seed"])

    if metadata_output_path and os.path.exists(metadata_output_path):
        raise OrchestrationError(
            "O0_PREFLIGHT",
            f"refusing to overwrite existing {metadata_output_path}")

    progress = _Progress(progress_path or output_path + ".progress.json")
    cache = _BoundaryCache(boundary_cache_dir or output_path + ".boundaries")
    sink = _RecordSink(output_path, "calibration_precommit_sha256", precommit_sha)
    existing_rows: dict[tuple, dict] = {}
    if sink.done:
        for row in load_jsonl(output_path):
            existing_rows[_row_key(row)] = row

    if backend is None:
        backend = RealBackend(checkpoint, manifest_raw)

    def common(task: dict) -> dict:
        return {
            "task_id": task["task_id"],
            "task_content_sha256": task["task_content_sha256"],
            "generator_setting_id": task["generator_setting_id"],
            "calibration_precommit_sha256": precommit_sha,
        }

    n_generated = 0
    for g, task in enumerate(tasks):
        rank = g % BANK_SIZE
        seeds = {i: derive_stream_seed(master, task["task_id"], i)
                 for i in range(BANK_SIZE)}

        def base_row(i: int, seed: int, branch: dict) -> dict:
            return {**common(task), "arm": "baseline", "alpha": 0.0,
                    "direction_id": None, "sign": None, "stream_index": i,
                    "seed": seed, "downstream_delta_rms": None,
                    "injected_rms": 0.0, **_score_fields(task, branch)}

        replay = {i: existing_rows.get((task["task_id"], "baseline", "0", i))
                  for i in range(BANK_SIZE)}
        need_any = any(
            (task["task_id"], arm, f"{float(a):.12g}", i) not in sink.done
            for arm, alphas in (("baseline", [0.0]), ("structured", grid))
            for a in alphas for i in range(BANK_SIZE))
        if not need_any:
            continue
        h_base = _run_baseline_bank(
            backend, task, seeds, sink, cache, base_row, replay)

        for alpha in grid:
            for d, s in SIGNED_ACTIONS:
                stream = (action_index(d, s) + rank) % BANK_SIZE
                key = (task["task_id"], "structured", f"{float(alpha):.12g}", stream)
                if key in sink.done:
                    continue
                branch = backend.generate(
                    task, seeds[stream], axes[d - 1], s * alpha)
                rho = downstream_delta_rms(
                    _boundary_of(branch), h_base, float(branch["injected_rms"]))
                sink.write({
                    **common(task), "arm": "structured", "alpha": float(alpha),
                    "direction_id": d, "sign": s, "stream_index": stream,
                    "seed": seeds[stream],
                    "downstream_delta_rms": rho,
                    "injected_rms": float(branch["injected_rms"]),
                    **_score_fields(task, branch)})
                n_generated += 1
                if n_generated % flush_every_rows == 0:
                    progress.flush({"tasks_completed": g})
        progress.flush({"tasks_completed": g + 1})

    total_expected = len(tasks) * BANK_SIZE * (1 + len(grid))
    if len(sink.done) != total_expected:
        raise OrchestrationError(
            "O7_COMPLETENESS",
            f"{len(sink.done)} rows present, {total_expected} sealed rows "
            f"expected — resume until complete before metadata is written")
    sink.close()
    # Finalization audit over the COMPLETE record set, including any rows
    # carried in from an earlier session: tokens must decode to their text and
    # the stored scoring must reproduce under the sealed parser/verifier.
    final_rows = load_jsonl(output_path)
    verify_token_text_binding(final_rows, backend.decode)
    verify_stored_scoring(final_rows, {t["task_id"]: t for t in tasks})
    progress.flush({"tasks_completed": len(tasks), "complete": True})
    metadata = {
        "elapsed_wall_seconds": progress.elapsed(),
        "calibration_precommit_sha256": precommit_sha,
    }
    tmp = metadata_output_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, metadata_output_path)
    return {
        "records": output_path,
        "records_sha256": sha256_file(output_path),
        "metadata": metadata_output_path,
        "metadata_sha256": sha256_file(metadata_output_path),
        "n_rows": len(sink.done),
        "elapsed_wall_seconds": metadata["elapsed_wall_seconds"],
    }


# ==========================================================================
# confirmatory orchestration
# ==========================================================================


ARM_GEOMETRY = {
    "baseline": (False, None),
    "zero_alpha": (True, None),
    "structured": (True, "structured"),
    "structured_secondary": (True, "structured"),
    "random": (True, "random"),
}


def orchestrate_confirmatory(
    manifest_path: str,
    artifact_paths_path: str,
    task_manifest_path: str,
    seed_matrix_path: str,
    provenance_path: str,
    pregeneration_seal_path: str,
    axis_package_path: str,
    checkpoint: str,
    output_path: str,
    progress_path: str | None = None,
    boundary_cache_dir: str | None = None,
    backend=None,
    flush_every_rows: int = 8,
) -> dict:
    man = Manifest(manifest_path)
    gate_G17_manifest_consistency(man)
    gate_G14_provenance(man, provenance_path)
    if not os.path.exists(pregeneration_seal_path):
        raise OrchestrationError(
            "O0_PREFLIGHT",
            "PREGENERATION_SEAL.json must exist before confirmatory generation")
    with open(pregeneration_seal_path) as fh:
        seal = json.load(fh)
    if seal.get("manifest_sha256") != man.sha256:
        raise OrchestrationError(
            "O0_PREFLIGHT", "pre-generation seal does not bind this manifest")
    prov_sha = sha256_file(provenance_path)
    if seal.get("run_provenance_sha256") != prov_sha:
        raise OrchestrationError(
            "O0_PREFLIGHT", "pre-generation seal does not bind RUN_PROVENANCE")
    pregen_sha = sha256_file(pregeneration_seal_path)

    paths = verify_artifact_paths(man.raw, artifact_paths_path)
    from verify_axis_artifact import verify as verify_axis
    axis_report = verify_axis(axis_package_path)
    if axis_report.get("verdict") != "SEALABLE":
        raise OrchestrationError(
            "O0_PREFLIGHT", "axis package did not verify as SEALABLE")

    structured_axes = load_axis_tensor(
        paths["artifact_hashes.structured_axis_tensor"],
        man.get("artifact_hashes.structured_axis_tensor"), "structured")
    random_axes = load_axis_tensor(
        paths["artifact_hashes.random_axis_tensor"],
        man.get("artifact_hashes.random_axis_tensor"), "random")

    from o1_analysis import load_task_manifest, validate_seed_matrix
    tasks = load_task_manifest(task_manifest_path)
    with open(seed_matrix_path) as fh:
        seed_matrix = json.load(fh)
    validate_seed_matrix(seed_matrix, tasks, man)

    arms = list(man.get("bank.arms_confirmatory"))
    for arm in arms:
        if arm not in ARM_GEOMETRY:
            raise OrchestrationError("O0_PREFLIGHT", f"unknown sealed arm {arm!r}")

    progress = _Progress(progress_path or output_path + ".progress.json")
    cache = _BoundaryCache(boundary_cache_dir or output_path + ".boundaries")
    sink = _RecordSink(output_path, "pregeneration_seal_sha256", pregen_sha)
    existing_rows: dict[tuple, dict] = {}
    if sink.done:
        for row in load_jsonl(output_path):
            existing_rows[_row_key(row)] = row

    if backend is None:
        backend = RealBackend(checkpoint, man.raw)

    def common(task: dict) -> dict:
        return {
            "task_id": task["task_id"],
            "stratum": task["stratum"],
            "task_content_sha256": task["task_content_sha256"],
            "generator_setting_id": task["generator_setting_id"],
            "run_provenance_sha256": prov_sha,
            "pregeneration_seal_sha256": pregen_sha,
        }

    n_generated = 0
    for task in tasks:
        rank = int(task["sealed_rank_within_stratum"]) % BANK_SIZE
        seeds = {i: int(seed_matrix[task["task_id"]][str(i)])
                 for i in range(BANK_SIZE)}

        def base_row(i: int, seed: int, branch: dict) -> dict:
            fields = _score_fields(task, branch)
            fields.pop("finish_reason", None)
            fields.pop("logits_finite", None)
            fields.pop("catastrophic_repetition", None)
            return {**common(task), "arm": "baseline", "alpha": 0.0,
                    "direction_id": None, "sign": None, "stream_index": i,
                    "seed": seed, "downstream_delta_rms": None,
                    "R": int(fields["well_formed"] and fields["verifier_correct"]),
                    **fields}

        need_any = False
        for arm in arms:
            alpha = man.arm_alpha(arm)
            for i in range(BANK_SIZE):
                if (task["task_id"], arm, f"{float(alpha):.12g}", i) not in sink.done:
                    need_any = True
        if not need_any:
            continue

        replay = {i: existing_rows.get((task["task_id"], "baseline", "0", i))
                  for i in range(BANK_SIZE)}
        h_base = _run_baseline_bank(
            backend, task, seeds, sink, cache, base_row, replay)

        for arm in arms:
            if arm == "baseline":
                continue
            alpha = float(man.arm_alpha(arm))
            is_intervention, geometry = ARM_GEOMETRY[arm]
            if geometry is None:
                # zero_alpha: full sham bank, one row per stream, no geometry.
                for i in range(BANK_SIZE):
                    key = (task["task_id"], arm, f"{alpha:.12g}", i)
                    if key in sink.done:
                        continue
                    branch = backend.generate(task, seeds[i], None, 0.0)
                    rho = downstream_delta_rms(
                        _boundary_of(branch), h_base, 0.0, is_zero_alpha=True)
                    fields = _score_fields(task, branch)
                    fields.pop("finish_reason", None)
                    fields.pop("logits_finite", None)
                    fields.pop("catastrophic_repetition", None)
                    sink.write({
                        **common(task), "arm": arm, "alpha": 0.0,
                        "direction_id": None, "sign": None, "stream_index": i,
                        "seed": seeds[i], "downstream_delta_rms": rho,
                        "R": int(fields["well_formed"] and fields["verifier_correct"]),
                        **fields})
                    n_generated += 1
                continue
            bank = structured_axes if geometry == "structured" else random_axes
            for d, s in SIGNED_ACTIONS:
                stream = (action_index(d, s) + rank) % BANK_SIZE
                key = (task["task_id"], arm, f"{alpha:.12g}", stream)
                if key in sink.done:
                    continue
                branch = backend.generate(task, seeds[stream], bank[d - 1], s * alpha)
                rho = downstream_delta_rms(
                    _boundary_of(branch), h_base, float(branch["injected_rms"]))
                fields = _score_fields(task, branch)
                fields.pop("finish_reason", None)
                fields.pop("logits_finite", None)
                fields.pop("catastrophic_repetition", None)
                sink.write({
                    **common(task), "arm": arm, "alpha": alpha,
                    "direction_id": d, "sign": s, "stream_index": stream,
                    "seed": seeds[stream], "downstream_delta_rms": rho,
                    "R": int(fields["well_formed"] and fields["verifier_correct"]),
                    **fields})
                n_generated += 1
                if n_generated % flush_every_rows == 0:
                    progress.flush({})
        progress.flush({})

    total_expected = len(tasks) * BANK_SIZE * len(arms)
    if len(sink.done) != total_expected:
        raise OrchestrationError(
            "O7_COMPLETENESS",
            f"{len(sink.done)} rows present, {total_expected} sealed rows "
            f"expected — resume until complete")
    sink.close()
    final_rows = load_jsonl(output_path)
    verify_token_text_binding(final_rows, backend.decode)
    verify_stored_scoring(final_rows, {t["task_id"]: t for t in tasks})
    progress.flush({"complete": True})
    return {
        "records": output_path,
        "records_sha256": sha256_file(output_path),
        "n_rows": len(sink.done),
        "elapsed_wall_seconds": progress.elapsed(),
    }


# ==========================================================================
# CLI
# ==========================================================================


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("calibration")
    c.add_argument("--manifest-design", required=True)
    c.add_argument("--artifact-paths", required=True)
    c.add_argument("--precommit", required=True)
    c.add_argument("--calibration-task-manifest", required=True)
    c.add_argument("--axis-package", required=True)
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--metadata-output", required=True)
    c.add_argument("--progress")
    c.add_argument("--boundary-cache-dir")

    f = sub.add_parser("confirmatory")
    f.add_argument("--manifest", required=True)
    f.add_argument("--artifact-paths", required=True)
    f.add_argument("--confirmatory-task-manifest", required=True)
    f.add_argument("--seed-matrix", required=True)
    f.add_argument("--run-provenance", required=True)
    f.add_argument("--pregeneration-seal", required=True)
    f.add_argument("--axis-package", required=True)
    f.add_argument("--checkpoint", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--progress")
    f.add_argument("--boundary-cache-dir")

    a = p.parse_args()
    if a.mode == "calibration":
        out = orchestrate_calibration(
            a.manifest_design, a.artifact_paths, a.precommit,
            a.calibration_task_manifest, a.axis_package, a.checkpoint,
            a.output, a.metadata_output, a.progress, a.boundary_cache_dir)
    else:
        out = orchestrate_confirmatory(
            a.manifest, a.artifact_paths, a.confirmatory_task_manifest,
            a.seed_matrix, a.run_provenance, a.pregeneration_seal,
            a.axis_package, a.checkpoint, a.output, a.progress,
            a.boundary_cache_dir)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
