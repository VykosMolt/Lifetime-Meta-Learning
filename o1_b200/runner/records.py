"""Canonical B200 record schema, construction, and full recomputation.

Every backend emits exactly this schema.  The scientific core is a strict
superset of the sealed v2.1 calibration row (same field names and values), so
sealed analysis code reads these records unchanged; every added field is
descriptive backend metadata and none of it may alter scientific analysis.

Volatile fields (timestamps, worker/batch identity, backend id, environment
digest) are excluded from cross-backend equivalence comparison but ARE bound
by the record hash, so a record cannot be silently edited after commit.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from . import sealed_import
from .identity import (
    canonical_json, domain_sha256, text_sha256, token_ids_sha256,
)
from .row_specs import RowSpec, row_id_of

sealed_import.ensure_sealed_path()

from o1_answer_parser_v2 import (  # noqa: E402 (sealed)
    has_catastrophic_repetition, parse_final_answer,
)
from o1_truth_table_verifier_v2 import verify_letter  # noqa: E402 (sealed)

RECORD_HASH_DOMAIN = "o1b200.record.v1"

# Scientific core: field names identical to the sealed v2.1 calibration row.
SCIENTIFIC_FIELDS = (
    "task_id", "task_content_sha256", "generator_setting_id", "arm", "alpha",
    "direction_id", "sign", "stream_index", "seed",
    "generated_token_ids", "generated_text", "generated_text_sha256",
    "generated_token_ids_sha256", "n_generated_tokens", "hit_max_tokens",
    "finish_reason", "parsed_answer", "well_formed", "verifier_correct",
    "logits_finite", "catastrophic_repetition",
    "downstream_delta_rms", "injected_rms", "R",
)

# Runner identity/binding fields (deterministic; part of equivalence).
BINDING_FIELDS = (
    "row_id", "exec_index", "bank", "prompt_sha256", "prompt_token_ids_sha256",
    "intended_alpha", "binding_key", "binding_sha256",
    "package_zip_sha256", "source_commit", "axis_package_sha256",
    "model_artifact_sha256", "tokenizer_sha256", "row_spec_manifest_sha256",
)

# Volatile descriptive fields (excluded from equivalence, included in hash).
VOLATILE_FIELDS = (
    "backend_id", "worker_id", "batch_id", "batch_size",
    "runtime_environment_sha256", "started_utc", "finished_utc",
    "realized_delta_rms",
)

ALL_FIELDS = SCIENTIFIC_FIELDS + BINDING_FIELDS + VOLATILE_FIELDS + ("record_hash",)


class RecordError(RuntimeError):
    pass


def record_hash(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "record_hash"}
    return domain_sha256(RECORD_HASH_DOMAIN, body)


def build_record(spec: RowSpec, gen: dict, *, rho: float | None,
                 prompt_token_ids: list[int], backend_id: str,
                 worker_id: str, batch_id: str, batch_size: int,
                 artifact_ids: dict, runtime_environment_sha256: str,
                 row_spec_manifest_sha256: str,
                 started_utc: str, finished_utc: str,
                 realized_delta_rms: float | None) -> dict:
    """Assemble the canonical record from a sealed generation result."""
    text = gen["generated_text"]
    toks = [int(t) for t in gen["generated_token_ids"]]
    parsed = parse_final_answer(text)
    if bool(gen.get("well_formed")) != (parsed is not None):
        raise RecordError("generator well_formed disagrees with sealed parser")
    well_formed = parsed is not None
    verifier_correct = bool(gen["verifier_correct"])
    rec = {
        # scientific core (sealed v2.1 names/values)
        "task_id": spec.task_id,
        "task_content_sha256": spec.task_content_sha256,
        "generator_setting_id": spec.generator_setting_id,
        "arm": spec.arm,
        "alpha": float(spec.alpha),
        "direction_id": spec.direction_id,
        "sign": spec.sign,
        "stream_index": spec.stream_index,
        "seed": spec.seed,
        "generated_token_ids": toks,
        "generated_text": text,
        "generated_text_sha256": text_sha256(text),
        "generated_token_ids_sha256": token_ids_sha256(toks),
        "n_generated_tokens": len(toks),
        "hit_max_tokens": gen["finish_reason"] == "max_new_tokens",
        "finish_reason": gen["finish_reason"],
        "parsed_answer": parsed,
        "well_formed": well_formed,
        "verifier_correct": verifier_correct,
        "logits_finite": bool(gen["logits_finite"]),
        "catastrophic_repetition": has_catastrophic_repetition(toks),
        "downstream_delta_rms": rho,
        "injected_rms": float(gen["injected_rms"]),
        "R": int(well_formed and verifier_correct),
        # binding
        "row_id": spec.row_id,
        "exec_index": spec.exec_index,
        "bank": spec.bank,
        "prompt_sha256": spec.prompt_sha256,
        "prompt_token_ids_sha256": token_ids_sha256(
            [int(t) for t in prompt_token_ids]),
        "intended_alpha": float(spec.alpha),
        "binding_key": spec.binding_key,
        "binding_sha256": spec.binding_sha256,
        "package_zip_sha256": artifact_ids["package_zip_sha256"],
        "source_commit": artifact_ids["source_commit"],
        "axis_package_sha256": artifact_ids["axis_package_sha256"],
        "model_artifact_sha256": artifact_ids["model_artifact_sha256"],
        "tokenizer_sha256": artifact_ids["tokenizer_sha256"],
        "row_spec_manifest_sha256": row_spec_manifest_sha256,
        # volatile
        "backend_id": backend_id,
        "worker_id": worker_id,
        "batch_id": batch_id,
        "batch_size": int(batch_size),
        "runtime_environment_sha256": runtime_environment_sha256,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "realized_delta_rms": realized_delta_rms,
    }
    rec["record_hash"] = record_hash(rec)
    return rec


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def scientific_core(record: dict) -> dict:
    return {k: record[k] for k in SCIENTIFIC_FIELDS}


def deterministic_view(record: dict) -> dict:
    return {k: record[k] for k in SCIENTIFIC_FIELDS + BINDING_FIELDS}


def verify_record(record: dict, *, task: dict, spec: RowSpec | None,
                  expected_artifact_ids: dict | None,
                  decode: Callable[[list[int]], str] | None = None) -> None:
    """Full recomputation.  Stored values may never override recomputed ones.

    Recomputes: schema completeness, row identity, action/seed mapping (via
    the RowSpec), text/token digests, parser result, well_formed,
    verifier_correct, R, artifact/package identity, and the record hash.
    Raises RecordError on the first discrepancy.
    """
    missing = [k for k in ALL_FIELDS if k not in record]
    extra = sorted(set(record) - set(ALL_FIELDS))
    if missing or extra:
        raise RecordError(f"schema mismatch: missing={missing} extra={extra}")
    if record_hash(record) != record["record_hash"]:
        raise RecordError("record_hash does not match record content")
    # identity recomputation
    identity = {
        "task_id": record["task_id"],
        "task_content_sha256": record["task_content_sha256"],
        "prompt_sha256": record["prompt_sha256"],
        "generator_setting_id": record["generator_setting_id"],
        "bank": record["bank"],
        "arm": record["arm"],
        "direction_id": record["direction_id"],
        "sign": record["sign"],
        "alpha": f"{float(record['alpha']):.12g}",
        "stream_index": record["stream_index"],
        "seed": record["seed"],
        "binding_key": record["binding_key"],
        "binding_sha256": record["binding_sha256"],
    }
    if row_id_of(identity) != record["row_id"]:
        raise RecordError("row_id does not match the row identity fields")
    if spec is not None:
        if spec.row_id != record["row_id"]:
            raise RecordError("record row_id disagrees with its RowSpec")
        if spec.exec_index != record["exec_index"]:
            raise RecordError("exec_index disagrees with the RowSpec")
        if spec.seed != record["seed"] or spec.stream_index != record["stream_index"]:
            raise RecordError("seed/stream mapping disagrees with the RowSpec")
        if (spec.arm != record["arm"] or spec.direction_id != record["direction_id"]
                or spec.sign != record["sign"]
                or float(spec.alpha) != float(record["alpha"])):
            raise RecordError("action mapping disagrees with the RowSpec")
    if float(record["intended_alpha"]) != float(record["alpha"]):
        raise RecordError("intended_alpha disagrees with alpha")
    # content digests
    toks = [int(t) for t in record["generated_token_ids"]]
    if token_ids_sha256(toks) != record["generated_token_ids_sha256"]:
        raise RecordError("generated_token_ids_sha256 does not match tokens")
    if text_sha256(record["generated_text"]) != record["generated_text_sha256"]:
        raise RecordError("generated_text_sha256 does not match text")
    if record["n_generated_tokens"] != len(toks):
        raise RecordError("n_generated_tokens does not match tokens")
    if decode is not None and decode(toks) != record["generated_text"]:
        raise RecordError("generated_token_ids do not decode to generated_text")
    # scoring recomputation (sealed parser/verifier are authoritative)
    if record["task_content_sha256"] != task["task_content_sha256"]:
        raise RecordError("record task_content_sha256 disagrees with the task")
    parsed = parse_final_answer(record["generated_text"])
    if record["parsed_answer"] != parsed:
        raise RecordError("stored parsed_answer disagrees with sealed parser")
    if bool(record["well_formed"]) != (parsed is not None):
        raise RecordError("stored well_formed disagrees with sealed parser")
    vc = verify_letter(task, parsed)
    if bool(record["verifier_correct"]) != vc:
        raise RecordError("stored verifier_correct disagrees with sealed verifier")
    if int(record["R"]) != int((parsed is not None) and vc):
        raise RecordError("stored R disagrees with recomputed R")
    if bool(record["catastrophic_repetition"]) != has_catastrophic_repetition(toks):
        raise RecordError("stored repetition flag disagrees with recomputation")
    if bool(record["hit_max_tokens"]) != (record["finish_reason"] == "max_new_tokens"):
        raise RecordError("hit_max_tokens disagrees with finish_reason")
    # artifact identity
    if expected_artifact_ids is not None:
        for key in ("package_zip_sha256", "source_commit", "axis_package_sha256",
                    "model_artifact_sha256", "tokenizer_sha256"):
            if record[key] != expected_artifact_ids[key]:
                raise RecordError(f"{key} disagrees with the expected identity")
    # transport surface constraints (values are checked by sealed transport
    # at generation; here we enforce the arm-level shape)
    if record["arm"] == "baseline":
        if record["downstream_delta_rms"] is not None:
            raise RecordError("baseline row must carry null transport")
    elif record["arm"] == "zero_alpha":
        if record["downstream_delta_rms"] != 0.0:
            raise RecordError("zero_alpha row must carry transport 0.0")
    else:
        rho = record["downstream_delta_rms"]
        if not isinstance(rho, (int, float)) or not (rho >= 0.0):
            raise RecordError("intervention row transport must be finite >= 0")
