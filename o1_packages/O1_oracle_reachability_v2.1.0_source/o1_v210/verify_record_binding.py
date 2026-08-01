#!/usr/bin/env python3
"""Independently re-decode sealed records and verify the token/text binding.

The analysis core is stdlib+numpy and cannot re-decode token ids, so within
`o1_analysis` / `calibration_analysis` a token list is bound only to its own
digest.  Token ids feed the mechanical repetition flag, which feeds the
coherence criteria, which select `alpha_star` — a red-team attack showed that
forged tokens carrying byte-identical text could therefore move the sealed
intervention magnitude and survive the full calibration seal.

This command closes that loop with the sealed tokenizer: every row's
`generated_token_ids` must decode exactly to its `generated_text`, both
digests must reproduce, and (with `--task-manifest`) `well_formed`,
`verifier_correct` and the repetition flag must reproduce under the sealed
parser, verifier and mechanical flag.  It is run by the sealed orchestrator at
finalization and is independently re-runnable by any auditor afterwards.

    python verify_record_binding.py \
      --records calibration_records.jsonl \
      --checkpoint <checkpoint_dir> \
      --task-manifest calibration_tasks.jsonl \
      --report RECORD_BINDING_REPORT.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from o1_answer_parser_v2 import has_catastrophic_repetition, parse_final_answer
from o1_analysis import load_jsonl, sha256_file, text_sha256, token_ids_sha256
from o1_truth_table_verifier_v2 import verify_letter


class RecordBindingError(RuntimeError):
    pass


def verify_records(records_path: str, checkpoint: str,
                   task_manifest_path: str | None = None) -> dict:
    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__ != "4.54.1":
        raise RecordBindingError(
            f"transformers must equal 4.54.1, got {transformers.__version__}")
    tok = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True)
    rows = load_jsonl(records_path)
    if not rows:
        raise RecordBindingError("record set is empty")

    tasks: dict[str, dict] = {}
    if task_manifest_path:
        for t in load_jsonl(task_manifest_path):
            tasks[t["task_id"]] = t

    n_scored = 0
    for i, row in enumerate(rows):
        toks = [int(t) for t in row["generated_token_ids"]]
        text = row["generated_text"]
        decoded = tok.decode(toks, skip_special_tokens=True)
        if decoded != text:
            raise RecordBindingError(
                f"row {i} (task {row.get('task_id')} arm {row.get('arm')}): "
                f"tokens do not decode to generated_text")
        if text_sha256(text) != row["generated_text_sha256"]:
            raise RecordBindingError(f"row {i}: generated_text_sha256 disagrees")
        if token_ids_sha256(toks) != row["generated_token_ids_sha256"]:
            raise RecordBindingError(
                f"row {i}: generated_token_ids_sha256 disagrees")
        if len(toks) != int(row["n_generated_tokens"]):
            raise RecordBindingError(f"row {i}: n_generated_tokens disagrees")
        if "catastrophic_repetition" in row:
            if bool(row["catastrophic_repetition"]) != has_catastrophic_repetition(toks):
                raise RecordBindingError(
                    f"row {i}: catastrophic_repetition disagrees with the "
                    f"mechanical flag recomputed from token ids")
        if tasks:
            task = tasks.get(row["task_id"])
            if task is None:
                raise RecordBindingError(
                    f"row {i}: task {row['task_id']!r} absent from the manifest")
            parsed = parse_final_answer(text)
            if bool(row["well_formed"]) != (parsed is not None):
                raise RecordBindingError(f"row {i}: well_formed disagrees")
            if bool(row["verifier_correct"]) != verify_letter(task, parsed):
                raise RecordBindingError(f"row {i}: verifier_correct disagrees")
            if "parsed_answer" in row and row.get("parsed_answer") != parsed:
                raise RecordBindingError(f"row {i}: parsed_answer disagrees")
            n_scored += 1

    return {
        "schema_version": "o1.record_binding.v1",
        "records": os.path.basename(records_path),
        "records_sha256": sha256_file(records_path),
        "n_rows": len(rows),
        "n_rows_rescored": n_scored,
        "tokenizer_source": checkpoint,
        "transformers": transformers.__version__,
        "verdict": "TOKEN_TEXT_BINDING_VERIFIED",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task-manifest")
    p.add_argument("--report")
    a = p.parse_args()
    out = verify_records(a.records, a.checkpoint, a.task_manifest)
    if a.report:
        if os.path.exists(a.report):
            raise RecordBindingError(f"refusing to overwrite {a.report}")
        with open(a.report, "x") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
