#!/usr/bin/env bash
# Pod-side entry for the COMBINED O1 + Foundation Learner session.
#
# This script starts the FL session supervisor, which runs the O1 pod-side
# entry as an OPAQUE configured subprocess FIRST, verifies and transfers the O1
# records, closes the O1 process, and only then runs the FL ladder
# (contract 13).  It NEVER modifies any O1 file, never imports an O1 module,
# and never writes under an O1 root.
#
# It refuses to start a real session whose configuration still carries
# UNRESOLVED fields.  The historical o1_entry_command gap is CLOSED: the
# O1 pod entry is the real production zero-touch entry
# (o1_b200/deploy/start_b300.sh); the operator may override it in the session
# config.  A DRESS REHEARSAL may proceed with unresolved
# fields and is loudly labelled DRESS_REHEARSAL in every artefact it writes.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: fl_b200_entry.sh --config <session_config.json> [--out <dir>] [--no-resume]

  --config   flb200.session_config.v1 JSON (required)
  --out      session output directory (default $FL_B200_OUT or
             /workspace/foundation_learner/session)
  --no-resume  ignore an existing journal and start from START_SESSION
USAGE
}

CONFIG=""
OUT="${FL_B200_OUT:-/workspace/foundation_learner/session}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:-}"; shift 2 ;;
    --out)    OUT="${2:-}"; shift 2 ;;
    --no-resume) EXTRA+=("--no-resume"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[fl_b200_entry] unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "[fl_b200_entry] REFUSED: --config is required" >&2
  usage
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[fl_b200_entry] REFUSED: session config not found: $CONFIG" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${FL_B200_PYTHON:-/opt/venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "[fl_b200_entry] REFUSED: python interpreter not executable: $PY" >&2
  echo "  (the FL package runs INSIDE the existing O1 container venv /opt/venv;" >&2
  echo "   set FL_B200_PYTHON to override for a local rehearsal)" >&2
  exit 2
fi

# Same runtime environment as the O1 container declares; nothing is installed.
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Pre-flight refusal on UNRESOLVED fields, BEFORE any work starts.  A rehearsal
# config (rehearsal=true) is exempt and is announced loudly.
"$PY" - "$CONFIG" <<'PYEOF'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    cfg = json.load(fh)
if cfg.get("schema") != "flb200.session_config.v1":
    raise SystemExit(f"[fl_b200_entry] REFUSED: {path} is not flb200.session_config.v1")

def unresolved(obj, prefix=""):
    out = []
    if isinstance(obj, str):
        if obj.startswith("UNRESOLVED"):
            out.append(prefix or "<root>")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += unresolved(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += unresolved(v, f"{prefix}[{i}]")
    return out

bad = unresolved(cfg)
if bad and not cfg.get("rehearsal", False):
    raise SystemExit(
        "[fl_b200_entry] REFUSED: unresolved session-config fields "
        f"{bad}. Bind every one of them before a real session.")
if bad:
    print("*** DRESS_REHEARSAL: unresolved fields present:", bad, "***",
          file=sys.stderr)
PYEOF

echo "[fl_b200_entry] O1 has absolute priority; FL runs only after O1 close."
echo "[fl_b200_entry] config: $CONFIG"
echo "[fl_b200_entry] out:    $OUT"
mkdir -p "$OUT"

# Episode-corpus ingestion.  The ~480 MB pregen tree is neither in Git nor in
# the image, and campaign/entry.py refuses a session whose pregen_root is
# absent — so this must happen before the supervisor starts, not when the
# ladder reaches for its first shard.  Every shard is hashed against
# SHARD_SUMS.json inside the fetch; a failure refuses here, still cheap.
# Credential scope BEFORE the corpus download: one HF_TOKEN must cover the
# pregen read and the durable-mirror write.  A read-only token trains for
# hours and then loses the journal and every checkpoint on eviction.
echo "[fl_b200_entry] credential scope preflight"
"$PY" -m foundation_learner.campaign.check_hf_scope \
  --config "$CONFIG" --out "$OUT/HF_SCOPE_REPORT.json"

echo "[fl_b200_entry] pregen ingestion (the episode corpus is not baked in)"
"$PY" -m foundation_learner.campaign.fetch_pregen \
  --config "$CONFIG" --out "$OUT/PREGEN_FETCH_REPORT.json"

exec "$PY" -m foundation_learner.campaign.session_supervisor \
  --config "$CONFIG" --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
