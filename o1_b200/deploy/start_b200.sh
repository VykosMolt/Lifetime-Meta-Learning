#!/usr/bin/env bash
# Provider-neutral launch entrypoint for the future B200 session.
# DOES NOTHING SCIENTIFIC BY ITSELF: it validates, then hands control to the
# zero-touch state machine, which enforces every gate (artifact hashes,
# environment, non-O1 equivalence, benchmark, frozen backend selection,
# external precommit verification, affordability) before calibration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${O1_B200_PYTHON:-/opt/venv/bin/python}"
OUT="${O1_B200_OUT:-/outputs}"

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="$ROOT"

mkdir -p "$OUT"

echo "[start_b200] environment validation"
"$PY" "$ROOT/o1_b200/deploy/validate_environment.py" --out "$OUT/environment_report.local.json"

echo "[start_b200] artifact verification"
"$PY" "$ROOT/o1_b200/deploy/verify_artifacts.py" \
  --manifest "${O1_B200_TRANSFER_MANIFEST:-$ROOT/o1_b200/deploy/TRANSFER_MANIFEST.json}"

echo "[start_b200] handing over to the zero-touch state machine"
echo "REFUSED: the production state-machine launch requires a selected"
echo "provider adapter and a populated budget policy; neither exists yet."
echo "This entrypoint intentionally stops here until the B200-access task"
echo "implements the selected provider adapter (runbook step 2)."
exit 64
