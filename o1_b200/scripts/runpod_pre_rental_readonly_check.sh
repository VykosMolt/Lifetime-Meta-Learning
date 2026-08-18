#!/usr/bin/env bash
# Read-only RunPod pre-rental check. GET-only by transport construction.
# Without RUNPOD_API_KEY: runs all local checks, reports live portion SKIPPED.
# Never creates a Pod. Never issues POST/PATCH/PUT/DELETE.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${O1_B200_PYTHON:-/home/moloch/ouro_project/venv/bin/python}"
OUT_DIR="${1:-$ROOT/o1_b200/reports}"
mkdir -p "$OUT_DIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT"

echo "[readonly-check] local pinned-schema + adapter unit tests (hermetic:"
echo "[readonly-check]  operator credentials stripped from the test processes)"
env -u RUNPOD_API_KEY -u RUNPOD_API_KEY_FILE \
  "$PY" "$ROOT/o1_b200/tests/test_runpod_adapter.py"
env -u RUNPOD_API_KEY -u RUNPOD_API_KEY_FILE \
  "$PY" "$ROOT/o1_b200/tests/test_runpod_interlock.py"

echo "[readonly-check] live read-only preflight (GET-only)"
"$PY" -m o1_b200.provider.runpod.preflight \
  --out "$OUT_DIR/RUNPOD_READONLY_PREFLIGHT.json"
