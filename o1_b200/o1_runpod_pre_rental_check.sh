#!/usr/bin/env bash
# RunPod pre-rental master check. Runs every local suite (base package, B200
# runner, RunPod adapter/mock/lifecycle/termination/budget/interlock/
# zero-touch rehearsal), schema pin, container record, artifact manifest,
# secret scan, dry-run request render, and — only if RUNPOD_API_KEY exists —
# the GET-only live preflight. Emits RUNPOD_PRE_RENTAL_READINESS.{json,md}.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${O1_B200_PYTHON:-/home/moloch/ouro_project/venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT"
exec "$PY" -m o1_b200.provider.runpod.pre_rental_check "$@"
