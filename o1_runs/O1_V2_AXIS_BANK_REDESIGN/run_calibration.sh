#!/usr/bin/env bash
# Launch or resume the sealed O1 v2.1 calibration. Idempotent: the sealed
# orchestrator appends only missing rows and refuses to mix runs.
set -euo pipefail

RUN=/home/moloch/ouro_worktrees/o1-v2-axis-bank-redesign/o1_runs/O1_V2_AXIS_BANK_REDESIGN
PKG=/home/moloch/ouro_worktrees/o1-v2-axis-bank-redesign/o1_packages/O1_oracle_reachability_v2.1.0_source/o1_v210
PY=/home/moloch/ouro_project/venv/bin/python

cd "$RUN"
export PYTHONDONTWRITEBYTECODE=1

# The sealed orchestrator asserts the deterministic environment BEFORE it
# imports anything that could set it, so it fails closed on a machine where
# the operator forgot. Set the sealed values here; the orchestrator verifies
# them against model.deterministic_flags in the precommitted manifest.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec "$PY" "$PKG/run_o1_v2_orchestrator.py" calibration \
  --manifest-design "$RUN/FREEZE_MANIFEST.precalibration.json" \
  --artifact-paths "$RUN/RUNTIME_ARTIFACT_PATHS.json" \
  --precommit "$RUN/CALIBRATION_PRECOMMIT.json" \
  --calibration-task-manifest "$RUN/COHORTS/calibration_tasks.jsonl" \
  --axis-package "$RUN/AXIS_PACKAGE_V2" \
  --checkpoint /home/moloch/ouro_project/models/ouro_rltt_local \
  --output "$RUN/calibration_records.jsonl" \
  --metadata-output "$RUN/CALIBRATION_METADATA.json" \
  --progress "$RUN/calibration_progress.json" \
  --boundary-cache-dir "$RUN/calibration_boundaries"
