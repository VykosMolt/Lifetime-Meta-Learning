#!/usr/bin/env bash
# Zero-touch RunPod B200 launcher — FOR THE LATER, SEPARATELY AUTHORIZED
# SESSION ONLY. Refuses before any mutating network call unless the complete
# live-mutation interlock is satisfied:
#   * --authorization <B300_RENTAL_AUTHORIZATION.json> (a REAL file, not the
#     template — templates fail validation by construction)
#   * RUNPOD_ALLOW_BILLABLE_MUTATIONS=YES_I_AUTHORIZE_THIS_RUN
#   * --execute-authorized-rental
#   * matching project/package/provider/budget/deployment identities,
#     unexpired, unused nonce.
# It then runs, with no further human interaction:
#   fresh quote -> pod request render + hash check -> create (reconciled,
#   duplicate-safe) -> watchdogs -> zero-touch state machine (hardware
#   validation, non-O1 equivalence, bounded benchmark, backend selection,
#   replacement precommit, external commit verification, affordability gate,
#   only then calibration) -> verified result transfer -> TERMINATE ->
#   confirm -> local diagnostics package.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${O1_B200_PYTHON:-/home/moloch/ouro_project/venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT"

AUTH=""
EXECUTE=""
OUT="$ROOT/o1_b200/reports/rental_session"
while [ $# -gt 0 ]; do
  case "$1" in
    --authorization) AUTH="$2"; shift 2;;
    --execute-authorized-rental) EXECUTE="--execute-authorized-rental"; shift;;
    --out) OUT="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

if [ -z "$AUTH" ]; then
  echo "REFUSED: --authorization <B300_RENTAL_AUTHORIZATION.json> is required." >&2
  echo "LIVE_MUTATION_NOT_AUTHORIZED" >&2
  exit 78
fi

exec "$PY" -m o1_b200.provider.runpod.zero_touch \
  --authorization "$AUTH" --out "$OUT" ${EXECUTE:+$EXECUTE}
