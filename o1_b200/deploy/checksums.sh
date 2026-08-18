#!/usr/bin/env bash
# Recompute and verify the package SHA256SUMS (both directions).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-verify}" in
  write)
    # Excludes regenerable per-run scratch (the same set the release
    # packager omits), so the manifest describes exactly the shipped tree.
    ( cd "$ROOT" && find . -type f \
        ! -name SHA256SUMS ! -path "./reports/local_runs/*" \
        ! -path "./reports/rehearsal_calibration/*" \
        ! -path "./reports/downloaded/*" ! -path "./reports/eq_*/*" \
        ! -name "*.pyc" ! -path "*/__pycache__/*" \
        | sed 's|^\./||' | sort \
        | while read -r f; do sha256sum "$f"; done ) > "$ROOT/SHA256SUMS"
    echo "wrote $ROOT/SHA256SUMS ($(wc -l < "$ROOT/SHA256SUMS") files)"
    ;;
  verify)
    ( cd "$ROOT" && sha256sum --quiet -c SHA256SUMS ) && echo "SHA256SUMS OK"
    ;;
  *)
    echo "usage: checksums.sh [write|verify]" >&2; exit 2;;
esac
