#!/usr/bin/env bash
# Build the deterministic O1_B300_RUNNER release archive.
#
# Deterministic by construction: sorted entry order, fixed timestamps, fixed
# permissions, no compression-level drift — so two builds of the same tree
# produce byte-identical archives with the same SHA-256.  Generated run
# outputs under reports/local_runs and every __pycache__ are excluded; the
# top-level reports (the validation evidence) are included.
#
# Historical packages are NEVER overwritten: the script refuses if the
# target archive already exists.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="$(cat "$ROOT/o1_b200/VERSION")"
NAME="${1:-O1_B300_RUNNER_v${VERSION}_RUNPOD_PRERENTAL}"
OUT="$ROOT/o1_packages/${NAME}.zip"

if [[ -e "$OUT" ]]; then
  echo "REFUSED: $OUT already exists; validated historical packages are" >&2
  echo "         never overwritten. Bump VERSION or pass a new name." >&2
  exit 3
fi

cd "$ROOT"
# refresh the package checksum manifest first: it ships inside the archive
./o1_b200/deploy/checksums.sh write >/dev/null

python3 - "$OUT" <<'PYEOF'
import os, stat, sys, zipfile

out = sys.argv[1]
root = os.getcwd()
FIXED_DATE = (1980, 1, 1, 0, 0, 0)

#: Regenerable per-run working directories under reports/ — scratch, not
#: evidence.  The evidence is the top-level *_REPORT.json files.
SCRATCH_REPORT_DIRS = ("local_runs", "rehearsal_calibration", "downloaded")


def included(path: str) -> bool:
    parts = path.split(os.sep)
    if "__pycache__" in parts or path.endswith(".pyc"):
        return False
    if parts[:2] == ["o1_b200", "reports"] and len(parts) > 3:
        sub = parts[2]
        if sub in SCRATCH_REPORT_DIRS or sub.startswith("eq_"):
            return False
    return True

files = []
for dirpath, dirnames, filenames in os.walk(os.path.join(root, "o1_b200")):
    dirnames.sort()
    for name in sorted(filenames):
        rel = os.path.relpath(os.path.join(dirpath, name), root)
        if included(rel):
            files.append(rel)
files.sort()

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for rel in files:
        info = zipfile.ZipInfo(rel, date_time=FIXED_DATE)
        mode = os.stat(rel).st_mode
        executable = bool(mode & stat.S_IXUSR)
        info.external_attr = (0o755 if executable else 0o644) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(rel, "rb") as fh:
            zf.writestr(info, fh.read())
print(f"{len(files)} files -> {out}")
PYEOF

( cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256" )
echo "archive: $OUT"
cat "$OUT.sha256"
