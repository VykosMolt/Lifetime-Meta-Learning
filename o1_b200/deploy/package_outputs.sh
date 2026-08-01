#!/usr/bin/env bash
# Output packaging for the future B200 session: bundles run records, logs,
# reports, and environment/benchmark artifacts into a checksummed archive for
# transfer back. Never includes the model checkpoint or any secret material.
set -euo pipefail

OUT_DIR="${1:?usage: package_outputs.sh <run_output_dir> <archive_path>}"
ARCHIVE="${2:?usage: package_outputs.sh <run_output_dir> <archive_path>}"

if [ ! -d "$OUT_DIR" ]; then
  echo "FATAL: $OUT_DIR is not a directory" >&2
  exit 1
fi

# refuse secret-looking files
if find "$OUT_DIR" -type f \( -name "*.pem" -o -name "*.key" -o -name "id_rsa*" \
     -o -name "*credentials*" -o -name "*.env" -o -name "*token*" \) | grep -q .; then
  echo "FATAL: secret-looking files present in $OUT_DIR; refusing" >&2
  exit 1
fi

( cd "$OUT_DIR" && find . -type f | sort | xargs sha256sum ) > "$OUT_DIR/SHA256SUMS.outputs"
tar --sort=name --owner=0 --group=0 --numeric-owner \
    --mtime='UTC 2020-01-01' -czf "$ARCHIVE" -C "$OUT_DIR" .
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
echo "packaged: $ARCHIVE"
cat "$ARCHIVE.sha256"
