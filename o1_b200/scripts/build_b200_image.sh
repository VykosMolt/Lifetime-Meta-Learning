#!/usr/bin/env bash
# Build the production B200 image locally (linux/amd64), record every
# environment fact, and emit CONTAINER_IMAGE_RECORD.json with the immutable
# image digest. Never embeds credentials; never includes the checkpoint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHEELS_SRC="${O1_B200_WHEELS:-/home/moloch/b200_build_cache/wheels}"
IMAGE_NAME="${1:-o1-b200-runner}"
VERSION_TAG="${2:-v0.2.0}"
OUT="$ROOT/o1_b200/provider/runpod/CONTAINER_IMAGE_RECORD.json"

cd "$ROOT"
rm -rf build_ctx && mkdir -p build_ctx/wheels
cp "$WHEELS_SRC"/torch-2.12.0.dev20260408+cu128-cp314-cp314-*.whl "$WHEELS_SRC"/triton-3.7.0+git282c8251-cp314-cp314-*.whl build_ctx/wheels/

# resolve the base image to an immutable digest BEFORE building
docker pull --platform linux/amd64 python:3.14-slim-bookworm >/dev/null
BASE_DIGEST=$(docker inspect --format '{{index .RepoDigests 0}}' python:3.14-slim-bookworm)

docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE="$BASE_DIGEST" \
  -f o1_b200/deploy/Dockerfile.b200 \
  -t "$IMAGE_NAME:$VERSION_TAG" .

IMAGE_ID=$(docker inspect --format '{{.Id}}' "$IMAGE_NAME:$VERSION_TAG")
DOCKERFILE_SHA=$(sha256sum o1_b200/deploy/Dockerfile.b200 | cut -d' ' -f1)
LOCK_SHA=$(sha256sum o1_b200/deploy/requirements.b200.lock | cut -d' ' -f1)
TORCH_WHEEL_SHA=$(grep -m1 " torch-" "$WHEELS_SRC/TORCH_WHEEL.sha256" | cut -d' ' -f1)
TRITON_WHEEL_SHA=$(grep -m1 " triton-" "$WHEELS_SRC/TORCH_WHEEL.sha256" | cut -d' ' -f1)

# wheel-level sm_100 verification on the CUDA host (containers build GPU-less)
WHEELCHECK_VENV="${O1_B200_WHEELCHECK:-/home/moloch/b200_build_cache/wheelcheck_venv}"
ARCH_LIST=$("$WHEELCHECK_VENV/bin/python" -c "import torch, json; al = torch.cuda.get_arch_list(); assert 'sm_100' in al, al; print(json.dumps(al))" 2>/dev/null)
echo "wheel arch list (host verification): $ARCH_LIST"

VERSIONS=$(docker run --rm --entrypoint /opt/venv/bin/python "$IMAGE_NAME:$VERSION_TAG" -c "
import json, platform, sys, torch, transformers, numpy
print(json.dumps({
  'python': sys.version.split()[0],
  'torch': torch.__version__,
  'torch_cuda_runtime': torch.version.cuda,
  'cudnn': torch.backends.cudnn.version(),
  'transformers': transformers.__version__,
  'numpy': numpy.__version__,
  'arch_list_source': 'host wheel verification (build has no GPU); re-verified on the B200 by validate_environment.py',
  'arch_list': json.loads('$ARCH_LIST'),
  'glibc': platform.libc_ver()[1],
}))")

python3 - "$OUT" <<EOF
import json, sys, time
out = sys.argv[1]
record = {
  "schema": "o1b200.container_image_record.v1",
  "build_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "platform": "linux/amd64",
  "image_name": "$IMAGE_NAME:$VERSION_TAG",
  "image_id_local": "$IMAGE_ID",
  "registry_digest_ref": "UNRESOLVED_UNTIL_PUSH (docker push prints repo@sha256:...)",
  "base_image_digest": "$BASE_DIGEST",
  "supported_driver_range": ">= R570 (NVIDIA Blackwell Compatibility Guide; CUDA 12.8 line)",
  "dockerfile_sha256": "$DOCKERFILE_SHA",
  "dependency_lock_sha256": "$LOCK_SHA",
  "torch_wheel_sha256": "$TORCH_WHEEL_SHA",
  "triton_wheel_sha256": "$TRITON_WHEEL_SHA",
  "environment": json.loads('''$VERSIONS'''),
  "system_packages": "base python:3.14-slim-bookworm only; no extra apt packages installed",
  "notes": "no FP8/FP4/quantization/speculative/vLLM/TensorRT-LLM/SGLang; no startup installation/compilation/clone/resolution; checkpoint never baked; no credentials",
}
with open(out, "w") as fh:
    json.dump(record, fh, indent=2, sort_keys=True); fh.write("\n")
print(json.dumps(record["environment"], indent=1))
print("image id:", record["image_id_local"])
EOF
rm -rf build_ctx
echo "BUILD OK -> $OUT"
