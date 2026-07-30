#!/usr/bin/env bash
# Probe: NeMo + Parakeet-TDT 0.6B v2 on our own worker base image, per arch.
#
# Usage:
#   ROCM_ARCH=gfx803 local-test/asr_backends/nemo_parakeet_rocm.sh
#   ROCM_ARCH=gfx1201 local-test/asr_backends/nemo_parakeet_rocm.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"

ROCM_ARCH="${ROCM_ARCH:?set ROCM_ARCH, e.g. gfx803 or gfx1201}"
ROCM_BASE_IMAGE="${ROCM_BASE_IMAGE:-ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-$ROCM_ARCH}"
IMAGE="nemo-parakeet-rocm:$ROCM_ARCH"
HF_CACHE="/tmp/nemo-parakeet-hf-cache"

mkdir -p "$HF_CACHE"
audio_path="$(fetch_test_audio)"

echo "==> building $IMAGE (base=$ROCM_BASE_IMAGE)"
docker build \
  --build-arg ROCM_BASE_IMAGE="$ROCM_BASE_IMAGE" \
  -f "$here/nemo-parakeet-rocm.Dockerfile" \
  -t "$IMAGE" "$here"

echo "==> transcribing $audio_path via NeMo on ROCm ($ROCM_ARCH)"
# shellcheck disable=SC2046
docker run --rm \
  $(rocm_docker_args) \
  -e HF_HOME=/hf-cache \
  -v "$HF_CACHE:/hf-cache" \
  -v "$(dirname "$audio_path"):/audios" \
  "$IMAGE" \
  "/audios/$(basename "$audio_path")"
