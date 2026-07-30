#!/usr/bin/env bash
# Probe: beecave-homelab/parakeet_rocm's own CLI package, installed on our
# worker base image instead of their rocm7.0 torch wheels. Answers whether
# their wrapper behaves any differently from calling NeMo directly - the
# underlying torch/NeMo/model stack is otherwise identical to
# nemo_parakeet_rocm.sh, so expect the same GPU verdict.
#
# Usage: ROCM_ARCH=gfx803 local-test/asr_backends/parakeet_rocm_ours.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"

ROCM_ARCH="${ROCM_ARCH:?set ROCM_ARCH, e.g. gfx803 or gfx1201}"
ROCM_BASE_IMAGE="${ROCM_BASE_IMAGE:-ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-$ROCM_ARCH}"
IMAGE="parakeet-rocm-ours:$ROCM_ARCH"
HF_CACHE="/tmp/nemo-parakeet-hf-cache"

mkdir -p "$HF_CACHE"
audio_path="$(fetch_test_audio)"

echo "==> building $IMAGE (base=$ROCM_BASE_IMAGE)"
docker build \
  --build-arg ROCM_BASE_IMAGE="$ROCM_BASE_IMAGE" \
  -f "$here/parakeet-rocm-ours.Dockerfile" \
  -t "$IMAGE" "$here"

echo "==> transcribing $audio_path via their CLI on ROCm ($ROCM_ARCH)"
# shellcheck disable=SC2046
docker run --rm \
  $(rocm_docker_args) \
  -e HF_HOME=/hf-cache \
  -v "$HF_CACHE:/hf-cache" \
  -v "$(dirname "$audio_path"):/audios" \
  -v /tmp/parakeet-rocm-out:/app/output \
  "$IMAGE" \
  transcribe "/audios/$(basename "$audio_path")"
