#!/usr/bin/env bash
# Probe: whisper.cpp built from source against the HIP/ROCm backend.
#
# No prebuilt image exists for this backend (whisper.cpp only publishes CUDA
# and Vulkan images), so this builds one - the long pole of this script.
# whisper.cpp gets its own build/probe per arch since it's a different GEMM
# path than CTranslate2-rocm; the fact that faster-whisper's kernels are
# broken on gfx803 says nothing about whether ggml's are.
#
# Usage:
#   ROCM_ARCH=gfx803 local-test/asr_backends/whisper_cpp_rocm.sh
#   ROCM_ARCH=gfx1201 local-test/asr_backends/whisper_cpp_rocm.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"

ROCM_ARCH="${ROCM_ARCH:?set ROCM_ARCH, e.g. gfx803 or gfx1201}"
# Same base image per arch as docker/Dockerfile and build-gfx803.sh use for
# the real worker image, so a pass here means it will bolt onto that image too.
ROCM_BASE_IMAGE="${ROCM_BASE_IMAGE:-ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-$ROCM_ARCH}"
IMAGE="whisper-cpp-rocm:$ROCM_ARCH"

audio_path="$(fetch_test_audio)"

echo "==> building $IMAGE (base=$ROCM_BASE_IMAGE, target=$ROCM_ARCH)"
docker build \
  --build-arg ROCM_BASE_IMAGE="$ROCM_BASE_IMAGE" \
  --build-arg AMDGPU_TARGETS="$ROCM_ARCH" \
  -f "$here/whisper-cpp-rocm.Dockerfile" \
  -t "$IMAGE" "$here"

echo "==> transcribing $audio_path on ROCm ($ROCM_ARCH)"
# shellcheck disable=SC2046
docker run --rm \
  $(rocm_docker_args) \
  -v "$(dirname "$audio_path"):/audios" \
  "$IMAGE" \
  -m /build/models/ggml-base.bin -f "/audios/$(basename "$audio_path")"
