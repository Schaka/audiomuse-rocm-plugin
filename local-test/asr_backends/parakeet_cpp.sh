#!/usr/bin/env bash
# Probe: parakeet.cpp built from source against the HIP/ROCm backend.
#
# No prebuilt image for HIP (only CPU/CUDA), so this builds one, same gap and
# same reasoning as whisper_cpp_rocm.sh. Only Parakeet models here - no Canary
# support in this project despite the name.
#
# Usage:
#   ROCM_ARCH=gfx803 local-test/asr_backends/parakeet_cpp.sh
#   ROCM_ARCH=gfx1201 MODEL=tdt-0.6b-v3-q8_0.gguf local-test/asr_backends/parakeet_cpp.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"

ROCM_ARCH="${ROCM_ARCH:?set ROCM_ARCH, e.g. gfx803 or gfx1201}"
# Same base image per arch as docker/Dockerfile and build-gfx803.sh use for
# the real worker image, so a pass here means it will bolt onto that image too.
ROCM_BASE_IMAGE="${ROCM_BASE_IMAGE:-ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-$ROCM_ARCH}"
MODEL="${MODEL:-tdt-0.6b-v3-q8_0.gguf}"
MODEL_DIR="/tmp/parakeet-cpp-models"
IMAGE="parakeet-cpp-rocm:$ROCM_ARCH"

mkdir -p "$MODEL_DIR"
audio_path="$(fetch_test_audio)"

if [ ! -f "$MODEL_DIR/$MODEL" ]; then
  echo "==> downloading $MODEL from huggingface.co/mudler/parakeet-cpp-gguf"
  curl -fL -o "$MODEL_DIR/$MODEL" \
    "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/$MODEL"
fi

echo "==> building $IMAGE (base=$ROCM_BASE_IMAGE, target=$ROCM_ARCH)"
docker build \
  --build-arg ROCM_BASE_IMAGE="$ROCM_BASE_IMAGE" \
  --build-arg AMDGPU_TARGETS="$ROCM_ARCH" \
  -f "$here/parakeet-cpp-rocm.Dockerfile" \
  -t "$IMAGE" "$here"

echo "==> transcribing $audio_path on ROCm ($ROCM_ARCH)"
# shellcheck disable=SC2046
docker run --rm \
  $(rocm_docker_args) \
  -v "$MODEL_DIR:/models" -v "$(dirname "$audio_path"):/audios" \
  "$IMAGE" \
  transcribe --model "/models/$MODEL" --input "/audios/$(basename "$audio_path")"
