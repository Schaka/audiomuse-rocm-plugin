#!/usr/bin/env bash
# Probe: parakeet.cpp built with the Vulkan backend on our own worker base
# image - cross-vendor, but built on the real image for the same reason as
# the HIP probes (does it bolt onto what we actually ship).
#
# Usage:
#   ROCM_ARCH=gfx803 local-test/asr_backends/parakeet_cpp_vulkan.sh
#   ROCM_ARCH=gfx1201 local-test/asr_backends/parakeet_cpp_vulkan.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"

ROCM_ARCH="${ROCM_ARCH:?set ROCM_ARCH, e.g. gfx803 or gfx1201}"
ROCM_BASE_IMAGE="${ROCM_BASE_IMAGE:-ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-$ROCM_ARCH}"
MODEL="${MODEL:-tdt-0.6b-v3-q8_0.gguf}"
MODEL_DIR="/tmp/parakeet-cpp-models"
IMAGE="parakeet-cpp-vulkan:$ROCM_ARCH"

mkdir -p "$MODEL_DIR"
audio_path="$(fetch_test_audio)"

if [ ! -f "$MODEL_DIR/$MODEL" ]; then
  echo "==> downloading $MODEL from huggingface.co/mudler/parakeet-cpp-gguf"
  curl -fL -o "$MODEL_DIR/$MODEL" \
    "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/$MODEL"
fi

echo "==> building $IMAGE (base=$ROCM_BASE_IMAGE)"
docker build \
  --build-arg ROCM_BASE_IMAGE="$ROCM_BASE_IMAGE" \
  -f "$here/parakeet-cpp-vulkan.Dockerfile" \
  -t "$IMAGE" "$here"

echo "==> transcribing $audio_path on Vulkan"
# shellcheck disable=SC2046
docker run --rm \
  $(vulkan_docker_args) \
  -v "$MODEL_DIR:/models" -v "$(dirname "$audio_path"):/audios" \
  "$IMAGE" \
  transcribe --model "/models/$MODEL" --input "/audios/$(basename "$audio_path")" --decoder tdt --json
