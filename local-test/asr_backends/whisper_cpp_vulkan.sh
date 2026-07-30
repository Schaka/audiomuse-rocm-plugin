#!/usr/bin/env bash
# Probe: whisper.cpp's prebuilt Vulkan image against this card.
#
# Vulkan is cross-vendor, so this is the cheapest of the four candidates to
# try - no source build, no arch-specific target string. If it works, it's a
# strong default for any AMD arch this plugin will ever run on, gfx803
# included, without needing a per-arch profile at all.
#
# Usage: local-test/asr_backends/whisper_cpp_vulkan.sh [model]
#   model defaults to "base" - see https://github.com/ggml-org/whisper.cpp/tree/master/models
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"

MODEL="${1:-base}"
MODEL_DIR="/tmp/whisper-cpp-models"
IMAGE="ghcr.io/ggml-org/whisper.cpp:main-vulkan"

mkdir -p "$MODEL_DIR"
audio_path="$(fetch_test_audio)"

echo "==> pulling $IMAGE"
docker pull "$IMAGE"

if [ ! -f "$MODEL_DIR/ggml-$MODEL.bin" ]; then
  echo "==> downloading ggml-$MODEL.bin"
  docker run --rm -v "$MODEL_DIR:/models" --entrypoint bash "$IMAGE" \
    -c "bash /app/models/download-ggml-model.sh $MODEL /models"
fi

echo "==> transcribing $audio_path with $MODEL on Vulkan"
# shellcheck disable=SC2046
docker run --rm \
  $(vulkan_docker_args) \
  -v "$MODEL_DIR:/models" -v "$(dirname "$audio_path"):/audios" \
  "$IMAGE" \
  "whisper-cli -m /models/ggml-$MODEL.bin -f /audios/$(basename "$audio_path")"
