#!/usr/bin/env bash
# Probe: beecave-homelab/parakeet_rocm, a native-ROCm port of NVIDIA's
# Parakeet-TDT 0.6B v2. Targets ROCm 6.4.1 directly - no gfx803 statement
# either way in its docs, so this is genuinely unknown until run. Its
# HSA_OVERRIDE_GFX_VERSION=10.3.0 default is meant for unlisted RDNA2-ish
# cards pretending to be gfx1030; it does not cross GCN4 (gfx803) into RDNA,
# so on gfx803 also try HSA_OVERRIDE_GFX_VERSION= (empty, i.e. unset) if the
# default fails - it may just fail cleanly either way.
#
# Usage:
#   local-test/asr_backends/parakeet_rocm.sh
#   HSA_OVERRIDE_GFX_VERSION= local-test/asr_backends/parakeet_rocm.sh   # gfx803, no override
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"

REPO_DIR="/tmp/parakeet_rocm"
IMAGE="parakeet-rocm:local"

if [ ! -d "$REPO_DIR" ]; then
  git clone --depth 1 https://github.com/beecave-homelab/parakeet_rocm "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only
fi

audio_path="$(fetch_test_audio)"

echo "==> building $IMAGE"
docker build -t "$IMAGE" "$REPO_DIR"

echo "==> transcribing $audio_path on ROCm"
# shellcheck disable=SC2046
docker run --rm \
  $(rocm_docker_args) \
  ${HSA_OVERRIDE_GFX_VERSION+-e HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE_GFX_VERSION"} \
  -v "$(dirname "$audio_path"):/audios" \
  --entrypoint parakeet-rocm \
  "$IMAGE" \
  transcribe "/audios/$(basename "$audio_path")"
