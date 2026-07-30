# Shared helpers for the standalone ASR backend probes in this directory.
# Sourced, not executed: `source "$(dirname "${BASH_SOURCE[0]}")/common.sh"`.

fail() { echo "error: $*" >&2; exit 1; }
warn() { echo "warning: $*" >&2; }

# Same clip used by ../verify_whisper_compute_types.py: openai/whisper's own
# DRM-free test file, speech rather than song lyrics on purpose - isolates the
# ASR backend from vocal separation/song quality, which is a separate concern.
JFK_FLAC_URL="https://github.com/openai/whisper/raw/main/tests/jfk.flac"
JFK_FLAC="/tmp/jfk.flac"
JFK_WAV="/tmp/jfk.wav"

fetch_test_audio() {
  command -v ffmpeg >/dev/null || fail "ffmpeg not found on host (needed to make a 16k mono wav)"
  [ -f "$JFK_FLAC" ] || { echo "downloading $JFK_FLAC_URL -> $JFK_FLAC" >&2; curl -fL -o "$JFK_FLAC" "$JFK_FLAC_URL" >&2; }
  [ -f "$JFK_WAV" ] || ffmpeg -y -loglevel error -i "$JFK_FLAC" -ar 16000 -ac 1 -c:a pcm_s16le "$JFK_WAV"
  echo "$JFK_WAV"
}

# ROCm device passthrough, same flags local-test/docker-compose-rocm.yaml uses
# for the worker service - kept in one place so a probe script is one line.
rocm_docker_args() {
  local render_gid video_gid
  render_gid="${RENDER_GID:-$(getent group render | cut -d: -f3 || true)}"
  video_gid="${VIDEO_GID:-$(getent group video | cut -d: -f3 || true)}"
  [ -e /dev/kfd ] || warn "/dev/kfd missing - amdgpu driver not loaded"
  [ -e /dev/dri ] || warn "/dev/dri missing - no render nodes to pass through"
  echo --device /dev/kfd --device /dev/dri \
    --group-add "${render_gid:-105}" --group-add "${video_gid:-39}" \
    --security-opt seccomp=unconfined --ipc host --cap-add SYS_PTRACE
}

# Vulkan only needs the render node, not /dev/kfd (no HIP context involved).
vulkan_docker_args() {
  [ -e /dev/dri ] || warn "/dev/dri missing - no render nodes to pass through"
  local video_gid
  video_gid="${VIDEO_GID:-$(getent group video | cut -d: -f3 || true)}"
  echo --device /dev/dri --group-add "${video_gid:-39}"
}
