#!/usr/bin/env bash
# Build and run the whole local stack for a Polaris / gfx803 test system.
#
# Wraps the two compose files with the gfx803-specific bits filled in: the
# ROCm 6.x base image (gfx803 is a separate package upstream, not another tag
# in the main arch matrix) and the host's render/video GIDs, which have to be
# passed numerically because the base image has no matching groups.
#
# Core is built from source by default, since the published core image does not
# have the plugin seams yet.
#
# Usage:
#   local-test/build-gfx803.sh                 # build core + images, then start
#   local-test/build-gfx803.sh --build-only    # build everything, start nothing
#   SKIP_CORE=1 local-test/build-gfx803.sh     # reuse the core image already built
#   AUDIOMUSE_CONTEXT=../../AudioMuse-AI local-test/build-gfx803.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose=(-f "$here/docker-compose-rocm.yaml" -f "$here/docker-compose-source.yaml")

build_only=0
[ "${1:-}" = "--build-only" ] && build_only=1

# gfx803 is ROCm 6.x on its own base package; the compose default targets RDNA4.
export ROCM_BASE_IMAGE="${ROCM_BASE_IMAGE:-ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803}"

# That ROCm 6 base cannot use CTranslate2's prebuilt ROCm wheel (it wants
# libhipblas.so.3, ROCm 7), so the ROCm fork is compiled from source for
# gfx803. This is the long pole of the image build.
export CT2_VARIANT="${CT2_VARIANT:-source}"
export CT2_REPO="${CT2_REPO:-https://github.com/arlo-phoenix/CTranslate2-rocm.git}"
export CT2_REF="${CT2_REF:-rocm}"
export ROCM_ARCH="${ROCM_ARCH:-gfx803}"
export AUDIOMUSE_CONTEXT="${AUDIOMUSE_CONTEXT:-https://github.com/Schaka/AudioMuse-AI.git#main}"
export CORE_IMAGE="${CORE_IMAGE:-audiomuse-ai-core:local}"

# Empty resolves back to the compose file's own defaults, which are frequently
# wrong for a given host - hence the warning rather than a silent fallback.
export RENDER_GID="${RENDER_GID:-$(getent group render | cut -d: -f3 || true)}"
export VIDEO_GID="${VIDEO_GID:-$(getent group video | cut -d: -f3 || true)}"

fail() { echo "error: $*" >&2; exit 1; }
warn() { echo "warning: $*" >&2; }

command -v docker >/dev/null || fail "docker not found"
docker compose version >/dev/null 2>&1 || fail "docker compose v2 not available"

[ -e /dev/kfd ] || warn "/dev/kfd missing - the amdgpu driver is not loaded, the worker will not see the GPU"
[ -e /dev/dri ] || warn "/dev/dri missing - no render nodes to pass through"
[ -n "$RENDER_GID" ] || warn "no 'render' group on this host; falling back to the compose default, which is probably wrong (check: ls -ln /dev/kfd)"
[ -n "$VIDEO_GID" ] || warn "no 'video' group on this host; falling back to the compose default"

echo "base image : $ROCM_BASE_IMAGE"
echo "core source: $AUDIOMUSE_CONTEXT -> $CORE_IMAGE"
echo "gids       : render=${RENDER_GID:-<compose default>} video=${VIDEO_GID:-<compose default>}"
echo

# Separate step, ahead of everything else: docker/Dockerfile does
# `FROM ${CORE_IMAGE}`, and compose does not order builds around that.
if [ "${SKIP_CORE:-0}" = "1" ]; then
  echo "==> skipping core build (SKIP_CORE=1)"
  docker image inspect "$CORE_IMAGE" >/dev/null 2>&1 \
    || fail "SKIP_CORE=1 but $CORE_IMAGE does not exist locally"
else
  echo "==> building core from source (slow: pulls ~5GB of analysis models)"
  docker compose "${compose[@]}" --profile core build audiomuse-ai-core
fi

echo "==> building the ROCm worker image"
docker compose "${compose[@]}" build

if [ "$build_only" = "1" ]; then
  echo "==> built; not starting (--build-only)"
  exit 0
fi

echo "==> starting the stack"
docker compose "${compose[@]}" up -d

cat <<EOF

Stack is up. Next:

  1. http://localhost:${FRONTEND_PORT:-8000} -> log in as admin
  2. Plugins > Repositories > add:  http://plugin-catalog:8099/manifest.json
  3. Catalog > install "ROCm Accelerator (AMD)" > Apply (restart)
  4. Run an analysis; the worker log should show the MIGraphX provider for
     musicnn and faster-whisper for lyrics. Confirm with rocm-smi on the host.

  logs: docker compose ${compose[*]} logs -f audiomuse-ai-worker
  stop: docker compose ${compose[*]} down
EOF
