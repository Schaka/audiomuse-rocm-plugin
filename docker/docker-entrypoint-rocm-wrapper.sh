#!/usr/bin/env bash
set -euo pipefail

# Wraps the upstream docker-entrypoint.sh instead of editing it, so this
# stays a plain diff-free vendor file across upstream updates. Only job here:
# invalidate the MIOpen/MIGraphX compiled-model cache volumes when the ROCm
# BASE_IMAGE's MIGraphX build changed (e.g. switching MIGRAPHX_REF between
# develop and a pinned release branch) -- a stale cache against a different
# MIGraphX build was observed to recompile forever instead of failing fast.
MIGRAPHX_VERSION_FILE=/opt/rocm/migraphx-version.txt
MIGRAPHX_CACHE_DIRS="/app/.cache/migraphx /app/.cache/miopen"
MIGRAPHX_CACHE_STAMP=/app/.cache/migraphx/.image-migraphx-version

if [ -f "$MIGRAPHX_VERSION_FILE" ]; then
  current_version="$(cat "$MIGRAPHX_VERSION_FILE")"
  stamped_version="$(cat "$MIGRAPHX_CACHE_STAMP" 2>/dev/null || true)"
  if [ "$current_version" != "$stamped_version" ]; then
    echo "ENTRYPOINT-WRAPPER: MIGraphX build changed (was '${stamped_version:-<none>}', now '${current_version}') - clearing compiled-model cache"
    for d in $MIGRAPHX_CACHE_DIRS; do
      find "$d" -mindepth 1 -delete 2>/dev/null || true
    done
    mkdir -p "$(dirname "$MIGRAPHX_CACHE_STAMP")"
    echo "$current_version" > "$MIGRAPHX_CACHE_STAMP"
  fi
fi

exec /app/docker-entrypoint.sh "$@"
