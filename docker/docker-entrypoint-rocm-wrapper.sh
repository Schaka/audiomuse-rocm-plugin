#!/usr/bin/env bash
set -euo pipefail

# Wraps the upstream docker-entrypoint.sh instead of editing it, so this
# stays a plain diff-free vendor file across upstream updates. Only job here:
# flag when the ROCm BASE_IMAGE's MIGraphX build changed since the cache
# volume was last used (e.g. switching MIGRAPHX_REF between develop and a
# pinned release branch) -- a stale cache against a different MIGraphX build
# was observed to recompile forever instead of failing fast. Given how costly
# a full recompile is (up to an hour+), this only warns instead of deleting;
# if the recompile-forever symptom shows up, clear the volume manually.
MIGRAPHX_VERSION_FILE=/opt/rocm/migraphx-version.txt
MIGRAPHX_CACHE_STAMP=/app/.cache/migraphx/.image-migraphx-version

if [ -f "$MIGRAPHX_VERSION_FILE" ]; then
  current_version="$(cat "$MIGRAPHX_VERSION_FILE")"
  stamped_version="$(cat "$MIGRAPHX_CACHE_STAMP" 2>/dev/null || true)"
  if [ "$current_version" != "$stamped_version" ]; then
    echo "ENTRYPOINT-WRAPPER: WARNING - MIGraphX build changed (was '${stamped_version:-<none>}', now '${current_version}') - compiled-model cache may be stale. If analysis hangs recompiling the same graph repeatedly, clear /app/.cache/migraphx and /app/.cache/miopen manually."
    mkdir -p "$(dirname "$MIGRAPHX_CACHE_STAMP")"
    echo "$current_version" > "$MIGRAPHX_CACHE_STAMP"
  fi
fi

exec /app/docker-entrypoint.sh "$@"
