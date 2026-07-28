#!/usr/bin/env bash
# Generate the two catalog files AudioMuse-AI fetches, from a versions array.
#
# Both are published as GitHub release assets, never committed: a workflow that
# committed them back to main would retrigger itself, and release assets give
# stable URLs with no server to host.
#
#   repository.json  what the user pastes into PLUGIN_REPOS. Carries only the
#                    plugin's identity plus a pluginUrl pointing at plugin.json
#                    (plugin/blueprint.py:_resolve_versions).
#   plugin.json      the full versions list. Each entry needs sourceUrl or core
#                    silently drops the plugin from the catalog
#                    (plugin/blueprint.py:_pick_version). targets/requirements
#                    are read from the top level of this doc, not from the
#                    entries (plugin/blueprint.py:_build_catalog_entry).
#
# Identity fields come from the in-repo manifest so there is one source of truth.
#
# Usage: catalog.sh <versions-json-file> <plugin-url> <outdir>
set -euo pipefail

versions="$1"
plugin_url="$2"
outdir="$3"
manifest="plugin/rocm_accelerator/plugin.json"

mkdir -p "$outdir"

jq -n \
  --slurpfile m "$manifest" \
  --slurpfile v "$versions" \
  '$m[0] | {
     id, name, author, description,
     targets: (.targets // []),
     requirements: (.requirements // []),
     versions: $v[0]
   }' > "$outdir/plugin.json"

jq -n \
  --slurpfile m "$manifest" \
  --arg url "$plugin_url" \
  '{plugins: [$m[0] | {id, name, author, description, pluginUrl: $url}]}' \
  > "$outdir/repository.json"
