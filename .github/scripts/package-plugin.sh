#!/usr/bin/env bash
# Build the installable plugin zip and print its md5.
#
# AudioMuse-AI's plugin manager extracts the zip and expects plugin.json either
# at the archive root or inside a single top-level directory
# (plugin/manager.py:_resolve_extract_root), so GitHub's auto-generated "Source
# code (zip)" cannot be used: it nests the manifest under
# <repo>-<sha>/plugin/rocm_accelerator/. This packs plugin/rocm_accelerator as
# the one top-level directory instead.
#
# The version is stamped into the in-zip manifest here rather than committed,
# so main never has to carry a release-specific value.
#
# Usage: package-plugin.sh <version> <output-zip>
set -euo pipefail

version="$1"
out="$(readlink -f "$2")"
src="plugin/rocm_accelerator"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cp -r "$src" "$work/rocm_accelerator"
jq --arg v "$version" '.version = $v' "$src/plugin.json" > "$work/rocm_accelerator/plugin.json"
find "$work/rocm_accelerator" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$work/rocm_accelerator" -name '*.pyc' -delete

rm -f "$out"
(cd "$work" && zip -qr "$out" rocm_accelerator)

md5sum "$out" | cut -d' ' -f1
