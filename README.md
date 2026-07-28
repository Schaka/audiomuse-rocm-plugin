# AudioMuse-AI ROCm plugin

AMD GPU acceleration for [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI),
as an installable plugin plus the worker image it needs.

Two separate things live here, released independently:

- **The plugin** (`plugin/rocm_accelerator`) — runs musicnn and the CLAP audio
  encoder through ONNX Runtime's `MIGraphXExecutionProvider` and swaps the
  lyrics ASR backend to faster-whisper. Installed from the Plugins page.
  See [its README](plugin/rocm_accelerator/README.md) for what it actually does.
- **The worker image** (`docker/Dockerfile`) — AudioMuse-AI's published image
  rebuilt on a ROCm base that has a MIGraphX-enabled onnxruntime and
  CTranslate2's ROCm build. The plugin is inert without it.

You need both. The plugin cannot install the ROCm runtime itself (a PyPI
`onnxruntime` would clobber the MIGraphX one), so it ships no pip requirements
and detects the missing provider and stays out of the way on any other image.

## Install the plugin

There is no server behind this. The plugin catalog is published as GitHub
release assets, so the URLs below are all you need.

1. In AudioMuse-AI, open **Plugins → Repositories** (the `PLUGIN_REPOS` setting)
   and add:

   ```
   https://github.com/Schaka/audiomuse-rocm-plugin/releases/latest/download/repository.json
   ```

2. Refresh the catalog. **ROCm Accelerator (AMD)** appears under Catalog.
3. Install it, then restart the worker.

### Unstable channel

Rebuilt from `main` on every plugin change, untested, expect breakage:

```
https://github.com/Schaka/audiomuse-rocm-plugin/releases/download/unstable/repository.json
```

Add one or the other, not both — they publish the same plugin id, and unstable
versions sort above the stable release they were built on top of.

> The plugin requires core **3.1.0 or newer**. AudioMuse-AI filters the catalog
> by `min_core_version`, so on an older core the entry is silently absent rather
> than shown as incompatible.

## Run the worker image

```
ghcr.io/schaka/audiomuse-ai-rocm:latest-<arch>
```

One tag per GPU arch, because baking several arches' ROCm kernels into one
image does not fit on a hosted runner's disk. Pick yours (`rocminfo | grep gfx`):

| Tag | GPUs |
| --- | --- |
| `latest-gfx1201`, `latest-gfx1200` | RDNA4 (RX 9070 …) |
| `latest-gfx1100`, `-gfx1101`, `-gfx1102` | RDNA3 (RX 7000) |
| `latest-gfx1150`, `-gfx1151` | Strix / Strix Halo APUs |
| `latest-gfx1030` | RDNA2 (RX 6000) |
| `latest-gfx900`, `-gfx906`, `-gfx908`, `-gfx90a`, `-gfx942` | Vega / CDNA |
| `latest-gfx803` | Polaris (RX 460–590), experimental |

Also published:

- `:<version>-<arch>` — pinned to an upstream AudioMuse-AI release
- `:unstable-<arch>` and `:unstable-<YYYYMMDD>-<arch>` — nightly, built against
  upstream's `:devel`

`local-test/docker-compose-rocm.yaml` is a working reference: device passthrough,
the MIOpen/MIGraphX cache volumes, and the entrypoint wrapper.

## Local testing

```bash
docker compose -f local-test/docker-compose-rocm.yaml up --build
```

Builds `docker/Dockerfile` from source instead of pulling, and runs a
`plugin-catalog` sidecar that packages `plugin/rocm_accelerator` from the
working tree and serves it over HTTP. Add
`http://plugin-catalog:8099/manifest.json` as a repository in the Plugins UI to
install whatever is currently checked out — no tag, no release, no GitHub round
trip. Override `ROCM_BASE_IMAGE` to test another arch.

## How releases work

Nothing in this repo is written back by CI — the catalog is built from release
assets, so no workflow can retrigger itself.

| Workflow | Trigger | Publishes |
| --- | --- | --- |
| `plugin-release.yml` | `v*` tag here | Release with the plugin zip, `plugin.json`, `repository.json` |
| `plugin-unstable.yml` | push to `main` touching `plugin/**` | Rolling `unstable` prerelease with the same three assets |
| `image-stable.yml` | poll, every 30 min | `:<version>-<arch>` + `:latest-<arch>` when upstream cuts a release |
| `image-unstable.yml` | poll, nightly | `:unstable-<arch>` when upstream's `:devel` digest changes |

The image workflows poll because GitHub cannot trigger a workflow here from a
push in a repository we do not own. Both keep their "last built against" marker
in the Actions cache, keyed on the upstream digest or version.

### Cutting a plugin release

The version is stamped into the manifest by CI, so there is nothing to bump by
hand — only `min_core_version` in `plugin/rocm_accelerator/plugin.json` is
authored, and the annotated tag's message becomes the changelog shown in the
Plugins UI:

```bash
git tag -a v0.1.0 -m "First release: MIGraphX provider for musicnn + faster-whisper ASR."
git push origin v0.1.0
```

Published versions are immutable: never move a tag, cut a new version instead.
The release workflow rebuilds the full version list from every past release, so
older versions stay installable and rollback keeps working.

## Base images

The ROCm bases come from
[Schaka/rocm-migraphx-ort-builder](https://github.com/Schaka/rocm-migraphx-ort-builder)
(`rocm-migraphx-ort-torch-builder:latest-<arch>`, and
`rocm-gfx803-ort-torch-builder:latest` for Polaris, which is a different ROCm
major on a different base).
