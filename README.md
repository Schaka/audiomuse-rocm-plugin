# AudioMuse-AI on AMD GPUs

AMD GPU acceleration for [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI):
musicnn and the CLAP audio encoder on the GPU via ONNX Runtime's MIGraphX
provider, and lyrics transcription on faster-whisper.

**Two pieces, and you need both:**

| | What | Why |
| --- | --- | --- |
| **Worker image** | `ghcr.io/schaka/audiomuse-ai-rocm:latest-<arch>` | AudioMuse-AI's published image rebuilt on a ROCm base, with a MIGraphX-enabled onnxruntime and CTranslate2's ROCm build |
| **Plugin** | `ROCm Accelerator (AMD)`, installed from the Plugins page | Wires those into the analysis pipeline |

A plugin cannot install a ROCm stack (a PyPI `onnxruntime` would replace the
MIGraphX build with a CPU-only one), so the runtime has to come from the image.
On any other image the plugin registers nothing.

Requires AudioMuse-AI **3.1.0 or newer**.

## Quick start

**1. Find your GPU's arch:**

```bash
rocminfo | grep gfx
```

**2. Pick the matching tag.** One tag per arch — several arches' ROCm kernels do
not fit in one image.

| Tag | GPUs |
| --- | --- |
| `latest-gfx1201`, `latest-gfx1200` | RDNA4 (RX 9070 …) |
| `latest-gfx1100`, `-gfx1101`, `-gfx1102` | RDNA3 (RX 7000) |
| `latest-gfx1150`, `-gfx1151` | Strix / Strix Halo APUs |
| `latest-gfx1030` | RDNA2 (RX 6000) |
| `latest-gfx900`, `-gfx906`, `-gfx908`, `-gfx90a`, `-gfx942` | Vega / CDNA |
| `latest-gfx803` | Polaris (RX 460–590) — experimental, see [docs/ARCH_NOTES.md](docs/ARCH_NOTES.md) |

Also published: `:<version>-<arch>` pinned to an upstream release, and
`:unstable-<arch>` / `:unstable-<YYYYMMDD>-<arch>` built nightly against
upstream's `:devel`.

**3. Run it.** [`examples/docker-compose.yaml`](examples/docker-compose.yaml) is
a complete stack — GPU passthrough, group ids, cache volumes:

```bash
ROCM_ARCH=gfx1030 docker compose -f examples/docker-compose.yaml up -d
```

**4. Add the plugin repository.** In the UI, **Plugins → Repositories**, add:

```
https://github.com/Schaka/audiomuse-rocm-plugin/releases/latest/download/repository.json
```

**5. Install.** Refresh the catalog, install **ROCm Accelerator (AMD)** from the
Catalog tab, apply the restart.

**6. Check it worked.** The worker log should show a MIGraphX provider chain for
musicnn and faster-whisper for lyrics; `rocm-smi` on the host should show the
worker using the GPU during an analysis.

## Where to get the plugin

Two routes. Either works; they publish the same plugin id, so **add one, not
both** — an unstable build sorts above the stable release it was built from.

### Community catalog (once submitted)

AudioMuse-AI ships the
[community catalog](https://github.com/NeptuneHub/AudioMuse-AI-plugins) as a
repository out of the box, so the plugin appears there with nothing to add.
Stable releases only.

> Not submitted yet. Until it is, use the repository below.

### This repository's own catalog

Gives you the unstable channel as well, which the community catalog does not
carry. Published as GitHub release assets — there is no server behind it.

```
# stable, tracks the latest release
https://github.com/Schaka/audiomuse-rocm-plugin/releases/latest/download/repository.json

# unstable: rebuilt from main on every plugin change, untested
https://github.com/Schaka/audiomuse-rocm-plugin/releases/download/unstable/repository.json
```

Replacing the community catalog entirely (rather than adding to it) is possible
with `PLUGIN_DEFAULT_REPO_URL`, but then no other community plugin is
installable. Adding a repository in the UI is the better option.

## Documentation

- [Plugin behavior, settings and environment](plugin/rocm_accelerator/README.md)
- [Per-arch findings](docs/ARCH_NOTES.md) — what was measured on which GPU, and
  why the plugin behaves differently there
- [Adding an arch profile](docs/ARCH_PROFILES.md) — wiring in behavior for a GPU
  generation that needs it

## Development

Two deliverables, two project directories, published by separate workflows:

| | Project | Published as |
| --- | --- | --- |
| `plugin/` | the Python plugin — `rocm_accelerator/` is the package that ships, plus `tests/`, `pytest.ini`, `requirements/dev.txt` | zip + catalog on a GitHub release |
| `docker/` | the ROCm worker image — `Dockerfile`, entrypoint, `requirements/rocm.txt` | `ghcr.io/schaka/audiomuse-ai-rocm` |

`local-test/` composes both, `docs/` and `examples/` belong to neither. The
image does not contain the plugin: it is installed at runtime through the
Plugins UI, which is why a plugin change never rebuilds an image.

### Tests

`plugin/` is the Python project — `pytest.ini`, the dev requirements and the
suite all live there:

```bash
cd plugin
pip install -r requirements/dev.txt
pytest
```

No GPU and no ROCm stack needed — the suite stubs the arch string and the
provider list. `jq` and `zip` have to be on `PATH` or the packaging tests skip.
They gate both publish workflows via `plugin-tests.yml`.

### Run against the working tree

```bash
docker compose -f local-test/docker-compose-rocm.yaml up --build
```

Builds `docker/Dockerfile` from source instead of pulling, and runs a
`plugin-catalog` sidecar that zips `plugin/rocm_accelerator` from the working
tree and serves it over HTTP. Add `http://plugin-catalog:8099/manifest.json` as
a repository to install whatever is checked out — no tag, no release, no GitHub
round trip. Override `ROCM_BASE_IMAGE` for another arch.

`local-test/build-gfx803.sh` does the same for gfx803, which needs a
source-built CTranslate2 and so several build args set together.

### Against an unreleased core

Add the `docker-compose-source.yaml` overlay to build core from a branch. Core
first, since the ROCm image does `FROM ${CORE_IMAGE}`:

```bash
COMPOSE="-f local-test/docker-compose-rocm.yaml -f local-test/docker-compose-source.yaml"
docker compose $COMPOSE --profile core build audiomuse-ai-core
docker compose $COMPOSE up --build
```

Defaults to `Schaka/AudioMuse-AI:main`. `AUDIOMUSE_CONTEXT` takes any Docker
build context — a git URL with a `#ref` fragment, or a local path (resolved from
`local-test/`) to iterate without pushing:

```bash
AUDIOMUSE_CONTEXT=../../AudioMuse-AI docker compose $COMPOSE --profile core build audiomuse-ai-core
```

### Testing a profile without the hardware

`register()` decides everything from the arch string and the available provider
list. Stub both, pass a recording `ctx`, and read off what would be registered —
see [docs/ARCH_PROFILES.md](docs/ARCH_PROFILES.md#testing-without-the-hardware).

## Releases

Nothing in this repo is written back by CI — the catalog is built from release
assets, so no workflow can retrigger itself.

| Workflow | Trigger | Publishes |
| --- | --- | --- |
| `plugin-release.yml` | `v*` tag here | Release with the plugin zip, `plugin.json`, `repository.json` |
| `plugin-unstable.yml` | push to `main` touching `plugin/**` | Rolling `unstable` prerelease, same three assets |
| `image-stable.yml` | poll, every 30 min | `:<version>-<arch>` + `:latest-<arch>` when upstream cuts a release |
| `image-unstable.yml` | poll, nightly | `:unstable-<arch>` when upstream's `:devel` digest changes |

The image workflows poll because a push in a repository we do not own cannot
trigger a workflow here. Both keep their "last built against" marker in the
Actions cache, keyed on the upstream digest or version.

### Cutting a plugin release

CI stamps the version into the manifest, so nothing needs bumping by hand —
only `min_core_version` in `plugin/rocm_accelerator/plugin.json` is authored. The
annotated tag's message becomes the changelog shown in the Plugins UI:

```bash
git tag -a v0.1.0 -m "First release: MIGraphX provider for musicnn + faster-whisper ASR."
git push origin v0.1.0
```

Published versions are immutable: never move a tag, cut a new version. The
release workflow rebuilds the version list from every past release, so older
versions stay installable and rollback keeps working.

## Base images

The ROCm bases come from
[Schaka/rocm-migraphx-ort-builder](https://github.com/Schaka/rocm-migraphx-ort-builder):
`rocm-migraphx-ort-torch-builder:latest-<arch>`, one tag per arch. The
`latest-gfx803` tag lives in the same package but is built from a ROCm 6.4.4
base by a separate workflow there, since ROCm 7 dropped Polaris support.
