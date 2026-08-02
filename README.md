# AudioMuse-AI on AMD GPUs

AMD GPU acceleration for [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI):
musicnn and the CLAP audio encoder on the GPU via ONNX Runtime's MIGraphX
provider, and lyrics transcription on faster-whisper.

**Two pieces, and you need both:**

| | What                                                      | Why |
| --- |-----------------------------------------------------------| --- |
| **Worker image** | `ghcr.io/schaka/audiomuse-ai-rocm:7.14-<arch>`            | AudioMuse-AI's published image rebuilt on a ROCm base, with a MIGraphX-enabled onnxruntime and CTranslate2's ROCm build |
| **Plugin** | `ROCm Accelerator (AMD)`, installed from the Plugins page | Wires those into the analysis pipeline |

A plugin cannot install a ROCm stack (a PyPI `onnxruntime` would replace the
MIGraphX build with a CPU-only one), so the runtime has to come from the image.
On any other image the plugin registers nothing.

Requires AudioMuse-AI **3.1.0 or newer**.

## 1. Pick your image

```bash
rocminfo | grep gfx
```

One tag per arch — several arches' ROCm kernels do not fit in one image.

| Tag | GPUs |
| --- | --- |
| `latest-gfx1201`, `-gfx1200` | RDNA4 (RX 9070 …) |
| `latest-gfx1100`, `-gfx1101`, `-gfx1102`, `-gfx1103` | RDNA3 (RX 7000, RDNA3 APUs) |
| `latest-gfx1150`, `-gfx1151`, `-gfx1152`, `-gfx1153` | Phoenix / Strix / Strix Halo APUs |
| `latest-gfx1030`, `-gfx1031`, `-gfx1032`, `-gfx1034`, `-gfx1035`, `-gfx1036` | RDNA2 (RX 6000, RDNA2 APUs) — not `gfx1033` (Steam Deck): no `rocm7.14-gfx1033` base image upstream |
| `latest-gfx1010`, `-gfx1011`, `-gfx1012` | RDNA1 (RX 5000) |
| `latest-gfx900`, `-gfx90c`, `-gfx906`, `-gfx908`, `-gfx90a`, `-gfx942`, `-gfx950` | Vega / CDNA |
| `latest-gfx803` | Polaris (RX 460–590) — experimental, see [docs/ARCH_NOTES.md](docs/ARCH_NOTES.md) |

Also published: `:<version>-<arch>` pinned to an upstream AudioMuse-AI
release (e.g. `:3.1.0-gfx1030`), for locking your worker to a specific
upstream version instead of tracking `latest`. Unrelated to the plugin's own
version — the plugin never ships inside this image. `:unstable-<arch>` /
`:unstable-<YYYYMMDD>-<arch>` are built nightly against upstream's `:devel`.

## 2. Wire it into your compose file

If you already run upstream's
[`docker-compose.yaml`](https://github.com/NeptuneHub/AudioMuse-AI/blob/main/deployment/docker-compose.yaml),
the change is: swap the worker's image, pass through the GPU, and mount a
cache volume. Nothing else in your existing stack needs to move.

```yaml
  audiomuse-ai-worker:
    image: ghcr.io/schaka/audiomuse-ai-rocm:latest-gfx1030  # <- your arch from step 1
    container_name: audiomuse-ai-worker-instance
    devices:
      - /dev/kfd
      - /dev/dri
    # image ships no render/video group; pass the host's numeric GIDs
    # (getent group render video)
    group_add:
      - "105"
      - "39"
    security_opt:
      - seccomp:unconfined
    ipc: host
    volumes:
      - migraphx-cache:/app/.cache/migraphx
      - miopen-cache:/app/.cache/miopen
```

The two cache volumes hold MIGraphX's and MIOpen's compiled-kernel caches.
Without them, every container restart recompiles the ONNX graphs and GPU
kernels from scratch, which costs minutes before the first analysis can start;
with the volume mounted, a restart reuses what was already compiled. See
[MIGraphX cache details](plugin/rocm_accelerator/README.md#compiled-model-cache)
for why it's split into `fp16`/`fp32` subdirectories internally.

A complete stack — Postgres, Redis, both services, GPU passthrough, group ids,
cache volumes — is at
[`examples/docker-compose.yaml`](examples/docker-compose.yaml). Copy it,
replace the arch in `x-rocm-image`, fill in your media server, `docker compose
up -d`.

## 3. Configure the plugin

Settings, environment variables and the compiled-model cache layout are
documented in the
[plugin README](plugin/rocm_accelerator/README.md#settings) — edit them from
**Plugins → ROCm Accelerator (AMD) → Settings** in the UI, which opens a raw
JSON editor for the whole settings object.

## 4. Get the plugin

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

Add it in **Plugins → Repositories**, refresh the catalog, install **ROCm
Accelerator (AMD)** from the Catalog tab, apply the restart.

Replacing the community catalog entirely (rather than adding to it) is possible
with `PLUGIN_DEFAULT_REPO_URL`, but then no other community plugin is
installable. Adding a repository in the UI is the better option.

**Check it worked:** the worker log should show a MIGraphX provider chain for
musicnn and faster-whisper for lyrics; `rocm-smi` on the host should show the
worker using the GPU during an analysis.

## Documentation

- [Plugin behavior, settings and environment](plugin/rocm_accelerator/README.md)
- [Per-arch findings](docs/ARCH_NOTES.md) — what was measured on which GPU, and
  why the plugin behaves differently there
- [Adding an arch profile](docs/ARCH_PROFILES.md) — wiring in behavior for a GPU
  generation that needs it

## Development

Building the image or plugin from source, running the test suite, iterating
against a working tree or an unreleased core: see
[DEVELOPMENT.md](DEVELOPMENT.md).

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

## Base images

The ROCm bases come from
[Schaka/rocm-migraphx-ort-builder](https://github.com/Schaka/rocm-migraphx-ort-builder):
`rocm-migraphx-ort-torch-builder:rocm7.14-<arch>`, one tag per arch. The
`latest-gfx803` tag lives in the same package but is built from a ROCm 6.4.4
base by a separate workflow there, since ROCm 7 dropped Polaris support — it
never got a `rocm7.14-gfx803` tag, so this repo's workflow still pulls
`6.4.4-gfx803` for that one arch. `gfx1033` (Steam Deck) also has no
`rocm7.14-gfx1033` base image, so it is dropped from the build matrix here.
