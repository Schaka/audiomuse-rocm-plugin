# Development

Two deliverables, two project directories, published by separate workflows:

| | Project | Published as |
| --- | --- | --- |
| `plugin/` | the Python plugin — `rocm_accelerator/` is the package that ships, plus `tests/`, `pytest.ini`, `requirements/dev.txt` | zip + catalog on a GitHub release |
| `docker/` | the ROCm worker image — `Dockerfile`, entrypoint, `requirements/rocm.txt` | `ghcr.io/schaka/audiomuse-ai-rocm` |

`local-test/` composes both, `docs/` and `examples/` belong to neither. The
image does not contain the plugin: it is installed at runtime through the
Plugins UI, which is why a plugin change never rebuilds an image.

## Tests

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

## Run against the working tree

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

## Against an unreleased core

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

## Testing a profile without the hardware

`register()` decides everything from the arch string and the available provider
list. Stub both, pass a recording `ctx`, and read off what would be registered —
see [docs/ARCH_PROFILES.md](docs/ARCH_PROFILES.md#testing-without-the-hardware).
