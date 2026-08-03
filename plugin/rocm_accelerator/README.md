# AMD GPU Hardware Acceleration

Runs AudioMuse-AI's analysis models on an AMD GPU. Install it on the
[ROCm worker image](https://github.com/Schaka/audiomuse-rocm-plugin); on any
other image it registers nothing and stays out of the way.

## What it does

- **musicnn and the CLAP audio encoder** run through ONNX Runtime's
  `MIGraphXExecutionProvider`, scoped to those two session labels. The Whisper
  encoder and decoder are excluded because MIGraphX cannot compile the decoder
  graph.
- **lyrics ASR** runs on faster-whisper (CTranslate2), whisper.cpp, or
  parakeet.cpp (NVIDIA Parakeet-TDT) instead of the built-in ONNX Whisper, for
  the same reason. Which engine and build variant (Vulkan/HIP) is picked via
  the `asr_backend`/`asr_backend_variant` settings below - see
  [ASR_BACKENDS.md](https://github.com/Schaka/audiomuse-rocm-plugin/blob/main/docs/ASR_BACKENDS.md)
  for what's confirmed working per arch.

CLAP's audio encoder runs on GPU like musicnn (see above). Two other things
stay on CPU: CLAP's *text* encoder, which runs Flask-side with
runtime-variable batch shapes, and clustering, because its library (RAPIDS
cuML) has no ROCm port and no GPU replacement has been built for it yet.

Some GPU generations need this set up differently — no fp16, a different
provider for one model, extra environment. That lives in
[`arch/`](arch/), one profile per generation, with the reasoning in
[ARCH_NOTES.md](https://github.com/Schaka/audiomuse-rocm-plugin/blob/main/docs/ARCH_NOTES.md)
and the how-to in
[ARCH_PROFILES.md](https://github.com/Schaka/audiomuse-rocm-plugin/blob/main/docs/ARCH_PROFILES.md).

## Settings

Edit from the Settings button on the admin Plugins page. That button opens a
raw JSON editor for the whole settings object, not a form with one field per
setting - type the keys below directly as JSON, e.g.:

```json
{
  "asr_backend": "whisper_cpp",
  "asr_backend_variant": "vulkan"
}
```

| Setting | Default | Effect |
| --- | --- | --- |
| `fp16_enable` | `true` | Sets `migraphx_fp16_enable`. Ignored on arches whose profile reports no usable fp16. |
| `asr_backend` | `faster_whisper` | Lyrics ASR engine: `faster_whisper` (CTranslate2), `whisper_cpp`, or `parakeet_cpp` (NVIDIA Parakeet-TDT). See [ASR_BACKENDS.md](https://github.com/Schaka/audiomuse-rocm-plugin/blob/main/docs/ASR_BACKENDS.md) for what's been confirmed working per arch. |
| `asr_backend_variant` | `vulkan` | `vulkan` or `hip`, for `whisper_cpp`/`parakeet_cpp` only (ignored for `faster_whisper`, which always uses CTranslate2's own HIP path). A combination an arch profile lists in `blocked_asr_backends` is refused with a log warning and falls back to `vulkan`, then to `faster_whisper` if even that is blocked. |

## Environment

Set by the worker image; override on the container only if you have a reason to.

| Variable | Default |
| --- | --- |
| `LYRICS_WHISPER_FASTER_DEVICE` | `cuda` (CTranslate2 mirrors the CUDA API on ROCm, so this means the AMD GPU) |
| `LYRICS_WHISPER_FASTER_COMPUTE_TYPE` | `float16` |
| `LYRICS_WHISPER_FASTER_MODEL_DIR` | `/app/model/faster-whisper-small` |
| `LYRICS_WHISPER_CPP_BIN_DIR` | `/opt/asr-backends/whisper-cpp` (holds `whisper-cli-vulkan`/`whisper-cli-hip`) |
| `LYRICS_WHISPER_CPP_MODEL` | `/app/model/whisper-cpp/ggml-small.bin` |
| `LYRICS_PARAKEET_CPP_BIN_DIR` | `/opt/asr-backends/parakeet-cpp` (holds `parakeet-cli-vulkan`/`parakeet-cli-hip`) |
| `LYRICS_PARAKEET_CPP_MODEL` | `/app/model/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf` |

An arch profile may set additional variables, but never overrides one already
set on the container.

## Compiled-model cache

MIGraphX caches compiled programs as `.mxr` files under `/app/.cache/migraphx`,
split into `fp16/` and `fp32/` subdirectories. The split is needed because
MIGraphX keys its artifacts on version, graph, arch and input shapes but *not*
precision — one shared directory would serve an fp32 artifact as a cache hit
after fp16 was switched on. Both sets stay valid, so flipping the setting back
costs no recompile.

Mount it as a volume to keep compilation results across restarts. The image
never clears this cache itself — given how long a recompile can take, a
still-valid cache is worth more than the disk space, so nothing here ever
deletes it on your behalf.

The entrypoint does check whether the image's MIGraphX build has changed
since the cache was last used (e.g. a base image rebuilt against a different
MIGraphX version), and logs a warning if so — a cache built against a
different MIGraphX build can, in rare cases, cause it to recompile the same
graph forever instead of using it. The warning is informational only, nothing
is deleted automatically. If you see it **and** also notice analysis stuck
recompiling the same model repeatedly (log stuck between `Model Compile:
Begin`/`Model Compile: End` far longer than your last known-good run), clear
`/app/.cache/migraphx` and `/app/.cache/miopen` yourself. If analysis is
proceeding normally, ignore the warning — the existing cache is still doing
its job.

### Speeding up the compile

MIGraphX compiles GPU kernels on the CPU, in parallel, controlled by
`MIGRAPHX_GPU_COMPILE_PARALLEL` (a MIGraphX variable, not set by the image —
add it yourself). It defaults to your core count, but that default doesn't
always resolve correctly in a container, so it can end up compiling on far
fewer threads than you have available. Setting it explicitly to your core
count on the worker can noticeably cut compile time:

```yaml
environment:
  MIGRAPHX_GPU_COMPILE_PARALLEL: "8"  # your worker's core count
```

### Avoiding avoidable recompiles

AudioMuse-AI's own `PER_SONG_MODEL_RELOAD` (default `true`) tears down and
rebuilds the musicnn ONNX sessions after every track. Each rebuild goes
through MIGraphX again — a cache hit is fast (well under a second), but
that's still session-construction overhead paid every track for no reason
once the cache is warm. Setting it to `false` on the worker keeps sessions
loaded across 20 tracks instead of 1 (CLAP already only recycles at album
end either way):

```yaml
environment:
  PER_SONG_MODEL_RELOAD: "false"
```

Trade-off, per core's own comment in `config.py`: `true` is safest for VRAM
(stable usage, no leak potential) at ~2-3s reload overhead per song; `false`
is faster but may see gradual VRAM growth on some systems. Worth trying
`false` first — revert if you see VRAM creeping up over a long library scan.

This only speeds up the CPU side of compilation — some of the compile work
still runs on the GPU regardless, so this won't make the GPU portion faster.
It will also peg every thread you give it at 100% CPU for the duration of the
compile, so don't set it above what you can spare on a shared host.

## Requirements

`requirements` in `plugin.json` is deliberately empty. A PyPI `onnxruntime`
pulled in as a dependency would replace the image's MIGraphX-enabled build with
a CPU-only one, so every library this plugin needs has to come from the image.

## Core seams used

1. `register_onnx_provider(name, options, only_models=, needs_static_shapes=)` —
   per-label provider scoping. `needs_static_shapes` makes core pin CLAP's
   symbolic time axis before it builds the session, which MIGraphX needs.
2. `register_analysis_provider('asr', factory)` — replaces the ASR component
   wholesale. The factory is resolved once per worker process, so the
   faster-whisper model stays loaded for a whole album like the built-in does.

Requires core 3.1.0 or newer.
