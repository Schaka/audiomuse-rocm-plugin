# ROCm Accelerator (AMD)

Runs AudioMuse-AI's analysis models on an AMD GPU. Install it on the
[ROCm worker image](https://github.com/Schaka/audiomuse-rocm-plugin); on any
other image it registers nothing and stays out of the way.

## What it does

- **musicnn and the CLAP audio encoder** run through ONNX Runtime's
  `MIGraphXExecutionProvider`, scoped to those two session labels. The Whisper
  encoder and decoder are excluded because MIGraphX cannot compile the decoder
  graph.
- **lyrics ASR** runs on faster-whisper (CTranslate2's ROCm backend) instead of
  the built-in ONNX Whisper, for the same reason.

CLAP's text encoder and clustering stay on CPU: the text encoder runs Flask-side
with runtime-variable batch shapes, and the clustering library (RAPIDS cuML) has
no ROCm port.

Some GPU generations need this set up differently — no fp16, a different
provider for one model, extra environment. That lives in
[`arch/`](arch/), one profile per generation, with the reasoning in
[ARCH_NOTES.md](https://github.com/Schaka/audiomuse-rocm-plugin/blob/main/docs/ARCH_NOTES.md)
and the how-to in
[ARCH_PROFILES.md](https://github.com/Schaka/audiomuse-rocm-plugin/blob/main/docs/ARCH_PROFILES.md).

## Settings

Edit from the Settings button on the admin Plugins page.

| Setting | Default | Effect |
| --- | --- | --- |
| `fp16_enable` | `true` | Sets `migraphx_fp16_enable`. Ignored on arches whose profile reports no usable fp16. |

## Environment

Set by the worker image; override on the container only if you have a reason to.

| Variable | Default |
| --- | --- |
| `LYRICS_WHISPER_FASTER_DEVICE` | `cuda` (CTranslate2 mirrors the CUDA API on ROCm, so this means the AMD GPU) |
| `LYRICS_WHISPER_FASTER_COMPUTE_TYPE` | `float16` |
| `LYRICS_WHISPER_FASTER_MODEL_DIR` | `/app/model/faster-whisper-small` |

An arch profile may set additional variables, but never overrides one already
set on the container.

## Compiled-model cache

MIGraphX caches compiled programs as `.mxr` files under `/app/.cache/migraphx`,
split into `fp16/` and `fp32/` subdirectories. The split is needed because
MIGraphX keys its artifacts on version, graph, arch and input shapes but *not*
precision — one shared directory would serve an fp32 artifact as a cache hit
after fp16 was switched on. Both sets stay valid, so flipping the setting back
costs no recompile.

Mount it as a volume to keep compilation results across restarts. The image's
entrypoint clears it by itself when the image's MIGraphX build changes.

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
