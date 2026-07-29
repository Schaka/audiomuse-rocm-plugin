# ROCm Accelerator (reference plugin)

Proof-of-concept AudioMuse-AI plugin that adds AMD GPU acceleration to the
analysis pipeline, using only two generic plugin seams so nothing AMD-specific
lives in core.

## What it does

- **musicnn and the CLAP audio encoder on the AMD GPU** via ONNX Runtime's
  `MIGraphXExecutionProvider`, registered with `only_models=["musicnn", "clap"]`
  so it never touches the Whisper encoder or decoder (MIGraphX can't parse the
  decoder graph). CLAP needs its symbolic time axis pinned to a static shape
  before MIGraphX can compile it; the provider is registered with
  `needs_static_shapes=True` and core does the pinning
  (`tasks/clap_analyzer.py:_prepared_model_bytes`).
- **lyrics ASR on the AMD GPU** by registering `faster_whisper.py` as the `asr`
  analysis provider (`register_analysis_provider('asr', ...)`), replacing the
  built-in ONNX Whisper backend that MIGraphX can't run.

CLAP's text encoder and clustering (RAPIDS cuML) stay on CPU: the text encoder
runs Flask-side with runtime-variable batch shapes, and cuML has no ROCm port.

## Requirements

Runs only on the **AudioMuse-AI ROCm worker image**, which provides the
MIGraphX-enabled onnxruntime, CTranslate2's ROCm build, faster-whisper and GPU
device access. `requirements` in `plugin.json` is intentionally empty: a PyPI
`onnxruntime` would clobber the image's MIGraphX build. On any other image the
plugin detects the missing provider and stays inert.

## Core seams it depends on

1. `register_onnx_provider(..., only_models=/exclude_models=, needs_static_shapes=)` -
   per-model provider scoping plus the static-shape opt-in (`plugin/api.py`,
   `tasks/analysis/song.py:resolve_providers`). Valid model labels are `musicnn`,
   `clap`, `clap_text`, `whisper_encoder`, `whisper_decoder`, `gte` and
   `silero_vad`; an unknown one is warned about and matches nothing.
2. `register_analysis_provider('asr', factory)` - component replacement, resolved
   by `lyrics/_asr_backend.py` before the built-in `whisper_onnx`. Core checks the
   backend exposes `load_whisper_model`/`transcribe`/`is_loaded`/`unload` and
   resolves the factory once per worker process (the default `cache=True`), so
   `whisper_faster`'s module-level singleton lives for the whole album as the
   built-in does.

## Plugin settings

- `fp16_enable` (default `true`) - enables `migraphx_fp16_enable` on the
  MIGraphX provider. A GPU page fault in a MIGraphX-compiled kernel
  (`mul_add_kernel` / `convert_mul_add_kernel`) has been seen on gfx1201
  (RX 9070 XT / RDNA4) during MusiCNN/CLAP inference, but it recurs with fp16
  both on and off, so it isn't fp16-specific - disabling fp16 here doesn't fix
  it, just gives up the throughput. Edit via the plugin's Settings button on
  the admin Plugins page (`{"fp16_enable": false}`) if you want it off anyway.

## Compiled-model cache

MIGraphX caches compiled programs as `.mxr` files under
`migraphx_model_cache_dir`, keyed on MIGraphX version, graph id, GPU arch and
input shapes. Precision is *not* part of that key, so the plugin points the
provider at a per-precision subdirectory - `/app/.cache/migraphx/fp16` or
`.../fp32`. Without the split, flipping `fp16_enable` would load the previous
precision's artifacts as cache hits and silently run at the wrong precision.
Both sets stay valid across a flip, so switching back costs no recompile.

**gfx803 (ROCm 6.4.4) exception:** that base's onnxruntime (1.21.1) MIGraphX
EP predates `migraphx_model_cache_dir` - passing it fails session creation
outright and ORT falls back to CPU. The plugin detects this by GPU arch
(`torch.cuda.get_device_properties(0).gcnArchName`) and instead uses the
older `migraphx_save_compiled_model`/`migraphx_save_model_path` and
`migraphx_load_compiled_model`/`migraphx_load_model_path` options (the path
keys do not repeat "compiled"), one file per model under the same fp16/fp32 subdirectory
(`musicnn.mxr`, `clap.mxr`). Unlike `migraphx_model_cache_dir` there's no
graph-id hash check on that file, so a stale one from a since-changed model
or input shape would load wrong - delete it (or the whole subdirectory) to
force a recompile.

`/app/.cache/migraphx` is a named volume in `local-test/docker-compose-rocm.yaml`;
the ROCm entrypoint wrapper clears it (subdirectories included) when the image's
MIGraphX build changes.

## Env (set by the ROCm image, override if needed)

- `LYRICS_WHISPER_FASTER_DEVICE` (default `cuda`; CTranslate2 mirrors the CUDA
  API on ROCm)
- `LYRICS_WHISPER_FASTER_COMPUTE_TYPE` (default `float16`)
- `LYRICS_WHISPER_FASTER_MODEL_DIR` (default `/app/model/faster-whisper-small`)
