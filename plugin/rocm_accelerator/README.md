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
(shelling out to `rocminfo`, not `torch.cuda` - see the "Why rocminfo, not
torch.cuda" note below) and instead uses the
older `migraphx_save_compiled_model`/`migraphx_save_model_path` and
`migraphx_load_compiled_model`/`migraphx_load_model_path` options (the path
keys do not repeat "compiled"), one file per model under the same fp16/fp32 subdirectory
(`musicnn.mxr`, `clap.mxr`). Unlike `migraphx_model_cache_dir` there's no
graph-id hash check on that file, so a stale one from a since-changed model
or input shape would load wrong - delete it (or the whole subdirectory) to
force a recompile.

**Why rocminfo, not torch.cuda:** `register()` runs once in the persistent RQ
worker process at startup, before RQ forks a fresh child process per job (RQ
calls these "horse" processes in its own logs). A HIP/CUDA context does not
survive `fork()` cleanly - initializing one in the long-lived parent (which
`torch.cuda.get_device_properties()` does) leaves every forked child with a
driver handle that looks initialized but isn't, so the child's first real GPU
call fails with a generic-looking error (MIGraphX's `hipMemGetInfo` raising
"Failed getting available memory: invalid argument" was the actual symptom
this caused here) - the same root cause behind the "Cannot re-initialize CUDA
in forked subprocess" warning core already logs elsewhere. `rocminfo` is a
separate process, so parsing its text output detects the arch without
initializing anything in the worker process itself, leaving GPU init to
happen for the first time in the actual job's forked child, same as it does
without this plugin.

`/app/.cache/migraphx` is a named volume in `local-test/docker-compose-rocm.yaml`;
the ROCm entrypoint wrapper clears it (subdirectories included) when the image's
MIGraphX build changes.

## CLAP audio: ROCMExecutionProvider fallback (gfx803 only)

CLAP's audio encoder has a Resize node that exports an explicit
`keep_aspect_ratio_policy` attribute (opset-19 exporter behavior). gfx803's
MIGraphX is pinned to `release/rocm-rel-6.4`
(`rocm-migraphx-ort-builder/gfx803/Dockerfile.gfx803`'s `MIGRAPHX_REF`), whose
ONNX parser throws on that attribute's mere presence - regardless of its value
or whether the graph's shapes are static or dynamic
(`parse_resize.cpp`: `"keep_aspect_ratio_policy is not supported!"`) - so CLAP
can never compile under `MIGraphXExecutionProvider` there; only musicnn
benefits from that provider on gfx803.

This is fixed upstream: MIGraphX's `develop` branch (what the ROCm 7 base for
gfx9xx/gfx1030+ builds against) added real support for the attribute
(`stretch`/absent is accepted; other policies still throw, now with a clearer
message), which is why this has not been seen on RX 9070 XT (gfx1201) or other
newer-base arches - CLAP compiles fine there.

Where the plain kernel-based `ROCMExecutionProvider` is available (gfx803's
ORT build, 1.21.1, still ships it - the ROCm 7 base used for gfx9xx/gfx1030+
dropped it, keeping only MIGraphX), the plugin registers it as CLAP's *only*
GPU provider on that arch: it runs Resize as an ordinary op rather than
parsing the graph ahead of time, so the attribute is a non-issue, and no
static-shape pinning or fp16 flags are needed. `_rocm_ep_available()`
naturally scopes this to gfx803 today without hardcoding the arch.

**MIGraphX is deliberately NOT registered for `clap` on these arches at all**
- not even alongside ROCM as a first attempt. Putting both providers in one
ORT session for CLAP was tried and confirmed unsafe on real gfx803 hardware:
MIGraphX does real GPU work on the parts of the graph it can compile before
reaching the unsupported Resize node, and handing that node off to
`ROCMExecutionProvider` within the same session SIGSEGVs the whole worker
process (exit 139) - not a catchable ONNX Runtime exception like a MIGraphX-
only session raises. Confirmed with a minimal repro
(`local-test/repro_clap_provider_chain.py`, plus the more general
`local-test/repro_migraphx_then_rocm_crash.py` showing the same fault from
any MIGraphX-session-then-ROCM-session sequence in one process).

Independent prior art for this combination being broken (found after the fact,
not the basis for the fix - the hardware repro above already was):
[microsoft/onnxruntime#14679](https://github.com/microsoft/onnxruntime/issues/14679)
reports `MIGraphXExecutionProvider` + `ROCMExecutionProvider` together in one
session (with CPU fallback) producing wrong/corrupted output vs. either
provider alone - still open/unresolved upstream as of this writing. Separately,
[immich-app/immich#27387](https://github.com/immich-app/immich/issues/27387)
(fixed by [#28444](https://github.com/immich-app/immich/pull/28444)) documents
a MIGraphX-*alone* SIGSEGV caused by concurrent/overlapping compile calls in one
process, fixed by serializing compiles with a lock - different mechanism, but
corroborates that MIGraphX's in-process state is fragile under multi-session
use in general. Neither report reproduces our exact `hip_global.cpp: Module not
initialized` message or is gfx803-specific - that mechanistic detail (HIP
module-table corruption on EP hand-off) is our own inference from the repro,
not independently documented anywhere found. The practical conclusion (never
share a session between these two providers) holds regardless of which exact
internal mechanism is at fault.

Since MIGraphX could never compile CLAP's Resize node here anyway, there's nothing
lost by skipping it - `clap`'s session on gfx803 is `[ROCMExecutionProvider,
CPUExecutionProvider]` only. musicnn is unaffected: its session never
contains ROCM, so it keeps `[MIGraphXExecutionProvider, CPUExecutionProvider]`
as before.

## Env (set by the ROCm image, override if needed)

- `LYRICS_WHISPER_FASTER_DEVICE` (default `cuda`; CTranslate2 mirrors the CUDA
  API on ROCm)
- `LYRICS_WHISPER_FASTER_COMPUTE_TYPE` (default `float16`)
- `LYRICS_WHISPER_FASTER_MODEL_DIR` (default `/app/model/faster-whisper-small`)
