# ASR backend findings

Speech-to-text options evaluated against real hardware, standalone
(`local-test/asr_backends/`), outside the plugin, before implementation. See
that directory for the probe scripts and Dockerfiles behind every result
below.

## gfx803 (Polaris: RX 460-590)

### Vulkan: works cleanly for everything

`whisper.cpp` and `parakeet.cpp`, both built with `GGML_VULKAN`/
`PARAKEET_GGML_VULKAN`, produce correct transcripts on this card - no crashes,
no garbage output, no per-arch tuning. Vulkan runs through the host's
RADV/mesa driver, a completely separate stack from ROCm/HIP with no
dependency on rocBLAS, Tensile, or MIOpen.

### HIP/ROCm

- `whisper.cpp` built with `GGML_HIP`: correct transcript, GPU used (`ROCm0`
  backend confirmed in logs).
- `faster-whisper` (CTranslate2-rocm): fixed, now the plugin's default on
  this arch. Works on exactly one configuration - **`float16` +
  `CT2_CUDA_ALLOCATOR=cub_caching` + the conv1d workspace patch**
  (`docker/patches/gfx803/conv1d-workspace-cap.patch`). Three separable bugs
  stacked on top of each other: CT2's default `hipMallocAsync` allocator
  page-faulting (fixed by cub_caching), MIOpen demanding a spurious ~1.44GB
  Conv1D workspace whose failed allocation surfaced as a fake `CUDA failed
  with error out of memory` (fixed by the patch), and the fp32 GEMM path
  computing garbage. `float32` still produces multilingual token salad
  (rocBLAS sgemm broken on this arch; same model + audio correct on CPU) and
  `int8_float32` silently returns empty text - both stay unusable. Full
  write-up: `ARCH_NOTES.md`, "faster-whisper on gfx803".
- Parakeet-TDT 0.6B via `parakeet.cpp` built with `PARAKEET_GGML_HIP`
  (~20+ layer Conformer encoder + TDT decoder): silent **empty** output on
  GPU, `exit 0`, no error, as of the last test. Same input on CPU (same
  binary, no `--device` passthrough): perfect transcript with word-level
  timestamps.

  **Needs re-testing** - the faster-whisper fixes above (allocator, workspace
  patch) landed after this probe and weren't yet applied when it ran.

### Verdict for gfx803

Vulkan for `whisper.cpp`/`parakeet.cpp` when HIP isn't confirmed working for
them; faster-whisper's HIP path is fixed and is the shipped default.
Parakeet-TDT via HIP is unconfirmed pending re-test.

## gfx900+ (Vega and newer)

Not part of the gfx803 investigation above - HIP is expected to actually work
correctly here (no history of the Tensile kernel-selection class of bug on
supported/current architectures), so `parakeet.cpp` with HIP is a real backend
option, not just Vulkan.

Confirmed working, smoke-tested against an RX 9070 XT (gfx1201) on the
plugin's own `rocm-migraphx-ort-torch-builder` base image (ROCm 7.14):

- `parakeet.cpp` built with `PARAKEET_GGML_HIP`: correct transcript,
  `ROCm0` backend, no crash, no garbage - the exact opposite of gfx803's
  result with the identical binary/model/build flags. Confirms the gfx803
  bug is arch-specific, not something inherent to HIP + Parakeet.

One packaging note from getting this running: gfx1201's base image is on
ROCm 7.14, which renamed the hipBLAS/rocBLAS dev packages
(`amdrocm-blas-dev7.14` etc.) - the `hipblas-dev`/`rocblas-dev` package names
that work on gfx803's ROCm 6.4 base don't exist there, but aren't needed
either: ROCm 7.14's base image already carries its own BLAS dev headers.

## Docker image implications

`faster_whisper`, `whisper_cpp` and `parakeet_cpp` are all shipped and
selectable via the `asr_backend` setting (see the
[plugin README](../plugin/rocm_accelerator/README.md#settings)). Remaining
notes from the investigation:

- Every backend's runtime dependencies need to be in the image the plugin
  ships (the plugin itself carries no pip requirements) - same pattern
  `docker/Dockerfile` already uses for CTranslate2/faster-whisper.
- **Models should be shared across backends where the same checkpoint
  works for more than one of them** (e.g. the Parakeet gguf checkpoint used
  by both `parakeet.cpp`'s Vulkan and HIP builds is the same file) - no
  reason to bake in duplicate copies of the same weights just because two
  backends can serve them.
- gfx803 only ever needs the Vulkan variant of each backend; gfx900+ images
  can carry HIP variants instead, keeping gfx803 images smaller and avoiding
  shipping a HIP path on that arch that's known broken for anything but
  whisper.cpp.

## Canary models

`parakeet.cpp` only covers the Parakeet family (CTC/RNNT/TDT/hybrid,
0.6B/1.1B/110M, English + multilingual v3) - no Canary support.
[CrispASR](https://github.com/CrispStrobe/CrispASR), a whisper.cpp fork with a
broader ggml model zoo, can run Canary GGUFs (`crispasr -m
canary-1b-v2.gguf`). Not evaluated against any of our arches yet; worth a look
if Canary support is ever needed.
