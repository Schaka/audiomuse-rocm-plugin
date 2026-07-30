# ASR backend findings

Speech-to-text options evaluated against real hardware, standalone
(`local-test/asr_backends/`), outside the plugin - before any of this gets
implemented for real. See that directory for the probe scripts and Dockerfiles
behind every result below.

## gfx803 (Polaris: RX 460-590)

Every GPU compute path was tried against this arch. Results, most to least
useful:

### Vulkan: the only backend that works cleanly for everything

`whisper.cpp` and `parakeet.cpp`, both built with `GGML_VULKAN`/
`PARAKEET_GGML_VULKAN` against this plugin's own worker base image, produce
correct transcripts on this card with no crashes, no garbage output, no
per-arch tuning. Vulkan runs through the host's RADV/mesa driver, a completely
separate stack from ROCm/HIP - it has no dependency on rocBLAS, Tensile, or
MIOpen at all, which turns out to matter a great deal here (see below).

### HIP/ROCm: broken for anything deeper than Whisper's 6-layer encoder

- `whisper.cpp` built with `GGML_HIP` works fine - correct transcript, GPU
  used (`ROCm0` backend confirmed in logs).
- `faster-whisper` (CTranslate2-rocm) works on exactly one configuration:
  **`float16` + `CT2_CUDA_ALLOCATOR=cub_caching` + the conv1d workspace patch**
  (`docker/patches/gfx803/conv1d-workspace-cap.patch`). The apparent
  every-compute-type breakage was three separable bugs stacked on top of each
  other - CT2's default `hipMallocAsync` allocator page-faulting (fixed by
  cub_caching), MIOpen demanding a spurious ~1.44GB Conv1D workspace whose
  failed allocation surfaced as a fake `CUDA failed with error out of memory`
  (fixed by the patch), and the fp32 GEMM path computing garbage. With the
  first two fixed, `float16` transcribes correctly and deterministically
  (30/30 reload-churn iterations on the JFK sample), while `float32` still
  produces multilingual token salad (rocBLAS sgemm broken on this arch;
  same model + audio correct on CPU) and `int8_float32` silently returns
  empty text - those two stay unusable. Full write-up:
  `ARCH_NOTES.md`, "faster-whisper on gfx803".
- Parakeet-TDT 0.6B (NVIDIA NeMo, ~20+ layer Conformer encoder + TDT decoder)
  is broken via HIP regardless of which framework serves it:
  - NeMo + PyTorch (our own working torch/torchaudio, forced fp32, forced
    "math" SDPA to rule out flash-attention specifically): degenerate
    repeated-token output, e.g. `"and and and and and and in in in and"`.
  - `parakeet.cpp` built with `PARAKEET_GGML_HIP` (a from-scratch
    implementation, zero shared code with NeMo/PyTorch): silent **empty**
    output on GPU, `exit 0`, no error at all. Same input on CPU (same
    binary, just no `--device` passthrough): perfect transcript with
    word-level timestamps.
  - beecave-homelab's `parakeet_rocm` project: their published wheels pin
    `torch==...+rocm7.0.0`, which has zero gfx803 device code (ROCm 7 dropped
    Polaris outright) - hard `CUDA is not available`, silent CPU fallback.
    Reinstalling *their actual CLI package* on our own working torch (no
    wheel pin at all) reproduces the exact same garbage as the raw NeMo test
    above - confirms the bug is in NeMo/PyTorch's HIP path itself, not
    anyone's packaging choice.
  - Official `rocm/onnxruntime:rocm6.4.4_ub24.04_ort1.21_torch2.8.0` image:
    doesn't even get that far - `RuntimeError: HIP error: invalid device
    function` on model load. AMD's official builds carry no gfx803 device
    code at all (dropped since ROCm 4.5, 2021); every point release since
    is running on unofficial community patches
    ([robertrosenbusch/gfx803_rocm](https://github.com/robertrosenbusch/gfx803_rocm/),
    [xuhuisheng/rocm-build](https://github.com/xuhuisheng/rocm-build)).

**Not a recent regression.** Checked whether some point release between
6.4.0 and 6.4.4 introduced this (downgrading would be pointless if it
predates the whole 6.4.x line). No rocBLAS/hipBLAS/MIOpen changelog entry in
that window touches gfx803/GCN4 at all - the only correctness fixes in that
window target CDNA/matrix-core GPUs, which gfx803 doesn't have. gfx803
device code was stripped from rocBLAS's own fat binary back at rocBLAS
4.0.0. So this isn't "6.4.4 broke something 6.4.0 had right" - it's a
structural, years-old gap in gfx803's (unofficial) Tensile kernel-selection
support that a shallow network like Whisper happens not to trigger and a
deep Conformer stack does. Historical precedent for gfx803-specific
Tensile/kernel-selection correctness bugs (as opposed to build failures)
goes back to ROCm 3.7
([ROCm/rocBLAS#1172](https://github.com/ROCm/rocBLAS/issues/1172),
[ROCm/ROCm#1265](https://github.com/ROCm/ROCm/issues/1265)) - same bug
class, not this specific incident, but it establishes this isn't a one-off.

### Verdict for gfx803

Use **Vulkan** for every model on this arch. It is the only path that is
both correct and doesn't need per-model kernel-selection workarounds. HIP
stays viable for whisper.cpp specifically if there's ever a reason to prefer
it, but there's no reason to when Vulkan already covers it too.

## gfx900+ (Vega and newer)

Not part of the gfx803 investigation above - HIP is expected to actually work
correctly here (no history of the Tensile kernel-selection class of bug on
supported/current architectures), so the plan is broader: offer
**parakeet.cpp with HIP** and **NeMo + Parakeet** (our own working
torch/torchaudio, no upstream wheel pins) as real backend options, not just
Vulkan.

Both confirmed working, smoke-tested against an RX 9070 XT (gfx1201) on the
plugin's own `rocm-migraphx-ort-torch-builder` base image (ROCm 7.14):

- `parakeet.cpp` built with `PARAKEET_GGML_HIP`: correct transcript,
  `ROCm0` backend, no crash, no garbage - the exact opposite of gfx803's
  result with the identical binary/model/build flags. Confirms the gfx803
  bug is arch-specific, not something inherent to HIP + Parakeet.
- NeMo + PyTorch, same setup as the gfx803 probe (our own torch, no forced
  fp32/SDPA overrides needed - defaulted to `torch.float32` on its own):
  perfect transcript, ~7.5s.

One packaging note from getting this running: gfx1201's base image is on
ROCm 7.14, which renamed the hipBLAS/rocBLAS dev packages
(`amdrocm-blas-dev7.14` etc.) - the `hipblas-dev`/`rocblas-dev` package names
that work on gfx803's ROCm 6.4 base don't exist there, but aren't needed
either: ROCm 7.14's base image already carries its own BLAS dev headers.

## Docker image implications

None of this is implemented in the plugin yet - standalone probes only, per
the validation-before-implementation rule for this investigation. If/when it
lands:

- Every backend's runtime dependencies need to be in the image the plugin
  ships (the plugin itself carries no pip requirements) - same pattern
  `docker/Dockerfile` already uses for CTranslate2/faster-whisper.
- **Models should be shared across backends where the same checkpoint
  works for more than one of them** (e.g. the Parakeet gguf checkpoint used
  by both `parakeet.cpp`'s Vulkan and HIP builds is the same file) - no
  reason to bake in duplicate copies of the same weights just because two
  backends can serve them.
- gfx803 only ever needs the Vulkan variant of each backend; gfx900+ images
  can carry HIP variants (and NeMo) instead, keeping gfx803 images smaller
  and avoiding shipping a HIP path on that arch that's known broken for
  anything but whisper.cpp.
