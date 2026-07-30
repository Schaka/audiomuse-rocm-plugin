# Per-arch findings

Why the arch profiles in `plugin/rocm_accelerator/arch/` look the way they do.
Everything here was reproduced on real hardware; nothing is inferred from
documentation alone.

## gfx803 (Polaris: RX 460–590)

Runs on a ROCm 6.4.4 base with onnxruntime 1.21.1 and MIGraphX
`release/rocm-rel-6.4`, because ROCm 7 dropped Polaris support outright — see
[rocm-migraphx-ort-builder/gfx803/README.md](https://github.com/Schaka/rocm-migraphx-ort-builder/blob/main/gfx803/README.md).
That older stack, not the GPU itself, causes most of what follows.

### No packed FP16 → `fp16_supported = False`

GCN 4 has no packed FP16 ALUs. FP16 math runs at a fraction of the FP32 rate,
so `migraphx_fp16_enable` buys no throughput and only adds precision risk.
Packed FP16 starts at Vega (gfx900).

### No `migraphx_model_cache_dir` → `supports_model_cache_dir = False`

That option postdates onnxruntime 1.21.1's MIGraphX EP. Passing it fails
session creation with `Invalid MIGraphX EP option: migraphx_model_cache_dir`,
and ORT then silently falls back to `CPUExecutionProvider` — a full CPU run
that looks like a successful one in the logs.

Confirmed with `strings` on the shipped `onnxruntime` shared object: no
`migraphx_model_cache_dir` symbol, but the older option set is there —
`migraphx_save_compiled_model` / `migraphx_save_model_path` and the `load_`
counterparts. Note the path keys do **not** repeat "compiled":
`migraphx_save_model_path`, not `migraphx_save_compiled_model_path`. The latter
is rejected as invalid and drops the EP to CPU the same way.

Those options take one literal file per session instead of keying a directory,
which is why `cache.per_model_options` does that keying by hand.

### CLAP cannot compile under MIGraphX → `migraphx_models()` drops it

CLAP's audio encoder has a Resize node carrying an explicit
`keep_aspect_ratio_policy` attribute (opset-19 exporter behavior). This
MIGraphX release's ONNX parser throws on the attribute's mere presence,
regardless of its value or whether shapes are static or dynamic
(`parse_resize.cpp`: `keep_aspect_ratio_policy is not supported!`).

Fixed on MIGraphX `develop`, which the ROCm 7 bases build against — absent or
`stretch` is accepted there, other policies throw with a clearer message. So
CLAP compiles fine on gfx1030+ / gfx9xx and this is gfx803-only.

### MIGraphX and the ROCM EP must never share a session

Putting both in CLAP's session was tried and **SIGSEGVs the whole worker
process** (exit 139), not a catchable ORT exception. MIGraphX does real GPU work
on the part of the graph it can compile before reaching the unsupported Resize
node; handing that node to `ROCMExecutionProvider` inside the same session
faults with `hip_global.cpp: Module not initialized`, consistent with a
corrupted HIP module table. The same fault follows any
MIGraphX-session-then-ROCM-session sequence in one process.

So on gfx803 CLAP gets `[ROCMExecutionProvider, CPUExecutionProvider]` in its
own session and MIGraphX is not offered for it at all. Nothing is lost: it could
never have compiled that node.

Independent corroboration, found after the hardware repro rather than as its
basis:

- [microsoft/onnxruntime#14679](https://github.com/microsoft/onnxruntime/issues/14679)
  — MIGraphX + ROCM in one session producing corrupted output vs. either alone;
  still open.
- [immich-app/immich#27387](https://github.com/immich-app/immich/issues/27387)
  (fixed by [#28444](https://github.com/immich-app/immich/pull/28444)) — a
  MIGraphX-*alone* SIGSEGV from concurrent compiles, fixed by serializing them.
  Different mechanism, but the same theme: MIGraphX's in-process state is
  fragile under multi-session use.

### musicnn must never use the ROCM EP

The ROCM EP **intermittently SIGSEGVs on Conv+Bias+Activation graphs** like
musicnn's heads. ORT's optimizer fuses those three ops into one node, which the
ROCM EP hands to MIOpen's Fusion Plan API; on gfx803 that path corrupts GPU
state and eventually faults with
`Memory access fault … Page not present or supervisor privilege`.

Ruled out on hardware, each individually: stale find-db (deleting the shipped
`gfx803_*.fdb.txt` changed nothing), a single bad solver
(`MIOPEN_DEBUG_CONV_WINOGRAD=0` no help; `MIOPEN_DEBUG_AMD_FUSED_WINOGRAD=0`
reduced but did not eliminate it), an async race
(`HIP_LAUNCH_BLOCKING=1` no help), and unstable VRAM (non-mining BIOS, lower
memory clock, extra cooling). It is not deterministic: the same input crashes on
one invocation and completes on the next.

MIGraphX is also simply faster for musicnn here (~22ms vs ~26–30ms mean per
inference, when the ROCM EP does not crash). Full write-up in the
[base image repo](https://github.com/Schaka/rocm-migraphx-ort-builder/blob/main/gfx803/README.md#known-runtime-issue-rocmexecutionprovider-crashes-on-fused-conv-musicnn-class-models).

### CLAP on the ROCM EP needs `ConvActivationFusion` disabled

CLAP is transformer-dominated, but its conv stem is enough to reach the same
MIOpen Fusion Plan path: ORT's `ConvActivationFusion` optimizer emits FusedConv
nodes for it, and the fused kernels (`miopenSp3AsmConvRxSU_CBA`,
`MIOpenConvUniBatchNormActiv`) intermittently kill the worker with
`Memory access fault … Page not present or supervisor privilege` on the first
inference after a session is created. Isolated on hardware (RX 470 8GB,
ROCm 6.4.4, ORT 1.21.1) with a bare create-session → run → destroy loop against
`model_epoch_36.onnx`:

- Session churn is the amplifier, not the cause: a churn loop usually dies
  within a handful of iterations, but a crash on the very first session of a
  fresh process was also observed. One long-lived session survived 180
  consecutive inferences. This matches production, where core reloads the CLAP
  audio model per track (`PER_SONG_MODEL_RELOAD`), so every track rolls the dice.
- The faulting access is a **read past the end of an ORT arena chunk holding
  the conv weights** (HSA fault address exactly at the chunk's end boundary,
  `VM fault … read from 'TC'` in dmesg — a compute kernel, not a DMA engine).
  Whether the over-read faults depends on whether the neighbouring page is
  mapped, hence the non-determinism as arena layout shifts between sessions.
- Ruled out on hardware: SDMA copies (`HSA_ENABLE_SDMA=0` still crashes) and
  async races (`AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` still crashes;
  the serialized log pins the fault on the fused kernels above).
- Disabling only `ConvActivationFusion` via the
  `optimization.disable_specified_optimizers` session config entry removes
  every fused kernel from the AMD log (the convs run as plain
  `miopenSp3AsmConvRxSU` / `naive_conv` + a separate ReLU) and a 200-iteration
  churn loop then survives where the baseline died within 2.

onnxruntime has no environment variable for this and core builds its own
`SessionOptions`, so the plugin applies the entry by wrapping
`onnxruntime.InferenceSession` at registration time — see
`ort_fusion_guard.py`. The guard keys on the provider chain, which on this
arch scopes it to exactly the CLAP session.

### faster-whisper on gfx803: `float16` is the only correct compute type

Isolated on the RX 470 with a bare load → transcribe → unload loop against the
JFK sample (a known transcript makes silent corruption visible), three
independent failure modes were separated that previously masked one another:

1. **Crash (`Memory access fault … Page not present`), any compute type.**
   CTranslate2's default HIP allocator is the stream-ordered `hipMallocAsync`
   mempool, and kernels (the serialized `AMD_LOG_LEVEL=4` trace pins a
   thrust/rocPRIM scatter from `indexed_fill`, CT2's beam-search token
   suppression) fault on its pages. `CT2_CUDA_ALLOCATOR=cub_caching` — which
   the gfx803 profile already sets — eliminates it: 10/10 fp32 and 30/30 fp16
   churn iterations clean vs a baseline that dies within one or two.
   `MIOPEN_FIND_MODE`, `MIOPEN_DEBUG_CONV_GEMM=0` and `HSA_ENABLE_SDMA=0` were
   each tried and change nothing — it is the allocator, full stop.
2. **Spurious `CUDA failed with error out of memory` mid-transcribe.**
   `miopenConvolutionForwardGetWorkSpaceSize` reports ~1.44GB for Whisper's
   first encoder Conv1D — the worst case across all solvers, driven by a
   GEMM fallback the find never actually picks (every applicable gfx803
   solver needs zero workspace). When VRAM is tight (CLAP/musicnn resident in
   the same worker) that allocation fails and kills the transcription. Fixed
   by `docker/patches/gfx803/conv1d-workspace-cap.patch` in the worker image.
3. **Silent wrong output.** With the crash and OOM out of the way:
   `float32` transcribes to multilingual token salad (~1300 chars of garbage
   for the 108-char JFK line; the same model and audio on CPU are correct, so
   it is the GPU fp32 GEMM path — rocBLAS sgemm on the resurrected r9nano
   Tensile logic), and `int8_float32` silently returns empty text.
   **`float16` transcribes correctly and deterministically** (30/30 identical
   correct transcripts across model reloads) — GCN 4's lack of packed fp16
   costs throughput, not correctness, and hgemm is evidently the code path
   that works. MIOpen's Conv1D is exonerated: it runs in fp32 under every
   compute type, including the correct fp16 runs.

So faster-whisper keeps CTranslate2's plain `float16` default on this arch;
`LYRICS_WHISPER_FASTER_COMPUTE_TYPE` still overrides. `float32` and
`int8_float32` must not be offered as "safe" fallbacks here — they do not
crash, they lie.

## gfx1201 (RDNA4: RX 9070 / XT)

- **GPU page faults in MIGraphX-compiled kernels** (`mul_add_kernel` /
  `convert_mul_add_kernel`) have been seen during musicnn/CLAP inference with
  fp16 both on *and* off. Not fp16-specific, so turning fp16 off is not a
  workaround — it only gives up throughput. No profile override for this.
- **CTranslate2's default HIP allocator faults mid-transcribe** (page not
  present), taking the host down with it. Fixed by `CT2_CUDA_ALLOCATOR=cub_caching`,
  which the worker image sets for every arch. Root cause is an upstream ROCm
  LLVM codegen bug, not CTranslate2 —
  [OpenNMT/CTranslate2#2021](https://github.com/OpenNMT/CTranslate2/issues/2021).
  That issue also shows a GPU load failing with "out of memory" on a card with
  free VRAM, which is why `whisper_faster` logs actual free/total VRAM on a
  failed load.

## gfx900 / gfx906 (Vega)

No plugin-side override yet. Both need rocBLAS rebuilt from source and
CTranslate2 compiled for the arch, but that is the worker image's job and is
already handled. FP16 throughput on Vega is untested here; if it turns out not
to pay off, that is a two-line `Gfx9Profile` — see
[ARCH_PROFILES.md](ARCH_PROFILES.md).

## Why arch detection shells out to rocminfo

`register()` runs once in the long-lived RQ worker process, which then forks a
child per job. A HIP context does not survive `fork()`: initializing one in the
parent (which `torch.cuda.get_device_properties()` does) leaves every child with
a driver handle that looks initialized but is not, and the child's first real GPU
call fails with a generic error — `hipMemGetInfo` raising `Failed getting
available memory: invalid argument` was the actual symptom. Same root cause as
the "Cannot re-initialize CUDA in forked subprocess" warning seen elsewhere.

`rocminfo` is a separate process with its own address space, so parsing its
output detects the arch without initializing anything here, and GPU init still
happens for the first time in the job's own child process.
