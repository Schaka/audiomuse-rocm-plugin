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

### CLAP can compile under MIGraphX now, but stays on the ROCM EP anyway

CLAP's audio encoder has a Resize node carrying an explicit
`keep_aspect_ratio_policy` attribute (opset-19 exporter behavior). This
MIGraphX release's ONNX parser used to throw on the attribute's mere presence
(`parse_resize.cpp`: `keep_aspect_ratio_policy is not supported!`) — fixed in
the base image before compiling MIGraphX
([parse-resize-fixes.patch](https://github.com/Schaka/rocm-migraphx-ort-builder/blob/main/gfx803/patches/migraphx/parse-resize-fixes.patch)).

Verified correct on real hardware (RX 470, `model_epoch_36.onnx`, cosine
similarity against the CPU EP output): MIGraphX now gives 0.98–0.997 across
repeated runs with different random inputs. `migraphx_models()` still doesn't
offer it for CLAP when the ROCM EP is available, though — the ROCM EP
(guarded, see below) benchmarks marginally faster for CLAP on the same
hardware (~12ms vs MIGraphX's ~13ms mean per inference), and switching would
be a wash at best. Worth re-benchmarking if that gap changes.

### MIGraphX and the ROCM EP must never share a session

Putting both in CLAP's session was tried and **SIGSEGVs the whole worker
process** (exit 139), not a catchable ORT exception. MIGraphX does real GPU work
on the part of the graph it can compile before reaching the unsupported Resize
node (now fixed in base image); handing that node to `ROCMExecutionProvider` inside the same session
faults with `hip_global.cpp: Module not initialized`, consistent with a
corrupted HIP module table. 

So on gfx803 CLAP gets `[ROCMExecutionProvider, CPUExecutionProvider]` in its
own session and MIGraphX is not offered for it at all. Nothing is lost: it could
never have compiled that node.

Independent corroboration, found after the hardware repro rather than as its
basis:

- [microsoft/onnxruntime#14679](https://github.com/microsoft/onnxruntime/issues/14679)
  — MIGraphX + ROCM in one session producing corrupted output vs. either alone;
  still open.

### musicnn stays on MIGraphX, not the ROCM EP

Not a correctness workaround anymore (see below) — MIGraphX is just faster
for musicnn here (~22ms vs ~26–30ms mean per inference), so `extra_providers()`
never offers it the ROCM EP.

### CLAP on the ROCM EP needs `ConvActivationFusion` disabled

CLAP's conv stem is enough to reach MIOpen's Fusion Plan path: ORT's
`ConvActivationFusion` optimizer fuses Conv+Bias+Activation into FusedConv
nodes, the ROCM EP hands those to MIOpen, and on this arch that produces wrong
output — and can still crash with `Memory access fault … Page not present or
supervisor privilege`.

The base image rebuilds MIOpen from source with a fix for one specific cause
of this (`patches/miopen/conv-direct-fwd-grouped-oob.sh`, an OOB read in
`ConvOclDirectFwd` for grouped/depthwise convs — see the [base image repo's
KERNEL_BUGS.md](https://github.com/Schaka/rocm-migraphx-ort-builder/blob/main/gfx803/KERNEL_BUGS.md#miopen-convactivationfusion-investigation)).
That fix is real but not sufficient: re-tested directly against a locally
built image carrying it (`LD_PRELOAD`/sgemm-shim and the rebuilt MIOpen both
confirmed present), `ConvActivationFusion` left enabled still gives
cos(CPU, ROCM) of 0.24–0.84 across repeated runs with different random inputs
(vs. 0.98+ with it disabled) and still crashed once during a short churn
test. So the plugin keeps disabling it itself.

`ort_fusion_guard.py` applies `optimization.disable_specified_optimizers` at
session-creation time — see `ProviderSpec.disable_optimizers` in
`arch/base.py`. The guard keys on the provider chain, which on this arch
scopes it to exactly the CLAP session.

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
