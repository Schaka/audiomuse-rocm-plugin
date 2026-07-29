# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""ROCm Accelerator plugin for AudioMuse-AI.

Wires AMD GPU acceleration into the analysis pipeline without forking core:

* musicnn and the CLAP audio encoder run on the AMD GPU through ONNX Runtime's
  MIGraphXExecutionProvider, scoped to those two models only. CLAP needs its
  symbolic time axis pinned to a static shape before MIGraphX can compile it,
  which core does for any provider registered with ``needs_static_shapes=True``
  (see ``tasks/clap_analyzer.py:_prepared_model_bytes``). The Whisper decoder
  has no such fixup, so it stays off this chain entirely.
* lyrics ASR is swapped to faster-whisper (CTranslate2's ROCm backend) because
  MIGraphX can't run the ONNX Whisper decoder at all.

The ROCm runtime (MIGraphX-enabled onnxruntime, CTranslate2 ROCm, faster-whisper,
GPU device access) is provided by the ROCm worker image, not this plugin. On any
other image the plugin registers nothing and stays inert.
"""

import logging
import os
import subprocess

from plugin.api import get_setting

logger = logging.getLogger("plugin.rocm_accelerator")

MIGRAPHX_CACHE_ROOT = "/app/.cache/migraphx"

# GCN 4 (Polaris - gfx803/802/805, e.g. RX 470/480/570/580) has no packed FP16
# throughput: FP16 math runs at a fraction of FP32 rate (or is emulated), so
# migraphx_fp16_enable buys no speedup and only adds precision risk. Packed
# FP16 starts at Vega (gfx900) and is present on every RDNA/CDNA part since.
NO_PACKED_FP16_ARCHES = {"gfx803", "gfx802", "gfx805"}


def _gpu_arch():
    # Deliberately shells out to rocminfo instead of asking torch.cuda: this
    # runs from register(), which fires once in the persistent RQ worker
    # process at startup - well before RQ fork()s a fresh child ("horse", per
    # RQ's own log lines) to actually run each job. A HIP/CUDA context does
    # not survive fork() cleanly; touching torch.cuda here would initialize
    # one in the long-lived parent, and every forked child then inherits a
    # driver handle that looks initialized but isn't, so the child's first
    # real GPU call (MIGraphX's hipMemGetInfo) fails with a generic-looking
    # "invalid argument" - exactly the failure this was chasing, and the
    # same root cause behind the "Cannot re-initialize CUDA in forked
    # subprocess" warning core already logs elsewhere. rocminfo is a separate
    # process with its own address space, so parsing its text output detects
    # the arch without initializing anything in this one.
    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
            if name.startswith("gfx"):
                return name
    return None


def _fp16_capable():
    arch = _gpu_arch()
    if arch is None:
        # Unknown - don't block a setting we can't verify; the arch's own EP
        # session creation is the backstop if fp16 truly isn't supported.
        return True
    return arch not in NO_PACKED_FP16_ARCHES


# gfx803's base is ROCm 6.4.4 (see image-build.yml / docker/Dockerfile's ct2
# stage comments - gfx803 is the one arch still on a ROCm 6 base). Its
# onnxruntime (1.21.1) MIGraphX EP build predates the migraphx_model_cache_dir
# option: passing it fails session creation with "Invalid MIGraphX EP option:
# migraphx_model_cache_dir" and ORT silently falls back to CPUExecutionProvider.
# Confirmed by `strings` on the shipped onnxruntime .so - it has no
# migraphx_model_cache_dir symbol, but does have the older, pre-cache-dir
# option set: migraphx_save_compiled_model / migraphx_save_model_path (and the
# load_ counterparts) - note the path keys do NOT repeat "compiled": it's
# migraphx_save_model_path, not migraphx_save_compiled_model_path (the latter
# is rejected as an invalid option and silently drops the whole EP to CPU).
# Those options still work, they just take one fixed file per session instead
# of auto-keying a whole directory, hence _compiled_model_options() below doing
# by hand what migraphx_model_cache_dir would otherwise do for it. Every other
# arch here is on the ROCm 7 base, which has migraphx_model_cache_dir.
NO_MODEL_CACHE_DIR_ARCHES = {"gfx803"}


def _model_cache_dir_supported():
    arch = _gpu_arch()
    if arch is None:
        return True
    return arch not in NO_MODEL_CACHE_DIR_ARCHES


def _asr_factory():
    # Imported lazily on the worker so non-ROCm containers never touch it.
    from . import whisper_faster

    return whisper_faster


def _migraphx_available():
    try:
        import onnxruntime as ort

        return "MIGraphXExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _rocm_ep_available():
    # Only gfx803's ORT build (1.21.1) still carries the plain kernel-based
    # ROCMExecutionProvider alongside MIGraphX (see rocm-migraphx-ort-builder's
    # gfx803/Dockerfile.gfx803: --use_rocm is passed there as a fallback for
    # graphs MIGraphX can't compile - gfx803 has no CK/MLIR to fuse with). Later
    # ORT builds (the ROCm 7 base used for gfx9xx/gfx1030+) dropped it, so this
    # check alone scopes the fallback below to where it actually exists.
    try:
        import onnxruntime as ort

        return "ROCMExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _faster_whisper_available():
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        return False


def _cache_dir(fp16):
    # MIGraphX keys its .mxr artifacts on MIGraphX version, graph id, GPU arch and
    # input shapes - not on precision. An fp32 artifact would therefore be loaded
    # as a hit after switching fp16 on, silently running the wrong precision. One
    # subdir per precision keeps both sets valid instead of wiping on every flip.
    cache_dir = os.path.join(MIGRAPHX_CACHE_ROOT, "fp16" if fp16 else "fp32")
    # save_compiled_model() in the MIGraphX EP has no error handling, so a missing
    # directory raises out of session creation rather than just skipping the save.
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _compiled_model_options(model_label, fp16):
    # gfx803 fallback for _model_cache_dir(): migraphx_save/load_compiled_model
    # each take one literal file, not a directory MIGraphX can key by graph id
    # itself - so this does that keying by hand, one file per (model, precision).
    # Existence of the file is the only signal available: first run for a given
    # model/precision writes it (save), every run after reads it back (load).
    # There's no hash-of-the-graph check like migraphx_model_cache_dir does, so
    # a stale file from a since-changed model/shape would load wrong - delete
    # the file (or the whole fp16/fp32 subdir) to force a recompile.
    path = os.path.join(_cache_dir(fp16), f"{model_label}.mxr")
    if os.path.exists(path):
        return {
            "migraphx_load_compiled_model": "True",
            "migraphx_load_model_path": path,
        }
    return {
        "migraphx_save_compiled_model": "True",
        "migraphx_save_model_path": path,
    }


def register(ctx):
    # Guard: this plugin only does anything on the ROCm worker image. On the CPU
    # or CUDA image the runtime is absent, so register nothing and leave analysis
    # exactly as it was rather than offering a provider that can't load.
    if not _migraphx_available():
        logger.warning(
            "MIGraphXExecutionProvider not available - ROCm Accelerator stays inert. "
            "Install this plugin on the AudioMuse-AI ROCm worker image."
        )
        return

    # fp16 defaults on. A GPU page fault (mul_add_kernel / convert_mul_add_kernel)
    # has been seen on gfx1201 (RX 9070 XT / RDNA4) during MusiCNN/CLAP inference
    # with fp16 both on and off, so it isn't fp16-specific - disabling fp16 buys
    # no safety, just throughput. Opt out via the plugin's settings if needed.
    fp16 = get_setting("fp16_enable", True)
    if fp16 and not _fp16_capable():
        logger.warning(
            "GPU arch %s has no packed FP16 throughput - ignoring fp16_enable "
            "and running MIGraphX in fp32.",
            _gpu_arch(),
        )
        fp16 = False
    options = {"device_id": 0}
    if fp16:
        options["migraphx_fp16_enable"] = "True"

    # needs_static_shapes tells core to pin the CLAP audio model's symbolic time
    # axis before it builds the session; MIGraphX cannot compile a dynamic dim.
    # Core does not know this provider's name, so without the flag CLAP would be
    # handed the dynamic graph and fail to compile.
    if _model_cache_dir_supported():
        options["migraphx_model_cache_dir"] = _cache_dir(fp16)
        ctx.register_onnx_provider(
            "MIGraphXExecutionProvider",
            options,
            only_models=["musicnn", "clap"],
            needs_static_shapes=True,
        )
    else:
        # No migraphx_model_cache_dir on this EP build: fall back to explicit
        # save/load-compiled-model file paths, which core's options dict is
        # per-registration, not per-session - so each model needs its own
        # registration (and its own file) rather than one shared dict, or
        # musicnn and clap would fight over the same compiled-model file.
        # resolve_providers() in core only matches the entry whose only_models
        # names the session's label, so registering the same provider name
        # twice like this is safe - each session only ever sees one of them.
        for label in ("musicnn", "clap"):
            model_options = dict(options)
            model_options.update(_compiled_model_options(label, fp16))
            ctx.register_onnx_provider(
                "MIGraphXExecutionProvider",
                model_options,
                only_models=[label],
                needs_static_shapes=True,
            )
        logger.warning(
            "GPU arch %s's MIGraphX EP predates migraphx_model_cache_dir - "
            "using migraphx_save/load_compiled_model per model instead "
            "(delete /app/.cache/migraphx to force a recompile).",
            _gpu_arch(),
        )
    logger.info(
        "Registered MIGraphX ONNX provider for musicnn and CLAP audio (AMD GPU, fp16=%s)",
        options.get("migraphx_fp16_enable", "0"),
    )

    # CLAP audio's Resize node exports an explicit keep_aspect_ratio_policy
    # attribute (opset-19 exporter behavior). gfx803's MIGraphX is pinned to
    # release/rocm-rel-6.4 (see MIGRAPHX_REF in rocm-migraphx-ort-builder's
    # gfx803/Dockerfile.gfx803), whose ONNX parser throws on that attribute's
    # mere presence regardless of value or static/dynamic shape
    # (parse_resize.cpp: "keep_aspect_ratio_policy is not supported!") - so
    # CLAP can never compile there; only musicnn benefits from the entry
    # above. This is fixed upstream on MIGraphX's develop branch (stretch/
    # absent now accepted), which is what the ROCm 7 base for gfx9xx/gfx1030+
    # builds against - CLAP compiles fine there, e.g. RX 9070 XT/gfx1201.
    # Where the plain kernel-based ROCMExecutionProvider is available (gfx803's
    # ORT build only - see _rocm_ep_available), register it as a CLAP-only
    # fallback: it runs Resize as an ordinary op instead of parsing the graph
    # ahead of time, so the attribute is a non-issue. No fp16/static-shape
    # options needed - it handles dynamic dims natively.
    if _rocm_ep_available():
        ctx.register_onnx_provider(
            "ROCMExecutionProvider",
            {"device_id": 0},
            only_models=["clap"],
        )
        logger.info("Registered ROCMExecutionProvider fallback for CLAP audio (AMD GPU)")

    if _faster_whisper_available():
        ctx.register_analysis_provider("asr", _asr_factory)
        logger.info("Registered faster-whisper as the ASR backend (AMD GPU)")
    else:
        logger.warning(
            "faster_whisper not importable - lyrics ASR stays on the ONNX backend (CPU). "
            "musicnn acceleration is unaffected."
        )
