# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""ROCm Accelerator plugin for AudioMuse-AI.

Runs the analysis models AMD GPUs can handle on the GPU, using two core seams:

* ``register_onnx_provider`` for musicnn and the CLAP audio encoder, via ONNX
  Runtime's MIGraphX provider. Scoped to those labels because the Whisper
  decoder's graph cannot be compiled by MIGraphX at all.
* ``register_analysis_provider('asr', ...)`` to swap lyrics ASR to
  faster-whisper, for the same reason.

Anything that has to differ per GPU generation lives in ``arch/``, not here.
The ROCm runtime itself comes from the ROCm worker image; on any other image
this registers nothing.
"""

import logging

from plugin.api import get_setting

from . import arch, cache, gpu
from .providers import MIGRAPHX

logger = logging.getLogger("plugin.rocm_accelerator")


def _asr_factory():
    # Imported lazily so non-ROCm images never load a faster-whisper backend
    # they have no libraries for.
    from . import whisper_faster

    return whisper_faster


def _resolve_fp16(profile, arch_name):
    fp16 = bool(get_setting("fp16_enable", True))
    if fp16 and not profile.fp16_supported:
        logger.warning(
            "GPU arch %s gains nothing from fp16 - ignoring fp16_enable and "
            "running MIGraphX in fp32.", arch_name,
        )
        return False
    return fp16


def _register_migraphx(ctx, profile, labels, fp16):
    options = {"device_id": 0}
    if fp16:
        options["migraphx_fp16_enable"] = "True"
    options.update(profile.migraphx_options())

    if profile.supports_model_cache_dir:
        options.update(cache.cache_dir_options(fp16))
        # needs_static_shapes makes core pin CLAP's symbolic time axis before
        # building the session; MIGraphX cannot compile a dynamic dim.
        ctx.register_onnx_provider(
            MIGRAPHX, options, only_models=list(labels), needs_static_shapes=True,
        )
        return

    # One registration per label, because the cache options are per-file and
    # core keeps one options dict per registration: sharing a dict would point
    # every model at the same compiled-model file. Core matches a session only
    # against the registration naming its label, so the repeated provider name
    # never puts two of these in one chain.
    for label in labels:
        model_options = dict(options)
        model_options.update(cache.per_model_options(label, fp16))
        ctx.register_onnx_provider(
            MIGRAPHX, model_options, only_models=[label], needs_static_shapes=True,
        )
    logger.info(
        "MIGraphX compiled-model cache: one file per model under %s "
        "(delete it to force a recompile)", cache.CACHE_ROOT,
    )


def register(ctx):
    providers = gpu.available_providers()
    if MIGRAPHX not in providers:
        logger.warning(
            "MIGraphXExecutionProvider not available - ROCm Accelerator stays inert. "
            "Install this plugin on the AudioMuse-AI ROCm worker image."
        )
        return

    arch_name = gpu.detect_arch()
    profile = arch.profile_for(arch_name)
    logger.info("GPU arch %s -> %s", arch_name or "unknown", profile)

    applied_env = arch.apply_env(profile)
    if applied_env:
        logger.info("%s set %s", profile, applied_env)

    fp16 = _resolve_fp16(profile, arch_name or "unknown")

    labels = profile.migraphx_models(providers)
    if labels:
        _register_migraphx(ctx, profile, labels, fp16)
        logger.info(
            "Registered MIGraphX ONNX provider for %s (AMD GPU, fp16=%s)",
            ", ".join(labels), fp16,
        )

    for spec in profile.extra_providers(providers):
        if spec.name not in providers:
            logger.warning(
                "%s asked for %s, which this onnxruntime build does not have - skipping",
                profile, spec.name,
            )
            continue
        ctx.register_onnx_provider(
            spec.name,
            dict(spec.options),
            only_models=list(spec.only_models) or None,
            needs_static_shapes=spec.needs_static_shapes,
        )
        logger.info(
            "Registered %s for %s (AMD GPU)",
            spec.name, ", ".join(spec.only_models) or "all models",
        )

    if gpu.faster_whisper_available():
        ctx.register_analysis_provider("asr", _asr_factory)
        logger.info("Registered faster-whisper as the ASR backend (AMD GPU)")
    else:
        logger.warning(
            "faster_whisper not importable - lyrics ASR stays on the ONNX backend (CPU). "
            "musicnn acceleration is unaffected."
        )
