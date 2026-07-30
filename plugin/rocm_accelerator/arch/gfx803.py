# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Polaris / GCN 4. Every deviation here is forced by its ROCm 6 based image.

Full findings behind each one: ``docs/ARCH_NOTES.md``.
"""

from ..providers import ROCM
from .base import ArchProfile, ProviderSpec


class Gfx803Profile(ArchProfile):
    arches = frozenset({"gfx803", "gfx802", "gfx805"})

    # GCN 4 has no packed FP16: fp16 math runs at a fraction of the fp32 rate,
    # so enabling it costs precision for no speedup.
    fp16_supported = False

    # This image's onnxruntime predates the migraphx_model_cache_dir option;
    # passing it fails session creation and drops the whole EP to CPU.
    supports_model_cache_dir = False

    # parakeet.cpp's HIP build returns a silent empty transcript on this arch
    # (exit 0, no exception) - confirmed via local-test/asr_backends/parakeet_cpp.sh,
    # while the same binary on Vulkan and whisper.cpp on either backend both
    # produce correct transcripts. See docs/ASR_BACKENDS.md for the full matrix.
    blocked_asr_backends = frozenset({("parakeet_cpp", "hip")})

    def migraphx_models(self, providers):
        # CLAP's audio graph has a Resize node this MIGraphX release refuses to
        # parse, so MIGraphX could never serve CLAP here - and pairing it with
        # the ROCM EP in one session segfaults the worker. Where the ROCM EP
        # exists, CLAP goes there alone (extra_providers) and MIGraphX keeps
        # musicnn only.
        if ROCM in providers:
            return ("musicnn",)
        return super().migraphx_models(providers)

    def extra_providers(self, providers):
        # musicnn deliberately does not get the ROCM EP: MIOpen's fusion path
        # for Conv+Bias+Activation corrupts GPU state on this arch and faults
        # non-deterministically. CLAP has no fused conv and has proven stable.
        if ROCM not in providers:
            return ()
        return (ProviderSpec(ROCM, {"device_id": 0}, only_models=("clap",)),)
