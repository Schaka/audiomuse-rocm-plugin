# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""gfx1201 (RDNA4: RX 9070 / XT). Full findings: ``docs/ARCH_NOTES.md``."""

from types import MappingProxyType

from .base import ArchProfile


class Gfx1201Profile(ArchProfile):
    arches = frozenset({"gfx1201"})

    # CTranslate2's default CUDA/HIP allocator (cuda_malloc_async, HIP's
    # hipMallocAsync/hipFreeAsync pool) triggers a GPU page fault mid-transcribe
    # on this arch - an upstream ROCm LLVM codegen bug, not CTranslate2 itself.
    # See OpenNMT/CTranslate2#2021. cub_caching sidesteps it.
    env = MappingProxyType({"CT2_CUDA_ALLOCATOR": "cub_caching"})
