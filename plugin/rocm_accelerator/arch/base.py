# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The arch profile contract: defaults every GPU arch gets, and what may differ.

A profile is pure declaration - it decides *what* to ask for, never how to
register it. See ``docs/ARCH_PROFILES.md`` for how to write one.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence, Tuple

# Session labels the MIGraphX provider is offered for by default. Labels are
# defined by core; anything else is ignored with a warning on its side.
DEFAULT_MIGRAPHX_MODELS = ("musicnn", "clap")

_NO_ENV: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True)
class ProviderSpec:
    """An extra ONNX provider a profile wants registered besides MIGraphX."""

    name: str
    options: Mapping[str, str] = field(default_factory=dict)
    only_models: Tuple[str, ...] = ()
    needs_static_shapes: bool = False


class ArchProfile:
    """What one GPU arch needs differently. The base class is the default path.

    Instantiated once per worker startup. Subclasses override only the members
    that actually differ for their arch and leave the rest inherited, so a new
    quirk stays a few lines and the defaults have exactly one definition.
    """

    #: Arch strings (as reported by rocminfo) this profile applies to.
    arches: frozenset = frozenset()

    #: Environment applied before onnxruntime or CTranslate2 are imported, for
    #: knobs those libraries only read at import time. A variable already set
    #: from outside is left alone. Read-only here so the inherited default
    #: cannot be mutated into every other profile; override it with a plain dict.
    env: Mapping[str, str] = _NO_ENV

    #: False where fp16 math buys no throughput, which makes the plugin's
    #: fp16_enable setting pure precision risk and so ignored.
    fp16_supported: bool = True

    #: False on EP builds without the migraphx_model_cache_dir option, which
    #: fails session creation outright when passed.
    supports_model_cache_dir: bool = True

    #: (asr_backend, asr_backend_variant) pairs known to fail on this arch
    #: without raising anything catchable - e.g. parakeet.cpp's HIP build
    #: returns a silent empty transcript on gfx803 (exit 0, no exception),
    #: which the normal "GPU load failed, fall back" path never sees. Refused
    #: at settings resolution instead of left to run and quietly produce no
    #: lyrics. See docs/ASR_BACKENDS.md for the findings behind each entry.
    blocked_asr_backends: frozenset = frozenset()

    def migraphx_options(self) -> Mapping[str, str]:
        """Extra MIGraphX EP options, merged over the ones built generically."""
        return {}

    def migraphx_models(self, providers: Sequence[str]) -> Tuple[str, ...]:
        """Session labels to offer MIGraphX for, given the available providers."""
        return DEFAULT_MIGRAPHX_MODELS

    def extra_providers(self, providers: Sequence[str]) -> Tuple[ProviderSpec, ...]:
        """Non-MIGraphX providers to register, given the available providers."""
        return ()

    def __str__(self) -> str:
        return type(self).__name__
