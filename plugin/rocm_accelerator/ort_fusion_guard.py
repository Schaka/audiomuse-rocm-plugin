# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Disable ONNX Runtime graph optimizers for sessions on a given provider.

Some of ORT's graph optimizers produce nodes whose GPU kernels are broken on a
specific arch: on gfx803, ``ConvActivationFusion`` emits FusedConv nodes that
the ROCM EP hands to MIOpen's Fusion Plan API, and those fused kernels
(``miopenSp3AsmConvRxSU_CBA``, ``MIOpenConvUniBatchNormActiv``) intermittently
read past the end of their weight buffers - a GPU page fault that aborts the
whole worker process. Unfused, the same convs are stable.

onnxruntime's only switch for this is the per-session config entry
``optimization.disable_specified_optimizers``; there is no environment
variable, and core builds its SessionOptions itself. Instead of a pass-through
seam in core's plugin API, the plugin wraps ``onnxruntime.InferenceSession``
at registration time: core resolves it lazily as a module attribute
(``import onnxruntime as ort`` + ``ort.InferenceSession(...)``), so every
analysis session built after ``register()`` goes through the guard. Sessions
whose provider chain doesn't name a guarded provider are passed through
untouched, which keeps the scope identical to the ProviderSpec that asked for
the guard (on gfx803: the ROCM EP, and with it exactly the CLAP session).
"""

import logging
from typing import Dict, Sequence, Tuple

logger = logging.getLogger(__name__)

_DISABLE_KEY = "optimization.disable_specified_optimizers"

# Provider name -> optimizer names to disable when a session uses it.
_rules: Dict[str, Tuple[str, ...]] = {}


def _provider_names(providers) -> Tuple[str, ...]:
    """Names from an InferenceSession ``providers`` list, which mixes plain
    strings and (name, options) tuples."""
    names = []
    for entry in providers or ():
        names.append(entry[0] if isinstance(entry, (tuple, list)) else entry)
    return tuple(names)


def _disabled_for(providers) -> Tuple[str, ...]:
    names = _provider_names(providers)
    disabled = []
    for provider, optimizers in _rules.items():
        if provider in names:
            for optimizer in optimizers:
                if optimizer not in disabled:
                    disabled.append(optimizer)
    return tuple(disabled)


def install(provider: str, optimizers: Sequence[str]) -> None:
    """Disable ``optimizers`` for every future session that uses ``provider``.

    Idempotent, and multiple calls accumulate rules into the one wrapper.
    """
    import onnxruntime as ort

    optimizers = tuple(optimizers)
    if not optimizers:
        return
    _rules[provider] = optimizers
    logger.info(
        "Sessions using %s will run without the %s graph optimizer(s)",
        provider, ", ".join(optimizers),
    )

    if getattr(ort.InferenceSession, "_rocm_accelerator_fusion_guard", False):
        return

    real_session = ort.InferenceSession

    class GuardedInferenceSession(real_session):
        _rocm_accelerator_fusion_guard = True

        def __init__(self, path_or_bytes, sess_options=None, providers=None,
                     provider_options=None, **kwargs):
            disabled = _disabled_for(providers)
            if disabled:
                if sess_options is None:
                    sess_options = ort.SessionOptions()
                sess_options.add_session_config_entry(
                    _DISABLE_KEY, ";".join(disabled))
            super().__init__(
                path_or_bytes,
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options,
                **kwargs,
            )

    GuardedInferenceSession.__name__ = real_session.__name__
    GuardedInferenceSession.__qualname__ = real_session.__qualname__
    ort.InferenceSession = GuardedInferenceSession
