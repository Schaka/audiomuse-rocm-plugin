# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""GPU arch detection and ONNX Runtime capability probes."""

import logging
import subprocess
from typing import Optional, Tuple

logger = logging.getLogger("plugin.rocm_accelerator.gpu")

_ROCMINFO_TIMEOUT = 10


def detect_arch() -> Optional[str]:
    """Return this machine's GPU arch (``"gfx1030"``, ...), or None if unknown.

    Parses ``rocminfo`` output rather than asking torch/HIP, because this runs in
    a process that later fork()s the workers doing the actual inference: a HIP
    context created here does not survive the fork, and the children would fail
    their first GPU call with a handle that looks initialized but is not. A
    separate process cannot leak a context into this one.
    """
    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True,
            timeout=_ROCMINFO_TIMEOUT, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        logger.debug("rocminfo unavailable - GPU arch unknown", exc_info=True)
        return None
    # rocminfo lists every agent, CPUs included; only GPU agents name a gfx ISA.
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
            if name.startswith("gfx"):
                return name
    return None


def available_providers() -> Tuple[str, ...]:
    """The execution providers this image's ONNX Runtime was built with."""
    try:
        import onnxruntime as ort

        return tuple(ort.get_available_providers())
    except Exception:
        logger.debug("onnxruntime not importable", exc_info=True)
        return ()


def faster_whisper_available() -> bool:
    """Whether the image ships a usable faster-whisper.

    Broad on purpose: a half-built CTranslate2 raises out of its extension
    module rather than an ImportError, and either way it cannot be used.
    """
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        return False
