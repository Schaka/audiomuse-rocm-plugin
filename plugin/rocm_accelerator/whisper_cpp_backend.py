# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""whisper.cpp ASR backend for the lyrics pipeline.

Registered as core's ``asr`` analysis provider, same contract
``whisper_faster.py`` implements: ``load_whisper_model`` / ``transcribe`` /
``is_loaded`` / ``unload`` / ``reset_session``, same return shape.

Unlike faster-whisper this wraps a compiled CLI binary (``whisper-cli``), not
an importable Python library, in one of two build variants (Vulkan or HIP -
see ``docs/ASR_BACKENDS.md``). There is no persistent in-process model to
hold: each ``transcribe()`` call shells out fresh. The CLI's own model-load
time is small (~400ms for the base model, measured in
``local-test/asr_backends/whisper_cpp_vulkan.sh``) and lyrics ASR runs once
per song, not at request-serving scale, so paying it per call is simpler than
managing a long-lived server process and its lifecycle.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("plugin.rocm_accelerator.whisper_cpp_backend")

SAMPLE_RATE = 16000
_TIMESTAMP_LINE = re.compile(r"^\[[\d:.]+\s*-->\s*[\d:.]+\]\s*(.*)$")
_LANG_LINE = re.compile(r"lang\s*=\s*([a-zA-Z-]+)")

_BIN_DIR = os.environ.get("LYRICS_WHISPER_CPP_BIN_DIR", "/opt/asr-backends/whisper-cpp").strip()
_MODEL = os.environ.get("LYRICS_WHISPER_CPP_MODEL", "/app/model/whisper-cpp/ggml-small.bin").strip()

_variant = "vulkan"
_validated = False


class WhisperCppLoadRefused(RuntimeError):
    """Raised when the binary/model can't be used; transcribe() degrades to empty."""


def configure(variant: str) -> None:
    global _variant, _validated
    if variant != _variant:
        _validated = False
    _variant = variant or "vulkan"


def _binary_path() -> str:
    return os.path.join(_BIN_DIR, f"whisper-cli-{_variant}")


def available(variant: Optional[str] = None) -> bool:
    binary = os.path.join(_BIN_DIR, f"whisper-cli-{variant or _variant}")
    return os.path.isfile(binary) and os.path.isfile(_MODEL)


def load_whisper_model():
    global _validated
    if _validated:
        return True
    if not available():
        raise WhisperCppLoadRefused(
            f"whisper-cli binary or model missing (bin={_binary_path()!r}, model={_MODEL!r})"
        )
    _validated = True
    logger.info("whisper.cpp ready (variant=%s, model=%s)", _variant, _MODEL)
    return True


def _run(wav_path: str, language: Optional[str]) -> subprocess.CompletedProcess:
    cmd = [_binary_path(), "-m", _MODEL, "-f", wav_path]
    if language:
        cmd += ["-l", language]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def transcribe(
    wav: np.ndarray, sr: int, language: Optional[str] = None
) -> Dict[str, object]:
    if sr != SAMPLE_RATE:
        import librosa

        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    audio = np.ascontiguousarray(wav, dtype=np.float32)
    duration = len(audio) / SAMPLE_RATE

    try:
        load_whisper_model()
    except WhisperCppLoadRefused as exc:
        logger.warning("whisper.cpp load refused: %s", exc)
        return {"text": "", "language": "", "duration": duration}

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE, subtype="PCM_16")
        try:
            proc = _run(tmp.name, language)
        except Exception as exc:
            logger.warning("whisper.cpp subprocess failed: %s", exc)
            return {"text": "", "language": "", "duration": duration}

    if proc.returncode != 0:
        logger.warning(
            "whisper.cpp exited %d: %s", proc.returncode, proc.stderr.strip()[-2000:]
        )
        return {"text": "", "language": "", "duration": duration}

    texts = []
    detected_lang = ""
    for line in proc.stdout.splitlines():
        m = _TIMESTAMP_LINE.match(line.strip())
        if m and m.group(1).strip():
            texts.append(m.group(1).strip())
        if not detected_lang:
            lm = _LANG_LINE.search(line)
            if lm:
                detected_lang = lm.group(1)

    result = {
        "text": " ".join(texts).strip(),
        "language": language or detected_lang,
        "duration": duration,
    }
    logger.info(
        "whisper.cpp (%s): %.1fs audio (lang=%r)",
        _variant, result["duration"], result["language"],
    )
    return result


def is_loaded() -> bool:
    return _validated


def unload() -> bool:
    global _validated
    was_validated = _validated
    _validated = False
    return was_validated


def reset_session() -> None:
    unload()
