# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""parakeet.cpp ASR backend for the lyrics pipeline (NVIDIA Parakeet-TDT).

Registered as core's ``asr`` analysis provider, same contract
``whisper_faster.py`` implements: ``load_whisper_model`` / ``transcribe`` /
``is_loaded`` / ``unload`` / ``reset_session``, same return shape.

Same CLI-per-call design as ``whisper_cpp_backend.py`` - see that module's
docstring for why. The HIP variant is confirmed broken on gfx803 (silent
empty output, see ``docs/ASR_BACKENDS.md``); ``Gfx803Profile.blocked_asr_backends``
refuses that combo before this module is ever reached.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("plugin.rocm_accelerator.parakeet_cpp_backend")

SAMPLE_RATE = 16000

_BIN_DIR = os.environ.get("LYRICS_PARAKEET_CPP_BIN_DIR", "/opt/asr-backends/parakeet-cpp").strip()
_MODEL = os.environ.get(
    "LYRICS_PARAKEET_CPP_MODEL", "/app/model/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf"
).strip()

_variant = "vulkan"
_validated = False


class ParakeetCppLoadRefused(RuntimeError):
    """Raised when the binary/model can't be used; transcribe() degrades to empty."""


def configure(variant: str) -> None:
    global _variant, _validated
    if variant != _variant:
        _validated = False
    _variant = variant or "vulkan"


def _binary_path() -> str:
    return os.path.join(_BIN_DIR, f"parakeet-cli-{_variant}")


def available(variant: Optional[str] = None) -> bool:
    binary = os.path.join(_BIN_DIR, f"parakeet-cli-{variant or _variant}")
    # size check, not just isfile(): the HIP build is skipped at image-build
    # time on gfx803 (confirmed broken) and an empty placeholder is copied in
    # its place instead - see docker/Dockerfile's parakeet-cpp-hip stage.
    return os.path.isfile(binary) and os.path.getsize(binary) > 0 and os.path.isfile(_MODEL)


def load_whisper_model():
    global _validated
    if _validated:
        return True
    if not available():
        raise ParakeetCppLoadRefused(
            f"parakeet-cli binary or model missing (bin={_binary_path()!r}, model={_MODEL!r})"
        )
    _validated = True
    logger.info("parakeet.cpp ready (variant=%s, model=%s)", _variant, _MODEL)
    return True


def _run(wav_path: str, language: Optional[str]) -> subprocess.CompletedProcess:
    cmd = [
        _binary_path(), "transcribe",
        "--model", _MODEL, "--input", wav_path,
        "--decoder", "tdt", "--json",
    ]
    if language:
        cmd += ["--lang", language]
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
    except ParakeetCppLoadRefused as exc:
        logger.warning("parakeet.cpp load refused: %s", exc)
        return {"text": "", "language": "", "duration": duration}

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE, subtype="PCM_16")
        try:
            proc = _run(tmp.name, language)
        except Exception as exc:
            logger.warning("parakeet.cpp subprocess failed: %s", exc)
            return {"text": "", "language": "", "duration": duration}

    if proc.returncode != 0:
        logger.warning(
            "parakeet.cpp exited %d: %s", proc.returncode, proc.stderr.strip()[-2000:]
        )
        return {"text": "", "language": "", "duration": duration}

    # Output is one JSON object on stdout: {"text": "...", "frame_sec": ...,
    # "words": [...], "tokens": [...]} - verified directly against this exact
    # binary/flags in local-test/asr_backends/parakeet_cpp.sh.
    try:
        payload = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        logger.warning("parakeet.cpp produced non-JSON stdout: %r", proc.stdout[-2000:])
        return {"text": "", "language": "", "duration": duration}

    text = (payload.get("text") or "").strip()
    words = payload.get("words") or []

    # No avg_logprob key: parakeet.cpp's per-word "conf" is a 0-1 confidence
    # score, not a log-probability - core's quality gate expects the latter's
    # scale (see whisper_faster.py), so passing "conf" under that key would
    # misrepresent it rather than just being absent.
    result = {
        "text": text,
        # parakeet.cpp's JSON carries no detected-language field (the model is
        # multilingual with automatic detection, but the CLI doesn't surface
        # it) - pass through the caller's hint if any, otherwise unknown.
        "language": language or "",
        "duration": duration,
    }
    logger.info(
        "parakeet.cpp (%s): %.1fs audio (%d words)",
        _variant, result["duration"], len(words),
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
