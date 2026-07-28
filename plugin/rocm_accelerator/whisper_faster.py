# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""faster-whisper (CTranslate2) ASR backend, shipped by the ROCm Accelerator plugin.

Drop-in replacement for core's whisper_onnx, registered via
``ctx.register_analysis_provider('asr', ...)``. Used on the AMD/ROCm image because
MIGraphX can't parse the ONNX Whisper decoder's dynamic If/KV-cache subgraphs;
CTranslate2 has a native ROCm HIP backend instead. Mirrors whisper_onnx's public
surface: load_whisper_model / transcribe / is_loaded / unload.

The faster-whisper and CTranslate2-ROCm libraries and the model come from the ROCm
worker image, not from the plugin's pip requirements (a PyPI onnxruntime would
clobber the image's MIGraphX build).

Main Features:
* Lazy, thread-safe model load with a CPU/int8 fallback when the GPU load fails,
  kept as a module singleton so core resolves the backend once per worker.
* transcribe returns core's dict contract (text/language/duration/avg_logprob),
  degrading to empty text instead of failing the run when the model is missing.
* is_loaded / unload mirror whisper_onnx so the per-album memory release works.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("plugin.rocm_accelerator.whisper_faster")

SAMPLE_RATE = 16000


def _beam_size() -> int:
    # config owns this setting; the fallback only covers core being unimportable.
    try:
        from plugin.api import config

        return int(config.LYRICS_ASR_BEAM_SIZE)
    except Exception:
        return 5


# CTranslate2 mirrors the CUDA API on ROCm, so device="cuda" targets an AMD GPU.
_DEVICE = os.environ.get("LYRICS_WHISPER_FASTER_DEVICE", "cuda").strip() or "cuda"
_COMPUTE_TYPE = os.environ.get("LYRICS_WHISPER_FASTER_COMPUTE_TYPE", "float16").strip() or "float16"
_MODEL_DIR = os.environ.get(
    "LYRICS_WHISPER_FASTER_MODEL_DIR", "/app/model/faster-whisper-small"
).strip()

_model = None
_model_dir: Optional[str] = None
_load_lock = threading.Lock()


class WhisperLoadRefused(RuntimeError):
    """Raised when the model cannot be loaded; transcribe() degrades to empty."""


def load_whisper_model():
    global _model, _model_dir
    if _model is not None and _model_dir == _MODEL_DIR:
        return _model
    with _load_lock:
        if _model is not None and _model_dir == _MODEL_DIR:
            return _model
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # library missing on non-ROCm images
            raise WhisperLoadRefused(f"faster_whisper import failed: {exc}") from exc

        model_src = _MODEL_DIR if os.path.isdir(_MODEL_DIR) else "small"
        try:
            _model = WhisperModel(model_src, device=_DEVICE, compute_type=_COMPUTE_TYPE)
        except Exception as exc:
            # Fall back to CPU/int8 rather than failing the whole lyrics run.
            logger.warning(
                "faster-whisper GPU load failed (device=%s, compute=%s): %s - "
                "falling back to CPU/int8",
                _DEVICE,
                _COMPUTE_TYPE,
                exc,
            )
            try:
                _model = WhisperModel(model_src, device="cpu", compute_type="int8")
            except Exception as exc2:
                raise WhisperLoadRefused(
                    f"faster_whisper load failed on GPU and CPU: {exc2}"
                ) from exc2
        _model_dir = _MODEL_DIR
        logger.info(
            "faster-whisper loaded (src=%s, device=%s, compute=%s)",
            model_src,
            _DEVICE,
            _COMPUTE_TYPE,
        )
        return _model


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
        model = load_whisper_model()
    except WhisperLoadRefused as exc:
        logger.warning("faster-whisper load refused: %s", exc)
        return {"text": "", "language": "", "duration": duration, "avg_logprob": float("-inf")}

    beam_size = _beam_size()
    # language=None lets faster-whisper auto-detect; VAD is done upstream
    # (silero_onnx), so leave faster-whisper's vad_filter off.
    segments, info = model.transcribe(
        audio,
        language=language or None,
        beam_size=beam_size,
        vad_filter=False,
    )

    texts = []
    logprobs = []
    try:
        for seg in segments:  # generator - consuming it runs the decode
            t = (seg.text or "").strip()
            if t:
                texts.append(t)
            lp = getattr(seg, "avg_logprob", None)
            if lp is not None:
                logprobs.append(float(lp))
    except Exception:
        # Decode runs lazily inside the generator; a mid-stream failure (GPU OOM,
        # CTranslate2 error) keeps whatever segments already decoded.
        logger.exception(
            "faster-whisper decode failed after %d segment(s) - returning partial text",
            len(texts),
        )

    full_text = " ".join(texts).strip()
    detected = getattr(info, "language", "") or ""
    avg_logprob = float(np.mean(logprobs)) if logprobs else float("-inf")
    info_dur = getattr(info, "duration", None)
    logger.info(
        "faster-whisper: %.1fs audio (lang=%r, beam=%d, avg_logprob=%.2f)",
        info_dur if info_dur else duration,
        detected,
        beam_size,
        avg_logprob,
    )
    return {
        "text": full_text,
        "language": detected,
        "duration": float(info_dur) if info_dur else duration,
        "avg_logprob": avg_logprob,
    }


def is_loaded() -> bool:
    return _model is not None


def unload() -> bool:
    global _model, _model_dir
    if _model is None and _model_dir is None:
        return False
    model = _model
    _model = None
    _model_dir = None
    try:
        del model
    except Exception:
        logger.exception("Error dropping faster-whisper model")
    try:
        import gc

        gc.collect()
    except Exception:
        logger.exception("Error during faster-whisper GC")
    try:
        from tasks.memory_utils import comprehensive_memory_cleanup

        comprehensive_memory_cleanup(force_cuda=False, reset_onnx_pool=False)
    except Exception:
        logger.exception("Error during memory cleanup on faster-whisper unload")
    logger.info("faster-whisper: model unloaded")
    return True


def reset_session() -> None:
    unload()
