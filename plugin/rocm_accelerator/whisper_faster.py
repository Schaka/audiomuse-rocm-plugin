# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""faster-whisper (CTranslate2) ASR backend for the lyrics pipeline.

Registered as core's ``asr`` analysis provider and so bound to that contract:
``load_whisper_model`` / ``transcribe`` / ``is_loaded`` / ``unload``, with
``transcribe`` returning ``text``, ``language``, ``duration`` and - only when a
confidence is actually known - ``avg_logprob``.

Main Features:
* Lazy, thread-safe model load with a CPU fallback when the GPU load fails,
  kept as a module singleton so one model serves a whole album.
* transcribe degrades to empty text rather than failing a run.

The libraries and the model come from the ROCm worker image; this module only
uses them.
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


def _default_compute_type() -> str:
    # gfx803/802/805 (Polaris) have no packed FP16 throughput - same finding
    # already applied to MIGraphX (see arch.gfx803.Gfx803Profile.fp16_supported).
    # CTranslate2's fp16 GEMM path on that arch doesn't just run slow, it trips
    # a spurious "out of memory" from the HIP allocator instead of decoding, so
    # this can't be left defaulted to float16 there like every other arch.
    from . import arch, gpu

    profile = arch.profile_for(gpu.detect_arch())
    return "float16" if profile.fp16_supported else "int8_float32"


# CTranslate2 mirrors the CUDA API on ROCm, so device="cuda" targets an AMD GPU.
_DEVICE = os.environ.get("LYRICS_WHISPER_FASTER_DEVICE", "cuda").strip() or "cuda"
_COMPUTE_TYPE = os.environ.get("LYRICS_WHISPER_FASTER_COMPUTE_TYPE", "").strip() or _default_compute_type()
_MODEL_DIR = os.environ.get(
    "LYRICS_WHISPER_FASTER_MODEL_DIR", "/app/model/faster-whisper-small"
).strip()

_model = None
_model_dir: Optional[str] = None
_load_lock = threading.Lock()


def _vram_status() -> str:
    # Asks HIP what it thinks is free, independent of CTranslate2's own
    # allocator - the number that tells a real OOM apart from an allocator
    # reporting one on a card with VRAM to spare.
    try:
        import torch

        if not torch.cuda.is_available():
            return "vram: no CUDA/HIP device visible to torch"
        free_b, total_b = torch.cuda.mem_get_info(0)
        return f"vram: {free_b / 2**20:.0f}MiB free / {total_b / 2**20:.0f}MiB total"
    except Exception as exc:
        return f"vram: unavailable ({exc})"


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
        loaded_device, loaded_compute = _DEVICE, _COMPUTE_TYPE
        try:
            _model = WhisperModel(model_src, device=_DEVICE, compute_type=_COMPUTE_TYPE)
        except Exception as exc:
            # Fall back to CPU rather than failing the whole lyrics run. int8 is
            # tried first for speed, but CTranslate2 only has an int8 CPU kernel
            # when built against oneDNN/MKL, so float32 has to back it up.
            # VRAM is logged because a spurious "out of memory" from the HIP
            # allocator is otherwise indistinguishable from a genuinely full card.
            logger.warning(
                "faster-whisper GPU load failed (device=%s, compute=%s): %s - "
                "falling back to CPU (%s)",
                _DEVICE,
                _COMPUTE_TYPE,
                exc,
                _vram_status(),
            )
            for cpu_compute in ("int8", "float32"):
                try:
                    _model = WhisperModel(model_src, device="cpu", compute_type=cpu_compute)
                    loaded_device, loaded_compute = "cpu", cpu_compute
                    break
                except Exception as exc2:
                    last_cpu_exc = exc2
                    logger.warning(
                        "faster-whisper CPU load failed (compute=%s): %s", cpu_compute, exc2
                    )
            else:
                raise WhisperLoadRefused(
                    f"faster_whisper load failed on GPU and CPU: {last_cpu_exc}"
                ) from last_cpu_exc
        _model_dir = _MODEL_DIR
        logger.info(
            "faster-whisper loaded (src=%s, device=%s, compute=%s)",
            model_src,
            loaded_device,
            loaded_compute,
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
        return {"text": "", "language": "", "duration": duration}

    beam_size = _beam_size()
    try:
        # language=None lets faster-whisper auto-detect; VAD is done upstream
        # (silero_onnx), so leave faster-whisper's vad_filter off. With
        # language=None this call itself runs detect_language's encode pass
        # eagerly (before the segments generator even exists), so a GPU
        # failure here has to be caught same as the segment loop below - the
        # generator's own try/except never sees it.
        segments, info = model.transcribe(
            audio,
            language=language or None,
            beam_size=beam_size,
            vad_filter=False,
        )
    except Exception as exc:
        logger.warning(
            "faster-whisper transcribe failed (%s) - %s", exc, _vram_status()
        )
        return {"text": "", "language": "", "duration": duration}

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
        # CTranslate2 error) keeps whatever segments already decoded. VRAM is
        # logged because a spurious "out of memory" from the HIP allocator is
        # otherwise indistinguishable from a genuinely full card.
        logger.exception(
            "faster-whisper decode failed after %d segment(s) - %s - returning partial text",
            len(texts),
            _vram_status(),
        )

    info_dur = getattr(info, "duration", None)
    result = {
        "text": " ".join(texts).strip(),
        "language": getattr(info, "language", "") or "",
        "duration": float(info_dur) if info_dur else duration,
    }
    # Omitted, not faked, when no segment reported one: core skips its quality
    # gate on a missing avg_logprob but would fail every transcript on a
    # placeholder value.
    if logprobs:
        result["avg_logprob"] = float(np.mean(logprobs))
    logger.info(
        "faster-whisper: %.1fs audio (lang=%r, beam=%d, avg_logprob=%s)",
        result["duration"],
        result["language"],
        beam_size,
        f"{result['avg_logprob']:.2f}" if "avg_logprob" in result else "n/a",
    )
    return result


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
