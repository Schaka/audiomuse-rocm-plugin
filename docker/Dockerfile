# syntax=docker/dockerfile:1
# AudioMuse-AI ROCm (AMD GPU) worker image - built in the plugin's own repo.
#
# App code, analysis models and HF cache all come from the published core
# image (CORE_IMAGE) - this repo carries no AudioMuse-AI source. The AMD
# behaviour itself is supplied at runtime by the ROCm Accelerator plugin
# (../plugin/rocm_accelerator), installed separately via the Plugins UI.
#
# This image's only job is the ROCm runtime the plugin can't install itself:
#   * onnxruntime with MIGraphXExecutionProvider (from BASE_IMAGE)
#   * CTranslate2's ROCm build + faster-whisper (installed below)
#   * the faster-whisper model, GPU env vars, ROCm system libs
#
# Build + run: see ../local-test/docker-compose-rocm.yaml

# Defaults are for a local one-off build only; CI always passes both explicitly
# (BASE_IMAGE per arch, CORE_IMAGE pinned to an upstream digest or version).
ARG BASE_IMAGE=ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx1201
ARG CORE_IMAGE=ghcr.io/neptunehub/audiomuse-ai:latest

# ============================================================================
# Stage 1: core - the published AudioMuse-AI image (app code + models + HF cache)
# ============================================================================
FROM ${CORE_IMAGE} AS core

# ============================================================================
# Stage 2: runner - ROCm base (MIGraphX ONNX Runtime) + app deps
# ============================================================================
FROM ${BASE_IMAGE} AS runner

SHELL ["/bin/bash", "-c"]

# Runtime libs + CLI tools the entrypoint/supervisord/debugging rely on. No
# version pins: the ROCm base's package set differs from the plain image.
RUN set -ux; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            python3 python3-pip python3-venv \
            libfftw3-double3 libyaml-0-2 libsamplerate0 libsndfile1 \
            libopenblas0 liblapack3 libgomp1 libpq5 \
            ffmpeg wget curl supervisor procps \
            git vim redis-tools strace iputils-ping \
            postgresql-common ca-certificates tzdata \
            && /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y \
            && DEBIAN_FRONTEND=noninteractive apt-get update \
            && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends postgresql-client-18; then \
            break; \
        fi; \
        n=$((n+1)); \
        echo "apt-get attempt $n failed - retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    rm -rf /var/lib/apt/lists/* && \
    apt-get remove -y python3-numpy || true && \
    apt-get autoremove -y || true && \
    rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY requirements/rocm.txt /app/requirements/rocm.txt

# Install into the base image's venv (/opt/venv), which already has onnxruntime
# built with MIGraphXExecutionProvider. Target the venv interpreter explicitly
# (not system python3.12) so deps land where the app actually runs; onnxruntime
# stays out of rocm.txt so it's never reinstalled/clobbered. common.txt comes
# from the core image (already installed there); only rocm.txt's extras
# (currently just tqdm) need adding on top.
ENV VIRTUAL_ENV=/opt/venv
RUN set -eux; \
    uv pip install --python "$VIRTUAL_ENV/bin/python3" --no-cache --index-strategy unsafe-best-match \
        -r /app/requirements/rocm.txt; \
    "$VIRTUAL_ENV/bin/python3" -c "import onnxruntime as ort; p=ort.get_available_providers(); print('ORT providers:', p); assert 'MIGraphXExecutionProvider' in p, 'MIGraphX EP missing'"; \
    find "$VIRTUAL_ENV/lib" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# faster-whisper on the AMD GPU via CTranslate2's ROCm HIP backend. The ROCm
# Accelerator plugin registers this as the ASR backend; the libs must be present
# in the image because the plugin ships no pip requirements. The wheel is a
# GitHub release asset (not on PyPI). Installed --no-deps: its onnxruntime
# dependency would otherwise clobber the base MIGraphX ORT with a CPU-only build.
ARG CT2_VERSION=4.8.1
RUN set -eux; \
    cd /tmp; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if wget "https://github.com/OpenNMT/CTranslate2/releases/download/v${CT2_VERSION}/rocm-python-wheels-Linux.zip"; then \
            break; \
        fi; \
        rm -f rocm-python-wheels-Linux.zip; \
        n=$((n+1)); \
        echo "wget attempt $n failed - retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    test -f rocm-python-wheels-Linux.zip; \
    "$VIRTUAL_ENV/bin/python3" -c "import zipfile; zipfile.ZipFile('rocm-python-wheels-Linux.zip').extractall('ct2whl')"; \
    py_ver="$("$VIRTUAL_ENV/bin/python3" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"; \
    whl="$(find ct2whl -name "ctranslate2-${CT2_VERSION}-${py_ver}-${py_ver}-*.whl" | head -1)"; \
    test -n "$whl"; \
    uv pip install --python "$VIRTUAL_ENV/bin/python3" --no-cache "$whl"; \
    uv pip install --python "$VIRTUAL_ENV/bin/python3" --no-cache --no-deps "faster-whisper==1.2.1"; \
    rm -rf /tmp/rocm-python-wheels-Linux.zip /tmp/ct2whl; \
    "$VIRTUAL_ENV/bin/python3" -c "import ctranslate2; print('ctranslate2', ctranslate2.__version__)"; \
    "$VIRTUAL_ENV/bin/python3" -c "from faster_whisper import WhisperModel; print('faster_whisper import OK')"; \
    "$VIRTUAL_ENV/bin/python3" -c "import onnxruntime as ort; p=ort.get_available_providers(); assert 'MIGraphXExecutionProvider' in p, 'MIGraphX EP lost after faster-whisper install'; print('MIGraphX EP intact:', p)"; \
    find "$VIRTUAL_ENV/lib" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Environment (baked into the image, not compose)
# ---------------------------------------------------------------------------
ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    IVF_DISK_CACHE_DIR=/app/ivf_cache \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    HF_XET_DISABLE=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONPATH=/app

# HSA_ENABLE_INTERRUPT=1: interrupt-driven GPU completion instead of CPU
# busy-polling (the default spin-wait pegs a CPU core). MIOPEN_*: kernel
# compile cache dir, mounted as a volume by compose to persist across restarts.
# LYRICS_WHISPER_FASTER_*: read by the plugin's faster-whisper backend.
# CT2_CUDA_ALLOCATOR=cub_caching: default CT2 CUDA/HIP allocator triggers a
# GPU memory access fault (page not present) mid-transcribe on gfx1201
# (RX 9070 XT / RDNA4), crashing the host. Reproduced locally with real audio
# and confirmed fixed by this allocator. Root cause is an upstream ROCm LLVM
# codegen bug for gfx1201, not CTranslate2 itself; see
# https://github.com/OpenNMT/CTranslate2/issues/2021
ENV HSA_ENABLE_INTERRUPT=1 \
    CT2_CUDA_ALLOCATOR=cub_caching \
    MIOPEN_USER_DB_PATH=/app/.cache/miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/app/.cache/miopen \
    LYRICS_WHISPER_FASTER_DEVICE=cuda \
    LYRICS_WHISPER_FASTER_COMPUTE_TYPE=float16 \
    LYRICS_WHISPER_FASTER_MODEL_DIR=/app/model/faster-whisper-small
RUN mkdir -p /app/.cache/miopen /app/.cache/migraphx && chmod 777 /app/.cache/miopen /app/.cache/migraphx

# Pre-fetch the faster-whisper model so the image is self-contained. Verified
# loadable on CPU here; the build host has no GPU so the GPU path runs at
# runtime instead.
RUN set -eux; \
    "$VIRTUAL_ENV/bin/python3" -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small', local_dir='/app/model/faster-whisper-small')"; \
    test -f /app/model/faster-whisper-small/model.bin; \
    "$VIRTUAL_ENV/bin/python3" -c "from faster_whisper import WhisperModel; WhisperModel('/app/model/faster-whisper-small', device='cpu', compute_type='int8'); print('faster-whisper model loads standalone OK (CPU)')"

# Application code, analysis models and HF cache from the published core
# image (last, for cache hits when only BASE_IMAGE/ROCm layers change). Docker
# COPY overlays onto the destination without deleting the faster-whisper model
# prefetched above (core image doesn't have that path).
COPY --from=core /app /app
COPY docker/docker-entrypoint-rocm-wrapper.sh /app/docker-entrypoint-rocm-wrapper.sh
COPY docker/supervisord-rocm.conf /etc/supervisor/conf.d/supervisord.conf
RUN chmod +x /app/docker-entrypoint.sh /app/docker-entrypoint-rocm-wrapper.sh && \
    test -f /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000
WORKDIR /workspace
ENTRYPOINT ["/app/docker-entrypoint-rocm-wrapper.sh"]
CMD []
