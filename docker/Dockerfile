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
ARG BASE_IMAGE=ghcr.io/schaka/rocm-migraphx-ort-torch-builder:rocm7.14-gfx1201
ARG CORE_IMAGE=ghcr.io/neptunehub/audiomuse-ai:latest

# Which CTranslate2 to use - see the ct2-* stages below. "prebuilt" takes
# upstream's ROCm 7 release wheel, which only carries device code for gfx1030
# and newer; "source" compiles for this image's arch, from CT2_REPO/CT2_REF.
# Three configurations in total, selected per arch in image-build.yml.
ARG CT2_VARIANT=prebuilt
ARG ROCM_ARCH=gfx1201
ARG CT2_VERSION=4.8.1

# whisper.cpp/parakeet.cpp: pinned to a tagged release, not HEAD - both
# projects churn multiple times a day (ggml is shared upstream and gets synced
# from llama.cpp constantly), so an unpinned `git clone --depth 1` is not
# reproducible: the exact same Dockerfile can pass or fail a build hours apart
# with zero changes on our side. Bump these deliberately, not by drift.
ARG WHISPER_CPP_REF=v1.9.1
ARG PARAKEET_CPP_REF=v0.4.0

# ============================================================================
# Stage 1: core - the published AudioMuse-AI image (app code + models + HF cache)
# ============================================================================
FROM ${CORE_IMAGE} AS core

# ============================================================================
# Stage 2a: ct2-prebuilt - CTranslate2's own ROCm release wheel
# ============================================================================
# Upstream ships one prebuilt ROCm wheel per release, linked against ROCm 7's
# hipBLAS (libhipblas.so.3). Correct for every arch on the ROCm 7 base.
FROM ${BASE_IMAGE} AS ct2-prebuilt
ARG CT2_VERSION
SHELL ["/bin/bash", "-c"]
# Both ct2 stages emit the same /ct2-out layout so the runner can consume
# either without knowing which was selected: wheel/ always holds the wheel,
# lib/ holds any shared libs the wheel does not already bundle (empty here -
# upstream's release wheels are auditwheel-repaired and self-contained).
# Fetched with the venv's python, not curl/wget: the ROCm bases ship neither,
# and an apt layer for a single download would drag the toolchain into an
# otherwise empty stage.
RUN set -eux; \
    mkdir -p /ct2-out/wheel /ct2-out/lib; \
    cd /tmp; \
    url="https://github.com/OpenNMT/CTranslate2/releases/download/v${CT2_VERSION}/rocm-python-wheels-Linux.zip"; \
    n=0; \
    until [ "$n" -ge 5 ]; do \
        if /opt/venv/bin/python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], 'rocm-python-wheels-Linux.zip')" "$url"; then \
            break; \
        fi; \
        rm -f rocm-python-wheels-Linux.zip; \
        n=$((n+1)); \
        echo "download attempt $n failed - retrying in $((n*n))s"; \
        sleep $((n*n)); \
    done; \
    test -f rocm-python-wheels-Linux.zip; \
    /opt/venv/bin/python3 -c "import zipfile; zipfile.ZipFile('rocm-python-wheels-Linux.zip').extractall('ct2whl')"; \
    py_ver="$(/opt/venv/bin/python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"; \
    whl="$(find ct2whl -name "ctranslate2-${CT2_VERSION}-${py_ver}-${py_ver}-*.whl" | head -1)"; \
    test -n "$whl"; \
    cp "$whl" /ct2-out/wheel/; \
    rm -rf /tmp/rocm-python-wheels-Linux.zip /tmp/ct2whl

# ============================================================================
# Stage 2b: ct2-source - CTranslate2 compiled from source for one arch
# ============================================================================
# Used for the two cases the prebuilt wheel does not cover, parametrized by
# CT2_REPO/CT2_REF rather than split into two near-identical stages:
#
#   * ROCm 6 bases (Polaris/gfx803) -> arlo-phoenix/CTranslate2-rocm, the
#     maintained HIP port. Upstream's wheel wants ROCm 7's libhipblas.so.3 and
#     dies at import there.
#   * Vega/CDNA (gfx900, gfx906, gfx908, gfx90a, gfx942) -> OpenNMT/CTranslate2
#     itself. The base is ROCm 7 so the wheel loads fine, but it is compiled
#     only for gfx1030 and newer (see prepare_build_environment_linux_rocm.sh),
#     so those arches would get no device code and fail at the first GPU
#     transcribe with "no kernel image is available". Building the same
#     upstream source for the actual arch is the whole fix - no fork needed.
#
# WITH_CUDNN=ON on the arlo-phoenix fork (gfx803): despite the CUDA-flavoured
# name, this is what actually enables their from-scratch MIOpen backend for
# Conv1D - hipDNN (the old cuDNN-alike) is unmaintained, so upstream CTranslate2
# has no HIP path for Conv1D at all, and Whisper's encoder opens with two
# Conv1D layers. Leaving this off builds fine and imports fine,
# but every GPU transcribe then fails at the first Conv1D with "Conv1D on GPU.
#
# CMAKE_POLICY_VERSION_MINIMUM covers third_party/cpu_features, which still asks
# for cmake_minimum_required below 3.5 - removed outright in the CMake 4 on the
# ROCm 7 bases (ROCm 6/gfx803 has CMake 3.x, where it is only a warning). The
# variable is simply unused on the older CMake, so one flag serves both.
FROM ${BASE_IMAGE} AS ct2-source
ARG ROCM_ARCH
ARG CT2_REPO=https://github.com/OpenNMT/CTranslate2.git
ARG CT2_REF=v4.8.1
SHELL ["/bin/bash", "-c"]
RUN set -eux; \
    DEBIAN_FRONTEND=noninteractive apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential libopenblas-dev; \
    rm -rf /var/lib/apt/lists/*
# Arch-specific source fixes, one directory per arch under docker/patches/.
# Each patch documents its own why/what in its header; applied right after the
# clone so a patch that stops applying fails the build loudly instead of
# silently shipping unpatched code.
COPY docker/patches/ /ct2-patches/
RUN set -eux; \
    git clone --branch "${CT2_REF}" --depth 1 --recursive "${CT2_REPO}" /ct2-src; \
    cd /ct2-src; \
    if compgen -G "/ct2-patches/${ROCM_ARCH}/*.patch" > /dev/null; then \
        for p in "/ct2-patches/${ROCM_ARCH}"/*.patch; do \
            echo "applying ${p}"; \
            git apply --verbose "${p}"; \
        done; \
    fi; \
    llvm_bin="$(hipconfig -l)"; \
    export HIPCXX="${llvm_bin}/clang"; \
    export HIP_PATH="$(hipconfig -R)"; \
    case "${CT2_REPO}" in \
        *arlo-phoenix*) with_cudnn=ON ;; \
        *) with_cudnn=OFF ;; \
    esac; \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_C_COMPILER="${llvm_bin}/clang" \
        -DCMAKE_CXX_COMPILER="${llvm_bin}/clang++" \
        -DWITH_MKL=OFF \
        -DWITH_OPENBLAS=ON \
        -DOPENMP_RUNTIME=COMP \
        -DWITH_HIP=ON \
        -DWITH_CUDNN="${with_cudnn}" \
        -DCMAKE_HIP_ARCHITECTURES="${ROCM_ARCH}" \
        -DBUILD_CLI=OFF \
        -DBUILD_TESTS=OFF; \
    set -o pipefail; \
    cmake --build build -j"$(nproc)" 2>&1 | tee /tmp/ct2-build.log || { \
        echo "=== first failing translation unit ==="; \
        grep -m1 -A30 '^FAILED:' /tmp/ct2-build.log; \
        exit 1; \
    }; \
    cmake --install build
# setup.py bdist_wheel is not auditwheel-repaired, so the wheel links against
# libctranslate2.so without bundling it - the runner needs the shared lib
# alongside it, unlike the prebuilt path.
#
# libomp comes along for the same reason: OPENMP_RUNTIME=COMP under ROCm's
# clang means -fopenmp=libomp, and that libomp lives in the ROCm LLVM tree,
# which the runner has no reason to carry (it installs libgomp1, GCC's, which
# is a different soname and does not satisfy the link).
RUN set -eux; \
    mkdir -p /ct2-out/wheel /ct2-out/lib; \
    cd /ct2-src/python; \
    /opt/venv/bin/python3 -m pip install --no-cache-dir -r install_requirements.txt; \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-} \
        /opt/venv/bin/python3 setup.py bdist_wheel; \
    cp dist/*.whl /ct2-out/wheel/; \
    cp -a /usr/local/lib/libctranslate2.so* /ct2-out/lib/; \
    omp="$(dirname "$(hipconfig -l)")/lib/libomp.so"; \
    test -f "$omp"; \
    cp -aL "$omp" /ct2-out/lib/

# ============================================================================
# Stage 2c: ct2 - whichever of the two above was selected
# ============================================================================
# BuildKit only builds the stages the target actually depends on, so the
# unselected variant is never executed - the source compile costs nothing on
# the 8 arches the prebuilt wheel already covers.
FROM ct2-${CT2_VARIANT} AS ct2

# ============================================================================
# Stages 3a-3d: whisper.cpp and parakeet.cpp, Vulkan and HIP variants
# ============================================================================
# Both engines confirmed working standalone (see docs/ASR_BACKENDS.md) -
# selectable via the plugin's asr_backend/asr_backend_variant settings, one
# CLI binary per (engine, variant) pair, both variants of an engine sharing
# one model file (copied once into the runner stage below).
#
# parakeet.cpp's HIP build is skipped on Polaris (gfx803/802/805): confirmed
# to produce silent empty transcripts there so there is nothing to
# gain from spending build time and image space on a binary the plugin will
# never select (Gfx803Profile.blocked_asr_backends refuses it). An empty
# placeholder is emitted in its place so the runner's COPY --from below has
# something to copy either way; whisper_cpp_backend.available()/
# parakeet_cpp_backend.available() check the file is non-empty, not just
# present, so the placeholder correctly reads as "unavailable".
FROM ${BASE_IMAGE} AS ggml-vulkan-deps
RUN set -eux; \
    DEBIAN_FRONTEND=noninteractive apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates \
        libvulkan-dev vulkan-tools glslang-tools mesa-vulkan-drivers glslc spirv-headers; \
    rm -rf /var/lib/apt/lists/*

# BUILD_SHARED_LIBS=OFF: statically links ggml/whisper/parakeet into the CLI
# binary itself, so only that one file needs copying into the runner stage -
# no libggml*.so/libwhisper.so siblings, no RPATH-into-a-build-tree-that-
# won't-exist-there problem. System libs (libvulkan.so.1, libstdc++, ...)
# stay dynamic - those are already in the runner image regardless.
SHELL ["/bin/bash", "-c"]

# ggml's Vulkan backend generates ~200+ shader variants via vulkan-shaders-gen,
# which forks a batch of concurrent glslc subprocesses (independent of this
# RUN's own -j). On resource-pressured shared CI runners that occasionally hits
# a transient fork()/pipe failure for one shader - but vulkan-shaders-gen.cpp
# catches that per-shader exception and only logs it, so the tool still exits
# 0 with that variant silently missing. ggml-vulkan.cpp references every
# variant unconditionally (not behind a feature-detect #ifdef), so the gap
# only surfaces later as a cryptic "undefined reference to
# matmul_id_subgroup_..." at the final link. The sed below turns that swallowed
# exception into an immediate abort, so a real transient failure is caught
# where it happens instead of silently corrupting the shader set; the retry
# loop then has something legitimate to act on. It's arch-independent (pure
# host-side shader compilation, not cross-compiled per AMDGPU_TARGETS), which
# is why it hits an unpredictable subset of the arch matrix each run. Same
# fork-failure -> undefined-reference pattern reported upstream, unfixed as of
# whisper.cpp v1.9.1: https://github.com/ggml-org/llama.cpp/issues/20868
FROM ggml-vulkan-deps AS whisper-cpp-vulkan
ARG WHISPER_CPP_REF
RUN set -eux; \
    git clone --depth 1 --branch "${WHISPER_CPP_REF}" https://github.com/ggml-org/whisper.cpp /build; \
    vk_gen="$(find /build -path '*ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp' -print -quit)"; \
    test -n "$vk_gen"; \
    sed -i '/Error executing command for/a\        std::exit(1);' "$vk_gen"
RUN set -eux; \
    n=0; \
    until [ "$n" -ge 3 ]; do \
        cmake -S /build -B /build/build -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON; \
        if cmake --build /build/build -j"$(nproc)" --target whisper-cli; then \
            break; \
        fi; \
        n=$((n+1)); \
        echo "whisper.cpp vulkan build attempt $n failed (likely shader-gen fork flakiness) - retrying"; \
    done; \
    test -x /build/build/bin/whisper-cli

FROM ggml-vulkan-deps AS parakeet-cpp-vulkan
ARG PARAKEET_CPP_REF
RUN set -eux; \
    git clone --recursive --depth 1 --branch "${PARAKEET_CPP_REF}" https://github.com/mudler/parakeet.cpp /build; \
    vk_gen="$(find /build -path '*ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp' -print -quit)"; \
    test -n "$vk_gen"; \
    sed -i '/Error executing command for/a\        std::exit(1);' "$vk_gen"
RUN set -eux; \
    n=0; \
    until [ "$n" -ge 3 ]; do \
        cmake -S /build -B /build/build -DPARAKEET_GGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON; \
        if cmake --build /build/build -j"$(nproc)" --target parakeet-cli; then \
            break; \
        fi; \
        n=$((n+1)); \
        echo "parakeet.cpp vulkan build attempt $n failed (likely shader-gen fork flakiness) - retrying"; \
    done; \
    test -x /build/build/examples/cli/parakeet-cli

FROM ${BASE_IMAGE} AS whisper-cpp-hip
ARG ROCM_ARCH
ARG WHISPER_CPP_REF
SHELL ["/bin/bash", "-c"]
# hipblas-dev/rocblas-dev: needed by name on gfx803's ROCm 6.4 base; gfx900+'s
# ROCm 7.14 base already carries its own BLAS dev headers under a different
# package name (amdrocm-blas-dev7.14 etc.), so this is a no-op there - hence
# the `|| true` rather than a hard failure on the newer bases.
RUN set -eux; \
    DEBIAN_FRONTEND=noninteractive apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates; \
    (DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends hipblas-dev rocblas-dev || true); \
    rm -rf /var/lib/apt/lists/*; \
    git clone --depth 1 --branch "${WHISPER_CPP_REF}" https://github.com/ggml-org/whisper.cpp /build; \
    cmake -S /build -B /build/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DGGML_HIP=1 -DAMDGPU_TARGETS="${ROCM_ARCH}"; \
    cmake --build /build/build -j"$(nproc)" --target whisper-cli

FROM ${BASE_IMAGE} AS parakeet-cpp-hip
ARG ROCM_ARCH
ARG PARAKEET_CPP_REF
SHELL ["/bin/bash", "-c"]
RUN set -eux; \
    case "${ROCM_ARCH}" in \
        gfx803|gfx802|gfx805) \
            echo "parakeet.cpp HIP skipped on ${ROCM_ARCH} - confirmed broken, see docs/ASR_BACKENDS.md"; \
            mkdir -p /build/build/examples/cli; \
            : > /build/build/examples/cli/parakeet-cli; \
            ;; \
        *) \
            DEBIAN_FRONTEND=noninteractive apt-get update; \
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                git cmake build-essential ca-certificates; \
            (DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends hipblas-dev rocblas-dev || true); \
            rm -rf /var/lib/apt/lists/*; \
            git clone --recursive --depth 1 --branch "${PARAKEET_CPP_REF}" https://github.com/mudler/parakeet.cpp /build; \
            cmake -S /build -B /build/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                -DPARAKEET_GGML_HIP=1 -DAMDGPU_TARGETS="${ROCM_ARCH}"; \
            cmake --build /build/build -j"$(nproc)" --target parakeet-cli; \
            ;; \
    esac

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
COPY docker/requirements/rocm.txt /app/requirements/rocm.txt
# common.txt is taken from the core image rather than vendored here so the app
# deps can never drift from the core release this image is built against. It
# has to be installed, not inherited: core installs with `uv pip install
# --system` into system site-packages, and only /app is copied from it, so
# nothing of common.txt reaches this image's venv on its own.
COPY --from=core /app/requirements/common.txt /app/requirements/common.txt

# Install into the base image's venv (/opt/venv), which already has onnxruntime
# built with MIGraphXExecutionProvider. Target the venv interpreter explicitly
# (not system python3.12) so deps land where the app actually runs; onnxruntime
# is deliberately absent from both requirements files so it is never reinstalled
# and the MIGraphX build never gets clobbered by a CPU-only wheel.
ENV VIRTUAL_ENV=/opt/venv
RUN set -eux; \
    uv pip install --python "$VIRTUAL_ENV/bin/python3" --no-cache --index-strategy unsafe-best-match \
        -r /app/requirements/rocm.txt -r /app/requirements/common.txt; \
    "$VIRTUAL_ENV/bin/python3" -c "import onnxruntime as ort; p=ort.get_available_providers(); print('ORT providers:', p); assert 'MIGraphXExecutionProvider' in p, 'MIGraphX EP missing'"; \
    "$VIRTUAL_ENV/bin/python3" -c "import psycopg2, av, huggingface_hub; print('common.txt deps OK')"; \
    find "$VIRTUAL_ENV/lib" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# faster-whisper on the AMD GPU via CTranslate2's ROCm HIP backend. The ROCm
# Accelerator plugin registers this as the ASR backend; the libs must be present
# in the image because the plugin ships no pip requirements. The wheel comes
# from the ct2 stage - upstream's ROCm 7 release wheel, or the ROCm fork built
# from source for a ROCm 6 base. faster-whisper itself is installed --no-deps:
# its onnxruntime dependency would otherwise clobber the base MIGraphX ORT with
# a CPU-only build.
COPY --from=ct2 /ct2-out /tmp/ct2-out
RUN set -eux; \
    if compgen -G '/tmp/ct2-out/lib/*' >/dev/null; then \
        cp -a /tmp/ct2-out/lib/. /usr/local/lib/; \
        ldconfig; \
    fi; \
    whl="$(find /tmp/ct2-out/wheel -name 'ctranslate2-*.whl' | head -1)"; \
    test -n "$whl"; \
    uv pip install --python "$VIRTUAL_ENV/bin/python3" --no-cache "$whl"; \
    uv pip install --python "$VIRTUAL_ENV/bin/python3" --no-cache --no-deps "faster-whisper==1.2.1"; \
    rm -rf /tmp/ct2-out; \
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
# CT2_CUDA_ALLOCATOR is deliberately NOT set here: it's arch-specific (see
# plugin/rocm_accelerator/arch/gfx1201.py), so every other arch gets to keep
# CTranslate2's own default allocator instead of being pinned to one arch's fix.
ENV HSA_ENABLE_INTERRUPT=1 \
    MIOPEN_USER_DB_PATH=/app/.cache/miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/app/.cache/miopen \
    LYRICS_WHISPER_FASTER_DEVICE=cuda \
    LYRICS_WHISPER_FASTER_MODEL_DIR=/app/model/faster-whisper-small
RUN mkdir -p /app/.cache/miopen /app/.cache/migraphx && chmod 777 /app/.cache/miopen /app/.cache/migraphx

# Pre-fetch the faster-whisper model so the image is self-contained. Verified
# loadable on CPU here; the build host has no GPU so the GPU path runs at
# runtime instead.
#
# The compute type is negotiated exactly as whisper_faster.py's CPU fallback
# does: int8 needs a oneDNN/MKL build of CTranslate2, which the OpenBLAS-only
# source builds are not, so float32 has to be accepted as a pass.
RUN set -eux; \
    "$VIRTUAL_ENV/bin/python3" -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small', local_dir='/app/model/faster-whisper-small')"; \
    test -f /app/model/faster-whisper-small/model.bin; \
    "$VIRTUAL_ENV/bin/python3" - <<'PY'
from faster_whisper import WhisperModel

for compute in ("int8", "float32"):
    try:
        WhisperModel("/app/model/faster-whisper-small", device="cpu", compute_type=compute)
    except ValueError as exc:
        print(f"compute_type={compute} unsupported: {exc}")
        continue
    print(f"faster-whisper model loads standalone OK (CPU, {compute})")
    break
else:
    raise SystemExit("faster-whisper model failed to load on CPU with any compute type")
PY

# whisper.cpp and parakeet.cpp: Vulkan and HIP CLI binaries, selectable via
# the plugin's asr_backend/asr_backend_variant settings (see
# plugin/rocm_accelerator/whisper_cpp_backend.py and parakeet_cpp_backend.py).
# Both variants of an engine share one model file - only the binary differs.
#
# Both engines build statically (BUILD_SHARED_LIBS=OFF above), so each is one
# self-contained binary - no siblings to bring along, no RPATH concerns.
COPY --from=whisper-cpp-vulkan /build/build/bin/whisper-cli /opt/asr-backends/whisper-cpp/whisper-cli-vulkan
COPY --from=whisper-cpp-hip /build/build/bin/whisper-cli /opt/asr-backends/whisper-cpp/whisper-cli-hip
COPY --from=parakeet-cpp-vulkan /build/build/examples/cli/parakeet-cli /opt/asr-backends/parakeet-cpp/parakeet-cli-vulkan
COPY --from=parakeet-cpp-hip /build/build/examples/cli/parakeet-cli /opt/asr-backends/parakeet-cpp/parakeet-cli-hip

RUN set -eux; \
    mkdir -p /app/model/whisper-cpp /app/model/parakeet-cpp; \
    curl -fL -o /app/model/whisper-cpp/ggml-small.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin; \
    curl -fL -o /app/model/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf \
        https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt-0.6b-v3-q8_0.gguf; \
    test -s /app/model/whisper-cpp/ggml-small.bin; \
    test -s /app/model/parakeet-cpp/tdt-0.6b-v3-q8_0.gguf

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
