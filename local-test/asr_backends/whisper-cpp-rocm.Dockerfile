# Source build: whisper.cpp ships no prebuilt HIP/ROCm image, only CUDA and
# Vulkan. Defaults to this plugin's own worker base
# (ghcr.io/schaka/rocm-migraphx-ort-torch-builder), the same image
# docker/Dockerfile builds CTranslate2 against - so a pass here answers
# "does this bolt onto our actual image", not just "does HIP work somewhere".
ARG ROCM_BASE_IMAGE=ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803
FROM ${ROCM_BASE_IMAGE}

ARG AMDGPU_TARGETS=gfx803

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ffmpeg ca-certificates \
    hipblas-dev rocblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 https://github.com/ggml-org/whisper.cpp .

RUN cmake -B build -DGGML_HIP=1 -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j --config Release

RUN bash ./models/download-ggml-model.sh base

ENTRYPOINT ["/build/build/bin/whisper-cli"]
