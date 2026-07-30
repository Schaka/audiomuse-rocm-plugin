# Source build: parakeet.cpp only publishes CPU/CUDA images; HIP is a CMake
# option (PARAKEET_GGML_HIP) with no prebuilt image, same gap as whisper.cpp.
# Same base-image reasoning as whisper-cpp-rocm.Dockerfile.
ARG ROCM_BASE_IMAGE=ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803
FROM ${ROCM_BASE_IMAGE}

ARG AMDGPU_TARGETS=gfx803

# hipblas-dev/rocblas-dev packages don't exist under ROCm 7.14's package
# naming (amdrocm-*7.14) on gfx1201's base image - that base already carries
# its own hipBLAS/rocBLAS dev headers (amdrocm-blas-dev7.14) automatically,
# so nothing extra is needed here. gfx803's ROCm 6.4 base uses the older
# hipblas-dev/rocblas-dev names instead, which is why this conditional exists.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ffmpeg ca-certificates \
    && (apt-get install -y --no-install-recommends hipblas-dev rocblas-dev || true) \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --recursive --depth 1 https://github.com/mudler/parakeet.cpp .

RUN cmake -B build -DPARAKEET_GGML_HIP=1 -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j

ENTRYPOINT ["/build/build/examples/cli/parakeet-cli"]
