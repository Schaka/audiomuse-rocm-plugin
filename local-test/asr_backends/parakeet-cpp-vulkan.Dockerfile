# parakeet.cpp Vulkan build, on this plugin's own worker base image - same
# reasoning as parakeet-cpp-rocm.Dockerfile and whisper-cpp-rocm.Dockerfile:
# answers "does this bolt onto our actual image", not just "does Vulkan work
# on some generic Ubuntu". The ROCm base already ships RADV/mesa userspace
# (musicnn/CLAP's MIGraphX EP needs it too), so this only adds the Vulkan
# SDK build deps on top, nothing GPU-driver-related.
ARG ROCM_BASE_IMAGE=ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803
FROM ${ROCM_BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ffmpeg ca-certificates \
    libvulkan-dev vulkan-tools glslang-tools mesa-vulkan-drivers glslc \
    && rm -rf /var/lib/apt/lists/*

# SPIRV-Headers isn't packaged on Ubuntu 22.04/24.04; it's header-only so a
# plain cmake install is enough to satisfy ggml-vulkan's find_package().
RUN git clone --depth 1 --branch vulkan-sdk-1.3.283.0 https://github.com/KhronosGroup/SPIRV-Headers /tmp/spirv-headers \
    && cmake -S /tmp/spirv-headers -B /tmp/spirv-headers/build -DCMAKE_INSTALL_PREFIX=/usr \
    && cmake --install /tmp/spirv-headers/build \
    && rm -rf /tmp/spirv-headers

WORKDIR /build
RUN git clone --recursive --depth 1 https://github.com/mudler/parakeet.cpp .

RUN cmake -B build -DPARAKEET_GGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j

ENTRYPOINT ["/build/build/examples/cli/parakeet-cli"]
