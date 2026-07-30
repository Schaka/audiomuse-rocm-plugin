# Our own approach: NeMo + Parakeet-TDT on top of this plugin's real worker
# base image, instead of parakeet_rocm's rocm7.0-pinned torch wheels (which
# are DOA on gfx803 - ROCm 7 dropped Polaris outright). The base already
# carries a working torch/torchaudio built for this exact arch, same one
# CTranslate2 gets compiled against in docker/Dockerfile - so this only adds
# NeMo itself and answers whether it survives on top of what we already ship.
ARG ROCM_BASE_IMAGE=ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803
FROM ${ROCM_BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg sox libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# --no-cache-dir only, no --no-deps: if NeMo's resolver wants to replace the
# base image's ROCm torch with a stock/CPU wheel, that is itself the finding.
RUN /opt/venv/bin/python3 -m pip install --no-cache-dir "nemo_toolkit[asr]<2.5.0,>=2.4.0"

COPY nemo_parakeet_transcribe.py /app-probe/transcribe.py
ENTRYPOINT ["/opt/venv/bin/python3", "/app-probe/transcribe.py"]
