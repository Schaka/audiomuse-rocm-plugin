# beecave-homelab/parakeet_rocm's actual CLI/webui package, installed on our
# own worker base image instead of their rocm7.0-pinned torch wheels - no
# wheel index, no version pin, just their package on top of the torch/NeMo
# we already have working here. Answers "does their CLI wrapper behave any
# differently from calling NeMo directly" (nemo-parakeet-rocm.Dockerfile),
# since the model/torch/NeMo underneath is otherwise identical either way.
ARG ROCM_BASE_IMAGE=ghcr.io/schaka/rocm-migraphx-ort-torch-builder:latest-gfx803
FROM ${ROCM_BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg sox libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/venv/bin/python3 -m pip install --no-cache-dir "nemo_toolkit[asr]<2.5.0,>=2.4.0"

# --no-deps: their pyproject.toml pulls the rocm7.0 torch/torchaudio wheels
# via a custom index, which would clobber the working torch this image
# already ships. Everything else their CLI needs (typer, rich) is small and
# not already present, so it needs its own install pass.
# --ignore-requires-python: their pyproject pins <3.11, this image's venv is
# 3.12 - that pin is almost certainly about their torch wheel's cp310/cp311-only
# builds, not their own (typer/rich) code, which has no 3.12 incompatibility.
RUN git clone --depth 1 https://github.com/beecave-homelab/parakeet_rocm /tmp/parakeet_rocm \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir --no-deps --ignore-requires-python -e /tmp/parakeet_rocm \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir typer rich

# NeMo call needs the same torch.distributed stub as nemo_parakeet_transcribe.py -
# this image's torch has no distributed backend (single-GPU source build).
# A .pth file's "import " lines are exec'd by site.py on every interpreter
# start for this site-packages dir - more reliable here than sitecustomize.py,
# which silently never ran (some path-ordering quirk in this base image).
RUN echo "import torch; hasattr(torch.distributed, 'is_initialized') or setattr(torch.distributed, 'is_initialized', lambda: False)" \
    > /opt/venv/lib/python3.12/site-packages/zz_torch_distributed_patch.pth

ENTRYPOINT ["/opt/venv/bin/parakeet-rocm"]
