"""Loads Parakeet-TDT 0.6B v3 via NeMo and transcribes one file. Run inside
nemo-parakeet-rocm.Dockerfile; reports which device the model actually ended
up on, since a silent CPU fallback would look like a pass otherwise.
"""

import sys

import torch

# The plugin's source-built torch has no torch.distributed backend (single-GPU
# builds skip USE_DISTRIBUTED), but NeMo's conformer encoder calls
# torch.distributed.is_initialized() unconditionally even for single-device
# inference. Stubbing it false is correct here - there is no process group.
if not hasattr(torch.distributed, "is_initialized"):
    torch.distributed.is_initialized = lambda: False

from nemo.collections.asr.models import ASRModel

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "/audios/jfk.wav"

    print(f"torch {torch.__version__}, cuda/hip available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    model = ASRModel.from_pretrained(MODEL_NAME)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.float()  # force fp32: gfx803 has no packed fp16 hardware (see
    # docs/ARCH_NOTES.md); autocast defaulting to fp16 here would silently run
    # broken math rather than error, same class of bug as CTranslate2's fp16 path.
    model.eval()
    print(f"model parameters on: {next(model.parameters()).device}, dtype: {next(model.parameters()).dtype}")

    # fp32 alone did not fix the garbage-loop output (still 'and and and...').
    # Next suspect: PyTorch's fused/flash SDPA kernels, which on ROCm run
    # through a from-scratch attention implementation with no official gfx803
    # coverage - same shape of problem as CTranslate2's fp16 GEMM path and
    # MIOpen's fused-conv path (see docs/ARCH_NOTES.md). Forcing the naive
    # "math" SDPA backend rules out fused-attention-kernel breakage.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        result = model.transcribe([audio_path])
    print(f"text={result[0].text!r}" if hasattr(result[0], "text") else f"result={result!r}")


if __name__ == "__main__":
    main()
