#!/usr/bin/env python3
"""Minimal repro: does MIGraphXExecutionProvider -> ROCMExecutionProvider, in
that order, in the same process, fault the GPU on gfx803?

Single model (musicnn_embedding.onnx), single tiny synthetic input, one
inference on MIGraphX, then one inference on ROCMExecutionProvider. Nothing
else - no benchmarking, no timing, no other samples. If this alone reproduces
the "Memory access fault... Page not present" crash seen during the EP
benchmark, it confirms the fault is the EP hand-off itself, not something
benchmark-specific.

Run inside the gfx803 image:
    docker cp local-test/repro_migraphx_then_rocm_crash.py <container>:/tmp/repro.py
    docker exec -it <container> python3 /tmp/repro.py
or the standalone docker run form (see chat for full flags).
"""

import sys

import numpy as np
import onnxruntime as ort

MODEL_PATH = "/app/model/musicnn_embedding.onnx"
INPUT_NAME = "model/Placeholder:0"
OUTPUT_NAME = "model/dense/BiasAdd:0"

# musicnn's expected patch shape: (batch, 187 frames, 96 mel bins).
DUMMY_INPUT = np.random.default_rng(0).standard_normal((1, 187, 96)).astype(np.float32)


def make_session(provider: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    provider_options = {"device_id": "0"}
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=[provider, "CPUExecutionProvider"],
        provider_options=[provider_options, {}],
        sess_options=opts,
    )
    actual = session.get_providers()[0]
    if actual != provider:
        print(f"WARNING: session silently fell back to {actual}, not {provider}", file=sys.stderr)
    return session


def main():
    print("[1/4] Creating MIGraphXExecutionProvider session...", flush=True)
    mgx_session = make_session("MIGraphXExecutionProvider")

    print("[2/4] Running one inference on MIGraphX...", flush=True)
    mgx_session.run([OUTPUT_NAME], {INPUT_NAME: DUMMY_INPUT})
    print("      MIGraphX inference OK.", flush=True)

    print("[3/4] Creating ROCMExecutionProvider session (same process)...", flush=True)
    rocm_session = make_session("ROCMExecutionProvider")

    print("[4/4] Running one inference on ROCMExecutionProvider...", flush=True)
    rocm_session.run([OUTPUT_NAME], {INPUT_NAME: DUMMY_INPUT})
    print("      ROCM inference OK.", flush=True)

    print("\nNo crash: MIGraphX -> ROCM in one process is safe on this build.")


if __name__ == "__main__":
    main()
