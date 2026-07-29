#!/usr/bin/env python3
"""Does the REAL production CLAP provider chain crash, not just the benchmark?

repro_migraphx_then_rocm_crash.py proved MIGraphX -> ROCM in one process
faults the GPU ("Module not initialized") after a FULL SUCCESSFUL MIGraphX
run. But the plugin's actual CLAP fallback (__init__.py) is structurally
different: it hands providers=[MIGraphX, ROCM, CPU] to ONE
ort.InferenceSession call, and MIGraphX fails at graph-PARSE time
(PARSE_Resize: keep_aspect_ratio_policy is not supported!) - before any real
kernel/module ever loads - then ORT falls through to ROCM within that same
session-creation call. This script reproduces that exact real path: loads
the actual CLAP audio ONNX model (same static-shape pinning core does),
builds one session with the same three-provider chain and options the
plugin registers, and runs it once. If this survives, the plugin's CLAP
fallback is safe as shipped; if it faults the same way, needs_static_shapes
alone isn't enough - the fallback needs a different fix (e.g. don't offer
MIGraphX for clap at all on gfx803, go straight ROCM -> CPU).

Run inside the gfx803 image:
    docker cp local-test/repro_clap_provider_chain.py <container>:/tmp/repro2.py
    docker exec -it <container> python3 /tmp/repro2.py

Uses the same migraphx_save_compiled_model/migraphx_load_compiled_model
options __init__.py's gfx803 fallback actually registered for clap (before
that registration was removed) - a first, options-bare version of this script
crashed with bare {"device_id": "0"} only, but that doesn't match production's
real options, so this isn't a clean confirmation by itself. Defaults to the
real /app/.cache/migraphx path (same as production) so mounting the same
migraphx-cache volume the worker uses reproduces an already-warm cache
exactly like whatever state the worker was in during a prior successful run -
pass --cache-dir to point elsewhere instead.
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

CLAP_AUDIO_MODEL_PATH = "/app/model/model_epoch_36.onnx"
SEGMENT_LENGTH_SAMPLES = 480000
CLAP_AUDIO_HOP_LENGTH = 480
MIGRAPHX_CACHE_ROOT = "/app/.cache/migraphx"


def prepared_model_bytes(model_path: str) -> bytes:
    frames = 1 + SEGMENT_LENGTH_SAMPLES // CLAP_AUDIO_HOP_LENGTH
    model = onnx.load(model_path, load_external_data=True)
    symbolic = {
        d.dim_param
        for inp in model.graph.input
        for d in inp.type.tensor_type.shape.dim
        if d.HasField("dim_param")
    }
    print(f"[1/4] Symbolic dims found: {symbolic or '(none)'}", flush=True)
    for name in symbolic:
        make_dim_param_fixed(model.graph, name, frames)
    return model.SerializeToString()


def dummy_inputs(session: ort.InferenceSession) -> dict:
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        np_type = np.float32 if "float" in inp.type else np.int64
        feed[inp.name] = np.zeros(shape, dtype=np_type)
    return feed


def compiled_model_options(cache_dir: str) -> dict:
    # Mirrors __init__.py's _compiled_model_options(): fp16 is always False on
    # gfx803 (no packed fp16), model_label is "clap" - same file the plugin
    # would have used before MIGraphX was removed from clap's provider list.
    sub = os.path.join(cache_dir, "fp32")
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, "clap.mxr")
    if os.path.exists(path):
        print(f"      Found existing {path} - using migraphx_load_compiled_model", flush=True)
        return {"migraphx_load_compiled_model": "True", "migraphx_load_model_path": path}
    print(f"      No {path} yet - using migraphx_save_compiled_model (this run will create it)", flush=True)
    return {"migraphx_save_compiled_model": "True", "migraphx_save_model_path": path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=MIGRAPHX_CACHE_ROOT)
    args = parser.parse_args()

    print(f"[0/4] Loading + pinning {CLAP_AUDIO_MODEL_PATH}...", flush=True)
    model_bytes = prepared_model_bytes(CLAP_AUDIO_MODEL_PATH)

    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False

    # Same chain + options shape as __init__.py's gfx803 fallback registered
    # for the 'clap' label before MIGraphX was removed from it: MIGraphX (with
    # the per-model save/load-compiled cache options), then ROCM, then CPU -
    # in ONE session.
    migraphx_opts = {"device_id": "0"}
    migraphx_opts.update(compiled_model_options(args.cache_dir))
    rocm_opts = {"device_id": "0"}

    print("[2/4] Creating ONE session with providers=[MIGraphX, ROCM, CPU]...", flush=True)
    try:
        session = ort.InferenceSession(
            model_bytes,
            providers=["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"],
            provider_options=[migraphx_opts, rocm_opts, {}],
            sess_options=sess_options,
        )
    except Exception as exc:
        print(f"      Session creation raised (not a crash, a catchable exception): {exc}")
        print("      This is fine - core's create_onnx_session() falls back to CPU on this.")
        return

    actual = session.get_providers()
    print(f"[3/4] Session created. Provider ORT actually picked: {actual[0]}", flush=True)

    print("[4/4] Running one inference...", flush=True)
    feed = dummy_inputs(session)
    outputs = [o.name for o in session.get_outputs()]
    session.run(outputs, feed)
    print("      Inference OK.")

    print("\nNo crash: the real CLAP provider chain is safe on this build.")


if __name__ == "__main__":
    main()
