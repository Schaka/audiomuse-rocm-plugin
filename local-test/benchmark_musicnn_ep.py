#!/usr/bin/env python3
"""Isolated MIGraphX vs ROCMExecutionProvider benchmark for the musicnn models.

Standalone - no AudioMuse-AI DB/Redis/Flask/plugin machinery involved. Loads
the same two ONNX models the real pipeline uses (musicnn_embedding.onnx,
musicnn_prediction.onnx, per tasks/analysis/song.py) directly with plain
onnxruntime, runs each candidate execution provider back to back on the same
inputs, and reports wall-clock inference time.

Run this inside the gfx803 worker image/container (needs onnxruntime, numpy,
librosa - all already present there):

    docker cp local-test/benchmark_musicnn_ep.py <worker-container>:/tmp/bench.py
    docker exec -it <worker-container> python3 /tmp/bench.py

By default it downloads two short, verified-license real music clips from
Wikimedia Commons (CC0/public domain, confirmed resolving and licensed as of
writing - see DOWNLOAD_SAMPLES below) and caches them locally, truncated to
15s each to keep runs quick. If the container has no network egress, it falls
back to synthesizing two short samples instead (sine sweep, pink noise) - no
download, no network, fully reproducible either way. Pass --no-download to
force synthetic, or --audio-dir to point at your own real files instead
(overrides both).

Preprocessing (mel-spectrogram patching) is NOT timed - only the two
session.run() calls are, since that's the only thing that differs between
execution providers. MIGraphX's first run per (model, shape) pays a one-time
JIT-compile cost (or a compiled-model save on gfx803, see the plugin's
_compiled_model_options) - warmup runs are discarded and reported separately
so steady-state numbers aren't skewed by that.

Each provider is benchmarked in its own subprocess (re-execs this same script
with a single --providers entry). This isn't just tidiness: a real hardware
fault was observed on gfx803 when MIGraphXExecutionProvider and
ROCMExecutionProvider ran back to back in one process ("Memory access
fault... Page not present"), fatal to the whole process with no Python
exception to catch. Process isolation means one provider's driver-level
crash can't take down the other's measurement or leave the GPU in a state
that corrupts it - see repro_migraphx_then_rocm_crash.py for a minimal
repro of that exact fault.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

_WORKER_FLAG = "--_worker-outfile"

EMBEDDING_INPUT = "model/Placeholder:0"
EMBEDDING_OUTPUT = "model/dense/BiasAdd:0"
PREDICTION_INPUT = "serving_default_model_Placeholder:0"
PREDICTION_OUTPUT = "PartitionedCall:0"

SAMPLE_RATE = 16000
N_MELS, HOP, N_FFT, FRAME = 96, 256, 512, 187


def prepare_spectrogram_patches(audio: np.ndarray, sr: int) -> np.ndarray | None:
    # Verbatim copy of tasks/analysis/song.py:prepare_spectrogram_patches so the
    # benchmark feeds the models exactly what the real pipeline would.
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        window="hann",
        center=False,
        power=2.0,
        norm="slaney",
        htk=False,
    )
    log_mel = np.log10(1 + 10000 * np.maximum(mel, 0.0))
    patches = [
        log_mel[:, i : i + FRAME] for i in range(0, log_mel.shape[1] - FRAME + 1, FRAME)
    ]
    if not patches:
        return None
    return np.array(patches).transpose(0, 2, 1).astype(np.float32)


SAMPLE_DURATION_S = 12  # kept short: at ~32s/inference observed on gfx803,
                        # even 2 samples x 1 warmup + 3 timed runs adds up fast


def synthesize_samples() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)

    def sine_sweep(duration):
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        f0, f1 = 80.0, 4000.0
        k = (f1 / f0) ** (1.0 / duration)
        phase = 2 * np.pi * f0 * (k**t - 1) / np.log(k)
        return 0.5 * np.sin(phase).astype(np.float32)

    def pink_noise(duration):
        n = int(SAMPLE_RATE * duration)
        white = rng.standard_normal(n)
        # Simple 1/f approximation via cumulative-sum + normalize.
        pink = np.cumsum(white)
        pink = pink / np.max(np.abs(pink))
        return (0.2 * pink).astype(np.float32)

    return {
        f"sine_sweep_{SAMPLE_DURATION_S}s": sine_sweep(SAMPLE_DURATION_S),
        f"pink_noise_{SAMPLE_DURATION_S}s": pink_noise(SAMPLE_DURATION_S),
    }


# Verified (HTTP 200 + license checked) 2026-07-29. Both short, real music,
# clearly licensed for this kind of use - not guessed/fabricated URLs.
DOWNLOAD_SAMPLES = {
    # CC0 1.0, 14s, "8 bars of a drift phonk instrumental at 140 BPM" (Zanahary, 2023)
    "phonk_sample": "https://upload.wikimedia.org/wikipedia/commons/2/2c/Phonk_sample.ogg",
    # Public domain, bansuri (Indian bamboo flute) recording, truncated to 15s on load
    "bansuri_sample": "https://upload.wikimedia.org/wikipedia/commons/1/1f/Sample2.ogg",
}


def download_samples(cache_dir: str, max_duration: float = 15.0) -> dict[str, np.ndarray] | None:
    import urllib.error
    import urllib.request

    import librosa

    os.makedirs(cache_dir, exist_ok=True)
    samples = {}
    for name, url in DOWNLOAD_SAMPLES.items():
        local_path = os.path.join(cache_dir, name + ".ogg")
        if not os.path.exists(local_path):
            print(f"Downloading {name} from {url} ...", file=sys.stderr)
            try:
                with urllib.request.urlopen(url, timeout=15) as resp, open(local_path, "wb") as f:
                    f.write(resp.read())
            except (urllib.error.URLError, OSError) as exc:
                print(f"Download failed ({exc}) - falling back to synthetic samples.", file=sys.stderr)
                return None
        try:
            audio, _ = librosa.load(local_path, sr=SAMPLE_RATE, mono=True, duration=max_duration)
        except Exception as exc:
            print(f"Failed to decode {local_path}: {exc} - falling back to synthetic samples.", file=sys.stderr)
            return None
        samples[name] = audio.astype(np.float32)
    return samples


def load_real_samples(audio_dir: str) -> dict[str, np.ndarray]:
    import librosa

    samples = {}
    for name in sorted(os.listdir(audio_dir)):
        path = os.path.join(audio_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception as exc:
            print(f"skip {name}: {exc}", file=sys.stderr)
            continue
        samples[name] = audio.astype(np.float32)
    return samples


def build_session(model_path: str, provider: str, options: dict, label: str):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False
    try:
        return ort.InferenceSession(
            model_path,
            providers=[provider, "CPUExecutionProvider"],
            provider_options=[options, {}],
            sess_options=sess_options,
        )
    except Exception as exc:
        print(f"{label}: failed to create session with {provider}: {exc}", file=sys.stderr)
        raise


def run_musicnn(embedding_sess, prediction_sess, patches: np.ndarray, batch_size: int):
    embedding_chunks = []
    for start in range(0, len(patches), batch_size):
        chunk = embedding_sess.run(
            [EMBEDDING_OUTPUT], {EMBEDDING_INPUT: patches[start : start + batch_size]}
        )[0]
        embedding_chunks.append(chunk)
    embeddings_per_patch = np.concatenate(embedding_chunks, axis=0)
    prediction_sess.run([PREDICTION_OUTPUT], {PREDICTION_INPUT: embeddings_per_patch})


def provider_options_for(name: str, cache_dir: str, fp16: bool) -> dict:
    if name == "MIGraphXExecutionProvider":
        opts = {"device_id": "0"}
        if fp16:
            opts["migraphx_fp16_enable"] = "True"
        # Mirror the plugin's gfx803 fallback: migraphx_model_cache_dir doesn't
        # exist on this ORT build, so use explicit save/load-compiled-model
        # file paths, one per (model, precision) - kept in a benchmark-only
        # cache dir so this never touches the real production cache.
        return opts
    if name == "ROCMExecutionProvider":
        return {"device_id": "0"}
    raise ValueError(name)


def compiled_model_path(cache_dir: str, model_label: str, fp16: bool) -> str:
    sub = os.path.join(cache_dir, "fp16" if fp16 else "fp32")
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"{model_label}.mxr")


def timed_runs(embedding_sess, prediction_sess, patches, batch_size, runs, warmup):
    times = []
    for i in range(warmup + runs):
        start = time.perf_counter()
        run_musicnn(embedding_sess, prediction_sess, patches, batch_size)
        elapsed = time.perf_counter() - start
        if i < warmup:
            if i == 0:
                cold_time = elapsed
        else:
            times.append(elapsed)
    return times, cold_time


def fmt_ms(values):
    ms = [v * 1000 for v in values]
    return (
        f"mean={statistics.mean(ms):.1f}ms "
        f"median={statistics.median(ms):.1f}ms "
        f"min={min(ms):.1f}ms max={max(ms):.1f}ms "
        f"stdev={statistics.pstdev(ms):.1f}ms"
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", help="Directory of real audio files (overrides download/synthetic samples)")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading sample clips, use synthetic samples instead")
    parser.add_argument("--sample-cache-dir", default="/tmp/musicnn_bench_audio_cache", help="Where downloaded sample clips are cached")
    parser.add_argument("--embedding-model", default="/app/model/musicnn_embedding.onnx")
    parser.add_argument("--prediction-model", default="/app/model/musicnn_prediction.onnx")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per sample per provider")
    parser.add_argument("--warmup", type=int, default=1, help="Discarded warmup runs (lets MIGraphX compile/cache)")
    parser.add_argument("--batch-size", type=int, default=8, help="Matches MUSICNN_BATCH_SIZE default")
    parser.add_argument("--fp16", action="store_true", help="Enable migraphx_fp16_enable (off by default - gfx803 has no packed fp16)")
    parser.add_argument("--cache-dir", default="/tmp/musicnn_bench_migraphx_cache")
    parser.add_argument(
        "--providers",
        default="MIGraphXExecutionProvider,ROCMExecutionProvider",
        help="Comma-separated providers to test",
    )
    parser.add_argument(_WORKER_FLAG, help=argparse.SUPPRESS)
    return parser


def run_one_provider(provider: str, args) -> dict:
    """Benchmark a single provider against every sample. Returns a JSON-able dict."""
    if args.audio_dir:
        samples = load_real_samples(args.audio_dir)
    elif args.no_download:
        samples = synthesize_samples()
    else:
        samples = download_samples(args.sample_cache_dir) or synthesize_samples()
    out = {"times": {}, "cold_ms": {}}

    for sample_name, audio in samples.items():
        patches = prepare_spectrogram_patches(audio, SAMPLE_RATE)
        if patches is None:
            print(f"{sample_name}: too short for a single patch, skipping")
            continue

        options = provider_options_for(provider, args.cache_dir, args.fp16)
        if provider == "MIGraphXExecutionProvider":
            emb_path = compiled_model_path(args.cache_dir, "embedding", args.fp16)
            pred_path = compiled_model_path(args.cache_dir, "prediction", args.fp16)
            emb_opts = dict(options)
            pred_opts = dict(options)
            for opts, path in ((emb_opts, emb_path), (pred_opts, pred_path)):
                if os.path.exists(path):
                    opts["migraphx_load_compiled_model"] = "True"
                    opts["migraphx_load_model_path"] = path
                else:
                    opts["migraphx_save_compiled_model"] = "True"
                    opts["migraphx_save_model_path"] = path
        else:
            emb_opts = pred_opts = options

        try:
            embedding_sess = build_session(args.embedding_model, provider, emb_opts, "embedding")
            prediction_sess = build_session(args.prediction_model, provider, pred_opts, "prediction")
        except Exception:
            continue

        actual_providers = embedding_sess.get_providers()
        if actual_providers[0] != provider:
            print(f"  {provider}: session silently fell back to {actual_providers[0]} for {sample_name} - skipping")
            continue

        times, cold = timed_runs(
            embedding_sess, prediction_sess, patches, args.batch_size, args.runs, args.warmup
        )
        out["times"][sample_name] = times
        out["cold_ms"][sample_name] = cold * 1000
        print(f"  {provider} / {sample_name}: cold={cold * 1000:.1f}ms  {fmt_ms(times)}", flush=True)

    return out


def run_worker(args):
    # Invoked as a re-exec'd subprocess for exactly one provider - see main().
    # Prints its own per-sample lines as it goes (parent just streams stdout),
    # then dumps the full result as JSON to the outfile for the parent to load.
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    assert len(providers) == 1, "worker mode takes exactly one provider"
    result = run_one_provider(providers[0], args)
    with open(getattr(args, "_worker_outfile"), "w") as f:
        json.dump(result, f)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if getattr(args, "_worker_outfile", None):
        run_worker(args)
        return

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    for p in providers:
        if p not in available:
            print(f"WARNING: {p} not in this ORT build's available providers ({sorted(available)}) - skipping", file=sys.stderr)
    providers = [p for p in providers if p in available]
    if not providers:
        print("No requested providers available on this ORT build. Nothing to benchmark.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.embedding_model) or not os.path.exists(args.prediction_model):
        print(f"Model files not found: {args.embedding_model}, {args.prediction_model}", file=sys.stderr)
        sys.exit(1)

    print(
        f"providers: {providers}, runs={args.runs} warmup={args.warmup} "
        f"(each provider runs in its own subprocess)\n"
    )

    # Rebuild the same CLI args to pass through to each subprocess, minus
    # --providers (overridden per worker) and the hidden worker flag.
    passthrough = [
        "--embedding-model", args.embedding_model,
        "--prediction-model", args.prediction_model,
        "--runs", str(args.runs),
        "--warmup", str(args.warmup),
        "--batch-size", str(args.batch_size),
        "--cache-dir", args.cache_dir,
        "--sample-cache-dir", args.sample_cache_dir,
    ]
    if args.fp16:
        passthrough.append("--fp16")
    if args.no_download:
        passthrough.append("--no-download")
    if args.audio_dir:
        passthrough += ["--audio-dir", args.audio_dir]

    results = {}
    crashed = []
    script_path = os.path.abspath(__file__)
    for provider in providers:
        print(f"=== {provider} (subprocess) ===", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            outfile = tmp.name
        cmd = [
            sys.executable, script_path,
            "--providers", provider,
            _WORKER_FLAG, outfile,
        ] + passthrough
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(
                f"  {provider}: subprocess exited with code {proc.returncode} "
                f"(negative = killed by signal - e.g. -11 is SIGSEGV) - "
                f"treating as a CRASH, not a valid measurement.\n"
            )
            crashed.append(provider)
        elif os.path.exists(outfile):
            with open(outfile) as f:
                results[provider] = json.load(f)
        try:
            os.unlink(outfile)
        except OSError:
            pass
        print()

    print("=== Overall (mean ms across all samples, lower is better) ===")
    overall = {}
    for provider, data in results.items():
        all_times = [t for times in data["times"].values() for t in times]
        if all_times:
            overall[provider] = statistics.mean(all_times) * 1000
    for provider, mean_ms in sorted(overall.items(), key=lambda kv: kv[1]):
        print(f"  {provider}: {mean_ms:.1f}ms")
    for provider in crashed:
        print(f"  {provider}: CRASHED - no valid measurement")
    if len(overall) == 2:
        (p1, t1), (p2, t2) = sorted(overall.items(), key=lambda kv: kv[1])
        speedup = t2 / t1 if t1 else float("inf")
        print(f"\n{p1} is {speedup:.2f}x faster than {p2} on average across these samples.")


if __name__ == "__main__":
    main()
