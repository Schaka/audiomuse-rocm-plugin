"""Arch detection from rocminfo output, and the capability probes."""

import subprocess
import types

import pytest

from plugin.rocm_accelerator import gpu

# Trimmed to the shape that matters: a CPU agent is listed before the GPU one,
# and both use the same "Name:" key.
ROCMINFO_OUTPUT = """
*******
Agent 1
*******
  Name:                    AMD Ryzen 7 5800X 8-Core Processor
  Marketing Name:          AMD Ryzen 7 5800X 8-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon RX 9070 XT
  Device Type:             GPU
"""


def fake_run(stdout="", exc=None):
    def run(*args, **kwargs):
        if exc is not None:
            raise exc
        return types.SimpleNamespace(stdout=stdout, returncode=0)

    return run


class TestDetectArch:
    def test_returns_the_gpu_agents_isa(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", fake_run(ROCMINFO_OUTPUT))

        assert gpu.detect_arch() == "gfx1201"

    def test_ignores_cpu_agents(self, monkeypatch):
        cpu_only = "  Name:                    AMD Ryzen 7 5800X\n  Device Type:      CPU\n"
        monkeypatch.setattr(subprocess, "run", fake_run(cpu_only))

        assert gpu.detect_arch() is None

    def test_takes_the_first_gpu_on_a_multi_gpu_host(self, monkeypatch):
        # device_id 0 is what the plugin asks the EP for, so the first agent is
        # the one whose quirks apply.
        monkeypatch.setattr(
            subprocess, "run", fake_run("  Name: gfx1030\n  Name: gfx1201\n")
        )

        assert gpu.detect_arch() == "gfx1030"

    @pytest.mark.parametrize("exc", [
        FileNotFoundError("rocminfo"),
        subprocess.TimeoutExpired("rocminfo", 10),
        subprocess.CalledProcessError(1, "rocminfo"),
        OSError("no permission"),
    ])
    def test_an_unusable_rocminfo_is_not_fatal(self, monkeypatch, exc):
        monkeypatch.setattr(subprocess, "run", fake_run(exc=exc))

        assert gpu.detect_arch() is None

    def test_does_not_initialize_a_gpu_context(self, monkeypatch):
        # A HIP context created here would not survive the worker's fork(), so
        # detection has to stay in a separate process.
        calls = []
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: (calls.append(a[0]), types.SimpleNamespace(
                stdout=ROCMINFO_OUTPUT, returncode=0))[1]
        )

        gpu.detect_arch()

        assert calls == [["rocminfo"]]


class TestAvailableProviders:
    def test_reports_what_onnxruntime_has(self, monkeypatch):
        ort = types.ModuleType("onnxruntime")
        ort.get_available_providers = lambda: ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
        monkeypatch.setitem(__import__("sys").modules, "onnxruntime", ort)

        assert gpu.available_providers() == (
            "MIGraphXExecutionProvider", "CPUExecutionProvider",
        )

    def test_empty_without_onnxruntime(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "onnxruntime", None)

        assert gpu.available_providers() == ()


def test_faster_whisper_probe_is_false_when_absent(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", None)

    assert gpu.faster_whisper_available() is False
