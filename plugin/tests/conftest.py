"""Test scaffolding: the plugin's core-side imports, faked.

The plugin lives inside AudioMuse-AI's ``plugin`` package at runtime and imports
``plugin.api`` from core, which is not vendored here. Both are stood up as
modules before the plugin is imported, so the tests run against this repo alone.
"""

import os
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _install_core_stubs():
    plugin_pkg = types.ModuleType("plugin")
    plugin_pkg.__path__ = [str(PLUGIN_DIR)]

    api = types.ModuleType("plugin.api")
    # Overwritten per test via the `settings` fixture.
    api._settings = {}
    api.get_setting = lambda key, default=None: api._settings.get(key, default)
    api.config = types.SimpleNamespace(LYRICS_ASR_BEAM_SIZE=5)

    sys.modules["plugin"] = plugin_pkg
    sys.modules["plugin.api"] = api
    return api


_api = _install_core_stubs()


class FakeSessionOptions:
    def __init__(self):
        self.config_entries = {}

    def add_session_config_entry(self, key, value):
        self.config_entries[key] = value


class FakeInferenceSession:
    """Records its constructor arguments; never touches a real runtime."""

    def __init__(self, path_or_bytes, sess_options=None, providers=None,
                 provider_options=None, **kwargs):
        self.path_or_bytes = path_or_bytes
        self.sess_options = sess_options
        self.providers = providers
        self.provider_options = provider_options


def _fresh_ort_module():
    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = FakeSessionOptions
    ort.InferenceSession = FakeInferenceSession
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    return ort


@pytest.fixture(autouse=True)
def fake_ort(monkeypatch):
    """A fresh fake onnxruntime per test, so the fusion guard's wrapping of
    ``ort.InferenceSession`` (module-global by design) cannot leak across tests."""
    from plugin.rocm_accelerator import ort_fusion_guard

    ort = _fresh_ort_module()
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr(ort_fusion_guard, "_rules", {})
    return ort


@pytest.fixture(autouse=True)
def _isolate_environ(monkeypatch):
    """Give every test a throwaway environment.

    Profiles set variables in the real ``os.environ``, and monkeypatch cannot
    undo a variable it did not set itself - without this, one test's profile
    would still be in effect in the next.
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))


@pytest.fixture
def settings():
    """The plugin's DB-stored settings, as `get_setting` will see them."""
    _api._settings = {}
    yield _api._settings
    _api._settings = {}


class RecordingContext:
    """Stands in for core's PluginContext, recording what a plugin registers."""

    def __init__(self):
        self.onnx_providers = []
        self.analysis_providers = {}

    def register_onnx_provider(self, name, options=None, position="before_cpu",
                               only_models=None, exclude_models=None,
                               needs_static_shapes=False):
        self.onnx_providers.append({
            "name": name,
            "options": dict(options or {}),
            "position": position,
            "only_models": only_models,
            "exclude_models": exclude_models,
            "needs_static_shapes": needs_static_shapes,
        })

    def register_analysis_provider(self, component, factory, cache=True):
        self.analysis_providers[component] = factory

    def providers_named(self, name):
        return [p for p in self.onnx_providers if p["name"] == name]

    def labels_for(self, name):
        """Every model label registered for `name`, flattened across entries."""
        labels = []
        for provider in self.providers_named(name):
            labels.extend(provider["only_models"] or [])
        return labels


@pytest.fixture
def ctx():
    return RecordingContext()


@pytest.fixture
def gpu(monkeypatch):
    """Fake hardware: set `gpu.arch` and `gpu.providers` to steer register()."""
    from plugin.rocm_accelerator import gpu as gpu_module

    fake = types.SimpleNamespace(
        arch="gfx1201",
        providers=("MIGraphXExecutionProvider", "CPUExecutionProvider"),
        faster_whisper=True,
    )
    monkeypatch.setattr(gpu_module, "detect_arch", lambda: fake.arch)
    monkeypatch.setattr(gpu_module, "available_providers", lambda: fake.providers)
    monkeypatch.setattr(gpu_module, "faster_whisper_available", lambda: fake.faster_whisper)
    return fake


@pytest.fixture
def cache_root(monkeypatch, tmp_path):
    """Redirect the compiled-model cache at a temp dir; register() creates it."""
    from plugin.rocm_accelerator import cache

    root = tmp_path / "migraphx"
    monkeypatch.setattr(cache, "CACHE_ROOT", str(root))
    return root
