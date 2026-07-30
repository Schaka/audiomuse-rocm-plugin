"""asr_backend/asr_backend_variant resolution: the fallback chain a blocked
combo goes through, and register()'s end-to-end behavior per backend.
"""

import pytest

from plugin.rocm_accelerator import _resolve_asr_backend, register
from plugin.rocm_accelerator.arch.base import ArchProfile

MIGRAPHX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"

pytestmark = pytest.mark.usefixtures("settings", "cache_root")


class TestResolveAsrBackend:
    def test_defaults_to_faster_whisper_vulkan(self):
        profile = ArchProfile()
        assert _resolve_asr_backend(profile) == ("faster_whisper", "vulkan")

    def test_honors_an_unblocked_selection(self, settings):
        settings["asr_backend"] = "whisper_cpp"
        settings["asr_backend_variant"] = "hip"
        profile = ArchProfile()

        assert _resolve_asr_backend(profile) == ("whisper_cpp", "hip")

    def test_falls_back_to_vulkan_when_the_variant_is_blocked(self, settings):
        settings["asr_backend"] = "parakeet_cpp"
        settings["asr_backend_variant"] = "hip"
        profile = ArchProfile()
        profile.blocked_asr_backends = frozenset({("parakeet_cpp", "hip")})

        assert _resolve_asr_backend(profile) == ("parakeet_cpp", "vulkan")

    def test_falls_back_to_faster_whisper_when_even_vulkan_is_blocked(self, settings):
        settings["asr_backend"] = "parakeet_cpp"
        settings["asr_backend_variant"] = "hip"
        profile = ArchProfile()
        profile.blocked_asr_backends = frozenset({
            ("parakeet_cpp", "hip"), ("parakeet_cpp", "vulkan"),
        })

        backend, _variant = _resolve_asr_backend(profile)
        assert backend == "faster_whisper"


class TestRegisterDispatchesByBackend:
    """Backend binaries/models don't exist in the test environment, so a
    non-faster_whisper selection resolves to "unavailable" and no ASR
    provider is registered - the behavior under test is that register()
    reaches that conclusion without raising, not that the backend runs.
    """

    def test_selecting_whisper_cpp_does_not_crash_registration(self, ctx, gpu, settings):
        settings["asr_backend"] = "whisper_cpp"

        register(ctx)  # binaries absent in the test env - must not raise

        assert "asr" not in ctx.analysis_providers

    def test_selecting_parakeet_cpp_does_not_crash_registration(self, ctx, gpu, settings):
        settings["asr_backend"] = "parakeet_cpp"

        register(ctx)

        assert "asr" not in ctx.analysis_providers

    def test_blocked_combo_on_gfx803_never_reaches_parakeet_cpp_hip(self, ctx, gpu, settings):
        gpu.arch = "gfx803"
        gpu.providers = (MIGRAPHX, CPU)
        settings["asr_backend"] = "parakeet_cpp"
        settings["asr_backend_variant"] = "hip"

        # Falls back to parakeet_cpp+vulkan (also unavailable in the test env,
        # binaries don't exist here), never faster_whisper, since only the
        # hip variant is blocked for this arch.
        register(ctx)

        assert "asr" not in ctx.analysis_providers
