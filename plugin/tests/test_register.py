"""What register() actually hands core, per arch and per available EP set."""

import pytest

from plugin.rocm_accelerator import register
from plugin.rocm_accelerator.arch import ArchProfile, ProviderSpec

MIGRAPHX = "MIGraphXExecutionProvider"
ROCM_EP = "ROCMExecutionProvider"
CPU = "CPUExecutionProvider"

pytestmark = pytest.mark.usefixtures("settings", "cache_root")


class TestInertWithoutRuntime:
    def test_registers_nothing_without_the_migraphx_provider(self, ctx, gpu):
        gpu.providers = (CPU,)

        register(ctx)

        assert ctx.onnx_providers == []
        assert ctx.analysis_providers == {}

    def test_does_not_even_probe_the_gpu(self, ctx, gpu, monkeypatch):
        gpu.providers = (CPU,)
        monkeypatch.setattr(
            "plugin.rocm_accelerator.gpu.detect_arch",
            lambda: pytest.fail("arch detection ran on a non-ROCm image"),
        )

        register(ctx)


class TestDefaultArch:
    def test_offers_migraphx_for_musicnn_and_clap_only(self, ctx, gpu):
        register(ctx)

        providers = ctx.providers_named(MIGRAPHX)
        assert len(providers) == 1
        assert providers[0]["only_models"] == ["musicnn", "clap"]

    def test_asks_core_to_pin_symbolic_shapes(self, ctx, gpu):
        register(ctx)

        assert providers_all(ctx, MIGRAPHX, "needs_static_shapes") == [True]

    def test_uses_the_cache_directory_option(self, ctx, gpu, cache_root):
        register(ctx)

        options = ctx.providers_named(MIGRAPHX)[0]["options"]
        assert options["migraphx_model_cache_dir"] == str(cache_root / "fp16")
        assert "migraphx_save_compiled_model" not in options

    def test_registers_faster_whisper_as_the_asr_backend(self, ctx, gpu):
        register(ctx)

        assert "asr" in ctx.analysis_providers

    def test_keeps_musicnn_on_the_gpu_when_faster_whisper_is_missing(self, ctx, gpu):
        gpu.faster_whisper = False

        register(ctx)

        assert ctx.analysis_providers == {}
        assert ctx.providers_named(MIGRAPHX)

    def test_registers_no_extra_providers(self, ctx, gpu):
        gpu.providers = (MIGRAPHX, ROCM_EP, CPU)

        register(ctx)

        assert ctx.providers_named(ROCM_EP) == []

    def test_an_unknown_arch_still_gets_the_gpu(self, ctx, gpu):
        # rocminfo can be absent or unparseable; refusing to accelerate then
        # would be a silent CPU downgrade on hardware that is probably fine.
        gpu.arch = None

        register(ctx)

        assert ctx.labels_for(MIGRAPHX) == ["musicnn", "clap"]


class TestFp16:
    def test_on_by_default(self, ctx, gpu):
        register(ctx)

        assert ctx.providers_named(MIGRAPHX)[0]["options"]["migraphx_fp16_enable"] == "True"

    def test_setting_turns_it_off(self, ctx, gpu, settings):
        settings["fp16_enable"] = False

        register(ctx)

        assert "migraphx_fp16_enable" not in ctx.providers_named(MIGRAPHX)[0]["options"]

    def test_precision_splits_the_cache_directory(self, ctx, gpu, settings, cache_root):
        settings["fp16_enable"] = False

        register(ctx)

        options = ctx.providers_named(MIGRAPHX)[0]["options"]
        assert options["migraphx_model_cache_dir"] == str(cache_root / "fp32")

    def test_profile_can_veto_the_setting(self, ctx, gpu, settings):
        settings["fp16_enable"] = True
        gpu.arch = "gfx803"

        register(ctx)

        for provider in ctx.providers_named(MIGRAPHX):
            assert "migraphx_fp16_enable" not in provider["options"]


class TestGfx803:
    @pytest.fixture(autouse=True)
    def polaris(self, gpu):
        gpu.arch = "gfx803"
        gpu.providers = (MIGRAPHX, ROCM_EP, CPU)
        return gpu

    def test_splits_the_two_models_across_the_two_providers(self, ctx):
        register(ctx)

        assert ctx.labels_for(MIGRAPHX) == ["musicnn"]
        assert ctx.labels_for(ROCM_EP) == ["clap"]

    def test_never_offers_two_gpu_providers_for_one_model(self, ctx):
        # A session holding both segfaults the worker, so no label may appear
        # under more than one provider.
        register(ctx)

        seen = {}
        for provider in ctx.onnx_providers:
            for label in provider["only_models"] or []:
                seen.setdefault(label, set()).add(provider["name"])
        assert all(len(names) == 1 for names in seen.values()), seen

    def test_the_rocm_ep_needs_no_shape_pinning(self, ctx):
        register(ctx)

        assert providers_all(ctx, ROCM_EP, "needs_static_shapes") == [False]

    def test_falls_back_to_per_model_cache_files(self, ctx, cache_root):
        register(ctx)

        options = ctx.providers_named(MIGRAPHX)[0]["options"]
        assert "migraphx_model_cache_dir" not in options
        assert options["migraphx_save_compiled_model"] == "True"
        assert options["migraphx_save_model_path"] == str(cache_root / "fp32" / "musicnn.mxr")

    def test_one_registration_and_cache_file_per_model(self, ctx, polaris, cache_root):
        # Core keeps one options dict per registration, so a shared dict would
        # point both models at the same compiled-model file. Two models land on
        # MIGraphX here because this build has no ROCM EP to hand CLAP to.
        polaris.providers = (MIGRAPHX, CPU)

        register(ctx)

        paths = [p["options"]["migraphx_save_model_path"] for p in ctx.providers_named(MIGRAPHX)]
        assert paths == [
            str(cache_root / "fp32" / "musicnn.mxr"),
            str(cache_root / "fp32" / "clap.mxr"),
        ]

    def test_loads_an_existing_compiled_model_instead_of_saving(self, ctx, cache_root):
        (cache_root / "fp32").mkdir(parents=True)
        (cache_root / "fp32" / "musicnn.mxr").write_bytes(b"compiled")

        register(ctx)

        options = ctx.providers_named(MIGRAPHX)[0]["options"]
        assert options["migraphx_load_compiled_model"] == "True"
        assert options["migraphx_load_model_path"] == str(cache_root / "fp32" / "musicnn.mxr")
        assert "migraphx_save_compiled_model" not in options

    def test_still_swaps_the_asr_backend(self, ctx):
        register(ctx)

        assert "asr" in ctx.analysis_providers


class TestProfileSeams:
    """A profile's declarations reach core unchanged."""

    @pytest.fixture
    def profile(self, monkeypatch):
        profile = ArchProfile()
        monkeypatch.setattr(
            "plugin.rocm_accelerator.arch.profile_for", lambda arch_name: profile
        )
        return profile

    def test_extra_migraphx_options_are_merged(self, ctx, gpu, profile):
        profile.migraphx_options = lambda: {"migraphx_exhaustive_tune": "True"}

        register(ctx)

        assert ctx.providers_named(MIGRAPHX)[0]["options"]["migraphx_exhaustive_tune"] == "True"

    def test_a_profile_can_keep_migraphx_out_entirely(self, ctx, gpu, profile):
        profile.migraphx_models = lambda providers: ()

        register(ctx)

        assert ctx.providers_named(MIGRAPHX) == []
        assert "asr" in ctx.analysis_providers

    def test_env_is_applied_before_registration(self, ctx, gpu, profile, monkeypatch):
        monkeypatch.delenv("ROCM_TEST_KNOB", raising=False)
        profile.env = {"ROCM_TEST_KNOB": "1"}

        register(ctx)

        import os

        assert os.environ["ROCM_TEST_KNOB"] == "1"

    def test_a_provider_missing_from_the_build_is_skipped(self, ctx, gpu, profile):
        profile.extra_providers = lambda providers: (
            ProviderSpec("NotBuiltExecutionProvider", {}, ("clap",)),
        )

        register(ctx)

        assert ctx.providers_named("NotBuiltExecutionProvider") == []
        assert ctx.providers_named(MIGRAPHX)

    def test_an_unscoped_extra_provider_registers_for_every_model(self, ctx, gpu, profile):
        gpu.providers = (MIGRAPHX, ROCM_EP, CPU)
        profile.extra_providers = lambda providers: (ProviderSpec(ROCM_EP, {"device_id": 0}),)

        register(ctx)

        assert ctx.providers_named(ROCM_EP)[0]["only_models"] is None


def providers_all(ctx, name, key):
    return [provider[key] for provider in ctx.providers_named(name)]
