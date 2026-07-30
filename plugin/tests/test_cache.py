"""The compiled-model cache layout and the two option shapes."""

import pytest

from plugin.rocm_accelerator import cache

pytestmark = pytest.mark.usefixtures("cache_root")


class TestCacheDir:
    def test_precision_gets_its_own_subdirectory(self, cache_root):
        # MIGraphX does not key artifacts on precision, so one shared directory
        # would serve an fp32 artifact as a hit after fp16 is switched on.
        assert cache.cache_dir(True) != cache.cache_dir(False)
        assert cache.cache_dir(True) == str(cache_root / "fp16")
        assert cache.cache_dir(False) == str(cache_root / "fp32")

    def test_the_directory_exists_afterwards(self, cache_root):
        # The EP's save path does no error handling and raises out of session
        # creation if the directory is missing.
        path = cache.cache_dir(True)

        assert (cache_root / "fp16").is_dir()
        assert cache.cache_dir(True) == path  # already there, still fine


class TestPerModelOptions:
    def test_saves_when_no_compiled_model_exists_yet(self, cache_root):
        options = cache.per_model_options("musicnn", fp16=False)

        assert options == {
            "migraphx_save_compiled_model": "True",
            "migraphx_save_model_path": str(cache_root / "fp32" / "musicnn.mxr"),
        }

    def test_loads_once_the_file_is_there(self, cache_root):
        path = cache_root / "fp32" / "musicnn.mxr"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"compiled")

        options = cache.per_model_options("musicnn", fp16=False)

        assert options == {
            "migraphx_load_compiled_model": "True",
            "migraphx_load_model_path": str(path),
        }

    def test_never_mixes_save_and_load(self, cache_root):
        # The EP rejects a session asking for both.
        for exists in (False, True):
            if exists:
                path = cache_root / "fp16" / "clap.mxr"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"compiled")
            options = cache.per_model_options("clap", fp16=True)
            assert ("migraphx_save_compiled_model" in options) != (
                "migraphx_load_compiled_model" in options
            )

    def test_models_do_not_share_a_file(self, cache_root):
        musicnn = cache.per_model_options("musicnn", fp16=True)["migraphx_save_model_path"]
        clap = cache.per_model_options("clap", fp16=True)["migraphx_save_model_path"]

        assert musicnn != clap

    def test_precisions_do_not_share_a_file(self, cache_root):
        fp16 = cache.per_model_options("musicnn", fp16=True)["migraphx_save_model_path"]
        fp32 = cache.per_model_options("musicnn", fp16=False)["migraphx_save_model_path"]

        assert fp16 != fp32


def test_cache_dir_options_names_the_directory(cache_root):
    assert cache.cache_dir_options(True) == {
        "migraphx_model_cache_dir": str(cache_root / "fp16")
    }
