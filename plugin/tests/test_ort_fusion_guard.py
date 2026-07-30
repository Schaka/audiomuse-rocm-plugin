"""The InferenceSession wrapper that keeps broken fused kernels off an EP."""

import sys

from plugin.rocm_accelerator import ort_fusion_guard

DISABLE_KEY = "optimization.disable_specified_optimizers"
ROCM_EP = "ROCMExecutionProvider"
CPU = "CPUExecutionProvider"


def _session(fake_ort, providers, sess_options=None):
    return sys.modules["onnxruntime"].InferenceSession(
        "model.onnx", sess_options=sess_options, providers=providers)


class TestInstall:
    def test_guarded_provider_gets_the_disable_entry(self, fake_ort):
        ort_fusion_guard.install(ROCM_EP, ("ConvActivationFusion",))

        session = _session(fake_ort, [(ROCM_EP, {"device_id": 0}), CPU])

        assert session.sess_options.config_entries == {
            DISABLE_KEY: "ConvActivationFusion"}

    def test_other_sessions_pass_through_untouched(self, fake_ort):
        ort_fusion_guard.install(ROCM_EP, ("ConvActivationFusion",))

        session = _session(fake_ort, [CPU])

        assert session.sess_options is None

    def test_plain_string_provider_names_match_too(self, fake_ort):
        ort_fusion_guard.install(ROCM_EP, ("ConvActivationFusion",))

        session = _session(fake_ort, [ROCM_EP, CPU])

        assert session.sess_options.config_entries[DISABLE_KEY] == \
            "ConvActivationFusion"

    def test_caller_supplied_options_are_reused_not_replaced(self, fake_ort):
        ort_fusion_guard.install(ROCM_EP, ("ConvActivationFusion",))
        opts = fake_ort.SessionOptions()

        session = _session(fake_ort, [ROCM_EP], sess_options=opts)

        assert session.sess_options is opts
        assert opts.config_entries[DISABLE_KEY] == "ConvActivationFusion"

    def test_multiple_installs_accumulate_into_one_wrapper(self, fake_ort):
        ort_fusion_guard.install(ROCM_EP, ("ConvActivationFusion",))
        first_wrapper = fake_ort.InferenceSession
        ort_fusion_guard.install("MIGraphXExecutionProvider", ("SomeOther",))

        assert fake_ort.InferenceSession is first_wrapper
        session = _session(
            fake_ort, [ROCM_EP, "MIGraphXExecutionProvider", CPU])
        assert session.sess_options.config_entries[DISABLE_KEY] == \
            "ConvActivationFusion;SomeOther"

    def test_no_optimizers_installs_nothing(self, fake_ort):
        original = fake_ort.InferenceSession

        ort_fusion_guard.install(ROCM_EP, ())

        assert fake_ort.InferenceSession is original

    def test_wrapper_is_a_subclass_so_isinstance_still_works(self, fake_ort):
        original = fake_ort.InferenceSession

        ort_fusion_guard.install(ROCM_EP, ("ConvActivationFusion",))

        session = _session(fake_ort, [ROCM_EP])
        assert isinstance(session, original)
