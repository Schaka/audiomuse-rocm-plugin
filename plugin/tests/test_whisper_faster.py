"""The ASR backend's side of core's contract."""

import numpy as np
import pytest

from plugin.rocm_accelerator import whisper_faster

REQUIRED_BY_CORE = ("load_whisper_model", "transcribe", "is_loaded", "unload")


class Segment:
    def __init__(self, text, avg_logprob=None):
        self.text = text
        if avg_logprob is not None:
            self.avg_logprob = avg_logprob


class FakeModel:
    """Stands in for faster_whisper.WhisperModel; transcribe() is a generator."""

    def __init__(self, segments, language="en", duration=12.0, fail_after=None):
        self._segments = segments
        self._language = language
        self._duration = duration
        self._fail_after = fail_after
        self.calls = []

    def transcribe(self, audio, language=None, beam_size=None, vad_filter=None):
        self.calls.append({
            "samples": len(audio),
            "language": language,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
        })
        info = type("Info", (), {"language": self._language, "duration": self._duration})()
        return self._generate(), info

    def _generate(self):
        for index, segment in enumerate(self._segments):
            if self._fail_after is not None and index == self._fail_after:
                raise RuntimeError("CTranslate2 decode blew up")
            yield segment


@pytest.fixture
def loaded(monkeypatch):
    """Install a FakeModel as what load_whisper_model() returns."""

    def install(model):
        monkeypatch.setattr(whisper_faster, "load_whisper_model", lambda: model)
        return model

    return install


def audio(seconds=1.0):
    return np.zeros(int(whisper_faster.SAMPLE_RATE * seconds), dtype=np.float32)


def test_exposes_every_method_core_requires():
    for name in REQUIRED_BY_CORE:
        assert callable(getattr(whisper_faster, name, None)), name


class TestTranscribeResult:
    def test_joins_segment_text_and_reports_the_language(self, loaded):
        loaded(FakeModel([Segment(" hello "), Segment("world"), Segment("  ")]))

        result = whisper_faster.transcribe(audio(), whisper_faster.SAMPLE_RATE)

        assert result["text"] == "hello world"
        assert result["language"] == "en"

    def test_averages_the_segment_logprobs(self, loaded):
        loaded(FakeModel([Segment("a", -0.5), Segment("b", -1.5)]))

        result = whisper_faster.transcribe(audio(), whisper_faster.SAMPLE_RATE)

        assert result["avg_logprob"] == pytest.approx(-1.0)

    def test_omits_avg_logprob_when_no_segment_reported_one(self, loaded):
        # Core skips its quality gate on a missing avg_logprob but would fail
        # every transcript on a placeholder value.
        loaded(FakeModel([Segment("a"), Segment("b")]))

        result = whisper_faster.transcribe(audio(), whisper_faster.SAMPLE_RATE)

        assert result["text"] == "a b"
        assert "avg_logprob" not in result

    def test_prefers_the_backends_own_duration(self, loaded):
        loaded(FakeModel([Segment("a")], duration=42.0))

        result = whisper_faster.transcribe(audio(seconds=1.0), whisper_faster.SAMPLE_RATE)

        assert result["duration"] == 42.0

    def test_falls_back_to_the_audio_length(self, loaded):
        loaded(FakeModel([Segment("a")], duration=None))

        result = whisper_faster.transcribe(audio(seconds=3.0), whisper_faster.SAMPLE_RATE)

        assert result["duration"] == pytest.approx(3.0)


class TestTranscribeDegradation:
    def test_a_refused_load_yields_empty_text_instead_of_raising(self, monkeypatch):
        def refuse():
            raise whisper_faster.WhisperLoadRefused("no libraries here")

        monkeypatch.setattr(whisper_faster, "load_whisper_model", refuse)

        result = whisper_faster.transcribe(audio(seconds=2.0), whisper_faster.SAMPLE_RATE)

        assert result["text"] == ""
        assert result["language"] == ""
        assert result["duration"] == pytest.approx(2.0)
        assert "avg_logprob" not in result

    def test_a_mid_stream_failure_keeps_the_decoded_segments(self, loaded):
        # Decoding happens lazily inside the generator, so a GPU fault partway
        # through still leaves usable text.
        loaded(FakeModel([Segment("kept", -0.4), Segment("lost", -0.4)], fail_after=1))

        result = whisper_faster.transcribe(audio(), whisper_faster.SAMPLE_RATE)

        assert result["text"] == "kept"
        assert result["avg_logprob"] == pytest.approx(-0.4)


class TestTranscribeInputs:
    def test_passes_the_requested_language_through(self, loaded):
        model = loaded(FakeModel([Segment("a")]))

        whisper_faster.transcribe(audio(), whisper_faster.SAMPLE_RATE, language="de")

        assert model.calls[0]["language"] == "de"

    def test_an_empty_language_means_auto_detect(self, loaded):
        model = loaded(FakeModel([Segment("a")]))

        whisper_faster.transcribe(audio(), whisper_faster.SAMPLE_RATE, language="")

        assert model.calls[0]["language"] is None

    def test_leaves_voice_activity_detection_to_the_pipeline(self, loaded):
        # Core already ran VAD upstream; doing it again would drop audio twice.
        model = loaded(FakeModel([Segment("a")]))

        whisper_faster.transcribe(audio(), whisper_faster.SAMPLE_RATE)

        assert model.calls[0]["vad_filter"] is False

    def test_resamples_audio_that_is_not_16k(self, loaded, monkeypatch):
        model = loaded(FakeModel([Segment("a")]))
        calls = {}

        def fake_resample(wav, orig_sr, target_sr):
            calls.update(orig_sr=orig_sr, target_sr=target_sr)
            return np.zeros(len(wav) // 3, dtype=np.float32)

        monkeypatch.setitem(
            __import__("sys").modules, "librosa",
            type("librosa", (), {"resample": staticmethod(fake_resample)}),
        )

        whisper_faster.transcribe(np.zeros(48000, dtype=np.float32), 48000)

        assert calls == {"orig_sr": 48000, "target_sr": whisper_faster.SAMPLE_RATE}
        assert model.calls[0]["samples"] == 16000


class TestUnload:
    @pytest.fixture(autouse=True)
    def no_core_cleanup(self, monkeypatch):
        # tasks.memory_utils belongs to core and is not vendored here.
        monkeypatch.setitem(__import__("sys").modules, "tasks", None)

    def test_reports_nothing_to_do_when_no_model_was_loaded(self):
        whisper_faster._model = None
        whisper_faster._model_dir = None

        assert whisper_faster.unload() is False
        assert whisper_faster.is_loaded() is False

    def test_drops_a_loaded_model(self):
        whisper_faster._model = object()
        whisper_faster._model_dir = "/app/model/faster-whisper-small"

        assert whisper_faster.unload() is True
        assert whisper_faster.is_loaded() is False

    def test_reset_session_unloads(self):
        whisper_faster._model = object()
        whisper_faster._model_dir = "/app/model/faster-whisper-small"

        whisper_faster.reset_session()

        assert whisper_faster.is_loaded() is False
