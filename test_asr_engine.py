"""Unit tests. Whisper and the microphone are mocked, so no model download is needed.

Run with:  python -m unittest test_asr_engine -v
"""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import app
import asr_engine as engine
from asr_engine import Segment, Transcriber, TranscriptionResult

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

FAKE_RAW = {
    "text": " Hello world. ",
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 1.5, "text": " Hello"},
        {"start": 1.5, "end": 3.25, "text": " world."},
    ],
}


def fake_whisper_module():
    model = mock.Mock()
    model.device = "cpu"
    model.transcribe.return_value = FAKE_RAW
    module = types.ModuleType("whisper")
    module.load_model = mock.Mock(return_value=model)
    return module, model


def sample_result():
    return TranscriptionResult(
        "Hello world.", "en", [Segment(0.0, 1.5, "Hello"), Segment(1.5, 3.25, "world.")]
    )


class TestFormatters(unittest.TestCase):
    def test_timestamp(self):
        self.assertEqual(engine.format_timestamp(0), "00:00:00,000")
        self.assertEqual(engine.format_timestamp(3.5), "00:00:03,500")
        self.assertEqual(engine.format_timestamp(3725.042), "01:02:05,042")

    def test_txt(self):
        self.assertEqual(engine.format_result(sample_result(), "txt"), "Hello world.")

    def test_srt(self):
        srt = engine.format_result(sample_result(), "srt")
        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nHello", srt)
        self.assertIn("2\n00:00:01,500 --> 00:00:03,250\nworld.", srt)

    def test_json(self):
        data = json.loads(engine.format_result(sample_result(), "json"))
        self.assertEqual(data["language"], "en")
        self.assertEqual(len(data["segments"]), 2)

    def test_bad_format(self):
        with self.assertRaises(ValueError):
            engine.format_result(sample_result(), "pdf")


class TestTranscriber(unittest.TestCase):
    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            Transcriber("huge")

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            Transcriber("tiny").transcribe_file("does_not_exist.wav")

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            with self.assertRaises(ValueError):
                Transcriber("tiny").transcribe_file(tmp.name)

    def test_transcribe_file(self):
        module, model = fake_whisper_module()
        with mock.patch.dict(sys.modules, {"whisper": module}):
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                result = Transcriber("tiny").transcribe_file(tmp.name, language="en")
        self.assertEqual(result.text, "Hello world.")
        self.assertEqual(result.language, "en")
        self.assertEqual(len(result.segments), 2)
        model.transcribe.assert_called_once()

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            Transcriber("tiny").transcribe_array([])

    def test_invalid_task(self):
        module, _ = fake_whisper_module()
        with mock.patch.dict(sys.modules, {"whisper": module}):
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                with self.assertRaises(ValueError):
                    Transcriber("tiny").transcribe_file(tmp.name, task="dance")


class TestCLI(unittest.TestCase):
    def test_file_command_saves_output(self):
        module, _ = fake_whisper_module()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules, {"whisper": module}):
            audio = os.path.join(d, "a.wav")
            open(audio, "wb").close()
            out = os.path.join(d, "out.srt")
            code = app.main(["file", audio, "-m", "tiny", "-f", "srt", "-o", out])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as fh:
                self.assertIn("-->", fh.read())

    def test_missing_file_returns_error_code(self):
        self.assertEqual(app.main(["file", "nope.wav"]), 1)

    def test_mic_command(self):
        module, _ = fake_whisper_module()
        with mock.patch.dict(sys.modules, {"whisper": module}), \
                mock.patch("asr_engine.record_for", return_value=[0.1, 0.2, 0.3]) as rec:
            code = app.main(["mic", "-d", "2", "-m", "tiny"])
        self.assertEqual(code, 0)
        rec.assert_called_once_with(2.0)

    def test_requires_subcommand(self):
        with self.assertRaises(SystemExit):
            app.main([])


@unittest.skipIf(np is None, "numpy not installed")
class TestRecorder(unittest.TestCase):
    def test_record_for(self):
        sd = types.ModuleType("sounddevice")
        sd.rec = mock.Mock(return_value=np.zeros((16000, 1), dtype="float32"))
        sd.wait = mock.Mock()
        with mock.patch.dict(sys.modules, {"sounddevice": sd}):
            audio = engine.record_for(1)
        self.assertEqual(audio.shape, (16000,))

    def test_invalid_duration(self):
        with self.assertRaises(ValueError):
            engine.record_for(0)


if __name__ == "__main__":
    unittest.main()
