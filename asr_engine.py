"""ASR engine: Whisper transcription, microphone recording and output formatting."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
VALID_MODELS = ("tiny", "base", "small", "medium", "large")
VALID_TASKS = ("transcribe", "translate")
SUPPORTED_FORMATS = ("txt", "srt", "json")
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".webm",
}
SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono audio


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------
@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str = "unknown"
    segments: List[Segment] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Transcriber
# ----------------------------------------------------------------------------
class Transcriber:
    """Loads a Whisper model lazily and transcribes files or raw audio arrays."""

    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        if model_name not in VALID_MODELS:
            raise ValueError(
                f"Invalid model '{model_name}'. Choose from: {', '.join(VALID_MODELS)}"
            )
        self.model_name = model_name
        self.device = device
        self._model = None
        self._fp16 = False

    def _load_model(self):
        if self._model is None:
            import whisper  # imported lazily so the CLI starts fast

            self._model = whisper.load_model(self.model_name, device=self.device)
            self._fp16 = str(getattr(self._model, "device", "cpu")).startswith("cuda")
        return self._model

    def transcribe_file(
        self, path: str, language: Optional[str] = None, task: str = "transcribe"
    ) -> TranscriptionResult:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return self._run(path, language, task)

    def transcribe_array(
        self, audio, language: Optional[str] = None, task: str = "transcribe"
    ) -> TranscriptionResult:
        """`audio` must be a mono float32 numpy array sampled at 16 kHz."""
        if audio is None or len(audio) == 0:
            raise ValueError("No audio data to transcribe.")
        return self._run(audio, language, task)

    def _run(self, audio, language, task) -> TranscriptionResult:
        if task not in VALID_TASKS:
            raise ValueError(f"Invalid task '{task}'. Choose from {VALID_TASKS}.")
        model = self._load_model()
        raw = model.transcribe(audio, language=language, task=task, fp16=self._fp16)
        segments = [
            Segment(float(s["start"]), float(s["end"]), s["text"])
            for s in raw.get("segments", [])
        ]
        return TranscriptionResult(
            text=raw.get("text", "").strip(),
            language=raw.get("language", language or "unknown"),
            segments=segments,
        )


# ----------------------------------------------------------------------------
# Microphone recording
# ----------------------------------------------------------------------------
def _sd():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:  # OSError: PortAudio missing
        raise RuntimeError(
            "Microphone support needs 'sounddevice' and PortAudio. "
            "Run: pip install sounddevice (Linux: sudo apt install libportaudio2)"
        ) from exc
    return sd


def record_for(duration: float, sample_rate: int = SAMPLE_RATE):
    """Record `duration` seconds and return a float32 numpy array."""
    if duration <= 0:
        raise ValueError("Duration must be greater than 0 seconds.")
    sd = _sd()
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def record_until_enter(sample_rate: int = SAMPLE_RATE):
    """Record until the user presses Enter and return a float32 numpy array."""
    import numpy as np

    sd = _sd()
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        input()
    if not chunks:
        return np.zeros(0, dtype="float32")
    return np.concatenate(chunks).flatten()


# ----------------------------------------------------------------------------
# Output formatting
# ----------------------------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    """Convert seconds to an SRT timestamp, e.g. 3.5 -> '00:00:03,500'."""
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def to_txt(result: TranscriptionResult) -> str:
    return result.text.strip()


def to_srt(result: TranscriptionResult) -> str:
    blocks = []
    for i, seg in enumerate(result.segments, start=1):
        blocks.append(
            f"{i}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text.strip()}\n"
        )
    return "\n".join(blocks)


def to_json(result: TranscriptionResult) -> str:
    return json.dumps(
        {
            "language": result.language,
            "text": result.text.strip(),
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in result.segments
            ],
        },
        indent=2,
        ensure_ascii=False,
    )


def format_result(result: TranscriptionResult, fmt: str) -> str:
    fmt = fmt.lower()
    if fmt == "txt":
        return to_txt(result)
    if fmt == "srt":
        return to_srt(result)
    if fmt == "json":
        return to_json(result)
    raise ValueError(f"Unsupported format '{fmt}'. Choose from {SUPPORTED_FORMATS}.")
