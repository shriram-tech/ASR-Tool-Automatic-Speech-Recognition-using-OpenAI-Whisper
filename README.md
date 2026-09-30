# ASR Tool – Automatic Speech Recognition using OpenAI Whisper

A command-line Automatic Speech Recognition (ASR) tool that converts speech to text **offline**. It can transcribe **audio files** as well as **live microphone input**, supports many languages, and can export results as plain text, SRT subtitles, or JSON.

**Task 2 – ASR Tool Implementation**

## Team

| Name | Register Number |
|------|-----------------|
| Shriram S | RA2311003050182 |
| Sugash P | RA2311003050222 |

## Features

- Transcribe audio files (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.aac`, `.mp4`, `.webm`, ...)
- Transcribe live speech from the microphone (fixed duration, or press Enter to stop)
- Five Whisper model sizes: `tiny`, `base`, `small`, `medium`, `large`
- Automatic language detection, or choose a language manually (e.g. `en`, `ta`, `hi`)
- Translate speech from any supported language into English
- Output as `txt`, `srt` (subtitles with timestamps) or `json`
- Runs fully offline after the first model download
- Unit tests with mocked Whisper and microphone (no model download needed)

## Tools & Technologies Used

| Tool / Technology | Purpose |
|-------------------|---------|
| Python 3.8+ | Programming language |
| [OpenAI Whisper](https://github.com/openai/whisper) | Speech recognition model |
| PyTorch | Deep-learning backend used by Whisper |
| FFmpeg | Decoding audio files |
| NumPy | Audio array handling |
| sounddevice (PortAudio) | Microphone recording |
| argparse | Command-line interface |
| unittest | Testing |
| Git & GitHub | Version control and hosting |

## Project Structure

```
asr-tool/
├── app.py                # Command-line interface (entry point)
├── asr_engine.py         # Whisper transcription, mic recording, output formatting
├── test_asr_engine.py    # Unit tests
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## How It Works

1. **Input** – an audio file is decoded with FFmpeg, or the microphone is recorded at 16 kHz mono.
2. **Recognition** – the audio is passed to a Whisper encoder–decoder transformer model, which predicts the text and timestamps.
3. **Output** – the result is printed or saved as TXT, SRT, or JSON.

## Installation

### 1. Prerequisites

- Python 3.8 or newer
- Git
- FFmpeg (needed for audio files)

```bash
# Windows
winget install ffmpeg        # or: choco install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg libportaudio2
```

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/asr-tool.git
cd asr-tool
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> The Whisper model downloads automatically on first run (`base` is about 140 MB). After that it works offline.

## Usage

### Transcribe an audio file

```bash
python app.py file sample.wav
```

### Transcribe from the microphone

```bash
# Record for 10 seconds
python app.py mic --duration 10

# Record until you press Enter
python app.py mic
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m`, `--model` | `tiny`, `base`, `small`, `medium`, `large` | `base` |
| `-l`, `--language` | Language code (`en`, `ta`, `hi`, ...) | auto-detect |
| `-t`, `--task` | `transcribe` or `translate` (to English) | `transcribe` |
| `-f`, `--format` | `txt`, `srt`, `json` | `txt` |
| `-o`, `--output` | Save result to a file | print to screen |
| `-d`, `--duration` | (mic only) seconds to record | until Enter |

### Examples

```bash
# Higher accuracy with the small model, saved as subtitles
python app.py file lecture.mp3 -m small -f srt -o lecture.srt

# Tamil speech translated into English text
python app.py file speech.wav -l ta -t translate

# 15 seconds from the mic, saved as JSON
python app.py mic -d 15 -f json -o result.json

# Help
python app.py --help
python app.py file --help
```

### Model guide

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~75 MB | Fastest | Lowest |
| base | ~140 MB | Fast | Good |
| small | ~460 MB | Medium | Better |
| medium | ~1.5 GB | Slow | High |
| large | ~3 GB | Slowest | Highest |

## Running the Tests

```bash
python -m unittest test_asr_engine -v
```

The tests cover the output formatters, input validation, the transcriber, the CLI, and the recorder. Whisper and the microphone are mocked, so they run quickly without downloading a model.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: ffmpeg` | Install FFmpeg and restart the terminal |
| `Microphone support needs 'sounddevice'...` | `pip install sounddevice` (Linux: `sudo apt install libportaudio2`) |
| Empty or poor transcription | Use a larger model (`-m small`) or set the language with `-l` |
| Slow on CPU | Use `-m tiny` or `-m base` |

## Limitations

- Accuracy depends on audio quality, accents, and background noise.
- Larger models need more RAM and run slowly without a GPU.
- Transcription runs after recording finishes (not word-by-word streaming).

## Future Improvements

- Real-time streaming transcription
- Web interface (Gradio/Streamlit)
- Speaker diarization
- Batch transcription of whole folders

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
