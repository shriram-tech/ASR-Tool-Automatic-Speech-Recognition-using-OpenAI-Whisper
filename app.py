"""ASR Tool - command-line app. Run `python app.py --help` for usage."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import asr_engine as engine

__version__ = "1.0.0"


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-m", "--model", default="base", choices=engine.VALID_MODELS,
                        help="Whisper model size (default: base)")
    common.add_argument("-l", "--language", default=None,
                        help="Language code, e.g. en, ta, hi (default: auto-detect)")
    common.add_argument("-t", "--task", default="transcribe", choices=engine.VALID_TASKS,
                        help="'translate' converts speech to English text")
    common.add_argument("-f", "--format", default="txt", choices=engine.SUPPORTED_FORMATS,
                        help="Output format (default: txt)")
    common.add_argument("-o", "--output", default=None,
                        help="Save the result to this file instead of printing")

    parser = argparse.ArgumentParser(
        prog="app.py", description="Offline speech-to-text using OpenAI Whisper."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_file = sub.add_parser("file", parents=[common], help="Transcribe an audio file")
    p_file.add_argument("path", help="Path to the audio file")

    p_mic = sub.add_parser("mic", parents=[common], help="Transcribe from the microphone")
    p_mic.add_argument("-d", "--duration", type=float, default=None,
                       help="Seconds to record (default: record until you press Enter)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        transcriber = engine.Transcriber(args.model)

        if args.command == "file":
            print(f"Transcribing '{args.path}' with Whisper '{args.model}' ...", file=sys.stderr)
            result = transcriber.transcribe_file(args.path, args.language, args.task)
        else:
            if args.duration:
                print(f"Recording for {args.duration} seconds... speak now!", file=sys.stderr)
                audio = engine.record_for(args.duration)
            else:
                print("Recording... press Enter to stop.", file=sys.stderr)
                audio = engine.record_until_enter()
            print("Transcribing ...", file=sys.stderr)
            result = transcriber.transcribe_array(audio, args.language, args.task)

        output = engine.format_result(result, args.format)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(output)
            print(f"Saved to {args.output}", file=sys.stderr)
        else:
            print(output)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
