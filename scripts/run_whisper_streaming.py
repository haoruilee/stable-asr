"""Run OpenAI Whisper over a Stable-ASR ASR manifest.

The script writes a raw Whisper-style JSONL export. Normalize it with
``scripts/export_streaming_transcript.py --schema whisper`` before evaluating
with the Stable-ASR streaming metrics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_asr.data.asr_manifest import ASRManifestRecord, load_asr_manifest
from stable_asr.data.formats.jsonl import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fp16", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        import whisper  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "ERROR: openai-whisper is not installed. Install it with "
            "`python -m pip install openai-whisper` or run another command adapter."
        ) from exc

    records = load_asr_manifest(args.manifest)
    if args.max_records is not None:
        records = records[: args.max_records]

    model = whisper.load_model(args.model, device=args.device)
    rows = [
        _transcribe_record(
            model,
            record,
            manifest_path=args.manifest,
            device=args.device,
            model_name=args.model,
            language=args.language,
            temperature=args.temperature,
            fp16=_resolve_fp16(args.fp16, args.device),
            word_timestamps=args.word_timestamps,
            verbose=args.verbose,
        )
        for record in records
    ]
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} raw Whisper record(s) to {args.output}")
    return 0


def _transcribe_record(
    model: Any,
    record: ASRManifestRecord,
    *,
    manifest_path: Path,
    device: str,
    model_name: str,
    language: str | None,
    temperature: float,
    fp16: bool,
    word_timestamps: bool,
    verbose: bool,
) -> dict[str, object]:
    audio_path = _resolve_audio(record.audio, manifest_path)
    start = time.perf_counter()
    result = model.transcribe(
        str(audio_path),
        language=language,
        temperature=temperature,
        fp16=fp16,
        word_timestamps=word_timestamps,
        verbose=verbose,
    )
    runtime = time.perf_counter() - start
    segments = [_segment_to_json(segment) for segment in result.get("segments", [])]
    inferred_duration = max((float(segment.get("end", 0.0)) for segment in segments), default=0.0)
    duration = record.duration if record.duration is not None else inferred_duration
    return {
        "id": record.id,
        "audio": record.audio,
        "reference": record.text,
        "text": str(result.get("text", "")).strip(),
        "duration": duration,
        "processing_time": runtime,
        "language": result.get("language") or language or record.language,
        "segments": segments,
        "metadata": {
            "runner": "openai-whisper",
            "model": model_name,
            "device": device,
            "source": record.source,
            "sample_rate": record.sample_rate,
        },
    }


def _segment_to_json(segment: dict[str, Any]) -> dict[str, object]:
    payload: dict[str, object] = {
        "start": float(segment.get("start", 0.0)),
        "end": float(segment.get("end", 0.0)),
        "text": str(segment.get("text", "")).strip(),
    }
    if "id" in segment:
        payload["id"] = segment["id"]
    words = segment.get("words")
    if isinstance(words, list):
        payload["words"] = [
            {
                "word": str(word.get("word", word.get("text", ""))).strip(),
                "start": float(word.get("start", 0.0)),
                "end": float(word.get("end", 0.0)),
            }
            for word in words
            if isinstance(word, dict)
        ]
    return payload


def _resolve_audio(audio: str, manifest_path: Path) -> Path:
    path = Path(audio)
    if path.is_absolute():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return Path.cwd() / path


def _resolve_fp16(value: str, device: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    return device != "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
