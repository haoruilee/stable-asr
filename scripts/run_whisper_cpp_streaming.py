"""Run whisper.cpp over a Stable-ASR ASR manifest.

This is a dependency-light bridge for C/C++ Whisper runtimes. It captures the
CLI transcript for each manifest row and writes a raw whisper_cpp-style JSONL
export that can be normalized with ``scripts/export_streaming_transcript.py``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_asr.data.asr_manifest import ASRManifestRecord, load_asr_manifest
from stable_asr.data.formats.jsonl import write_jsonl


TIMESTAMP_LINE = re.compile(r"^\s*\[[^\]]+\]\s*(.*)$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Path to a whisper.cpp ggml model file.")
    parser.add_argument("--binary", default="whisper-cli")
    parser.add_argument("--language")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    args = parser.parse_args()

    records = load_asr_manifest(args.manifest)
    if args.max_records is not None:
        records = records[: args.max_records]

    rows = [
        _transcribe_record(
            record,
            manifest_path=args.manifest,
            binary=args.binary,
            model_path=args.model,
            language=args.language,
            extra_args=args.extra_arg,
            timeout_sec=args.timeout_sec,
        )
        for record in records
    ]
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} raw whisper.cpp record(s) to {args.output}")
    return 0


def _transcribe_record(
    record: ASRManifestRecord,
    *,
    manifest_path: Path,
    binary: str,
    model_path: str,
    language: str | None,
    extra_args: list[str],
    timeout_sec: float,
) -> dict[str, object]:
    audio_path = _resolve_audio(record.audio, manifest_path)
    command = [binary, "-m", model_path, "-f", str(audio_path)]
    if language:
        command.extend(["-l", language])
    command.extend(extra_args)

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    runtime = time.perf_counter() - start
    text = _parse_stdout(completed.stdout)
    return {
        "id": record.id,
        "audio": record.audio,
        "reference": record.text,
        "text": text,
        "duration": record.duration or 0.0,
        "processing_time": runtime,
        "metadata": {
            "runner": "whisper.cpp",
            "binary": binary,
            "model": model_path,
            "source": record.source,
            "sample_rate": record.sample_rate,
            "stderr": completed.stderr[-4000:],
        },
    }


def _parse_stdout(stdout: str) -> str:
    lines: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = TIMESTAMP_LINE.match(line)
        lines.append(match.group(1).strip() if match else line)
    return " ".join(lines).strip()


def _resolve_audio(audio: str, manifest_path: Path) -> Path:
    path = Path(audio)
    if path.is_absolute():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return Path.cwd() / path


if __name__ == "__main__":
    raise SystemExit(main())
