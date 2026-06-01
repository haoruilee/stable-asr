"""Run FunASR over a Stable-ASR ASR manifest.

The script writes a raw FunASR-style JSONL export. Normalize it with
``scripts/export_streaming_transcript.py --schema funasr`` before evaluating
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
    parser.add_argument("--model", default="paraformer-zh")
    parser.add_argument("--vad-model")
    parser.add_argument("--punc-model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--batch-size-s", type=int, default=60)
    parser.add_argument("--disable-update", action="store_true")
    args = parser.parse_args()

    try:
        from funasr import AutoModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "ERROR: funasr is not installed. Install it with "
            "`python -m pip install funasr` or run another command adapter."
        ) from exc

    records = load_asr_manifest(args.manifest)
    if args.max_records is not None:
        records = records[: args.max_records]

    model_kwargs: dict[str, object] = {"model": args.model, "device": args.device}
    if args.vad_model:
        model_kwargs["vad_model"] = args.vad_model
    if args.punc_model:
        model_kwargs["punc_model"] = args.punc_model
    if args.disable_update:
        model_kwargs["disable_update"] = True

    try:
        model = AutoModel(**model_kwargs)
    except TypeError:
        model_kwargs.pop("disable_update", None)
        model = AutoModel(**model_kwargs)

    rows = [
        _transcribe_record(
            model,
            record,
            manifest_path=args.manifest,
            model_name=args.model,
            device=args.device,
            batch_size_s=args.batch_size_s,
        )
        for record in records
    ]
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} raw FunASR record(s) to {args.output}")
    return 0


def _transcribe_record(
    model: Any,
    record: ASRManifestRecord,
    *,
    manifest_path: Path,
    model_name: str,
    device: str,
    batch_size_s: int,
) -> dict[str, object]:
    audio_path = _resolve_audio(record.audio, manifest_path)
    start = time.perf_counter()
    result = model.generate(input=str(audio_path), batch_size_s=batch_size_s)
    runtime = time.perf_counter() - start
    row = _first_result(result)
    row.update(
        {
            "key": record.id,
            "wav": record.audio,
            "reference": record.text,
            "duration": record.duration or _duration_from_funasr(row),
            "processing_time": runtime,
            "metadata": {
                **(row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}),
                "runner": "funasr",
                "model": model_name,
                "device": device,
                "source": record.source,
                "sample_rate": record.sample_rate,
            },
        }
    )
    return _json_safe(row)


def _first_result(result: Any) -> dict[str, object]:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return dict(first)
    if isinstance(result, dict):
        return dict(result)
    return {"text": str(result or "")}


def _duration_from_funasr(row: dict[str, object]) -> float:
    sentence_info = row.get("sentence_info")
    if not isinstance(sentence_info, list):
        return 0.0
    end_ms = 0.0
    for item in sentence_info:
        if isinstance(item, dict):
            value = item.get("end", item.get("end_ms", 0.0))
            if isinstance(value, (int, float)):
                end_ms = max(end_ms, float(value))
    return end_ms / 1000.0


def _resolve_audio(audio: str, manifest_path: Path) -> Path:
    path = Path(audio)
    if path.is_absolute():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return Path.cwd() / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
