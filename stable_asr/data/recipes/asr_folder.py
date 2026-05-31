"""Prepare utterance-level ASR manifests from local metadata tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from stable_asr.data.asr_manifest import ASRManifestRecord, write_asr_manifest
from stable_asr.data.formats.jsonl import iter_jsonl

ID_ALIASES = ("id", "utt_id", "utterance_id", "key", "audio_id")
AUDIO_ALIASES = ("audio", "audio_path", "wav", "path", "file", "wav_path")
TEXT_ALIASES = ("text", "transcript", "transcription", "reference", "sentence")
SAMPLE_RATE_ALIASES = ("sample_rate", "sr", "sampling_rate")
DURATION_ALIASES = ("duration", "duration_sec", "audio_duration")
LANGUAGE_ALIASES = ("language", "lang", "locale")
SOURCE_ALIASES = ("source", "dataset", "corpus")
SPLIT_ALIASES = ("split", "subset", "partition")
SPEAKER_ALIASES = ("speaker_id", "speaker", "spk", "speakerid")


def prepare_asr_manifest(
    input_path: str | Path,
    output_path: str | Path,
    *,
    audio_root: str | Path | None = None,
    default_sample_rate: int = 16000,
    default_language: str = "unknown",
    default_source: str = "asr_manifest",
    default_split: str | None = None,
    id_field: str | None = None,
    audio_field: str | None = None,
    text_field: str | None = None,
    duration_field: str | None = None,
    speaker_field: str | None = None,
) -> list[ASRManifestRecord]:
    """Normalize TSV/CSV/JSONL ASR metadata into the Stable-ASR ASR manifest."""

    rows = _read_rows(input_path)
    records = prepare_asr_manifest_rows(
        rows,
        audio_root=audio_root,
        default_sample_rate=default_sample_rate,
        default_language=default_language,
        default_source=default_source,
        default_split=default_split,
        id_field=id_field,
        audio_field=audio_field,
        text_field=text_field,
        duration_field=duration_field,
        speaker_field=speaker_field,
    )
    write_asr_manifest(output_path, records)
    return records


def prepare_asr_manifest_rows(
    rows: Iterable[dict[str, Any]],
    *,
    audio_root: str | Path | None = None,
    default_sample_rate: int = 16000,
    default_language: str = "unknown",
    default_source: str = "asr_manifest",
    default_split: str | None = None,
    id_field: str | None = None,
    audio_field: str | None = None,
    text_field: str | None = None,
    duration_field: str | None = None,
    speaker_field: str | None = None,
) -> list[ASRManifestRecord]:
    records: list[ASRManifestRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index + 1} must be an object")
        normalized = {str(key): value for key, value in row.items() if key is not None}
        try:
            records.append(
                _record_from_row(
                    normalized,
                    index=index,
                    audio_root=Path(audio_root) if audio_root is not None else None,
                    default_sample_rate=default_sample_rate,
                    default_language=default_language,
                    default_source=default_source,
                    default_split=default_split,
                    id_field=id_field,
                    audio_field=audio_field,
                    text_field=text_field,
                    duration_field=duration_field,
                    speaker_field=speaker_field,
                )
            )
        except ValueError as exc:
            raise ValueError(f"row {index + 1}: {exc}") from exc
    return records


def _record_from_row(
    row: dict[str, Any],
    *,
    index: int,
    audio_root: Path | None,
    default_sample_rate: int,
    default_language: str,
    default_source: str,
    default_split: str | None,
    id_field: str | None,
    audio_field: str | None,
    text_field: str | None,
    duration_field: str | None,
    speaker_field: str | None,
) -> ASRManifestRecord:
    picked: set[str] = set()
    record_id, key = _pick(row, ID_ALIASES, explicit=id_field, required=True)
    picked.add(key)
    audio, key = _pick(row, AUDIO_ALIASES, explicit=audio_field, required=True)
    picked.add(key)
    text, key = _pick(row, TEXT_ALIASES, explicit=text_field, required=True)
    picked.add(key)

    sample_rate, key = _pick(row, SAMPLE_RATE_ALIASES, required=False)
    if key:
        picked.add(key)
    language, key = _pick(row, LANGUAGE_ALIASES, required=False)
    if key:
        picked.add(key)
    source, key = _pick(row, SOURCE_ALIASES, required=False)
    if key:
        picked.add(key)
    split, key = _pick(row, SPLIT_ALIASES, required=False)
    if key:
        picked.add(key)
    duration, key = _pick(row, DURATION_ALIASES, explicit=duration_field, required=False)
    if key:
        picked.add(key)
    speaker_id, key = _pick(row, SPEAKER_ALIASES, explicit=speaker_field, required=False)
    if key:
        picked.add(key)

    metadata = {
        key: value
        for key, value in row.items()
        if key not in picked and value is not None and value != ""
    }
    metadata["source_row_index"] = index
    if audio_root is not None:
        metadata["audio_root"] = str(audio_root)

    return ASRManifestRecord.from_dict(
        {
            "id": str(record_id),
            "audio": _normalize_audio(str(audio), audio_root=audio_root),
            "sample_rate": _coerce_int(sample_rate, default=default_sample_rate),
            "text": str(text),
            "language": str(language or default_language),
            "source": str(source or default_source),
            "duration": _coerce_optional_float(duration),
            "split": _optional_str(split or default_split),
            "speaker_id": _optional_str(speaker_id),
            "metadata": metadata,
        }
    )


def _read_rows(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [row for _, row in iter_jsonl(path)]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("records", "data", "utterances"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list or contain records/data/utterances")
        return [row for row in payload]
    if suffix in {".tsv", ".csv", ".txt"}:
        delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"{path} is missing a header row")
            return [dict(row) for row in reader]
    raise ValueError(f"unsupported ASR metadata input suffix: {suffix}")


def _pick(
    row: dict[str, Any],
    aliases: tuple[str, ...],
    *,
    explicit: str | None = None,
    required: bool,
) -> tuple[Any, str | None]:
    keys = (explicit,) if explicit is not None else aliases
    for key in keys:
        if key is None:
            continue
        value = row.get(key)
        if value is not None and value != "":
            return value, key
    if required:
        expected = explicit or "/".join(aliases)
        raise ValueError(f"missing required ASR metadata field: {expected}")
    return None, None


def _normalize_audio(value: str, *, audio_root: Path | None) -> str:
    path = Path(value)
    if audio_root is not None and not path.is_absolute():
        return str((audio_root / path).as_posix())
    return value


def _coerce_int(value: Any, *, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_rate must be an integer, got {value!r}") from exc


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"duration must be a number, got {value!r}") from exc


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
