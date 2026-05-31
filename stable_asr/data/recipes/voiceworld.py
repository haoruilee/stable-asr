"""Prepare real VoiceWorld turn manifests from local annotation tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from stable_asr.data.formats.jsonl import iter_jsonl
from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.data.registry import write_turn_records

ID_ALIASES = ("id", "episode_id", "window_id", "sample_id", "utterance_id", "key")
AUDIO_ALIASES = ("audio", "audio_path", "wav", "path", "file", "mixed_audio")
TEXT_ALIASES = ("text", "transcript", "reference", "utterance")
ASR_TEXT_ALIASES = ("asr_text", "hypothesis", "asr_hypothesis")
SAMPLE_RATE_ALIASES = ("sample_rate", "sr", "sampling_rate")
START_ALIASES = ("start", "start_time", "window_start", "begin_time", "start_sec", "start_ms")
END_ALIASES = ("end", "end_time", "window_end", "end_sec", "end_ms")
DURATION_ALIASES = ("duration", "duration_sec", "window_sec", "duration_ms")
TURN_LABEL_ALIASES = ("turn_label", "label", "turn_state", "state")
ACTION_LABEL_ALIASES = ("action_label", "action", "expected_action", "target_action")
ASSISTANT_SPEAKING_ALIASES = ("assistant_speaking", "system_speaking", "bot_speaking", "tts_active")
OVERLAP_ALIASES = ("overlap", "has_overlap", "is_overlap")
LANGUAGE_ALIASES = ("language", "lang", "locale")
SOURCE_ALIASES = ("source", "dataset", "corpus")
SCENARIO_ALIASES = ("scenario", "scenario_id", "case", "task")
METADATA_ALIASES = ("metadata", "meta")
DEFAULT_VOICEWORLD_FACTOR_FIELDS = (
    "pause_ms",
    "vad_pause_ms",
    "duration_ms",
    "snr_db",
    "reverb",
    "speaking_rate",
    "overlap_offset_ms",
    "network_jitter_ms",
    "farfield_distance_m",
    "code_switch_ratio",
    "accent",
    "speaker_id",
    "tts_voice",
    "asr_error_rate",
)


def prepare_voiceworld_manifest(
    input_path: str | Path,
    output_path: str | Path,
    *,
    audio_root: str | Path | None = None,
    default_sample_rate: int = 16000,
    default_language: str = "unknown",
    default_source: str = "voiceworld_real",
    default_start: float = 0.0,
    factor_fields: Iterable[str] = DEFAULT_VOICEWORLD_FACTOR_FIELDS,
    id_field: str | None = None,
    audio_field: str | None = None,
    text_field: str | None = None,
    scenario_field: str | None = None,
    turn_label_field: str | None = None,
    action_label_field: str | None = None,
) -> list[TurnManifestRecord]:
    """Normalize TSV/CSV/JSONL VoiceWorld annotations into a turn manifest."""

    records = prepare_voiceworld_manifest_rows(
        _read_rows(input_path),
        audio_root=audio_root,
        default_sample_rate=default_sample_rate,
        default_language=default_language,
        default_source=default_source,
        default_start=default_start,
        factor_fields=factor_fields,
        id_field=id_field,
        audio_field=audio_field,
        text_field=text_field,
        scenario_field=scenario_field,
        turn_label_field=turn_label_field,
        action_label_field=action_label_field,
    )
    write_turn_records(output_path, records, format="jsonl")
    return records


def prepare_voiceworld_manifest_rows(
    rows: Iterable[dict[str, Any]],
    *,
    audio_root: str | Path | None = None,
    default_sample_rate: int = 16000,
    default_language: str = "unknown",
    default_source: str = "voiceworld_real",
    default_start: float = 0.0,
    factor_fields: Iterable[str] = DEFAULT_VOICEWORLD_FACTOR_FIELDS,
    id_field: str | None = None,
    audio_field: str | None = None,
    text_field: str | None = None,
    scenario_field: str | None = None,
    turn_label_field: str | None = None,
    action_label_field: str | None = None,
) -> list[TurnManifestRecord]:
    records: list[TurnManifestRecord] = []
    audio_root_path = Path(audio_root) if audio_root is not None else None
    factor_set = {str(field) for field in factor_fields if str(field)}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index + 1} must be an object")
        normalized = {str(key): value for key, value in row.items() if key is not None}
        try:
            records.append(
                _record_from_row(
                    normalized,
                    index=index,
                    audio_root=audio_root_path,
                    default_sample_rate=default_sample_rate,
                    default_language=default_language,
                    default_source=default_source,
                    default_start=default_start,
                    factor_fields=factor_set,
                    id_field=id_field,
                    audio_field=audio_field,
                    text_field=text_field,
                    scenario_field=scenario_field,
                    turn_label_field=turn_label_field,
                    action_label_field=action_label_field,
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
    default_start: float,
    factor_fields: set[str],
    id_field: str | None,
    audio_field: str | None,
    text_field: str | None,
    scenario_field: str | None,
    turn_label_field: str | None,
    action_label_field: str | None,
) -> TurnManifestRecord:
    picked: set[str] = set()
    record_id, key = _pick(row, ID_ALIASES, explicit=id_field, required=True)
    picked.add(str(key))
    audio, key = _pick(row, AUDIO_ALIASES, explicit=audio_field, required=True)
    picked.add(str(key))
    turn_label, key = _pick(row, TURN_LABEL_ALIASES, explicit=turn_label_field, required=True)
    picked.add(str(key))
    action_label, key = _pick(row, ACTION_LABEL_ALIASES, explicit=action_label_field, required=True)
    picked.add(str(key))
    scenario, key = _pick(row, SCENARIO_ALIASES, explicit=scenario_field, required=True)
    picked.add(str(key))

    text, key = _pick(row, TEXT_ALIASES, explicit=text_field, required=False)
    _maybe_add(picked, key)
    asr_text, key = _pick(row, ASR_TEXT_ALIASES, required=False)
    _maybe_add(picked, key)
    sample_rate, key = _pick(row, SAMPLE_RATE_ALIASES, required=False)
    _maybe_add(picked, key)
    start, start_key = _pick(row, START_ALIASES, required=False)
    _maybe_add(picked, start_key)
    end, end_key = _pick(row, END_ALIASES, required=False)
    _maybe_add(picked, end_key)
    duration, duration_key = _pick(row, DURATION_ALIASES, required=False)
    _maybe_add(picked, duration_key)
    assistant_speaking, key = _pick(row, ASSISTANT_SPEAKING_ALIASES, required=False)
    _maybe_add(picked, key)
    overlap, key = _pick(row, OVERLAP_ALIASES, required=False)
    _maybe_add(picked, key)
    language, key = _pick(row, LANGUAGE_ALIASES, required=False)
    _maybe_add(picked, key)
    source, key = _pick(row, SOURCE_ALIASES, required=False)
    _maybe_add(picked, key)
    metadata, key = _pick(row, METADATA_ALIASES, required=False)
    _maybe_add(picked, key)

    start_sec = _time_seconds(start, start_key, default=default_start)
    end_sec = _resolve_end_seconds(end, end_key, duration, duration_key, start_sec)
    metadata_payload = _metadata_dict(metadata)
    for key, value in row.items():
        if key not in picked and value is not None and value != "":
            metadata_payload[key] = _coerce_metadata_value(value)
    for factor in factor_fields:
        if factor in row and row[factor] not in (None, ""):
            metadata_payload[factor] = _coerce_metadata_value(row[factor])
    metadata_payload["source_row_index"] = index
    if audio_root is not None:
        metadata_payload["audio_root"] = str(audio_root)

    return TurnManifestRecord.from_dict(
        {
            "id": str(record_id),
            "audio": _normalize_audio(str(audio), audio_root=audio_root),
            "sample_rate": _coerce_int(sample_rate, default=default_sample_rate),
            "start": start_sec,
            "end": end_sec,
            "turn_label": str(turn_label),
            "action_label": str(action_label),
            "assistant_speaking": _coerce_bool(assistant_speaking, default=False),
            "overlap": _coerce_bool(overlap, default=False),
            "language": str(language or default_language),
            "source": str(source or default_source),
            "text": _optional_str(text),
            "asr_text": _optional_str(asr_text),
            "scenario": str(scenario),
            "metadata": metadata_payload,
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
            for key in ("records", "data", "episodes"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list or contain records/data/episodes")
        return [row for row in payload]
    if suffix in {".tsv", ".csv", ".txt"}:
        delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"{path} is missing a header row")
            return [dict(row) for row in reader]
    raise ValueError(f"unsupported VoiceWorld metadata input suffix: {suffix}")


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
        raise ValueError(f"missing required VoiceWorld metadata field: {expected}")
    return None, None


def _maybe_add(picked: set[str], key: str | None) -> None:
    if key:
        picked.add(key)


def _normalize_audio(value: str, *, audio_root: Path | None) -> str:
    path = Path(value)
    if audio_root is not None and not path.is_absolute():
        return str((audio_root / path).as_posix())
    return value


def _resolve_end_seconds(end: Any, end_key: str | None, duration: Any, duration_key: str | None, start: float) -> float:
    if end is not None and end != "":
        return _time_seconds(end, end_key, default=0.0)
    if duration is None or duration == "":
        raise ValueError("missing end/end_time/window_end or duration/duration_sec/duration_ms")
    return start + _time_seconds(duration, duration_key, default=0.0)


def _time_seconds(value: Any, key: str | None, *, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key or 'time'} must be a number, got {value!r}") from exc
    if key and key.endswith("_ms"):
        return number / 1000.0
    return number


def _coerce_int(value: Any, *, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_rate must be an integer, got {value!r}") from exc


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"boolean field must be true/false, got {value!r}")


def _metadata_dict(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("metadata must be a JSON object string when provided") from exc
        if not isinstance(parsed, dict):
            raise ValueError("metadata must be a JSON object")
        return dict(parsed)
    raise ValueError("metadata must be an object or JSON object string")


def _coerce_metadata_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return value
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." not in text:
            return int(text)
        return float(text)
    except ValueError:
        return value


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
