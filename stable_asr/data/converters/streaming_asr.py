"""Converters for external streaming ASR transcript schemas.

The normalized output is the Stable-ASR ``StreamingASRRecord`` JSONL schema
consumed by ``eval-streaming-asr`` and the streaming adapter comparison tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from stable_asr.data.formats.jsonl import iter_jsonl, write_jsonl
from stable_asr.streaming.types import StreamingASRRecord

ASR_TRANSCRIPT_SCHEMAS = (
    "whisper",
    "funasr",
    "whisper_cpp",
    "whisperx",
    "qwen3_asr",
    "firered_asr2s",
    "sensevoice",
    "moonshine",
    "whisperkit",
)
GENERIC_VENDOR_SCHEMAS = frozenset(
    {
        "whisper_cpp",
        "whisperx",
        "qwen3_asr",
        "firered_asr2s",
        "sensevoice",
        "moonshine",
        "whisperkit",
    }
)


def convert_streaming_asr_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    schema: str,
) -> int:
    rows = [row for _, row in iter_jsonl(input_path)]
    records = convert_streaming_asr_rows(rows, schema=schema)
    write_jsonl(output_path, [record.to_dict() for record in records])
    return len(records)


def convert_streaming_asr_rows(
    rows: Iterable[dict[str, Any]],
    *,
    schema: str,
) -> list[StreamingASRRecord]:
    if schema not in ASR_TRANSCRIPT_SCHEMAS:
        raise ValueError(
            f"unknown ASR transcript schema {schema!r}; expected one of {ASR_TRANSCRIPT_SCHEMAS}"
        )

    records: list[StreamingASRRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index + 1} must be a JSON object")
        if schema == "whisper":
            records.append(_convert_whisper_row(row, index=index))
        elif schema == "funasr":
            records.append(_convert_funasr_row(row, index=index))
        elif schema in GENERIC_VENDOR_SCHEMAS:
            records.append(_convert_generic_vendor_row(row, index=index, schema=schema))
    return records


def _convert_whisper_row(row: dict[str, Any], *, index: int) -> StreamingASRRecord:
    segments = _list_of_dicts(row.get("segments"))
    words = _normalize_words(_pick(row, "word_timestamps", "words"))
    if not words and segments:
        words = _segment_words(segments)

    partials = _normalize_partials(row.get("partials"))
    if not partials and segments:
        partials = _partials_from_segments(segments)

    payload = _common_payload(
        row,
        index=index,
        schema="whisper",
        words=words,
        partials=partials,
        inferred_end=_max_time_from_segments(segments),
    )
    return StreamingASRRecord.from_dict(payload)


def _convert_funasr_row(row: dict[str, Any], *, index: int) -> StreamingASRRecord:
    sentences = _list_of_dicts(_pick(row, "sentence_info", "sentences"))
    words = _normalize_words(_pick(row, "word_timestamps", "words"))
    if not words:
        words = _funasr_timestamp_words(row)

    partials = _normalize_partials(row.get("partials"))
    if not partials and sentences:
        partials = _partials_from_sentences(sentences)

    payload = _common_payload(
        row,
        index=index,
        schema="funasr",
        words=words,
        partials=partials,
        inferred_end=max(_max_time_from_segments(sentences), _max_time_from_words(words)),
    )
    return StreamingASRRecord.from_dict(payload)


def _convert_generic_vendor_row(row: dict[str, Any], *, index: int, schema: str) -> StreamingASRRecord:
    """Normalize common transcript exports from command-backed upstream ASR systems.

    The supported projects do not share a stable JSON contract. This converter
    intentionally accepts a conservative intersection of fields used by modern
    ASR exporters: segments/sentences/chunks, word/timestamp lists, partial
    events, duration/runtime fields, and optional language/speaker metadata.
    Project-specific scripts should do any heavyweight inference and write one
    JSON object per utterance in this shape before Stable-ASR evaluates it.
    """

    segments = _list_of_dicts(_pick(row, "segments", "sentences", "chunks", "results"))
    words = _normalize_words(
        _pick(row, "word_timestamps", "words", "tokens", "alignment", "word_segments")
    )
    if not words:
        words = _timestamp_words(row)
    if not words and segments:
        words = _segment_words(segments)

    partials = _normalize_partials(
        _pick(row, "partials", "partial_results", "events", "streaming_events")
    )
    if not partials and segments:
        partials = _partials_from_segments(segments)

    payload = _common_payload(
        row,
        index=index,
        schema=schema,
        words=words,
        partials=partials,
        inferred_end=max(_max_time_from_segments(segments), _max_time_from_words(words)),
    )
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        for source_key, metadata_key in (
            ("language", "language"),
            ("lang", "language"),
            ("language_id", "language"),
            ("speaker", "speaker"),
            ("speaker_id", "speaker"),
            ("lid", "language"),
            ("emotion", "emotion"),
        ):
            value = _optional_str(row.get(source_key))
            if value is not None and metadata_key not in metadata:
                metadata[metadata_key] = value
    return StreamingASRRecord.from_dict(payload)


def _common_payload(
    row: dict[str, Any],
    *,
    index: int,
    schema: str,
    words: list[dict[str, object]],
    partials: list[dict[str, object]],
    inferred_end: float,
) -> dict[str, object]:
    record_id = _optional_str(_pick(row, "id", "utt_id", "key", "audio_id"))
    if record_id is None:
        record_id = f"{schema}_{index:06d}"

    final_text = _optional_str(_pick(row, "final_text", "hypothesis", "text", "pred", "transcript")) or ""
    reference = _optional_str(_pick(row, "reference", "ref", "target")) or ""

    endpoint_time = _optional_seconds(_pick(row, "endpoint_time", "finalized_at", "finalization_time"))
    speech_end_time = _optional_seconds(
        _pick(row, "speech_end_time", "end_of_speech", "reference_endpoint_time")
    )
    duration = _duration(row)
    if duration is None:
        duration = max(
            inferred_end,
            endpoint_time or 0.0,
            speech_end_time or 0.0,
            _max_time_from_partials(partials),
        )
    if endpoint_time is None:
        endpoint_time = max(_max_time_from_partials(partials), speech_end_time or 0.0, duration)

    if not partials:
        partials = [{"time": endpoint_time or duration, "text": final_text, "is_final": True}]
    elif not any(bool(partial.get("is_final")) for partial in partials):
        partials[-1] = {**partials[-1], "is_final": True}

    metadata = dict(row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {})
    metadata.update(
        {
            "source_schema": schema,
            "source_row_index": index,
            "source_row": row,
        }
    )
    audio = _optional_str(_pick(row, "audio", "audio_path", "wav", "path", "file"))
    if audio is not None:
        metadata["audio"] = audio

    payload: dict[str, object] = {
        "id": record_id,
        "reference": reference,
        "final_text": final_text,
        "audio_duration": duration or 0.0,
        "processing_time": _processing_time(row),
        "speech_end_time": speech_end_time,
        "endpoint_time": endpoint_time,
        "word_timestamps": words,
        "reference_word_timestamps": _normalize_words(
            _pick(row, "reference_word_timestamps", "reference_words")
        ),
        "partials": partials,
        "metadata": metadata,
    }
    return payload


def _normalize_partials(value: Any) -> list[dict[str, object]]:
    partials: list[dict[str, object]] = []
    for item in _list_of_dicts(value):
        time = _optional_seconds(
            _pick(
                item,
                "time",
                "time_ms",
                "end",
                "end_ms",
                "timestamp",
                "timestamp_ms",
                "event_time",
                "finalized_at",
                "t",
            )
        )
        if time is None:
            continue
        partials.append(
            {
                "time": time,
                "text": _optional_str(_pick(item, "text", "hypothesis", "transcript", "partial")) or "",
                "is_final": bool(item.get("is_final", item.get("final", False))),
            }
        )
    return partials


def _partials_from_segments(segments: list[dict[str, Any]]) -> list[dict[str, object]]:
    partials: list[dict[str, object]] = []
    cumulative: list[str] = []
    for segment in segments:
        text = _optional_str(_pick(segment, "text", "sentence", "transcript", "hypothesis"))
        end = _optional_seconds(
            _pick(segment, "end", "end_time", "timestamp_end", "end_ms", "to", "t1")
        )
        if text:
            cumulative.append(text)
        if end is not None:
            partials.append(
                {
                    "time": end,
                    "text": " ".join(cumulative).strip(),
                    "is_final": False,
                }
            )
    if partials:
        partials[-1]["is_final"] = True
    return partials


def _partials_from_sentences(sentences: list[dict[str, Any]]) -> list[dict[str, object]]:
    partials: list[dict[str, object]] = []
    cumulative: list[str] = []
    for sentence in sentences:
        text = _optional_str(_pick(sentence, "text", "sentence", "transcript"))
        end = _optional_seconds(_pick(sentence, "end", "end_time", "timestamp_end"))
        if text:
            cumulative.append(text)
        if end is not None:
            partials.append(
                {
                    "time": end,
                    "text": " ".join(cumulative).strip(),
                    "is_final": False,
                }
            )
    if partials:
        partials[-1]["is_final"] = True
    return partials


def _normalize_words(value: Any) -> list[dict[str, object]]:
    words: list[dict[str, object]] = []
    for item in _list_of_dicts(value):
        start = _optional_seconds(
            _pick(
                item,
                "start",
                "begin",
                "start_time",
                "timestamp_start",
                "start_ms",
                "begin_ms",
                "offset_start",
                "from",
                "t0",
            )
        )
        end = _optional_seconds(
            _pick(
                item,
                "end",
                "finish",
                "end_time",
                "timestamp_end",
                "end_ms",
                "finish_ms",
                "offset_end",
                "to",
                "t1",
            )
        )
        if start is None or end is None:
            continue
        words.append(
            {
                "word": _optional_str(_pick(item, "word", "text", "token", "piece")) or "",
                "start": start,
                "end": end,
            }
        )
    return words


def _segment_words(segments: list[dict[str, Any]]) -> list[dict[str, object]]:
    words: list[dict[str, object]] = []
    for segment in segments:
        words.extend(
            _normalize_words(_pick(segment, "words", "word_timestamps", "tokens", "alignment"))
        )
    return words


def _funasr_timestamp_words(row: dict[str, Any]) -> list[dict[str, object]]:
    return _timestamp_words(row)


def _timestamp_words(row: dict[str, Any]) -> list[dict[str, object]]:
    timestamps = _pick(row, "timestamp", "timestamps", "word_offsets", "token_timestamps")
    if not isinstance(timestamps, list):
        return []

    text = _optional_str(_pick(row, "final_text", "hypothesis", "text", "pred", "transcript")) or ""
    tokens = _tokens_for_timestamps(text, len(timestamps))
    if len(tokens) != len(timestamps):
        return []

    words: list[dict[str, object]] = []
    for token, item in zip(tokens, timestamps):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = _optional_seconds(item[0])
        end = _optional_seconds(item[1])
        if start is None or end is None:
            continue
        words.append({"word": token, "start": start, "end": end})
    return words


def _tokens_for_timestamps(text: str, expected: int) -> list[str]:
    spaced = text.split()
    if len(spaced) == expected:
        return spaced
    compact = [char for char in text if not char.isspace()]
    if len(compact) == expected:
        return compact
    return []


def _duration(row: dict[str, Any]) -> float | None:
    duration_ms = _optional_float(_pick(row, "duration_ms", "audio_duration_ms"))
    if duration_ms is not None:
        return duration_ms / 1000.0
    return _optional_seconds(_pick(row, "audio_duration", "duration", "duration_sec", "duration_s"))


def _processing_time(row: dict[str, Any]) -> float:
    processing_ms = _optional_float(
        _pick(row, "processing_time_ms", "runtime_ms", "elapsed_ms", "latency_ms")
    )
    if processing_ms is not None:
        return processing_ms / 1000.0
    return (
        _optional_seconds(
            _pick(row, "processing_time", "runtime", "runtime_sec", "elapsed_sec", "latency_sec")
        )
        or 0.0
    )


def _max_time_from_segments(segments: list[dict[str, Any]]) -> float:
    times = [
        value
        for segment in segments
        for value in (
            _optional_seconds(_pick(segment, "end", "end_time", "timestamp_end")),
            _max_time_from_words(_normalize_words(segment.get("words"))),
        )
        if value is not None
    ]
    return max(times, default=0.0)


def _max_time_from_words(words: list[dict[str, object]]) -> float:
    return max((float(word["end"]) for word in words if "end" in word), default=0.0)


def _max_time_from_partials(partials: list[dict[str, object]]) -> float:
    return max((float(partial["time"]) for partial in partials if "time" in partial), default=0.0)


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_seconds(value: Any) -> float | None:
    number = _optional_float(value)
    if number is None:
        return None
    if abs(number) >= 100.0:
        return number / 1000.0
    return number
