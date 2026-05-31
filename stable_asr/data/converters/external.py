"""Flexible JSONL converters for external turn-taking datasets.

These converters are intentionally conservative. They support common field
names used by EasyTurn-style, Full-Duplex-Bench-style, and SmartTurn-style manifests, preserve
the original row in metadata, and normalize labels/actions into the Stable-ASR
schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from stable_asr.data.formats.jsonl import iter_jsonl, write_jsonl
from stable_asr.data.manifest import TurnManifestRecord

EXTERNAL_SCHEMAS = ("easyturn", "full_duplex_bench", "smart_turn")


def convert_external_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    schema: str,
    default_sample_rate: int = 16000,
    default_language: str = "unknown",
) -> int:
    rows = [row for _, row in iter_jsonl(input_path)]
    records = convert_rows(
        rows,
        schema=schema,
        default_sample_rate=default_sample_rate,
        default_language=default_language,
    )
    write_jsonl(output_path, [record.to_dict() for record in records])
    return len(records)


def convert_rows(
    rows: Iterable[dict[str, Any]],
    *,
    schema: str,
    default_sample_rate: int = 16000,
    default_language: str = "unknown",
) -> list[TurnManifestRecord]:
    if schema not in EXTERNAL_SCHEMAS:
        raise ValueError(f"unknown external schema {schema!r}; expected one of {EXTERNAL_SCHEMAS}")

    records: list[TurnManifestRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index + 1} must be a JSON object")
        records.append(
            _convert_row(
                row,
                index=index,
                schema=schema,
                default_sample_rate=default_sample_rate,
                default_language=default_language,
            )
        )
    return records


def _convert_row(
    row: dict[str, Any],
    *,
    index: int,
    schema: str,
    default_sample_rate: int,
    default_language: str,
) -> TurnManifestRecord:
    scenario = _optional_str(_pick(row, "scenario", "task", "category", "case", "type"))
    raw_label = _pick(
        row,
        "turn_label",
        "label",
        "state",
        "turn_state",
        "class",
        "is_complete",
        "complete_probability",
        "completion_probability",
        "turn_complete_probability",
        "prob_complete",
    )
    raw_action = _pick(row, "action_label", "action", "expected_action", "target_action")
    turn_label = _normalize_label(raw_label, scenario=scenario, schema=schema)
    action_label = _normalize_action(raw_action, turn_label=turn_label, scenario=scenario, schema=schema)

    assistant_speaking = _as_bool(
        _pick(row, "assistant_speaking", "system_speaking", "bot_speaking", "tts_active"),
        default=_default_assistant_speaking(scenario, action_label),
    )
    overlap = _as_bool(
        _pick(row, "overlap", "has_overlap", "is_overlap"),
        default=_default_overlap(scenario, action_label),
    )

    start = _as_float(_pick(row, "start", "start_time", "window_start"), default=0.0)
    end = _end_time(row, start=start)
    record_id = _optional_str(_pick(row, "id", "episode_id", "utterance_id", "sample_id"))
    if record_id is None:
        record_id = f"{schema}_{index:06d}"

    metadata = dict(_optional_dict(row.get("metadata")))
    metadata.update(
        {
            "source_schema": schema,
            "source_row_index": index,
            "source_row": row,
        }
    )

    return TurnManifestRecord.from_dict(
        {
            "id": record_id,
            "audio": _audio_path(row),
            "sample_rate": _as_int(_pick(row, "sample_rate", "sr"), default=default_sample_rate),
            "start": start,
            "end": end,
            "text": _optional_str(_pick(row, "text", "transcript", "reference", "utterance")),
            "asr_text": _optional_str(_pick(row, "asr_text", "hypothesis", "asr_hypothesis")),
            "turn_label": turn_label,
            "action_label": action_label,
            "assistant_speaking": assistant_speaking,
            "overlap": overlap,
            "scenario": scenario,
            "language": _optional_str(_pick(row, "language", "lang")) or default_language,
            "source": schema,
            "metadata": metadata,
        }
    )


def _normalize_label(value: Any, *, scenario: str | None, schema: str) -> str:
    if value is None and schema == "full_duplex_bench":
        return _label_from_scenario(scenario)
    if isinstance(value, bool):
        return "complete" if value else "incomplete"
    if isinstance(value, (int, float)):
        return "complete" if float(value) >= 0.5 else "incomplete"
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "complete": "complete",
        "completed": "complete",
        "finish": "complete",
        "finished": "complete",
        "take_turn": "complete",
        "interruption": "complete",
        "user_interruption": "complete",
        "incomplete": "incomplete",
        "unfinished": "incomplete",
        "partial": "incomplete",
        "continue": "incomplete",
        "backchannel": "backchannel",
        "back_channel": "backchannel",
        "bc": "backchannel",
        "listener_backchannel": "backchannel",
        "wait": "wait",
        "hold": "wait",
        "ignore": "wait",
        "ambient": "wait",
        "side_conversation": "wait",
        "not_complete": "incomplete",
        "not_finished": "incomplete",
        "finished": "complete",
    }
    if text in aliases:
        return aliases[text]
    if scenario:
        return _label_from_scenario(scenario)
    raise ValueError(f"cannot normalize turn label from {value!r}")


def _label_from_scenario(scenario: str | None) -> str:
    text = (scenario or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "backchannel" in text:
        return "backchannel"
    if "interrupt" in text:
        return "complete"
    if "incomplete" in text or "pause" in text:
        return "incomplete"
    if "side" in text or "ambient" in text or "wait" in text or "stop" in text:
        return "wait"
    return "complete"


def _normalize_action(value: Any, *, turn_label: str, scenario: str | None, schema: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "take_turn": "take_turn",
        "respond": "take_turn",
        "reply": "take_turn",
        "keep_listening": "keep_listening",
        "listen": "keep_listening",
        "continue_speaking": "continue_speaking",
        "continue_tts": "continue_speaking",
        "stop_tts_and_listen": "stop_tts_and_listen",
        "barge_in": "stop_tts_and_listen",
        "stop": "stop_tts_and_listen",
        "ignore": "ignore",
        "hold": "hold",
        "wait": "hold",
        "light_ack": "light_ack",
    }
    if text in aliases:
        return aliases[text]

    scenario_text = (scenario or "").lower().replace("-", "_").replace(" ", "_")
    if "interrupt" in scenario_text:
        return "stop_tts_and_listen"
    if "backchannel" in scenario_text:
        return "continue_speaking"
    if "side" in scenario_text or "ambient" in scenario_text:
        return "ignore"
    if turn_label == "complete":
        return "take_turn"
    if turn_label == "incomplete":
        return "keep_listening"
    if turn_label == "backchannel":
        return "continue_speaking"
    return "hold"


def _default_assistant_speaking(scenario: str | None, action_label: str) -> bool:
    scenario_text = (scenario or "").lower()
    return action_label in {"continue_speaking", "stop_tts_and_listen"} or "interrupt" in scenario_text


def _default_overlap(scenario: str | None, action_label: str) -> bool:
    scenario_text = (scenario or "").lower()
    return action_label in {"continue_speaking", "stop_tts_and_listen"} or "overlap" in scenario_text


def _audio_path(row: dict[str, Any]) -> str:
    value = _pick(row, "audio", "audio_path", "wav", "path", "file")
    if not isinstance(value, str) or not value:
        raise ValueError("external row is missing an audio path")
    return value


def _end_time(row: dict[str, Any], *, start: float) -> float:
    value = _pick(row, "end", "end_time", "window_end")
    if value is not None:
        return _as_float(value, default=start + 2.0)
    duration = _pick(row, "duration", "duration_sec", "window_sec")
    if duration is not None:
        return start + _as_float(duration, default=2.0)
    duration_ms = _pick(row, "duration_ms")
    if duration_ms is not None:
        return start + _as_float(duration_ms, default=2000.0) / 1000.0
    return start + 2.0


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    return float(value)
