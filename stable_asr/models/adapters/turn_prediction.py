"""Adapter for externally generated turn prediction JSONL files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.formats.jsonl import iter_jsonl, write_jsonl
from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.turn.labels import TURN_LABELS
from stable_asr.turn.types import TurnPrediction


PREDICTION_SCHEMAS = ("generic", "smart_turn", "easyturn", "vap")


@dataclass(frozen=True)
class TurnPredictionRow:
    id: str
    probs: dict[str, float]
    timestamp: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnPredictionRow":
        if not isinstance(data, dict):
            raise ValueError("turn prediction row must be a JSON object")
        row_id = data.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("turn prediction row requires non-empty string id")

        if "probs" in data:
            probs = _parse_probs(data["probs"])
        else:
            probs = _parse_label_confidence(data)

        timestamp = data.get("timestamp")
        if timestamp is not None:
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise ValueError("timestamp must be numeric when present")
            timestamp = float(timestamp)

        return cls(id=row_id, probs=probs, timestamp=timestamp)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"id": self.id, "probs": self.probs}
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        return payload


class TurnPredictionManifestAdapter:
    """Serve turn predictions from an external model output manifest."""

    def __init__(self, rows: list[TurnPredictionRow]) -> None:
        if not rows:
            raise ValueError("turn prediction manifest must contain at least one row")
        predictions: dict[str, TurnPredictionRow] = {}
        for row in rows:
            if row.id in predictions:
                raise ValueError(f"duplicate prediction id: {row.id}")
            predictions[row.id] = row
        self._predictions = predictions

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TurnPredictionManifestAdapter":
        rows: list[TurnPredictionRow] = []
        for line_number, item in iter_jsonl(path):
            try:
                rows.append(TurnPredictionRow.from_dict(item))
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
        return cls(rows)

    def predict(self, record: TurnManifestRecord) -> TurnPrediction:
        row = self._predictions.get(record.id)
        if row is None:
            raise KeyError(f"missing prediction for record id: {record.id}")
        return TurnPrediction(
            probs=dict(row.probs),
            timestamp=record.end if row.timestamp is None else row.timestamp,
        )


def load_turn_prediction_jsonl(path: str | Path) -> TurnPredictionManifestAdapter:
    return TurnPredictionManifestAdapter.from_jsonl(path)


def export_turn_predictions_jsonl(
    records: list[TurnManifestRecord],
    predictor: Any,
    output_path: str | Path,
) -> list[TurnPredictionRow]:
    """Run a turn predictor and write Stable-ASR prediction JSONL rows."""

    rows: list[TurnPredictionRow] = []
    for record in records:
        prediction = predictor.predict(record)
        rows.append(
            TurnPredictionRow(
                id=record.id,
                probs=dict(prediction.probs),
                timestamp=prediction.timestamp,
            )
        )
    write_jsonl(output_path, [row.to_dict() for row in rows])
    return rows


def convert_turn_prediction_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    schema: str,
) -> int:
    if schema not in PREDICTION_SCHEMAS:
        raise ValueError(f"unknown prediction schema {schema!r}; expected one of {PREDICTION_SCHEMAS}")

    rows: list[TurnPredictionRow] = []
    for line_number, item in iter_jsonl(input_path):
        try:
            rows.append(convert_turn_prediction_row(item, schema=schema))
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
    write_jsonl(output_path, [row.to_dict() for row in rows])
    return len(rows)


def convert_turn_prediction_row(data: dict[str, Any], *, schema: str) -> TurnPredictionRow:
    if schema == "generic":
        return TurnPredictionRow.from_dict(data)
    if schema == "smart_turn":
        return _convert_smart_turn_row(data)
    if schema == "easyturn":
        return _convert_easyturn_prediction_row(data)
    if schema == "vap":
        return _convert_vap_prediction_row(data)
    raise ValueError(f"unknown prediction schema {schema!r}; expected one of {PREDICTION_SCHEMAS}")


def _convert_smart_turn_row(data: dict[str, Any]) -> TurnPredictionRow:
    row_id = _row_id(data)
    complete = _pick_number(
        data,
        "complete_probability",
        "turn_complete_probability",
        "prob_complete",
        "p_complete",
        "score",
    )
    if complete is None:
        label = data.get("label", data.get("prediction"))
        if label is not None:
            normalized_label = _normalize_label(label)
            complete = 1.0 if normalized_label == "complete" else 0.0
        else:
            raise ValueError("smart_turn row requires a completion probability or label")
    complete = _clamp_probability(complete, "complete_probability")
    return TurnPredictionRow(
        id=row_id,
        probs=_normalize_probs(
            {
                "complete": complete,
                "incomplete": 1.0 - complete,
                "backchannel": 0.0,
                "wait": 0.0,
            }
        ),
        timestamp=_timestamp(data),
    )


def _convert_easyturn_prediction_row(data: dict[str, Any]) -> TurnPredictionRow:
    row_id = _row_id(data)
    probs = data.get("probs", data.get("probabilities"))
    if isinstance(probs, dict):
        return TurnPredictionRow(id=row_id, probs=_parse_probs(_alias_prob_keys(probs)), timestamp=_timestamp(data))

    direct_probs = {
        "complete": _pick_number(data, "complete", "complete_probability", "prob_complete", "p_complete"),
        "incomplete": _pick_number(data, "incomplete", "incomplete_probability", "prob_incomplete", "p_incomplete"),
        "backchannel": _pick_number(data, "backchannel", "backchannel_probability", "prob_backchannel", "p_backchannel"),
        "wait": _pick_number(data, "wait", "wait_probability", "prob_wait", "p_wait"),
    }
    if any(value is not None for value in direct_probs.values()):
        return TurnPredictionRow(
            id=row_id,
            probs=_parse_probs({label: value or 0.0 for label, value in direct_probs.items()}),
            timestamp=_timestamp(data),
        )

    label = data.get("label", data.get("prediction", data.get("state", data.get("turn_state"))))
    if label is None:
        raise ValueError("easyturn row requires probs, class probabilities, or a label")
    confidence = data.get("confidence", data.get("score", 1.0))
    return TurnPredictionRow.from_dict(
        {
            "id": row_id,
            "label": _normalize_label(label),
            "confidence": confidence,
            "timestamp": _timestamp(data),
        }
    )


def _convert_vap_prediction_row(data: dict[str, Any]) -> TurnPredictionRow:
    """Map VAP-style future activity scores to Stable-ASR turn labels.

    VAP systems usually expose continuous future voice-activity estimates rather
    than discrete turn labels. Stable-ASR treats high future user activity as
    `incomplete` and high future assistant/system activity as `complete`.
    Optional backchannel/wait scores can override those two-way dynamics when an
    exporter provides them.
    """

    row_id = _row_id(data)
    probs = data.get("probs", data.get("probabilities"))
    if isinstance(probs, dict) and any(str(label) in TURN_LABELS for label in probs):
        return TurnPredictionRow(id=row_id, probs=_parse_probs(_alias_prob_keys(probs)), timestamp=_timestamp(data))
    score_source = dict(data)
    if isinstance(probs, dict):
        score_source.update(probs)

    user_future = _pick_probability(
        score_source,
        "user_future_activity",
        "future_user_activity",
        "future_user_va",
        "p_user_future",
        "p_future_user",
        "p_user_speech_future",
        "user_future",
    )
    assistant_future = _pick_probability(
        score_source,
        "assistant_future_activity",
        "system_future_activity",
        "future_assistant_activity",
        "future_system_activity",
        "future_assistant_va",
        "future_system_va",
        "p_assistant_future",
        "p_system_future",
        "p_future_assistant",
        "p_future_system",
        "assistant_future",
        "system_future",
    )
    complete = _pick_probability(score_source, "complete", "complete_probability", "prob_complete", "p_complete")
    incomplete = _pick_probability(score_source, "incomplete", "incomplete_probability", "prob_incomplete", "p_incomplete")
    backchannel = _pick_probability(
        score_source,
        "backchannel",
        "backchannel_probability",
        "prob_backchannel",
        "p_backchannel",
        "listener_backchannel_probability",
        "p_listener_backchannel",
    )
    wait = _pick_probability(score_source, "wait", "wait_probability", "prob_wait", "p_wait", "hold_probability", "p_hold")

    next_speaker = data.get("next_speaker", data.get("predicted_next_speaker"))
    if complete is None and _is_assistant_speaker(next_speaker):
        complete = 1.0
    if incomplete is None and _is_user_speaker(next_speaker):
        incomplete = 1.0

    if complete is None and assistant_future is not None:
        complete = assistant_future
    if incomplete is None and user_future is not None:
        incomplete = user_future
    if complete is None and user_future is not None:
        complete = 1.0 - user_future
    if incomplete is None and assistant_future is not None:
        incomplete = 1.0 - assistant_future

    scores = {
        "complete": complete or 0.0,
        "incomplete": incomplete or 0.0,
        "backchannel": backchannel or 0.0,
        "wait": wait or 0.0,
    }
    if sum(scores.values()) > 0.0:
        return TurnPredictionRow(id=row_id, probs=_normalize_probs(scores), timestamp=_timestamp(data))

    label = data.get("label", data.get("prediction", data.get("state", data.get("turn_state"))))
    if label is None:
        raise ValueError("vap row requires future activity scores, next_speaker, turn probabilities, or a label")
    confidence = data.get("confidence", data.get("score", 1.0))
    return TurnPredictionRow.from_dict(
        {
            "id": row_id,
            "label": _normalize_label(label),
            "confidence": confidence,
            "timestamp": _timestamp(data),
        }
    )


def _parse_probs(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError("probs must be a non-empty object")
    probs = {label: 0.0 for label in TURN_LABELS}
    for label, raw_score in value.items():
        if label not in TURN_LABELS:
            raise ValueError(f"unknown turn label in probs: {label}")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError(f"probability for {label} must be numeric")
        score = float(raw_score)
        if score < 0.0:
            raise ValueError(f"probability for {label} must be non-negative")
        probs[label] = score
    return _normalize_probs(probs)


def _parse_label_confidence(data: dict[str, Any]) -> dict[str, float]:
    label = data.get("label")
    if label not in TURN_LABELS:
        raise ValueError(f"label must be one of {sorted(TURN_LABELS)} when probs is absent")
    confidence = data.get("confidence", 1.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    floor = (1.0 - confidence) / (len(TURN_LABELS) - 1)
    probs = {candidate: floor for candidate in TURN_LABELS}
    probs[str(label)] = confidence
    return _normalize_probs(probs)


def _normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    if total <= 0.0:
        raise ValueError("probabilities must sum to a positive value")
    return {label: value / total for label, value in probs.items()}


def _row_id(data: dict[str, Any]) -> str:
    value = data.get("id", data.get("record_id", data.get("utterance_id", data.get("sample_id"))))
    if not isinstance(value, str) or not value:
        raise ValueError("prediction row requires id, record_id, utterance_id, or sample_id")
    return value


def _timestamp(data: dict[str, Any]) -> float | None:
    value = data.get("timestamp", data.get("time", data.get("end_time")))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timestamp must be numeric when present")
    return float(value)


def _pick_number(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        return float(value)
    return None


def _pick_probability(data: dict[str, Any], *keys: str) -> float | None:
    value = _pick_number(data, *keys)
    if value is None:
        return None
    return _clamp_probability(value, keys[0] if keys else "probability")


def _clamp_probability(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _alias_prob_keys(probs: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "complete_probability": "complete",
        "prob_complete": "complete",
        "p_complete": "complete",
        "incomplete_probability": "incomplete",
        "prob_incomplete": "incomplete",
        "p_incomplete": "incomplete",
        "backchannel_probability": "backchannel",
        "prob_backchannel": "backchannel",
        "p_backchannel": "backchannel",
        "wait_probability": "wait",
        "prob_wait": "wait",
        "p_wait": "wait",
        "hold_probability": "wait",
        "p_hold": "wait",
    }
    return {aliases.get(str(label), str(label)): score for label, score in probs.items()}


def _is_user_speaker(value: Any) -> bool:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text in {"user", "speaker", "current_speaker", "human"}


def _is_assistant_speaker(value: Any) -> bool:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text in {"assistant", "system", "agent", "bot", "other_speaker"}


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "complete": "complete",
        "completed": "complete",
        "finished": "complete",
        "finish": "complete",
        "turn_complete": "complete",
        "take_turn": "complete",
        "incomplete": "incomplete",
        "unfinished": "incomplete",
        "partial": "incomplete",
        "not_complete": "incomplete",
        "backchannel": "backchannel",
        "back_channel": "backchannel",
        "bc": "backchannel",
        "wait": "wait",
        "hold": "wait",
    }
    if text not in aliases:
        raise ValueError(f"unknown prediction label: {value!r}")
    return aliases[text]
