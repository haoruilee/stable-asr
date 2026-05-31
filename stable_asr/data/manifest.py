"""Stable-ASR turn manifest schema and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stable_asr.data.formats.jsonl import iter_jsonl
from stable_asr.turn.labels import ACTION_LABELS, TURN_LABELS


class ManifestError(ValueError):
    """Raised when a manifest record cannot be parsed."""


@dataclass(frozen=True)
class TurnManifestRecord:
    """One turn-taking training or evaluation window.

    The schema is intentionally small in v0 so external datasets can be
    converted without committing to a heavy storage layer.
    """

    id: str
    audio: str
    sample_rate: int
    start: float
    end: float
    turn_label: str
    action_label: str
    assistant_speaking: bool
    overlap: bool
    language: str
    source: str
    text: str | None = None
    asr_text: str | None = None
    scenario: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnManifestRecord":
        if not isinstance(data, dict):
            raise ManifestError("record must be a JSON object")

        required = {
            "id",
            "audio",
            "sample_rate",
            "start",
            "end",
            "turn_label",
            "action_label",
            "assistant_speaking",
            "overlap",
            "language",
            "source",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ManifestError(f"missing required field(s): {', '.join(missing)}")

        record = cls(
            id=_require_str(data, "id"),
            audio=_require_str(data, "audio"),
            sample_rate=_require_int(data, "sample_rate"),
            start=_require_float(data, "start"),
            end=_require_float(data, "end"),
            turn_label=_require_str(data, "turn_label"),
            action_label=_require_str(data, "action_label"),
            assistant_speaking=_require_bool(data, "assistant_speaking"),
            overlap=_require_bool(data, "overlap"),
            language=_require_str(data, "language"),
            source=_require_str(data, "source"),
            text=_optional_str(data, "text"),
            asr_text=_optional_str(data, "asr_text"),
            scenario=_optional_str(data, "scenario"),
            metadata=_optional_dict(data, "metadata"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ManifestError("sample_rate must be positive")
        if self.start < 0:
            raise ManifestError("start must be non-negative")
        if self.end <= self.start:
            raise ManifestError("end must be greater than start")
        if self.turn_label not in TURN_LABELS:
            raise ManifestError(
                f"unknown turn_label {self.turn_label!r}; expected one of {sorted(TURN_LABELS)}"
            )
        if self.action_label not in ACTION_LABELS:
            raise ManifestError(
                f"unknown action_label {self.action_label!r}; expected one of {sorted(ACTION_LABELS)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestValidationReport:
    path: str
    records: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "records": self.records,
            "errors": self.errors,
            "ok": self.ok,
        }

    def to_text(self) -> str:
        if self.ok:
            return f"OK: {self.path} contains {self.records} valid record(s)."
        lines = [f"ERROR: {self.path} has {len(self.errors)} validation error(s)."]
        lines.extend(f"- {error}" for error in self.errors)
        return "\n".join(lines)


def load_manifest(path: str | Path) -> list[TurnManifestRecord]:
    """Load a JSONL manifest and raise on the first invalid record."""

    records: list[TurnManifestRecord] = []
    for line_number, item in iter_jsonl(path):
        try:
            records.append(TurnManifestRecord.from_dict(item))
        except ManifestError as exc:
            raise ManifestError(f"line {line_number}: {exc}") from exc
    return records


def validate_manifest(path: str | Path) -> ManifestValidationReport:
    """Validate a manifest without raising for record-level failures."""

    errors: list[str] = []
    records = 0
    path = Path(path)

    try:
        iterator = iter_jsonl(path)
        for line_number, item in iterator:
            try:
                TurnManifestRecord.from_dict(item)
                records += 1
            except ManifestError as exc:
                errors.append(f"line {line_number}: {exc}")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))

    return ManifestValidationReport(path=str(path), records=records, errors=errors)


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{key} must be a non-empty string")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestError(f"{key} must be a string when present")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{key} must be an integer")
    return value


def _require_float(data: dict[str, Any], key: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{key} must be a number")
    return float(value)


def _require_bool(data: dict[str, Any], key: str) -> bool:
    value = data[key]
    if not isinstance(value, bool):
        raise ManifestError(f"{key} must be a boolean")
    return value


def _optional_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestError(f"{key} must be an object when present")
    return dict(value)

