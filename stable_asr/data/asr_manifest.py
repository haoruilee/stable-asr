"""Stable-ASR ASR corpus manifest schema and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stable_asr.data.formats.jsonl import iter_jsonl, write_jsonl


class ASRManifestError(ValueError):
    """Raised when an ASR manifest record cannot be parsed."""


@dataclass(frozen=True)
class ASRManifestRecord:
    """One utterance-level ASR corpus record.

    This schema is intentionally separate from the turn/action window manifest:
    public ASR corpora usually provide audio paths and reference transcripts,
    but not endpointing or full-duplex action labels.
    """

    id: str
    audio: str
    sample_rate: int
    text: str
    language: str
    source: str
    duration: float | None = None
    split: str | None = None
    speaker_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ASRManifestRecord":
        if not isinstance(data, dict):
            raise ASRManifestError("record must be a JSON object")

        required = {"id", "audio", "sample_rate", "text", "language", "source"}
        missing = sorted(required - data.keys())
        if missing:
            raise ASRManifestError(f"missing required field(s): {', '.join(missing)}")

        record = cls(
            id=_require_str(data, "id"),
            audio=_require_str(data, "audio"),
            sample_rate=_require_int(data, "sample_rate"),
            text=_require_str(data, "text"),
            language=_require_str(data, "language"),
            source=_require_str(data, "source"),
            duration=_optional_float(data, "duration"),
            split=_optional_str(data, "split"),
            speaker_id=_optional_str(data, "speaker_id"),
            metadata=_optional_dict(data, "metadata"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ASRManifestError("sample_rate must be positive")
        if self.duration is not None and self.duration <= 0:
            raise ASRManifestError("duration must be positive when present")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ASRManifestValidationReport:
    path: str
    records: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "records": self.records,
            "errors": self.errors,
            "ok": self.ok,
        }

    def to_text(self) -> str:
        if self.ok:
            return f"OK: {self.path} contains {self.records} valid ASR record(s)."
        lines = [f"ERROR: {self.path} has {len(self.errors)} validation error(s)."]
        lines.extend(f"- {error}" for error in self.errors)
        return "\n".join(lines)


def load_asr_manifest(path: str | Path) -> list[ASRManifestRecord]:
    records: list[ASRManifestRecord] = []
    for line_number, item in iter_jsonl(path):
        try:
            records.append(ASRManifestRecord.from_dict(item))
        except ASRManifestError as exc:
            raise ASRManifestError(f"line {line_number}: {exc}") from exc
    return records


def write_asr_manifest(path: str | Path, records: list[ASRManifestRecord]) -> None:
    write_jsonl(path, [record.to_dict() for record in records])


def validate_asr_manifest(path: str | Path) -> ASRManifestValidationReport:
    errors: list[str] = []
    records = 0
    path = Path(path)

    try:
        for line_number, item in iter_jsonl(path):
            try:
                ASRManifestRecord.from_dict(item)
                records += 1
            except ASRManifestError as exc:
                errors.append(f"line {line_number}: {exc}")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))

    return ASRManifestValidationReport(path=str(path), records=records, errors=errors)


def summarize_asr_records(records: list[ASRManifestRecord]) -> dict[str, object]:
    durations = [record.duration for record in records if record.duration is not None]
    return {
        "records": len(records),
        "total_duration_sec": round(sum(durations), 6),
        "has_duration_records": len(durations),
        "sample_rates": _counts(str(record.sample_rate) for record in records),
        "languages": _counts(record.language for record in records),
        "sources": _counts(record.source for record in records),
        "splits": _counts(record.split for record in records if record.split is not None),
        "speakers": len({record.speaker_id for record in records if record.speaker_id}),
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ASRManifestError(f"{key} must be a non-empty string")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ASRManifestError(f"{key} must be a string when present")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data[key]
    if isinstance(value, bool):
        raise ASRManifestError(f"{key} must be an integer")
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError as exc:
            raise ASRManifestError(f"{key} must be an integer") from exc
    if not isinstance(value, int):
        raise ASRManifestError(f"{key} must be an integer")
    return value


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ASRManifestError(f"{key} must be a number when present")
    try:
        return float(value)
    except ValueError as exc:
        raise ASRManifestError(f"{key} must be a number when present") from exc


def _optional_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ASRManifestError(f"{key} must be an object when present")
    return dict(value)
