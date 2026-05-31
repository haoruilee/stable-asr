"""Validation utilities for turn prediction manifests."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.models.adapters.turn_prediction import TurnPredictionRow


@dataclass(frozen=True)
class TurnPredictionValidationReport:
    dataset_records: int
    prediction_rows: int
    valid_prediction_rows: int
    invalid_rows: list[str]
    duplicate_dataset_ids: list[str]
    duplicate_prediction_ids: list[str]
    missing_ids: list[str]
    extra_ids: list[str]
    allow_extra: bool = False
    dataset_path: str | None = None
    predictions_path: str | None = None

    @property
    def ok(self) -> bool:
        return (
            not self.invalid_rows
            and not self.duplicate_dataset_ids
            and not self.duplicate_prediction_ids
            and not self.missing_ids
            and (self.allow_extra or not self.extra_ids)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "dataset_path": self.dataset_path,
            "predictions_path": self.predictions_path,
            "dataset_records": self.dataset_records,
            "prediction_rows": self.prediction_rows,
            "valid_prediction_rows": self.valid_prediction_rows,
            "invalid_rows": self.invalid_rows,
            "duplicate_dataset_ids": self.duplicate_dataset_ids,
            "duplicate_prediction_ids": self.duplicate_prediction_ids,
            "missing_ids": self.missing_ids,
            "extra_ids": self.extra_ids,
            "allow_extra": self.allow_extra,
        }

    def to_text(self) -> str:
        status = "OK" if self.ok else "ERROR"
        lines = [
            f"{status}: turn prediction manifest validation",
            f"dataset_records: {self.dataset_records}",
            f"prediction_rows: {self.prediction_rows}",
            f"valid_prediction_rows: {self.valid_prediction_rows}",
        ]
        if self.dataset_path:
            lines.append(f"dataset: {self.dataset_path}")
        if self.predictions_path:
            lines.append(f"predictions: {self.predictions_path}")
        lines.extend(_issue_lines("invalid_rows", self.invalid_rows))
        lines.extend(_issue_lines("duplicate_dataset_ids", self.duplicate_dataset_ids))
        lines.extend(_issue_lines("duplicate_prediction_ids", self.duplicate_prediction_ids))
        lines.extend(_issue_lines("missing_ids", self.missing_ids))
        if self.extra_ids and self.allow_extra:
            lines.extend(_issue_lines("extra_ids_allowed", self.extra_ids))
        else:
            lines.extend(_issue_lines("extra_ids", self.extra_ids))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            "# Stable-ASR Turn Prediction Validation",
            "",
            f"- status: {'OK' if self.ok else 'ERROR'}",
            f"- dataset_records: {self.dataset_records}",
            f"- prediction_rows: {self.prediction_rows}",
            f"- valid_prediction_rows: {self.valid_prediction_rows}",
            f"- allow_extra: {self.allow_extra}",
        ]
        if self.dataset_path:
            lines.append(f"- dataset: `{self.dataset_path}`")
        if self.predictions_path:
            lines.append(f"- predictions: `{self.predictions_path}`")
        lines.append("")
        for title, values in [
            ("Invalid Rows", self.invalid_rows),
            ("Duplicate Dataset IDs", self.duplicate_dataset_ids),
            ("Duplicate Prediction IDs", self.duplicate_prediction_ids),
            ("Missing IDs", self.missing_ids),
            ("Extra IDs", self.extra_ids),
        ]:
            lines.append(f"## {title}")
            if values:
                lines.extend(f"- `{value}`" for value in values)
            else:
                lines.append("- none")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def validate_turn_prediction_jsonl(
    records: list[TurnManifestRecord],
    predictions_path: str | Path,
    *,
    allow_extra: bool = False,
    dataset_path: str | Path | None = None,
) -> TurnPredictionValidationReport:
    dataset_ids = [record.id for record in records]
    duplicate_dataset_ids = _duplicates(dataset_ids)
    dataset_id_set = set(dataset_ids)

    prediction_rows = 0
    valid_rows: list[TurnPredictionRow] = []
    invalid_rows: list[str] = []
    predictions_path = Path(predictions_path)

    try:
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                prediction_rows += 1
                try:
                    item = json.loads(stripped)
                    valid_rows.append(TurnPredictionRow.from_dict(item))
                except (json.JSONDecodeError, ValueError) as exc:
                    invalid_rows.append(f"line {line_number}: {exc}")
    except OSError as exc:
        invalid_rows.append(str(exc))

    prediction_ids = [row.id for row in valid_rows]
    prediction_id_set = set(prediction_ids)

    return TurnPredictionValidationReport(
        dataset_records=len(records),
        prediction_rows=prediction_rows,
        valid_prediction_rows=len(valid_rows),
        invalid_rows=invalid_rows,
        duplicate_dataset_ids=duplicate_dataset_ids,
        duplicate_prediction_ids=_duplicates(prediction_ids),
        missing_ids=sorted(dataset_id_set - prediction_id_set),
        extra_ids=sorted(prediction_id_set - dataset_id_set),
        allow_extra=allow_extra,
        dataset_path=str(dataset_path) if dataset_path is not None else None,
        predictions_path=str(predictions_path),
    )


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _issue_lines(name: str, values: list[str]) -> list[str]:
    if not values:
        return []
    return [f"{name}: {len(values)}", *(f"  - {value}" for value in values)]
