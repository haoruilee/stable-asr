"""Leakage audits for train/dev/test turn manifest splits."""

from __future__ import annotations

from dataclasses import dataclass

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.data.split import SPLIT_NAMES


DEFAULT_LEAKAGE_FIELDS = ("id", "audio", "metadata.asr_record_id", "metadata.conversation_id")


@dataclass(frozen=True)
class SplitLeak:
    field: str
    value: str
    splits: tuple[str, ...]
    record_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "value": self.value,
            "splits": list(self.splits),
            "record_ids": list(self.record_ids),
        }


@dataclass(frozen=True)
class TurnSplitAuditReport:
    records_by_split: dict[str, int]
    leakage_fields: tuple[str, ...]
    leaks: list[SplitLeak]
    missing_splits: list[str]

    @property
    def ok(self) -> bool:
        return not self.leaks and not self.missing_splits

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "records_by_split": self.records_by_split,
            "leakage_fields": list(self.leakage_fields),
            "leaks": [leak.to_dict() for leak in self.leaks],
            "missing_splits": self.missing_splits,
        }

    def to_text(self) -> str:
        lines = [
            f"turn_split_audit: {'OK' if self.ok else 'FAILED'}",
            "records_by_split: "
            + ", ".join(f"{name}={count}" for name, count in self.records_by_split.items()),
            "leakage_fields: " + ", ".join(self.leakage_fields),
            f"leaks: {len(self.leaks)}",
        ]
        if self.missing_splits:
            lines.append("missing_splits: " + ", ".join(self.missing_splits))
        if self.leaks:
            lines.append("leak_examples:")
            for leak in self.leaks[:25]:
                lines.append(
                    f"- {leak.field}={leak.value!r} splits={','.join(leak.splits)} "
                    f"records={','.join(leak.record_ids[:6])}"
                )
            if len(self.leaks) > 25:
                lines.append(f"- ... {len(self.leaks) - 25} more leak(s)")
        return "\n".join(lines)


def audit_turn_splits(
    splits: dict[str, list[TurnManifestRecord]],
    *,
    leakage_fields: tuple[str, ...] = DEFAULT_LEAKAGE_FIELDS,
) -> TurnSplitAuditReport:
    missing_splits = [name for name in SPLIT_NAMES if name not in splits]
    records_by_split = {name: len(splits.get(name, [])) for name in SPLIT_NAMES}
    leaks: list[SplitLeak] = []

    for field in leakage_fields:
        values: dict[str, dict[str, list[str]]] = {}
        for split_name, records in splits.items():
            for record in records:
                value = _field_value(record, field)
                if not value:
                    continue
                split_values = values.setdefault(value, {})
                split_values.setdefault(split_name, []).append(record.id)
        for value, split_values in values.items():
            split_names = tuple(sorted(split_values))
            if len(split_names) <= 1:
                continue
            record_ids = tuple(
                record_id
                for split_name in split_names
                for record_id in sorted(split_values[split_name])
            )
            leaks.append(SplitLeak(field=field, value=value, splits=split_names, record_ids=record_ids))

    leaks.sort(key=lambda leak: (leak.field, leak.value))
    return TurnSplitAuditReport(
        records_by_split=records_by_split,
        leakage_fields=leakage_fields,
        leaks=leaks,
        missing_splits=missing_splits,
    )


def _field_value(record: TurnManifestRecord, field: str) -> str:
    if field.startswith("metadata."):
        value = record.metadata.get(field.removeprefix("metadata."))
    else:
        value = getattr(record, field, None)
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return repr(value)
