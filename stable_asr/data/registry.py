"""Data format registry for turn manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from stable_asr.data.formats.jsonl import write_jsonl
from stable_asr.data.formats.lance import load_lance, write_lance
from stable_asr.data.formats.parquet import load_parquet, write_parquet
from stable_asr.data.manifest import TurnManifestRecord, load_manifest


class TurnFormat(Protocol):
    name: str
    suffixes: tuple[str, ...]

    def load(self, path: str | Path) -> list[TurnManifestRecord]:
        ...

    def write(self, path: str | Path, records: list[TurnManifestRecord]) -> None:
        ...


@dataclass(frozen=True)
class JsonlTurnFormat:
    name: str = "jsonl"
    suffixes: tuple[str, ...] = (".jsonl",)

    def load(self, path: str | Path) -> list[TurnManifestRecord]:
        return load_manifest(path)

    def write(self, path: str | Path, records: list[TurnManifestRecord]) -> None:
        write_jsonl(path, [record.to_dict() for record in records])


@dataclass(frozen=True)
class ParquetTurnFormat:
    name: str = "parquet"
    suffixes: tuple[str, ...] = (".parquet", ".pq")

    def load(self, path: str | Path) -> list[TurnManifestRecord]:
        return load_parquet(path)

    def write(self, path: str | Path, records: list[TurnManifestRecord]) -> None:
        write_parquet(path, records)


@dataclass(frozen=True)
class LanceTurnFormat:
    name: str = "lance"
    suffixes: tuple[str, ...] = (".lance",)

    def load(self, path: str | Path) -> list[TurnManifestRecord]:
        return load_lance(path)

    def write(self, path: str | Path, records: list[TurnManifestRecord]) -> None:
        write_lance(path, records)


class FormatRegistry:
    def __init__(self) -> None:
        self._formats: dict[str, TurnFormat] = {}

    def register(self, data_format: TurnFormat) -> None:
        if data_format.name in self._formats:
            raise ValueError(f"format already registered: {data_format.name}")
        self._formats[data_format.name] = data_format

    def get(self, name: str) -> TurnFormat:
        try:
            return self._formats[name]
        except KeyError as exc:
            raise ValueError(f"unknown format {name!r}; available: {self.names()}") from exc

    def names(self) -> list[str]:
        return sorted(self._formats)

    def detect(self, path: str | Path) -> TurnFormat:
        suffix = Path(path).suffix.lower()
        for data_format in self._formats.values():
            if suffix in data_format.suffixes:
                return data_format
        raise ValueError(f"could not detect data format from suffix {suffix!r}")


TURN_FORMATS = FormatRegistry()
TURN_FORMATS.register(JsonlTurnFormat())
TURN_FORMATS.register(ParquetTurnFormat())
TURN_FORMATS.register(LanceTurnFormat())


def load_turn_records(path: str | Path, *, format: str | None = None) -> list[TurnManifestRecord]:
    data_format = TURN_FORMATS.get(format) if format else TURN_FORMATS.detect(path)
    return data_format.load(path)


def write_turn_records(
    path: str | Path,
    records: list[TurnManifestRecord],
    *,
    format: str | None = None,
) -> None:
    data_format = TURN_FORMATS.get(format) if format else TURN_FORMATS.detect(path)
    data_format.write(path, records)


def convert_turn_manifest(
    source: str | Path,
    dest: str | Path,
    *,
    source_format: str | None = None,
    dest_format: str | None = None,
) -> int:
    records = load_turn_records(source, format=source_format)
    write_turn_records(dest, records, format=dest_format)
    return len(records)


def summarize_records(records: list[TurnManifestRecord]) -> dict[str, object]:
    labels: dict[str, int] = {}
    actions: dict[str, int] = {}
    scenarios: dict[str, int] = {}
    languages: dict[str, int] = {}

    for record in records:
        labels[record.turn_label] = labels.get(record.turn_label, 0) + 1
        actions[record.action_label] = actions.get(record.action_label, 0) + 1
        if record.scenario:
            scenarios[record.scenario] = scenarios.get(record.scenario, 0) + 1
        languages[record.language] = languages.get(record.language, 0) + 1

    return {
        "records": len(records),
        "turn_labels": dict(sorted(labels.items())),
        "action_labels": dict(sorted(actions.items())),
        "scenarios": dict(sorted(scenarios.items())),
        "languages": dict(sorted(languages.items())),
    }
