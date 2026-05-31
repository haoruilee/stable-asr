"""Small data-layer benchmark helpers."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from stable_asr.data.formats.lance import take_lance
from stable_asr.data.registry import load_turn_records, write_turn_records
from stable_asr.data.manifest import TurnManifestRecord


@dataclass(frozen=True)
class DataBenchmarkRow:
    format: str
    records: int
    write_seconds: float
    read_seconds: float
    size_bytes: int
    output_path: str
    sample_count: int = 0
    sample_seconds: float = 0.0
    samples_per_second: float = 0.0
    sample_strategy: str = "disabled"

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "records": self.records,
            "write_seconds": self.write_seconds,
            "read_seconds": self.read_seconds,
            "size_bytes": self.size_bytes,
            "output_path": self.output_path,
            "sample_count": self.sample_count,
            "sample_seconds": self.sample_seconds,
            "samples_per_second": self.samples_per_second,
            "sample_strategy": self.sample_strategy,
        }


def benchmark_data_formats(
    records: list[TurnManifestRecord],
    *,
    output_dir: str | Path,
    formats: list[str],
    sample_count: int = 0,
    seed: int = 0,
) -> list[DataBenchmarkRow]:
    if not records:
        raise ValueError("records must not be empty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[DataBenchmarkRow] = []
    for name in formats:
        suffix = _format_suffix(name)
        output_path = output_dir / f"turn_manifest{suffix}"

        write_start = time.perf_counter()
        write_turn_records(output_path, records, format=name)
        write_seconds = time.perf_counter() - write_start

        read_start = time.perf_counter()
        loaded = load_turn_records(output_path, format=name)
        read_seconds = time.perf_counter() - read_start
        sample_seconds = 0.0
        samples_per_second = 0.0
        sample_strategy = "disabled"
        if sample_count > 0:
            indices = _sample_indices(len(loaded), sample_count=sample_count, seed=seed)
            sample_strategy = "lance_take" if name == "lance" else "load_all_select"
            sample_start = time.perf_counter()
            sampled = _sample_records(output_path, name, indices)
            sample_seconds = time.perf_counter() - sample_start
            if len(sampled) != sample_count:
                raise RuntimeError(f"sampled {len(sampled)} records from {name}, expected {sample_count}")
            samples_per_second = sample_count / sample_seconds if sample_seconds > 0 else float("inf")

        rows.append(
            DataBenchmarkRow(
                format=name,
                records=len(loaded),
                write_seconds=write_seconds,
                read_seconds=read_seconds,
                size_bytes=_path_size_bytes(output_path),
                output_path=str(output_path),
                sample_count=sample_count,
                sample_seconds=sample_seconds,
                samples_per_second=samples_per_second,
                sample_strategy=sample_strategy,
            )
        )
    return rows


def _format_suffix(name: str) -> str:
    suffixes = {
        "jsonl": ".jsonl",
        "parquet": ".parquet",
        "lance": ".lance",
    }
    return suffixes.get(name, f".{name}")


def _path_size_bytes(path: Path) -> int:
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return path.stat().st_size


def _sample_indices(records: int, *, sample_count: int, seed: int) -> list[int]:
    if records <= 0:
        raise ValueError("records must be positive")
    rng = random.Random(seed)
    return [rng.randrange(records) for _ in range(sample_count)]


def _sample_records(path: Path, name: str, indices: list[int]) -> list[TurnManifestRecord]:
    if name == "lance":
        return take_lance(path, indices)
    records = load_turn_records(path, format=name)
    return [records[index] for index in indices]
