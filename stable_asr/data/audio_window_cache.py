"""Materialized audio-window cache and benchmarks.

This module is the speech counterpart of stable-worldmodel's Lance-first data
layer: turn manifests remain the interchange format, while training windows can
be materialized into columnar rows for fast random access.
"""

from __future__ import annotations

import random
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.audio import load_wav_mono
from stable_asr.data.manifest import TurnManifestRecord


WINDOW_CACHE_FORMATS = ("source_wav", "parquet", "lance")


@dataclass(frozen=True)
class AudioWindowBenchmarkRow:
    format: str
    records: int
    write_seconds: float
    sample_count: int
    sample_seconds: float
    samples_per_second: float
    speedup_vs_source_wav: float
    size_bytes: int
    output_path: str
    sample_strategy: str

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "records": self.records,
            "write_seconds": self.write_seconds,
            "sample_count": self.sample_count,
            "sample_seconds": self.sample_seconds,
            "samples_per_second": self.samples_per_second,
            "speedup_vs_source_wav": self.speedup_vs_source_wav,
            "size_bytes": self.size_bytes,
            "output_path": self.output_path,
            "sample_strategy": self.sample_strategy,
        }


def benchmark_audio_window_formats(
    records: list[TurnManifestRecord],
    *,
    output_dir: str | Path,
    formats: list[str] | tuple[str, ...] = WINDOW_CACHE_FORMATS,
    sample_count: int = 1000,
    seed: int = 0,
    max_records: int | None = None,
    audio_root: str | Path | None = None,
) -> list[AudioWindowBenchmarkRow]:
    """Benchmark source WAV reads against materialized Parquet/Lance rows."""

    if not records:
        raise ValueError("records must not be empty")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    requested = list(formats)
    unknown = sorted(set(requested) - set(WINDOW_CACHE_FORMATS))
    if unknown:
        raise ValueError(f"unknown audio window cache format(s): {', '.join(unknown)}")

    records = list(records[:max_records]) if max_records else list(records)
    if not records:
        raise ValueError("no records selected for benchmark")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = _sample_indices(len(records), sample_count=sample_count, seed=seed)

    rows: list[AudioWindowBenchmarkRow] = []
    baseline_sps = 0.0
    for name in requested:
        if name == "source_wav":
            row = _benchmark_source_wav(records, indices, audio_root=audio_root)
            baseline_sps = row.samples_per_second
            rows.append(row)
            continue
        if name == "parquet":
            row = _benchmark_materialized_format(
                name,
                records,
                indices,
                output_path=output_dir / "audio_windows.parquet",
                audio_root=audio_root,
                source_samples_per_second=baseline_sps,
            )
            rows.append(row)
            continue
        if name == "lance":
            row = _benchmark_materialized_format(
                name,
                records,
                indices,
                output_path=output_dir / "audio_windows.lance",
                audio_root=audio_root,
                source_samples_per_second=baseline_sps,
            )
            rows.append(row)
            continue
    if baseline_sps <= 0.0:
        baseline = next((row for row in rows if row.format == "source_wav"), None)
        baseline_sps = baseline.samples_per_second if baseline else 0.0
    if baseline_sps > 0.0:
        rows = [
            AudioWindowBenchmarkRow(
                format=row.format,
                records=row.records,
                write_seconds=row.write_seconds,
                sample_count=row.sample_count,
                sample_seconds=row.sample_seconds,
                samples_per_second=row.samples_per_second,
                speedup_vs_source_wav=row.samples_per_second / baseline_sps,
                size_bytes=row.size_bytes,
                output_path=row.output_path,
                sample_strategy=row.sample_strategy,
            )
            for row in rows
        ]
    return rows


def materialize_audio_windows(
    records: list[TurnManifestRecord],
    output_path: str | Path,
    *,
    format: str,
    audio_root: str | Path | None = None,
) -> int:
    if format not in {"parquet", "lance"}:
        raise ValueError("audio window materialization supports only parquet or lance")
    rows = [_record_to_window_row(record, audio_root=audio_root) for record in records]
    if format == "parquet":
        _write_window_parquet(output_path, rows)
    else:
        _write_window_lance(output_path, rows)
    return len(rows)


def _benchmark_source_wav(
    records: list[TurnManifestRecord],
    indices: list[int],
    *,
    audio_root: str | Path | None,
) -> AudioWindowBenchmarkRow:
    start = time.perf_counter()
    total_samples = 0
    for index in indices:
        samples = _record_to_window_samples(records[index], audio_root=audio_root)
        total_samples += len(samples)
    sample_seconds = time.perf_counter() - start
    if total_samples <= 0:
        raise RuntimeError("source_wav benchmark decoded no samples")
    return AudioWindowBenchmarkRow(
        format="source_wav",
        records=len(records),
        write_seconds=0.0,
        sample_count=len(indices),
        sample_seconds=sample_seconds,
        samples_per_second=len(indices) / sample_seconds if sample_seconds > 0 else float("inf"),
        speedup_vs_source_wav=1.0,
        size_bytes=_source_audio_size_bytes(records, audio_root=audio_root),
        output_path="source_audio",
        sample_strategy="open_wav_per_sample",
    )


def _benchmark_materialized_format(
    name: str,
    records: list[TurnManifestRecord],
    indices: list[int],
    *,
    output_path: Path,
    audio_root: str | Path | None,
    source_samples_per_second: float,
) -> AudioWindowBenchmarkRow:
    write_start = time.perf_counter()
    materialize_audio_windows(records, output_path, format=name, audio_root=audio_root)
    write_seconds = time.perf_counter() - write_start

    sample_start = time.perf_counter()
    if name == "parquet":
        total_samples = _sample_parquet_windows(output_path, indices)
        sample_strategy = "parquet_read_columns_select"
    else:
        total_samples = _sample_lance_windows(output_path, indices)
        sample_strategy = "lance_take"
    sample_seconds = time.perf_counter() - sample_start
    if total_samples <= 0:
        raise RuntimeError(f"{name} benchmark decoded no samples")
    samples_per_second = len(indices) / sample_seconds if sample_seconds > 0 else float("inf")
    return AudioWindowBenchmarkRow(
        format=name,
        records=len(records),
        write_seconds=write_seconds,
        sample_count=len(indices),
        sample_seconds=sample_seconds,
        samples_per_second=samples_per_second,
        speedup_vs_source_wav=(samples_per_second / source_samples_per_second)
        if source_samples_per_second > 0
        else 0.0,
        size_bytes=_path_size_bytes(output_path),
        output_path=str(output_path),
        sample_strategy=sample_strategy,
    )


def _record_to_window_row(
    record: TurnManifestRecord,
    *,
    audio_root: str | Path | None,
) -> dict[str, Any]:
    samples = _record_to_window_samples(record, audio_root=audio_root)
    return {
        "id": record.id,
        "audio": record.audio,
        "sample_rate": record.sample_rate,
        "start": record.start,
        "end": record.end,
        "turn_label": record.turn_label,
        "action_label": record.action_label,
        "language": record.language,
        "scenario": record.scenario,
        "num_samples": len(samples),
        "pcm_f32": _f32_bytes(samples),
    }


def _record_to_window_samples(
    record: TurnManifestRecord,
    *,
    audio_root: str | Path | None,
) -> list[float]:
    path = Path(record.audio)
    if not path.is_absolute() and audio_root is not None:
        path = Path(audio_root) / path
    samples, sample_rate = load_wav_mono(path)
    if sample_rate != record.sample_rate:
        raise ValueError(f"sample rate mismatch for {path}: audio={sample_rate}, manifest={record.sample_rate}")
    start = max(0, int(round(record.start * sample_rate)))
    end = min(len(samples), int(round(record.end * sample_rate)))
    if end <= start:
        raise ValueError(f"empty audio window for {record.id}: start={record.start} end={record.end}")
    return samples[start:end]


def _write_window_parquet(path: str | Path, rows: list[dict[str, Any]]) -> None:
    pa, pq = _require_pyarrow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(_columns(rows)), path)


def _write_window_lance(path: str | Path, rows: list[dict[str, Any]]) -> None:
    pa, lance = _require_pyarrow_lance()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(
        pa.table(_columns(rows)),
        str(path),
        mode="overwrite",
        commit_message="stable-asr audio window cache write",
    )


def _sample_parquet_windows(path: str | Path, indices: list[int]) -> int:
    _, pq = _require_pyarrow()
    table = pq.read_table(path, columns=["pcm_f32"])
    payloads = table.column("pcm_f32").to_pylist()
    return sum(len(_f32_array(payloads[index])) for index in indices)


def _sample_lance_windows(path: str | Path, indices: list[int]) -> int:
    lance = _require_lance()
    table = lance.dataset(str(path)).take(indices, columns=["pcm_f32"])
    payloads = table.column("pcm_f32").to_pylist()
    return sum(len(_f32_array(payload)) for payload in payloads)


def _columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not rows:
        raise ValueError("rows must not be empty")
    return {key: [row.get(key) for row in rows] for key in rows[0]}


def _sample_indices(records: int, *, sample_count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(records) for _ in range(sample_count)]


def _f32_bytes(samples: list[float]) -> bytes:
    return array("f", samples).tobytes()


def _f32_array(payload: bytes) -> array:
    values = array("f")
    values.frombytes(payload)
    return values


def _source_audio_size_bytes(records: list[TurnManifestRecord], *, audio_root: str | Path | None) -> int:
    paths = set()
    for record in records:
        path = Path(record.audio)
        if not path.is_absolute() and audio_root is not None:
            path = Path(audio_root) / path
        paths.add(path)
    return sum(path.stat().st_size for path in paths if path.exists() and path.is_file())


def _path_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return path.stat().st_size


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Audio-window Parquet support requires pyarrow. Install stable-asr[data].") from exc
    return pa, pq


def _require_lance():
    try:
        import lance
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Audio-window Lance support requires pylance. Install stable-asr[lance].") from exc
    if not hasattr(lance, "dataset") or not hasattr(lance, "write_dataset"):
        raise RuntimeError("Audio-window Lance support requires the pylance package that imports as 'lance'.")
    return lance


def _require_pyarrow_lance():
    try:
        import pyarrow as pa
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Audio-window Lance support requires pyarrow. Install stable-asr[lance].") from exc
    return pa, _require_lance()

