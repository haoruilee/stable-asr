"""Cached NanoTurn audio-feature storage.

The cache stores the pooled log-mel feature vector used by NanoTurn v0. It is
intended for repeated training/evaluation loops where re-opening and decoding
source audio dominates iteration time.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.train.features import AUDIO_FEATURE_NAMES, record_to_logmel_features
from stable_asr.turn.nanoturn import require_torch, torch


TRAIN_FEATURE_BENCHMARK_FORMATS = ("source_audio", "source_audio_file_cache", "parquet", "lance")
FEATURE_CACHE_FORMATS = ("parquet", "lance")


@dataclass(frozen=True)
class TrainFeatureBenchmarkRow:
    format: str
    records: int
    write_seconds: float
    sample_count: int
    sample_seconds: float
    samples_per_second: float
    speedup_vs_source_audio: float
    output_path: str
    size_bytes: int
    sample_strategy: str

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "records": self.records,
            "write_seconds": self.write_seconds,
            "sample_count": self.sample_count,
            "sample_seconds": self.sample_seconds,
            "samples_per_second": self.samples_per_second,
            "speedup_vs_source_audio": self.speedup_vs_source_audio,
            "output_path": self.output_path,
            "size_bytes": self.size_bytes,
            "sample_strategy": self.sample_strategy,
        }


def ensure_logmel_feature_cache(
    records: list[TurnManifestRecord],
    path: str | Path,
    *,
    format: str | None = None,
    mode: str = "auto",
    audio_root: str | Path | None = None,
) -> int:
    """Ensure a feature cache exists and return its record count."""

    path = Path(path)
    mode = mode.lower()
    if mode not in {"auto", "read", "write", "off"}:
        raise ValueError("feature cache mode must be one of: auto, read, write, off")
    if mode == "off":
        return 0
    if mode == "read":
        if not path.exists():
            raise RuntimeError(f"feature cache does not exist: {path}")
        return count_logmel_feature_cache(path, format=format)
    if mode == "auto" and path.exists():
        return count_logmel_feature_cache(path, format=format)
    return write_logmel_feature_cache(records, path, format=format, audio_root=audio_root)


def write_logmel_feature_cache(
    records: list[TurnManifestRecord],
    path: str | Path,
    *,
    format: str | None = None,
    audio_root: str | Path | None = None,
) -> int:
    require_torch()
    if not records:
        raise ValueError("records must not be empty")
    resolved_format = _resolve_feature_cache_format(path, format=format)
    rows = []
    audio_cache: dict[Path, tuple[list[float], int]] = {}
    for record in records:
        features = record_to_logmel_features(record, audio_root=audio_root, audio_cache=audio_cache).detach().cpu().tolist()
        rows.append(
            {
                "id": record.id,
                "audio": record.audio,
                "start": record.start,
                "end": record.end,
                "sample_rate": record.sample_rate,
                **{name: float(features[index]) for index, name in enumerate(AUDIO_FEATURE_NAMES)},
            }
        )
    if resolved_format == "parquet":
        _write_feature_parquet(path, rows)
    elif resolved_format == "lance":
        _write_feature_lance(path, rows)
    else:  # pragma: no cover - guarded by resolver.
        raise ValueError(f"unsupported feature cache format: {resolved_format}")
    return len(rows)


def load_logmel_feature_cache(
    path: str | Path,
    *,
    format: str | None = None,
    record_ids: list[str] | None = None,
):
    require_torch()
    resolved_format = _resolve_feature_cache_format(path, format=format)
    table = _read_feature_table(path, resolved_format)
    return _table_to_feature_tensor(table, record_ids=record_ids)


def count_logmel_feature_cache(path: str | Path, *, format: str | None = None) -> int:
    resolved_format = _resolve_feature_cache_format(path, format=format)
    table = _read_feature_table(path, resolved_format, columns=["id"])
    return table.num_rows


def benchmark_train_feature_cache(
    records: list[TurnManifestRecord],
    *,
    output_dir: str | Path,
    formats: list[str] | tuple[str, ...] = TRAIN_FEATURE_BENCHMARK_FORMATS,
    sample_count: int = 1000,
    seed: int = 0,
    max_records: int | None = None,
    audio_root: str | Path | None = None,
) -> list[TrainFeatureBenchmarkRow]:
    require_torch()
    if not records:
        raise ValueError("records must not be empty")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    unknown = sorted(set(formats) - set(TRAIN_FEATURE_BENCHMARK_FORMATS))
    if unknown:
        raise ValueError(f"unknown training feature benchmark format(s): {', '.join(unknown)}")

    records = list(records[:max_records]) if max_records else list(records)
    if not records:
        raise ValueError("no records selected for benchmark")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = _sample_indices(len(records), sample_count=sample_count, seed=seed)

    rows: list[TrainFeatureBenchmarkRow] = []
    source_sps = 0.0
    for name in formats:
        if name == "source_audio":
            row = _benchmark_source_features(records, indices, audio_root=audio_root, share_audio_cache=False)
            source_sps = row.samples_per_second
            rows.append(row)
        elif name == "source_audio_file_cache":
            rows.append(_benchmark_source_features(records, indices, audio_root=audio_root, share_audio_cache=True))
        elif name in FEATURE_CACHE_FORMATS:
            rows.append(
                _benchmark_cached_features(
                    name,
                    records,
                    indices,
                    output_dir=output_dir,
                    audio_root=audio_root,
                    source_samples_per_second=source_sps,
                )
            )
    if source_sps <= 0.0:
        baseline = next((row for row in rows if row.format == "source_audio"), None)
        source_sps = baseline.samples_per_second if baseline else 0.0
    if source_sps > 0.0:
        rows = [
            TrainFeatureBenchmarkRow(
                format=row.format,
                records=row.records,
                write_seconds=row.write_seconds,
                sample_count=row.sample_count,
                sample_seconds=row.sample_seconds,
                samples_per_second=row.samples_per_second,
                speedup_vs_source_audio=row.samples_per_second / source_sps,
                output_path=row.output_path,
                size_bytes=row.size_bytes,
                sample_strategy=row.sample_strategy,
            )
            for row in rows
        ]
    return rows


def _benchmark_source_features(
    records: list[TurnManifestRecord],
    indices: list[int],
    *,
    audio_root: str | Path | None,
    share_audio_cache: bool,
) -> TrainFeatureBenchmarkRow:
    audio_cache: dict[Path, tuple[list[float], int]] | None = {} if share_audio_cache else None
    start = time.perf_counter()
    features = [
        record_to_logmel_features(records[index], audio_root=audio_root, audio_cache=audio_cache)
        for index in indices
    ]
    tensor = torch.stack(features)
    sample_seconds = time.perf_counter() - start
    if tensor.shape[0] != len(indices):
        raise RuntimeError("source feature benchmark produced the wrong number of samples")
    samples_per_second = len(indices) / sample_seconds if sample_seconds > 0 else float("inf")
    return TrainFeatureBenchmarkRow(
        format="source_audio_file_cache" if share_audio_cache else "source_audio",
        records=len(records),
        write_seconds=0.0,
        sample_count=len(indices),
        sample_seconds=sample_seconds,
        samples_per_second=samples_per_second,
        speedup_vs_source_audio=samples_per_second,
        output_path="source_audio",
        size_bytes=0,
        sample_strategy="wav_decode_once_per_file_plus_stft" if share_audio_cache else "wav_decode_stft_per_sample",
    )


def _benchmark_cached_features(
    name: str,
    records: list[TurnManifestRecord],
    indices: list[int],
    *,
    output_dir: Path,
    audio_root: str | Path | None,
    source_samples_per_second: float,
) -> TrainFeatureBenchmarkRow:
    output_path = output_dir / f"logmel_features.{name}"
    write_start = time.perf_counter()
    write_logmel_feature_cache(records, output_path, format=name, audio_root=audio_root)
    write_seconds = time.perf_counter() - write_start

    sample_start = time.perf_counter()
    if name == "lance":
        tensor = load_logmel_feature_cache_by_indices(output_path, format=name, indices=indices)
        sample_strategy = "lance_take_cached_logmel"
    else:
        tensor = load_logmel_feature_cache_by_indices(output_path, format=name, indices=indices)
        sample_strategy = "parquet_cached_logmel_select"
    sample_seconds = time.perf_counter() - sample_start
    if tensor.shape[0] != len(indices):
        raise RuntimeError(f"{name} cache benchmark produced the wrong number of samples")
    samples_per_second = len(indices) / sample_seconds if sample_seconds > 0 else float("inf")
    return TrainFeatureBenchmarkRow(
        format=name,
        records=len(records),
        write_seconds=write_seconds,
        sample_count=len(indices),
        sample_seconds=sample_seconds,
        samples_per_second=samples_per_second,
        speedup_vs_source_audio=(samples_per_second / source_samples_per_second)
        if source_samples_per_second > 0
        else 0.0,
        output_path=str(output_path),
        size_bytes=_path_size_bytes(output_path),
        sample_strategy=sample_strategy,
    )


def load_logmel_feature_cache_by_indices(path: str | Path, *, format: str | None = None, indices: list[int]):
    resolved_format = _resolve_feature_cache_format(path, format=format)
    if resolved_format == "lance":
        lance = _require_lance()
        table = lance.dataset(str(path)).take(indices, columns=["id", *AUDIO_FEATURE_NAMES])
    else:
        table = _read_feature_table(path, resolved_format, columns=["id", *AUDIO_FEATURE_NAMES])
        table = table.take(indices)
    return _table_to_feature_tensor(table)


def _table_to_feature_tensor(table: Any, *, record_ids: list[str] | None = None):
    columns = table.to_pydict()
    ids = [str(item) for item in columns["id"]]
    rows = [
        [float(columns[name][index]) for name in AUDIO_FEATURE_NAMES]
        for index in range(len(ids))
    ]
    if record_ids is not None:
        by_id = {record_id: rows[index] for index, record_id in enumerate(ids)}
        missing = [record_id for record_id in record_ids if record_id not in by_id]
        if missing:
            raise RuntimeError(f"feature cache is missing {len(missing)} record id(s): {missing[:3]}")
        rows = [by_id[record_id] for record_id in record_ids]
    return torch.tensor(rows, dtype=torch.float32)


def _write_feature_parquet(path: str | Path, rows: list[dict[str, Any]]) -> None:
    pa, pq = _require_pyarrow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(_columns(rows)), path)


def _write_feature_lance(path: str | Path, rows: list[dict[str, Any]]) -> None:
    pa, lance = _require_pyarrow_lance()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(
        pa.table(_columns(rows)),
        str(path),
        mode="overwrite",
        commit_message="stable-asr logmel feature cache write",
    )


def _read_feature_table(path: str | Path, format: str, columns: list[str] | None = None):
    if format == "parquet":
        _, pq = _require_pyarrow()
        return pq.read_table(path, columns=columns)
    if format == "lance":
        lance = _require_lance()
        return lance.dataset(str(path)).to_table(columns=columns)
    raise ValueError(f"unsupported feature cache format: {format}")


def _resolve_feature_cache_format(path: str | Path, *, format: str | None) -> str:
    if format:
        if format not in FEATURE_CACHE_FORMATS:
            raise ValueError(f"unknown feature cache format: {format}")
        return format
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".lance":
        return "lance"
    raise ValueError(f"could not detect feature cache format from suffix {suffix!r}")


def _columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not rows:
        raise ValueError("rows must not be empty")
    return {key: [row.get(key) for row in rows] for key in rows[0]}


def _sample_indices(records: int, *, sample_count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(records) for _ in range(sample_count)]


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
        raise RuntimeError("Log-mel feature cache requires pyarrow. Install stable-asr[data].") from exc
    return pa, pq


def _require_lance():
    try:
        import lance
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Log-mel feature cache Lance support requires pylance. Install stable-asr[lance].") from exc
    if not hasattr(lance, "dataset") or not hasattr(lance, "write_dataset"):
        raise RuntimeError("Log-mel feature cache Lance support requires the pylance package that imports as 'lance'.")
    return lance


def _require_pyarrow_lance():
    try:
        import pyarrow as pa
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Log-mel feature cache Lance support requires pyarrow. Install stable-asr[lance].") from exc
    return pa, _require_lance()

