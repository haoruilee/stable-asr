"""Latency and artifact-size benchmarks for turn predictors."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.eval.report import MarkdownReport, dict_table
from stable_asr.eval.turn_eval import TurnPredictor


@dataclass(frozen=True)
class TurnBenchmarkReport:
    records: int
    warmup: int
    repeat: int
    predictions: int
    total_audio_seconds: float
    total_seconds: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_predictions_per_sec: float
    rtf: float
    artifact_bytes: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "warmup": self.warmup,
            "repeat": self.repeat,
            "predictions": self.predictions,
            "total_audio_seconds": self.total_audio_seconds,
            "total_seconds": self.total_seconds,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "throughput_predictions_per_sec": self.throughput_predictions_per_sec,
            "rtf": self.rtf,
            "artifact_bytes": self.artifact_bytes,
        }

    def to_markdown(self) -> str:
        report = MarkdownReport("Stable-ASR Turn Benchmark")
        report.add_section(
            "Summary",
            dict_table(
                [
                    {
                        "records": self.records,
                        "repeat": self.repeat,
                        "avg_latency_ms": f"{self.avg_latency_ms:.4f}",
                        "p50_latency_ms": f"{self.p50_latency_ms:.4f}",
                        "p95_latency_ms": f"{self.p95_latency_ms:.4f}",
                        "throughput": f"{self.throughput_predictions_per_sec:.2f}",
                        "rtf": f"{self.rtf:.6f}",
                    }
                ]
            ),
        )
        if self.artifact_bytes:
            report.add_section(
                "Artifacts",
                dict_table(
                    [
                        {"artifact": name, "bytes": size}
                        for name, size in self.artifact_bytes.items()
                    ]
                ),
            )
        return report.to_markdown()


def benchmark_turn_predictor(
    records: list[TurnManifestRecord],
    predictor: TurnPredictor,
    *,
    warmup: int = 1,
    repeat: int = 5,
    artifact_paths: list[str | Path] | None = None,
) -> TurnBenchmarkReport:
    if not records:
        raise ValueError("records must not be empty")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeat < 1:
        raise ValueError("repeat must be at least 1")

    for _ in range(warmup):
        for record in records:
            predictor.predict(record)

    latencies: list[float] = []
    total_start = time.perf_counter()
    for _ in range(repeat):
        for record in records:
            start = time.perf_counter()
            predictor.predict(record)
            latencies.append(time.perf_counter() - start)
    total_seconds = time.perf_counter() - total_start

    predictions = len(latencies)
    total_audio_seconds = sum(record.duration for record in records) * repeat
    latencies_ms = [latency * 1000.0 for latency in latencies]
    return TurnBenchmarkReport(
        records=len(records),
        warmup=warmup,
        repeat=repeat,
        predictions=predictions,
        total_audio_seconds=total_audio_seconds,
        total_seconds=total_seconds,
        avg_latency_ms=sum(latencies_ms) / len(latencies_ms),
        p50_latency_ms=statistics.median(latencies_ms),
        p95_latency_ms=_percentile(latencies_ms, 0.95),
        throughput_predictions_per_sec=predictions / total_seconds if total_seconds > 0 else 0.0,
        rtf=total_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0,
        artifact_bytes=_artifact_sizes(artifact_paths or []),
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def _artifact_sizes(paths: list[str | Path]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists() and path.is_file():
            sizes[str(path)] = path.stat().st_size
    return sizes
