"""Chunk and lookahead sensitivity sweeps for streaming ASR fixtures."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from stable_asr.eval.report import MarkdownReport, dict_table
from stable_asr.streaming.metrics import StreamingASRReport, evaluate_streaming_records
from stable_asr.streaming.types import PartialHypothesis, StreamingASRRecord


@dataclass(frozen=True)
class StreamingScheduleSweepRow:
    chunk_ms: int
    lookahead_ms: int
    report: StreamingASRReport

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_ms": self.chunk_ms,
            "lookahead_ms": self.lookahead_ms,
            **self.report.to_dict(),
        }


@dataclass(frozen=True)
class StreamingScheduleSweepReport:
    rows: list[StreamingScheduleSweepRow]

    def to_dict(self) -> dict[str, object]:
        return {"rows": [row.to_dict() for row in self.rows]}

    def to_markdown(self) -> str:
        report = MarkdownReport("Stable-ASR Streaming ASR Schedule Sweep")
        report.add_section("Chunk And Lookahead", dict_table(_sweep_rows(self.rows)))
        return report.to_markdown()


def sweep_streaming_schedule(
    records: list[StreamingASRRecord],
    *,
    chunk_ms_values: list[int],
    lookahead_ms_values: list[int],
) -> StreamingScheduleSweepReport:
    if not records:
        raise ValueError("records must not be empty")
    if not chunk_ms_values:
        raise ValueError("chunk_ms_values must not be empty")
    if not lookahead_ms_values:
        raise ValueError("lookahead_ms_values must not be empty")

    rows: list[StreamingScheduleSweepRow] = []
    for chunk_ms in chunk_ms_values:
        if chunk_ms <= 0:
            raise ValueError("chunk_ms values must be positive")
        for lookahead_ms in lookahead_ms_values:
            if lookahead_ms < 0:
                raise ValueError("lookahead_ms values must be non-negative")
            scheduled = [_schedule_record(record, chunk_ms=chunk_ms, lookahead_ms=lookahead_ms) for record in records]
            rows.append(
                StreamingScheduleSweepRow(
                    chunk_ms=chunk_ms,
                    lookahead_ms=lookahead_ms,
                    report=evaluate_streaming_records(scheduled),
                )
            )
    return StreamingScheduleSweepReport(rows=rows)


def _schedule_record(record: StreamingASRRecord, *, chunk_ms: int, lookahead_ms: int) -> StreamingASRRecord:
    partials = [
        PartialHypothesis(
            time=_scheduled_time(partial.time, chunk_ms=chunk_ms, lookahead_ms=lookahead_ms),
            text=partial.text,
            is_final=partial.is_final,
        )
        for partial in record.partials
    ]
    endpoint_time = (
        _scheduled_time(record.endpoint_time, chunk_ms=chunk_ms, lookahead_ms=lookahead_ms)
        if record.endpoint_time is not None
        else None
    )
    return replace(record, partials=partials, endpoint_time=endpoint_time)


def _scheduled_time(time_sec: float, *, chunk_ms: int, lookahead_ms: int) -> float:
    chunk_sec = chunk_ms / 1000.0
    lookahead_sec = lookahead_ms / 1000.0
    return round(math.ceil(time_sec / chunk_sec) * chunk_sec + lookahead_sec, 6)


def _sweep_rows(rows: list[StreamingScheduleSweepRow]) -> list[dict[str, object]]:
    table_rows: list[dict[str, object]] = []
    for row in rows:
        metrics = row.report.to_dict()
        table_rows.append(
            {
                "chunk_ms": row.chunk_ms,
                "lookahead_ms": row.lookahead_ms,
                "wer": f"{metrics['wer']:.4f}",
                "rtf": f"{metrics['rtf']:.4f}",
                "first_partial_latency": f"{metrics['first_partial_latency']:.4f}",
                "final_latency": f"{metrics['final_latency']:.4f}",
                "endpoint_delay": f"{metrics['endpoint_delay']:.4f}",
                "partial_revision_rate": f"{metrics['partial_revision_rate']:.4f}",
                "stable_prefix_ratio": f"{metrics['stable_prefix_ratio']:.4f}",
                "timestamp_drift": f"{metrics['timestamp_drift']:.4f}",
            }
        )
    return table_rows
