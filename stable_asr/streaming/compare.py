"""Compare multiple streaming ASR transcript adapters under one evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stable_asr.eval.report import MarkdownReport, dict_table
from stable_asr.models.adapters.asr import StreamingASRAdapter
from stable_asr.models.adapters.transcript import transcript_jsonl_adapter
from stable_asr.streaming.metrics import StreamingASRReport, evaluate_streaming_records


@dataclass(frozen=True)
class StreamingASRComparisonRow:
    adapter: str
    input_path: str
    report: StreamingASRReport

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter": self.adapter,
            "input_path": self.input_path,
            **self.report.to_dict(),
        }


@dataclass(frozen=True)
class StreamingASRComparisonReport:
    rows: list[StreamingASRComparisonRow]

    def to_dict(self) -> dict[str, object]:
        return {"rows": [row.to_dict() for row in self.rows]}

    def to_markdown(self) -> str:
        report = MarkdownReport("Stable-ASR Streaming ASR Adapter Comparison")
        report.add_section("Adapters", dict_table(_comparison_rows(self.rows)))
        return report.to_markdown()


def compare_streaming_transcript_jsonl(inputs: list[tuple[str, str | Path]]) -> StreamingASRComparisonReport:
    return compare_streaming_adapters([transcript_jsonl_adapter(adapter, path) for adapter, path in inputs])


def compare_streaming_adapters(adapters: list[StreamingASRAdapter]) -> StreamingASRComparisonReport:
    if not adapters:
        raise ValueError("at least one adapter is required")
    rows: list[StreamingASRComparisonRow] = []
    seen: set[str] = set()
    for adapter in adapters:
        if not adapter.name:
            raise ValueError("adapter name must not be empty")
        if adapter.name in seen:
            raise ValueError(f"duplicate adapter name: {adapter.name}")
        seen.add(adapter.name)
        records = adapter.load_records()
        input_path = str(getattr(adapter, "path", "adapter"))
        rows.append(
            StreamingASRComparisonRow(
                adapter=adapter.name,
                input_path=input_path,
                report=evaluate_streaming_records(records),
            )
        )
    return StreamingASRComparisonReport(rows=rows)


def _comparison_rows(rows: list[StreamingASRComparisonRow]) -> list[dict[str, object]]:
    table_rows: list[dict[str, object]] = []
    for row in rows:
        metrics = row.report.to_dict()
        table_rows.append(
            {
                "adapter": row.adapter,
                "records": metrics["records"],
                "wer": f"{metrics['wer']:.4f}",
                "cer": f"{metrics['cer']:.4f}",
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
