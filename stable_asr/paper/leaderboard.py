"""Leaderboard-ready exports from paper result artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.paper.tables import load_paper_results


@dataclass(frozen=True)
class LeaderboardRow:
    suite: str
    task: str
    system: str
    slice: str
    metric: str
    value: float
    unit: str
    higher_is_better: bool
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "task": self.task,
            "system": self.system,
            "slice": self.slice,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "source": self.source,
        }


LEADERBOARD_COLUMNS = (
    "suite",
    "task",
    "system",
    "slice",
    "metric",
    "value",
    "unit",
    "higher_is_better",
    "source",
)


def leaderboard_rows(results: dict[str, object], *, source: str = "paper_results.json") -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    rows.extend(_baseline_rows(results, source=source))
    rows.extend(_turn_latency_rows(results, source=source))
    rows.extend(_data_rows(results, source=source))
    rows.extend(_asr_manifest_rows(results, source=source))
    rows.extend(_scenario_rows(results, source=source))
    rows.extend(_policy_rows(results, source=source))
    rows.extend(_streaming_rows(results, source=source))
    return rows


def export_leaderboard(
    results_path: str | Path,
    output_path: str | Path,
    *,
    format: str = "jsonl",
) -> str:
    results_path = Path(results_path)
    rows = leaderboard_rows(load_paper_results(results_path), source=str(results_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "jsonl":
        output_path.write_text(
            "\n".join(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
    elif format == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(LEADERBOARD_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow(row.to_dict())
    else:
        raise ValueError("format must be 'jsonl' or 'csv'")
    return str(output_path)


def _baseline_rows(results: dict[str, object], *, source: str) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    baselines = _dict(results.get("baselines"))
    for system, payload in baselines.items():
        payload = _dict(payload)
        classification = _dict(payload.get("classification"))
        interaction = _dict(payload.get("interaction"))
        rows.extend(
            [
                _row("turn_quality", system, "overall", "accuracy", classification.get("accuracy"), "rate", True, source),
                _row("turn_quality", system, "overall", "macro_f1", classification.get("macro_f1"), "rate", True, source),
                _row("turn_quality", system, "overall", "false_complete_rate", interaction.get("false_complete_rate"), "rate", False, source),
                _row("turn_quality", system, "overall", "missed_interrupt_rate", interaction.get("missed_interrupt_rate"), "rate", False, source),
            ]
        )
    return [row for row in rows if row is not None]


def _turn_latency_rows(results: dict[str, object], *, source: str) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    benchmarks = _dict(results.get("turn_benchmarks"))
    for system, payload in benchmarks.items():
        payload = _dict(payload)
        artifact_bytes = _dict(payload.get("artifact_bytes"))
        artifact_size = sum(float(value) for value in artifact_bytes.values())
        rows.extend(
            [
                _row("turn_latency", system, "overall", "avg_latency_ms", payload.get("avg_latency_ms"), "ms", False, source),
                _row("turn_latency", system, "overall", "p95_latency_ms", payload.get("p95_latency_ms"), "ms", False, source),
                _row("turn_latency", system, "overall", "rtf", payload.get("rtf"), "ratio", False, source),
                _row("turn_latency", system, "overall", "throughput_predictions_per_sec", payload.get("throughput_predictions_per_sec"), "pred/s", True, source),
                _row("turn_latency", system, "overall", "artifact_bytes", artifact_size, "bytes", False, source),
            ]
        )
    return [row for row in rows if row is not None]


def _data_rows(results: dict[str, object], *, source: str) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    benchmark = _dict(_dict(results.get("data")).get("benchmark"))
    if benchmark.get("status") != "completed":
        return rows
    for payload in benchmark.get("rows", []):
        payload = _dict(payload)
        system = str(payload.get("format", "unknown"))
        sample_strategy = str(payload.get("sample_strategy", "disabled"))
        rows.extend(
            [
                _row("data_layer", system, sample_strategy, "write_seconds", payload.get("write_seconds"), "s", False, source),
                _row("data_layer", system, sample_strategy, "read_seconds", payload.get("read_seconds"), "s", False, source),
                _row("data_layer", system, sample_strategy, "size_bytes", payload.get("size_bytes"), "bytes", False, source),
                _row("data_layer", system, sample_strategy, "sample_seconds", payload.get("sample_seconds"), "s", False, source),
                _row("data_layer", system, sample_strategy, "samples_per_second", payload.get("samples_per_second"), "samples/s", True, source),
            ]
        )
    return [row for row in rows if row is not None]


def _asr_manifest_rows(results: dict[str, object], *, source: str) -> list[LeaderboardRow]:
    recipe = _dict(_dict(results.get("data")).get("asr_manifest_recipe"))
    if not recipe:
        return []
    summary = _dict(recipe.get("summary"))
    validation = _dict(recipe.get("validation"))
    return [
        row
        for row in [
            _row("asr_manifest_recipe", "metadata_table", "asr_fixture", "records", recipe.get("records"), "count", True, source),
            _row(
                "asr_manifest_recipe",
                "metadata_table",
                "asr_fixture",
                "valid",
                1.0 if validation.get("ok") else 0.0,
                "bool",
                True,
                source,
            ),
            _row(
                "asr_manifest_recipe",
                "metadata_table",
                "asr_fixture",
                "total_duration_sec",
                summary.get("total_duration_sec"),
                "s",
                True,
                source,
            ),
            _row("asr_manifest_recipe", "metadata_table", "asr_fixture", "speakers", summary.get("speakers"), "count", True, source),
        ]
        if row is not None
    ]


def _scenario_rows(results: dict[str, object], *, source: str) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    scenarios = _dict(_dict(results.get("scenarios")).get("by_scenario"))
    for scenario, payload in scenarios.items():
        payload = _dict(payload)
        classification = _dict(payload.get("classification"))
        interaction = _dict(payload.get("interaction"))
        rows.extend(
            [
                _row("voiceworld", "scenario_evaluator", str(scenario), "accuracy", classification.get("accuracy"), "rate", True, source),
                _row("voiceworld", "scenario_evaluator", str(scenario), "macro_f1", classification.get("macro_f1"), "rate", True, source),
                _row("voiceworld", "scenario_evaluator", str(scenario), "false_complete_rate", interaction.get("false_complete_rate"), "rate", False, source),
                _row("voiceworld", "scenario_evaluator", str(scenario), "missed_interrupt_rate", interaction.get("missed_interrupt_rate"), "rate", False, source),
            ]
        )
    return [row for row in rows if row is not None]


def _policy_rows(results: dict[str, object], *, source: str) -> list[LeaderboardRow]:
    best = _dict(_dict(results.get("policy_search")).get("best"))
    if not best:
        return []
    return [
        row
        for row in [
            _row("policy_search", "best_policy", "overall", "objective_score", best.get("score"), "cost", False, source)
        ]
        if row is not None
    ]


def _streaming_rows(results: dict[str, object], *, source: str) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    streaming = _dict(results.get("streaming_asr"))
    comparison = _dict(streaming.get("adapter_comparison"))
    comparison_rows = comparison.get("rows")
    if isinstance(comparison_rows, list):
        for payload in comparison_rows:
            payload = _dict(payload)
            system = str(payload.get("adapter", "unknown"))
            rows.extend(_streaming_metric_rows(system, "adapter", payload, source=source))
    else:
        rows.extend(_streaming_metric_rows("streaming_fixture", "overall", _dict(streaming.get("metrics")), source=source))

    sweep = _dict(streaming.get("schedule_sweep"))
    sweep_rows = sweep.get("rows")
    if isinstance(sweep_rows, list):
        for payload in sweep_rows:
            payload = _dict(payload)
            slice_name = f"chunk={payload.get('chunk_ms')}ms/lookahead={payload.get('lookahead_ms')}ms"
            rows.extend(_streaming_metric_rows("schedule_sweep", slice_name, payload, source=source))
    command_adapter = _dict(streaming.get("command_adapter"))
    command_metrics = _dict(command_adapter.get("metrics"))
    if command_metrics:
        rows.extend(
            _streaming_metric_rows(
                str(command_adapter.get("adapter", "command_fixture")),
                "command",
                command_metrics,
                source=source,
            )
        )
    conversion_rows = streaming.get("asr_transcript_conversions")
    if isinstance(conversion_rows, list):
        for payload in conversion_rows:
            payload = _dict(payload)
            schema = str(payload.get("schema", "unknown"))
            metrics = _dict(payload.get("metrics"))
            rows.extend(_asr_transcript_metric_rows(schema, metrics, source=source))
    return [row for row in rows if row is not None]


def _asr_transcript_metric_rows(system: str, payload: dict[str, Any], *, source: str) -> list[LeaderboardRow | None]:
    return [
        _row("asr_transcript_conversion", system, "converted_schema", "wer", payload.get("wer"), "rate", False, source),
        _row("asr_transcript_conversion", system, "converted_schema", "cer", payload.get("cer"), "rate", False, source),
        _row("asr_transcript_conversion", system, "converted_schema", "rtf", payload.get("rtf"), "ratio", False, source),
        _row("asr_transcript_conversion", system, "converted_schema", "endpoint_delay", payload.get("endpoint_delay"), "s", False, source),
        _row("asr_transcript_conversion", system, "converted_schema", "partial_revision_rate", payload.get("partial_revision_rate"), "rate", False, source),
        _row("asr_transcript_conversion", system, "converted_schema", "timestamp_drift", payload.get("timestamp_drift"), "s", False, source),
    ]


def _streaming_metric_rows(system: str, slice_name: str, payload: dict[str, Any], *, source: str) -> list[LeaderboardRow | None]:
    return [
        _row("streaming_asr", system, slice_name, "wer", payload.get("wer"), "rate", False, source),
        _row("streaming_asr", system, slice_name, "cer", payload.get("cer"), "rate", False, source),
        _row("streaming_asr", system, slice_name, "rtf", payload.get("rtf"), "ratio", False, source),
        _row("streaming_asr", system, slice_name, "first_partial_latency", payload.get("first_partial_latency"), "s", False, source),
        _row("streaming_asr", system, slice_name, "final_latency", payload.get("final_latency"), "s", False, source),
        _row("streaming_asr", system, slice_name, "endpoint_delay", payload.get("endpoint_delay"), "s", False, source),
        _row("streaming_asr", system, slice_name, "partial_revision_rate", payload.get("partial_revision_rate"), "rate", False, source),
        _row("streaming_asr", system, slice_name, "stable_prefix_ratio", payload.get("stable_prefix_ratio"), "rate", True, source),
        _row("streaming_asr", system, slice_name, "timestamp_drift", payload.get("timestamp_drift"), "s", False, source),
    ]


def _row(
    task: str,
    system: str,
    slice_name: str,
    metric: str,
    value: object,
    unit: str,
    higher_is_better: bool,
    source: str,
) -> LeaderboardRow | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return LeaderboardRow(
        suite="stable_asr_v0",
        task=task,
        system=str(system),
        slice=str(slice_name),
        metric=metric,
        value=numeric,
        unit=unit,
        higher_is_better=higher_is_better,
        source=source,
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
