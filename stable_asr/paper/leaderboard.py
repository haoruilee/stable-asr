"""Leaderboard-ready exports from paper result artifacts."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.paper.suites import load_benchmark_suite, validate_benchmark_suite
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


@dataclass(frozen=True)
class LeaderboardValidationIssue:
    line: int
    field: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"line": self.line, "field": self.field, "detail": self.detail}


@dataclass(frozen=True)
class LeaderboardValidationReport:
    ok: bool
    path: str
    rows: int
    tasks: dict[str, int]
    issues: list[LeaderboardValidationIssue]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "path": self.path,
            "rows": self.rows,
            "tasks": self.tasks,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_text(self) -> str:
        lines = [f"leaderboard_validation: {'OK' if self.ok else 'FAILED'}", f"rows: {self.rows}"]
        for issue in self.issues:
            lines.append(f"- line {issue.line} {issue.field}: {issue.detail}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        rows = [{"task": task, "rows": count} for task, count in sorted(self.tasks.items())]
        issue_rows = [
            {"line": issue.line, "field": issue.field, "detail": issue.detail}
            for issue in self.issues
        ]
        return "\n".join(
            [
                "# Stable-ASR Leaderboard Validation",
                "",
                f"- status: `{'OK' if self.ok else 'FAILED'}`",
                f"- rows: `{self.rows}`",
                f"- issues: `{len(self.issues)}`",
                "",
                "## Task Coverage",
                "",
                dict_table(rows) if rows else "No rows.",
                "",
                "## Issues",
                "",
                dict_table(issue_rows) if issue_rows else "No issues.",
                "",
            ]
        )


@dataclass(frozen=True)
class LeaderboardRankedRow:
    rank: int
    suite: str
    task: str
    slice: str
    metric: str
    system: str
    value: float
    unit: str
    higher_is_better: bool
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "suite": self.suite,
            "task": self.task,
            "slice": self.slice,
            "metric": self.metric,
            "system": self.system,
            "value": self.value,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "source": self.source,
        }


@dataclass(frozen=True)
class LeaderboardReport:
    ok: bool
    path: str
    suite: str
    rows: int
    groups: int
    top_k: int
    ranked_rows: list[LeaderboardRankedRow]
    validation: LeaderboardValidationReport

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "path": self.path,
            "suite": self.suite,
            "rows": self.rows,
            "groups": self.groups,
            "top_k": self.top_k,
            "ranked_rows": [row.to_dict() for row in self.ranked_rows],
            "validation": self.validation.to_dict(),
        }

    def to_text(self) -> str:
        lines = [
            f"leaderboard_report: {'OK' if self.ok else 'FAILED'}",
            f"path: {self.path}",
            f"suite: {self.suite}",
            f"rows: {self.rows}",
            f"groups: {self.groups}",
            f"ranked_rows: {len(self.ranked_rows)}",
        ]
        if not self.validation.ok:
            lines.append(self.validation.to_text())
        return "\n".join(lines)

    def to_markdown(self) -> str:
        summary_rows = [
            {
                "task": row.task,
                "slice": row.slice,
                "metric": row.metric,
                "rank": row.rank,
                "system": row.system,
                "value": _format_value(row.value),
                "unit": row.unit,
                "direction": "higher" if row.higher_is_better else "lower",
                "source": row.source,
            }
            for row in self.ranked_rows
        ]
        lines = [
            "# Stable-ASR Leaderboard Report",
            "",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- suite: `{self.suite}`",
            f"- rows: `{self.rows}`",
            f"- groups: `{self.groups}`",
            f"- top_k: `{self.top_k}`",
            "",
            "## Ranked Metrics",
            "",
            dict_table(summary_rows) if summary_rows else "No valid ranked rows.",
        ]
        if not self.validation.ok:
            lines.extend(["", "## Validation", "", self.validation.to_markdown()])
        return "\n".join(lines)


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


def validate_leaderboard_jsonl(
    path: str | Path,
    *,
    suite: dict[str, Any] | None = None,
    require_known_systems: bool = False,
    require_known_slices: bool = False,
    require_complete_suite: bool = False,
) -> LeaderboardValidationReport:
    suite = suite or load_benchmark_suite()
    suite_validation = validate_benchmark_suite(suite)
    if not suite_validation.ok:
        raise ValueError("; ".join(suite_validation.errors))

    path = Path(path)
    issues: list[LeaderboardValidationIssue] = []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    task_counts: dict[str, int] = {}
    task_specs = {str(task["id"]): task for task in suite["tasks"]}
    expected_suite = str(suite.get("leaderboard_suite", suite.get("id", "stable_asr_v0")))

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(LeaderboardValidationIssue(line_number, "json", str(exc)))
                continue
            if not isinstance(payload, dict):
                issues.append(LeaderboardValidationIssue(line_number, "row", "row must be a JSON object"))
                continue
            rows.append(payload)
            _validate_leaderboard_row(
                payload,
                line_number=line_number,
                expected_suite=expected_suite,
                task_specs=task_specs,
                require_known_systems=require_known_systems,
                require_known_slices=require_known_slices,
                seen=seen,
                issues=issues,
            )
            task = str(payload.get("task", ""))
            if task:
                task_counts[task] = task_counts.get(task, 0) + 1

    if not rows:
        issues.append(LeaderboardValidationIssue(0, "rows", "leaderboard file has no rows"))
    if require_complete_suite:
        for task_id in task_specs:
            if task_counts.get(task_id, 0) == 0:
                issues.append(LeaderboardValidationIssue(0, "task", f"missing task rows: {task_id}"))

    return LeaderboardValidationReport(
        ok=not issues,
        path=str(path),
        rows=len(rows),
        tasks=task_counts,
        issues=issues,
    )


def leaderboard_report(
    path: str | Path,
    *,
    suite: dict[str, Any] | None = None,
    top_k: int = 3,
    require_known_systems: bool = False,
    require_known_slices: bool = False,
    require_complete_suite: bool = False,
) -> LeaderboardReport:
    suite = suite or load_benchmark_suite()
    validation = validate_leaderboard_jsonl(
        path,
        suite=suite,
        require_known_systems=require_known_systems,
        require_known_slices=require_known_slices,
        require_complete_suite=require_complete_suite,
    )
    suite_id = str(suite.get("leaderboard_suite", suite.get("id", "stable_asr_v0")))
    rows = _load_leaderboard_rows(path) if validation.ok else []
    ranked_rows = _rank_leaderboard_rows(rows, top_k=max(1, top_k))
    groups = len({(row.task, row.slice, row.metric) for row in rows})
    return LeaderboardReport(
        ok=validation.ok,
        path=str(path),
        suite=suite_id,
        rows=validation.rows,
        groups=groups,
        top_k=max(1, top_k),
        ranked_rows=ranked_rows,
        validation=validation,
    )


def _validate_leaderboard_row(
    payload: dict[str, Any],
    *,
    line_number: int,
    expected_suite: str,
    task_specs: dict[str, dict[str, Any]],
    require_known_systems: bool,
    require_known_slices: bool,
    seen: set[tuple[str, str, str, str, str]],
    issues: list[LeaderboardValidationIssue],
) -> None:
    for column in LEADERBOARD_COLUMNS:
        if column not in payload:
            issues.append(LeaderboardValidationIssue(line_number, column, "missing required field"))

    suite = str(payload.get("suite", ""))
    task = str(payload.get("task", ""))
    system = str(payload.get("system", ""))
    slice_name = str(payload.get("slice", ""))
    metric = str(payload.get("metric", ""))
    unit = str(payload.get("unit", ""))
    source = str(payload.get("source", ""))

    if suite != expected_suite:
        issues.append(LeaderboardValidationIssue(line_number, "suite", f"expected {expected_suite}, got {suite}"))
    if not system:
        issues.append(LeaderboardValidationIssue(line_number, "system", "system must be non-empty"))
    if not slice_name:
        issues.append(LeaderboardValidationIssue(line_number, "slice", "slice must be non-empty"))
    if not source:
        issues.append(LeaderboardValidationIssue(line_number, "source", "source must be non-empty"))

    task_spec = task_specs.get(task)
    if task_spec is None:
        issues.append(LeaderboardValidationIssue(line_number, "task", f"unknown task: {task}"))
        task_spec = {}
    else:
        metrics = {
            str(item["name"]): item
            for item in task_spec.get("metrics", [])
            if isinstance(item, dict) and "name" in item
        }
        metric_spec = metrics.get(metric)
        if metric_spec is None:
            issues.append(LeaderboardValidationIssue(line_number, "metric", f"unknown metric for task {task}: {metric}"))
        else:
            if unit != str(metric_spec.get("unit", "")):
                issues.append(
                    LeaderboardValidationIssue(
                        line_number,
                        "unit",
                        f"expected {metric_spec.get('unit')}, got {unit}",
                    )
                )
            expected_direction = bool(metric_spec.get("higher_is_better"))
            if payload.get("higher_is_better") is not expected_direction:
                issues.append(
                    LeaderboardValidationIssue(
                        line_number,
                        "higher_is_better",
                        f"expected {expected_direction}, got {payload.get('higher_is_better')}",
                    )
                )
        if require_known_systems and system not in {str(item) for item in task_spec.get("systems", [])}:
            issues.append(LeaderboardValidationIssue(line_number, "system", f"unknown system for task {task}: {system}"))
        if require_known_slices and slice_name not in {str(item) for item in task_spec.get("slices", [])}:
            issues.append(LeaderboardValidationIssue(line_number, "slice", f"unknown slice for task {task}: {slice_name}"))

    try:
        value = float(payload.get("value"))
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError):
        issues.append(LeaderboardValidationIssue(line_number, "value", "value must be a finite number"))

    key = (suite, task, system, slice_name, metric)
    if key in seen:
        issues.append(LeaderboardValidationIssue(line_number, "row", "duplicate suite/task/system/slice/metric row"))
    seen.add(key)


def _load_leaderboard_rows(path: str | Path) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            rows.append(
                LeaderboardRow(
                    suite=str(payload["suite"]),
                    task=str(payload["task"]),
                    system=str(payload["system"]),
                    slice=str(payload["slice"]),
                    metric=str(payload["metric"]),
                    value=float(payload["value"]),
                    unit=str(payload["unit"]),
                    higher_is_better=bool(payload["higher_is_better"]),
                    source=str(payload["source"]),
                )
            )
    return rows


def _rank_leaderboard_rows(rows: list[LeaderboardRow], *, top_k: int) -> list[LeaderboardRankedRow]:
    ranked: list[LeaderboardRankedRow] = []
    groups: dict[tuple[str, str, str], list[LeaderboardRow]] = {}
    for row in rows:
        groups.setdefault((row.task, row.slice, row.metric), []).append(row)
    for key in sorted(groups):
        group_rows = groups[key]
        if not group_rows:
            continue
        higher_is_better = group_rows[0].higher_is_better
        ordered = sorted(
            group_rows,
            key=lambda row: (-row.value if higher_is_better else row.value, row.system),
        )
        previous_value: float | None = None
        previous_rank = 0
        for index, row in enumerate(ordered, start=1):
            rank = previous_rank if previous_value is not None and row.value == previous_value else index
            previous_rank = rank
            previous_value = row.value
            if rank > top_k:
                continue
            ranked.append(
                LeaderboardRankedRow(
                    rank=rank,
                    suite=row.suite,
                    task=row.task,
                    slice=row.slice,
                    metric=row.metric,
                    system=row.system,
                    value=row.value,
                    unit=row.unit,
                    higher_is_better=row.higher_is_better,
                    source=row.source,
                )
            )
    return ranked


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


def _format_value(value: float) -> str:
    return f"{value:.6g}"
