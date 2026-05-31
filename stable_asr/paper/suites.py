"""Machine-readable benchmark suite definitions for paper artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table


DEFAULT_SUITE_ID = "stable_asr_v0"


DEFAULT_BENCHMARK_SUITE: dict[str, Any] = {
    "id": DEFAULT_SUITE_ID,
    "version": "0.1.0",
    "title": "Stable-ASR v0 Paper Benchmark Suite",
    "description": (
        "A reproducible smoke-to-paper benchmark suite covering turn-taking quality, "
        "latency, data-layer behavior, ASR manifest recipes, VoiceWorld scenarios, "
        "policy search, streaming ASR metrics, and external ASR transcript conversion."
    ),
    "leaderboard_suite": "stable_asr_v0",
    "required_artifacts": [
        "paper_results.json",
        "artifact_manifest.json",
        "ARTIFACT_INDEX.md",
        "benchmark_suite.json",
        "BENCHMARK_SUITE.md",
        "leaderboard.jsonl",
        "leaderboard.csv",
    ],
    "tasks": [
        {
            "id": "turn_quality",
            "title": "Turn-Taking Quality",
            "coverage": "system_slice_metric",
            "systems": ["rule_endpoint", "vad_pause", "text_turn", "prediction_manifest", "nanoturn_pico"],
            "slices": ["overall"],
            "metrics": [
                {"name": "accuracy", "unit": "rate", "higher_is_better": True},
                {"name": "macro_f1", "unit": "rate", "higher_is_better": True},
                {"name": "false_complete_rate", "unit": "rate", "higher_is_better": False},
                {"name": "missed_interrupt_rate", "unit": "rate", "higher_is_better": False},
            ],
        },
        {
            "id": "turn_latency",
            "title": "Turn Predictor Latency",
            "coverage": "system_slice_metric",
            "systems": ["rule_endpoint", "vad_pause", "text_turn", "prediction_manifest", "nanoturn_pico"],
            "slices": ["overall"],
            "metrics": [
                {"name": "avg_latency_ms", "unit": "ms", "higher_is_better": False},
                {"name": "p95_latency_ms", "unit": "ms", "higher_is_better": False},
                {"name": "rtf", "unit": "ratio", "higher_is_better": False},
                {"name": "throughput_predictions_per_sec", "unit": "pred/s", "higher_is_better": True},
                {"name": "artifact_bytes", "unit": "bytes", "higher_is_better": False},
            ],
        },
        {
            "id": "data_layer",
            "title": "Data Layer",
            "systems": ["jsonl", "parquet", "lance"],
            "slices": ["random_sampling"],
            "metrics": [
                {"name": "write_seconds", "unit": "s", "higher_is_better": False},
                {"name": "read_seconds", "unit": "s", "higher_is_better": False},
                {"name": "size_bytes", "unit": "bytes", "higher_is_better": False},
                {"name": "samples_per_second", "unit": "samples/s", "higher_is_better": True},
            ],
        },
        {
            "id": "asr_manifest_recipe",
            "title": "ASR Manifest Recipe",
            "systems": ["metadata_table"],
            "slices": ["asr_fixture"],
            "metrics": [
                {"name": "records", "unit": "count", "higher_is_better": True},
                {"name": "valid", "unit": "bool", "higher_is_better": True},
                {"name": "total_duration_sec", "unit": "s", "higher_is_better": True},
                {"name": "speakers", "unit": "count", "higher_is_better": True},
            ],
        },
        {
            "id": "voiceworld",
            "title": "VoiceWorld Scenario Robustness",
            "coverage": "system_slice_metric",
            "systems": ["scenario_evaluator"],
            "slices": [
                "incomplete_pause",
                "backchannel",
                "wait_stop",
                "user_interruption",
                "side_conversation",
                "ambient_speech",
                "noisy_farfield",
                "code_switching",
            ],
            "metrics": [
                {"name": "accuracy", "unit": "rate", "higher_is_better": True},
                {"name": "macro_f1", "unit": "rate", "higher_is_better": True},
                {"name": "false_complete_rate", "unit": "rate", "higher_is_better": False},
                {"name": "missed_interrupt_rate", "unit": "rate", "higher_is_better": False},
            ],
        },
        {
            "id": "policy_search",
            "title": "Cost-Sensitive Policy Search",
            "coverage": "system_slice_metric",
            "systems": ["best_policy"],
            "slices": ["overall"],
            "metrics": [
                {"name": "objective_score", "unit": "cost", "higher_is_better": False},
            ],
        },
        {
            "id": "streaming_asr",
            "title": "Streaming ASR",
            "systems": ["balanced_fixture", "fast_unstable_fixture", "schedule_sweep", "command_fixture"],
            "slices": ["adapter", "chunk/lookahead"],
            "metrics": [
                {"name": "wer", "unit": "rate", "higher_is_better": False},
                {"name": "cer", "unit": "rate", "higher_is_better": False},
                {"name": "rtf", "unit": "ratio", "higher_is_better": False},
                {"name": "first_partial_latency", "unit": "s", "higher_is_better": False},
                {"name": "final_latency", "unit": "s", "higher_is_better": False},
                {"name": "endpoint_delay", "unit": "s", "higher_is_better": False},
                {"name": "partial_revision_rate", "unit": "rate", "higher_is_better": False},
                {"name": "stable_prefix_ratio", "unit": "rate", "higher_is_better": True},
                {"name": "timestamp_drift", "unit": "s", "higher_is_better": False},
            ],
        },
        {
            "id": "asr_transcript_conversion",
            "title": "External ASR Transcript Conversion",
            "coverage": "system_slice_metric",
            "systems": ["whisper", "funasr"],
            "slices": ["converted_schema"],
            "metrics": [
                {"name": "wer", "unit": "rate", "higher_is_better": False},
                {"name": "cer", "unit": "rate", "higher_is_better": False},
                {"name": "rtf", "unit": "ratio", "higher_is_better": False},
                {"name": "endpoint_delay", "unit": "s", "higher_is_better": False},
                {"name": "partial_revision_rate", "unit": "rate", "higher_is_better": False},
                {"name": "timestamp_drift", "unit": "s", "higher_is_better": False},
            ],
        },
    ],
}


@dataclass(frozen=True)
class BenchmarkSuiteValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "benchmark_suite: OK"
        return "benchmark_suite: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class BenchmarkSuiteCoverage:
    ok: bool
    missing: list[str]
    rows: int

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "missing": self.missing, "rows": self.rows}

    def to_text(self) -> str:
        if self.ok:
            return f"benchmark_suite_coverage: OK ({self.rows} row(s))"
        return "benchmark_suite_coverage: FAILED\n" + "\n".join(f"- {item}" for item in self.missing)


def load_benchmark_suite(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_BENCHMARK_SUITE))
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("benchmark suite must be a JSON object")
    return payload


def write_benchmark_suite_json(path: str | Path, suite: dict[str, Any] | None = None) -> str:
    suite = suite or load_benchmark_suite()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_benchmark_suite(suite: dict[str, Any]) -> BenchmarkSuiteValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "leaderboard_suite", "tasks"):
        if key not in suite:
            errors.append(f"missing top-level key: {key}")
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks must be a non-empty list")
        return BenchmarkSuiteValidation(ok=False, errors=errors)

    seen: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"task {index} must be an object")
            continue
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            errors.append(f"task {index} missing id")
        elif task_id in seen:
            errors.append(f"duplicate task id: {task_id}")
        else:
            seen.add(task_id)
        for key in ("title", "systems", "slices", "metrics"):
            if key not in task:
                errors.append(f"task {task_id or index} missing {key}")
        coverage = task.get("coverage", "system_metric")
        if coverage not in {"system_metric", "system_slice_metric"}:
            errors.append(f"task {task_id or index} has unknown coverage mode: {coverage}")
        metrics = task.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            errors.append(f"task {task_id or index} metrics must be non-empty")
            continue
        for metric_index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                errors.append(f"task {task_id or index} metric {metric_index} must be an object")
                continue
            for key in ("name", "unit", "higher_is_better"):
                if key not in metric:
                    errors.append(f"task {task_id or index} metric {metric_index} missing {key}")
            if "higher_is_better" in metric and not isinstance(metric["higher_is_better"], bool):
                errors.append(f"task {task_id or index} metric {metric_index} higher_is_better must be boolean")
    return BenchmarkSuiteValidation(ok=not errors, errors=errors)


def audit_benchmark_suite_coverage(
    results: dict[str, object],
    *,
    suite: dict[str, Any] | None = None,
) -> BenchmarkSuiteCoverage:
    suite = suite or load_benchmark_suite()
    validation = validate_benchmark_suite(suite)
    if not validation.ok:
        return BenchmarkSuiteCoverage(ok=False, missing=validation.errors, rows=0)

    from stable_asr.paper.leaderboard import leaderboard_rows

    rows = leaderboard_rows(results)
    row_index: dict[tuple[str, str], dict[str, set[str]]] = {}
    for row in rows:
        row_index.setdefault((row.task, row.system), {}).setdefault(row.slice, set()).add(row.metric)

    missing: list[str] = []
    for task in suite["tasks"]:
        task_id = str(task["id"])
        systems = [str(system) for system in task.get("systems", [])]
        slices = [str(slice_name) for slice_name in task.get("slices", [])]
        metrics = [str(metric["name"]) for metric in task.get("metrics", []) if isinstance(metric, dict)]
        coverage = str(task.get("coverage", "system_metric"))
        if not any(row.task == task_id for row in rows):
            missing.append(f"{task_id}: no leaderboard rows")
            continue
        for system in systems:
            system_slices = row_index.get((task_id, system), {})
            if not system_slices:
                missing.append(f"{task_id}/{system}: no rows")
                continue
            if coverage == "system_slice_metric":
                for slice_name in slices:
                    slice_metrics = system_slices.get(slice_name, set())
                    if not slice_metrics:
                        missing.append(f"{task_id}/{system}/{slice_name}: no rows")
                        continue
                    for metric in metrics:
                        if metric not in slice_metrics:
                            missing.append(f"{task_id}/{system}/{slice_name}/{metric}: missing metric")
            else:
                available_metrics = set().union(*system_slices.values())
                for metric in metrics:
                    if metric not in available_metrics:
                        missing.append(f"{task_id}/{system}/{metric}: missing metric")
    return BenchmarkSuiteCoverage(ok=not missing, missing=missing, rows=len(rows))


def benchmark_suite_markdown(suite: dict[str, Any]) -> str:
    validation = validate_benchmark_suite(suite)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    lines = [
        f"# {suite['title']}",
        "",
        f"- id: `{suite['id']}`",
        f"- version: `{suite['version']}`",
        f"- leaderboard suite: `{suite['leaderboard_suite']}`",
        "",
        str(suite.get("description", "")),
        "",
        "## Tasks",
        "",
        dict_table(_task_rows(suite)),
        "",
        "## Required Artifacts",
        "",
    ]
    for artifact in suite.get("required_artifacts", []):
        lines.append(f"- `{artifact}`")
    lines.append("")
    return "\n".join(lines)


def _task_rows(suite: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for task in suite["tasks"]:
        metrics = task.get("metrics", [])
        rows.append(
            {
                "task": task["id"],
                "title": task["title"],
                "coverage": task.get("coverage", "system_metric"),
                "systems": len(task.get("systems", [])),
                "slices": len(task.get("slices", [])),
                "metrics": ", ".join(str(metric["name"]) for metric in metrics if isinstance(metric, dict)),
            }
        )
    return rows
