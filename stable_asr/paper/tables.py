"""Paper table extraction from structured result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from stable_asr.eval.report import dict_table


PAPER_TABLES = (
    "baselines",
    "turn_benchmark",
    "data",
    "asr_manifest_recipe",
    "failure_cases",
    "streaming",
    "streaming_failures",
    "streaming_sweep",
    "asr_transcript_conversions",
    "scenarios",
    "policy",
)


def load_paper_results(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("paper results must be a JSON object")
    return payload


def paper_table(results_path: str | Path, table: str) -> str:
    results = load_paper_results(results_path)
    if table == "baselines":
        return baseline_table(results)
    if table == "turn_benchmark":
        return turn_benchmark_table(results)
    if table == "data":
        return data_table(results)
    if table == "asr_manifest_recipe":
        return asr_manifest_recipe_table(results)
    if table == "failure_cases":
        return failure_cases_table(results)
    if table == "streaming":
        return streaming_table(results)
    if table == "streaming_failures":
        return streaming_failures_table(results)
    if table == "streaming_sweep":
        return streaming_sweep_table(results)
    if table == "asr_transcript_conversions":
        return asr_transcript_conversion_table(results)
    if table == "scenarios":
        return scenario_table(results)
    if table == "policy":
        return policy_table(results)
    raise ValueError(f"unknown paper table: {table}")


def baseline_table(results: dict[str, object]) -> str:
    baselines = results["baselines"]
    if not isinstance(baselines, dict):
        raise ValueError("missing baselines object")
    rows = []
    for name, payload in baselines.items():
        if not isinstance(payload, dict):
            continue
        classification = payload["classification"]
        interaction = payload["interaction"]
        rows.append(
            {
                "baseline": name,
                "accuracy": f"{classification['accuracy']:.4f}",
                "macro_f1": f"{classification['macro_f1']:.4f}",
                "false_complete_rate": f"{interaction['false_complete_rate']:.4f}",
                "missed_interrupt_rate": f"{interaction['missed_interrupt_rate']:.4f}",
            }
        )
    return dict_table(rows)


def turn_benchmark_table(results: dict[str, object]) -> str:
    benchmarks = results["turn_benchmarks"]
    if not isinstance(benchmarks, dict):
        raise ValueError("missing turn_benchmarks object")
    rows = []
    for name, payload in benchmarks.items():
        if not isinstance(payload, dict):
            continue
        artifact_bytes = payload.get("artifact_bytes", {})
        artifact_size = sum(artifact_bytes.values()) if isinstance(artifact_bytes, dict) else 0
        rows.append(
            {
                "baseline": name,
                "avg_latency_ms": f"{payload['avg_latency_ms']:.4f}",
                "p50_latency_ms": f"{payload['p50_latency_ms']:.4f}",
                "p95_latency_ms": f"{payload['p95_latency_ms']:.4f}",
                "throughput": f"{payload['throughput_predictions_per_sec']:.2f}",
                "rtf": f"{payload['rtf']:.6f}",
                "artifact_bytes": artifact_size,
            }
        )
    return dict_table(rows)


def data_table(results: dict[str, object]) -> str:
    data = results["data"]
    if not isinstance(data, dict):
        raise ValueError("missing data object")
    benchmark = data["benchmark"]
    if not isinstance(benchmark, dict) or benchmark.get("status") != "completed":
        return str(benchmark.get("reason", "data benchmark unavailable"))
    rows = []
    for row in benchmark["rows"]:
        rows.append(
            {
                "format": row["format"],
                "records": row["records"],
                "write_seconds": f"{row['write_seconds']:.6f}",
                "read_seconds": f"{row['read_seconds']:.6f}",
                "size_bytes": row["size_bytes"],
                "sample_count": row.get("sample_count", 0),
                "sample_seconds": f"{float(row.get('sample_seconds', 0.0)):.6f}",
                "samples_per_second": f"{float(row.get('samples_per_second', 0.0)):.2f}",
                "sample_strategy": row.get("sample_strategy", "disabled"),
            }
        )
    return dict_table(rows)


def asr_manifest_recipe_table(results: dict[str, object]) -> str:
    data = results["data"]
    if not isinstance(data, dict):
        raise ValueError("missing data object")
    recipe = data.get("asr_manifest_recipe")
    if not isinstance(recipe, dict):
        raise ValueError("missing asr_manifest_recipe object")
    summary = recipe.get("summary", {})
    validation = recipe.get("validation", {})
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(validation, dict):
        validation = {}
    return dict_table(
        [
            {
                "records": recipe.get("records", 0),
                "valid": validation.get("ok", False),
                "languages": ", ".join(sorted(_dict_keys(summary.get("languages", {})))),
                "sources": ", ".join(sorted(_dict_keys(summary.get("sources", {})))),
                "splits": ", ".join(sorted(_dict_keys(summary.get("splits", {})))),
                "total_duration_sec": f"{float(summary.get('total_duration_sec', 0.0)):.2f}",
            }
        ]
    )


def failure_cases_table(results: dict[str, object]) -> str:
    baselines = results["baselines"]
    if not isinstance(baselines, dict):
        raise ValueError("missing baselines object")
    rows = []
    for baseline, payload in baselines.items():
        payload = payload if isinstance(payload, dict) else {}
        analysis = payload.get("failure_analysis", {})
        if not isinstance(analysis, dict):
            analysis = {}
        category_counts = analysis.get("category_counts", {})
        scenario_counts = analysis.get("scenario_counts", {})
        if not isinstance(category_counts, dict):
            category_counts = {}
        if not isinstance(scenario_counts, dict):
            scenario_counts = {}
        top_scenarios = ", ".join(str(name) for name in list(scenario_counts.keys())[:3])
        if not category_counts:
            rows.append(
                {
                    "baseline": baseline,
                    "category": "none",
                    "count": 0,
                    "top_scenarios": "",
                }
            )
            continue
        for category, count in category_counts.items():
            rows.append(
                {
                    "baseline": baseline,
                    "category": category,
                    "count": count,
                    "top_scenarios": top_scenarios,
                }
            )
    return dict_table(rows)


def streaming_table(results: dict[str, object]) -> str:
    streaming = results["streaming_asr"]
    if not isinstance(streaming, dict):
        raise ValueError("missing streaming_asr object")
    comparison = streaming.get("adapter_comparison")
    if isinstance(comparison, dict) and isinstance(comparison.get("rows"), list):
        rows = []
        for row in comparison["rows"]:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "adapter": row["adapter"],
                    "records": row["records"],
                    "wer": f"{float(row['wer']):.4f}",
                    "cer": f"{float(row['cer']):.4f}",
                    "rtf": f"{float(row['rtf']):.4f}",
                    "endpoint_delay": f"{float(row['endpoint_delay']):.4f}",
                    "partial_revision_rate": f"{float(row['partial_revision_rate']):.4f}",
                    "timestamp_drift": f"{float(row['timestamp_drift']):.4f}",
                }
            )
        if rows:
            return dict_table(rows)
    metrics = streaming["metrics"]
    return dict_table(
        [
            {
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
        ]
    )


def streaming_failures_table(results: dict[str, object]) -> str:
    streaming = results["streaming_asr"]
    if not isinstance(streaming, dict):
        raise ValueError("missing streaming_asr object")

    rows: list[dict[str, object]] = []
    _append_streaming_failure_rows(rows, "overall", _dict(_dict(streaming.get("metrics")).get("failure_analysis")))

    comparison = _dict(streaming.get("adapter_comparison"))
    comparison_rows = comparison.get("rows", [])
    if isinstance(comparison_rows, list):
        for row in comparison_rows:
            row = _dict(row)
            adapter = str(row.get("adapter", "unknown"))
            _append_streaming_failure_rows(rows, adapter, _dict(row.get("failure_analysis")))

    command_adapter = _dict(streaming.get("command_adapter"))
    command_metrics = _dict(command_adapter.get("metrics"))
    if command_metrics:
        adapter = str(command_adapter.get("adapter", "command_adapter"))
        _append_streaming_failure_rows(rows, adapter, _dict(command_metrics.get("failure_analysis")))

    if not rows:
        return dict_table([{"source": "overall", "category": "none", "count": 0, "top_cases": ""}])
    return dict_table(rows)


def _append_streaming_failure_rows(rows: list[dict[str, object]], source: str, analysis: dict[str, object]) -> None:
    category_counts = analysis.get("category_counts", {})
    cases = analysis.get("cases", [])
    if not isinstance(category_counts, dict) or not category_counts:
        rows.append({"source": source, "category": "none", "count": 0, "top_cases": ""})
        return
    for category, count in category_counts.items():
        top_cases = []
        if isinstance(cases, list):
            for case in cases:
                case = _dict(case)
                if case.get("category") != category:
                    continue
                top_cases.append(str(case.get("id", "")))
                if len(top_cases) == 3:
                    break
        rows.append(
            {
                "source": source,
                "category": category,
                "count": count,
                "top_cases": ", ".join(item for item in top_cases if item),
            }
        )


def streaming_sweep_table(results: dict[str, object]) -> str:
    streaming = results["streaming_asr"]
    if not isinstance(streaming, dict):
        raise ValueError("missing streaming_asr object")
    sweep = streaming.get("schedule_sweep")
    if not isinstance(sweep, dict) or not isinstance(sweep.get("rows"), list):
        raise ValueError("missing streaming schedule sweep")
    rows = []
    for row in sweep["rows"]:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "chunk_ms": row["chunk_ms"],
                "lookahead_ms": row["lookahead_ms"],
                "first_partial_latency": f"{float(row['first_partial_latency']):.4f}",
                "final_latency": f"{float(row['final_latency']):.4f}",
                "endpoint_delay": f"{float(row['endpoint_delay']):.4f}",
                "partial_revision_rate": f"{float(row['partial_revision_rate']):.4f}",
                "timestamp_drift": f"{float(row['timestamp_drift']):.4f}",
            }
        )
    return dict_table(rows)


def asr_transcript_conversion_table(results: dict[str, object]) -> str:
    streaming = results["streaming_asr"]
    if not isinstance(streaming, dict):
        raise ValueError("missing streaming_asr object")
    conversions = streaming.get("asr_transcript_conversions")
    if not isinstance(conversions, list):
        raise ValueError("missing ASR transcript conversions")
    rows = []
    for conversion in conversions:
        if not isinstance(conversion, dict):
            continue
        metrics = conversion.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append(
            {
                "schema": conversion["schema"],
                "records": conversion["records"],
                "wer": f"{float(metrics.get('wer', 0.0)):.4f}",
                "cer": f"{float(metrics.get('cer', 0.0)):.4f}",
                "rtf": f"{float(metrics.get('rtf', 0.0)):.4f}",
                "endpoint_delay": f"{float(metrics.get('endpoint_delay', 0.0)):.4f}",
                "partial_revision_rate": f"{float(metrics.get('partial_revision_rate', 0.0)):.4f}",
                "timestamp_drift": f"{float(metrics.get('timestamp_drift', 0.0)):.4f}",
            }
        )
    return dict_table(rows)


def scenario_table(results: dict[str, object]) -> str:
    scenarios = results["scenarios"]
    if not isinstance(scenarios, dict):
        raise ValueError("missing scenarios object")
    rows = []
    for scenario, payload in scenarios["by_scenario"].items():
        rows.append(
            {
                "scenario": scenario,
                "records": len(payload["examples"]),
                "accuracy": f"{payload['classification']['accuracy']:.4f}",
                "macro_f1": f"{payload['classification']['macro_f1']:.4f}",
                "false_complete_rate": f"{payload['interaction']['false_complete_rate']:.4f}",
                "missed_interrupt_rate": f"{payload['interaction']['missed_interrupt_rate']:.4f}",
            }
        )
    return dict_table(rows)


def policy_table(results: dict[str, object]) -> str:
    policy_search = results["policy_search"]
    if not isinstance(policy_search, dict):
        raise ValueError("missing policy_search object")
    best = policy_search["best"]
    config = best["config"]
    return dict_table(
        [
            {
                "score": f"{best['score']:.4f}",
                "complete_threshold": config["complete_threshold"],
                "backchannel_threshold": config["backchannel_threshold"],
                "wait_threshold": config["wait_threshold"],
                "interrupt_min_confidence": config["interrupt_min_confidence"],
                "trials": len(policy_search["trials"]),
            }
        ]
    )


def _dict_keys(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    return [str(key) for key in value.keys()]


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}
