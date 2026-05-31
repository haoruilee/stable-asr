"""Generate paper-facing case-study artifacts from failure analyses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.manifest import TurnManifestRecord, load_manifest
from stable_asr.eval.report import dict_table
from stable_asr.models.adapters.transcript import load_streaming_transcript_jsonl
from stable_asr.paper.tables import load_paper_results
from stable_asr.streaming.types import StreamingASRRecord


@dataclass(frozen=True)
class PaperCaseStudyArtifacts:
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {"json": self.json_path, "markdown": self.markdown_path}


def paper_case_studies(
    results_path: str | Path,
    output_dir: str | Path,
    *,
    max_turn_cases_per_baseline: int = 8,
    max_streaming_cases_per_source: int = 8,
) -> PaperCaseStudyArtifacts:
    """Write JSON and Markdown case-study artifacts for paper appendices."""

    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_paper_results(results_path)
    payload = build_case_studies(
        results,
        max_turn_cases_per_baseline=max_turn_cases_per_baseline,
        max_streaming_cases_per_source=max_streaming_cases_per_source,
    )
    json_path = output_dir / "case_studies.json"
    markdown_path = output_dir / "CASE_STUDIES.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(case_studies_markdown(payload, results_path=results_path), encoding="utf-8")
    return PaperCaseStudyArtifacts(json_path=str(json_path), markdown_path=str(markdown_path))


def build_case_studies(
    results: dict[str, object],
    *,
    max_turn_cases_per_baseline: int = 8,
    max_streaming_cases_per_source: int = 8,
) -> dict[str, object]:
    turn_records = _load_turn_record_index(results)
    streaming_records = _load_streaming_record_indexes(results)
    turn_cases = _turn_case_rows(
        results,
        turn_records=turn_records,
        max_cases_per_baseline=max_turn_cases_per_baseline,
    )
    streaming_cases = _streaming_case_rows(
        results,
        streaming_records=streaming_records,
        max_cases_per_source=max_streaming_cases_per_source,
    )
    return {
        "turn_cases": turn_cases,
        "streaming_cases": streaming_cases,
        "summary": {
            "turn_cases": len(turn_cases),
            "streaming_cases": len(streaming_cases),
            "turn_categories": _counts(row["category"] for row in turn_cases),
            "streaming_categories": _counts(row["category"] for row in streaming_cases),
        },
    }


def case_studies_markdown(payload: dict[str, object], *, results_path: str | Path) -> str:
    summary = _dict(payload.get("summary"))
    turn_cases = [row for row in payload.get("turn_cases", []) if isinstance(row, dict)]
    streaming_cases = [row for row in payload.get("streaming_cases", []) if isinstance(row, dict)]

    lines = [
        "# Stable-ASR Case Studies",
        "",
        f"Results source: `{results_path}`",
        "",
        "## Summary",
        "",
        dict_table(
            [
                {
                    "turn_cases": summary.get("turn_cases", 0),
                    "streaming_cases": summary.get("streaming_cases", 0),
                }
            ]
        ),
        "",
        "## Turn-Taking Failures",
        "",
        dict_table(_compact_turn_rows(turn_cases)),
        "",
        "## Streaming ASR Failures",
        "",
        dict_table(_compact_streaming_rows(streaming_cases)),
        "",
    ]
    return "\n".join(lines)


def _load_turn_record_index(results: dict[str, object]) -> dict[str, TurnManifestRecord]:
    data = _dict(results.get("data"))
    manifest_path = data.get("manifest_path")
    if not isinstance(manifest_path, str):
        return {}
    path = Path(manifest_path)
    if not path.exists():
        return {}
    return {record.id: record for record in load_manifest(path)}


def _load_streaming_record_indexes(results: dict[str, object]) -> dict[str, dict[str, StreamingASRRecord]]:
    streaming = _dict(results.get("streaming_asr"))
    indexes: dict[str, dict[str, StreamingASRRecord]] = {}

    fixture_path = streaming.get("fixture_path")
    if isinstance(fixture_path, str):
        indexes["overall"] = _load_streaming_index(fixture_path)

    adapter_paths = streaming.get("adapter_fixture_paths")
    if isinstance(adapter_paths, dict):
        for name, path in adapter_paths.items():
            if isinstance(name, str) and isinstance(path, str):
                indexes[name] = _load_streaming_index(path)

    command_adapter = _dict(streaming.get("command_adapter"))
    command_output = command_adapter.get("output_path")
    command_name = command_adapter.get("adapter", "command_adapter")
    if isinstance(command_output, str) and isinstance(command_name, str):
        indexes[command_name] = _load_streaming_index(command_output)

    return indexes


def _load_streaming_index(path: str) -> dict[str, StreamingASRRecord]:
    raw_path = Path(path)
    if not raw_path.exists():
        return {}
    return {record.id: record for record in load_streaming_transcript_jsonl(raw_path)}


def _turn_case_rows(
    results: dict[str, object],
    *,
    turn_records: dict[str, TurnManifestRecord],
    max_cases_per_baseline: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baselines = _dict(results.get("baselines"))
    for baseline, payload in baselines.items():
        analysis = _dict(_dict(payload).get("failure_analysis"))
        cases = analysis.get("cases", [])
        if not isinstance(cases, list):
            continue
        for case in cases[:max_cases_per_baseline]:
            if not isinstance(case, dict):
                continue
            record = turn_records.get(str(case.get("id")))
            rows.append(_turn_case_row(str(baseline), case, record))
    return rows


def _turn_case_row(
    baseline: str,
    case: dict[str, object],
    record: TurnManifestRecord | None,
) -> dict[str, object]:
    metadata = record.metadata if record is not None else {}
    return {
        "baseline": baseline,
        "id": case.get("id", ""),
        "category": case.get("category", ""),
        "severity": case.get("severity", 0),
        "scenario": case.get("scenario") or (record.scenario if record else ""),
        "audio": record.audio if record else "",
        "text": record.text if record else "",
        "asr_text": record.asr_text if record else "",
        "true_label": case.get("true_label", ""),
        "pred_label": case.get("pred_label", ""),
        "true_action": case.get("true_action", ""),
        "pred_action": case.get("pred_action", ""),
        "confidence": case.get("confidence", 0.0),
        "reason": case.get("reason", ""),
        "snr_db": metadata.get("snr_db", ""),
        "reverb": metadata.get("reverb", ""),
        "speaking_rate": metadata.get("speaking_rate", ""),
        "overlap_offset_ms": metadata.get("overlap_offset_ms", ""),
        "network_jitter_ms": metadata.get("network_jitter_ms", ""),
    }


def _streaming_case_rows(
    results: dict[str, object],
    *,
    streaming_records: dict[str, dict[str, StreamingASRRecord]],
    max_cases_per_source: int,
) -> list[dict[str, object]]:
    streaming = _dict(results.get("streaming_asr"))
    rows: list[dict[str, object]] = []

    metrics = _dict(streaming.get("metrics"))
    rows.extend(
        _streaming_cases_from_analysis(
            "overall",
            _dict(metrics.get("failure_analysis")),
            streaming_records.get("overall", {}),
            max_cases=max_cases_per_source,
        )
    )

    comparison = _dict(streaming.get("adapter_comparison"))
    comparison_rows = comparison.get("rows", [])
    if isinstance(comparison_rows, list):
        for row in comparison_rows:
            row = _dict(row)
            source = str(row.get("adapter", "unknown"))
            rows.extend(
                _streaming_cases_from_analysis(
                    source,
                    _dict(row.get("failure_analysis")),
                    streaming_records.get(source, {}),
                    max_cases=max_cases_per_source,
                )
            )

    command_adapter = _dict(streaming.get("command_adapter"))
    command_source = str(command_adapter.get("adapter", "command_adapter"))
    command_metrics = _dict(command_adapter.get("metrics"))
    if command_metrics:
        rows.extend(
            _streaming_cases_from_analysis(
                command_source,
                _dict(command_metrics.get("failure_analysis")),
                streaming_records.get(command_source, {}),
                max_cases=max_cases_per_source,
            )
        )
    return rows


def _streaming_cases_from_analysis(
    source: str,
    analysis: dict[str, object],
    record_index: dict[str, StreamingASRRecord],
    *,
    max_cases: int,
) -> list[dict[str, object]]:
    cases = analysis.get("cases", [])
    if not isinstance(cases, list):
        return []
    rows: list[dict[str, object]] = []
    for case in cases[:max_cases]:
        if not isinstance(case, dict):
            continue
        record = record_index.get(str(case.get("id")))
        rows.append(_streaming_case_row(source, case, record))
    return rows


def _streaming_case_row(
    source: str,
    case: dict[str, object],
    record: StreamingASRRecord | None,
) -> dict[str, object]:
    metadata = record.metadata if record is not None else {}
    return {
        "source": source,
        "id": case.get("id", ""),
        "category": case.get("category", ""),
        "severity": case.get("severity", 0),
        "value": case.get("value", 0.0),
        "threshold": case.get("threshold", 0.0),
        "audio": metadata.get("audio", ""),
        "reference": case.get("reference") or (record.reference if record else ""),
        "final_text": case.get("final_text") or (record.final_text if record else ""),
        "audio_duration": record.audio_duration if record else "",
        "processing_time": record.processing_time if record else "",
        "speech_end_time": record.speech_end_time if record else "",
        "endpoint_time": record.endpoint_time if record else "",
        "partials": len(record.partials) if record else 0,
        "reason": case.get("reason", ""),
    }


def _compact_turn_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "baseline": row.get("baseline", ""),
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "scenario": row.get("scenario", ""),
            "audio": row.get("audio", ""),
            "text": row.get("text", ""),
            "true_action": row.get("true_action", ""),
            "pred_action": row.get("pred_action", ""),
        }
        for row in rows
    ]


def _compact_streaming_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "source": row.get("source", ""),
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "value": row.get("value", ""),
            "threshold": row.get("threshold", ""),
            "audio": row.get("audio", ""),
            "reference": row.get("reference", ""),
            "final_text": row.get("final_text", ""),
        }
        for row in rows
    ]


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
