"""External turn prediction submission packages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.registry import TURN_FORMATS, load_turn_records
from stable_asr.eval.report import dict_table
from stable_asr.eval.turn_eval import TurnEvalReport, evaluate_turn_records
from stable_asr.models.adapters import TurnPredictionManifestAdapter, validate_turn_prediction_jsonl
from stable_asr.models.adapters.transcript import load_streaming_transcript_jsonl
from stable_asr.models.adapters.prediction_audit import TurnPredictionValidationReport
from stable_asr.paper.leaderboard import (
    LeaderboardMergeReport,
    LeaderboardRow,
    merge_leaderboard_jsonl,
    validate_leaderboard_jsonl,
)
from stable_asr.paper.suites import load_benchmark_suite
from stable_asr.schema_validation import SchemaFileValidationReport, validate_schema_file
from stable_asr.streaming.metrics import StreamingASRReport, evaluate_streaming_records
from stable_asr.turn.policy import TurnPolicy, TurnPolicyConfig


TURN_SUBMISSION_VERSION = "turn_submission_v0"
STREAMING_SUBMISSION_VERSION = "streaming_submission_v0"


@dataclass(frozen=True)
class TurnSubmissionArtifacts:
    output_dir: str
    manifest: str
    summary_markdown: str
    schema_validation: dict[str, str]
    prediction_validation: dict[str, str]
    evaluation: dict[str, str]
    leaderboard: dict[str, str]
    leaderboard_validation: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "manifest": self.manifest,
            "summary_markdown": self.summary_markdown,
            "schema_validation": self.schema_validation,
            "prediction_validation": self.prediction_validation,
            "evaluation": self.evaluation,
            "leaderboard": self.leaderboard,
            "leaderboard_validation": self.leaderboard_validation,
        }


@dataclass(frozen=True)
class TurnSubmissionReport:
    ok: bool
    system: str
    dataset: str
    predictions: str
    records: int
    schema_validation: SchemaFileValidationReport
    prediction_validation: TurnPredictionValidationReport
    evaluation: TurnEvalReport | None
    leaderboard_validation_ok: bool | None
    artifacts: TurnSubmissionArtifacts

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "version": TURN_SUBMISSION_VERSION,
            "system": self.system,
            "dataset": self.dataset,
            "predictions": self.predictions,
            "records": self.records,
            "schema_validation": self.schema_validation.to_dict(),
            "prediction_validation": self.prediction_validation.to_dict(),
            "evaluation": self.evaluation.to_dict() if self.evaluation is not None else None,
            "leaderboard_validation_ok": self.leaderboard_validation_ok,
            "artifacts": self.artifacts.to_dict(),
        }

    def to_markdown(self) -> str:
        metric_rows = []
        if self.evaluation is not None:
            metric_rows = [
                {
                    "metric": "accuracy",
                    "value": f"{self.evaluation.classification.accuracy:.6f}",
                },
                {
                    "metric": "macro_f1",
                    "value": f"{self.evaluation.classification.macro_f1:.6f}",
                },
                {
                    "metric": "false_complete_rate",
                    "value": f"{self.evaluation.interaction['false_complete_rate']:.6f}",
                },
                {
                    "metric": "missed_interrupt_rate",
                    "value": f"{self.evaluation.interaction['missed_interrupt_rate']:.6f}",
                },
            ]
        artifacts = self.artifacts.to_dict()
        artifact_rows = [
            {"section": section, "path": value}
            for section, value in artifacts.items()
            if isinstance(value, str)
        ]
        for section, paths in artifacts.items():
            if isinstance(paths, dict):
                for name, path in paths.items():
                    artifact_rows.append({"section": f"{section}:{name}", "path": path})

        return "\n".join(
            [
                "# Stable-ASR Turn Submission",
                "",
                f"- status: `{'OK' if self.ok else 'FAILED'}`",
                f"- system: `{self.system}`",
                f"- dataset: `{self.dataset}`",
                f"- predictions: `{self.predictions}`",
                f"- records: `{self.records}`",
                f"- schema_validation: `{'OK' if self.schema_validation.ok else 'FAILED'}`",
                f"- prediction_validation: `{'OK' if self.prediction_validation.ok else 'FAILED'}`",
                f"- leaderboard_validation: `{self.leaderboard_validation_ok}`",
                "",
                "## Metrics",
                "",
                dict_table(metric_rows) if metric_rows else "No metrics; validation failed before evaluation.",
                "",
                "## Artifacts",
                "",
                dict_table(artifact_rows),
                "",
            ]
        )


@dataclass(frozen=True)
class StreamingSubmissionArtifacts:
    output_dir: str
    manifest: str
    summary_markdown: str
    schema_validation: dict[str, str]
    evaluation: dict[str, str]
    leaderboard: dict[str, str]
    leaderboard_validation: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "manifest": self.manifest,
            "summary_markdown": self.summary_markdown,
            "schema_validation": self.schema_validation,
            "evaluation": self.evaluation,
            "leaderboard": self.leaderboard,
            "leaderboard_validation": self.leaderboard_validation,
        }


@dataclass(frozen=True)
class StreamingSubmissionReport:
    ok: bool
    system: str
    input_path: str
    slice_name: str
    records: int
    schema_validation: SchemaFileValidationReport
    evaluation: StreamingASRReport | None
    leaderboard_validation_ok: bool | None
    artifacts: StreamingSubmissionArtifacts

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "version": STREAMING_SUBMISSION_VERSION,
            "system": self.system,
            "input_path": self.input_path,
            "slice": self.slice_name,
            "records": self.records,
            "schema_validation": self.schema_validation.to_dict(),
            "evaluation": self.evaluation.to_dict() if self.evaluation is not None else None,
            "leaderboard_validation_ok": self.leaderboard_validation_ok,
            "artifacts": self.artifacts.to_dict(),
        }

    def to_markdown(self) -> str:
        metric_rows = _streaming_metric_table(self.evaluation)
        artifacts = self.artifacts.to_dict()
        artifact_rows = [
            {"section": section, "path": value}
            for section, value in artifacts.items()
            if isinstance(value, str)
        ]
        for section, paths in artifacts.items():
            if isinstance(paths, dict):
                for name, path in paths.items():
                    artifact_rows.append({"section": f"{section}:{name}", "path": path})

        return "\n".join(
            [
                "# Stable-ASR Streaming Submission",
                "",
                f"- status: `{'OK' if self.ok else 'FAILED'}`",
                f"- system: `{self.system}`",
                f"- input: `{self.input_path}`",
                f"- slice: `{self.slice_name}`",
                f"- records: `{self.records}`",
                f"- schema_validation: `{'OK' if self.schema_validation.ok else 'FAILED'}`",
                f"- leaderboard_validation: `{self.leaderboard_validation_ok}`",
                "",
                "## Metrics",
                "",
                dict_table(metric_rows) if metric_rows else "No metrics; validation failed before evaluation.",
                "",
                "## Artifacts",
                "",
                dict_table(artifact_rows),
                "",
            ]
        )


@dataclass(frozen=True)
class SubmissionIndexEntry:
    manifest: str
    version: str
    task: str
    system: str
    ok: bool
    records: int
    leaderboard: str | None
    summary_markdown: str | None
    issues: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest,
            "version": self.version,
            "task": self.task,
            "system": self.system,
            "ok": self.ok,
            "records": self.records,
            "leaderboard": self.leaderboard,
            "summary_markdown": self.summary_markdown,
            "issues": self.issues,
        }


@dataclass(frozen=True)
class SubmissionIndexArtifacts:
    output_dir: str
    index_json: str
    summary_markdown: str
    leaderboard: str | None
    leaderboard_validation: dict[str, str]
    leaderboard_report: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "index_json": self.index_json,
            "summary_markdown": self.summary_markdown,
            "leaderboard": self.leaderboard,
            "leaderboard_validation": self.leaderboard_validation,
            "leaderboard_report": self.leaderboard_report,
        }


@dataclass(frozen=True)
class SubmissionIndexReport:
    ok: bool
    root: str
    submissions: list[SubmissionIndexEntry]
    leaderboard_merge: LeaderboardMergeReport | None
    artifacts: SubmissionIndexArtifacts

    @property
    def valid_submissions(self) -> int:
        return sum(1 for submission in self.submissions if submission.ok and submission.leaderboard and not submission.issues)

    @property
    def failed_submissions(self) -> int:
        return sum(1 for submission in self.submissions if not submission.ok or submission.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "root": self.root,
            "submissions": [submission.to_dict() for submission in self.submissions],
            "valid_submissions": self.valid_submissions,
            "failed_submissions": self.failed_submissions,
            "leaderboard_merge": self.leaderboard_merge.to_dict() if self.leaderboard_merge is not None else None,
            "artifacts": self.artifacts.to_dict(),
        }

    def to_text(self) -> str:
        lines = [
            f"submission_index: {'OK' if self.ok else 'FAILED'}",
            f"root: {self.root}",
            f"submissions: {len(self.submissions)}",
            f"valid_submissions: {self.valid_submissions}",
            f"failed_submissions: {self.failed_submissions}",
        ]
        if self.leaderboard_merge is not None:
            lines.append(f"leaderboard: {self.leaderboard_merge.output_path}")
        for submission in self.submissions:
            if submission.issues:
                lines.append(f"- {submission.manifest}: " + "; ".join(submission.issues))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        rows = [
            {
                "task": submission.task,
                "system": submission.system,
                "status": "OK" if submission.ok and not submission.issues else "FAILED",
                "records": submission.records,
                "leaderboard": submission.leaderboard or "",
                "issues": "; ".join(submission.issues),
            }
            for submission in self.submissions
        ]
        lines = [
            "# Stable-ASR Submission Index",
            "",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- root: `{self.root}`",
            f"- submissions: `{len(self.submissions)}`",
            f"- valid_submissions: `{self.valid_submissions}`",
            f"- failed_submissions: `{self.failed_submissions}`",
            "",
            "## Submissions",
            "",
            dict_table(rows) if rows else "No submissions found.",
        ]
        if self.leaderboard_merge is not None:
            lines.extend(["", "## Leaderboard Merge", "", self.leaderboard_merge.to_markdown()])
        return "\n".join(lines) + "\n"


def build_turn_submission(
    *,
    dataset: str | Path,
    predictions: str | Path,
    output_dir: str | Path,
    system: str,
    dataset_format: str | None = None,
    allow_extra: bool = False,
    complete_threshold: float = 0.75,
    suite_path: str | Path | None = None,
) -> TurnSubmissionReport:
    """Build an auditable turn-prediction submission package."""

    if not system:
        raise ValueError("system must be a non-empty string")
    if dataset_format is not None and dataset_format not in TURN_FORMATS.names():
        raise ValueError(f"unknown dataset format: {dataset_format}")

    dataset = Path(dataset)
    predictions = Path(predictions)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_turn_records(dataset, format=dataset_format)
    schema_validation = validate_schema_file(
        predictions,
        schema_id="stable_asr.turn_prediction_record.v0",
    )
    prediction_validation = validate_turn_prediction_jsonl(
        records,
        predictions,
        allow_extra=allow_extra,
        dataset_path=dataset,
    )

    artifacts = TurnSubmissionArtifacts(
        output_dir=str(output_dir),
        manifest=str(output_dir / "submission.json"),
        summary_markdown=str(output_dir / "SUBMISSION.md"),
        schema_validation={
            "json": str(output_dir / "schema_validation.json"),
            "markdown": str(output_dir / "SCHEMA_VALIDATION.md"),
        },
        prediction_validation={
            "json": str(output_dir / "prediction_validation.json"),
            "markdown": str(output_dir / "PREDICTION_VALIDATION.md"),
        },
        evaluation={
            "json": str(output_dir / "turn_eval.json"),
            "markdown": str(output_dir / "TURN_EVAL.md"),
        },
        leaderboard={
            "jsonl": str(output_dir / "leaderboard.jsonl"),
        },
        leaderboard_validation={
            "json": str(output_dir / "leaderboard_validation.json"),
            "markdown": str(output_dir / "LEADERBOARD_VALIDATION.md"),
        },
    )

    _write_json(artifacts.schema_validation["json"], schema_validation.to_dict())
    _write_text(artifacts.schema_validation["markdown"], schema_validation.to_markdown())
    _write_json(artifacts.prediction_validation["json"], prediction_validation.to_dict())
    _write_text(artifacts.prediction_validation["markdown"], prediction_validation.to_markdown())

    evaluation: TurnEvalReport | None = None
    leaderboard_validation_ok: bool | None = None
    if schema_validation.ok and prediction_validation.ok:
        predictor = TurnPredictionManifestAdapter.from_jsonl(predictions)
        policy = TurnPolicy(TurnPolicyConfig(complete_threshold=complete_threshold))
        evaluation = evaluate_turn_records(records, predictor=predictor, policy=policy)
        _write_json(artifacts.evaluation["json"], evaluation.to_dict())
        _write_text(artifacts.evaluation["markdown"], evaluation.to_markdown())

        rows = _turn_quality_leaderboard_rows(
            evaluation,
            system=system,
            source=str(predictions),
        )
        _write_leaderboard_jsonl(artifacts.leaderboard["jsonl"], rows)
        suite = load_benchmark_suite(suite_path) if suite_path is not None else load_benchmark_suite()
        leaderboard_validation = validate_leaderboard_jsonl(artifacts.leaderboard["jsonl"], suite=suite)
        leaderboard_validation_ok = leaderboard_validation.ok
        _write_json(artifacts.leaderboard_validation["json"], leaderboard_validation.to_dict())
        _write_text(artifacts.leaderboard_validation["markdown"], leaderboard_validation.to_markdown())
    else:
        _write_json(artifacts.evaluation["json"], {"ok": False, "skipped": "validation_failed"})
        _write_text(artifacts.evaluation["markdown"], "# Stable-ASR Turn Evaluation\n\nSkipped because validation failed.\n")
        _write_text(artifacts.leaderboard["jsonl"], "")
        _write_json(artifacts.leaderboard_validation["json"], {"ok": False, "skipped": "validation_failed"})
        _write_text(
            artifacts.leaderboard_validation["markdown"],
            "# Stable-ASR Leaderboard Validation\n\nSkipped because validation failed.\n",
        )

    report = TurnSubmissionReport(
        ok=bool(schema_validation.ok and prediction_validation.ok and leaderboard_validation_ok),
        system=system,
        dataset=str(dataset),
        predictions=str(predictions),
        records=len(records),
        schema_validation=schema_validation,
        prediction_validation=prediction_validation,
        evaluation=evaluation,
        leaderboard_validation_ok=leaderboard_validation_ok,
        artifacts=artifacts,
    )
    _write_json(artifacts.manifest, report.to_dict())
    _write_text(artifacts.summary_markdown, report.to_markdown())
    return report


def index_submission_directory(
    root: str | Path,
    output_dir: str | Path,
    *,
    suite: dict[str, Any] | None = None,
    top_k: int = 3,
    require_known_systems: bool = False,
    require_known_slices: bool = False,
    require_complete_suite: bool = False,
) -> SubmissionIndexReport:
    """Index submission packages and produce a merged leaderboard report."""

    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = sorted(root.rglob("submission.json")) if root.exists() else []
    entries = [_load_submission_index_entry(path) for path in manifest_paths]
    leaderboard_inputs = [
        Path(entry.leaderboard)
        for entry in entries
        if entry.ok and not entry.issues and entry.leaderboard and Path(entry.leaderboard).exists()
    ]

    artifacts = SubmissionIndexArtifacts(
        output_dir=str(output_dir),
        index_json=str(output_dir / "submissions_index.json"),
        summary_markdown=str(output_dir / "SUBMISSIONS.md"),
        leaderboard=str(output_dir / "leaderboard.jsonl") if leaderboard_inputs else None,
        leaderboard_validation={
            "json": str(output_dir / "leaderboard_validation.json"),
            "markdown": str(output_dir / "LEADERBOARD_VALIDATION.md"),
        },
        leaderboard_report={
            "json": str(output_dir / "leaderboard_report.json"),
            "markdown": str(output_dir / "LEADERBOARD_REPORT.md"),
        },
    )

    leaderboard_merge: LeaderboardMergeReport | None = None
    if leaderboard_inputs:
        leaderboard_merge = merge_leaderboard_jsonl(
            leaderboard_inputs,
            output_dir / "leaderboard.jsonl",
            suite=suite,
            top_k=top_k,
            require_known_systems=require_known_systems,
            require_known_slices=require_known_slices,
            require_complete_suite=require_complete_suite,
        )
        _write_json(artifacts.leaderboard_validation["json"], leaderboard_merge.validation.to_dict())
        _write_text(artifacts.leaderboard_validation["markdown"], leaderboard_merge.validation.to_markdown())
        _write_json(artifacts.leaderboard_report["json"], leaderboard_merge.report.to_dict())
        _write_text(artifacts.leaderboard_report["markdown"], leaderboard_merge.report.to_markdown())
    else:
        _write_json(
            artifacts.leaderboard_validation["json"],
            {"ok": False, "skipped": "no valid leaderboard inputs"},
        )
        _write_text(
            artifacts.leaderboard_validation["markdown"],
            "# Stable-ASR Leaderboard Validation\n\nSkipped because no valid leaderboard inputs were found.\n",
        )
        _write_json(
            artifacts.leaderboard_report["json"],
            {"ok": False, "skipped": "no valid leaderboard inputs"},
        )
        _write_text(
            artifacts.leaderboard_report["markdown"],
            "# Stable-ASR Leaderboard Report\n\nSkipped because no valid leaderboard inputs were found.\n",
        )

    ok = bool(
        entries
        and leaderboard_merge is not None
        and leaderboard_merge.ok
        and all(entry.ok and not entry.issues for entry in entries)
    )
    report = SubmissionIndexReport(
        ok=ok,
        root=str(root),
        submissions=entries,
        leaderboard_merge=leaderboard_merge,
        artifacts=artifacts,
    )
    _write_json(artifacts.index_json, report.to_dict())
    _write_text(artifacts.summary_markdown, report.to_markdown())
    return report


def build_streaming_submission(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    system: str,
    slice_name: str = "submission",
    suite_path: str | Path | None = None,
) -> StreamingSubmissionReport:
    """Build an auditable streaming ASR submission package."""

    if not system:
        raise ValueError("system must be a non-empty string")
    if not slice_name:
        raise ValueError("slice_name must be a non-empty string")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_validation = validate_schema_file(
        input_path,
        schema_id="stable_asr.streaming_asr_record.v0",
    )
    artifacts = StreamingSubmissionArtifacts(
        output_dir=str(output_dir),
        manifest=str(output_dir / "submission.json"),
        summary_markdown=str(output_dir / "SUBMISSION.md"),
        schema_validation={
            "json": str(output_dir / "schema_validation.json"),
            "markdown": str(output_dir / "SCHEMA_VALIDATION.md"),
        },
        evaluation={
            "json": str(output_dir / "streaming_eval.json"),
            "markdown": str(output_dir / "STREAMING_EVAL.md"),
        },
        leaderboard={
            "jsonl": str(output_dir / "leaderboard.jsonl"),
        },
        leaderboard_validation={
            "json": str(output_dir / "leaderboard_validation.json"),
            "markdown": str(output_dir / "LEADERBOARD_VALIDATION.md"),
        },
    )
    _write_json(artifacts.schema_validation["json"], schema_validation.to_dict())
    _write_text(artifacts.schema_validation["markdown"], schema_validation.to_markdown())

    evaluation: StreamingASRReport | None = None
    leaderboard_validation_ok: bool | None = None
    records = schema_validation.records
    if schema_validation.ok:
        streaming_records = load_streaming_transcript_jsonl(input_path)
        records = len(streaming_records)
        evaluation = evaluate_streaming_records(streaming_records)
        _write_json(artifacts.evaluation["json"], evaluation.to_dict())
        _write_text(artifacts.evaluation["markdown"], _streaming_report_markdown(evaluation))

        rows = _streaming_leaderboard_rows(
            evaluation,
            system=system,
            slice_name=slice_name,
            source=str(input_path),
        )
        _write_leaderboard_jsonl(artifacts.leaderboard["jsonl"], rows)
        suite = load_benchmark_suite(suite_path) if suite_path is not None else load_benchmark_suite()
        leaderboard_validation = validate_leaderboard_jsonl(artifacts.leaderboard["jsonl"], suite=suite)
        leaderboard_validation_ok = leaderboard_validation.ok
        _write_json(artifacts.leaderboard_validation["json"], leaderboard_validation.to_dict())
        _write_text(artifacts.leaderboard_validation["markdown"], leaderboard_validation.to_markdown())
    else:
        _write_json(artifacts.evaluation["json"], {"ok": False, "skipped": "schema_validation_failed"})
        _write_text(
            artifacts.evaluation["markdown"],
            "# Stable-ASR Streaming ASR Evaluation\n\nSkipped because schema validation failed.\n",
        )
        _write_text(artifacts.leaderboard["jsonl"], "")
        _write_json(artifacts.leaderboard_validation["json"], {"ok": False, "skipped": "schema_validation_failed"})
        _write_text(
            artifacts.leaderboard_validation["markdown"],
            "# Stable-ASR Leaderboard Validation\n\nSkipped because schema validation failed.\n",
        )

    report = StreamingSubmissionReport(
        ok=bool(schema_validation.ok and leaderboard_validation_ok),
        system=system,
        input_path=str(input_path),
        slice_name=slice_name,
        records=records,
        schema_validation=schema_validation,
        evaluation=evaluation,
        leaderboard_validation_ok=leaderboard_validation_ok,
        artifacts=artifacts,
    )
    _write_json(artifacts.manifest, report.to_dict())
    _write_text(artifacts.summary_markdown, report.to_markdown())
    return report


def _turn_quality_leaderboard_rows(
    report: TurnEvalReport,
    *,
    system: str,
    source: str,
) -> list[LeaderboardRow]:
    return [
        LeaderboardRow(
            suite="stable_asr_v0",
            task="turn_quality",
            system=system,
            slice="overall",
            metric="accuracy",
            value=report.classification.accuracy,
            unit="rate",
            higher_is_better=True,
            source=source,
        ),
        LeaderboardRow(
            suite="stable_asr_v0",
            task="turn_quality",
            system=system,
            slice="overall",
            metric="macro_f1",
            value=report.classification.macro_f1,
            unit="rate",
            higher_is_better=True,
            source=source,
        ),
        LeaderboardRow(
            suite="stable_asr_v0",
            task="turn_quality",
            system=system,
            slice="overall",
            metric="false_complete_rate",
            value=report.interaction["false_complete_rate"],
            unit="rate",
            higher_is_better=False,
            source=source,
        ),
        LeaderboardRow(
            suite="stable_asr_v0",
            task="turn_quality",
            system=system,
            slice="overall",
            metric="missed_interrupt_rate",
            value=report.interaction["missed_interrupt_rate"],
            unit="rate",
            higher_is_better=False,
            source=source,
        ),
    ]


def _streaming_leaderboard_rows(
    report: StreamingASRReport,
    *,
    system: str,
    slice_name: str,
    source: str,
) -> list[LeaderboardRow]:
    specs = [
        ("wer", report.wer, "rate", False),
        ("cer", report.cer, "rate", False),
        ("rtf", report.rtf, "ratio", False),
        ("first_partial_latency", report.first_partial_latency, "s", False),
        ("final_latency", report.final_latency, "s", False),
        ("endpoint_delay", report.endpoint_delay, "s", False),
        ("partial_revision_rate", report.partial_revision_rate, "rate", False),
        ("stable_prefix_ratio", report.stable_prefix_ratio, "rate", True),
        ("timestamp_drift", report.timestamp_drift, "s", False),
    ]
    return [
        LeaderboardRow(
            suite="stable_asr_v0",
            task="streaming_asr",
            system=system,
            slice=slice_name,
            metric=metric,
            value=value,
            unit=unit,
            higher_is_better=higher_is_better,
            source=source,
        )
        for metric, value, unit, higher_is_better in specs
    ]


def _streaming_report_markdown(report: StreamingASRReport) -> str:
    return "\n".join(
        [
            "# Stable-ASR Streaming ASR Evaluation",
            "",
            f"- records: `{report.records}`",
            "",
            "## Metrics",
            "",
            dict_table(_streaming_metric_table(report)),
            "",
            "## Failure Analysis",
            "",
            report.failure_analysis.to_markdown(max_cases=20) or "- no failures",
            "",
        ]
    )


def _streaming_metric_table(report: StreamingASRReport | None) -> list[dict[str, str]]:
    if report is None:
        return []
    return [
        {"metric": "wer", "value": f"{report.wer:.6f}"},
        {"metric": "cer", "value": f"{report.cer:.6f}"},
        {"metric": "rtf", "value": f"{report.rtf:.6f}"},
        {"metric": "first_partial_latency", "value": f"{report.first_partial_latency:.6f}"},
        {"metric": "final_latency", "value": f"{report.final_latency:.6f}"},
        {"metric": "endpoint_delay", "value": f"{report.endpoint_delay:.6f}"},
        {"metric": "partial_revision_rate", "value": f"{report.partial_revision_rate:.6f}"},
        {"metric": "stable_prefix_ratio", "value": f"{report.stable_prefix_ratio:.6f}"},
        {"metric": "timestamp_drift", "value": f"{report.timestamp_drift:.6f}"},
    ]


def _write_leaderboard_jsonl(path: str | Path, rows: list[LeaderboardRow]) -> None:
    text = "\n".join(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) for row in rows)
    _write_text(path, text + ("\n" if text else ""))


def _load_submission_index_entry(manifest_path: Path) -> SubmissionIndexEntry:
    issues: list[str] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return SubmissionIndexEntry(
            manifest=str(manifest_path),
            version="unknown",
            task="unknown",
            system="unknown",
            ok=False,
            records=0,
            leaderboard=None,
            summary_markdown=None,
            issues=[str(exc)],
        )
    if not isinstance(payload, dict):
        issues.append("submission manifest must be a JSON object")
        payload = {}

    version = str(payload.get("version", "unknown"))
    task = _submission_task(version)
    system = str(payload.get("system", "unknown"))
    ok = bool(payload.get("ok"))
    records = _safe_int(payload.get("records"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        issues.append("missing artifacts object")
    leaderboard = _artifact_path(manifest_path, artifacts.get("leaderboard"), "jsonl")
    summary = _artifact_path(manifest_path, artifacts.get("summary_markdown"), None)
    if ok and not leaderboard:
        issues.append("missing leaderboard artifact")
    if leaderboard and not Path(leaderboard).exists():
        issues.append(f"leaderboard artifact not found: {leaderboard}")
    if not system or system == "unknown":
        issues.append("missing system")
    if task == "unknown":
        issues.append(f"unknown submission version: {version}")

    return SubmissionIndexEntry(
        manifest=str(manifest_path),
        version=version,
        task=task,
        system=system,
        ok=ok,
        records=records,
        leaderboard=leaderboard,
        summary_markdown=summary,
        issues=issues,
    )


def _submission_task(version: str) -> str:
    if version == TURN_SUBMISSION_VERSION:
        return "turn_quality"
    if version == STREAMING_SUBMISSION_VERSION:
        return "streaming_asr"
    return "unknown"


def _artifact_path(manifest_path: Path, payload: object, key: str | None) -> str | None:
    if isinstance(payload, dict) and key is not None:
        raw = payload.get(key)
    else:
        raw = payload
    if raw is None:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path)
    sibling = manifest_path.parent / path.name
    if sibling.exists():
        return str(sibling)
    return str(path)


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
