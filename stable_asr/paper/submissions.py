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
from stable_asr.models.adapters.prediction_audit import TurnPredictionValidationReport
from stable_asr.paper.leaderboard import LeaderboardRow, validate_leaderboard_jsonl
from stable_asr.paper.suites import load_benchmark_suite
from stable_asr.schema_validation import SchemaFileValidationReport, validate_schema_file
from stable_asr.turn.policy import TurnPolicy, TurnPolicyConfig


TURN_SUBMISSION_VERSION = "turn_submission_v0"


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


def _write_leaderboard_jsonl(path: str | Path, rows: list[LeaderboardRow]) -> None:
    text = "\n".join(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) for row in rows)
    _write_text(path, text + ("\n" if text else ""))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
