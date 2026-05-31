"""Compare multiple turn predictors on the same manifest."""

from __future__ import annotations

from dataclasses import dataclass

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.eval.report import MarkdownReport, dict_table
from stable_asr.eval.turn_eval import TurnEvalReport, TurnPredictor, evaluate_turn_records
from stable_asr.turn.policy import TurnPolicy


@dataclass(frozen=True)
class TurnComparisonRow:
    name: str
    kind: str
    records: int
    accuracy: float
    macro_f1: float
    false_complete_rate: float
    premature_response_rate: float
    missed_interrupt_rate: float
    missed_complete_rate: float
    failures: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "records": self.records,
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
            "false_complete_rate": round(self.false_complete_rate, 6),
            "premature_response_rate": round(self.premature_response_rate, 6),
            "missed_interrupt_rate": round(self.missed_interrupt_rate, 6),
            "missed_complete_rate": round(self.missed_complete_rate, 6),
            "failures": self.failures,
        }


@dataclass(frozen=True)
class TurnComparisonReport:
    dataset: str
    rows: list[TurnComparisonRow]
    reports: dict[str, TurnEvalReport]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "rows": [row.to_dict() for row in self.rows],
            "reports": {name: report.to_dict() for name, report in self.reports.items()},
        }

    def to_markdown(self) -> str:
        report = MarkdownReport("Stable-ASR Turn Comparison")
        report.add_section("Dataset", f"- dataset: `{self.dataset}`\n- systems: `{len(self.rows)}`")
        report.add_section("Leaderboard", dict_table([row.to_dict() for row in self.rows]))
        for row in self.rows:
            eval_report = self.reports[row.name]
            report.add_section(
                f"Failure Summary: {row.name}",
                eval_report.failure_analysis.to_markdown(max_cases=10) or "- no failures",
            )
        return report.to_markdown()


def compare_turn_predictors(
    records: list[TurnManifestRecord],
    predictors: list[tuple[str, str, TurnPredictor]],
    *,
    dataset: str,
    policy: TurnPolicy | None = None,
) -> TurnComparisonReport:
    if not records:
        raise ValueError("records must not be empty")
    if not predictors:
        raise ValueError("at least one predictor is required")

    rows: list[TurnComparisonRow] = []
    reports: dict[str, TurnEvalReport] = {}
    seen: set[str] = set()
    for name, kind, predictor in predictors:
        if name in seen:
            raise ValueError(f"duplicate predictor name: {name}")
        seen.add(name)
        eval_report = evaluate_turn_records(records, predictor=predictor, policy=policy)
        reports[name] = eval_report
        rows.append(
            TurnComparisonRow(
                name=name,
                kind=kind,
                records=len(records),
                accuracy=eval_report.classification.accuracy,
                macro_f1=eval_report.classification.macro_f1,
                false_complete_rate=eval_report.interaction["false_complete_rate"],
                premature_response_rate=eval_report.interaction["premature_response_rate"],
                missed_interrupt_rate=eval_report.interaction["missed_interrupt_rate"],
                missed_complete_rate=eval_report.interaction["missed_complete_rate"],
                failures=eval_report.failure_analysis.total_failures,
            )
        )
    rows.sort(key=lambda row: (-row.macro_f1, row.false_complete_rate, row.name))
    return TurnComparisonReport(dataset=dataset, rows=rows, reports=reports)


@dataclass(frozen=True)
class TurnSplitComparisonReport:
    splits: dict[str, TurnComparisonReport]

    def to_dict(self) -> dict[str, object]:
        return {
            "splits": {name: report.to_dict() for name, report in self.splits.items()},
            "rows": self.rows(),
        }

    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for split_name, report in self.splits.items():
            for row in report.rows:
                payload = row.to_dict()
                payload["split"] = split_name
                rows.append(payload)
        return sorted(rows, key=lambda row: (str(row["name"]), str(row["split"])))

    def to_markdown(self) -> str:
        report = MarkdownReport("Stable-ASR Turn Split Comparison")
        report.add_section("Leaderboard", dict_table(self.rows()))
        for split_name, split_report in self.splits.items():
            report.add_section(
                f"Split: {split_name}",
                dict_table([row.to_dict() for row in split_report.rows]),
            )
        return report.to_markdown()


def compare_turn_predictors_on_splits(
    split_records: dict[str, list[TurnManifestRecord]],
    predictors: list[tuple[str, str, TurnPredictor]],
    *,
    policy: TurnPolicy | None = None,
) -> TurnSplitComparisonReport:
    if not split_records:
        raise ValueError("split_records must not be empty")
    reports = {
        split_name: compare_turn_predictors(
            records,
            predictors,
            dataset=split_name,
            policy=policy,
        )
        for split_name, records in split_records.items()
    }
    return TurnSplitComparisonReport(splits=reports)
