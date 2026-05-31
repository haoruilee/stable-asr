"""Turn-taking evaluation loop and report objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.eval.failures import TurnFailureSummary, mine_turn_failures
from stable_asr.eval.report import MarkdownReport, dict_table
from stable_asr.eval.turn_metrics import ClassificationReport, classification_report
from stable_asr.turn.labels import TURN_LABELS
from stable_asr.turn.policy import TurnPolicy
from stable_asr.turn.types import TurnPrediction


class TurnPredictor(Protocol):
    def predict(self, record: TurnManifestRecord) -> TurnPrediction:
        ...


@dataclass(frozen=True)
class TurnEvalExample:
    id: str
    true_label: str
    pred_label: str
    true_action: str
    pred_action: str
    confidence: float
    scenario: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "true_label": self.true_label,
            "pred_label": self.pred_label,
            "true_action": self.true_action,
            "pred_action": self.pred_action,
            "confidence": round(self.confidence, 6),
            "scenario": self.scenario,
        }


@dataclass(frozen=True)
class TurnEvalReport:
    classification: ClassificationReport
    interaction: dict[str, float]
    examples: list[TurnEvalExample]
    failure_analysis: TurnFailureSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification.to_dict(),
            "interaction": self.interaction,
            "examples": [example.to_dict() for example in self.examples],
            "failure_analysis": self.failure_analysis.to_dict(),
        }

    def to_markdown(self) -> str:
        report = MarkdownReport("Stable-ASR Turn Evaluation")
        report.add_section(
            "Summary",
            "\n".join(
                [
                    f"- accuracy: {self.classification.accuracy:.4f}",
                    f"- macro_f1: {self.classification.macro_f1:.4f}",
                    f"- false_complete_rate: {self.interaction['false_complete_rate']:.4f}",
                    f"- premature_response_rate: {self.interaction['premature_response_rate']:.4f}",
                    f"- missed_interrupt_rate: {self.interaction['missed_interrupt_rate']:.4f}",
                ]
            ),
        )
        report.add_section(
            "Per-Label Metrics",
            dict_table(
                [
                    {
                        "label": label,
                        "precision": f"{self.classification.precision[label]:.4f}",
                        "recall": f"{self.classification.recall[label]:.4f}",
                        "f1": f"{self.classification.f1[label]:.4f}",
                        "support": self.classification.support[label],
                    }
                    for label in self.classification.labels
                ]
            ),
        )
        report.add_section(
            "Examples",
            dict_table([example.to_dict() for example in self.examples[:20]]),
        )
        failure_markdown = self.failure_analysis.to_markdown(max_cases=20)
        if failure_markdown:
            report.add_section("Failure Analysis", failure_markdown)
        return report.to_markdown()


def evaluate_turn_records(
    records: list[TurnManifestRecord],
    predictor: TurnPredictor,
    policy: TurnPolicy | None = None,
    labels: list[str] | None = None,
) -> TurnEvalReport:
    if labels is None:
        labels = sorted(TURN_LABELS)
    if policy is None:
        policy = TurnPolicy()

    y_true: list[str] = []
    y_pred: list[str] = []
    examples: list[TurnEvalExample] = []

    for record in records:
        prediction = predictor.predict(record)
        pred_label = prediction.label
        action = policy.decide(prediction, assistant_speaking=record.assistant_speaking)

        y_true.append(record.turn_label)
        y_pred.append(pred_label)
        examples.append(
            TurnEvalExample(
                id=record.id,
                true_label=record.turn_label,
                pred_label=pred_label,
                true_action=record.action_label,
                pred_action=action.action,
                confidence=prediction.confidence,
                scenario=record.scenario,
            )
        )

    return TurnEvalReport(
        classification=classification_report(y_true, y_pred, labels=labels),
        interaction=interaction_metrics(examples),
        examples=examples,
        failure_analysis=mine_turn_failures(examples),
    )


def interaction_metrics(examples: list[TurnEvalExample]) -> dict[str, float]:
    non_complete = [example for example in examples if example.true_label != "complete"]
    complete = [example for example in examples if example.true_label == "complete"]
    interruptions = [
        example for example in examples if example.true_action == "stop_tts_and_listen"
    ]

    false_complete = [
        example for example in non_complete if example.pred_label == "complete"
    ]
    premature_responses = [
        example for example in non_complete if example.pred_action == "take_turn"
    ]
    missed_interrupts = [
        example
        for example in interruptions
        if example.pred_action != "stop_tts_and_listen"
    ]
    missed_completes = [
        example
        for example in complete
        if example.pred_action not in {"take_turn", "stop_tts_and_listen"}
    ]

    return {
        "false_complete_rate": _safe_div(len(false_complete), len(non_complete)),
        "premature_response_rate": _safe_div(len(premature_responses), len(non_complete)),
        "missed_interrupt_rate": _safe_div(len(missed_interrupts), len(interruptions)),
        "missed_complete_rate": _safe_div(len(missed_completes), len(complete)),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
