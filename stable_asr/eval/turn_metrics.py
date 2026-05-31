"""Small dependency-free turn classification metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationReport:
    labels: list[str]
    accuracy: float
    macro_f1: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    support: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": self.labels,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


def classification_report(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> ClassificationReport:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    support: dict[str, int] = {}

    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        support[label] = sum(t == label for t in y_true)
        precision[label] = _safe_div(tp, tp + fp)
        recall[label] = _safe_div(tp, tp + fn)
        f1[label] = _safe_div(2 * precision[label] * recall[label], precision[label] + recall[label])

    accuracy = _safe_div(sum(t == p for t, p in zip(y_true, y_pred)), len(y_true))
    macro_f1 = _safe_div(sum(f1.values()), len(labels))
    return ClassificationReport(
        labels=list(labels),
        accuracy=accuracy,
        macro_f1=macro_f1,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

