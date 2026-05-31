"""Failure mining utilities for turn-taking evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from stable_asr.eval.report import dict_table


@dataclass(frozen=True)
class TurnFailureCase:
    id: str
    category: str
    severity: int
    true_label: str
    pred_label: str
    true_action: str
    pred_action: str
    confidence: float
    scenario: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "category": self.category,
            "severity": self.severity,
            "true_label": self.true_label,
            "pred_label": self.pred_label,
            "true_action": self.true_action,
            "pred_action": self.pred_action,
            "confidence": round(self.confidence, 6),
            "scenario": self.scenario,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TurnFailureSummary:
    total_failures: int
    category_counts: dict[str, int]
    scenario_counts: dict[str, int]
    cases: list[TurnFailureCase]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_failures": self.total_failures,
            "category_counts": self.category_counts,
            "scenario_counts": self.scenario_counts,
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_markdown(self, *, max_cases: int = 20) -> str:
        sections = []
        if self.category_counts:
            sections.append(
                "### Failure Taxonomy\n\n"
                + dict_table(
                    [
                        {"category": category, "count": count}
                        for category, count in self.category_counts.items()
                    ]
                )
            )
        if self.scenario_counts:
            sections.append(
                "### Failure Scenarios\n\n"
                + dict_table(
                    [
                        {"scenario": scenario, "count": count}
                        for scenario, count in self.scenario_counts.items()
                    ]
                )
            )
        if self.cases:
            sections.append(
                "### Representative Failures\n\n"
                + dict_table([case.to_dict() for case in self.cases[:max_cases]])
            )
        return "\n\n".join(sections)


def mine_turn_failures(
    examples: Iterable[Any],
    *,
    max_cases: int = 50,
) -> TurnFailureSummary:
    """Categorize interaction-level failures from turn evaluation examples."""

    failures: list[TurnFailureCase] = []
    for example in examples:
        case = _failure_case(example)
        if case is not None:
            failures.append(case)

    failures.sort(key=lambda case: (-case.severity, -case.confidence, case.category, case.id))
    category_counts = _counts(case.category for case in failures)
    scenario_counts = _counts(case.scenario or "unknown" for case in failures)
    return TurnFailureSummary(
        total_failures=len(failures),
        category_counts=category_counts,
        scenario_counts=scenario_counts,
        cases=failures[:max_cases],
    )


def _failure_case(example: Any) -> TurnFailureCase | None:
    true_label = str(getattr(example, "true_label"))
    pred_label = str(getattr(example, "pred_label"))
    true_action = str(getattr(example, "true_action"))
    pred_action = str(getattr(example, "pred_action"))

    category: str | None = None
    severity = 1
    reason = "predicted turn label differs from reference"

    if true_action == "stop_tts_and_listen" and pred_action != "stop_tts_and_listen":
        category = "missed_interrupt"
        severity = 5
        reason = "user interruption was not converted into a stop-and-listen action"
    elif true_label != "complete" and pred_label == "complete":
        category = "false_complete"
        severity = 4
        reason = "non-complete user state was classified as complete"
    elif true_label != "complete" and pred_action == "take_turn":
        category = "premature_response"
        severity = 4
        reason = "policy would make the assistant take the turn before user completion"
    elif true_label == "backchannel" and pred_action != "continue_speaking":
        category = "backchannel_break"
        severity = 3
        reason = "listener backchannel did not preserve assistant speaking flow"
    elif true_label == "wait" and pred_action not in {"hold", "keep_listening"}:
        category = "wait_violation"
        severity = 3
        reason = "wait/hold user intent was not preserved"
    elif true_label == "complete" and pred_action not in {"take_turn", "stop_tts_and_listen"}:
        category = "missed_complete"
        severity = 2
        reason = "complete user turn did not lead to a response action"
    elif true_label != pred_label:
        category = "classification_error"
        severity = 1

    if category is None:
        return None
    return TurnFailureCase(
        id=str(getattr(example, "id")),
        category=category,
        severity=severity,
        true_label=true_label,
        pred_label=pred_label,
        true_action=true_action,
        pred_action=pred_action,
        confidence=float(getattr(example, "confidence")),
        scenario=getattr(example, "scenario"),
        reason=reason,
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
