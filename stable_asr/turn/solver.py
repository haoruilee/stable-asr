"""Policy search utilities for turn-taking decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.eval.turn_eval import TurnEvalReport, TurnPredictor, evaluate_turn_records
from stable_asr.turn.policy import TurnPolicy, TurnPolicyConfig


DEFAULT_OBJECTIVE_WEIGHTS = {
    "false_complete_rate": 3.0,
    "missed_interrupt_rate": 5.0,
    "premature_response_rate": 3.0,
    "missed_complete_rate": 1.0,
}


@dataclass(frozen=True)
class PolicyTrial:
    config: TurnPolicyConfig
    score: float
    interaction: dict[str, float]
    accuracy: float
    macro_f1: float

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "score": self.score,
            "interaction": self.interaction,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
        }


@dataclass(frozen=True)
class ThresholdSearchResult:
    best: PolicyTrial
    trials: list[PolicyTrial]
    objective_weights: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "best": self.best.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
            "objective_weights": self.objective_weights,
        }


def threshold_search(
    records: list[TurnManifestRecord],
    predictor: TurnPredictor,
    *,
    objective_weights: dict[str, float] | None = None,
    complete_thresholds: list[float] | None = None,
    backchannel_thresholds: list[float] | None = None,
    wait_thresholds: list[float] | None = None,
    interrupt_thresholds: list[float] | None = None,
) -> ThresholdSearchResult:
    if not records:
        raise ValueError("records must not be empty")
    weights = objective_weights or DEFAULT_OBJECTIVE_WEIGHTS
    complete_thresholds = complete_thresholds or [0.50, 0.65, 0.75, 0.85]
    backchannel_thresholds = backchannel_thresholds or [0.50, 0.70]
    wait_thresholds = wait_thresholds or [0.50, 0.70]
    interrupt_thresholds = interrupt_thresholds or [0.50, 0.75, 0.90]

    trials: list[PolicyTrial] = []
    for complete, backchannel, wait, interrupt in product(
        complete_thresholds,
        backchannel_thresholds,
        wait_thresholds,
        interrupt_thresholds,
    ):
        config = TurnPolicyConfig(
            complete_threshold=complete,
            backchannel_threshold=backchannel,
            wait_threshold=wait,
            interrupt_min_confidence=interrupt,
        )
        report = evaluate_turn_records(records, predictor, policy=TurnPolicy(config))
        trials.append(_trial(config, report, weights))

    best = min(trials, key=lambda trial: (trial.score, -trial.macro_f1))
    return ThresholdSearchResult(best=best, trials=trials, objective_weights=dict(weights))


def score_report(report: TurnEvalReport, weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_OBJECTIVE_WEIGHTS
    return sum(report.interaction.get(metric, 0.0) * weight for metric, weight in weights.items())


def _trial(
    config: TurnPolicyConfig,
    report: TurnEvalReport,
    weights: dict[str, float],
) -> PolicyTrial:
    return PolicyTrial(
        config=config,
        score=score_report(report, weights),
        interaction=report.interaction,
        accuracy=report.classification.accuracy,
        macro_f1=report.classification.macro_f1,
    )

