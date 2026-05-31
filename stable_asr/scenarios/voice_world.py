"""VoiceWorld scenario evaluation for turn-taking policies."""

from __future__ import annotations

from dataclasses import dataclass

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.eval.report import MarkdownReport, dict_table
from stable_asr.eval.turn_eval import TurnEvalReport, TurnPredictor, evaluate_turn_records
from stable_asr.scenarios.synthetic_turn import generate_synthetic_turn_records
from stable_asr.turn.policy import TurnPolicy, TurnPolicyConfig


@dataclass(frozen=True)
class ScenarioEvalReport:
    suite: str
    seed: int
    overall: TurnEvalReport
    by_scenario: dict[str, TurnEvalReport]
    factor_summary: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "seed": self.seed,
            "overall": self.overall.to_dict(),
            "by_scenario": {
                scenario: report.to_dict() for scenario, report in self.by_scenario.items()
            },
            "factor_summary": self.factor_summary,
        }

    def to_markdown(self) -> str:
        report = MarkdownReport("Stable-ASR VoiceWorld Scenario Evaluation")
        report.add_section(
            "Summary",
            "\n".join(
                [
                    f"- suite: {self.suite}",
                    f"- seed: {self.seed}",
                    f"- records: {len(self.overall.examples)}",
                    f"- accuracy: {self.overall.classification.accuracy:.4f}",
                    f"- macro_f1: {self.overall.classification.macro_f1:.4f}",
                ]
            ),
        )
        report.add_section("Scenario Breakdown", dict_table(_scenario_rows(self.by_scenario)))
        report.add_section("Factors", _factor_markdown(self.factor_summary))
        return report.to_markdown()


def evaluate_voice_world(
    predictor: TurnPredictor,
    *,
    episodes: int = 25,
    seed: int = 0,
    suite: str = "zh_turn_mini_v0",
    policy_config: TurnPolicyConfig | None = None,
) -> ScenarioEvalReport:
    records = generate_synthetic_turn_records(episodes=episodes, seed=seed)
    return evaluate_voice_world_records(
        records,
        predictor,
        seed=seed,
        suite=suite,
        policy_config=policy_config,
    )


def evaluate_voice_world_records(
    records: list[TurnManifestRecord],
    predictor: TurnPredictor,
    *,
    seed: int = 0,
    suite: str = "custom",
    policy_config: TurnPolicyConfig | None = None,
) -> ScenarioEvalReport:
    if not records:
        raise ValueError("records must not be empty")

    config = policy_config or TurnPolicyConfig()
    overall = evaluate_turn_records(records, predictor, policy=TurnPolicy(config))
    grouped: dict[str, list[TurnManifestRecord]] = {}
    for record in records:
        grouped.setdefault(record.scenario or "unknown", []).append(record)
    by_scenario = {
        scenario: evaluate_turn_records(group, predictor, policy=TurnPolicy(config))
        for scenario, group in sorted(grouped.items())
    }
    return ScenarioEvalReport(
        suite=suite,
        seed=seed,
        overall=overall,
        by_scenario=by_scenario,
        factor_summary=_factor_summary(records),
    )


def _scenario_rows(by_scenario: dict[str, TurnEvalReport]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario, report in by_scenario.items():
        rows.append(
            {
                "scenario": scenario,
                "records": len(report.examples),
                "accuracy": f"{report.classification.accuracy:.4f}",
                "macro_f1": f"{report.classification.macro_f1:.4f}",
                "false_complete_rate": f"{report.interaction['false_complete_rate']:.4f}",
                "premature_response_rate": f"{report.interaction['premature_response_rate']:.4f}",
                "missed_interrupt_rate": f"{report.interaction['missed_interrupt_rate']:.4f}",
            }
        )
    return rows


def _factor_summary(records: list[TurnManifestRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    keys = (
        "snr_db",
        "reverb",
        "speaking_rate",
        "overlap_offset_ms",
        "network_jitter_ms",
        "farfield_distance_m",
        "code_switch_ratio",
        "accent",
    )
    for record in records:
        for key in keys:
            value = str(record.metadata.get(key, "unknown"))
            summary.setdefault(key, {})
            summary[key][value] = summary[key].get(value, 0) + 1
    return {key: dict(sorted(values.items())) for key, values in sorted(summary.items())}


def _factor_markdown(summary: dict[str, dict[str, int]]) -> str:
    sections = []
    for factor, values in summary.items():
        rows = [{"value": value, "count": count} for value, count in values.items()]
        sections.append(f"### {factor}\n\n{dict_table(rows)}")
    return "\n\n".join(sections)
