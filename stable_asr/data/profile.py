"""Turn manifest profiling for training and benchmark readiness checks."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.eval.report import dict_table
from stable_asr.turn.labels import ACTION_LABELS, TURN_LABELS


@dataclass(frozen=True)
class TurnDataProfile:
    records: int
    total_duration_sec: float
    duration_stats: dict[str, float]
    turn_labels: dict[str, int]
    action_labels: dict[str, int]
    scenarios: dict[str, int]
    languages: dict[str, int]
    sources: dict[str, int]
    assistant_speaking: dict[str, int]
    overlap: dict[str, int]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "ok": self.ok,
            "total_duration_sec": self.total_duration_sec,
            "duration_stats": self.duration_stats,
            "turn_labels": self.turn_labels,
            "action_labels": self.action_labels,
            "scenarios": self.scenarios,
            "languages": self.languages,
            "sources": self.sources,
            "assistant_speaking": self.assistant_speaking,
            "overlap": self.overlap,
            "warnings": self.warnings,
        }

    def to_text(self) -> str:
        lines = [
            "turn_data_profile:",
            f"- records: {self.records}",
            f"- total_duration_sec: {self.total_duration_sec:.3f}",
            f"- duration_mean_sec: {self.duration_stats.get('mean', 0.0):.3f}",
            f"- turn_labels: {json.dumps(self.turn_labels, ensure_ascii=False, sort_keys=True)}",
            f"- action_labels: {json.dumps(self.action_labels, ensure_ascii=False, sort_keys=True)}",
            f"- scenarios: {json.dumps(self.scenarios, ensure_ascii=False, sort_keys=True)}",
            f"- warnings: {len(self.warnings)}",
        ]
        lines.extend(f"  - {warning}" for warning in self.warnings)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            "# Stable-ASR Turn Data Profile",
            "",
            f"- records: `{self.records}`",
            f"- status: `{'OK' if self.ok else 'WARN'}`",
            f"- total_duration_sec: `{self.total_duration_sec:.3f}`",
            "",
            "## Duration",
            "",
            dict_table([{key: value for key, value in self.duration_stats.items()}]),
            "",
            "## Distributions",
            "",
            "### Turn Labels",
            "",
            dict_table(_count_rows(self.turn_labels)),
            "",
            "### Actions",
            "",
            dict_table(_count_rows(self.action_labels)),
            "",
            "### Scenarios",
            "",
            dict_table(_count_rows(self.scenarios)),
            "",
            "### Languages",
            "",
            dict_table(_count_rows(self.languages)),
            "",
            "## Warnings",
            "",
        ]
        if self.warnings:
            lines.extend(f"- {warning}" for warning in self.warnings)
        else:
            lines.append("- none")
        return "\n".join(lines).rstrip() + "\n"


def profile_turn_records(
    records: list[TurnManifestRecord],
    *,
    min_records: int = 1,
    warn_label_imbalance: float = 0.85,
    require_all_turn_labels: bool = False,
) -> TurnDataProfile:
    durations = [record.duration for record in records]
    turn_labels = _counts(record.turn_label for record in records)
    action_labels = _counts(record.action_label for record in records)
    scenarios = _counts(record.scenario or "" for record in records if record.scenario)
    languages = _counts(record.language for record in records)
    sources = _counts(record.source for record in records)
    assistant_speaking = _counts("true" if record.assistant_speaking else "false" for record in records)
    overlap = _counts("true" if record.overlap else "false" for record in records)
    duration_stats = _duration_stats(durations)
    warnings = _warnings(
        records,
        turn_labels=turn_labels,
        action_labels=action_labels,
        scenarios=scenarios,
        min_records=min_records,
        warn_label_imbalance=warn_label_imbalance,
        require_all_turn_labels=require_all_turn_labels,
    )
    return TurnDataProfile(
        records=len(records),
        total_duration_sec=round(sum(durations), 6),
        duration_stats=duration_stats,
        turn_labels=turn_labels,
        action_labels=action_labels,
        scenarios=scenarios,
        languages=languages,
        sources=sources,
        assistant_speaking=assistant_speaking,
        overlap=overlap,
        warnings=warnings,
    )


def _warnings(
    records: list[TurnManifestRecord],
    *,
    turn_labels: dict[str, int],
    action_labels: dict[str, int],
    scenarios: dict[str, int],
    min_records: int,
    warn_label_imbalance: float,
    require_all_turn_labels: bool,
) -> list[str]:
    warnings: list[str] = []
    if len(records) < min_records:
        warnings.append(f"record_count_below_minimum:{len(records)}<{min_records}")
    if require_all_turn_labels:
        missing = sorted(set(TURN_LABELS).difference(turn_labels))
        if missing:
            warnings.append("missing_turn_labels:" + ",".join(missing))
    unknown_turn_labels = sorted(label for label in turn_labels if label not in TURN_LABELS)
    if unknown_turn_labels:
        warnings.append("unknown_turn_labels:" + ",".join(unknown_turn_labels))
    unknown_actions = sorted(label for label in action_labels if label not in ACTION_LABELS)
    if unknown_actions:
        warnings.append("unknown_action_labels:" + ",".join(unknown_actions))
    if records and turn_labels:
        largest = max(turn_labels.values()) / len(records)
        if largest >= warn_label_imbalance and len(turn_labels) > 1:
            warnings.append(f"turn_label_imbalance:{largest:.3f}")
        if len(turn_labels) == 1:
            only_label = next(iter(turn_labels))
            warnings.append(f"single_turn_label:{only_label}")
    if not scenarios:
        warnings.append("missing_scenario_metadata")
    if any(record.duration <= 0 for record in records):
        warnings.append("non_positive_duration")
    short_records = sum(1 for record in records if record.duration < 0.2)
    if short_records:
        warnings.append(f"very_short_windows:{short_records}")
    return warnings


def _duration_stats(durations: list[float]) -> dict[str, float]:
    if not durations:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "min": round(min(durations), 6),
        "max": round(max(durations), 6),
        "mean": round(statistics.fmean(durations), 6),
        "median": round(statistics.median(durations), 6),
    }


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _count_rows(counts: dict[str, int]) -> list[dict[str, object]]:
    total = sum(counts.values())
    return [
        {
            "name": name,
            "count": count,
            "ratio": round(count / total, 6) if total else 0.0,
        }
        for name, count in counts.items()
    ]
