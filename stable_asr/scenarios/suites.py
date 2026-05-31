"""Machine-readable VoiceWorld scenario suite definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.scenarios.synthetic_turn import SCENARIO_NAMES
from stable_asr.turn.labels import ACTION_LABELS, TURN_LABELS


DEFAULT_SCENARIO_SUITE: dict[str, Any] = {
    "id": "stable_asr_voiceworld_v0",
    "version": "0.1.0",
    "title": "Stable-ASR VoiceWorld v0 Scenario Suite",
    "description": (
        "Seedable full-duplex turn-taking and streaming-ASR-control scenarios "
        "for incomplete pauses, backchannels, wait/hold commands, interruptions, "
        "side speech, ambient speech, noisy far-field speech, and code-switching."
    ),
    "generator": "stable_asr.scenarios.synthetic_turn.generate_synthetic_turn_records",
    "seed_protocol": "Scenario id is selected by episode_index % len(scenarios); factors are sampled from a local random.Random(seed).",
    "sample_rate": 16000,
    "scenarios": [
        {
            "id": "normal_question",
            "title": "Normal User Question",
            "turn_label": "complete",
            "action_label": "take_turn",
            "assistant_speaking": False,
            "overlap": False,
            "expected_behavior": "assistant should take the turn after the user finishes",
        },
        {
            "id": "incomplete_pause",
            "title": "Incomplete Mid-Utterance Pause",
            "turn_label": "incomplete",
            "action_label": "keep_listening",
            "assistant_speaking": False,
            "overlap": False,
            "expected_behavior": "assistant should keep listening and avoid premature response",
        },
        {
            "id": "backchannel",
            "title": "Listener Backchannel",
            "turn_label": "backchannel",
            "action_label": "continue_speaking",
            "assistant_speaking": True,
            "overlap": True,
            "expected_behavior": "assistant should treat short acknowledgement as flow-preserving",
        },
        {
            "id": "wait_stop",
            "title": "Wait Or Hold Command",
            "turn_label": "wait",
            "action_label": "hold",
            "assistant_speaking": False,
            "overlap": False,
            "expected_behavior": "assistant should hold instead of answering immediately",
        },
        {
            "id": "user_interruption",
            "title": "User Interruption",
            "turn_label": "complete",
            "action_label": "stop_tts_and_listen",
            "assistant_speaking": True,
            "overlap": True,
            "expected_behavior": "assistant should stop speaking and listen to the user",
        },
        {
            "id": "side_conversation",
            "title": "Side Conversation",
            "turn_label": "wait",
            "action_label": "ignore",
            "assistant_speaking": False,
            "overlap": True,
            "expected_behavior": "assistant should ignore speech not addressed to it",
        },
        {
            "id": "ambient_speech",
            "title": "Ambient Speech",
            "turn_label": "wait",
            "action_label": "ignore",
            "assistant_speaking": False,
            "overlap": True,
            "expected_behavior": "assistant should reject background speech",
        },
        {
            "id": "noisy_farfield",
            "title": "Noisy Far-Field User Speech",
            "turn_label": "complete",
            "action_label": "take_turn",
            "assistant_speaking": False,
            "overlap": False,
            "expected_behavior": "assistant should still take the turn under noise and distance",
        },
        {
            "id": "code_switching",
            "title": "Chinese-English Code Switching",
            "turn_label": "complete",
            "action_label": "take_turn",
            "assistant_speaking": False,
            "overlap": False,
            "expected_behavior": "assistant should handle mixed-language completion",
        },
    ],
    "factors": [
        {"name": "pause_ms", "type": "integer_range", "values": "scenario_specific"},
        {"name": "vad_pause_ms", "type": "integer_range", "values": "pause_ms +/- 40"},
        {"name": "duration_ms", "type": "integer_range", "values": "scenario_specific"},
        {"name": "snr_db", "type": "choice", "values": [-12, -8, -5, 0, 5, 10, 20]},
        {"name": "reverb", "type": "choice", "values": ["none", "small_room", "large_room", "hallway"]},
        {"name": "speaking_rate", "type": "choice", "values": [0.8, 1.0, 1.2, 1.5]},
        {"name": "overlap_offset_ms", "type": "choice", "values": [0, 100, 300, 500]},
        {"name": "network_jitter_ms", "type": "choice", "values": [0, 50, 100, 300]},
        {"name": "farfield_distance_m", "type": "choice", "values": [0.5, 2.0, 3.5, 5.0]},
        {"name": "code_switch_ratio", "type": "choice", "values": [0.0, 0.25, 0.4, 0.55]},
        {"name": "accent", "type": "choice", "values": ["standard", "regional", "non_native"]},
    ],
    "metrics": [
        "accuracy",
        "macro_f1",
        "false_complete_rate",
        "premature_response_rate",
        "missed_interrupt_rate",
        "failure_case_mining",
    ],
}


@dataclass(frozen=True)
class ScenarioSuiteValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "scenario_suite: OK"
        return "scenario_suite: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


def load_scenario_suite(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_SCENARIO_SUITE))
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("scenario suite must be a JSON object")
    return payload


def write_scenario_suite_json(path: str | Path, suite: dict[str, Any] | None = None) -> str:
    suite = suite or load_scenario_suite()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_scenario_suite(suite: dict[str, Any]) -> ScenarioSuiteValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "scenarios", "factors", "metrics"):
        if key not in suite:
            errors.append(f"missing top-level key: {key}")

    scenarios = suite.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        errors.append("scenarios must be a non-empty list")
        return ScenarioSuiteValidation(ok=False, errors=errors)

    seen: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            errors.append(f"scenario {index} must be an object")
            continue
        scenario_id = scenario.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            errors.append(f"scenario {index} missing id")
        elif scenario_id in seen:
            errors.append(f"duplicate scenario id: {scenario_id}")
        else:
            seen.add(scenario_id)
        for key in ("title", "turn_label", "action_label", "expected_behavior"):
            if key not in scenario:
                errors.append(f"scenario {scenario_id or index} missing {key}")
        if scenario.get("turn_label") not in TURN_LABELS:
            errors.append(f"scenario {scenario_id or index} has invalid turn_label")
        if scenario.get("action_label") not in ACTION_LABELS:
            errors.append(f"scenario {scenario_id or index} has invalid action_label")

    missing_generated = sorted(set(SCENARIO_NAMES).difference(seen))
    if missing_generated:
        errors.append("missing generated scenario(s): " + ", ".join(missing_generated))

    factors = suite.get("factors")
    if not isinstance(factors, list) or not factors:
        errors.append("factors must be a non-empty list")
    else:
        factor_names = set()
        for index, factor in enumerate(factors):
            if not isinstance(factor, dict):
                errors.append(f"factor {index} must be an object")
                continue
            name = factor.get("name")
            if not isinstance(name, str) or not name:
                errors.append(f"factor {index} missing name")
            elif name in factor_names:
                errors.append(f"duplicate factor name: {name}")
            else:
                factor_names.add(name)
            for key in ("type", "values"):
                if key not in factor:
                    errors.append(f"factor {name or index} missing {key}")

    metrics = suite.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append("metrics must be a non-empty list")
    return ScenarioSuiteValidation(ok=not errors, errors=errors)


def scenario_suite_markdown(suite: dict[str, Any]) -> str:
    validation = validate_scenario_suite(suite)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    lines = [
        f"# {suite['title']}",
        "",
        f"- id: `{suite['id']}`",
        f"- version: `{suite['version']}`",
        f"- generator: `{suite.get('generator', '')}`",
        "",
        str(suite.get("description", "")),
        "",
        "## Scenarios",
        "",
        dict_table(_scenario_rows(suite)),
        "",
        "## Factors",
        "",
        dict_table(_factor_rows(suite)),
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- `{metric}`" for metric in suite.get("metrics", []))
    lines.append("")
    return "\n".join(lines)


def _scenario_rows(suite: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "scenario": item["id"],
            "turn_label": item["turn_label"],
            "action_label": item["action_label"],
            "assistant_speaking": item.get("assistant_speaking", False),
            "overlap": item.get("overlap", False),
            "expected_behavior": item["expected_behavior"],
        }
        for item in suite["scenarios"]
    ]


def _factor_rows(suite: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in suite["factors"]:
        values = item.get("values", "")
        rows.append(
            {
                "factor": item["name"],
                "type": item["type"],
                "values": ", ".join(str(value) for value in values) if isinstance(values, list) else values,
            }
        )
    return rows
