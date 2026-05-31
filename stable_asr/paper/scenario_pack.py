"""Build self-contained VoiceWorld scenario starter packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stable_asr.data.manifest import load_manifest
from stable_asr.data.recipes import prepare_voiceworld_manifest
from stable_asr.eval.report import dict_table
from stable_asr.scenarios.suites import (
    load_scenario_suite,
    scenario_suite_markdown,
    validate_scenario_suite,
    write_scenario_suite_json,
)


SCENARIO_PACK_VERSION = "scenario_pack_v0"


@dataclass(frozen=True)
class ScenarioPackReport:
    output_dir: str
    files: dict[str, str]
    commands: list[str]
    scenario_suite_ok: bool
    sample_manifest_ok: bool
    sample_records: int

    @property
    def ok(self) -> bool:
        return self.scenario_suite_ok and self.sample_manifest_ok and self.sample_records > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "version": SCENARIO_PACK_VERSION,
            "output_dir": self.output_dir,
            "files": self.files,
            "commands": self.commands,
            "scenario_suite_ok": self.scenario_suite_ok,
            "sample_manifest_ok": self.sample_manifest_ok,
            "sample_records": self.sample_records,
        }

    def to_markdown(self) -> str:
        file_rows = [{"name": name, "path": path} for name, path in sorted(self.files.items())]
        status_rows = [
            {"check": "scenario_suite", "ok": _yes_no(self.scenario_suite_ok)},
            {"check": "sample_manifest", "ok": _yes_no(self.sample_manifest_ok)},
            {"check": "sample_records", "ok": str(self.sample_records)},
        ]
        return "\n".join(
            [
                "# Stable-ASR VoiceWorld Scenario Pack",
                "",
                f"- status: `{'OK' if self.ok else 'FAILED'}`",
                f"- version: `{SCENARIO_PACK_VERSION}`",
                f"- output_dir: `{self.output_dir}`",
                "",
                "## Included Files",
                "",
                dict_table(file_rows),
                "",
                "## Readiness Checks",
                "",
                dict_table(status_rows),
                "",
                "## Run From This Directory",
                "",
                "```bash",
                "\n".join(self.commands),
                "```",
                "",
            ]
        )


def build_scenario_pack(
    output_dir: str | Path,
    *,
    suite_path: str | Path | None = None,
) -> ScenarioPackReport:
    """Write a starter pack for contributing and evaluating VoiceWorld scenarios."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = load_scenario_suite(suite_path)
    suite_validation = validate_scenario_suite(suite)
    if not suite_validation.ok:
        raise ValueError(suite_validation.to_text())

    files: dict[str, str] = {}
    _write_json(output_dir / "manifest.json", {"version": SCENARIO_PACK_VERSION, "status": "building"})
    files["scenario_suite_json"] = write_scenario_suite_json(output_dir / "configs" / "scenario_suite.json", suite)
    files["scenario_suite_markdown"] = _write_text(
        output_dir / "configs" / "SCENARIO_SUITE.md",
        scenario_suite_markdown(suite),
    )
    files["metadata"] = _write_text(output_dir / "data" / "voiceworld_metadata.tsv", _metadata_tsv())
    files["data_readme"] = _write_text(output_dir / "data" / "README.md", _data_readme())
    files["sample_manifest"] = str(output_dir / "data" / "voiceworld_manifest.jsonl")
    prepare_voiceworld_manifest(
        output_dir / "data" / "voiceworld_metadata.tsv",
        output_dir / "data" / "voiceworld_manifest.jsonl",
        default_language="zh",
        default_source="scenario_pack_demo",
    )
    commands = _scenario_commands()
    files["commands_markdown"] = _write_text(output_dir / "COMMANDS.md", _commands_markdown(commands))
    files["commands_script"] = _write_text(output_dir / "commands.sh", _commands_script(commands))

    records = load_manifest(output_dir / "data" / "voiceworld_manifest.jsonl")
    report = ScenarioPackReport(
        output_dir=str(output_dir),
        files=files,
        commands=commands,
        scenario_suite_ok=suite_validation.ok,
        sample_manifest_ok=bool(records),
        sample_records=len(records),
    )
    files["readme"] = _write_text(output_dir / "README.md", report.to_markdown())
    _write_json(output_dir / "manifest.json", report.to_dict())
    return report


def _scenario_commands() -> list[str]:
    return [
        "stable-asr scenario-suite --suite configs/scenario_suite.json --validate-only",
        "stable-asr prepare-voiceworld --input data/voiceworld_metadata.tsv --output runs/voiceworld_manifest.jsonl --language zh --source scenario_pack_demo",
        "stable-asr validate-manifest runs/voiceworld_manifest.jsonl",
        "stable-asr profile-turn-data --dataset runs/voiceworld_manifest.jsonl --report reports/voiceworld_profile.md",
        "stable-asr eval-scenario --dataset runs/voiceworld_manifest.jsonl --baseline vad_pause --report reports/voiceworld_eval.md --json-output reports/voiceworld_eval.json",
    ]


def _metadata_tsv() -> str:
    header = [
        "id",
        "audio",
        "text",
        "scenario",
        "turn_label",
        "action_label",
        "assistant_speaking",
        "overlap",
        "start_ms",
        "duration_ms",
        "language",
        "source",
        "snr_db",
        "reverb",
        "speaking_rate",
        "overlap_offset_ms",
        "network_jitter_ms",
        "farfield_distance_m",
        "code_switch_ratio",
        "accent",
    ]
    rows = [
        ("vw_pack_001", "audio/normal_question.wav", "我想问一下今天的天气", "normal_question", "complete", "take_turn", "false", "false", 0, 1600, "zh", "scenario_pack_demo", 20, "none", 1.0, 0, 0, 0.8, 0.0, "standard"),
        ("vw_pack_002", "audio/incomplete_pause.wav", "我想问一下今天北京", "incomplete_pause", "incomplete", "keep_listening", "false", "false", 0, 1450, "zh", "scenario_pack_demo", 10, "small_room", 0.9, 0, 50, 1.0, 0.0, "standard"),
        ("vw_pack_003", "audio/backchannel.wav", "嗯嗯", "backchannel", "backchannel", "continue_speaking", "true", "true", 0, 650, "zh", "scenario_pack_demo", 20, "small_room", 1.0, 200, 10, 0.8, 0.0, "standard"),
        ("vw_pack_004", "audio/wait_stop.wav", "先别说我想一下", "wait_stop", "wait", "hold", "false", "false", 0, 1200, "zh", "scenario_pack_demo", 15, "none", 0.8, 0, 0, 0.8, 0.0, "standard"),
        ("vw_pack_005", "audio/user_interruption.wav", "等一下不是这个", "user_interruption", "complete", "stop_tts_and_listen", "true", "true", 100, 900, "zh", "scenario_pack_demo", 10, "none", 1.1, 120, 30, 1.0, 0.0, "standard"),
        ("vw_pack_006", "audio/side_conversation.wav", "你帮我拿一下水", "side_conversation", "wait", "ignore", "false", "true", 0, 1350, "zh", "scenario_pack_demo", 5, "large_room", 1.2, 300, 80, 2.0, 0.0, "regional"),
        ("vw_pack_007", "audio/ambient_speech.wav", "电视里的新闻声音", "ambient_speech", "wait", "ignore", "false", "true", 0, 1800, "zh", "scenario_pack_demo", 0, "hallway", 1.0, 500, 100, 3.5, 0.0, "standard"),
        ("vw_pack_008", "audio/noisy_farfield.wav", "帮我查一下航班状态", "noisy_farfield", "complete", "take_turn", "false", "false", 0, 1700, "zh", "scenario_pack_demo", -5, "large_room", 1.0, 0, 100, 5.0, 0.0, "regional"),
        ("vw_pack_009", "audio/code_switching.wav", "帮我 book 一个 meeting room", "code_switching", "complete", "take_turn", "false", "false", 0, 1600, "zh_en", "scenario_pack_demo", 10, "small_room", 1.2, 0, 50, 1.0, 0.4, "non_native"),
    ]
    lines = ["\t".join(header)]
    lines.extend("\t".join(str(value) for value in row) for row in rows)
    return "\n".join(lines) + "\n"


def _commands_markdown(commands: list[str]) -> str:
    return "\n".join(
        [
            "# Stable-ASR Scenario Pack Commands",
            "",
            "Run these commands from the scenario pack root.",
            "",
            "```bash",
            "\n".join(commands),
            "```",
            "",
        ]
    )


def _commands_script(commands: list[str]) -> str:
    return "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", "", *commands, ""])


def _data_readme() -> str:
    return "\n".join(
        [
            "# VoiceWorld Scenario Fixture Data",
            "",
            "- `voiceworld_metadata.tsv`: editable scenario annotations covering every built-in VoiceWorld v0 scenario.",
            "- `voiceworld_manifest.jsonl`: normalized Stable-ASR turn manifest generated from the metadata table.",
            "",
            "Audio paths are placeholders for contribution workflow testing. Replace them with real WAV/FLAC files before using the pack as final-scale evidence.",
            "",
        ]
    )


def _write_json(path: str | Path, payload: dict[str, object]) -> str:
    return _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: str | Path, text: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
