"""Build a self-contained contributor onboarding pack."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.paper.acquisition_pack import build_final_acquisition_pack
from stable_asr.paper.adapter_pack import build_adapter_pack
from stable_asr.paper.benchmark_pack import build_benchmark_pack
from stable_asr.paper.final_pack import build_final_pack
from stable_asr.paper.scenario_pack import build_scenario_pack
from stable_asr.references import (
    reference_workqueue_from_registries,
    reference_workqueue_jsonl,
    reference_workqueue_markdown,
)
from stable_asr.resources import resolve_platform_path


CONTRIBUTOR_PACK_VERSION = "contributor_pack_v0"


@dataclass(frozen=True)
class ContributorPackReport:
    output_dir: str
    files: dict[str, str]
    commands: list[str]
    pack_statuses: dict[str, bool]
    template_files: list[str]

    @property
    def ok(self) -> bool:
        return all(self.pack_statuses.values()) and bool(self.commands) and bool(self.template_files)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "version": CONTRIBUTOR_PACK_VERSION,
            "output_dir": self.output_dir,
            "files": self.files,
            "commands": self.commands,
            "pack_statuses": self.pack_statuses,
            "template_files": self.template_files,
        }

    def to_markdown(self) -> str:
        pack_rows = [
            {"pack": name, "status": "OK" if ok else "FAILED", "path": f"packs/{name}"}
            for name, ok in sorted(self.pack_statuses.items())
        ]
        file_rows = [{"name": name, "path": path} for name, path in sorted(self.files.items())]
        return "\n".join(
            [
                "# Stable-ASR Contributor Pack",
                "",
                f"- status: `{'OK' if self.ok else 'FAILED'}`",
                f"- version: `{CONTRIBUTOR_PACK_VERSION}`",
                f"- output_dir: `{self.output_dir}`",
                "",
                (
                    "This pack is the single entrypoint for external contributors. "
                    "It gathers benchmark submissions, ASR adapters, VoiceWorld scenarios, "
                    "final-run planning, final input acquisition, and GitHub workflow templates."
                ),
                "",
                "## Included Packs",
                "",
                dict_table(pack_rows),
                "",
                "## Included Files",
                "",
                dict_table(file_rows),
                "",
                "## Run From This Directory",
                "",
                "```bash",
                "\n".join(self.commands),
                "```",
                "",
            ]
        )


def build_contributor_pack(output_dir: str | Path) -> ContributorPackReport:
    """Write a unified contributor pack for all public contribution tracks."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    _write_json(output_dir / "manifest.json", {"version": CONTRIBUTOR_PACK_VERSION, "status": "building"})

    reports = {
        "benchmark_pack": build_benchmark_pack(output_dir / "packs" / "benchmark_pack"),
        "adapter_pack": build_adapter_pack(output_dir / "packs" / "adapter_pack"),
        "scenario_pack": build_scenario_pack(output_dir / "packs" / "scenario_pack"),
        "final_pack": build_final_pack(output_dir / "packs" / "final_pack"),
        "final_acquisition_pack": build_final_acquisition_pack(output_dir / "packs" / "final_acquisition_pack"),
    }
    pack_statuses = {name: report.ok for name, report in reports.items()}
    for name, report in reports.items():
        files[f"{name}:readme"] = report.files["readme"]
        files[f"{name}:commands"] = report.files["commands_script"]
        files[f"{name}:manifest"] = str(output_dir / "packs" / name / "manifest.json")

    template_files = _copy_templates(output_dir)
    for path in template_files:
        files[f"template:{Path(path).name}"] = path

    reference_workqueue = reference_workqueue_from_registries(required_priorities=("p0", "p1"))
    files["reference_workqueue_json"] = _write_json(
        output_dir / "references" / "reference_workqueue.json",
        reference_workqueue,
    )
    files["reference_workqueue_jsonl"] = _write_text(
        output_dir / "references" / "reference_workqueue.jsonl",
        reference_workqueue_jsonl(reference_workqueue),
    )
    files["reference_workqueue_markdown"] = _write_text(
        output_dir / "references" / "REFERENCE_WORKQUEUE.md",
        reference_workqueue_markdown(reference_workqueue),
    )

    commands = _starter_commands()
    files["tracks"] = _write_text(output_dir / "CONTRIBUTION_TRACKS.md", _tracks_markdown())
    files["commands_markdown"] = _write_text(output_dir / "COMMANDS.md", _commands_markdown(commands))
    files["commands_script"] = _write_text(output_dir / "commands.sh", _commands_script(commands))

    report = ContributorPackReport(
        output_dir=str(output_dir),
        files=files,
        commands=commands,
        pack_statuses=pack_statuses,
        template_files=template_files,
    )
    files["readme"] = _write_text(output_dir / "README.md", report.to_markdown())
    _write_json(output_dir / "manifest.json", report.to_dict())
    return report


def _starter_commands() -> list[str]:
    return [
        "(cd packs/benchmark_pack && bash commands.sh)",
        "(cd packs/adapter_pack && bash commands.sh)",
        "(cd packs/scenario_pack && bash commands.sh)",
        "(cd packs/final_pack && bash commands.sh)",
        "(cd packs/final_acquisition_pack && bash commands.sh)",
        "stable-asr reference-workqueue --output references/REFERENCE_WORKQUEUE_CURRENT.md",
    ]


def _copy_templates(output_dir: Path) -> list[str]:
    copied: list[str] = []
    template_root = output_dir / "github_templates"
    issue_dir = resolve_platform_path(".github/ISSUE_TEMPLATE")
    if issue_dir.exists():
        destination = template_root / "ISSUE_TEMPLATE"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(issue_dir, destination)
        copied.extend(str(path) for path in sorted(destination.glob("*.yml")))
    pr_template = resolve_platform_path(".github/PULL_REQUEST_TEMPLATE.md")
    if pr_template.exists():
        destination = template_root / "PULL_REQUEST_TEMPLATE.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pr_template, destination)
        copied.append(str(destination))
    return copied


def _tracks_markdown() -> str:
    rows = [
        {
            "track": "Reference collection and license review",
            "pack": "references/REFERENCE_WORKQUEUE.md",
            "github_template": "asr_adapter.yml, voiceworld_scenario.yml",
            "first_command": "stable-asr reference-workqueue --output runs/REFERENCE_WORKQUEUE.md",
        },
        {
            "track": "Benchmark submission",
            "pack": "packs/benchmark_pack",
            "github_template": "benchmark_submission.yml",
            "first_command": "stable-asr benchmark-pack --output-dir runs/benchmark_pack",
        },
        {
            "track": "External ASR adapter",
            "pack": "packs/adapter_pack",
            "github_template": "asr_adapter.yml",
            "first_command": "stable-asr adapter-pack --output-dir runs/adapter_pack",
        },
        {
            "track": "VoiceWorld scenario",
            "pack": "packs/scenario_pack",
            "github_template": "voiceworld_scenario.yml",
            "first_command": "stable-asr scenario-pack --output-dir runs/scenario_pack",
        },
        {
            "track": "Final run planning",
            "pack": "packs/final_pack",
            "github_template": "PULL_REQUEST_TEMPLATE.md",
            "first_command": "stable-asr final-pack --output-dir runs/final_pack",
        },
        {
            "track": "Final input acquisition",
            "pack": "packs/final_acquisition_pack",
            "github_template": "final_data_acquisition.yml",
            "first_command": "stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack && stable-asr final-assignment-audit --input runs/final_acquisition_pack/acquisition/assignments.json",
        },
    ]
    return "\n".join(
        [
            "# Stable-ASR Contribution Tracks",
            "",
            "Use this index to choose the right starter pack and GitHub template.",
            "",
            dict_table(rows),
            "",
        ]
    )


def _commands_markdown(commands: list[str]) -> str:
    return "\n".join(
        [
            "# Stable-ASR Contributor Pack Commands",
            "",
            "Run these commands from the contributor pack root.",
            "",
            "```bash",
            "\n".join(commands),
            "```",
            "",
        ]
    )


def _commands_script(commands: list[str]) -> str:
    return "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", "", *commands, ""])


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    return _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: str | Path, text: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return str(path)
