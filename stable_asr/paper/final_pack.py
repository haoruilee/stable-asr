"""Build self-contained final-run starter packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.models.registry import (
    load_model_registry,
    model_registry_markdown,
    write_model_registry_json,
)
from stable_asr.paper.evidence import final_evidence_matrix
from stable_asr.paper.final_config import (
    audit_final_run_files,
    build_final_run_action_plan,
    final_run_config_markdown,
    final_run_file_audit_markdown,
    load_final_run_config,
    scaffold_final_run,
    validate_final_run_config,
    write_final_run_config_json,
)
from stable_asr.paper.final_experiments import (
    final_experiments_markdown,
    load_final_experiments,
    validate_final_experiments,
    write_final_experiments_json,
)
from stable_asr.paper.final_inputs import (
    final_input_collection_report,
    load_final_input_collections,
    validate_final_input_collections,
    write_final_input_collections_json,
)
from stable_asr.references import (
    asr_collections_markdown,
    asr_collections_source_manifest,
    load_asr_collections,
    load_turn_collections,
    reference_workqueue_assignments,
    reference_workqueue_assignments_markdown,
    reference_workqueue_assignments_tsv,
    reference_workqueue_from_registries,
    reference_workqueue_jsonl,
    reference_workqueue_markdown,
    turn_collections_markdown,
    turn_collections_source_manifest,
    write_asr_collections_json,
    write_turn_collections_json,
)
from stable_asr.resources import resolve_platform_path
from stable_asr.scenarios.suites import (
    load_scenario_suite,
    scenario_suite_markdown,
    write_scenario_suite_json,
)


FINAL_PACK_VERSION = "final_pack_v0"
PACK_FINAL_CONFIG_PATH = "configs/final/paper_final.json"
PACK_FINAL_INPUTS_PATH = "configs/final/input_collections.json"
PACK_FINAL_EXPERIMENTS_PATH = "configs/paper/final_experiments.json"
PACK_SCENARIO_SUITE_PATH = "configs/scenarios/stable_asr_voiceworld_v0.json"
PACK_MODEL_REGISTRY_PATH = "configs/models/stable_asr_models.json"
PACK_ASR_COLLECTIONS_PATH = "configs/references/asr_collections.json"
PACK_TURN_COLLECTIONS_PATH = "configs/references/turn_collections.json"


@dataclass(frozen=True)
class FinalPackReport:
    output_dir: str
    files: dict[str, str]
    commands: list[str]
    config_ok: bool
    input_collections_ok: bool
    final_experiments_ok: bool
    scaffold_entries: int
    final_ready: bool
    missing_required: list[str]

    @property
    def ok(self) -> bool:
        return all(
            [
                self.config_ok,
                self.input_collections_ok,
                self.final_experiments_ok,
                self.scaffold_entries > 0,
                bool(self.commands),
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "version": FINAL_PACK_VERSION,
            "output_dir": self.output_dir,
            "files": self.files,
            "commands": self.commands,
            "config_ok": self.config_ok,
            "input_collections_ok": self.input_collections_ok,
            "final_experiments_ok": self.final_experiments_ok,
            "scaffold_entries": self.scaffold_entries,
            "final_ready": self.final_ready,
            "missing_required": self.missing_required,
        }

    def to_markdown(self) -> str:
        file_rows = [{"name": name, "path": path} for name, path in sorted(self.files.items())]
        status_rows = [
            {"check": "pack_build", "status": "OK" if self.ok else "FAILED"},
            {"check": "final_ready", "status": "READY" if self.final_ready else "NOT_READY"},
            {"check": "config", "status": _yes_no(self.config_ok)},
            {"check": "input_collections", "status": _yes_no(self.input_collections_ok)},
            {"check": "final_experiments", "status": _yes_no(self.final_experiments_ok)},
            {"check": "scaffold_entries", "status": str(self.scaffold_entries)},
            {"check": "missing_required", "status": str(len(self.missing_required))},
        ]
        lines = [
            "# Stable-ASR Final Run Starter Pack",
            "",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- final_ready: `{'READY' if self.final_ready else 'NOT_READY'}`",
            f"- version: `{FINAL_PACK_VERSION}`",
            f"- output_dir: `{self.output_dir}`",
            "",
            (
                "This pack centralizes final-scale configs, input collection plans, "
                "experiment runbooks, evidence audits, and directory scaffolding. "
                "It intentionally does not include real corpora, raw predictions, "
                "NanoTurn checkpoints, or benchmark outputs."
            ),
            "",
            "## Readiness Checks",
            "",
            dict_table(status_rows),
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
        if self.missing_required:
            lines.extend(["## Missing Required Inputs", ""])
            lines.extend(f"- `{path}`" for path in self.missing_required)
            lines.append("")
        return "\n".join(lines)


def build_final_pack(
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    input_collections_path: str | Path | None = None,
    final_experiments_path: str | Path | None = None,
    scenario_suite_path: str | Path | None = None,
) -> FinalPackReport:
    """Write a starter workspace for final-scale Stable-ASR platform runs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_final_run_config(config_path)
    config_validation = validate_final_run_config(config)
    input_collections = load_final_input_collections(input_collections_path)
    input_validation = validate_final_input_collections(input_collections)
    experiments = load_final_experiments(final_experiments_path)
    experiment_validation = validate_final_experiments(experiments)
    if not config_validation.ok:
        raise ValueError(config_validation.to_text())
    if not input_validation.ok:
        raise ValueError(input_validation.to_text())
    if not experiment_validation.ok:
        raise ValueError(experiment_validation.to_text())

    files: dict[str, str] = {}
    _write_json(output_dir / "manifest.json", {"version": FINAL_PACK_VERSION, "status": "building"})

    files["final_config_json"] = write_final_run_config_json(output_dir / PACK_FINAL_CONFIG_PATH, config)
    files["final_config_markdown"] = _write_text(
        output_dir / "reports" / "FINAL_RUN_CONFIG.md",
        final_run_config_markdown(config),
    )
    files["final_input_collections_json"] = write_final_input_collections_json(
        output_dir / PACK_FINAL_INPUTS_PATH,
        input_collections,
    )
    files["final_experiments_json"] = write_final_experiments_json(
        output_dir / PACK_FINAL_EXPERIMENTS_PATH,
        experiments,
    )
    files["final_experiments_markdown"] = _write_text(
        output_dir / "reports" / "FINAL_EXPERIMENTS.md",
        final_experiments_markdown(experiments),
    )

    scenario_suite = load_scenario_suite(scenario_suite_path)
    files["scenario_suite_json"] = write_scenario_suite_json(output_dir / PACK_SCENARIO_SUITE_PATH, scenario_suite)
    files["scenario_suite_markdown"] = _write_text(
        output_dir / "reports" / "SCENARIO_SUITE.md",
        scenario_suite_markdown(scenario_suite),
    )
    files["asr_command_config"] = _copy_json_to_pack(
        config["asr_command_config"],
        output_dir / str(config["asr_command_config"]),
    )
    models = load_model_registry()
    files["model_registry_json"] = write_model_registry_json(output_dir / PACK_MODEL_REGISTRY_PATH, models)
    files["model_registry_markdown"] = _write_text(
        output_dir / "reports" / "MODELS.md",
        model_registry_markdown(models),
    )
    collections = load_asr_collections()
    files["asr_collections_json"] = write_asr_collections_json(output_dir / PACK_ASR_COLLECTIONS_PATH, collections)
    files["asr_collections_markdown"] = _write_text(
        output_dir / "reports" / "ASR_COLLECTIONS.md",
        asr_collections_markdown(collections),
    )
    files["asr_collections_source_manifest"] = _write_json(
        output_dir / "reports" / "asr_collection_source_manifest.json",
        asr_collections_source_manifest(collections),
    )
    turn_collections = load_turn_collections()
    files["turn_collections_json"] = write_turn_collections_json(
        output_dir / PACK_TURN_COLLECTIONS_PATH,
        turn_collections,
    )
    files["turn_collections_markdown"] = _write_text(
        output_dir / "reports" / "TURN_COLLECTIONS.md",
        turn_collections_markdown(turn_collections),
    )
    files["turn_collections_source_manifest"] = _write_json(
        output_dir / "reports" / "turn_collection_source_manifest.json",
        turn_collections_source_manifest(turn_collections),
    )
    reference_workqueue = reference_workqueue_from_registries(
        asr_registry=collections,
        turn_registry=turn_collections,
        required_priorities=("p0", "p1"),
    )
    files["reference_workqueue_json"] = _write_json(
        output_dir / "reports" / "reference_workqueue.json",
        reference_workqueue,
    )
    files["reference_workqueue_jsonl"] = _write_text(
        output_dir / "reports" / "reference_workqueue.jsonl",
        reference_workqueue_jsonl(reference_workqueue),
    )
    files["reference_workqueue_markdown"] = _write_text(
        output_dir / "reports" / "REFERENCE_WORKQUEUE.md",
        reference_workqueue_markdown(reference_workqueue),
    )
    reference_assignments = reference_workqueue_assignments(reference_workqueue)
    files["reference_assignments_json"] = _write_json(
        output_dir / "reports" / "reference_assignments.json",
        reference_assignments,
    )
    files["reference_assignments_tsv"] = _write_text(
        output_dir / "reports" / "reference_assignments.tsv",
        reference_workqueue_assignments_tsv(reference_assignments),
    )
    files["reference_assignments_markdown"] = _write_text(
        output_dir / "reports" / "REFERENCE_ASSIGNMENTS.md",
        reference_workqueue_assignments_markdown(reference_assignments),
    )

    file_audit = audit_final_run_files(config, repo_root=output_dir)
    missing_required = [f"{check.name}: {check.path}" for check in file_audit.checks if check.required and not check.ok]
    files["file_audit_json"] = _write_json(output_dir / "reports" / "final_run_file_audit.json", file_audit.to_dict())
    files["file_audit_markdown"] = _write_text(
        output_dir / "reports" / "FINAL_RUN_FILE_AUDIT.md",
        final_run_file_audit_markdown(file_audit),
    )

    action_plan = build_final_run_action_plan(
        config,
        repo_root=output_dir,
        config_path=PACK_FINAL_CONFIG_PATH,
    )
    files["action_plan_json"] = _write_json(output_dir / "reports" / "final_run_action_plan.json", action_plan.to_dict())
    files["action_plan_markdown"] = _write_text(
        output_dir / "reports" / "FINAL_RUN_ACTION_PLAN.md",
        action_plan.to_markdown(),
    )

    input_report = final_input_collection_report(input_collections, config=config, repo_root=output_dir)
    files["input_collections_status_json"] = _write_json(
        output_dir / "reports" / "final_input_collection_status.json",
        input_report.to_dict(),
    )
    files["input_collections_markdown"] = _write_text(
        output_dir / "reports" / "FINAL_INPUT_COLLECTIONS.md",
        input_report.to_markdown(),
    )

    evidence_report = final_evidence_matrix(
        repo_root=output_dir,
        registry_path=output_dir / PACK_FINAL_EXPERIMENTS_PATH,
        config_path=output_dir / PACK_FINAL_CONFIG_PATH,
        artifacts_dir=output_dir / "runs" / "final" / "artifacts",
    )
    files["evidence_matrix_json"] = _write_json(
        output_dir / "reports" / "final_evidence_matrix.json",
        evidence_report.to_dict(),
    )
    files["evidence_matrix_markdown"] = _write_text(
        output_dir / "reports" / "FINAL_EVIDENCE_MATRIX.md",
        evidence_report.to_markdown(),
    )

    scaffold_report = scaffold_final_run(config, repo_root=output_dir)
    files["scaffold_json"] = _write_json(output_dir / "reports" / "final_run_scaffold.json", scaffold_report.to_dict())
    files["scaffold_text"] = _write_text(output_dir / "reports" / "FINAL_RUN_SCAFFOLD.txt", scaffold_report.to_text())

    files["data_readme"] = _write_text(output_dir / "data" / "README.md", _data_readme())
    files["runs_readme"] = _write_text(output_dir / "runs" / "README.md", _runs_readme())
    files["manual_commands"] = _write_text(output_dir / "NEXT_COMMANDS.md", _manual_commands(config))

    commands = _starter_commands()
    files["commands_markdown"] = _write_text(output_dir / "COMMANDS.md", _commands_markdown(commands))
    files["commands_script"] = _write_text(output_dir / "commands.sh", _commands_script(commands))

    report = FinalPackReport(
        output_dir=str(output_dir),
        files=files,
        commands=commands,
        config_ok=config_validation.ok,
        input_collections_ok=input_validation.ok,
        final_experiments_ok=experiment_validation.ok,
        scaffold_entries=len(scaffold_report.entries),
        final_ready=file_audit.ok and evidence_report.final_ready,
        missing_required=missing_required,
    )
    files["readme"] = _write_text(output_dir / "README.md", report.to_markdown())
    _write_json(output_dir / "manifest.json", report.to_dict())
    return report


def _starter_commands() -> list[str]:
    return [
        f"stable-asr final-config --config {PACK_FINAL_CONFIG_PATH} --validate-only",
        f"stable-asr final-experiments --registry {PACK_FINAL_EXPERIMENTS_PATH} --validate-only",
        (
            f"stable-asr final-inputs --registry {PACK_FINAL_INPUTS_PATH} "
            f"--config {PACK_FINAL_CONFIG_PATH} --repo-root . "
            "--output reports/FINAL_INPUT_COLLECTIONS_CURRENT.md"
        ),
        (
            "stable-asr paper-evidence-matrix "
            f"--registry {PACK_FINAL_EXPERIMENTS_PATH} "
            f"--config {PACK_FINAL_CONFIG_PATH} "
            "--repo-root . --artifacts-dir runs/final/artifacts "
            "--output reports/FINAL_EVIDENCE_MATRIX_CURRENT.md"
        ),
        (
            "stable-asr reference-workqueue "
            "--asr-registry configs/references/asr_collections.json "
            "--turn-registry configs/references/turn_collections.json "
            "--output reports/REFERENCE_WORKQUEUE_CURRENT.md"
        ),
        (
            "stable-asr reference-workqueue --format assignments-markdown "
            "--asr-registry configs/references/asr_collections.json "
            "--turn-registry configs/references/turn_collections.json "
            "--output reports/REFERENCE_ASSIGNMENTS_CURRENT.md"
        ),
        (
            f"stable-asr final-config --config {PACK_FINAL_CONFIG_PATH} --repo-root . "
            "--plan-missing --output reports/FINAL_RUN_ACTION_PLAN_CURRENT.md"
        ),
        f"stable-asr final-config --config {PACK_FINAL_CONFIG_PATH} --repo-root . --scaffold",
        (
            f"stable-asr final-config --config {PACK_FINAL_CONFIG_PATH} --repo-root . "
            "--check-files --output reports/FINAL_RUN_FILE_AUDIT_CURRENT.md || true"
        ),
    ]


def _commands_markdown(commands: list[str]) -> str:
    return "\n".join(
        [
            "# Stable-ASR Final Pack Commands",
            "",
            "Run these commands from the final pack root.",
            "",
            "The last audit command is non-blocking because a fresh starter pack is expected to report `NOT_READY` until real corpora, predictions, checkpoints, and result files are staged.",
            "",
            "```bash",
            "\n".join(commands),
            "```",
            "",
        ]
    )


def _commands_script(commands: list[str]) -> str:
    return "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", "", *commands, ""])


def _manual_commands(config: dict[str, Any]) -> str:
    commands = "\n".join(str(command) for command in config.get("commands", []))
    return "\n".join(
        [
            "# Final-Scale Run Commands",
            "",
            "These commands are copied from the final run config. Run them only after replacing scaffold directories with real data and adapter outputs.",
            "",
            "```bash",
            commands,
            "```",
            "",
        ]
    )


def _data_readme() -> str:
    return "\n".join(
        [
            "# Final Data Staging",
            "",
            "Place or symlink real upstream corpora, VoiceWorld annotations, audio, and external prediction exports under this tree.",
            "",
            "This starter pack only creates directories and README hints. Empty directories are not benchmark evidence.",
            "",
        ]
    )


def _runs_readme() -> str:
    return "\n".join(
        [
            "# Final Run Outputs",
            "",
            "Generated manifests, NanoTurn checkpoints, reports, paper artifacts, and archives should be written under `runs/final`.",
            "",
            "Run `stable-asr final-config --check-files` before claiming final readiness.",
            "",
        ]
    )


def _copy_json_to_pack(source: str | Path, destination: str | Path) -> str:
    source_path = resolve_platform_path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"required final pack source not found: {source}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(destination)


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    return _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: str | Path, text: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return str(path)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
