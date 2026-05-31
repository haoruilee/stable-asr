"""Build final-scale input acquisition starter packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.paper.final_config import (
    build_final_run_action_plan,
    final_run_config_markdown,
    load_final_run_config,
    validate_final_run_config,
    write_final_run_config_json,
)
from stable_asr.paper.final_inputs import (
    final_input_collection_report,
    load_final_input_collections,
    validate_final_input_collections,
    write_final_input_collections_json,
)
from stable_asr.resources import resolve_platform_path


FINAL_ACQUISITION_PACK_VERSION = "final_acquisition_pack_v0"
PACK_FINAL_CONFIG_PATH = "configs/final/paper_final.json"
PACK_FINAL_INPUTS_PATH = "configs/final/input_collections.json"


@dataclass(frozen=True)
class AcquisitionChecklistRow:
    collection_id: str
    title: str
    category: str
    priority: str
    required: bool
    license: str
    path_kind: str
    path: str
    status: str
    source_urls: list[str]
    command: str
    verification: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return {
            "collection_id": self.collection_id,
            "title": self.title,
            "category": self.category,
            "priority": self.priority,
            "required": self.required,
            "license": self.license,
            "path_kind": self.path_kind,
            "path": self.path,
            "status": self.status,
            "source_urls": self.source_urls,
            "command": self.command,
            "verification": self.verification,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class FinalAcquisitionPackReport:
    output_dir: str
    files: dict[str, str]
    commands: list[str]
    collections: int
    checklist_rows: int
    missing_required: list[str]
    license_review_items: int
    config_ok: bool
    input_collections_ok: bool

    @property
    def ok(self) -> bool:
        return (
            self.config_ok
            and self.input_collections_ok
            and self.collections > 0
            and self.checklist_rows > 0
            and bool(self.commands)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "version": FINAL_ACQUISITION_PACK_VERSION,
            "output_dir": self.output_dir,
            "files": self.files,
            "commands": self.commands,
            "collections": self.collections,
            "checklist_rows": self.checklist_rows,
            "missing_required": self.missing_required,
            "license_review_items": self.license_review_items,
            "config_ok": self.config_ok,
            "input_collections_ok": self.input_collections_ok,
        }

    def to_markdown(self) -> str:
        status_rows = [
            {"check": "pack_build", "status": "OK" if self.ok else "FAILED"},
            {"check": "collections", "status": str(self.collections)},
            {"check": "checklist_rows", "status": str(self.checklist_rows)},
            {"check": "missing_required", "status": str(len(self.missing_required))},
            {"check": "license_review_items", "status": str(self.license_review_items)},
        ]
        file_rows = [{"name": name, "path": path} for name, path in sorted(self.files.items())]
        lines = [
            "# Stable-ASR Final Acquisition Pack",
            "",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- version: `{FINAL_ACQUISITION_PACK_VERSION}`",
            f"- output_dir: `{self.output_dir}`",
            "",
            (
                "This pack turns the final input collection registry into a data "
                "staging checklist for corpora, VoiceWorld recordings, external "
                "turn predictions, ASR adapter exports, NanoTurn artifacts, and "
                "paper bundle outputs. It does not download, copy, or fabricate "
                "benchmark data."
            ),
            "",
            "## Status",
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


def build_final_acquisition_pack(
    output_dir: str | Path,
    *,
    input_collections_path: str | Path | None = None,
    config_path: str | Path | None = None,
    repo_root: str | Path = ".",
) -> FinalAcquisitionPackReport:
    """Write a collaborator-facing acquisition checklist for final-scale inputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root)

    config = load_final_run_config(config_path)
    config_validation = validate_final_run_config(config)
    registry = load_final_input_collections(input_collections_path)
    registry_validation = validate_final_input_collections(registry)
    if not config_validation.ok:
        raise ValueError(config_validation.to_text())
    if not registry_validation.ok:
        raise ValueError(registry_validation.to_text())

    files: dict[str, str] = {}
    _write_json(output_dir / "manifest.json", {"version": FINAL_ACQUISITION_PACK_VERSION, "status": "building"})
    files["final_config_json"] = write_final_run_config_json(output_dir / PACK_FINAL_CONFIG_PATH, config)
    files["final_config_markdown"] = _write_text(
        output_dir / "reports" / "FINAL_RUN_CONFIG.md",
        final_run_config_markdown(config),
    )
    files["input_collections_json"] = write_final_input_collections_json(
        output_dir / PACK_FINAL_INPUTS_PATH,
        registry,
    )
    files["asr_command_config"] = _copy_json_to_pack(
        config["asr_command_config"],
        output_dir / str(config["asr_command_config"]),
    )

    report = final_input_collection_report(registry, config=config, repo_root=repo_root)
    files["input_status_json"] = _write_json(
        output_dir / "reports" / "final_input_collection_status.json",
        report.to_dict(),
    )
    files["input_status_markdown"] = _write_text(
        output_dir / "reports" / "FINAL_INPUT_COLLECTIONS.md",
        report.to_markdown(),
    )

    action_plan = build_final_run_action_plan(
        config,
        repo_root=repo_root,
        config_path=PACK_FINAL_CONFIG_PATH,
    )
    files["action_plan_json"] = _write_json(
        output_dir / "reports" / "final_run_action_plan.json",
        action_plan.to_dict(),
    )
    files["action_plan_markdown"] = _write_text(
        output_dir / "reports" / "FINAL_RUN_ACTION_PLAN.md",
        action_plan.to_markdown(),
    )

    checklist = _checklist_rows(registry, repo_root=repo_root)
    files["checklist_json"] = _write_json(
        output_dir / "acquisition" / "staging_checklist.json",
        {"rows": [row.to_dict() for row in checklist]},
    )
    files["checklist_tsv"] = _write_text(
        output_dir / "acquisition" / "staging_checklist.tsv",
        _checklist_tsv(checklist),
    )
    files["acquisition_markdown"] = _write_text(
        output_dir / "acquisition" / "DATA_ACQUISITION.md",
        _acquisition_markdown(registry, checklist),
    )
    license_items = _license_review_rows(registry)
    files["license_review_markdown"] = _write_text(
        output_dir / "acquisition" / "LICENSE_REVIEW.md",
        _license_review_markdown(license_items),
    )
    files["voiceworld_checklist"] = _write_text(
        output_dir / "acquisition" / "VOICEWORLD_RECORDING_CHECKLIST.md",
        _voiceworld_recording_checklist(registry),
    )
    files["handoff_template"] = _write_text(
        output_dir / "acquisition" / "HANDOFF_TEMPLATE.md",
        _handoff_template(),
    )

    commands = _starter_commands()
    files["commands_markdown"] = _write_text(output_dir / "COMMANDS.md", _commands_markdown(commands))
    files["commands_script"] = _write_text(output_dir / "commands.sh", _commands_script(commands))

    final_report = FinalAcquisitionPackReport(
        output_dir=str(output_dir),
        files=files,
        commands=commands,
        collections=len(registry["collections"]),
        checklist_rows=len(checklist),
        missing_required=report.missing_required,
        license_review_items=len(license_items),
        config_ok=config_validation.ok,
        input_collections_ok=registry_validation.ok,
    )
    files["readme"] = _write_text(output_dir / "README.md", final_report.to_markdown())
    _write_json(output_dir / "manifest.json", final_report.to_dict())
    return final_report


def _checklist_rows(registry: dict[str, Any], *, repo_root: Path) -> list[AcquisitionChecklistRow]:
    rows: list[AcquisitionChecklistRow] = []
    for collection in registry["collections"]:
        commands = list(collection.get("commands", []))
        verification = list(collection.get("verification", []))
        base = {
            "collection_id": str(collection["id"]),
            "title": str(collection["title"]),
            "category": str(collection["category"]),
            "priority": str(collection["priority"]),
            "required": bool(collection["required"]),
            "license": str(collection["license"]),
            "source_urls": list(collection.get("source_urls", [])),
            "command": commands[0] if commands else "",
            "verification": verification[0] if verification else "",
            "notes": str(collection.get("notes", "")),
        }
        for path in collection.get("required_paths", []):
            rows.append(
                AcquisitionChecklistRow(
                    **base,
                    path_kind="required_input",
                    path=str(path),
                    status=_path_status(str(path), repo_root=repo_root, missing="missing"),
                )
            )
        for path in collection.get("generated_paths", []):
            rows.append(
                AcquisitionChecklistRow(
                    **base,
                    path_kind="generated_artifact",
                    path=str(path),
                    status=_path_status(str(path), repo_root=repo_root, missing="pending"),
                )
            )
    return rows


def _license_review_rows(registry: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    needs_review = {
        "see_upstream",
        "depends_on_external_system",
        "project_or_recording_consent",
        "derived_from_upstream_inputs",
        "depends_on_input_audio",
        "depends_on_input_predictions",
    }
    for collection in registry["collections"]:
        license_name = str(collection["license"])
        if license_name in needs_review or license_name.startswith("depends_on"):
            rows.append(
                {
                    "collection_id": str(collection["id"]),
                    "title": str(collection["title"]),
                    "license": license_name,
                    "source_urls": "; ".join(collection.get("source_urls", [])) or "project generated",
                    "required_action": _license_action(license_name),
                }
            )
    return rows


def _license_action(license_name: str) -> str:
    if license_name == "project_or_recording_consent":
        return "record consent, speaker permission, and redistribution constraints before publishing manifests"
    if license_name == "derived_from_upstream_inputs":
        return "inherit and document upstream data licenses for generated manifests and splits"
    if license_name == "see_upstream":
        return "review upstream license, model card, and terms before storing artifacts"
    if license_name.startswith("depends_on"):
        return "document the external system or input-data license in the final handoff"
    return "review before release"


def _path_status(path: str, *, repo_root: Path, missing: str) -> str:
    target = Path(path)
    candidate = target if target.is_absolute() else repo_root / target
    return "present" if candidate.exists() else missing


def _checklist_tsv(rows: list[AcquisitionChecklistRow]) -> str:
    headers = [
        "collection_id",
        "title",
        "category",
        "priority",
        "required",
        "license",
        "path_kind",
        "path",
        "status",
        "source_urls",
        "command",
        "verification",
        "notes",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        payload = row.to_dict()
        values: list[str] = []
        for header in headers:
            value = payload[header]
            if isinstance(value, list):
                text = "; ".join(str(item) for item in value)
            else:
                text = str(value)
            values.append(_tsv_cell(text))
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


def _acquisition_markdown(registry: dict[str, Any], rows: list[AcquisitionChecklistRow]) -> str:
    grouped: dict[str, list[AcquisitionChecklistRow]] = {}
    for row in rows:
        grouped.setdefault(row.collection_id, []).append(row)
    lines = [
        "# Stable-ASR Final Data Acquisition",
        "",
        "Use this file to assign and track final-scale data staging. It is a checklist, not evidence.",
        "",
    ]
    for collection in registry["collections"]:
        collection_id = str(collection["id"])
        path_rows = [
            {
                "path_kind": row.path_kind,
                "path": row.path,
                "status": row.status,
            }
            for row in grouped.get(collection_id, [])
        ]
        lines.extend(
            [
                f"## {collection['title']}",
                "",
                f"- id: `{collection_id}`",
                f"- category: `{collection['category']}`",
                f"- priority: `{collection['priority']}`",
                f"- required: `{collection['required']}`",
                f"- license: `{collection['license']}`",
                f"- source_urls: `{'; '.join(collection.get('source_urls', [])) or 'none'}`",
                "",
                str(collection.get("notes", "")),
                "",
                "Paths:",
                "",
                dict_table(path_rows),
                "",
                "Commands:",
                "",
            ]
        )
        lines.extend(f"```bash\n{command}\n```" for command in collection.get("commands", []))
        lines.extend(["", "Verification:", ""])
        lines.extend(f"```bash\n{command}\n```" for command in collection.get("verification", []))
        lines.append("")
    return "\n".join(lines)


def _license_review_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Stable-ASR Final License Review",
        "",
        "Review these items before publishing final artifacts. Stable-ASR should link to upstream projects and user-provided outputs without vendoring or relicensing external assets.",
        "",
    ]
    lines.append(dict_table(rows) if rows else "No license review items detected.")
    lines.append("")
    return "\n".join(lines)


def _voiceworld_recording_checklist(registry: dict[str, Any]) -> str:
    voiceworld = next((item for item in registry["collections"] if item.get("id") == "voiceworld_real"), None)
    paths = voiceworld.get("required_paths", []) if isinstance(voiceworld, dict) else []
    return "\n".join(
        [
            "# VoiceWorld Recording Checklist",
            "",
            "Use this checklist before treating scenario audio as final evidence.",
            "",
            "- speaker consent or synthetic-data provenance recorded",
            "- redistribution policy recorded for every audio file",
            "- scenario id assigned from `configs/scenarios/stable_asr_voiceworld_v0.json`",
            "- factor annotations present for SNR, reverb, accent, overlap, assistant state, language, and distance when applicable",
            "- audio files are referenced by stable relative paths",
            "- `stable-asr prepare-voiceworld` and `stable-asr final-config --audit-voiceworld-real` pass",
            "",
            "Configured required paths:",
            "",
            *(f"- `{path}`" for path in paths),
            "",
        ]
    )


def _handoff_template() -> str:
    return "\n".join(
        [
            "# Final Input Handoff",
            "",
            "- owner:",
            "- collection_id:",
            "- staged_paths:",
            "- source_urls:",
            "- license_or_consent_notes:",
            "- commands_run:",
            "- verification_outputs:",
            "- known_gaps:",
            "",
        ]
    )


def _starter_commands() -> list[str]:
    return [
        f"stable-asr final-inputs --registry {PACK_FINAL_INPUTS_PATH} --config {PACK_FINAL_CONFIG_PATH} --repo-root . --output reports/FINAL_INPUT_COLLECTIONS_CURRENT.md",
        f"stable-asr final-config --config {PACK_FINAL_CONFIG_PATH} --repo-root . --plan-missing --output reports/FINAL_RUN_ACTION_PLAN_CURRENT.md",
        f"stable-asr final-config --config {PACK_FINAL_CONFIG_PATH} --repo-root . --check-files --output reports/FINAL_RUN_FILE_AUDIT_CURRENT.md || true",
    ]


def _commands_markdown(commands: list[str]) -> str:
    return "\n".join(
        [
            "# Stable-ASR Final Acquisition Commands",
            "",
            "Run these commands from the acquisition pack root after staging files.",
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


def _copy_json_to_pack(source: str | Path, destination: str | Path) -> str:
    source_path = resolve_platform_path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"required acquisition pack source not found: {source}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(destination)


def _write_text(path: str | Path, text: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return str(path)


def _tsv_cell(text: str) -> str:
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
