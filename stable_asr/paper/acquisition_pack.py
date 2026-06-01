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
from stable_asr.paper.handoff import final_handoff_schema_markdown, final_handoff_template
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
class AcquisitionAssignmentRow:
    collection_id: str
    title: str
    category: str
    priority: str
    required: bool
    owner: str
    due_date: str
    status: str
    blocking_release: bool
    missing_required_paths: list[str]
    pending_generated_paths: list[str]
    license: str
    license_review_required: bool
    handoff_required: bool
    source_urls: list[str]
    next_command: str
    verification: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return {
            "collection_id": self.collection_id,
            "title": self.title,
            "category": self.category,
            "priority": self.priority,
            "required": self.required,
            "owner": self.owner,
            "due_date": self.due_date,
            "status": self.status,
            "blocking_release": self.blocking_release,
            "missing_required_paths": self.missing_required_paths,
            "pending_generated_paths": self.pending_generated_paths,
            "license": self.license,
            "license_review_required": self.license_review_required,
            "handoff_required": self.handoff_required,
            "source_urls": self.source_urls,
            "next_command": self.next_command,
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
    assignment_rows: int
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
            and self.assignment_rows > 0
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
            "assignment_rows": self.assignment_rows,
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
            {"check": "assignment_rows", "status": str(self.assignment_rows)},
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


@dataclass(frozen=True)
class FinalAssignmentAuditReport:
    ok: bool
    path: str
    rows: int
    blocking_release: list[str]
    unassigned: list[str]
    missing_due_dates: list[str]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "path": self.path,
            "rows": self.rows,
            "blocking_release": self.blocking_release,
            "unassigned": self.unassigned,
            "missing_due_dates": self.missing_due_dates,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def to_markdown(self) -> str:
        rows = [
            {"check": "rows", "value": self.rows},
            {"check": "blocking_release", "value": len(self.blocking_release)},
            {"check": "unassigned", "value": len(self.unassigned)},
            {"check": "missing_due_dates", "value": len(self.missing_due_dates)},
            {"check": "errors", "value": len(self.errors)},
            {"check": "warnings", "value": len(self.warnings)},
        ]
        lines = [
            "# Stable-ASR Final Assignment Audit",
            "",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- assignments: `{self.path}`",
            "",
            dict_table(rows),
            "",
            "## Blocking Release",
            "",
        ]
        lines.extend(f"- `{item}`" for item in self.blocking_release) if self.blocking_release else lines.append("- none")
        lines.extend(["", "## Unassigned", ""])
        lines.extend(f"- `{item}`" for item in self.unassigned) if self.unassigned else lines.append("- none")
        lines.extend(["", "## Missing Due Dates", ""])
        lines.extend(f"- `{item}`" for item in self.missing_due_dates) if self.missing_due_dates else lines.append("- none")
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{item}`" for item in self.errors) if self.errors else lines.append("- none")
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{item}`" for item in self.warnings) if self.warnings else lines.append("- none")
        return "\n".join(lines) + "\n"


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
    assignments = _assignment_rows(registry, checklist)
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
    files["assignments_json"] = _write_json(
        output_dir / "acquisition" / "assignments.json",
        {"rows": [row.to_dict() for row in assignments]},
    )
    files["assignments_tsv"] = _write_text(
        output_dir / "acquisition" / "assignments.tsv",
        _assignments_tsv(assignments),
    )
    files["assignments_markdown"] = _write_text(
        output_dir / "acquisition" / "ASSIGNMENTS.md",
        _assignments_markdown(assignments),
    )
    issue_templates = _issue_templates(registry, checklist, assignments)
    files["issue_index_json"] = _write_json(
        output_dir / "acquisition" / "issues.json",
        {"issues": [_issue_manifest_row(issue) for issue in issue_templates]},
    )
    files["issue_index_markdown"] = _write_text(
        output_dir / "acquisition" / "ISSUE_INDEX.md",
        _issue_index_markdown(issue_templates),
    )
    for issue in issue_templates:
        files[f"issue_template:{issue['collection_id']}"] = _write_text(
            output_dir / str(issue["path"]),
            str(issue["markdown"]),
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
    files["handoff_json_template"] = _write_json(
        output_dir / "acquisition" / "handoff_template.json",
        final_handoff_template(registry),
    )
    files["handoff_schema_markdown"] = _write_text(
        output_dir / "acquisition" / "HANDOFF_SCHEMA.md",
        final_handoff_schema_markdown(),
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
        assignment_rows=len(assignments),
        missing_required=report.missing_required,
        license_review_items=len(license_items),
        config_ok=config_validation.ok,
        input_collections_ok=registry_validation.ok,
    )
    files["readme"] = _write_text(output_dir / "README.md", final_report.to_markdown())
    _write_json(output_dir / "manifest.json", final_report.to_dict())
    return final_report


def audit_acquisition_assignments(
    path: str | Path,
    *,
    require_owner: bool = False,
    require_due_date: bool = False,
    require_ready: bool = False,
) -> FinalAssignmentAuditReport:
    """Audit a filled final-acquisition assignment tracker."""

    assignment_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    blocking_release: list[str] = []
    unassigned: list[str] = []
    missing_due_dates: list[str] = []

    try:
        payload = json.loads(assignment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return FinalAssignmentAuditReport(
            ok=False,
            path=str(assignment_path),
            rows=0,
            blocking_release=[],
            unassigned=[],
            missing_due_dates=[],
            errors=[str(exc)],
            warnings=[],
        )

    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        errors.append("rows must be a non-empty list")
        rows = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index}: must be an object")
            continue
        _audit_assignment_row(
            index,
            row,
            require_owner=require_owner,
            require_due_date=require_due_date,
            require_ready=require_ready,
            blocking_release=blocking_release,
            unassigned=unassigned,
            missing_due_dates=missing_due_dates,
            errors=errors,
            warnings=warnings,
        )

    return FinalAssignmentAuditReport(
        ok=not errors,
        path=str(assignment_path),
        rows=len(rows),
        blocking_release=blocking_release,
        unassigned=unassigned,
        missing_due_dates=missing_due_dates,
        errors=errors,
        warnings=warnings,
    )


def _audit_assignment_row(
    index: int,
    row: dict[str, Any],
    *,
    require_owner: bool,
    require_due_date: bool,
    require_ready: bool,
    blocking_release: list[str],
    unassigned: list[str],
    missing_due_dates: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    collection_id = row.get("collection_id")
    if not isinstance(collection_id, str) or not collection_id.strip():
        errors.append(f"row {index}:collection_id:missing")
        collection_id = f"row_{index}"
    label = str(collection_id)

    status = row.get("status")
    if not isinstance(status, str) or not status.strip():
        errors.append(f"{label}:status:missing")
    owner = str(row.get("owner", "")).strip()
    if not owner or owner == "unassigned":
        unassigned.append(label)
        message = f"{label}:owner:unassigned"
        if require_owner:
            errors.append(message)
        else:
            warnings.append(message)
    due_date = str(row.get("due_date", "")).strip()
    if not due_date:
        missing_due_dates.append(label)
        message = f"{label}:due_date:missing"
        if require_due_date:
            errors.append(message)
        else:
            warnings.append(message)

    row_blocking = bool(row.get("blocking_release"))
    if row_blocking:
        blocking_release.append(label)
        message = f"{label}:blocking_release"
        if require_ready:
            errors.append(message)
        else:
            warnings.append(message)

    for field in ("missing_required_paths", "pending_generated_paths", "source_urls"):
        value = row.get(field, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"{label}:{field}:invalid")


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


def _assignment_rows(
    registry: dict[str, Any],
    checklist: list[AcquisitionChecklistRow],
) -> list[AcquisitionAssignmentRow]:
    grouped: dict[str, list[AcquisitionChecklistRow]] = {}
    for row in checklist:
        grouped.setdefault(row.collection_id, []).append(row)

    rows: list[AcquisitionAssignmentRow] = []
    for collection in registry["collections"]:
        collection_id = str(collection["id"])
        path_rows = grouped.get(collection_id, [])
        missing_required = [
            row.path for row in path_rows if row.path_kind == "required_input" and row.status != "present"
        ]
        pending_generated = [
            row.path for row in path_rows if row.path_kind == "generated_artifact" and row.status != "present"
        ]
        required = bool(collection["required"])
        blocking_release = required and (bool(missing_required) or bool(pending_generated))
        rows.append(
            AcquisitionAssignmentRow(
                collection_id=collection_id,
                title=str(collection["title"]),
                category=str(collection["category"]),
                priority=str(collection["priority"]),
                required=required,
                owner=str(collection.get("owner", "unassigned")),
                due_date=str(collection.get("due_date", "")),
                status=_assignment_status(
                    required=required,
                    missing_required=missing_required,
                    pending_generated=pending_generated,
                ),
                blocking_release=blocking_release,
                missing_required_paths=missing_required,
                pending_generated_paths=pending_generated,
                license=str(collection["license"]),
                license_review_required=_needs_license_review(str(collection["license"])),
                handoff_required=required or bool(missing_required) or bool(pending_generated),
                source_urls=list(collection.get("source_urls", [])),
                next_command=_first_string(collection.get("commands", [])),
                verification=_first_string(collection.get("verification", [])),
                notes=str(collection.get("notes", "")),
            )
        )
    return rows


def _issue_templates(
    registry: dict[str, Any],
    checklist: list[AcquisitionChecklistRow],
    assignments: list[AcquisitionAssignmentRow],
) -> list[dict[str, Any]]:
    grouped_paths: dict[str, list[AcquisitionChecklistRow]] = {}
    for row in checklist:
        grouped_paths.setdefault(row.collection_id, []).append(row)
    assignments_by_id = {row.collection_id: row for row in assignments}

    issues: list[dict[str, Any]] = []
    for collection in registry["collections"]:
        collection_id = str(collection["id"])
        assignment = assignments_by_id[collection_id]
        path_rows = grouped_paths.get(collection_id, [])
        labels = [
            "final-data",
            f"priority:{assignment.priority}",
            f"category:{assignment.category}",
            "blocking-release" if assignment.blocking_release else "non-blocking",
        ]
        path = f"acquisition/issues/{_issue_slug(collection_id)}.md"
        issue = {
            "collection_id": collection_id,
            "title": f"[Final data] {assignment.title}",
            "labels": labels,
            "path": path,
            "status": assignment.status,
            "blocking_release": assignment.blocking_release,
            "owner": assignment.owner,
            "due_date": assignment.due_date,
            "required": assignment.required,
            "license": assignment.license,
            "license_review_required": assignment.license_review_required,
            "handoff_required": assignment.handoff_required,
            "missing_required_paths": assignment.missing_required_paths,
            "pending_generated_paths": assignment.pending_generated_paths,
            "source_urls": assignment.source_urls,
            "commands": list(collection.get("commands", [])),
            "verification": list(collection.get("verification", [])),
            "markdown": "",
        }
        issue["markdown"] = _issue_markdown(issue, collection=collection, path_rows=path_rows)
        issues.append(issue)
    return issues


def _issue_manifest_row(issue: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in issue.items() if key != "markdown"}


def _issue_index_markdown(issues: list[dict[str, Any]]) -> str:
    rows = [
        {
            "collection_id": issue["collection_id"],
            "priority": issue["labels"][1],
            "status": issue["status"],
            "blocking_release": str(issue["blocking_release"]),
            "issue_template": issue["path"],
        }
        for issue in issues
    ]
    lines = [
        "# Stable-ASR Final Acquisition Issue Index",
        "",
        (
            "These issue templates convert the final input registry into owner-ready "
            "tasks. They are coordination artifacts only; final evidence still needs "
            "real staged paths, verification outputs, and a checksummed handoff."
        ),
        "",
        dict_table(rows),
        "",
    ]
    return "\n".join(lines)


def _issue_markdown(
    issue: dict[str, Any],
    *,
    collection: dict[str, Any],
    path_rows: list[AcquisitionChecklistRow],
) -> str:
    path_table = [
        {
            "kind": row.path_kind,
            "path": row.path,
            "status": row.status,
        }
        for row in path_rows
    ]
    lines = [
        f"# {issue['title']}",
        "",
        "## Metadata",
        "",
        f"- collection_id: `{issue['collection_id']}`",
        f"- category: `{collection['category']}`",
        f"- priority: `{collection['priority']}`",
        f"- required: `{collection['required']}`",
        f"- status: `{issue['status']}`",
        f"- blocking_release: `{issue['blocking_release']}`",
        f"- owner: `{issue['owner']}`",
        f"- due_date: `{issue['due_date'] or 'TBD'}`",
        f"- labels: `{', '.join(issue['labels'])}`",
        f"- license: `{issue['license']}`",
        f"- license_review_required: `{issue['license_review_required']}`",
        f"- handoff_required: `{issue['handoff_required']}`",
        "",
        "## Why This Matters",
        "",
        str(collection.get("notes", "")) or "Final-scale input required by the Stable-ASR release gates.",
        "",
        "## Source URLs",
        "",
    ]
    if issue["source_urls"]:
        lines.extend(f"- {url}" for url in issue["source_urls"])
    else:
        lines.append("- none")
    lines.extend(["", "## Paths", "", dict_table(path_table), "", "## Commands", ""])
    if issue["commands"]:
        lines.extend(f"```bash\n{command}\n```" for command in issue["commands"])
    else:
        lines.append("- none")
    lines.extend(["", "## Verification", ""])
    if issue["verification"]:
        lines.extend(f"```bash\n{command}\n```" for command in issue["verification"])
    else:
        lines.append("- none")
    handoff_note = (
        "Do not attach upstream data, model weights, private recordings, or long copied snippets to the issue "
        "unless the license review explicitly allows redistribution."
    )
    lines.extend(
        [
            "",
            "## Acceptance Checklist",
            "",
            "- [ ] Owner and due date are set in `acquisition/assignments.json`.",
            "- [ ] Required input paths are staged with real data, not smoke fixtures.",
            "- [ ] Generated artifacts are produced by the commands above.",
            "- [ ] Verification commands pass and outputs are recorded.",
            "- [ ] License or consent notes are filled for this collection.",
            "- [ ] `acquisition/handoff_template.json` contains this collection with staged paths.",
            "- [ ] `stable-asr final-handoff-checksums` has populated byte sizes and SHA256 values.",
            "- [ ] `stable-asr final-handoff-audit --require-checksums` passes for the handoff.",
            "",
            "## Handoff Notes",
            "",
            handoff_note,
            "",
        ]
    )
    return "\n".join(lines)


def _issue_slug(collection_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in collection_id).strip("_")


def _assignment_status(
    *,
    required: bool,
    missing_required: list[str],
    pending_generated: list[str],
) -> str:
    if missing_required:
        return "blocked_missing_required_input" if required else "optional_missing_required_input"
    if pending_generated:
        return "pending_generated_artifacts" if required else "optional_pending_generated_artifacts"
    return "ready_for_handoff" if required else "optional_ready_for_handoff"


def _license_review_rows(registry: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for collection in registry["collections"]:
        license_name = str(collection["license"])
        if _needs_license_review(license_name):
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


def _needs_license_review(license_name: str) -> bool:
    needs_review = {
        "see_upstream",
        "depends_on_external_system",
        "project_or_recording_consent",
        "derived_from_upstream_inputs",
        "depends_on_input_audio",
        "depends_on_input_predictions",
    }
    return license_name in needs_review or license_name.startswith("depends_on")


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


def _first_string(values: object) -> str:
    if isinstance(values, list) and values:
        return str(values[0])
    return ""


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


def _assignments_tsv(rows: list[AcquisitionAssignmentRow]) -> str:
    headers = [
        "collection_id",
        "title",
        "category",
        "priority",
        "required",
        "owner",
        "due_date",
        "status",
        "blocking_release",
        "missing_required_paths",
        "pending_generated_paths",
        "license",
        "license_review_required",
        "handoff_required",
        "source_urls",
        "next_command",
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


def _assignments_markdown(rows: list[AcquisitionAssignmentRow]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "collection_id": row.collection_id,
                "priority": row.priority,
                "required": str(row.required),
                "owner": row.owner,
                "due_date": row.due_date or "TBD",
                "status": row.status,
                "blocking_release": str(row.blocking_release),
                "missing_required_paths": "; ".join(row.missing_required_paths) or "none",
                "pending_generated_paths": "; ".join(row.pending_generated_paths) or "none",
                "license_review_required": str(row.license_review_required),
            }
        )
    blocking = [row for row in rows if row.blocking_release]
    lines = [
        "# Stable-ASR Final Acquisition Assignments",
        "",
        (
            "Use this tracker to assign owners and due dates for final-scale inputs. "
            "It is intentionally separate from the evidence handoff: update this "
            "file while coordinating work, then use `HANDOFF_TEMPLATE.md` or "
            "`handoff_template.json` for the final verified submission."
        ),
        "",
        f"- collections: `{len(rows)}`",
        f"- blocking_release: `{len(blocking)}`",
        "",
        dict_table(table_rows),
        "",
        "## Owner Workflow",
        "",
        "1. Set `owner` and `due_date` in `assignments.json` or this Markdown copy.",
        "2. Stage the required paths and generated artifacts listed for the collection.",
        "3. Run the collection verification command from `DATA_ACQUISITION.md`.",
        "4. Fill `handoff_template.json`, run `stable-asr final-handoff-checksums`, validate it with `stable-asr validate-schema-file --schema-id stable_asr.final_handoff.v0`, then run `stable-asr final-handoff-audit --require-checksums`.",
        "",
    ]
    return "\n".join(lines)


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
        f"stable-asr final-inputs --registry {PACK_FINAL_INPUTS_PATH} --config {PACK_FINAL_CONFIG_PATH} --repo-root . --require-checksums --output reports/FINAL_INPUT_COLLECTIONS_CURRENT.md",
        f"stable-asr final-config --config {PACK_FINAL_CONFIG_PATH} --repo-root . --plan-missing --output reports/FINAL_RUN_ACTION_PLAN_CURRENT.md",
        f"stable-asr final-config --config {PACK_FINAL_CONFIG_PATH} --repo-root . --check-files --output reports/FINAL_RUN_FILE_AUDIT_CURRENT.md || true",
        "stable-asr final-assignment-audit --input acquisition/assignments.json --output reports/FINAL_ASSIGNMENT_AUDIT.md || true",
        "stable-asr final-handoff-checksums --input acquisition/handoff_template.json --repo-root . --output acquisition/handoff_template.json || true",
        "stable-asr validate-schema-file --input acquisition/handoff_template.json --schema-id stable_asr.final_handoff.v0 --output reports/FINAL_HANDOFF_SCHEMA_VALIDATION.md || true",
        "stable-asr final-handoff-audit --input acquisition/handoff_template.json --repo-root . --require-checksums --output reports/FINAL_HANDOFF_AUDIT.md || true",
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
