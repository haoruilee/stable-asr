"""Unified work queue for reference collection and adapter tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.references.collections import asr_collections_source_manifest, load_asr_collections
from stable_asr.references.turn_collections import turn_collections_source_manifest, load_turn_collections


DEFAULT_REFERENCE_WORKQUEUE_PRIORITIES = ("p0", "p1")


@dataclass(frozen=True)
class ReferenceWorkQueueValidation:
    ok: bool
    errors: list[str]

    def to_text(self) -> str:
        if self.ok:
            return "reference_workqueue: OK"
        return "reference_workqueue: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class ReferenceAssignmentAuditReport:
    ok: bool
    path: str
    rows: int
    blocking_release: list[str]
    unassigned: list[str]
    missing_due_dates: list[str]
    missing_evidence: list[str]
    missing_license_reviews: list[str]
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
            "missing_evidence": self.missing_evidence,
            "missing_license_reviews": self.missing_license_reviews,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def to_markdown(self) -> str:
        rows = [
            {"check": "rows", "value": self.rows},
            {"check": "blocking_release", "value": len(self.blocking_release)},
            {"check": "unassigned", "value": len(self.unassigned)},
            {"check": "missing_due_dates", "value": len(self.missing_due_dates)},
            {"check": "missing_evidence", "value": len(self.missing_evidence)},
            {"check": "missing_license_reviews", "value": len(self.missing_license_reviews)},
            {"check": "errors", "value": len(self.errors)},
            {"check": "warnings", "value": len(self.warnings)},
        ]
        lines = [
            "# Stable-ASR Reference Assignment Audit",
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
        lines.extend(["", "## Missing Evidence", ""])
        lines.extend(f"- `{item}`" for item in self.missing_evidence) if self.missing_evidence else lines.append("- none")
        lines.extend(["", "## Missing License Reviews", ""])
        lines.extend(f"- `{item}`" for item in self.missing_license_reviews) if self.missing_license_reviews else lines.append("- none")
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{item}`" for item in self.errors) if self.errors else lines.append("- none")
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{item}`" for item in self.warnings) if self.warnings else lines.append("- none")
        return "\n".join(lines) + "\n"


def reference_workqueue_from_registries(
    *,
    asr_registry: dict[str, Any] | None = None,
    turn_registry: dict[str, Any] | None = None,
    required_priorities: tuple[str, ...] = DEFAULT_REFERENCE_WORKQUEUE_PRIORITIES,
) -> dict[str, object]:
    """Build a unified contributor work queue from ASR and turn registries."""

    asr_manifest = asr_collections_source_manifest(asr_registry or load_asr_collections())
    turn_manifest = turn_collections_source_manifest(turn_registry or load_turn_collections())
    return reference_workqueue_from_source_manifests(
        asr_manifest,
        turn_manifest,
        required_priorities=required_priorities,
    )


def reference_workqueue_from_source_manifests(
    *source_manifests: dict[str, Any],
    required_priorities: tuple[str, ...] = DEFAULT_REFERENCE_WORKQUEUE_PRIORITIES,
) -> dict[str, object]:
    """Build a unified contributor work queue from source manifests."""

    priority_set = set(required_priorities)
    tasks: list[dict[str, object]] = []
    for manifest in source_manifests:
        collection_type = str(manifest.get("collection_type", ""))
        sources = manifest.get("sources", [])
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict):
                continue
            priority = str(source.get("priority", ""))
            if priority not in priority_set:
                continue
            reference_id = str(source.get("reference_id", ""))
            task_id = f"{collection_type}:{reference_id}"
            license_review_required = bool(source.get("license_review_required"))
            tasks.append(
                {
                    "task_id": task_id,
                    "collection_type": collection_type,
                    "reference_id": reference_id,
                    "name": str(source.get("name", "")),
                    "category": str(source.get("category", "")),
                    "priority": priority,
                    "acquisition_track": str(source.get("acquisition_track", "")),
                    "evidence_target": str(source.get("evidence_target", "")),
                    "license": str(source.get("license", "")),
                    "license_review_required": license_review_required,
                    "license_review_target": str(source.get("license_review_target", "")),
                    "policy": str(source.get("policy", "")),
                    "status": _task_status(license_review_required),
                    "next_action": _next_action(source),
                    "blocked_by": _blocked_by(license_review_required),
                    "source_url": str(source.get("source_url", "")),
                    "docs_url": str(source.get("docs_url", "")),
                    "stable_asr_actions": list(source.get("stable_asr_actions", [])),
                    "reference_use": str(source.get("reference_use", "")),
                }
            )
    tasks.sort(key=lambda task: (str(task["priority"]), str(task["collection_type"]), str(task["task_id"])))
    return {
        "id": "stable_asr_reference_workqueue_v0",
        "version": "0.1.0",
        "generated_by": "stable-asr reference-workqueue",
        "required_priorities": list(required_priorities),
        "tasks": tasks,
    }


def validate_reference_workqueue(workqueue: dict[str, Any]) -> ReferenceWorkQueueValidation:
    errors: list[str] = []
    for key in ("id", "version", "generated_by", "required_priorities", "tasks"):
        if key not in workqueue:
            errors.append(f"missing top-level key: {key}")
    tasks = workqueue.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks must be a non-empty list")
        return ReferenceWorkQueueValidation(ok=False, errors=errors)
    seen: set[str] = set()
    required = {
        "task_id",
        "collection_type",
        "reference_id",
        "name",
        "priority",
        "category",
        "acquisition_track",
        "evidence_target",
        "license",
        "license_review_required",
        "license_review_target",
        "policy",
        "status",
        "next_action",
        "blocked_by",
        "source_url",
        "docs_url",
        "stable_asr_actions",
        "reference_use",
    }
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"task {index} must be an object")
            continue
        task_id = str(task.get("task_id", ""))
        if not task_id:
            errors.append(f"task {index} missing task_id")
        elif task_id in seen:
            errors.append(f"duplicate task_id: {task_id}")
        else:
            seen.add(task_id)
        missing = sorted(required.difference(task))
        if missing:
            errors.append(f"task {task_id or index} missing: {', '.join(missing)}")
        if task.get("collection_type") not in {"asr", "turn"}:
            errors.append(f"task {task_id or index} collection_type must be asr or turn")
        if task.get("priority") not in {"p0", "p1", "p2"}:
            errors.append(f"task {task_id or index} priority must be p0, p1, or p2")
        if not isinstance(task.get("license_review_required"), bool):
            errors.append(f"task {task_id or index} license_review_required must be boolean")
        actions = task.get("stable_asr_actions", [])
        if not isinstance(actions, list) or not actions:
            errors.append(f"task {task_id or index} stable_asr_actions must be a non-empty list")
    return ReferenceWorkQueueValidation(ok=not errors, errors=errors)


def reference_workqueue_markdown(workqueue: dict[str, Any]) -> str:
    validation = validate_reference_workqueue(workqueue)
    if not validation.ok:
        raise ValueError(validation.to_text())
    tasks = workqueue["tasks"]
    rows = [
        {
            "task": task["task_id"],
            "priority": task["priority"],
            "track": task["acquisition_track"],
            "evidence": task["evidence_target"],
            "license_review": "yes" if task["license_review_required"] else "no",
            "status": task["status"],
        }
        for task in tasks
    ]
    review_count = sum(1 for task in tasks if task["license_review_required"])
    lines = [
        "# Stable-ASR Reference Work Queue",
        "",
        "This queue is generated from the ASR and turn reference source manifests. It is a contributor-facing task list, not evidence that the work has been completed.",
        "",
        f"- status: `{'OK' if validation.ok else 'FAILED'}`",
        f"- required_priorities: `{', '.join(workqueue.get('required_priorities', []))}`",
        f"- tasks: `{len(tasks)}`",
        f"- license_review_required: `{review_count}`",
        "",
        dict_table(rows),
        "",
        "## Evidence Rule",
        "",
        "A task is complete only when the evidence target exists, passes its schema or audit command, and any license review target is filled before vendoring or redistribution.",
    ]
    return "\n".join(lines)


def reference_workqueue_jsonl(workqueue: dict[str, Any]) -> str:
    validation = validate_reference_workqueue(workqueue)
    if not validation.ok:
        raise ValueError(validation.to_text())
    return "\n".join(json.dumps(task, ensure_ascii=False, sort_keys=True) for task in workqueue["tasks"]) + "\n"


def reference_workqueue_assignments(workqueue: dict[str, Any]) -> dict[str, object]:
    """Turn a reference work queue into an owner-fillable assignment tracker."""

    validation = validate_reference_workqueue(workqueue)
    if not validation.ok:
        raise ValueError(validation.to_text())
    rows = []
    for task in workqueue["tasks"]:
        rows.append(
            {
                "task_id": task["task_id"],
                "collection_type": task["collection_type"],
                "reference_id": task["reference_id"],
                "name": task["name"],
                "priority": task["priority"],
                "category": task["category"],
                "owner": "",
                "due_date": "",
                "status": _assignment_status(task),
                "blocking_release": task["priority"] == "p0",
                "evidence_target": task["evidence_target"],
                "license": task["license"],
                "license_review_required": task["license_review_required"],
                "license_review_target": task["license_review_target"],
                "next_action": task["next_action"],
                "blocked_by": task["blocked_by"],
                "source_url": task["source_url"],
                "docs_url": task["docs_url"],
                "notes": "",
            }
        )
    return {
        "id": "stable_asr_reference_assignments_v0",
        "version": "0.1.0",
        "generated_by": "stable-asr reference-workqueue --format assignments-json",
        "source_workqueue_id": workqueue["id"],
        "rows": rows,
    }


def reference_workqueue_assignments_tsv(assignments: dict[str, Any]) -> str:
    rows = assignments.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("reference assignments rows must be a non-empty list")
    columns = [
        "task_id",
        "collection_type",
        "reference_id",
        "name",
        "priority",
        "category",
        "owner",
        "due_date",
        "status",
        "blocking_release",
        "evidence_target",
        "license_review_required",
        "license_review_target",
        "next_action",
        "blocked_by",
        "source_url",
        "docs_url",
        "notes",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append("\t".join(_tsv_cell(row.get(column, "")) for column in columns))
    return "\n".join(lines) + "\n"


def reference_workqueue_assignments_markdown(assignments: dict[str, Any]) -> str:
    rows = assignments.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("reference assignments rows must be a non-empty list")
    table_rows = [
        {
            "task": row["task_id"],
            "priority": row["priority"],
            "owner": row["owner"] or "unassigned",
            "due": row["due_date"] or "unset",
            "status": row["status"],
            "blocking": "yes" if row["blocking_release"] else "no",
            "evidence": row["evidence_target"],
        }
        for row in rows
        if isinstance(row, dict)
    ]
    return "\n".join(
        [
            "# Stable-ASR Reference Assignments",
            "",
            f"- status: `TEMPLATE`",
            f"- source_workqueue_id: `{assignments.get('source_workqueue_id', '')}`",
            f"- rows: `{len(table_rows)}`",
            "",
            dict_table(table_rows),
            "",
            "## Owner Workflow",
            "",
            "1. Fill `owner` and `due_date` for every P0 task first.",
            "2. Keep `blocking_release` true for P0 tasks until the evidence target exists and any license review is complete.",
            "3. Do not vendor upstream code, weights, fixtures, or long snippets before the linked license review is filled.",
            "4. Update `status` to `ready_for_review` only after evidence and review artifacts are staged.",
            "",
        ]
    )


def audit_reference_assignments(
    path: str | Path,
    *,
    repo_root: str | Path = ".",
    require_owner: bool = False,
    require_due_date: bool = False,
    require_ready: bool = False,
) -> ReferenceAssignmentAuditReport:
    """Audit a filled reference assignment tracker."""

    assignment_path = Path(path)
    repo_root = Path(repo_root)
    errors: list[str] = []
    warnings: list[str] = []
    blocking_release: list[str] = []
    unassigned: list[str] = []
    missing_due_dates: list[str] = []
    missing_evidence: list[str] = []
    missing_license_reviews: list[str] = []

    try:
        payload = json.loads(assignment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ReferenceAssignmentAuditReport(
            ok=False,
            path=str(assignment_path),
            rows=0,
            blocking_release=[],
            unassigned=[],
            missing_due_dates=[],
            missing_evidence=[],
            missing_license_reviews=[],
            errors=[str(exc)],
            warnings=[],
        )

    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        errors.append("rows must be a non-empty list")
        rows = []

    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index}: must be an object")
            continue
        _audit_reference_assignment_row(
            index,
            row,
            repo_root=repo_root,
            seen=seen,
            require_owner=require_owner,
            require_due_date=require_due_date,
            require_ready=require_ready,
            blocking_release=blocking_release,
            unassigned=unassigned,
            missing_due_dates=missing_due_dates,
            missing_evidence=missing_evidence,
            missing_license_reviews=missing_license_reviews,
            errors=errors,
            warnings=warnings,
        )

    return ReferenceAssignmentAuditReport(
        ok=not errors,
        path=str(assignment_path),
        rows=len(rows),
        blocking_release=blocking_release,
        unassigned=unassigned,
        missing_due_dates=missing_due_dates,
        missing_evidence=missing_evidence,
        missing_license_reviews=missing_license_reviews,
        errors=errors,
        warnings=warnings,
    )


def write_reference_workqueue_json(path: str | Path, workqueue: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workqueue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _task_status(license_review_required: bool) -> str:
    if license_review_required:
        return "link_or_command_adapter_until_license_review"
    return "ready_for_adapter_or_bridge_work"


def _next_action(source: dict[str, Any]) -> str:
    actions = source.get("stable_asr_actions", [])
    if isinstance(actions, list) and actions:
        return str(actions[0])
    return str(source.get("acquisition_track", ""))


def _blocked_by(license_review_required: bool) -> list[str]:
    if license_review_required:
        return ["license_review_before_vendoring"]
    return []


def _assignment_status(task: dict[str, Any]) -> str:
    if task.get("license_review_required"):
        return "blocked_license_review"
    return "needs_evidence"


def _audit_reference_assignment_row(
    index: int,
    row: dict[str, Any],
    *,
    repo_root: Path,
    seen: set[str],
    require_owner: bool,
    require_due_date: bool,
    require_ready: bool,
    blocking_release: list[str],
    unassigned: list[str],
    missing_due_dates: list[str],
    missing_evidence: list[str],
    missing_license_reviews: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        errors.append(f"row {index}:task_id:missing")
        task_id = f"row_{index}"
    label = str(task_id)
    if label in seen:
        errors.append(f"{label}:duplicate")
    seen.add(label)

    for field in ("collection_type", "reference_id", "priority", "status", "evidence_target"):
        if not isinstance(row.get(field), str) or not str(row.get(field)).strip():
            errors.append(f"{label}:{field}:missing")
    if row.get("collection_type") not in {"asr", "turn"}:
        errors.append(f"{label}:collection_type:invalid")
    if row.get("priority") not in {"p0", "p1", "p2"}:
        errors.append(f"{label}:priority:invalid")
    if not isinstance(row.get("blocking_release"), bool):
        errors.append(f"{label}:blocking_release:invalid")
    if not isinstance(row.get("license_review_required"), bool):
        errors.append(f"{label}:license_review_required:invalid")
    blocked_by = row.get("blocked_by", [])
    if not isinstance(blocked_by, list) or not all(isinstance(item, str) for item in blocked_by):
        errors.append(f"{label}:blocked_by:invalid")

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

    if row.get("blocking_release") is True:
        blocking_release.append(label)
        message = f"{label}:blocking_release"
        if require_ready:
            errors.append(message)
        else:
            warnings.append(message)

    evidence_target = str(row.get("evidence_target", "")).strip()
    if evidence_target and not _target_exists(evidence_target, repo_root=repo_root):
        missing_evidence.append(label)
        message = f"{label}:evidence:missing:{evidence_target}"
        if require_ready or str(row.get("status", "")) in {"ready_for_review", "complete"}:
            errors.append(message)
        else:
            warnings.append(message)

    license_review_target = str(row.get("license_review_target", "")).strip()
    if row.get("license_review_required") is True and license_review_target:
        if not _target_exists(license_review_target, repo_root=repo_root):
            missing_license_reviews.append(label)
            message = f"{label}:license_review:missing:{license_review_target}"
            if require_ready or str(row.get("status", "")) in {"ready_for_review", "complete"}:
                errors.append(message)
            else:
                warnings.append(message)


def _target_exists(path: str, *, repo_root: Path) -> bool:
    target = Path(path)
    if not target.is_absolute():
        target = repo_root / target
    return target.exists()


def _tsv_cell(value: object) -> str:
    if isinstance(value, list):
        value = ",".join(str(item) for item in value)
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")
