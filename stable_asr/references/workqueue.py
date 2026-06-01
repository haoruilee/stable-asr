"""Unified work queue for reference collection and adapter tasks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.references.collections import asr_collections_source_manifest, load_asr_collections
from stable_asr.references.turn_collections import turn_collections_source_manifest, load_turn_collections


DEFAULT_REFERENCE_WORKQUEUE_PRIORITIES = ("p0", "p1")
EVIDENCE_MARKDOWN_REQUIRED_SECTIONS = (
    "Upstream version and source",
    "Inputs used",
    "Command, script, or bridge implementation notes",
    "Output paths and schema or validation commands",
    "Metrics, examples, or failure notes relevant to Stable-ASR",
    "License and redistribution decision",
)
LICENSE_REVIEW_REQUIRED_SECTIONS = (
    "## Decision",
    "status:",
    "reviewer:",
    "approved_uses:",
    "prohibited_uses:",
    "required_notices:",
)


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


@dataclass(frozen=True)
class ReferenceEvidenceAuditRow:
    task_id: str
    collection_type: str
    reference_id: str
    priority: str
    evidence_target: str
    evidence_present: bool
    license_review_required: bool
    license_review_target: str
    license_review_present: bool
    evidence_content_checked: bool
    evidence_content_ok: bool
    evidence_content_errors: list[str]
    license_review_content_checked: bool
    license_review_content_ok: bool
    license_review_content_errors: list[str]
    ok: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "collection_type": self.collection_type,
            "reference_id": self.reference_id,
            "priority": self.priority,
            "evidence_target": self.evidence_target,
            "evidence_present": self.evidence_present,
            "license_review_required": self.license_review_required,
            "license_review_target": self.license_review_target,
            "license_review_present": self.license_review_present,
            "evidence_content_checked": self.evidence_content_checked,
            "evidence_content_ok": self.evidence_content_ok,
            "evidence_content_errors": self.evidence_content_errors,
            "license_review_content_checked": self.license_review_content_checked,
            "license_review_content_ok": self.license_review_content_ok,
            "license_review_content_errors": self.license_review_content_errors,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class ReferenceEvidenceAuditReport:
    ok: bool
    repo_root: str
    require_content: bool
    rows: list[ReferenceEvidenceAuditRow]
    missing_evidence: list[str]
    missing_license_reviews: list[str]
    incomplete_evidence: list[str]
    incomplete_license_reviews: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "repo_root": self.repo_root,
            "require_content": self.require_content,
            "rows": [row.to_dict() for row in self.rows],
            "missing_evidence": self.missing_evidence,
            "missing_license_reviews": self.missing_license_reviews,
            "incomplete_evidence": self.incomplete_evidence,
            "incomplete_license_reviews": self.incomplete_license_reviews,
        }

    def to_markdown(self) -> str:
        table_rows = [
            {
                "task": row.task_id,
                "priority": row.priority,
                "evidence": "yes" if row.evidence_present else "no",
                "license_review": (
                    "yes"
                    if row.license_review_required and row.license_review_present
                    else "missing"
                    if row.license_review_required
                    else "n/a"
                ),
                "status": "READY" if row.ok else "MISSING",
                "target": row.evidence_target,
            }
            for row in self.rows
        ]
        lines = [
            "# Stable-ASR Reference Evidence Audit",
            "",
            f"- status: `{'READY' if self.ok else 'NOT_READY'}`",
            f"- repo_root: `{self.repo_root}`",
            f"- require_content: `{self.require_content}`",
            f"- tasks: `{len(self.rows)}`",
            f"- missing_evidence: `{len(self.missing_evidence)}`",
            f"- missing_license_reviews: `{len(self.missing_license_reviews)}`",
            f"- incomplete_evidence: `{len(self.incomplete_evidence)}`",
            f"- incomplete_license_reviews: `{len(self.incomplete_license_reviews)}`",
            "",
            dict_table(table_rows),
            "",
            "## Missing Evidence",
            "",
        ]
        lines.extend(f"- `{item}`" for item in self.missing_evidence) if self.missing_evidence else lines.append("- none")
        lines.extend(["", "## Missing License Reviews", ""])
        lines.extend(f"- `{item}`" for item in self.missing_license_reviews) if self.missing_license_reviews else lines.append("- none")
        lines.extend(["", "## Incomplete Evidence", ""])
        lines.extend(f"- `{item}`" for item in self.incomplete_evidence) if self.incomplete_evidence else lines.append("- none")
        lines.extend(["", "## Incomplete License Reviews", ""])
        lines.extend(f"- `{item}`" for item in self.incomplete_license_reviews) if self.incomplete_license_reviews else lines.append("- none")
        lines.extend(
            [
                "",
                "## Evidence Rule",
                "",
                (
                    "Collection registry entries are plans. A task becomes release evidence only after its "
                    "`evidence_target` exists and any required license review target is present."
                ),
            ]
        )
        return "\n".join(lines) + "\n"

    def to_text(self) -> str:
        lines = [
            f"reference_evidence_audit: {'READY' if self.ok else 'NOT_READY'}",
            f"repo_root: {self.repo_root}",
            f"require_content: {self.require_content}",
            f"tasks: {len(self.rows)}",
            f"missing_evidence: {len(self.missing_evidence)}",
            f"missing_license_reviews: {len(self.missing_license_reviews)}",
            f"incomplete_evidence: {len(self.incomplete_evidence)}",
            f"incomplete_license_reviews: {len(self.incomplete_license_reviews)}",
        ]
        for row in self.rows:
            marker = "OK" if row.ok else "MISSING"
            lines.append(f"- {marker} {row.task_id}: {row.evidence_target}")
        return "\n".join(lines)


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


def reference_workqueue_evidence_markdown(workqueue: dict[str, Any]) -> str:
    """Render contributor-facing evidence templates for each reference task."""

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
            "license_review": task["license_review_target"] if task["license_review_required"] else "not required",
        }
        for task in tasks
    ]
    lines = [
        "# Stable-ASR Reference Evidence Templates",
        "",
        (
            "These templates describe the minimum evidence expected for each ASR or turn reference task. "
            "They are not evidence by themselves; copy the relevant section into the target file only after "
            "running the external system, writing the bridge note, or completing the license review."
        ),
        "",
        f"- source_workqueue_id: `{workqueue.get('id', '')}`",
        f"- tasks: `{len(tasks)}`",
        "",
        dict_table(rows),
        "",
        "## Acceptance Rule",
        "",
        "A task is complete only when:",
        "",
        "- its `evidence_target` exists and contains concrete commands, versions, inputs, outputs, and validation results",
        "- any required `license_review_target` exists before vendoring code, weights, fixtures, or long snippets",
        "- `stable-asr reference-workqueue --audit-evidence --require-content --repo-root .` reports the task as present",
        "",
    ]
    for task in tasks:
        blocked_by = task.get("blocked_by", [])
        blocked = ", ".join(str(item) for item in blocked_by) if isinstance(blocked_by, list) and blocked_by else "none"
        actions = task.get("stable_asr_actions", [])
        action_text = ", ".join(str(item) for item in actions) if isinstance(actions, list) else ""
        lines.extend(
            [
                f"## {task['task_id']}",
                "",
                f"- name: `{task['name']}`",
                f"- priority: `{task['priority']}`",
                f"- collection_type: `{task['collection_type']}`",
                f"- category: `{task['category']}`",
                f"- acquisition_track: `{task['acquisition_track']}`",
                f"- evidence_target: `{task['evidence_target']}`",
                f"- license: `{task['license']}`",
                f"- license_review_required: `{task['license_review_required']}`",
                f"- license_review_target: `{task['license_review_target']}`",
                f"- blocked_by: `{blocked}`",
                f"- source_url: `{task['source_url']}`",
                f"- docs_url: `{task['docs_url']}`",
                f"- stable_asr_actions: `{action_text}`",
                "",
                "Required evidence sections:",
                "",
                "1. Upstream version and source",
                "2. Inputs used",
                "3. Command, script, or bridge implementation notes",
                "4. Output paths and schema or validation commands",
                "5. Metrics, examples, or failure notes relevant to Stable-ASR",
                "6. License and redistribution decision",
                "",
                "Validation commands:",
                "",
                "```bash",
                "stable-asr reference-workqueue --audit-evidence --require-content --repo-root . --output runs/REFERENCE_EVIDENCE_AUDIT.md",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def reference_workqueue_issues_markdown(workqueue: dict[str, Any]) -> str:
    """Render issue-ready Markdown tasks for collecting reference evidence."""

    validation = validate_reference_workqueue(workqueue)
    if not validation.ok:
        raise ValueError(validation.to_text())
    tasks = workqueue["tasks"]
    rows = [
        {
            "task": task["task_id"],
            "priority": task["priority"],
            "track": task["acquisition_track"],
            "license_review": "yes" if task["license_review_required"] else "no",
        }
        for task in tasks
    ]
    lines = [
        "# Stable-ASR Reference Collection Issues",
        "",
        (
            "Use these issue stubs to assign upstream ASR, turn-taking, and full-duplex collection work. "
            "They are task descriptions only; they do not count as evidence until the strict evidence audit passes."
        ),
        "",
        f"- source_workqueue_id: `{workqueue.get('id', '')}`",
        f"- tasks: `{len(tasks)}`",
        "",
        dict_table(rows),
        "",
        "## Shared Acceptance",
        "",
        "- The evidence target contains concrete upstream version, input, command, output, metric, and failure-note details.",
        "- Required license reviews are completed before vendoring code, weights, fixtures, or long snippets.",
        "- `stable-asr reference-workqueue --audit-evidence --require-content --repo-root .` reports the task as READY.",
        "",
    ]
    for task in tasks:
        actions = task.get("stable_asr_actions", [])
        action_text = "\n".join(f"- [ ] `{action}`" for action in actions) if isinstance(actions, list) else "- [ ] collect evidence"
        labels = ", ".join(
            [
                "stable-asr",
                "reference-collection",
                str(task["collection_type"]),
                str(task["priority"]),
            ]
        )
        license_command = (
            f"- [ ] Fill `{task['license_review_target']}` and record the redistribution decision."
            if task["license_review_required"]
            else "- [x] No separate license review required by registry policy."
        )
        lines.extend(
            [
                f"## {task['task_id']}",
                "",
                f"**Title:** Collect `{task['name']}` reference evidence for Stable-ASR",
                "",
                f"**Labels:** `{labels}`",
                "",
                "### Context",
                "",
                f"- collection_type: `{task['collection_type']}`",
                f"- reference_id: `{task['reference_id']}`",
                f"- priority: `{task['priority']}`",
                f"- category: `{task['category']}`",
                f"- acquisition_track: `{task['acquisition_track']}`",
                f"- source_url: `{task['source_url']}`",
                f"- docs_url: `{task['docs_url']}`",
                f"- reference_use: {task['reference_use']}",
                "",
                "### Work",
                "",
                action_text,
                license_command,
                f"- [ ] Write evidence to `{task['evidence_target']}`.",
                "- [ ] Run the strict evidence audit and attach the failing/passing summary.",
                "",
                "### Commands",
                "",
                "```bash",
                "stable-asr reference-workqueue --format evidence-markdown --output runs/REFERENCE_EVIDENCE_TEMPLATES.md",
                "stable-asr reference-workqueue --audit-evidence --require-content --repo-root . --output runs/REFERENCE_EVIDENCE_AUDIT.md",
                "```",
                "",
                "### Acceptance",
                "",
                f"- [ ] `{task['evidence_target']}` exists and passes strict content audit.",
                license_command,
                "- [ ] No vendored upstream asset is added unless the license review explicitly allows it.",
                "",
            ]
        )
    return "\n".join(lines)


def reference_workqueue_license_review_markdown(workqueue: dict[str, Any]) -> str:
    """Render license-review templates for workqueue tasks that require review."""

    validation = validate_reference_workqueue(workqueue)
    if not validation.ok:
        raise ValueError(validation.to_text())
    review_tasks = [task for task in workqueue["tasks"] if task["license_review_required"]]
    rows = [
        {
            "task": task["task_id"],
            "priority": task["priority"],
            "license": task["license"],
            "target": task["license_review_target"],
        }
        for task in review_tasks
    ]
    lines = [
        "# Stable-ASR Reference License Review Templates",
        "",
        (
            "Fill these templates before copying upstream code, weights, generated fixtures, datasets, or long snippets. "
            "Keeping a template pending is allowed for link-only notes and command adapters, but it is not release evidence."
        ),
        "",
        f"- source_workqueue_id: `{workqueue.get('id', '')}`",
        f"- review_required: `{len(review_tasks)}`",
        "",
        dict_table(rows) if rows else "No reference tasks require manual license review.",
        "",
        "## Shared Review Rules",
        "",
        "- Confirm the upstream repository, model, dataset, and documentation licenses separately when they differ.",
        "- Record whether Stable-ASR uses only links and command adapters or copies any upstream asset.",
        "- Do not commit copied upstream code, weights, generated fixtures, datasets, or long snippets unless approved below.",
        "- A copied review is complete only when `status`, `reviewer`, `approved_uses`, `prohibited_uses`, and `required_notices` are filled.",
        "",
    ]
    for task in review_tasks:
        actions = task.get("stable_asr_actions", [])
        action_text = ", ".join(str(action) for action in actions) if isinstance(actions, list) else ""
        lines.extend(
            [
                f"## {task['task_id']}",
                "",
                f"Copy this section into `{task['license_review_target']}` after a human review is complete.",
                "",
                f"# License Review: {task['name']}",
                "",
                f"- reference_id: `{task['reference_id']}`",
                f"- collection_type: `{task['collection_type']}`",
                f"- priority: `{task['priority']}`",
                f"- declared_license: `{task['license']}`",
                f"- default_policy: `{task['policy']}`",
                f"- source_url: `{task['source_url']}`",
                f"- docs_url: `{task['docs_url']}`",
                f"- intended_stable_asr_use: {task['reference_use']}",
                f"- planned_actions: {action_text}",
                "",
                "### Review Checklist",
                "",
                "- [ ] Confirm upstream repository license.",
                "- [ ] Confirm model, dataset, checkpoint, fixture, and documentation licenses if separate.",
                "- [ ] Decide whether Stable-ASR remains link/command-adapter-only or may redistribute specific assets.",
                "- [ ] Record required notices, attribution text, redistribution limits, and commercial-use limits.",
                "- [ ] Confirm no copied upstream asset is committed before this review is approved.",
                "",
                "## Decision",
                "",
                "- status: pending",
                "- reviewer:",
                "- reviewed_at:",
                "- approved_uses:",
                "- prohibited_uses:",
                "- required_notices:",
                "- notes:",
                "",
            ]
        )
    return "\n".join(lines)


def reference_workqueue_jsonl(workqueue: dict[str, Any]) -> str:
    validation = validate_reference_workqueue(workqueue)
    if not validation.ok:
        raise ValueError(validation.to_text())
    return "\n".join(json.dumps(task, ensure_ascii=False, sort_keys=True) for task in workqueue["tasks"]) + "\n"


def audit_reference_workqueue_evidence(
    workqueue: dict[str, Any],
    *,
    repo_root: str | Path = ".",
    require_content: bool = False,
) -> ReferenceEvidenceAuditReport:
    """Check whether reference workqueue evidence and license-review targets exist."""

    validation = validate_reference_workqueue(workqueue)
    if not validation.ok:
        raise ValueError(validation.to_text())
    root = Path(repo_root)
    rows: list[ReferenceEvidenceAuditRow] = []
    missing_evidence: list[str] = []
    missing_license_reviews: list[str] = []
    incomplete_evidence: list[str] = []
    incomplete_license_reviews: list[str] = []
    for task in workqueue["tasks"]:
        evidence_target = str(task["evidence_target"])
        evidence_present = _target_exists(evidence_target, repo_root=root)
        evidence_content_errors = (
            _evidence_content_errors(evidence_target, repo_root=root)
            if require_content and evidence_present
            else []
        )
        evidence_content_checked = require_content and evidence_present
        evidence_content_ok = not evidence_content_errors
        license_review_required = bool(task["license_review_required"])
        license_review_target = str(task["license_review_target"])
        license_review_present = (not license_review_required) or _target_exists(license_review_target, repo_root=root)
        license_review_content_errors = (
            _license_review_content_errors(license_review_target, repo_root=root)
            if require_content and license_review_required and license_review_present
            else []
        )
        license_review_content_checked = require_content and license_review_required and license_review_present
        license_review_content_ok = not license_review_content_errors
        task_id = str(task["task_id"])
        if not evidence_present:
            missing_evidence.append(f"{task_id}:{evidence_target}")
        if license_review_required and not license_review_present:
            missing_license_reviews.append(f"{task_id}:{license_review_target}")
        if evidence_content_errors:
            incomplete_evidence.append(f"{task_id}:{evidence_target}:{'; '.join(evidence_content_errors)}")
        if license_review_content_errors:
            incomplete_license_reviews.append(
                f"{task_id}:{license_review_target}:{'; '.join(license_review_content_errors)}"
            )
        rows.append(
            ReferenceEvidenceAuditRow(
                task_id=task_id,
                collection_type=str(task["collection_type"]),
                reference_id=str(task["reference_id"]),
                priority=str(task["priority"]),
                evidence_target=evidence_target,
                evidence_present=evidence_present,
                license_review_required=license_review_required,
                license_review_target=license_review_target,
                license_review_present=license_review_present,
                evidence_content_checked=evidence_content_checked,
                evidence_content_ok=evidence_content_ok,
                evidence_content_errors=evidence_content_errors,
                license_review_content_checked=license_review_content_checked,
                license_review_content_ok=license_review_content_ok,
                license_review_content_errors=license_review_content_errors,
                ok=evidence_present and license_review_present and evidence_content_ok and license_review_content_ok,
            )
        )
    return ReferenceEvidenceAuditReport(
        ok=all(row.ok for row in rows),
        repo_root=str(root),
        require_content=require_content,
        rows=rows,
        missing_evidence=missing_evidence,
        missing_license_reviews=missing_license_reviews,
        incomplete_evidence=incomplete_evidence,
        incomplete_license_reviews=incomplete_license_reviews,
    )


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
    return _target_path(path, repo_root=repo_root).exists()


def _target_path(path: str, *, repo_root: Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = repo_root / target
    return target


def _evidence_content_errors(path: str, *, repo_root: Path) -> list[str]:
    target = _target_path(path, repo_root=repo_root)
    if target.is_dir():
        return [] if any(target.iterdir()) else ["directory is empty"]
    suffix = target.suffix.lower()
    if suffix == ".jsonl":
        return _jsonl_content_errors(target)
    if suffix == ".json":
        return _json_content_errors(target)
    if suffix in {".tsv", ".csv"}:
        return _delimited_content_errors(target)
    return _markdown_section_errors(target, required_sections=EVIDENCE_MARKDOWN_REQUIRED_SECTIONS)


def _license_review_content_errors(path: str, *, repo_root: Path) -> list[str]:
    target = _target_path(path, repo_root=repo_root)
    errors = _markdown_section_errors(target, required_sections=LICENSE_REVIEW_REQUIRED_SECTIONS)
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return errors
    lowered = text.lower()
    if re.search(r"^\s*-\s*status:\s*pending\s*$", lowered, flags=re.MULTILINE):
        errors.append("decision status is still pending")
    if re.search(r"^\s*-\s*reviewer:\s*$", text, flags=re.MULTILINE):
        errors.append("reviewer is blank")
    return errors


def _jsonl_content_errors(path: Path) -> list[str]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError:
        return ["file is not utf-8 text"]
    if not lines:
        return ["jsonl has no records"]
    errors: list[str] = []
    for index, line in enumerate(lines, start=1):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {index} is not valid JSON: {exc.msg}")
            break
    return errors


def _json_content_errors(path: Path) -> list[str]:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return ["file is not utf-8 text"]
    except json.JSONDecodeError as exc:
        return [f"file is not valid JSON: {exc.msg}"]
    return []


def _delimited_content_errors(path: Path) -> list[str]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError:
        return ["file is not utf-8 text"]
    return [] if len(lines) >= 2 else ["file needs a header and at least one data row"]


def _markdown_section_errors(path: Path, *, required_sections: tuple[str, ...]) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return ["file is not utf-8 text"]
    errors: list[str] = []
    markers = {section: _find_markdown_marker(lines, section) for section in required_sections}
    missing = [section for section, marker in markers.items() if marker is None]
    if missing:
        errors.append("missing section(s): " + ", ".join(missing))
    for section, marker in markers.items():
        if marker is None:
            continue
        if section.endswith(":"):
            if not _has_filled_markdown_content(marker[1]):
                errors.append(f"field `{section}` is blank")
        elif not section.startswith("#"):
            content_lines = [marker[1], *_markdown_section_body(lines, marker[0], required_sections)]
            if not _has_filled_markdown_content("\n".join(content_lines)):
                errors.append(f"section `{section}` has no filled content")
    return errors


def _find_markdown_marker(lines: list[str], marker: str) -> tuple[int, str] | None:
    normalized_marker = marker.strip().lower()
    if marker.startswith("#"):
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.lower().startswith(normalized_marker):
                return index, stripped[len(marker) :].strip()
        return None

    marker_name = marker.rstrip(":").strip()
    marker_requires_colon = marker.endswith(":")
    pattern = re.compile(
        rf"^\s*(?:#+\s*|\d+\.\s+|[-*]\s+)?{re.escape(marker_name)}"
        rf"{':' if marker_requires_colon else ':?'}\s*(?P<inline>.*)$",
        flags=re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            return index, match.group("inline").strip()
    return None


def _markdown_section_body(lines: list[str], start_index: int, required_sections: tuple[str, ...]) -> list[str]:
    body: list[str] = []
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            body.append(line)
            continue
        if stripped.lower() in {"validation commands:", "required evidence sections:"}:
            break
        if stripped.startswith("#"):
            break
        if any(_find_markdown_marker([line], section) is not None for section in required_sections):
            break
        body.append(line)
    return body


def _has_filled_markdown_content(text: str) -> bool:
    placeholder_pattern = re.compile(r"^(?:tbd|todo|placeholder|fill me|\.\.\.)$", flags=re.IGNORECASE)
    for line in text.splitlines():
        stripped = line.strip().strip("`*_ ")
        if not stripped or stripped in {"-", ":", "```"}:
            continue
        if placeholder_pattern.match(stripped):
            continue
        return True
    return False


def _tsv_cell(value: object) -> str:
    if isinstance(value, list):
        value = ",".join(str(item) for item in value)
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")
