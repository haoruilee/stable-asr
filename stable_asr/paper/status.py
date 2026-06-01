"""Single-page paper readiness status report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.paper.acquisition_pack import audit_acquisition_assignments
from stable_asr.paper.audit import audit_paper_artifacts
from stable_asr.paper.final_config import audit_final_run_files, load_final_run_config
from stable_asr.paper.handoff import audit_final_handoff
from stable_asr.paper.parity import audit_paper_parity
from stable_asr.schema_validation import validate_schema_file


@dataclass(frozen=True)
class FinalAssignmentStatus:
    ready: bool
    assignment_tracker: str
    assignment_audit: str
    missing: list[str]
    errors: list[str]
    blocking_release: list[str]
    unassigned: list[str]
    missing_due_dates: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "assignment_tracker": self.assignment_tracker,
            "assignment_audit": self.assignment_audit,
            "missing": self.missing,
            "errors": self.errors,
            "blocking_release": self.blocking_release,
            "unassigned": self.unassigned,
            "missing_due_dates": self.missing_due_dates,
        }


@dataclass(frozen=True)
class FinalHandoffStatus:
    ready: bool
    handoff: str
    handoff_schema_validation: str
    handoff_audit: str
    missing: list[str]
    errors: list[str]
    warnings: list[str]
    checked_paths: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "handoff": self.handoff,
            "handoff_schema_validation": self.handoff_schema_validation,
            "handoff_audit": self.handoff_audit,
            "missing": self.missing,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_paths": self.checked_paths,
        }


@dataclass(frozen=True)
class PaperStatusNextAction:
    id: str
    title: str
    status: str
    reason: str
    commands: list[str]
    artifacts: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "reason": self.reason,
            "commands": self.commands,
            "artifacts": self.artifacts,
        }


@dataclass(frozen=True)
class PaperStatusReport:
    ok: bool
    smoke_ready: bool
    structural_ready: bool
    final_ready: bool
    final_inputs_ready: bool
    missing_final_inputs: list[str]
    final_assignment_ready: bool
    final_assignment: FinalAssignmentStatus
    final_handoff_ready: bool
    final_handoff: FinalHandoffStatus
    next_actions: list[PaperStatusNextAction]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "smoke_ready": self.smoke_ready,
            "structural_ready": self.structural_ready,
            "final_ready": self.final_ready,
            "final_inputs_ready": self.final_inputs_ready,
            "missing_final_inputs": self.missing_final_inputs,
            "final_assignment_ready": self.final_assignment_ready,
            "final_assignment": self.final_assignment.to_dict(),
            "final_handoff_ready": self.final_handoff_ready,
            "final_handoff": self.final_handoff.to_dict(),
            "next_actions": [action.to_dict() for action in self.next_actions],
        }

    def to_markdown(self) -> str:
        rows = [
            {"gate": "doctor", "status": "OK" if self.ok else "FAILED", "meaning": "required configs and environment checks"},
            {"gate": "smoke_ready", "status": _status(self.smoke_ready), "meaning": "paper result and artifact bundle shape"},
            {"gate": "structural_ready", "status": _status(self.structural_ready), "meaning": "stable-worldmodel-style structural evidence"},
            {"gate": "final_inputs_ready", "status": _status(self.final_inputs_ready), "meaning": "real final corpora/splits/predictions exist"},
            {
                "gate": "final_assignment_ready",
                "status": _status(self.final_assignment_ready),
                "meaning": "final input owners, due dates, release blockers, and audit artifact",
            },
            {
                "gate": "final_handoff_ready",
                "status": _status(self.final_handoff_ready),
                "meaning": "filled final handoff passes path, license, verification, and checksum audit",
            },
            {
                "gate": "final_ready",
                "status": _status(self.final_ready),
                "meaning": "no final-scale parity, input, assignment, or handoff gaps remain",
            },
        ]
        lines = [
            "# Stable-ASR Paper Status",
            "",
            dict_table(rows),
            "",
            "## Missing Final Inputs",
            "",
        ]
        if self.missing_final_inputs:
            lines.extend(f"- `{path}`" for path in self.missing_final_inputs)
        else:
            lines.append("- None")
        lines.extend(["", "## Next Actions", ""])
        if self.next_actions:
            lines.append(
                dict_table(
                    [
                        {
                            "id": action.id,
                            "status": action.status,
                            "title": action.title,
                            "artifacts": len(action.artifacts),
                        }
                        for action in self.next_actions
                    ]
                )
            )
            for action in self.next_actions:
                lines.extend(
                    [
                        "",
                        f"### {action.id}",
                        "",
                        f"- status: `{action.status}`",
                        f"- reason: {action.reason}",
                        "",
                        "Commands:",
                        "",
                    ]
                )
                _extend_markdown_list(lines, action.commands)
                lines.extend(["", "Artifacts:", ""])
                _extend_markdown_list(lines, action.artifacts)
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "## Final Assignment Gate",
                "",
                f"- assignment tracker: `{self.final_assignment.assignment_tracker}`",
                f"- assignment audit: `{self.final_assignment.assignment_audit}`",
                "",
                "Missing evidence:",
                "",
            ]
        )
        _extend_markdown_list(lines, self.final_assignment.missing)
        lines.extend(["", "Errors:", ""])
        _extend_markdown_list(lines, self.final_assignment.errors)
        lines.extend(["", "Release blockers:", ""])
        _extend_markdown_list(lines, self.final_assignment.blocking_release)
        lines.extend(["", "Unassigned:", ""])
        _extend_markdown_list(lines, self.final_assignment.unassigned)
        lines.extend(["", "Missing due dates:", ""])
        _extend_markdown_list(lines, self.final_assignment.missing_due_dates)
        lines.extend(
            [
                "",
                "## Final Handoff Gate",
                "",
                f"- handoff: `{self.final_handoff.handoff}`",
                f"- handoff schema validation: `{self.final_handoff.handoff_schema_validation}`",
                f"- handoff audit: `{self.final_handoff.handoff_audit}`",
                "",
                "Missing evidence:",
                "",
            ]
        )
        _extend_markdown_list(lines, self.final_handoff.missing)
        lines.extend(["", "Errors:", ""])
        _extend_markdown_list(lines, self.final_handoff.errors)
        lines.extend(["", "Warnings:", ""])
        _extend_markdown_list(lines, self.final_handoff.warnings)
        lines.extend(["", "Checked paths:", ""])
        _extend_markdown_list(lines, self.final_handoff.checked_paths)
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                (
                    "A smoke-ready or structurally ready report does not mean the final paper experiments are complete. "
                    "Final readiness requires real final inputs and a parity audit with no remaining final-scale gaps."
                ),
                "",
            ]
        )
        return "\n".join(lines)


def paper_status(
    *,
    repo_root: str | Path = ".",
    release_dir: str | Path | None = None,
    results_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> PaperStatusReport:
    repo_root = Path(repo_root)
    if release_dir is not None:
        release_dir = Path(release_dir)
        if results_path is None:
            results_path = release_dir / "paper" / "paper_results.json"
        if artifacts_dir is None:
            artifacts_dir = release_dir / "artifacts"
    from stable_asr.doctor import run_doctor

    doctor = run_doctor(repo_root=repo_root, check_final_files=True)
    smoke_ready = False
    if results_path is not None and artifacts_dir is not None:
        smoke_ready = audit_paper_artifacts(results_path, artifacts_dir).ok
    parity = audit_paper_parity(repo_root=repo_root, results_path=results_path, artifacts_dir=artifacts_dir)
    config = load_final_run_config(repo_root / "configs" / "final" / "paper_final.json")
    final_files = audit_final_run_files(config, repo_root=repo_root)
    assignment = _final_assignment_status(config, repo_root=repo_root)
    handoff = _final_handoff_status(config, repo_root=repo_root)
    missing = [
        check.path
        for check in final_files.checks
        if check.required and not check.ok
    ]
    next_actions = _paper_status_next_actions(
        config,
        missing_final_inputs=missing,
        final_inputs_ready=final_files.ok,
        assignment=assignment,
        handoff=handoff,
        final_ready=parity.final_ready and final_files.ok and assignment.ready and handoff.ready,
    )
    return PaperStatusReport(
        ok=doctor.ok,
        smoke_ready=smoke_ready,
        structural_ready=parity.ok,
        final_ready=parity.final_ready and final_files.ok and assignment.ready and handoff.ready,
        final_inputs_ready=final_files.ok,
        missing_final_inputs=missing,
        final_assignment_ready=assignment.ready,
        final_assignment=assignment,
        final_handoff_ready=handoff.ready,
        final_handoff=handoff,
        next_actions=next_actions,
    )


def write_paper_status_json(report: PaperStatusReport, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def write_paper_status_markdown(report: PaperStatusReport, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_markdown(), encoding="utf-8")
    return str(path)


def _status(ok: bool) -> str:
    return "READY" if ok else "NOT_READY"


def _final_assignment_status(config: dict[str, Any], *, repo_root: Path) -> FinalAssignmentStatus:
    artifacts = config.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    tracker = _resolve_repo_path(
        artifacts.get("assignment_tracker", "runs/final_acquisition_pack/acquisition/assignments.json"),
        repo_root=repo_root,
    )
    audit = _resolve_repo_path(
        artifacts.get("assignment_audit", "runs/final/FINAL_ASSIGNMENT_AUDIT.md"),
        repo_root=repo_root,
    )
    missing: list[str] = []
    errors: list[str] = []
    blocking_release: list[str] = []
    unassigned: list[str] = []
    missing_due_dates: list[str] = []

    if not tracker.exists():
        missing.append(str(tracker))
    else:
        report = audit_acquisition_assignments(
            tracker,
            require_owner=True,
            require_due_date=True,
            require_ready=True,
        )
        errors.extend(report.errors)
        blocking_release.extend(report.blocking_release)
        unassigned.extend(report.unassigned)
        missing_due_dates.extend(report.missing_due_dates)

    if not audit.exists():
        missing.append(str(audit))

    return FinalAssignmentStatus(
        ready=not missing and not errors,
        assignment_tracker=str(tracker),
        assignment_audit=str(audit),
        missing=missing,
        errors=errors,
        blocking_release=blocking_release,
        unassigned=unassigned,
        missing_due_dates=missing_due_dates,
    )


def _final_handoff_status(config: dict[str, Any], *, repo_root: Path) -> FinalHandoffStatus:
    artifacts = config.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    handoff = _resolve_repo_path(
        artifacts.get("handoff", "runs/final/FINAL_INPUT_HANDOFF.json"),
        repo_root=repo_root,
    )
    handoff_schema_validation = _resolve_repo_path(
        artifacts.get("handoff_schema_validation", "runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md"),
        repo_root=repo_root,
    )
    handoff_audit = _resolve_repo_path(
        artifacts.get("handoff_audit", "runs/final/FINAL_HANDOFF_AUDIT.md"),
        repo_root=repo_root,
    )
    missing: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    checked_paths: list[str] = []
    if not handoff.exists():
        missing.append(str(handoff))
    else:
        schema_report = validate_schema_file(handoff, schema_id="stable_asr.final_handoff.v0")
        errors.extend(f"schema:{issue.path}:{issue.detail}" for issue in schema_report.issues)
        report = audit_final_handoff(handoff, repo_root=repo_root, require_checksums=True)
        errors.extend(report.errors)
        warnings.extend(report.warnings)
        checked_paths.extend(report.checked_paths)
    if not handoff_schema_validation.exists():
        missing.append(str(handoff_schema_validation))
    if not handoff_audit.exists():
        missing.append(str(handoff_audit))
    return FinalHandoffStatus(
        ready=not missing and not errors,
        handoff=str(handoff),
        handoff_schema_validation=str(handoff_schema_validation),
        handoff_audit=str(handoff_audit),
        missing=missing,
        errors=errors,
        warnings=warnings,
        checked_paths=checked_paths,
    )


def _paper_status_next_actions(
    config: dict[str, Any],
    *,
    missing_final_inputs: list[str],
    final_inputs_ready: bool,
    assignment: FinalAssignmentStatus,
    handoff: FinalHandoffStatus,
    final_ready: bool,
) -> list[PaperStatusNextAction]:
    artifacts = config.get("artifacts", {}) if isinstance(config.get("artifacts", {}), dict) else {}
    paper_results = str(artifacts.get("paper_results", "runs/final/paper_results.json"))
    bundle_dir = str(artifacts.get("bundle_dir", "runs/final/artifacts"))
    model_card = str(artifacts.get("model_card", "runs/final/MODEL_CARD.md"))
    archive = str(artifacts.get("artifact_archive", "runs/final/artifacts.tar.gz"))
    config_path = "configs/final/paper_final.json"
    actions = [
        PaperStatusNextAction(
            id="collect_final_inputs",
            title="Collect real final corpora, turn splits, VoiceWorld records, external predictions, and ASR command exports",
            status="done" if final_inputs_ready else "needed",
            reason=(
                "all required final input paths exist"
                if final_inputs_ready
                else f"{len(missing_final_inputs)} required final input path(s) are missing"
            ),
            commands=[
                f"stable-asr final-config --config {config_path} --check-files",
                f"stable-asr final-config --config {config_path} --plan-missing --output runs/final/FINAL_RUN_ACTION_PLAN.md",
                "stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack --repo-root .",
            ],
            artifacts=["runs/final/FINAL_RUN_ACTION_PLAN.md", *missing_final_inputs],
        ),
        PaperStatusNextAction(
            id="fill_final_assignment",
            title="Assign owners, due dates, release-blocker state, and evidence handoff status",
            status="done" if assignment.ready else "needed",
            reason=(
                "assignment tracker and strict audit are ready"
                if assignment.ready
                else _gate_reason(
                    missing=len(assignment.missing),
                    errors=len(assignment.errors),
                    blockers=len(assignment.blocking_release),
                    unassigned=len(assignment.unassigned),
                    missing_due_dates=len(assignment.missing_due_dates),
                )
            ),
            commands=[
                "stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack --repo-root .",
                (
                    f"stable-asr final-assignment-audit --input {assignment.assignment_tracker} "
                    f"--require-owner --require-due-date --require-ready --output {assignment.assignment_audit}"
                ),
            ],
            artifacts=[assignment.assignment_tracker, assignment.assignment_audit],
        ),
        PaperStatusNextAction(
            id="complete_final_handoff",
            title="Fill final handoff metadata, checksums, schema validation, and handoff audit",
            status="done" if handoff.ready else "needed",
            reason=(
                "handoff, schema validation, and checksum audit are ready"
                if handoff.ready
                else _gate_reason(
                    missing=len(handoff.missing),
                    errors=len(handoff.errors),
                    warnings=len(handoff.warnings),
                    checked_paths=len(handoff.checked_paths),
                )
            ),
            commands=[
                f"stable-asr final-handoff-template --output {handoff.handoff}",
                f"stable-asr final-handoff-checksums --input {handoff.handoff} --repo-root . --output {handoff.handoff}",
                f"stable-asr validate-schema-file --input {handoff.handoff} --schema-id stable_asr.final_handoff.v0 --output {handoff.handoff_schema_validation}",
                f"stable-asr final-handoff-audit --input {handoff.handoff} --repo-root . --require-checksums --output {handoff.handoff_audit}",
            ],
            artifacts=[handoff.handoff, handoff.handoff_schema_validation, handoff.handoff_audit],
        ),
        PaperStatusNextAction(
            id="assemble_final_release",
            title="Build final paper results, artifact bundle, model card, archive, and release audit",
            status="done" if final_ready else ("ready" if final_inputs_ready and assignment.ready and handoff.ready else "blocked"),
            reason=(
                "final release gates are ready"
                if final_ready
                else "requires final inputs, assignment audit, handoff audit, and final parity gates"
            ),
            commands=[
                f"stable-asr final-results --config {config_path} --output {paper_results}",
                f"stable-asr paper-bundle --results {paper_results} --output-dir {bundle_dir}",
                f"stable-asr make-card model --input configs/models/stable_asr_models.json --model-id nanoturn_pico --metrics runs/final/nanoturn/metrics.json --output {model_card}",
                f"stable-asr paper-archive --artifacts-dir {bundle_dir} --output {archive}",
                f"stable-asr paper-archive-verify --archive {archive}",
                f"stable-asr paper-release-audit --repo-root . --results {paper_results} --artifacts-dir {bundle_dir} --model-card {model_card} --require-final-ready",
            ],
            artifacts=[paper_results, bundle_dir, model_card, archive],
        ),
    ]
    return actions


def _gate_reason(**counts: int) -> str:
    parts = [f"{name}={count}" for name, count in counts.items() if count]
    return "; ".join(parts) if parts else "gate is not ready"


def _resolve_repo_path(path: object, *, repo_root: Path) -> Path:
    resolved = Path(str(path))
    if resolved.is_absolute():
        return resolved
    return repo_root / resolved


def _extend_markdown_list(lines: list[str], items: list[str]) -> None:
    if items:
        lines.extend(f"- `{item}`" for item in items)
    else:
        lines.append("- None")
