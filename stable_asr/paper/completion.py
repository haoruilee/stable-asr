"""Goal-level completion audit for Stable-ASR release work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stable_asr.eval.report import dict_table
from stable_asr.paper.parity import audit_paper_parity
from stable_asr.paper.platform_parity import audit_platform_parity
from stable_asr.paper.status import paper_status
from stable_asr.references import audit_reference_workqueue_evidence, reference_workqueue_from_registries
from stable_asr.roadmap import load_roadmap, roadmap_status


OBJECTIVE_STATEMENT = "完成路线图,形成优秀的平台和论文,提供有价值的仓库"


@dataclass(frozen=True)
class CompletionAuditItem:
    requirement: str
    evidence: str
    command: str
    ok: bool
    detail: str
    blockers: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement": self.requirement,
            "evidence": self.evidence,
            "command": self.command,
            "ok": self.ok,
            "detail": self.detail,
            "blockers": self.blockers,
        }


@dataclass(frozen=True)
class CompletionAuditReport:
    ok: bool
    objective: str
    repo_root: str
    items: list[CompletionAuditItem]

    @property
    def blockers(self) -> list[str]:
        rows: list[str] = []
        for item in self.items:
            rows.extend(f"{item.requirement}: {blocker}" for blocker in item.blockers)
        return rows

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "objective": self.objective,
            "repo_root": self.repo_root,
            "items": [item.to_dict() for item in self.items],
            "blockers": self.blockers,
        }

    def to_text(self) -> str:
        lines = [
            f"completion_audit: {'READY' if self.ok else 'NOT_READY'}",
            f"objective: {self.objective}",
            f"repo_root: {self.repo_root}",
            f"items: {len(self.items)}",
            f"blockers: {len(self.blockers)}",
        ]
        for item in self.items:
            status = "OK" if item.ok else "MISSING"
            lines.append(f"- {status} {item.requirement}: {item.detail}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        rows = [
            {
                "requirement": item.requirement,
                "status": "READY" if item.ok else "NOT_READY",
                "evidence": item.evidence,
                "blockers": len(item.blockers),
            }
            for item in self.items
        ]
        lines = [
            "# Stable-ASR Completion Audit",
            "",
            f"- objective: `{self.objective}`",
            f"- status: `{'READY' if self.ok else 'NOT_READY'}`",
            f"- repo_root: `{self.repo_root}`",
            f"- blockers: `{len(self.blockers)}`",
            "",
            "## Prompt-To-Artifact Checklist",
            "",
            dict_table(rows),
            "",
            "## Requirement Details",
            "",
        ]
        for item in self.items:
            lines.extend(
                [
                    f"### {item.requirement}",
                    "",
                    f"- status: `{'READY' if item.ok else 'NOT_READY'}`",
                    f"- evidence: `{item.evidence}`",
                    f"- command: `{item.command}`",
                    f"- detail: {item.detail}",
                    "",
                    "Blockers:",
                    "",
                ]
            )
            if item.blockers:
                lines.extend(f"- `{blocker}`" for blocker in item.blockers)
            else:
                lines.append("- none")
            lines.append("")
        lines.extend(
            [
                "## Completion Rule",
                "",
                (
                    "This audit is intentionally stricter than smoke tests. The goal is complete only when every "
                    "row is READY, including real final inputs, external reference evidence, final assignment, "
                    "final handoff, and final-scale parity gates."
                ),
                "",
            ]
        )
        return "\n".join(lines)


def completion_audit(
    *,
    repo_root: str | Path = ".",
    release_dir: str | Path | None = None,
    results_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> CompletionAuditReport:
    """Audit whether the user-level Stable-ASR objective is actually complete."""

    root = Path(repo_root)
    if release_dir is not None:
        release_path = Path(release_dir)
        if results_path is None:
            results_path = release_path / "paper" / "paper_results.json"
        if artifacts_dir is None:
            artifacts_dir = release_path / "artifacts"

    status = paper_status(
        repo_root=root,
        results_path=results_path,
        artifacts_dir=artifacts_dir,
    )
    roadmap = roadmap_status(load_roadmap(root / "configs" / "roadmap" / "stable_asr_roadmap.json"), repo_root=root)
    platform = audit_platform_parity(repo_root=root)
    parity = audit_paper_parity(repo_root=root, results_path=results_path, artifacts_dir=artifacts_dir)
    reference = audit_reference_workqueue_evidence(
        reference_workqueue_from_registries(),
        repo_root=root,
        require_content=True,
    )

    items = [
        CompletionAuditItem(
            requirement="roadmap",
            evidence="configs/roadmap/stable_asr_roadmap.json",
            command="stable-asr roadmap-status --require-final-ready",
            ok=roadmap.ok and roadmap.final_scale_ready,
            detail=(
                f"{len(roadmap.missing_required_artifacts)} missing current artifact(s); "
                f"final_scale_ready={'YES' if roadmap.final_scale_ready else 'NO'}"
            ),
            blockers=_roadmap_blockers(roadmap),
        ),
        CompletionAuditItem(
            requirement="stable_worldmodel_style_platform",
            evidence="configs/platform/stable_worldmodel_parity.json",
            command="stable-asr platform-parity --registry configs/platform/stable_worldmodel_parity.json",
            ok=platform.ok,
            detail=f"missing_count={platform.missing_count}",
            blockers=_platform_blockers(platform),
        ),
        CompletionAuditItem(
            requirement="paper_smoke_bundle",
            evidence=_paper_bundle_evidence(results_path, artifacts_dir),
            command="stable-asr paper-audit --results <results> --artifacts-dir <artifacts>",
            ok=status.smoke_ready,
            detail="paper result and artifact shape are audit-ready" if status.smoke_ready else "paper smoke bundle missing or failing",
            blockers=[] if status.smoke_ready else [_paper_bundle_evidence(results_path, artifacts_dir)],
        ),
        CompletionAuditItem(
            requirement="paper_structural_parity",
            evidence="configs/paper/paper_parity_checklist.json",
            command="stable-asr paper-parity-audit --results <results> --artifacts-dir <artifacts> --require-final",
            ok=parity.ok and parity.final_ready,
            detail=f"structural_ok={parity.ok}; final_ready={parity.final_ready}",
            blockers=_paper_parity_blockers(parity),
        ),
        CompletionAuditItem(
            requirement="final_inputs",
            evidence="configs/final/paper_final.json",
            command="stable-asr final-config --config configs/final/paper_final.json --check-files",
            ok=status.final_inputs_ready,
            detail=f"missing_final_inputs={len(status.missing_final_inputs)}",
            blockers=status.missing_final_inputs,
        ),
        CompletionAuditItem(
            requirement="external_reference_evidence",
            evidence="configs/references/asr_collections.json + configs/references/turn_collections.json",
            command="stable-asr reference-workqueue --audit-evidence --require-content --repo-root .",
            ok=reference.ok,
            detail=(
                f"missing_evidence={len(reference.missing_evidence)}; "
                f"missing_license_reviews={len(reference.missing_license_reviews)}; "
                f"incomplete_evidence={len(reference.incomplete_evidence)}; "
                f"incomplete_license_reviews={len(reference.incomplete_license_reviews)}"
            ),
            blockers=[
                *reference.missing_evidence,
                *reference.missing_license_reviews,
                *reference.incomplete_evidence,
                *reference.incomplete_license_reviews,
            ],
        ),
        CompletionAuditItem(
            requirement="final_assignment",
            evidence=status.final_assignment.assignment_tracker,
            command="stable-asr final-assignment-audit --require-owner --require-due-date --require-ready",
            ok=status.final_assignment_ready,
            detail=(
                f"missing={len(status.final_assignment.missing)}; "
                f"errors={len(status.final_assignment.errors)}"
            ),
            blockers=[
                *status.final_assignment.missing,
                *status.final_assignment.errors,
                *status.final_assignment.blocking_release,
                *status.final_assignment.unassigned,
                *status.final_assignment.missing_due_dates,
            ],
        ),
        CompletionAuditItem(
            requirement="final_handoff",
            evidence=status.final_handoff.handoff,
            command="stable-asr final-handoff-audit --require-checksums",
            ok=status.final_handoff_ready,
            detail=(
                f"missing={len(status.final_handoff.missing)}; "
                f"errors={len(status.final_handoff.errors)}; "
                f"checked_paths={len(status.final_handoff.checked_paths)}"
            ),
            blockers=[*status.final_handoff.missing, *status.final_handoff.errors, *status.final_handoff.warnings],
        ),
        CompletionAuditItem(
            requirement="final_release_ready",
            evidence="runs/final/paper_results.json + runs/final/artifacts",
            command="stable-asr paper-release-audit --require-final-ready",
            ok=status.final_ready,
            detail="final paper release gates are ready" if status.final_ready else "final paper release gates are not ready",
            blockers=[] if status.final_ready else ["final_ready:NOT_READY"],
        ),
    ]
    return CompletionAuditReport(
        ok=all(item.ok for item in items),
        objective=OBJECTIVE_STATEMENT,
        repo_root=str(root),
        items=items,
    )


def write_completion_audit_json(report: CompletionAuditReport, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def write_completion_audit_markdown(report: CompletionAuditReport, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_markdown(), encoding="utf-8")
    return str(path)


def _roadmap_blockers(report: object) -> list[str]:
    blockers = [artifact.path for artifact in report.missing_required_artifacts]
    if report.final_readiness and not report.final_readiness.ready:
        blockers.extend(report.final_readiness.blockers)
    return blockers


def _platform_blockers(report: object) -> list[str]:
    blockers: list[str] = []
    for check in report.checks:
        blockers.extend(check.missing_paths)
        blockers.extend(check.missing_commands)
        blockers.extend(check.missing_markers)
    return blockers


def _paper_parity_blockers(report: object) -> list[str]:
    blockers: list[str] = []
    for check in report.checks:
        blockers.extend(check.missing)
        blockers.extend(check.final_scale_requirements)
    return blockers


def _paper_bundle_evidence(results_path: str | Path | None, artifacts_dir: str | Path | None) -> str:
    results = str(results_path) if results_path is not None else "missing results path"
    artifacts = str(artifacts_dir) if artifacts_dir is not None else "missing artifacts dir"
    return f"{results} + {artifacts}"
