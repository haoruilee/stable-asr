"""Machine-readable roadmap status for Stable-ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.resources import resolve_platform_path


DEFAULT_ROADMAP_PATH = Path("configs/roadmap/stable_asr_roadmap.json")
MILESTONE_STATUSES = {"complete", "active", "planned"}


@dataclass(frozen=True)
class RoadmapValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "roadmap: OK"
        return "roadmap: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class RoadmapArtifactCheck:
    milestone_id: str
    path: str
    exists: bool
    required_now: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "milestone_id": self.milestone_id,
            "path": self.path,
            "exists": self.exists,
            "required_now": self.required_now,
        }


@dataclass(frozen=True)
class RoadmapMilestoneStatus:
    id: str
    title: str
    status: str
    objective: str
    artifacts: list[RoadmapArtifactCheck]
    commands: list[str]
    success_criteria: list[str]

    @property
    def required_artifacts(self) -> list[RoadmapArtifactCheck]:
        return [artifact for artifact in self.artifacts if artifact.required_now]

    @property
    def missing_required_artifacts(self) -> list[RoadmapArtifactCheck]:
        return [artifact for artifact in self.required_artifacts if not artifact.exists]

    @property
    def planned_artifacts(self) -> list[RoadmapArtifactCheck]:
        return [artifact for artifact in self.artifacts if not artifact.required_now]

    @property
    def ok(self) -> bool:
        return not self.missing_required_artifacts

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "objective": self.objective,
            "ok": self.ok,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "missing_required_artifacts": [artifact.path for artifact in self.missing_required_artifacts],
            "planned_artifacts": [artifact.path for artifact in self.planned_artifacts],
            "commands": self.commands,
            "success_criteria": self.success_criteria,
        }


@dataclass(frozen=True)
class RoadmapFinalReadiness:
    checked: bool
    ready: bool
    missing_required_inputs: int
    final_input_blockers: int
    blocked_experiments: int
    missing_expected_artifacts: int
    asr_command_blockers: int
    blockers: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "checked": self.checked,
            "ready": self.ready,
            "missing_required_inputs": self.missing_required_inputs,
            "final_input_blockers": self.final_input_blockers,
            "blocked_experiments": self.blocked_experiments,
            "missing_expected_artifacts": self.missing_expected_artifacts,
            "asr_command_blockers": self.asr_command_blockers,
            "blockers": self.blockers,
            "error": self.error,
        }


@dataclass(frozen=True)
class RoadmapStatusReport:
    id: str
    version: str
    title: str
    validation: RoadmapValidation
    milestones: list[RoadmapMilestoneStatus]
    final_readiness: RoadmapFinalReadiness | None = None

    @property
    def ok(self) -> bool:
        return self.validation.ok and all(milestone.ok for milestone in self.milestones)

    @property
    def final_scale_ready(self) -> bool:
        return bool(self.final_readiness and self.final_readiness.ready)

    @property
    def missing_required_artifacts(self) -> list[RoadmapArtifactCheck]:
        missing: list[RoadmapArtifactCheck] = []
        for milestone in self.milestones:
            missing.extend(milestone.missing_required_artifacts)
        return missing

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "validation": self.validation.to_dict(),
            "milestones": [milestone.to_dict() for milestone in self.milestones],
            "final_readiness": self.final_readiness.to_dict() if self.final_readiness else None,
            "missing_required_artifacts": [
                artifact.to_dict() for artifact in self.missing_required_artifacts
            ],
        }

    def to_text(self) -> str:
        lines = [
            f"roadmap_status: {'OK' if self.ok else 'FAILED'}",
            f"id: {self.id}",
            f"version: {self.version}",
            f"milestones: {len(self.milestones)}",
            f"missing_required_artifacts: {len(self.missing_required_artifacts)}",
            f"final_scale_ready: {_yes_no(self.final_scale_ready) if self.final_readiness else 'NOT_CHECKED'}",
        ]
        if self.final_readiness:
            lines.extend(
                [
                    f"final_missing_required_inputs: {self.final_readiness.missing_required_inputs}",
                    f"final_blocked_experiments: {self.final_readiness.blocked_experiments}",
                    f"final_missing_expected_artifacts: {self.final_readiness.missing_expected_artifacts}",
                    f"final_asr_command_blockers: {self.final_readiness.asr_command_blockers}",
                ]
            )
        for milestone in self.milestones:
            present = len([artifact for artifact in milestone.required_artifacts if artifact.exists])
            total = len(milestone.required_artifacts)
            lines.append(
                f"- {milestone.id}: {milestone.status} "
                f"required_artifacts={present}/{total} "
                f"planned_artifacts={len(milestone.planned_artifacts)} "
                f"status={'OK' if milestone.ok else 'MISSING'}"
            )
        if self.missing_required_artifacts:
            lines.append("missing:")
            lines.extend(f"  - {artifact.milestone_id}: {artifact.path}" for artifact in self.missing_required_artifacts)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        rows = []
        for milestone in self.milestones:
            present = len([artifact for artifact in milestone.required_artifacts if artifact.exists])
            total = len(milestone.required_artifacts)
            rows.append(
                {
                    "milestone": milestone.id,
                    "status": milestone.status,
                    "required_artifacts": f"{present}/{total}",
                    "planned_artifacts": len(milestone.planned_artifacts),
                    "ok": "yes" if milestone.ok else "no",
                    "objective": milestone.objective,
                }
            )
        lines = [
            f"# {self.title}",
            "",
            f"- id: `{self.id}`",
            f"- version: `{self.version}`",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- missing_required_artifacts: `{len(self.missing_required_artifacts)}`",
            f"- final_scale_ready: `{_yes_no(self.final_scale_ready) if self.final_readiness else 'NOT_CHECKED'}`",
            "",
            "## Milestones",
            "",
            dict_table(rows),
            "",
            "## Missing Required Artifacts",
            "",
        ]
        if self.missing_required_artifacts:
            lines.extend(
                f"- `{artifact.milestone_id}`: `{artifact.path}`"
                for artifact in self.missing_required_artifacts
            )
        else:
            lines.append("- none")
        lines.extend(["", "## Final-Scale Readiness", ""])
        if self.final_readiness is None:
            lines.append("- not checked")
        else:
            readiness = self.final_readiness
            lines.extend(
                [
                    f"- status: `{'READY' if readiness.ready else 'NOT_READY'}`",
                    f"- missing_required_inputs: `{readiness.missing_required_inputs}`",
                    f"- final_input_blockers: `{readiness.final_input_blockers}`",
                    f"- blocked_experiments: `{readiness.blocked_experiments}`",
                    f"- missing_expected_artifacts: `{readiness.missing_expected_artifacts}`",
                    f"- asr_command_blockers: `{readiness.asr_command_blockers}`",
                ]
            )
            if readiness.error:
                lines.append(f"- error: `{readiness.error}`")
            if readiness.blockers:
                lines.extend(["", "### Final Blockers", ""])
                lines.extend(f"- `{blocker}`" for blocker in readiness.blockers[:20])
                if len(readiness.blockers) > 20:
                    lines.append(f"- ... {len(readiness.blockers) - 20} more")
        lines.extend(["", "## Commands", ""])
        for milestone in self.milestones:
            lines.append(f"### {milestone.id}: {milestone.title}")
            lines.extend(f"- `{command}`" for command in milestone.commands)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def load_roadmap(path: str | Path | None = None) -> dict[str, Any]:
    roadmap_path = resolve_platform_path(Path(path) if path else DEFAULT_ROADMAP_PATH)
    with roadmap_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("roadmap registry must be a JSON object")
    return payload


def validate_roadmap(roadmap: dict[str, Any]) -> RoadmapValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "milestones"):
        if key not in roadmap:
            errors.append(f"missing top-level key: {key}")

    milestones = roadmap.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        errors.append("milestones must be a non-empty list")
        return RoadmapValidation(ok=False, errors=errors)

    seen: set[str] = set()
    for index, milestone in enumerate(milestones):
        if not isinstance(milestone, dict):
            errors.append(f"milestone {index} must be an object")
            continue
        milestone_id = milestone.get("id")
        if not isinstance(milestone_id, str) or not milestone_id:
            errors.append(f"milestone {index} missing id")
        elif milestone_id in seen:
            errors.append(f"duplicate milestone id: {milestone_id}")
        else:
            seen.add(milestone_id)
        for key in ("title", "status", "objective", "artifacts", "commands", "success_criteria"):
            if key not in milestone:
                errors.append(f"milestone {milestone_id or index} missing {key}")
        status = milestone.get("status")
        if status not in MILESTONE_STATUSES:
            errors.append(f"milestone {milestone_id or index} status must be one of {sorted(MILESTONE_STATUSES)}")
        artifacts = milestone.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"milestone {milestone_id or index} artifacts must be a non-empty list")
        else:
            for artifact_index, artifact in enumerate(artifacts):
                if not isinstance(artifact, dict):
                    errors.append(f"milestone {milestone_id or index} artifact {artifact_index} must be an object")
                    continue
                path = artifact.get("path")
                if not isinstance(path, str) or not path:
                    errors.append(f"milestone {milestone_id or index} artifact {artifact_index} missing path")
                if not isinstance(artifact.get("required_now"), bool):
                    errors.append(
                        f"milestone {milestone_id or index} artifact {path or artifact_index} required_now must be a bool"
                    )
        for list_key in ("commands", "success_criteria"):
            values = milestone.get(list_key)
            if not isinstance(values, list) or not values or not all(isinstance(item, str) and item for item in values):
                errors.append(f"milestone {milestone_id or index} {list_key} must be a non-empty string list")
    return RoadmapValidation(ok=not errors, errors=errors)


def roadmap_status(
    roadmap: dict[str, Any],
    *,
    repo_root: str | Path = ".",
    include_final_readiness: bool = True,
) -> RoadmapStatusReport:
    validation = validate_roadmap(roadmap)
    milestones: list[RoadmapMilestoneStatus] = []
    repo_root = Path(repo_root)
    if validation.ok:
        for milestone in roadmap["milestones"]:
            artifacts = [
                RoadmapArtifactCheck(
                    milestone_id=str(milestone["id"]),
                    path=str(artifact["path"]),
                    exists=(repo_root / str(artifact["path"])).exists(),
                    required_now=bool(artifact["required_now"]),
                )
                for artifact in milestone["artifacts"]
            ]
            milestones.append(
                RoadmapMilestoneStatus(
                    id=str(milestone["id"]),
                    title=str(milestone["title"]),
                    status=str(milestone["status"]),
                    objective=str(milestone["objective"]),
                    artifacts=artifacts,
                    commands=[str(command) for command in milestone["commands"]],
                    success_criteria=[str(item) for item in milestone["success_criteria"]],
                )
            )
    final_readiness = (
        _roadmap_final_readiness(repo_root)
        if include_final_readiness and _has_final_milestone(milestones)
        else None
    )
    return RoadmapStatusReport(
        id=str(roadmap.get("id", "")),
        version=str(roadmap.get("version", "")),
        title=str(roadmap.get("title", "Stable-ASR Roadmap")),
        validation=validation,
        milestones=milestones,
        final_readiness=final_readiness,
    )


def _has_final_milestone(milestones: list[RoadmapMilestoneStatus]) -> bool:
    return any(milestone.id == "m5_final_scale_evidence" for milestone in milestones)


def _roadmap_final_readiness(repo_root: Path) -> RoadmapFinalReadiness:
    try:
        from stable_asr.paper.evidence import final_evidence_matrix
        from stable_asr.paper.final_inputs import final_input_collection_report

        input_report = final_input_collection_report(repo_root=repo_root)
        evidence_report = final_evidence_matrix(repo_root=repo_root)
        blockers = [
            *[f"input:{path}" for path in input_report.missing_required],
            *[f"evidence:{blocker}" for blocker in evidence_report.final_input_blockers],
            *[f"asr_command:{blocker}" for blocker in evidence_report.asr_command_blockers],
        ]
        ready = input_report.ok and evidence_report.final_ready
        return RoadmapFinalReadiness(
            checked=True,
            ready=ready,
            missing_required_inputs=len(input_report.missing_required),
            final_input_blockers=len(evidence_report.final_input_blockers),
            blocked_experiments=evidence_report.blocked_experiment_count,
            missing_expected_artifacts=evidence_report.missing_artifact_count,
            asr_command_blockers=len(evidence_report.asr_command_blockers),
            blockers=blockers,
        )
    except (OSError, ValueError) as exc:
        return RoadmapFinalReadiness(
            checked=False,
            ready=False,
            missing_required_inputs=0,
            final_input_blockers=0,
            blocked_experiments=0,
            missing_expected_artifacts=0,
            asr_command_blockers=0,
            blockers=[],
            error=str(exc),
        )


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"
