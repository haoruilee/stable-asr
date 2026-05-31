"""Machine-readable roadmap status for Stable-ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table


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
class RoadmapStatusReport:
    id: str
    version: str
    title: str
    validation: RoadmapValidation
    milestones: list[RoadmapMilestoneStatus]

    @property
    def ok(self) -> bool:
        return self.validation.ok and all(milestone.ok for milestone in self.milestones)

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
        ]
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
        lines.extend(["", "## Commands", ""])
        for milestone in self.milestones:
            lines.append(f"### {milestone.id}: {milestone.title}")
            lines.extend(f"- `{command}`" for command in milestone.commands)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def load_roadmap(path: str | Path | None = None) -> dict[str, Any]:
    roadmap_path = Path(path) if path else DEFAULT_ROADMAP_PATH
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
    return RoadmapStatusReport(
        id=str(roadmap.get("id", "")),
        version=str(roadmap.get("version", "")),
        title=str(roadmap.get("title", "Stable-ASR Roadmap")),
        validation=validation,
        milestones=milestones,
    )
