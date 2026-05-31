"""Final-scale evidence matrix for paper run planning and audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.paper.final_config import (
    FinalRunPathCheck,
    audit_final_run_files,
    load_final_run_config,
)
from stable_asr.paper.final_experiments import load_final_experiments, validate_final_experiments
from stable_asr.streaming.command_compare import ASRCommandConfigAuditReport, audit_asr_command_config


_EXPERIMENT_BLOCKERS: dict[str, tuple[str, ...]] = {
    "real_data_layer_benchmark": ("corpus:", "turn_split:train"),
    "external_turn_baselines": ("turn_split:test", "external_prediction:", "turn_split:train"),
    "real_voiceworld_scenarios": ("turn_split:voiceworld_real", "voiceworld_real:", "turn_split:train"),
    "policy_transfer": ("turn_split:dev", "turn_split:test", "turn_split:voiceworld_real"),
    "real_streaming_asr_systems": ("corpus:", "asr_command_config"),
    "final_reproducibility_bundle": ("",),
}


@dataclass(frozen=True)
class FinalEvidenceArtifactCheck:
    name: str
    path: str
    exists: bool
    checked: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "exists": self.exists,
            "checked": self.checked,
        }


@dataclass(frozen=True)
class FinalEvidenceExperiment:
    id: str
    title: str
    registry_status: str
    computed_status: str
    priority: str
    paper_section: str
    blockers: list[str]
    expected_artifacts: list[FinalEvidenceArtifactCheck]
    commands: list[str]
    metrics: list[str]
    success_criteria: list[str]

    @property
    def missing_artifacts(self) -> list[FinalEvidenceArtifactCheck]:
        return [artifact for artifact in self.expected_artifacts if artifact.checked and not artifact.exists]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "registry_status": self.registry_status,
            "computed_status": self.computed_status,
            "priority": self.priority,
            "paper_section": self.paper_section,
            "blockers": self.blockers,
            "expected_artifacts": [artifact.to_dict() for artifact in self.expected_artifacts],
            "missing_artifacts": [artifact.to_dict() for artifact in self.missing_artifacts],
            "commands": self.commands,
            "metrics": self.metrics,
            "success_criteria": self.success_criteria,
        }


@dataclass(frozen=True)
class FinalEvidenceMatrixReport:
    ok: bool
    final_ready: bool
    repo_root: str
    artifacts_dir: str | None
    experiments: list[FinalEvidenceExperiment]
    final_input_blockers: list[str]
    asr_command_blockers: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "final_ready": self.final_ready,
            "repo_root": self.repo_root,
            "artifacts_dir": self.artifacts_dir,
            "blocked_experiments": self.blocked_experiment_count,
            "missing_artifacts": self.missing_artifact_count,
            "final_input_blockers": self.final_input_blockers,
            "asr_command_blockers": self.asr_command_blockers,
            "experiments": [experiment.to_dict() for experiment in self.experiments],
        }

    @property
    def blocked_experiment_count(self) -> int:
        return sum(1 for experiment in self.experiments if experiment.blockers)

    @property
    def missing_artifact_count(self) -> int:
        return sum(len(experiment.missing_artifacts) for experiment in self.experiments)

    def to_markdown(self) -> str:
        rows = [
            {
                "experiment": experiment.id,
                "priority": experiment.priority,
                "registry_status": experiment.registry_status,
                "computed_status": experiment.computed_status,
                "blockers": len(experiment.blockers),
                "missing_artifacts": len(experiment.missing_artifacts),
                "commands": len(experiment.commands),
            }
            for experiment in self.experiments
        ]
        lines = [
            "# Stable-ASR Final Evidence Matrix",
            "",
            f"- status: `{'READY' if self.final_ready else 'NOT_READY'}`",
            f"- blocked_experiments: `{self.blocked_experiment_count}`",
            f"- missing_artifacts: `{self.missing_artifact_count}`",
            f"- final_input_blockers: `{len(self.final_input_blockers)}`",
            f"- asr_command_blockers: `{len(self.asr_command_blockers)}`",
            "",
            "## Experiment Summary",
            "",
            dict_table(rows),
        ]
        for experiment in self.experiments:
            lines.extend(_experiment_markdown(experiment))
        return "\n".join(lines)


def final_evidence_matrix(
    *,
    repo_root: str | Path = ".",
    registry_path: str | Path | None = None,
    config_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> FinalEvidenceMatrixReport:
    """Build a concrete final-scale evidence matrix from checked-in runbooks."""

    root = Path(repo_root)
    registry = load_final_experiments(_default_path(root, registry_path, "configs/paper/final_experiments.json"))
    validation = validate_final_experiments(registry)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    config = load_final_run_config(_default_path(root, config_path, "configs/final/paper_final.json"))
    file_audit = audit_final_run_files(config, repo_root=root)
    missing_required = [check for check in file_audit.checks if check.required and not check.ok]
    command_audit = audit_asr_command_config(
        _resolve(str(config["asr_command_config"]), root=root),
        repo_root=root,
        min_adapters=4,
        require_input_manifest=True,
    )
    command_blockers = _asr_command_blockers(command_audit)
    artifact_root = Path(artifacts_dir) if artifacts_dir is not None else None
    experiments = [
        _experiment_evidence(
            experiment,
            config=config,
            root=root,
            artifact_root=artifact_root,
            missing_required=missing_required,
            command_blockers=command_blockers,
        )
        for experiment in registry["experiments"]
    ]
    final_ready = all(experiment.computed_status == "completed" for experiment in experiments)
    return FinalEvidenceMatrixReport(
        ok=True,
        final_ready=final_ready,
        repo_root=str(root),
        artifacts_dir=str(artifact_root) if artifact_root else None,
        experiments=experiments,
        final_input_blockers=[f"{check.name}: {check.path}" for check in missing_required],
        asr_command_blockers=command_blockers,
    )


def _experiment_evidence(
    experiment: dict[str, Any],
    *,
    config: dict[str, Any],
    root: Path,
    artifact_root: Path | None,
    missing_required: list[FinalRunPathCheck],
    command_blockers: list[str],
) -> FinalEvidenceExperiment:
    experiment_id = str(experiment["id"])
    blockers = _matching_blockers(experiment_id, missing_required)
    if experiment_id in {"real_streaming_asr_systems", "final_reproducibility_bundle"}:
        blockers.extend(command_blockers)
    artifacts = [
        _artifact_check(str(name), config=config, root=root, artifact_root=artifact_root)
        for name in experiment.get("expected_artifacts", [])
    ]
    missing_artifacts = [artifact for artifact in artifacts if artifact.checked and not artifact.exists]
    registry_status = str(experiment["status"])
    if blockers:
        computed_status = "blocked"
    elif missing_artifacts:
        computed_status = "artifact_missing"
    elif registry_status == "completed":
        computed_status = "completed"
    else:
        computed_status = "ready_to_run"
    return FinalEvidenceExperiment(
        id=experiment_id,
        title=str(experiment["title"]),
        registry_status=registry_status,
        computed_status=computed_status,
        priority=str(experiment["priority"]),
        paper_section=str(experiment["paper_section"]),
        blockers=blockers,
        expected_artifacts=artifacts,
        commands=[str(command) for command in experiment.get("commands", [])],
        metrics=[str(metric) for metric in experiment.get("metrics", [])],
        success_criteria=[str(item) for item in experiment.get("success_criteria", [])],
    )


def _matching_blockers(experiment_id: str, missing_required: list[FinalRunPathCheck]) -> list[str]:
    prefixes = _EXPERIMENT_BLOCKERS.get(experiment_id, ())
    blockers: list[str] = []
    for check in missing_required:
        if any(prefix == "" or check.name.startswith(prefix) for prefix in prefixes):
            blockers.append(f"{check.name}: {check.path}")
    return blockers


def _asr_command_blockers(report: ASRCommandConfigAuditReport) -> list[str]:
    blockers = list(report.errors)
    for adapter in report.adapters:
        for missing in adapter.missing_required_inputs:
            blockers.append(f"{adapter.name}: missing required input {missing}")
    return blockers


def _artifact_check(
    name: str,
    *,
    config: dict[str, Any],
    root: Path,
    artifact_root: Path | None,
) -> FinalEvidenceArtifactCheck:
    path = _artifact_path(name, config=config, root=root, artifact_root=artifact_root)
    checked = path is not None
    exists = bool(path and path.exists())
    return FinalEvidenceArtifactCheck(
        name=name,
        path=str(path) if path else "",
        exists=exists,
        checked=checked,
    )


def _artifact_path(
    name: str,
    *,
    config: dict[str, Any],
    root: Path,
    artifact_root: Path | None,
) -> Path | None:
    run_root_artifacts = {
        "DATASET_CARD.md",
        "EXPERIMENT_CARD.md",
        "FINAL_ASSIGNMENT_AUDIT.md",
        "FINAL_HANDOFF_AUDIT.md",
        "FINAL_INPUT_HANDOFF.json",
        "MODEL_CARD.md",
        "paper.tex",
        "PAPER_DRAFT.md",
        "artifacts.tar.gz",
        "artifacts.tar.gz.sha256",
    }
    if artifact_root is not None and name in run_root_artifacts:
        return artifact_root.parent / name
    artifact_map = {
        "DATASET_CARD.md": config["artifacts"].get("dataset_card"),
        "EXPERIMENT_CARD.md": config["artifacts"].get("experiment_card"),
        "FINAL_ASSIGNMENT_AUDIT.md": config["artifacts"].get("assignment_audit"),
        "FINAL_HANDOFF_AUDIT.md": config["artifacts"].get("handoff_audit"),
        "FINAL_INPUT_HANDOFF.json": config["artifacts"].get("handoff"),
        "MODEL_CARD.md": config["artifacts"].get("model_card"),
        "paper.tex": config["artifacts"].get("latex_draft"),
        "PAPER_DRAFT.md": config["artifacts"].get("markdown_draft"),
        "artifacts.tar.gz": config["artifacts"].get("artifact_archive"),
        "artifacts.tar.gz.sha256": _archive_sha256_path(config["artifacts"].get("artifact_archive")),
    }
    if name in artifact_map and artifact_map[name]:
        return _resolve(str(artifact_map[name]), root=root)
    if artifact_root is not None:
        return artifact_root / name
    bundle_dir = config.get("artifacts", {}).get("bundle_dir")
    if isinstance(bundle_dir, str) and bundle_dir:
        return _resolve(bundle_dir, root=root) / name
    return None


def _archive_sha256_path(archive_path: Any) -> str | None:
    if not isinstance(archive_path, str) or not archive_path:
        return None
    return archive_path + ".sha256"


def _experiment_markdown(experiment: FinalEvidenceExperiment) -> list[str]:
    lines = [
        "",
        f"## {experiment.id}",
        "",
        f"- title: {experiment.title}",
        f"- registry_status: `{experiment.registry_status}`",
        f"- computed_status: `{experiment.computed_status}`",
        f"- priority: `{experiment.priority}`",
        f"- paper_section: `{experiment.paper_section}`",
        "",
        "Blockers:",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in experiment.blockers) if experiment.blockers else lines.append("- none")
    lines.extend(["", "Expected artifacts:", ""])
    for artifact in experiment.expected_artifacts:
        if not artifact.checked:
            status = "UNCHECKED"
        else:
            status = "OK" if artifact.exists else "MISSING"
        lines.append(f"- `{status}` `{artifact.name}` -> `{artifact.path}`")
    lines.extend(["", "Success criteria:", ""])
    lines.extend(f"- {item}" for item in experiment.success_criteria)
    lines.append("")
    return lines


def _resolve(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _default_path(root: Path, path: str | Path | None, relative: str) -> Path | None:
    if path is not None:
        return Path(path)
    candidate = root / relative
    return candidate if candidate.exists() else None
