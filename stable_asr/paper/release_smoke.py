"""One-command paper release smoke workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.doctor import DoctorReport, run_doctor
from stable_asr.paper.archive import (
    PaperArchiveVerificationReport,
    paper_artifact_archive,
    verify_paper_artifact_archive,
)
from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.audit import PaperReleaseAuditReport, audit_paper_release
from stable_asr.paper.cards import dataset_card, experiment_card, model_card
from stable_asr.paper.draft import paper_draft
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.latex import paper_latex
from stable_asr.paper.status import PaperStatusReport, paper_status


@dataclass(frozen=True)
class PaperReleaseSmokeResult:
    output_dir: str
    results_path: str
    artifacts_dir: str
    markdown_draft: str
    latex_draft: str
    dataset_card: str
    experiment_card: str
    model_card: str
    artifact_archive: str
    artifact_archive_sha256: str
    artifact_archive_verification_json: str
    artifact_archive_verification_markdown: str
    archive_verification: PaperArchiveVerificationReport
    release_audit_json: str
    release_audit_markdown: str
    audit: PaperReleaseAuditReport
    paper_status_markdown: str
    paper_status: PaperStatusReport
    release_environment: DoctorReport

    @property
    def ok(self) -> bool:
        return self.audit.ok and self.archive_verification.ok

    @property
    def final_ready(self) -> bool:
        return self.paper_status.final_ready

    @property
    def final_assignment_ready(self) -> bool:
        return self.paper_status.final_assignment_ready

    @property
    def final_handoff_ready(self) -> bool:
        return self.paper_status.final_handoff_ready

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "final_ready": self.final_ready,
            "final_assignment_ready": self.final_assignment_ready,
            "final_handoff_ready": self.final_handoff_ready,
            "output_dir": self.output_dir,
            "results_path": self.results_path,
            "artifacts_dir": self.artifacts_dir,
            "markdown_draft": self.markdown_draft,
            "latex_draft": self.latex_draft,
            "dataset_card": self.dataset_card,
            "experiment_card": self.experiment_card,
            "model_card": self.model_card,
            "artifact_archive": self.artifact_archive,
            "artifact_archive_sha256": self.artifact_archive_sha256,
            "artifact_archive_verification_json": self.artifact_archive_verification_json,
            "artifact_archive_verification_markdown": self.artifact_archive_verification_markdown,
            "archive_verification": self.archive_verification.to_dict(),
            "release_audit_json": self.release_audit_json,
            "release_audit_markdown": self.release_audit_markdown,
            "audit": self.audit.to_dict(),
            "paper_status_markdown": self.paper_status_markdown,
            "paper_status": self.paper_status.to_dict(),
            "release_environment": self.release_environment.to_dict(),
        }

    def to_text(self) -> str:
        missing = [check for check in self.audit.checks if not check.ok]
        lines = [
            f"paper_release_smoke: {'READY' if self.ok else 'NOT_READY'}",
            f"final_scale_ready: {'YES' if self.final_ready else 'NO'}",
            f"results: {self.results_path}",
            f"artifacts: {self.artifacts_dir}",
            f"markdown_draft: {self.markdown_draft}",
            f"latex_draft: {self.latex_draft}",
            f"dataset_card: {self.dataset_card}",
            f"experiment_card: {self.experiment_card}",
            f"model_card: {self.model_card}",
            f"artifact_archive: {self.artifact_archive}",
            f"artifact_archive_sha256: {self.artifact_archive_sha256}",
            f"artifact_archive_verification_json: {self.artifact_archive_verification_json}",
            f"artifact_archive_verification_markdown: {self.artifact_archive_verification_markdown}",
            f"archive_verification: {'OK' if self.archive_verification.ok else 'FAILED'}",
            f"release_audit_json: {self.release_audit_json}",
            f"release_audit_markdown: {self.release_audit_markdown}",
            f"paper_status_markdown: {self.paper_status_markdown}",
            f"release_environment_ready: {'YES' if self.release_environment.release_environment_ready else 'NO'}",
            f"final_inputs_ready: {'YES' if self.paper_status.final_inputs_ready else 'NO'}",
            f"final_assignment_ready: {'YES' if self.final_assignment_ready else 'NO'}",
            f"final_handoff_ready: {'YES' if self.final_handoff_ready else 'NO'}",
            f"missing_gates: {len(missing)}",
        ]
        if not self.release_environment.release_environment_ready:
            lines.append("release_environment_hint: python -m pip install -e \".[lance,train]\"")
            lines.extend(
                f"- release_env/{check.category}/{check.name}: {check.detail}"
                for check in self.release_environment.checks
                if check.required and not check.ok
            )
        lines.extend(f"- archive_verification/{error}" for error in self.archive_verification.errors)
        lines.extend(f"- {check.gate}/{check.name}: {check.detail}" for check in missing)
        return "\n".join(lines)


def run_paper_release_smoke(
    output_dir: str | Path,
    *,
    episodes: int = 9,
    seed: int = 6,
    train_model: bool = True,
    repo_root: str | Path = ".",
    dataset_manifest: str | Path = "examples/data/turn_demo.jsonl",
) -> PaperReleaseSmokeResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_run = run_paper_smoke(output_dir / "paper", episodes=episodes, seed=seed, train_model=train_model)
    bundle = paper_artifact_bundle(paper_run.results_path, output_dir / "artifacts")
    markdown = paper_draft(paper_run.results_path, output_dir / "PAPER_DRAFT.md", artifacts_dir=bundle.output_dir)
    latex = paper_latex(paper_run.results_path, output_dir / "paper.tex", artifacts_dir=bundle.output_dir)
    dataset = dataset_card(dataset_manifest, output_dir / "DATASET_CARD.md")
    experiment = experiment_card(paper_run.results_path, output_dir / "EXPERIMENT_CARD.md")
    nanoturn_metrics_path = None
    if paper_run.results.get("nanoturn", {}).get("status") == "completed":
        nanoturn_metrics_path = paper_run.results["nanoturn"].get("metrics_path")
    model = model_card(
        "configs/models/stable_asr_models.json",
        output_dir / "MODEL_CARD.md",
        model_id="nanoturn_pico",
        metrics_path=nanoturn_metrics_path,
    )

    audit = audit_paper_release(
        repo_root=repo_root,
        results_path=paper_run.results_path,
        artifacts_dir=bundle.output_dir,
        markdown_draft=markdown,
        latex_draft=latex,
        dataset_card=dataset,
        experiment_card=experiment,
        model_card=model,
    )
    audit_json = output_dir / "release_audit.json"
    audit_markdown = output_dir / "RELEASE_AUDIT.md"
    audit_json.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_markdown.write_text(audit.to_text() + "\n", encoding="utf-8")
    status = paper_status(repo_root=repo_root, results_path=paper_run.results_path, artifacts_dir=bundle.output_dir)
    archive = paper_artifact_archive(bundle.output_dir, output_dir / "artifacts.tar.gz")
    archive_verification = verify_paper_artifact_archive(archive.archive_path)
    archive_verification_json = output_dir / "archive_verification.json"
    archive_verification_markdown = output_dir / "ARCHIVE_VERIFICATION.md"
    archive_verification_json.write_text(
        json.dumps(archive_verification.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    archive_verification_markdown.write_text(archive_verification.to_text() + "\n", encoding="utf-8")
    release_environment = run_doctor(repo_root=repo_root, check_release_env=True)

    return PaperReleaseSmokeResult(
        output_dir=str(output_dir),
        results_path=paper_run.results_path,
        artifacts_dir=bundle.output_dir,
        markdown_draft=markdown,
        latex_draft=latex,
        dataset_card=dataset,
        experiment_card=experiment,
        model_card=model,
        artifact_archive=archive.archive_path,
        artifact_archive_sha256=archive.sha256_path,
        artifact_archive_verification_json=str(archive_verification_json),
        artifact_archive_verification_markdown=str(archive_verification_markdown),
        archive_verification=archive_verification,
        release_audit_json=str(audit_json),
        release_audit_markdown=str(audit_markdown),
        audit=audit,
        paper_status_markdown=bundle.paper_status["markdown"],
        paper_status=status,
        release_environment=release_environment,
    )
