"""One-command paper release smoke workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.audit import PaperReleaseAuditReport, audit_paper_release
from stable_asr.paper.cards import dataset_card, experiment_card
from stable_asr.paper.draft import paper_draft
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.latex import paper_latex


@dataclass(frozen=True)
class PaperReleaseSmokeResult:
    output_dir: str
    results_path: str
    artifacts_dir: str
    markdown_draft: str
    latex_draft: str
    dataset_card: str
    experiment_card: str
    release_audit_json: str
    release_audit_markdown: str
    audit: PaperReleaseAuditReport

    @property
    def ok(self) -> bool:
        return self.audit.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output_dir": self.output_dir,
            "results_path": self.results_path,
            "artifacts_dir": self.artifacts_dir,
            "markdown_draft": self.markdown_draft,
            "latex_draft": self.latex_draft,
            "dataset_card": self.dataset_card,
            "experiment_card": self.experiment_card,
            "release_audit_json": self.release_audit_json,
            "release_audit_markdown": self.release_audit_markdown,
            "audit": self.audit.to_dict(),
        }

    def to_text(self) -> str:
        missing = [check for check in self.audit.checks if not check.ok]
        lines = [
            f"paper_release_smoke: {'READY' if self.ok else 'NOT_READY'}",
            f"results: {self.results_path}",
            f"artifacts: {self.artifacts_dir}",
            f"markdown_draft: {self.markdown_draft}",
            f"latex_draft: {self.latex_draft}",
            f"dataset_card: {self.dataset_card}",
            f"experiment_card: {self.experiment_card}",
            f"release_audit_json: {self.release_audit_json}",
            f"release_audit_markdown: {self.release_audit_markdown}",
            f"missing_gates: {len(missing)}",
        ]
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

    audit = audit_paper_release(
        repo_root=repo_root,
        results_path=paper_run.results_path,
        artifacts_dir=bundle.output_dir,
        markdown_draft=markdown,
        latex_draft=latex,
        dataset_card=dataset,
        experiment_card=experiment,
    )
    audit_json = output_dir / "release_audit.json"
    audit_markdown = output_dir / "RELEASE_AUDIT.md"
    audit_json.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_markdown.write_text(audit.to_text() + "\n", encoding="utf-8")

    return PaperReleaseSmokeResult(
        output_dir=str(output_dir),
        results_path=paper_run.results_path,
        artifacts_dir=bundle.output_dir,
        markdown_draft=markdown,
        latex_draft=latex,
        dataset_card=dataset,
        experiment_card=experiment,
        release_audit_json=str(audit_json),
        release_audit_markdown=str(audit_markdown),
        audit=audit,
    )
