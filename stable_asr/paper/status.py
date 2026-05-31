"""Single-page paper readiness status report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stable_asr.eval.report import dict_table
from stable_asr.paper.audit import audit_paper_artifacts
from stable_asr.paper.final_config import audit_final_run_files, load_final_run_config
from stable_asr.paper.parity import audit_paper_parity


@dataclass(frozen=True)
class PaperStatusReport:
    ok: bool
    smoke_ready: bool
    structural_ready: bool
    final_ready: bool
    final_inputs_ready: bool
    missing_final_inputs: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "smoke_ready": self.smoke_ready,
            "structural_ready": self.structural_ready,
            "final_ready": self.final_ready,
            "final_inputs_ready": self.final_inputs_ready,
            "missing_final_inputs": self.missing_final_inputs,
        }

    def to_markdown(self) -> str:
        rows = [
            {"gate": "doctor", "status": "OK" if self.ok else "FAILED", "meaning": "required configs and environment checks"},
            {"gate": "smoke_ready", "status": _status(self.smoke_ready), "meaning": "paper result and artifact bundle shape"},
            {"gate": "structural_ready", "status": _status(self.structural_ready), "meaning": "stable-worldmodel-style structural evidence"},
            {"gate": "final_inputs_ready", "status": _status(self.final_inputs_ready), "meaning": "real final corpora/splits/predictions exist"},
            {"gate": "final_ready", "status": _status(self.final_ready), "meaning": "no final-scale parity gaps remain"},
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
    results_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> PaperStatusReport:
    repo_root = Path(repo_root)
    from stable_asr.doctor import run_doctor

    doctor = run_doctor(repo_root=repo_root, check_final_files=True)
    smoke_ready = False
    if results_path is not None and artifacts_dir is not None:
        smoke_ready = audit_paper_artifacts(results_path, artifacts_dir).ok
    parity = audit_paper_parity(repo_root=repo_root, results_path=results_path, artifacts_dir=artifacts_dir)
    final_files = audit_final_run_files(load_final_run_config(repo_root / "configs" / "final" / "paper_final.json"), repo_root=repo_root)
    missing = [
        check.path
        for check in final_files.checks
        if check.required and not check.ok
    ]
    return PaperStatusReport(
        ok=doctor.ok,
        smoke_ready=smoke_ready,
        structural_ready=parity.ok,
        final_ready=parity.final_ready and final_files.ok,
        final_inputs_ready=final_files.ok,
        missing_final_inputs=missing,
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
