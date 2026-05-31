"""Paper artifact bundle generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stable_asr.paper.case_studies import paper_case_studies
from stable_asr.paper.claims import paper_claims
from stable_asr.paper.final_experiments import (
    final_experiments_markdown,
    load_final_experiments,
    write_final_experiments_json,
)
from stable_asr.paper.final_config import (
    audit_final_run_files,
    final_run_file_audit_markdown,
    final_run_config_markdown,
    load_final_run_config,
    write_final_run_config_json,
)
from stable_asr.paper.figures import PAPER_FIGURES, paper_figure
from stable_asr.paper.leaderboard import export_leaderboard
from stable_asr.paper.parity import audit_paper_parity, load_paper_parity_checklist, paper_parity_markdown
from stable_asr.paper.status import paper_status, write_paper_status_json, write_paper_status_markdown
from stable_asr.paper.suites import benchmark_suite_markdown, load_benchmark_suite, write_benchmark_suite_json
from stable_asr.paper.tables import PAPER_TABLES, paper_table
from stable_asr.data.sources import data_sources_markdown, load_data_sources, write_data_sources_json
from stable_asr.models.adapters.registry import (
    adapter_registry_markdown,
    load_adapter_registry,
    write_adapter_registry_json,
)
from stable_asr.scenarios.suites import scenario_suite_markdown, load_scenario_suite, write_scenario_suite_json


@dataclass(frozen=True)
class PaperArtifactBundle:
    output_dir: str
    index_path: str
    manifest_path: str
    tables: dict[str, str]
    figures: dict[str, str]
    leaderboards: dict[str, str]
    benchmark_suite: dict[str, str]
    data_sources: dict[str, str]
    adapter_registry: dict[str, str]
    scenario_suite: dict[str, str]
    case_studies: dict[str, str]
    paper_parity: dict[str, str]
    final_experiments: dict[str, str]
    final_run_config: dict[str, str]
    final_run_file_audit: dict[str, str]
    paper_status: dict[str, str]
    claims: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "index_path": self.index_path,
            "manifest_path": self.manifest_path,
            "tables": self.tables,
            "figures": self.figures,
            "leaderboards": self.leaderboards,
            "benchmark_suite": self.benchmark_suite,
            "data_sources": self.data_sources,
            "adapter_registry": self.adapter_registry,
            "scenario_suite": self.scenario_suite,
            "case_studies": self.case_studies,
            "paper_parity": self.paper_parity,
            "final_experiments": self.final_experiments,
            "final_run_config": self.final_run_config,
            "final_run_file_audit": self.final_run_file_audit,
            "paper_status": self.paper_status,
            "claims": self.claims,
        }


def paper_artifact_bundle(results_path: str | Path, output_dir: str | Path) -> PaperArtifactBundle:
    """Generate all smoke-run paper tables, figures, and an artifact index."""

    results_path = Path(results_path)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, str] = {}
    for name in PAPER_TABLES:
        path = tables_dir / f"{name}.md"
        path.write_text(paper_table(results_path, name) + "\n", encoding="utf-8")
        tables[name] = str(path)

    figures: dict[str, str] = {}
    for name in PAPER_FIGURES:
        path = figures_dir / f"{name}.svg"
        figures[name] = paper_figure(results_path, name, path)

    leaderboards = {
        "jsonl": export_leaderboard(results_path, output_dir / "leaderboard.jsonl", format="jsonl"),
        "csv": export_leaderboard(results_path, output_dir / "leaderboard.csv", format="csv"),
    }
    suite = load_benchmark_suite()
    benchmark_suite = {
        "json": write_benchmark_suite_json(output_dir / "benchmark_suite.json", suite),
        "markdown": str(output_dir / "BENCHMARK_SUITE.md"),
    }
    Path(benchmark_suite["markdown"]).write_text(benchmark_suite_markdown(suite), encoding="utf-8")
    sources = load_data_sources()
    data_sources = {
        "json": write_data_sources_json(output_dir / "data_sources.json", sources),
        "markdown": str(output_dir / "DATA_SOURCES.md"),
    }
    Path(data_sources["markdown"]).write_text(data_sources_markdown(sources), encoding="utf-8")
    adapters = load_adapter_registry()
    adapter_registry = {
        "json": write_adapter_registry_json(output_dir / "adapter_registry.json", adapters),
        "markdown": str(output_dir / "ADAPTERS.md"),
    }
    Path(adapter_registry["markdown"]).write_text(adapter_registry_markdown(adapters), encoding="utf-8")
    voiceworld_suite = load_scenario_suite()
    scenario_suite = {
        "json": write_scenario_suite_json(output_dir / "scenario_suite.json", voiceworld_suite),
        "markdown": str(output_dir / "SCENARIO_SUITE.md"),
    }
    Path(scenario_suite["markdown"]).write_text(scenario_suite_markdown(voiceworld_suite), encoding="utf-8")
    case_study_artifacts = paper_case_studies(results_path, output_dir)
    case_studies = case_study_artifacts.to_dict()

    index_path = output_dir / "ARTIFACT_INDEX.md"
    manifest_path = output_dir / "artifact_manifest.json"
    # Seed these files before the parity audit, which checks that the paper
    # bundle contains an artifact index and manifest. They are rewritten below
    # with the final bundle payload.
    index_path.write_text("# Stable-ASR Paper Artifact Index\n", encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")

    parity_report = audit_paper_parity(
        checklist=load_paper_parity_checklist(),
        repo_root=Path("."),
        results_path=results_path,
        artifacts_dir=output_dir,
    )
    paper_parity = {
        "json": str(output_dir / "paper_parity.json"),
        "markdown": str(output_dir / "PAPER_PARITY.md"),
    }
    Path(paper_parity["json"]).write_text(
        json.dumps(parity_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(paper_parity["markdown"]).write_text(paper_parity_markdown(parity_report), encoding="utf-8")
    final_experiment_registry = load_final_experiments()
    final_experiments = {
        "json": write_final_experiments_json(output_dir / "final_experiments.json", final_experiment_registry),
        "markdown": str(output_dir / "FINAL_EXPERIMENTS.md"),
    }
    Path(final_experiments["markdown"]).write_text(
        final_experiments_markdown(final_experiment_registry),
        encoding="utf-8",
    )
    final_config = load_final_run_config()
    final_run_config = {
        "json": write_final_run_config_json(output_dir / "final_run_config.json", final_config),
        "markdown": str(output_dir / "FINAL_RUN_CONFIG.md"),
    }
    Path(final_run_config["markdown"]).write_text(final_run_config_markdown(final_config), encoding="utf-8")
    final_file_report = audit_final_run_files(final_config, repo_root=Path("."))
    final_run_file_audit = {
        "json": str(output_dir / "final_run_file_audit.json"),
        "markdown": str(output_dir / "FINAL_RUN_FILE_AUDIT.md"),
    }
    Path(final_run_file_audit["json"]).write_text(
        json.dumps(final_file_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(final_run_file_audit["markdown"]).write_text(
        final_run_file_audit_markdown(final_file_report),
        encoding="utf-8",
    )
    status_report = paper_status(repo_root=Path("."), results_path=results_path, artifacts_dir=output_dir)
    paper_status_artifacts = {
        "json": write_paper_status_json(status_report, output_dir / "paper_status.json"),
        "markdown": write_paper_status_markdown(status_report, output_dir / "PAPER_STATUS.md"),
    }

    # Create provisional bundle files so the claim audit can verify that the
    # paper reproducibility artifacts exist. They are rewritten below with the
    # final claim artifact paths included.
    provisional = PaperArtifactBundle(
        output_dir=str(output_dir),
        index_path=str(index_path),
        manifest_path=str(manifest_path),
        tables=tables,
        figures=figures,
        leaderboards=leaderboards,
        benchmark_suite=benchmark_suite,
        data_sources=data_sources,
        adapter_registry=adapter_registry,
        scenario_suite=scenario_suite,
        case_studies=case_studies,
        paper_parity=paper_parity,
        final_experiments=final_experiments,
        final_run_config=final_run_config,
        final_run_file_audit=final_run_file_audit,
        paper_status=paper_status_artifacts,
        claims={},
    )
    manifest_path.write_text(json.dumps(provisional.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(_artifact_index(results_path, provisional), encoding="utf-8")

    claim_artifacts = paper_claims(results_path, output_dir)
    claims = claim_artifacts.to_dict()
    bundle = PaperArtifactBundle(
        output_dir=str(output_dir),
        index_path=str(index_path),
        manifest_path=str(manifest_path),
        tables=tables,
        figures=figures,
        leaderboards=leaderboards,
        benchmark_suite=benchmark_suite,
        data_sources=data_sources,
        adapter_registry=adapter_registry,
        scenario_suite=scenario_suite,
        case_studies=case_studies,
        paper_parity=paper_parity,
        final_experiments=final_experiments,
        final_run_config=final_run_config,
        final_run_file_audit=final_run_file_audit,
        paper_status=paper_status_artifacts,
        claims=claims,
    )
    manifest_path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(_artifact_index(results_path, bundle), encoding="utf-8")
    return bundle


def _artifact_index(results_path: Path, bundle: PaperArtifactBundle) -> str:
    lines = [
        "# Stable-ASR Paper Artifact Index",
        "",
        f"Results source: `{results_path}`",
        "",
        "## Tables",
        "",
    ]
    for name, path in bundle.tables.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Figures", ""])
    for name, path in bundle.figures.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Leaderboards", ""])
    for name, path in bundle.leaderboards.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Benchmark Suite", ""])
    for name, path in bundle.benchmark_suite.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Data Sources", ""])
    for name, path in bundle.data_sources.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Adapter Registry", ""])
    for name, path in bundle.adapter_registry.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Scenario Suite", ""])
    for name, path in bundle.scenario_suite.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Case Studies", ""])
    for name, path in bundle.case_studies.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Paper Parity", ""])
    for name, path in bundle.paper_parity.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Final Experiments", ""])
    for name, path in bundle.final_experiments.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Final Run Config", ""])
    for name, path in bundle.final_run_config.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Final Run File Audit", ""])
    for name, path in bundle.final_run_file_audit.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Paper Status", ""])
    for name, path in bundle.paper_status.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Claims", ""])
    for name, path in bundle.claims.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            f"stable-asr paper-bundle --results {results_path} --output-dir {bundle.output_dir}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
