"""Paper artifact bundle generation."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from stable_asr.paper.acquisition_pack import build_final_acquisition_pack
from stable_asr.paper.adapter_pack import build_adapter_pack
from stable_asr.paper.benchmark_pack import build_benchmark_pack
from stable_asr.paper.contributor_pack import build_contributor_pack
from stable_asr.paper.final_pack import build_final_pack
from stable_asr.paper.scenario_pack import build_scenario_pack
from stable_asr.paper.case_studies import paper_case_studies
from stable_asr.paper.claims import paper_claims
from stable_asr.paper.completion import completion_audit, write_completion_audit_json, write_completion_audit_markdown
from stable_asr.paper.evidence import final_evidence_matrix
from stable_asr.paper.final_experiments import (
    final_experiments_markdown,
    load_final_experiments,
    write_final_experiments_json,
)
from stable_asr.paper.final_config import (
    audit_final_run_files,
    build_final_run_action_plan,
    final_run_file_audit_markdown,
    final_run_config_markdown,
    load_final_run_config,
    write_final_run_config_json,
)
from stable_asr.paper.figures import PAPER_FIGURES, paper_figure
from stable_asr.paper.final_inputs import (
    final_input_collection_report,
    load_final_input_collections,
    write_final_input_collections_json,
)
from stable_asr.paper.integrity import artifact_integrity_manifest, write_artifact_integrity
from stable_asr.paper.leaderboard import export_leaderboard, leaderboard_report, validate_leaderboard_jsonl
from stable_asr.paper.parity import audit_paper_parity, load_paper_parity_checklist, paper_parity_markdown
from stable_asr.paper.platform_parity import audit_platform_parity
from stable_asr.paper.provenance import paper_bundle_provenance, write_paper_provenance
from stable_asr.paper.status import paper_status, write_paper_status_json, write_paper_status_markdown
from stable_asr.paper.suites import benchmark_suite_markdown, load_benchmark_suite, write_benchmark_suite_json
from stable_asr.paper.tables import PAPER_TABLES, load_paper_results, paper_table
from stable_asr.data.sources import data_sources_markdown, load_data_sources, write_data_sources_json
from stable_asr.models.adapters.registry import (
    adapter_registry_markdown,
    load_adapter_registry,
    write_adapter_registry_json,
)
from stable_asr.models.registry import (
    load_model_registry,
    model_registry_markdown,
    write_model_registry_json,
)
from stable_asr.references import (
    asr_collections_acquisition_markdown,
    asr_collections_bibtex,
    asr_collections_markdown,
    asr_collections_reference_markdown,
    asr_collections_source_manifest,
    audit_reference_workqueue_evidence,
    audit_asr_collection_coverage,
    audit_asr_collection_licenses,
    audit_asr_collection_readiness,
    audit_turn_collection_coverage,
    load_asr_collections,
    load_turn_collections,
    reference_workqueue_assignments,
    reference_workqueue_assignments_markdown,
    reference_workqueue_assignments_tsv,
    reference_workqueue_evidence_markdown,
    reference_workqueue_from_registries,
    reference_workqueue_jsonl,
    reference_workqueue_markdown,
    turn_collections_acquisition_markdown,
    turn_collections_markdown,
    turn_collections_source_manifest,
    write_asr_collections_json,
    write_turn_collections_json,
)
from stable_asr.roadmap import load_roadmap, roadmap_status
from stable_asr.scenarios.suites import scenario_suite_markdown, load_scenario_suite, write_scenario_suite_json
from stable_asr.paper.cards import model_card_markdown, model_card_payload, write_model_card_json
from stable_asr.schemas import load_schema_registry, schema_registry_markdown, write_schema_registry_json


@dataclass(frozen=True)
class PaperArtifactBundle:
    output_dir: str
    index_path: str
    manifest_path: str
    results: dict[str, str]
    artifact_integrity: dict[str, str]
    provenance: dict[str, str]
    tables: dict[str, str]
    figures: dict[str, str]
    leaderboards: dict[str, str]
    leaderboard_validation: dict[str, str]
    leaderboard_reports: dict[str, str]
    benchmark_suite: dict[str, str]
    starter_packs: dict[str, str]
    data_sources: dict[str, str]
    adapter_registry: dict[str, str]
    model_registry: dict[str, str]
    model_cards: dict[str, str]
    schema_registry: dict[str, str]
    asr_collections: dict[str, str]
    turn_collections: dict[str, str]
    reference_workqueue: dict[str, str]
    scenario_suite: dict[str, str]
    case_studies: dict[str, str]
    paper_parity: dict[str, str]
    platform_parity: dict[str, str]
    platform_catalog: dict[str, str]
    final_experiments: dict[str, str]
    final_input_collections: dict[str, str]
    final_run_config: dict[str, str]
    final_run_file_audit: dict[str, str]
    final_run_action_plan: dict[str, str]
    final_evidence_matrix: dict[str, str]
    paper_status: dict[str, str]
    roadmap_status: dict[str, str]
    completion_audit: dict[str, str]
    claims: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "index_path": self.index_path,
            "manifest_path": self.manifest_path,
            "results": self.results,
            "artifact_integrity": self.artifact_integrity,
            "provenance": self.provenance,
            "tables": self.tables,
            "figures": self.figures,
            "leaderboards": self.leaderboards,
            "leaderboard_validation": self.leaderboard_validation,
            "leaderboard_reports": self.leaderboard_reports,
            "benchmark_suite": self.benchmark_suite,
            "starter_packs": self.starter_packs,
            "data_sources": self.data_sources,
            "adapter_registry": self.adapter_registry,
            "model_registry": self.model_registry,
            "model_cards": self.model_cards,
            "schema_registry": self.schema_registry,
            "asr_collections": self.asr_collections,
            "turn_collections": self.turn_collections,
            "reference_workqueue": self.reference_workqueue,
            "scenario_suite": self.scenario_suite,
            "case_studies": self.case_studies,
            "paper_parity": self.paper_parity,
            "platform_parity": self.platform_parity,
            "platform_catalog": self.platform_catalog,
            "final_experiments": self.final_experiments,
            "final_input_collections": self.final_input_collections,
            "final_run_config": self.final_run_config,
            "final_run_file_audit": self.final_run_file_audit,
            "final_run_action_plan": self.final_run_action_plan,
            "final_evidence_matrix": self.final_evidence_matrix,
            "paper_status": self.paper_status,
            "roadmap_status": self.roadmap_status,
            "completion_audit": self.completion_audit,
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

    results_artifacts = {"json": str(output_dir / "paper_results.json")}
    _copy_results(results_path, Path(results_artifacts["json"]))

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
    leaderboard_validation_report = validate_leaderboard_jsonl(leaderboards["jsonl"], require_complete_suite=True)
    leaderboard_validation = {
        "json": str(output_dir / "leaderboard_validation.json"),
        "markdown": str(output_dir / "LEADERBOARD_VALIDATION.md"),
    }
    Path(leaderboard_validation["json"]).write_text(
        json.dumps(leaderboard_validation_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(leaderboard_validation["markdown"]).write_text(
        leaderboard_validation_report.to_markdown(),
        encoding="utf-8",
    )
    leaderboard_ranking_report = leaderboard_report(
        leaderboards["jsonl"],
        require_complete_suite=True,
    )
    leaderboard_reports = {
        "json": str(output_dir / "leaderboard_report.json"),
        "markdown": str(output_dir / "LEADERBOARD_REPORT.md"),
    }
    Path(leaderboard_reports["json"]).write_text(
        json.dumps(leaderboard_ranking_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(leaderboard_reports["markdown"]).write_text(
        leaderboard_ranking_report.to_markdown(),
        encoding="utf-8",
    )
    suite = load_benchmark_suite()
    benchmark_suite = {
        "json": write_benchmark_suite_json(output_dir / "benchmark_suite.json", suite),
        "markdown": str(output_dir / "BENCHMARK_SUITE.md"),
    }
    Path(benchmark_suite["markdown"]).write_text(benchmark_suite_markdown(suite), encoding="utf-8")
    starter_packs = _write_starter_packs(output_dir)
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
    models = load_model_registry()
    model_registry = {
        "json": write_model_registry_json(output_dir / "model_registry.json", models),
        "markdown": str(output_dir / "MODELS.md"),
    }
    Path(model_registry["markdown"]).write_text(model_registry_markdown(models), encoding="utf-8")
    results = load_paper_results(results_path)
    nanoturn = results.get("nanoturn", {}) if isinstance(results, dict) else {}
    nanoturn_metrics = nanoturn.get("metrics") if isinstance(nanoturn, dict) else None
    nanoturn_metrics_path = nanoturn.get("metrics_path") if isinstance(nanoturn, dict) else None
    if not isinstance(nanoturn_metrics, dict):
        nanoturn_metrics = None
    if not isinstance(nanoturn_metrics_path, str):
        nanoturn_metrics_path = None
    model_payload = model_card_payload(
        "configs/models/stable_asr_models.json",
        model_id=str(nanoturn_metrics.get("model_type", "nanoturn_pico")) if nanoturn_metrics else "nanoturn_pico",
        metrics_path=nanoturn_metrics_path,
        metrics=nanoturn_metrics,
    )
    model_cards = {
        "json": write_model_card_json(model_payload, output_dir / "model_card.json"),
        "markdown": str(output_dir / "MODEL_CARD.md"),
    }
    Path(model_cards["markdown"]).write_text(model_card_markdown(model_payload), encoding="utf-8")
    schemas = load_schema_registry()
    schema_registry = {
        "json": write_schema_registry_json(output_dir / "schema_registry.json", schemas),
        "markdown": str(output_dir / "SCHEMAS.md"),
    }
    Path(schema_registry["markdown"]).write_text(schema_registry_markdown(schemas), encoding="utf-8")
    asr_reference_registry = load_asr_collections()
    asr_reference_coverage = audit_asr_collection_coverage(
        asr_reference_registry,
        adapters,
        required_priorities=("p0", "p1"),
    )
    asr_reference_readiness = audit_asr_collection_readiness(
        asr_reference_registry,
        adapters,
        required_priorities=("p0", "p1"),
    )
    asr_reference_licenses = audit_asr_collection_licenses(
        asr_reference_registry,
        required_priorities=("p0", "p1"),
    )
    asr_collections = {
        "json": write_asr_collections_json(output_dir / "asr_collections.json", asr_reference_registry),
        "markdown": str(output_dir / "ASR_COLLECTIONS.md"),
        "paper_markdown": str(output_dir / "ASR_REFERENCES.md"),
        "bibtex": str(output_dir / "ASR_REFERENCES.bib"),
        "acquisition_markdown": str(output_dir / "ASR_COLLECTION_ACQUISITION.md"),
        "source_manifest_json": str(output_dir / "asr_collection_source_manifest.json"),
        "license_json": str(output_dir / "asr_collection_license_review.json"),
        "license_markdown": str(output_dir / "ASR_COLLECTION_LICENSE_REVIEW.md"),
        "coverage_json": str(output_dir / "asr_collection_coverage.json"),
        "coverage_markdown": str(output_dir / "ASR_COLLECTION_COVERAGE.md"),
        "readiness_json": str(output_dir / "asr_collection_readiness.json"),
        "readiness_markdown": str(output_dir / "ASR_COLLECTION_READINESS.md"),
    }
    Path(asr_collections["markdown"]).write_text(asr_collections_markdown(asr_reference_registry), encoding="utf-8")
    Path(asr_collections["paper_markdown"]).write_text(
        asr_collections_reference_markdown(asr_reference_registry),
        encoding="utf-8",
    )
    Path(asr_collections["bibtex"]).write_text(asr_collections_bibtex(asr_reference_registry), encoding="utf-8")
    Path(asr_collections["acquisition_markdown"]).write_text(
        asr_collections_acquisition_markdown(asr_reference_registry),
        encoding="utf-8",
    )
    Path(asr_collections["source_manifest_json"]).write_text(
        json.dumps(asr_collections_source_manifest(asr_reference_registry), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(asr_collections["license_json"]).write_text(
        json.dumps(asr_reference_licenses.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(asr_collections["license_markdown"]).write_text(asr_reference_licenses.to_markdown(), encoding="utf-8")
    Path(asr_collections["coverage_json"]).write_text(
        json.dumps(asr_reference_coverage.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(asr_collections["coverage_markdown"]).write_text(asr_reference_coverage.to_markdown(), encoding="utf-8")
    Path(asr_collections["readiness_json"]).write_text(
        json.dumps(asr_reference_readiness.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(asr_collections["readiness_markdown"]).write_text(
        asr_reference_readiness.to_markdown(),
        encoding="utf-8",
    )
    turn_reference_registry = load_turn_collections()
    turn_reference_coverage = audit_turn_collection_coverage(
        turn_reference_registry,
        sources,
        adapters,
        required_priorities=("p0",),
    )
    turn_collections = {
        "json": write_turn_collections_json(output_dir / "turn_collections.json", turn_reference_registry),
        "markdown": str(output_dir / "TURN_COLLECTIONS.md"),
        "acquisition_markdown": str(output_dir / "TURN_COLLECTION_ACQUISITION.md"),
        "source_manifest_json": str(output_dir / "turn_collection_source_manifest.json"),
        "coverage_json": str(output_dir / "turn_collection_coverage.json"),
        "coverage_markdown": str(output_dir / "TURN_COLLECTION_COVERAGE.md"),
    }
    Path(turn_collections["markdown"]).write_text(turn_collections_markdown(turn_reference_registry), encoding="utf-8")
    Path(turn_collections["acquisition_markdown"]).write_text(
        turn_collections_acquisition_markdown(turn_reference_registry),
        encoding="utf-8",
    )
    Path(turn_collections["source_manifest_json"]).write_text(
        json.dumps(turn_collections_source_manifest(turn_reference_registry), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(turn_collections["coverage_json"]).write_text(
        json.dumps(turn_reference_coverage.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(turn_collections["coverage_markdown"]).write_text(turn_reference_coverage.to_markdown(), encoding="utf-8")
    reference_workqueue_report = reference_workqueue_from_registries(
        asr_registry=asr_reference_registry,
        turn_registry=turn_reference_registry,
        required_priorities=("p0", "p1"),
    )
    reference_workqueue = {
        "json": str(output_dir / "reference_workqueue.json"),
        "jsonl": str(output_dir / "reference_workqueue.jsonl"),
        "markdown": str(output_dir / "REFERENCE_WORKQUEUE.md"),
        "evidence_templates_markdown": str(output_dir / "REFERENCE_EVIDENCE_TEMPLATES.md"),
        "evidence_audit_json": str(output_dir / "reference_evidence_audit.json"),
        "evidence_audit_markdown": str(output_dir / "REFERENCE_EVIDENCE_AUDIT.md"),
        "assignments_json": str(output_dir / "reference_assignments.json"),
        "assignments_tsv": str(output_dir / "reference_assignments.tsv"),
        "assignments_markdown": str(output_dir / "REFERENCE_ASSIGNMENTS.md"),
    }
    reference_assignments = reference_workqueue_assignments(reference_workqueue_report)
    reference_evidence_report = audit_reference_workqueue_evidence(reference_workqueue_report, repo_root=Path("."))
    Path(reference_workqueue["json"]).write_text(
        json.dumps(reference_workqueue_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(reference_workqueue["jsonl"]).write_text(
        reference_workqueue_jsonl(reference_workqueue_report),
        encoding="utf-8",
    )
    Path(reference_workqueue["markdown"]).write_text(
        reference_workqueue_markdown(reference_workqueue_report),
        encoding="utf-8",
    )
    Path(reference_workqueue["evidence_templates_markdown"]).write_text(
        reference_workqueue_evidence_markdown(reference_workqueue_report),
        encoding="utf-8",
    )
    Path(reference_workqueue["evidence_audit_json"]).write_text(
        json.dumps(reference_evidence_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(reference_workqueue["evidence_audit_markdown"]).write_text(
        reference_evidence_report.to_markdown(),
        encoding="utf-8",
    )
    Path(reference_workqueue["assignments_json"]).write_text(
        json.dumps(reference_assignments, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(reference_workqueue["assignments_tsv"]).write_text(
        reference_workqueue_assignments_tsv(reference_assignments),
        encoding="utf-8",
    )
    Path(reference_workqueue["assignments_markdown"]).write_text(
        reference_workqueue_assignments_markdown(reference_assignments),
        encoding="utf-8",
    )
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
    artifact_integrity = {
        "json": str(output_dir / "artifact_hashes.json"),
        "markdown": str(output_dir / "ARTIFACT_HASHES.md"),
    }
    provenance = {
        "json": str(output_dir / "provenance.json"),
        "markdown": str(output_dir / "PROVENANCE.md"),
    }
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
    platform_report = audit_platform_parity(repo_root=Path("."))
    platform_parity = {
        "json": str(output_dir / "platform_parity.json"),
        "markdown": str(output_dir / "PLATFORM_PARITY.md"),
    }
    Path(platform_parity["json"]).write_text(
        json.dumps(platform_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(platform_parity["markdown"]).write_text(platform_report.to_markdown(), encoding="utf-8")
    platform_catalog = _write_platform_catalog(output_dir)
    final_experiment_registry = load_final_experiments()
    final_experiments = {
        "json": write_final_experiments_json(output_dir / "final_experiments.json", final_experiment_registry),
        "markdown": str(output_dir / "FINAL_EXPERIMENTS.md"),
    }
    Path(final_experiments["markdown"]).write_text(
        final_experiments_markdown(final_experiment_registry),
        encoding="utf-8",
    )
    final_input_registry = load_final_input_collections()
    final_input_report = final_input_collection_report(final_input_registry, repo_root=Path("."))
    final_input_collections = {
        "json": write_final_input_collections_json(output_dir / "final_input_collections.json", final_input_registry),
        "audit_json": str(output_dir / "final_input_collection_status.json"),
        "markdown": str(output_dir / "FINAL_INPUT_COLLECTIONS.md"),
    }
    Path(final_input_collections["audit_json"]).write_text(
        json.dumps(final_input_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(final_input_collections["markdown"]).write_text(final_input_report.to_markdown(), encoding="utf-8")
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
    final_action_report = build_final_run_action_plan(final_config, repo_root=Path("."))
    final_run_action_plan = {
        "json": str(output_dir / "final_run_action_plan.json"),
        "markdown": str(output_dir / "FINAL_RUN_ACTION_PLAN.md"),
    }
    Path(final_run_action_plan["json"]).write_text(
        json.dumps(final_action_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(final_run_action_plan["markdown"]).write_text(
        final_action_report.to_markdown(),
        encoding="utf-8",
    )
    final_evidence_report = final_evidence_matrix(artifacts_dir=output_dir)
    final_evidence = {
        "json": str(output_dir / "final_evidence_matrix.json"),
        "markdown": str(output_dir / "FINAL_EVIDENCE_MATRIX.md"),
    }
    Path(final_evidence["json"]).write_text(
        json.dumps(final_evidence_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(final_evidence["markdown"]).write_text(final_evidence_report.to_markdown(), encoding="utf-8")
    status_report = paper_status(repo_root=Path("."), results_path=results_path, artifacts_dir=output_dir)
    paper_status_artifacts = {
        "json": write_paper_status_json(status_report, output_dir / "paper_status.json"),
        "markdown": write_paper_status_markdown(status_report, output_dir / "PAPER_STATUS.md"),
    }
    roadmap_report = roadmap_status(load_roadmap(), repo_root=Path("."))
    roadmap_status_artifacts = {
        "json": str(output_dir / "roadmap_status.json"),
        "markdown": str(output_dir / "ROADMAP_STATUS.md"),
    }
    Path(roadmap_status_artifacts["json"]).write_text(
        json.dumps(roadmap_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(roadmap_status_artifacts["markdown"]).write_text(roadmap_report.to_markdown(), encoding="utf-8")

    # Create provisional bundle files so the claim audit can verify that the
    # paper reproducibility artifacts exist. They are rewritten below with the
    # final claim artifact paths included.
    provisional = PaperArtifactBundle(
        output_dir=str(output_dir),
        index_path=str(index_path),
        manifest_path=str(manifest_path),
        results=results_artifacts,
        artifact_integrity=artifact_integrity,
        provenance=provenance,
        tables=tables,
        figures=figures,
        leaderboards=leaderboards,
        leaderboard_validation=leaderboard_validation,
        leaderboard_reports=leaderboard_reports,
        benchmark_suite=benchmark_suite,
        starter_packs=starter_packs,
        data_sources=data_sources,
        adapter_registry=adapter_registry,
        model_registry=model_registry,
        model_cards=model_cards,
        schema_registry=schema_registry,
        asr_collections=asr_collections,
        turn_collections=turn_collections,
        reference_workqueue=reference_workqueue,
        scenario_suite=scenario_suite,
        case_studies=case_studies,
        paper_parity=paper_parity,
        platform_parity=platform_parity,
        platform_catalog=platform_catalog,
        final_experiments=final_experiments,
        final_input_collections=final_input_collections,
        final_run_config=final_run_config,
        final_run_file_audit=final_run_file_audit,
        final_run_action_plan=final_run_action_plan,
        final_evidence_matrix=final_evidence,
        paper_status=paper_status_artifacts,
        roadmap_status=roadmap_status_artifacts,
        completion_audit={},
        claims={},
    )
    manifest_path.write_text(json.dumps(provisional.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(_artifact_index(results_path, provisional), encoding="utf-8")
    _write_bundle_provenance(results_path, provisional)
    _write_bundle_integrity(provisional)

    claim_artifacts = paper_claims(results_path, output_dir)
    claims = claim_artifacts.to_dict()
    _write_completion_audit_placeholder(output_dir)
    completion_report = completion_audit(repo_root=Path("."), results_path=results_path, artifacts_dir=output_dir)
    completion_artifacts = {
        "json": write_completion_audit_json(completion_report, output_dir / "completion_audit.json"),
        "markdown": write_completion_audit_markdown(completion_report, output_dir / "COMPLETION_AUDIT.md"),
    }
    bundle = PaperArtifactBundle(
        output_dir=str(output_dir),
        index_path=str(index_path),
        manifest_path=str(manifest_path),
        results=results_artifacts,
        artifact_integrity=artifact_integrity,
        provenance=provenance,
        tables=tables,
        figures=figures,
        leaderboards=leaderboards,
        leaderboard_validation=leaderboard_validation,
        leaderboard_reports=leaderboard_reports,
        benchmark_suite=benchmark_suite,
        starter_packs=starter_packs,
        data_sources=data_sources,
        adapter_registry=adapter_registry,
        model_registry=model_registry,
        model_cards=model_cards,
        schema_registry=schema_registry,
        asr_collections=asr_collections,
        turn_collections=turn_collections,
        reference_workqueue=reference_workqueue,
        scenario_suite=scenario_suite,
        case_studies=case_studies,
        paper_parity=paper_parity,
        platform_parity=platform_parity,
        platform_catalog=platform_catalog,
        final_experiments=final_experiments,
        final_input_collections=final_input_collections,
        final_run_config=final_run_config,
        final_run_file_audit=final_run_file_audit,
        final_run_action_plan=final_run_action_plan,
        final_evidence_matrix=final_evidence,
        paper_status=paper_status_artifacts,
        roadmap_status=roadmap_status_artifacts,
        completion_audit=completion_artifacts,
        claims=claims,
    )
    manifest_path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(_artifact_index(results_path, bundle), encoding="utf-8")
    _write_bundle_provenance(results_path, bundle)
    _write_bundle_integrity(bundle)
    return bundle


def _artifact_index(results_path: Path, bundle: PaperArtifactBundle) -> str:
    lines = [
        "# Stable-ASR Paper Artifact Index",
        "",
        f"Results source: `{results_path}`",
        "",
        "## Results",
        "",
    ]
    for name, path in bundle.results.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Tables", ""])
    for name, path in bundle.tables.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Figures", ""])
    for name, path in bundle.figures.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Leaderboards", ""])
    for name, path in bundle.leaderboards.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Leaderboard Validation", ""])
    for name, path in bundle.leaderboard_validation.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Leaderboard Reports", ""])
    for name, path in bundle.leaderboard_reports.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Artifact Integrity", ""])
    for name, path in bundle.artifact_integrity.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Provenance", ""])
    for name, path in bundle.provenance.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Benchmark Suite", ""])
    for name, path in bundle.benchmark_suite.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Starter Packs", ""])
    for name, path in bundle.starter_packs.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Data Sources", ""])
    for name, path in bundle.data_sources.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Adapter Registry", ""])
    for name, path in bundle.adapter_registry.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Model Registry", ""])
    for name, path in bundle.model_registry.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Model Cards", ""])
    for name, path in bundle.model_cards.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Schema Registry", ""])
    for name, path in bundle.schema_registry.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## ASR Reference Collections", ""])
    for name, path in bundle.asr_collections.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Turn Reference Collections", ""])
    for name, path in bundle.turn_collections.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Reference Work Queue", ""])
    for name, path in bundle.reference_workqueue.items():
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
    lines.extend(["", "## Platform Parity", ""])
    for name, path in bundle.platform_parity.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Platform Catalog", ""])
    for name, path in bundle.platform_catalog.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Final Experiments", ""])
    for name, path in bundle.final_experiments.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Final Input Collections", ""])
    for name, path in bundle.final_input_collections.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Final Run Config", ""])
    for name, path in bundle.final_run_config.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Final Run File Audit", ""])
    for name, path in bundle.final_run_file_audit.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Final Run Action Plan", ""])
    for name, path in bundle.final_run_action_plan.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Final Evidence Matrix", ""])
    for name, path in bundle.final_evidence_matrix.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Paper Status", ""])
    for name, path in bundle.paper_status.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Roadmap Status", ""])
    for name, path in bundle.roadmap_status.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Completion Audit", ""])
    for name, path in bundle.completion_audit.items():
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


def _write_bundle_integrity(bundle: PaperArtifactBundle) -> dict[str, str]:
    output_dir = Path(bundle.output_dir)
    report = artifact_integrity_manifest(_bundle_artifact_paths(bundle), root=output_dir)
    return write_artifact_integrity(
        report,
        bundle.artifact_integrity["json"],
        bundle.artifact_integrity["markdown"],
    )


def _write_bundle_provenance(results_path: Path, bundle: PaperArtifactBundle) -> dict[str, str]:
    report = paper_bundle_provenance(results_path, bundle.output_dir)
    return write_paper_provenance(
        report,
        bundle.provenance["json"],
        bundle.provenance["markdown"],
    )


def _write_platform_catalog(output_dir: Path) -> dict[str, str]:
    from stable_asr.catalog import build_platform_catalog, write_platform_catalog_json, write_platform_catalog_markdown

    report = build_platform_catalog(repo_root=Path("."))
    artifacts = {
        "json": str(output_dir / "platform_catalog.json"),
        "markdown": str(output_dir / "PLATFORM_CATALOG.md"),
    }
    write_platform_catalog_json(report, artifacts["json"])
    write_platform_catalog_markdown(report, artifacts["markdown"])
    return artifacts


def _write_completion_audit_placeholder(output_dir: Path) -> None:
    """Break the self-reference between paper-audit and the completion artifact."""

    (output_dir / "completion_audit.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "COMPLETION_AUDIT.md").write_text(
        "# Stable-ASR Completion Audit\n\n## Prompt-To-Artifact Checklist\n\nPending final render.\n",
        encoding="utf-8",
    )


def _bundle_artifact_paths(bundle: PaperArtifactBundle) -> list[str]:
    paths = [bundle.index_path, bundle.manifest_path]
    sections = (
        bundle.results,
        bundle.provenance,
        bundle.tables,
        bundle.figures,
        bundle.leaderboards,
        bundle.leaderboard_validation,
        bundle.leaderboard_reports,
        bundle.benchmark_suite,
        bundle.starter_packs,
        bundle.data_sources,
        bundle.adapter_registry,
        bundle.model_registry,
        bundle.model_cards,
        bundle.schema_registry,
        bundle.asr_collections,
        bundle.turn_collections,
        bundle.reference_workqueue,
        bundle.scenario_suite,
        bundle.case_studies,
        bundle.paper_parity,
        bundle.platform_parity,
        bundle.platform_catalog,
        bundle.final_experiments,
        bundle.final_input_collections,
        bundle.final_run_config,
        bundle.final_run_file_audit,
        bundle.final_run_action_plan,
        bundle.final_evidence_matrix,
        bundle.paper_status,
        bundle.roadmap_status,
        bundle.completion_audit,
        bundle.claims,
    )
    for section in sections:
        paths.extend(section.values())
    return paths


def _copy_results(results_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.resolve() == output_path.resolve():
        return
    shutil.copyfile(results_path, output_path)


def _write_starter_packs(output_dir: Path) -> dict[str, str]:
    packs_dir = output_dir / "starter_packs"
    benchmark = build_benchmark_pack(packs_dir / "benchmark_pack")
    adapter = build_adapter_pack(packs_dir / "adapter_pack")
    scenario = build_scenario_pack(packs_dir / "scenario_pack")
    final = build_final_pack(packs_dir / "final_pack")
    acquisition = build_final_acquisition_pack(packs_dir / "final_acquisition_pack")
    contributor = build_contributor_pack(packs_dir / "contributor_pack")
    paths: dict[str, str] = {}
    for prefix, files in (
        ("benchmark_pack", benchmark.files),
        ("adapter_pack", adapter.files),
        ("scenario_pack", scenario.files),
        ("final_pack", final.files),
        ("final_acquisition_pack", acquisition.files),
        ("contributor_pack", contributor.files),
    ):
        for name, path in sorted(files.items()):
            paths[f"{prefix}:{name}"] = path
    return paths
