from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.figures import PAPER_FIGURES
from stable_asr.paper.tables import PAPER_TABLES


def test_paper_artifact_bundle_generates_tables_figures_and_index(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=5, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")

    assert Path(bundle.index_path).exists()
    assert Path(bundle.manifest_path).exists()
    assert set(bundle.results) == {"json"}
    assert set(bundle.tables) == set(PAPER_TABLES)
    assert set(bundle.figures) == set(PAPER_FIGURES)
    assert set(bundle.leaderboards) == {"jsonl", "csv"}
    assert set(bundle.leaderboard_validation) == {"json", "markdown"}
    assert set(bundle.leaderboard_reports) == {"json", "markdown"}
    assert set(bundle.artifact_integrity) == {"json", "markdown"}
    assert set(bundle.provenance) == {"json", "markdown"}
    assert set(bundle.benchmark_suite) == {"json", "markdown"}
    assert "benchmark_pack:readme" in bundle.starter_packs
    assert "benchmark_pack:commands_script" in bundle.starter_packs
    assert "adapter_pack:readme" in bundle.starter_packs
    assert "adapter_pack:command_config" in bundle.starter_packs
    assert "scenario_pack:readme" in bundle.starter_packs
    assert "scenario_pack:metadata" in bundle.starter_packs
    assert "final_pack:readme" in bundle.starter_packs
    assert "final_pack:action_plan_markdown" in bundle.starter_packs
    assert "final_acquisition_pack:readme" in bundle.starter_packs
    assert "final_acquisition_pack:checklist_tsv" in bundle.starter_packs
    assert "final_acquisition_pack:assignments_json" in bundle.starter_packs
    assert "final_acquisition_pack:assignments_markdown" in bundle.starter_packs
    assert "final_acquisition_pack:issue_index_markdown" in bundle.starter_packs
    assert "final_acquisition_pack:issue_template:librispeech_dev_clean" in bundle.starter_packs
    assert "final_acquisition_pack:handoff_json_template" in bundle.starter_packs
    assert "final_acquisition_pack:handoff_schema_markdown" in bundle.starter_packs
    assert "contributor_pack:readme" in bundle.starter_packs
    assert "contributor_pack:tracks" in bundle.starter_packs
    assert "contributor_pack:reference_workqueue_markdown" in bundle.starter_packs
    assert "contributor_pack:reference_evidence_audit_markdown" in bundle.starter_packs
    assert "contributor_pack:reference_evidence_templates_markdown" in bundle.starter_packs
    assert "contributor_pack:reference_assignments_markdown" in bundle.starter_packs
    assert set(bundle.data_sources) == {"json", "markdown"}
    assert set(bundle.adapter_registry) == {"json", "markdown"}
    assert set(bundle.model_registry) == {"json", "markdown"}
    assert set(bundle.model_cards) == {"json", "markdown"}
    assert set(bundle.schema_registry) == {"json", "markdown"}
    assert set(bundle.asr_collections) == {
        "json",
        "markdown",
        "paper_markdown",
        "bibtex",
        "acquisition_markdown",
        "source_manifest_json",
        "license_json",
        "license_markdown",
        "coverage_json",
        "coverage_markdown",
        "readiness_json",
        "readiness_markdown",
    }
    assert set(bundle.turn_collections) == {
        "json",
        "markdown",
        "acquisition_markdown",
        "source_manifest_json",
        "coverage_json",
        "coverage_markdown",
    }
    assert set(bundle.reference_workqueue) == {
        "json",
        "jsonl",
        "markdown",
        "evidence_templates_markdown",
        "evidence_audit_json",
        "evidence_audit_markdown",
        "assignments_json",
        "assignments_tsv",
        "assignments_markdown",
    }
    assert set(bundle.scenario_suite) == {"json", "markdown"}
    assert set(bundle.case_studies) == {"json", "markdown"}
    assert set(bundle.paper_parity) == {"json", "markdown"}
    assert set(bundle.platform_parity) == {"json", "markdown"}
    assert set(bundle.platform_catalog) == {"json", "markdown"}
    assert set(bundle.final_experiments) == {"json", "markdown"}
    assert set(bundle.final_input_collections) == {"json", "audit_json", "markdown"}
    assert set(bundle.final_run_config) == {"json", "markdown"}
    assert set(bundle.final_run_file_audit) == {"json", "markdown"}
    assert set(bundle.final_run_action_plan) == {"json", "markdown"}
    assert set(bundle.final_evidence_matrix) == {"json", "markdown"}
    assert set(bundle.paper_status) == {"json", "markdown"}
    assert set(bundle.roadmap_status) == {"json", "markdown"}
    assert set(bundle.claims) == {"json", "markdown"}
    assert "Stable-ASR Paper Artifact Index" in Path(bundle.index_path).read_text(encoding="utf-8")
    assert Path(bundle.results["json"]).read_text(encoding="utf-8") == Path(result.results_path).read_text(encoding="utf-8")
    assert "## Results" in Path(bundle.index_path).read_text(encoding="utf-8")
    assert "rule_endpoint" in Path(bundle.tables["baselines"]).read_text(encoding="utf-8")
    assert "Stable-ASR Platform Architecture" in Path(bundle.figures["architecture"]).read_text(encoding="utf-8")
    assert "Baseline Macro F1" in Path(bundle.figures["baselines"]).read_text(encoding="utf-8")
    assert "turn_quality" in Path(bundle.leaderboards["jsonl"]).read_text(encoding="utf-8")
    assert "Stable-ASR Leaderboard Validation" in Path(bundle.leaderboard_validation["markdown"]).read_text(encoding="utf-8")
    assert "status: `OK`" in Path(bundle.leaderboard_validation["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Leaderboard Report" in Path(bundle.leaderboard_reports["markdown"]).read_text(encoding="utf-8")
    assert "ranked_rows" in Path(bundle.leaderboard_reports["json"]).read_text(encoding="utf-8")
    assert "Stable-ASR Artifact Integrity" in Path(bundle.artifact_integrity["markdown"]).read_text(encoding="utf-8")
    assert "sha256" in Path(bundle.artifact_integrity["json"]).read_text(encoding="utf-8")
    assert "Artifact Integrity" in Path(bundle.index_path).read_text(encoding="utf-8")
    assert "Stable-ASR Paper Provenance" in Path(bundle.provenance["markdown"]).read_text(encoding="utf-8")
    assert "generated_at_utc" in Path(bundle.provenance["json"]).read_text(encoding="utf-8")
    assert "Provenance" in Path(bundle.index_path).read_text(encoding="utf-8")
    assert "asr_transcript_conversion" in Path(bundle.benchmark_suite["markdown"]).read_text(encoding="utf-8")
    assert "Starter Packs" in Path(bundle.index_path).read_text(encoding="utf-8")
    assert "Stable-ASR Benchmark Starter Pack" in Path(bundle.starter_packs["benchmark_pack:readme"]).read_text(encoding="utf-8")
    assert "Stable-ASR External ASR Adapter Pack" in Path(bundle.starter_packs["adapter_pack:readme"]).read_text(encoding="utf-8")
    assert "Stable-ASR VoiceWorld Scenario Pack" in Path(bundle.starter_packs["scenario_pack:readme"]).read_text(encoding="utf-8")
    assert "Stable-ASR Final Run Starter Pack" in Path(bundle.starter_packs["final_pack:readme"]).read_text(encoding="utf-8")
    assert "Stable-ASR Final Acquisition Pack" in Path(bundle.starter_packs["final_acquisition_pack:readme"]).read_text(encoding="utf-8")
    assert "Stable-ASR Final Acquisition Assignments" in Path(
        bundle.starter_packs["final_acquisition_pack:assignments_markdown"]
    ).read_text(encoding="utf-8")
    assert "Stable-ASR Final Acquisition Issue Index" in Path(
        bundle.starter_packs["final_acquisition_pack:issue_index_markdown"]
    ).read_text(encoding="utf-8")
    assert "Acceptance Checklist" in Path(
        bundle.starter_packs["final_acquisition_pack:issue_template:librispeech_dev_clean"]
    ).read_text(encoding="utf-8")
    assert "Stable-ASR Contributor Pack" in Path(bundle.starter_packs["contributor_pack:readme"]).read_text(encoding="utf-8")
    assert "Stable-ASR Reference Work Queue" in Path(
        bundle.starter_packs["contributor_pack:reference_workqueue_markdown"]
    ).read_text(encoding="utf-8")
    assert "Stable-ASR Reference Assignments" in Path(
        bundle.starter_packs["contributor_pack:reference_assignments_markdown"]
    ).read_text(encoding="utf-8")
    assert "synthetic_voiceworld" in Path(bundle.data_sources["markdown"]).read_text(encoding="utf-8")
    assert "command_streaming_asr" in Path(bundle.adapter_registry["markdown"]).read_text(encoding="utf-8")
    assert "nanoturn_pico" in Path(bundle.model_registry["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Model Card: NanoTurn Pico" in Path(bundle.model_cards["markdown"]).read_text(encoding="utf-8")
    assert "model_id" in Path(bundle.model_cards["json"]).read_text(encoding="utf-8")
    assert "Stable-ASR Schema Registry" in Path(bundle.schema_registry["markdown"]).read_text(encoding="utf-8")
    assert "stable_asr.turn_manifest_record.v0" in Path(bundle.schema_registry["json"]).read_text(encoding="utf-8")
    assert "Stable-ASR Reference Collections" in Path(bundle.asr_collections["markdown"]).read_text(encoding="utf-8")
    assert "@misc{stableasr_ref_funasr" in Path(bundle.asr_collections["bibtex"]).read_text(encoding="utf-8")
    assert "Stable-ASR Paper Reference Notes" in Path(bundle.asr_collections["paper_markdown"]).read_text(encoding="utf-8")
    assert "ASR Collection Acquisition Plan" in Path(bundle.asr_collections["acquisition_markdown"]).read_text(
        encoding="utf-8"
    )
    assert "stable_asr_asr_reference_source_manifest_v0" in Path(bundle.asr_collections["source_manifest_json"]).read_text(
        encoding="utf-8"
    )
    assert "ASR Collection License Review" in Path(bundle.asr_collections["license_markdown"]).read_text(
        encoding="utf-8"
    )
    assert "license_review_required" in Path(bundle.asr_collections["license_json"]).read_text(encoding="utf-8")
    assert "funasr" in Path(bundle.asr_collections["coverage_markdown"]).read_text(encoding="utf-8")
    assert "required_priorities: `p0, p1`" in Path(bundle.asr_collections["coverage_markdown"]).read_text(encoding="utf-8")
    assert "ASR Collection Readiness" in Path(bundle.asr_collections["readiness_markdown"]).read_text(encoding="utf-8")
    assert "license_review_needed" in Path(bundle.asr_collections["readiness_json"]).read_text(encoding="utf-8")
    assert "Turn And Full-Duplex Reference Collections" in Path(bundle.turn_collections["markdown"]).read_text(
        encoding="utf-8"
    )
    assert "Turn Collection Acquisition Plan" in Path(bundle.turn_collections["acquisition_markdown"]).read_text(
        encoding="utf-8"
    )
    assert "stable_asr_turn_reference_source_manifest_v0" in Path(
        bundle.turn_collections["source_manifest_json"]
    ).read_text(encoding="utf-8")
    assert "missing_required: `0`" in Path(bundle.turn_collections["coverage_markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Reference Work Queue" in Path(bundle.reference_workqueue["markdown"]).read_text(encoding="utf-8")
    assert "stable_asr_reference_workqueue_v0" in Path(bundle.reference_workqueue["json"]).read_text(encoding="utf-8")
    assert '"task_id": "asr:funasr"' in Path(bundle.reference_workqueue["jsonl"]).read_text(encoding="utf-8")
    assert "Reference Evidence Audit" in Path(bundle.reference_workqueue["evidence_audit_markdown"]).read_text(
        encoding="utf-8"
    )
    assert "Stable-ASR Reference Evidence Templates" in Path(
        bundle.reference_workqueue["evidence_templates_markdown"]
    ).read_text(encoding="utf-8")
    assert '"ok": false' in Path(bundle.reference_workqueue["evidence_audit_json"]).read_text(encoding="utf-8")
    assert "Stable-ASR Reference Assignments" in Path(bundle.reference_workqueue["assignments_markdown"]).read_text(
        encoding="utf-8"
    )
    assert "stable_asr_reference_assignments_v0" in Path(bundle.reference_workqueue["assignments_json"]).read_text(
        encoding="utf-8"
    )
    assert "blocked_license_review" in Path(bundle.reference_workqueue["assignments_tsv"]).read_text(encoding="utf-8")
    assert "Reference Work Queue" in Path(bundle.index_path).read_text(encoding="utf-8")
    assert "Stable-ASR VoiceWorld v0 Scenario Suite" in Path(bundle.scenario_suite["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Case Studies" in Path(bundle.case_studies["markdown"]).read_text(encoding="utf-8")
    assert "final-scale ready" in Path(bundle.paper_parity["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Stable-WorldModel Repository Parity" in Path(bundle.platform_parity["markdown"]).read_text(
        encoding="utf-8"
    )
    assert "repository_identity" in Path(bundle.platform_parity["json"]).read_text(encoding="utf-8")
    assert "Stable-ASR Platform Catalog" in Path(bundle.platform_catalog["markdown"]).read_text(encoding="utf-8")
    assert "stable_asr_sources_v0" in Path(bundle.platform_catalog["json"]).read_text(encoding="utf-8")
    assert "real_data_layer_benchmark" in Path(bundle.final_experiments["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Final Input Collections" in Path(bundle.final_input_collections["markdown"]).read_text(encoding="utf-8")
    assert "stable_asr_final_input_collections_v0" in Path(bundle.final_input_collections["json"]).read_text(encoding="utf-8")
    assert "librispeech_dev_clean" in Path(bundle.final_run_config["markdown"]).read_text(encoding="utf-8")
    assert "Final Run File Audit" in Path(bundle.final_run_file_audit["markdown"]).read_text(encoding="utf-8")
    assert "Final Run Action Plan" in Path(bundle.final_run_action_plan["markdown"]).read_text(encoding="utf-8")
    assert "Final Evidence Matrix" in Path(bundle.final_evidence_matrix["markdown"]).read_text(encoding="utf-8")
    assert "final_run_action_plan" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "final_input_collections" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "final_evidence_matrix" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "platform_parity" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "platform_catalog" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "model_cards" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "schema_registry" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "turn_collections" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "reference_workqueue" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "starter_packs" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "artifact_integrity" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "provenance" in Path(bundle.manifest_path).read_text(encoding="utf-8")
    assert "Stable-ASR Paper Status" in Path(bundle.paper_status["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Platform Roadmap" in Path(bundle.roadmap_status["markdown"]).read_text(encoding="utf-8")
    assert "Stable-ASR Claim Evidence Matrix" in Path(bundle.claims["markdown"]).read_text(encoding="utf-8")
