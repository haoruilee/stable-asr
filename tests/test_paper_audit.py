import json
from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.audit import audit_paper_artifacts, audit_paper_release
from stable_asr.paper.cards import dataset_card, experiment_card, model_card
from stable_asr.paper.draft import paper_draft
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.latex import paper_latex


def test_paper_audit_accepts_results_and_bundle(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)
    paper_artifact_bundle(result.results_path, tmp_path / "artifacts")

    report = audit_paper_artifacts(result.results_path, tmp_path / "artifacts")

    assert report.ok
    assert "paper_audit: OK" in report.to_text()
    assert "results:json" in report.to_text()
    assert "benchmark_suite:required_artifacts" in report.to_text()
    assert "platform_parity:json" in report.to_text()
    assert "platform_parity:markdown" in report.to_text()
    assert "leaderboard_report:markdown" in report.to_text()
    assert "artifact_integrity:sha256" in report.to_text()
    assert "provenance:json" in report.to_text()
    assert "starter_pack:scenario_manifest" in report.to_text()
    assert "starter_pack:final_manifest" in report.to_text()
    assert "starter_pack:final_acquisition_checklist" in report.to_text()
    assert "starter_pack:final_acquisition_assignments" in report.to_text()
    assert "starter_pack:final_acquisition_assignment_audit_command" in report.to_text()
    assert "starter_pack:final_acquisition_handoff_template" in report.to_text()
    assert "starter_pack:final_acquisition_handoff_schema" in report.to_text()
    assert "starter_pack:contributor_tracks" in report.to_text()
    assert "model_card:markdown" in report.to_text()
    assert "model_registry:markdown" in report.to_text()
    assert "schema_registry:markdown" in report.to_text()
    assert "final_input_collections:markdown" in report.to_text()
    assert "asr_collections:source_manifest" in report.to_text()
    assert "asr_collections:source_manifest_content" in report.to_text()
    assert "asr_collection_readiness:markdown" in report.to_text()
    assert "turn_collections:source_manifest" in report.to_text()
    assert "turn_collections:source_manifest_content" in report.to_text()
    assert "turn_collection_coverage:markdown" in report.to_text()
    assert report.to_dict()["ok"] is True


def test_paper_audit_rejects_tampered_reference_source_manifest(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")
    manifest = Path(bundle.asr_collections["source_manifest_json"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sources"] = payload["sources"][:-1]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_paper_artifacts(result.results_path, bundle.output_dir)

    assert not report.ok
    assert "asr_collections:source_manifest_content" in report.to_text()
    assert "missing:" in report.to_text()


def test_paper_audit_requires_four_asr_transcript_conversions(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)
    payload = json.loads(Path(result.results_path).read_text(encoding="utf-8"))
    payload["streaming_asr"]["asr_transcript_conversions"] = payload["streaming_asr"][
        "asr_transcript_conversions"
    ][:3]
    truncated = tmp_path / "paper_results_three_asr_schemas.json"
    truncated.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_paper_artifacts(truncated)

    assert not report.ok
    assert "asr_transcript_conversions" in report.to_text()
    assert "3/4 conversion(s)" in report.to_text()


def test_paper_audit_rejects_incomplete_bundle(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)

    report = audit_paper_artifacts(result.results_path, tmp_path / "missing_artifacts")

    assert not report.ok
    assert "artifact_index" in report.to_text()


def test_paper_audit_rejects_tampered_artifact_bundle(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")
    Path(bundle.tables["baselines"]).write_text("tampered\n", encoding="utf-8")

    report = audit_paper_artifacts(result.results_path, bundle.output_dir)

    assert not report.ok
    assert "artifact_integrity:sha256" in report.to_text()
    assert "mismatched: tables/baselines.md" in report.to_text()


def test_paper_audit_rejects_tampered_results_copy(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")
    Path(bundle.results["json"]).write_text("{}\n", encoding="utf-8")

    report = audit_paper_artifacts(result.results_path, bundle.output_dir)

    assert not report.ok
    assert "results:json" in report.to_text()
    assert "hash mismatch" in report.to_text()


def test_paper_audit_rejects_missing_benchmark_required_artifact(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")
    Path(bundle.leaderboards["csv"]).unlink()

    report = audit_paper_artifacts(result.results_path, bundle.output_dir)

    assert not report.ok
    assert "benchmark_suite:required_artifacts" in report.to_text()
    assert "leaderboard.csv" in report.to_text()


def test_paper_release_audit_reports_remaining_release_gaps(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")
    markdown = Path(paper_draft(result.results_path, tmp_path / "PAPER_DRAFT.md", artifacts_dir=bundle.output_dir))
    latex = Path(paper_latex(result.results_path, tmp_path / "paper.tex", artifacts_dir=bundle.output_dir))
    data_card = Path(dataset_card("examples/data/turn_demo.jsonl", tmp_path / "DATASET_CARD.md"))
    experiment = Path(experiment_card(result.results_path, tmp_path / "EXPERIMENT_CARD.md"))

    report = audit_paper_release(
        repo_root=Path("."),
        results_path=result.results_path,
        artifacts_dir=bundle.output_dir,
        markdown_draft=markdown,
        latex_draft=latex,
        dataset_card=data_card,
        experiment_card=experiment,
    )

    text = report.to_text()
    assert not report.ok
    assert "paper_release_audit: NOT_READY" in text
    assert "OK software/manifest_in" in text
    assert "OK software/source_manifest_content" in text
    assert "OK software/wheel_data_files" in text
    assert "OK software/schema_registry" in text
    assert "OK software/ci_wheel_smoke" in text
    assert "OK software/ci_lance_smoke" in text
    assert "OK software/pull_request_template" in text
    assert "OK software/issue_template_final_data" in text
    assert "OK software/issue_template_asr_adapter" in text
    assert "OK software/issue_template_voiceworld" in text
    assert "OK software/issue_template_benchmark_submission" in text
    assert "OK software/final_streaming_transcript_export_bridge" in text
    assert "OK software/license" in text
    assert "OK software/contributing" in text
    assert "OK software/security" in text
    assert "OK software/code_of_conduct" in text
    assert "OK data/external_data_sources" in text
    assert "data/lance_data_layer" in text
    assert "baseline/failure_case_mining" in text
    assert "baseline/nanoturn_release_baseline" in text
    assert "streaming/streaming_failure_mining" in text
    assert "scenario/scenario_suite_schema" in text
    assert "adapter/adapter_registry_schema" in text
    assert "paper/paper_parity_schema" in text
    assert "OK software/platform_parity_checklist" in text
    assert "paper/final_experiments_schema" in text
    assert "paper/final_run_config_schema" in text
    assert "reference/asr_collections_schema" in text
    assert "OK reference/asr_collections_coverage" in text
    assert "OK reference/asr_collections_readiness" in text
    assert "OK reference/turn_collections_schema" in text
    assert "OK reference/turn_collections_coverage" in text
    assert "OK scenario/scenario_suite_coverage" in text


def test_paper_release_audit_infers_release_smoke_outputs(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")
    paper_draft(result.results_path, tmp_path / "PAPER_DRAFT.md", artifacts_dir=bundle.output_dir)
    paper_latex(result.results_path, tmp_path / "paper.tex", artifacts_dir=bundle.output_dir)
    dataset_card("examples/data/turn_demo.jsonl", tmp_path / "DATASET_CARD.md")
    experiment_card(result.results_path, tmp_path / "EXPERIMENT_CARD.md")
    model_card("configs/models/stable_asr_models.json", tmp_path / "MODEL_CARD.md", model_id="nanoturn_pico")

    report = audit_paper_release(
        repo_root=Path("."),
        results_path=result.results_path,
        artifacts_dir=bundle.output_dir,
    )

    text = report.to_text()
    assert "OK paper/markdown_draft" in text
    assert "OK paper/latex_draft" in text
    assert "OK data/dataset_card" in text
    assert "OK paper/experiment_card" in text
    assert "OK model/model_card" in text


def test_paper_release_audit_resolves_platform_assets_from_empty_repo_root(tmp_path: Path, monkeypatch) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    monkeypatch.chdir(tmp_path)

    report = audit_paper_release(repo_root=empty_root)
    text = report.to_text()

    assert "OK software/pyproject" in text
    assert "OK software/manifest_in" in text
    assert "OK software/mkdocs_config" in text
    assert "OK software/ci_workflow" in text
    assert "OK software/pull_request_template" in text
    assert "OK software/issue_template_final_data" in text
    assert "OK software/asr_manifest_schema" in text
    assert "OK paper/citation" in text
    assert "OK software/docs_site" in text
