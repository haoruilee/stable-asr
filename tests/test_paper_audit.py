from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.audit import audit_paper_artifacts, audit_paper_release
from stable_asr.paper.cards import dataset_card, experiment_card
from stable_asr.paper.draft import paper_draft
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.latex import paper_latex


def test_paper_audit_accepts_results_and_bundle(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)
    paper_artifact_bundle(result.results_path, tmp_path / "artifacts")

    report = audit_paper_artifacts(result.results_path, tmp_path / "artifacts")

    assert report.ok
    assert "paper_audit: OK" in report.to_text()
    assert report.to_dict()["ok"] is True


def test_paper_audit_rejects_incomplete_bundle(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=6, train_model=False)

    report = audit_paper_artifacts(result.results_path, tmp_path / "missing_artifacts")

    assert not report.ok
    assert "artifact_index" in report.to_text()


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
    assert "OK software/license" in text
    assert "OK software/contributing" in text
    assert "OK data/external_data_sources" in text
    assert "data/lance_data_layer" in text
    assert "baseline/failure_case_mining" in text
    assert "baseline/nanoturn_release_baseline" in text
    assert "streaming/streaming_failure_mining" in text
    assert "scenario/scenario_suite_schema" in text
    assert "adapter/adapter_registry_schema" in text
    assert "paper/paper_parity_schema" in text
    assert "paper/final_experiments_schema" in text
    assert "paper/final_run_config_schema" in text
    assert "reference/asr_collections_schema" in text
    assert "OK reference/asr_collections_coverage" in text
    assert "OK scenario/scenario_suite_coverage" in text
