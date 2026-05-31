from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.evidence import final_evidence_matrix
from stable_asr.paper.experiments import run_paper_smoke


def test_final_evidence_matrix_reports_final_scale_blockers(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=17, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")

    report = final_evidence_matrix(artifacts_dir=bundle.output_dir)
    markdown = report.to_markdown()

    assert report.ok
    assert not report.final_ready
    assert report.blocked_experiment_count > 0
    assert "Stable-ASR Final Evidence Matrix" in markdown
    assert "real_data_layer_benchmark" in markdown
    assert "data/librispeech/LibriSpeech/dev-clean" in markdown
    assert "real_streaming_asr_systems" in {experiment.id for experiment in report.experiments}


def test_final_evidence_matrix_artifact_checks_are_serializable(tmp_path: Path) -> None:
    report = final_evidence_matrix(artifacts_dir=tmp_path / "missing_artifacts")
    payload = report.to_dict()

    assert payload["ok"] is True
    assert payload["final_ready"] is False
    assert payload["blocked_experiments"]
    assert any(
        artifact["checked"]
        for experiment in payload["experiments"]
        for artifact in experiment["expected_artifacts"]
    )
