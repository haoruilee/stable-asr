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


def test_final_evidence_matrix_maps_archive_to_final_config_path() -> None:
    report = final_evidence_matrix()
    reproducibility = next(experiment for experiment in report.experiments if experiment.id == "final_reproducibility_bundle")
    archive = next(artifact for artifact in reproducibility.expected_artifacts if artifact.name == "artifacts.tar.gz")
    assignment_audit = next(artifact for artifact in reproducibility.expected_artifacts if artifact.name == "FINAL_ASSIGNMENT_AUDIT.md")
    handoff = next(artifact for artifact in reproducibility.expected_artifacts if artifact.name == "FINAL_INPUT_HANDOFF.json")
    handoff_audit = next(artifact for artifact in reproducibility.expected_artifacts if artifact.name == "FINAL_HANDOFF_AUDIT.md")

    assert archive.path == "runs/final/artifacts.tar.gz"
    assert assignment_audit.path == "runs/final/FINAL_ASSIGNMENT_AUDIT.md"
    assert handoff.path == "runs/final/FINAL_INPUT_HANDOFF.json"
    assert handoff_audit.path == "runs/final/FINAL_HANDOFF_AUDIT.md"


def test_final_evidence_matrix_maps_archive_next_to_explicit_artifacts_dir(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    report = final_evidence_matrix(artifacts_dir=artifact_root)
    reproducibility = next(experiment for experiment in report.experiments if experiment.id == "final_reproducibility_bundle")
    archive = next(artifact for artifact in reproducibility.expected_artifacts if artifact.name == "artifacts.tar.gz")
    model_card = next(artifact for artifact in reproducibility.expected_artifacts if artifact.name == "MODEL_CARD.md")
    latex = next(artifact for artifact in reproducibility.expected_artifacts if artifact.name == "paper.tex")
    assignment_audit = next(artifact for artifact in reproducibility.expected_artifacts if artifact.name == "FINAL_ASSIGNMENT_AUDIT.md")
    handoff = next(artifact for artifact in reproducibility.expected_artifacts if artifact.name == "FINAL_INPUT_HANDOFF.json")
    handoff_audit = next(artifact for artifact in reproducibility.expected_artifacts if artifact.name == "FINAL_HANDOFF_AUDIT.md")

    assert archive.path == str(tmp_path / "artifacts.tar.gz")
    assert model_card.path == str(tmp_path / "MODEL_CARD.md")
    assert latex.path == str(tmp_path / "paper.tex")
    assert assignment_audit.path == str(tmp_path / "FINAL_ASSIGNMENT_AUDIT.md")
    assert handoff.path == str(tmp_path / "FINAL_INPUT_HANDOFF.json")
    assert handoff_audit.path == str(tmp_path / "FINAL_HANDOFF_AUDIT.md")
