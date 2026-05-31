from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.status import paper_status


def test_paper_status_summarizes_smoke_and_final_gaps(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=11, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")

    report = paper_status(results_path=result.results_path, artifacts_dir=bundle.output_dir)
    markdown = report.to_markdown()

    assert report.ok
    assert report.smoke_ready
    assert report.structural_ready
    assert not report.final_ready
    assert not report.final_inputs_ready
    assert "Stable-ASR Paper Status" in markdown
    assert "final_inputs_ready" in markdown
    assert "data/librispeech/LibriSpeech/dev-clean" in markdown


def test_paper_status_without_results_is_not_smoke_ready() -> None:
    report = paper_status()

    assert report.ok
    assert not report.smoke_ready
    assert not report.final_ready
