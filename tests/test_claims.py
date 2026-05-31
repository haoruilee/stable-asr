from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.claims import audit_claims, claims_markdown, paper_claims
from stable_asr.paper.experiments import run_paper_smoke


def test_audit_claims_accepts_complete_smoke_bundle(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=3, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")

    report = audit_claims(results_path=result.results_path, artifacts_dir=bundle.output_dir)

    assert report.ok
    assert "claim_audit: OK" in report.to_text()
    assert {check.claim_id for check in report.checks}.issuperset(
        {"data_layer", "baseline_zoo", "streaming_asr_eval", "paper_reproducibility"}
    )


def test_audit_claims_reports_missing_artifacts(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=3, train_model=False)

    report = audit_claims(results_path=result.results_path, artifacts_dir=tmp_path / "missing")

    assert not report.ok
    assert "artifact:" in report.to_text()


def test_paper_claims_writes_json_and_markdown(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=3, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")
    artifacts = paper_claims(result.results_path, tmp_path / "claims")
    markdown = Path(artifacts.markdown_path).read_text(encoding="utf-8")

    assert Path(artifacts.json_path).exists()
    assert "Stable-ASR Claim Evidence Matrix" in markdown
    assert "data_layer" in markdown

    report = audit_claims(results_path=result.results_path, artifacts_dir=bundle.output_dir)
    assert "Repository evidence" in claims_markdown(report)
