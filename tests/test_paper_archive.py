from pathlib import Path
import tarfile

import pytest

from stable_asr.paper.archive import paper_artifact_archive
from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.experiments import run_paper_smoke


def test_paper_artifact_archive_writes_tarball_and_sha256(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=4, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")

    report = paper_artifact_archive(bundle.output_dir, tmp_path / "stable_asr_artifacts.tar.gz")

    assert report.ok
    assert Path(report.archive_path).exists()
    assert Path(report.sha256_path).exists()
    assert report.integrity_ok
    assert report.required_artifacts_ok
    assert "paper_results.json" in report.files
    assert Path(report.sha256_path).read_text(encoding="utf-8").startswith(report.sha256)
    with tarfile.open(report.archive_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "stable-asr-artifacts/paper_results.json" in names
    assert "stable-asr-artifacts/artifact_hashes.json" in names
    assert "stable-asr-artifacts/PROVENANCE.md" in names


def test_paper_artifact_archive_rejects_invalid_bundle(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=4, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")
    Path(bundle.leaderboards["csv"]).unlink()

    with pytest.raises(ValueError, match="artifact integrity verification failed"):
        paper_artifact_archive(bundle.output_dir, tmp_path / "broken.tar.gz")
