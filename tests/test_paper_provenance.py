from pathlib import Path

from stable_asr.paper.provenance import paper_bundle_provenance, write_paper_provenance


def test_paper_bundle_provenance_records_results_configs_and_git(tmp_path: Path) -> None:
    results = tmp_path / "paper_results.json"
    results.write_text('{"meta":{"artifact_version":"test"}}\n', encoding="utf-8")

    report = paper_bundle_provenance(
        results,
        tmp_path / "artifacts",
        repo_root=Path("."),
        config_paths=["pyproject.toml"],
    )
    outputs = write_paper_provenance(report, tmp_path / "provenance.json", tmp_path / "PROVENANCE.md")

    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    json_text = Path(outputs["json"]).read_text(encoding="utf-8")

    assert report.results.exists
    assert report.results.sha256 is not None
    assert report.configs[0].path == "pyproject.toml"
    assert report.configs[0].exists
    assert "Stable-ASR Paper Provenance" in markdown
    assert "stable-asr paper-bundle" in markdown
    assert '"git"' in json_text
