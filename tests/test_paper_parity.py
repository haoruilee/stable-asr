from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.parity import (
    audit_paper_parity,
    load_paper_parity_checklist,
    paper_parity_markdown,
    validate_paper_parity_checklist,
    write_paper_parity_checklist_json,
)


def test_default_paper_parity_checklist_matches_config() -> None:
    checklist = load_paper_parity_checklist()
    config_checklist = load_paper_parity_checklist("configs/paper/paper_parity_checklist.json")

    assert validate_paper_parity_checklist(checklist).ok
    assert validate_paper_parity_checklist(config_checklist).ok
    assert checklist["id"] == "stable_asr_paper_parity_v0"
    assert [item["id"] for item in checklist["items"]] == [item["id"] for item in config_checklist["items"]]


def test_paper_parity_markdown_and_json_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "paper_parity_checklist.json"
    write_paper_parity_checklist_json(output)
    checklist = load_paper_parity_checklist(output)

    assert validate_paper_parity_checklist(checklist).ok
    assert "baseline_zoo" in {item["id"] for item in checklist["items"]}


def test_paper_parity_audit_covers_smoke_artifacts_but_not_final_scale(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=7, train_model=False)
    bundle = paper_artifact_bundle(result.results_path, tmp_path / "artifacts")

    report = audit_paper_parity(results_path=result.results_path, artifacts_dir=bundle.output_dir)
    markdown = paper_parity_markdown(report)

    assert report.ok
    assert not report.final_ready
    assert "final-scale ready: `NO`" in markdown
    assert "real public ASR corpus" in markdown


def test_paper_parity_audit_reports_missing_bundle(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=8, seed=7, train_model=False)

    report = audit_paper_parity(results_path=result.results_path, artifacts_dir=tmp_path / "missing")

    assert not report.ok
    assert "artifact:" in report.to_text()
