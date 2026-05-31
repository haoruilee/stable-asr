import json
from pathlib import Path

from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.final_config import write_final_run_config_json
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
    assert not report.final_assignment_ready
    assert report.final_assignment.missing
    assert "Stable-ASR Paper Status" in markdown
    assert "final_inputs_ready" in markdown
    assert "final_assignment_ready" in markdown
    assert "runs/final_acquisition_pack/acquisition/assignments.json" in markdown
    assert "data/librispeech/LibriSpeech/dev-clean" in markdown


def test_paper_status_without_results_is_not_smoke_ready() -> None:
    report = paper_status()

    assert report.ok
    assert not report.smoke_ready
    assert not report.final_ready


def test_paper_status_accepts_strict_final_assignment_gate(tmp_path: Path) -> None:
    write_final_run_config_json(tmp_path / "configs/final/paper_final.json")
    assignments = tmp_path / "runs/final_acquisition_pack/acquisition/assignments.json"
    assignment_audit = tmp_path / "runs/final/FINAL_ASSIGNMENT_AUDIT.md"
    assignments.parent.mkdir(parents=True)
    assignment_audit.parent.mkdir(parents=True)
    payload = {
        "rows": [
            {
                "collection_id": "unit",
                "status": "ready_for_handoff",
                "owner": "owner",
                "due_date": "2026-06-30",
                "blocking_release": False,
                "missing_required_paths": [],
                "pending_generated_paths": [],
                "source_urls": [],
            }
        ]
    }
    assignments.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assignment_audit.write_text("# audit\n", encoding="utf-8")

    report = paper_status(repo_root=tmp_path)
    payload = report.to_dict()

    assert report.final_assignment_ready
    assert payload["final_assignment_ready"] is True
    assert payload["final_assignment"]["missing"] == []
    assert "final_assignment_ready" in report.to_markdown()
