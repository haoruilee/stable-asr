import json
import hashlib
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
    assert isinstance(report.final_inputs_ready, bool)
    assert isinstance(report.final_assignment_ready, bool)
    assert isinstance(report.final_handoff_ready, bool)
    if not report.final_assignment_ready:
        assert report.final_assignment.missing
    assert {action.id for action in report.next_actions} == {
        "collect_final_inputs",
        "fill_final_assignment",
        "complete_final_handoff",
        "assemble_final_release",
    }
    assert report.next_actions[0].status in {"needed", "done"}
    assert "Stable-ASR Paper Status" in markdown
    assert "final_inputs_ready" in markdown
    assert "final_assignment_ready" in markdown
    assert "final_handoff_ready" in markdown
    assert "Next Actions" in markdown
    assert "stable-asr final-config --config configs/final/paper_final.json --plan-missing" in markdown
    assert "stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack" in markdown
    assert "runs/final_acquisition_pack/acquisition/assignments.json" in markdown
    assert "runs/final/FINAL_INPUT_HANDOFF.json" in markdown
    assert "runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md" in markdown
    assert "collect_final_inputs" in markdown


def test_paper_status_infers_release_smoke_paths(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    result = run_paper_smoke(release_dir / "paper", episodes=8, seed=11, train_model=False)
    paper_artifact_bundle(result.results_path, release_dir / "artifacts")

    report = paper_status(release_dir=release_dir)

    assert report.smoke_ready
    assert report.structural_ready
    assert not report.final_ready


def test_paper_status_without_results_is_not_smoke_ready() -> None:
    report = paper_status()

    assert report.ok
    assert not report.smoke_ready
    assert not report.final_ready
    assert report.next_actions[-1].status in {"blocked", "ready"}


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
    assignment_action = next(action for action in report.next_actions if action.id == "fill_final_assignment")
    assert assignment_action.status == "done"
    assert "final_assignment_ready" in report.to_markdown()


def test_paper_status_accepts_strict_final_handoff_gate(tmp_path: Path) -> None:
    write_final_run_config_json(tmp_path / "configs/final/paper_final.json")
    staged = tmp_path / "data.txt"
    staged.write_text("stable-asr\n", encoding="utf-8")
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    handoff = tmp_path / "runs/final/FINAL_INPUT_HANDOFF.json"
    handoff_schema_validation = tmp_path / "runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md"
    handoff_audit = tmp_path / "runs/final/FINAL_HANDOFF_AUDIT.md"
    handoff.parent.mkdir(parents=True)
    handoff.write_text(
        json.dumps(
            {
                "version": "stable_asr_final_handoff_v0",
                "entries": [
                    {
                        "collection_id": "unit_collection",
                        "owner": "owner",
                        "staged_paths": ["data.txt"],
                        "source_urls": ["https://example.com/source"],
                        "license_or_consent_notes": "local fixture with project permission",
                        "commands_run": ["echo build"],
                        "verification_outputs": ["pytest"],
                        "checksums": [{"path": "data.txt", "sha256": digest, "bytes": staged.stat().st_size}],
                        "known_gaps": [],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff_schema_validation.write_text("# schema validation\n", encoding="utf-8")
    handoff_audit.write_text("# audit\n", encoding="utf-8")

    report = paper_status(repo_root=tmp_path)
    payload = report.to_dict()

    assert report.final_handoff_ready
    assert payload["final_handoff_ready"] is True
    assert payload["final_handoff"]["missing"] == []
    assert payload["final_handoff"]["handoff_schema_validation"].endswith("runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md")
    assert payload["final_handoff"]["checked_paths"] == ["data.txt"]
    handoff_action = next(action for action in report.next_actions if action.id == "complete_final_handoff")
    assert handoff_action.status == "done"
    assert "final_handoff_ready" in report.to_markdown()
