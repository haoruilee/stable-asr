from pathlib import Path

from stable_asr.paper.final_inputs import (
    final_input_collection_report,
    final_input_collections_markdown,
    load_final_input_collections,
    validate_final_input_collections,
)


def test_final_input_collections_validate_and_report_missing_inputs() -> None:
    registry = load_final_input_collections("configs/final/input_collections.json")
    validation = validate_final_input_collections(registry)
    report = final_input_collection_report(registry)

    assert validation.ok
    assert not report.ok
    assert "data/librispeech/LibriSpeech/dev-clean" in report.missing_required
    assert "final_input_collections: NOT_READY" in report.to_text()


def test_final_input_collections_markdown_marks_ready_paths(tmp_path: Path) -> None:
    (tmp_path / "required").mkdir()
    registry = {
        "id": "unit_final_inputs",
        "version": "0.1.0",
        "title": "Unit Final Inputs",
        "collections": [
            {
                "id": "unit",
                "title": "Unit",
                "category": "test",
                "priority": "p0",
                "required": True,
                "license": "test",
                "source_urls": [],
                "required_paths": ["required"],
                "generated_paths": ["generated.json"],
                "commands": ["echo collect"],
                "verification": ["echo verify"],
            }
        ],
    }

    markdown = final_input_collections_markdown(registry, repo_root=tmp_path)

    assert "status: `READY`" in markdown
    assert "present: `required`" in markdown
    assert "pending: `generated.json`" in markdown
