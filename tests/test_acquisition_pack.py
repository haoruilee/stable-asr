from pathlib import Path

from stable_asr.paper.acquisition_pack import build_final_acquisition_pack


def test_build_final_acquisition_pack_writes_staging_checklists(tmp_path: Path) -> None:
    report = build_final_acquisition_pack(tmp_path / "acquisition_pack", repo_root=tmp_path)

    assert report.ok
    assert report.collections > 0
    assert report.checklist_rows > report.collections
    assert report.missing_required
    assert report.license_review_items > 0

    output_dir = Path(report.output_dir)
    assert (output_dir / "README.md").exists()
    assert (output_dir / "COMMANDS.md").exists()
    assert (output_dir / "commands.sh").exists()
    assert (output_dir / "configs" / "final" / "paper_final.json").exists()
    assert (output_dir / "configs" / "final" / "input_collections.json").exists()
    assert (output_dir / "configs" / "final" / "asr_command_compare.json").exists()
    assert (output_dir / "acquisition" / "staging_checklist.tsv").exists()
    assert (output_dir / "acquisition" / "staging_checklist.json").exists()
    assert (output_dir / "acquisition" / "DATA_ACQUISITION.md").exists()
    assert (output_dir / "acquisition" / "LICENSE_REVIEW.md").exists()
    assert (output_dir / "acquisition" / "VOICEWORLD_RECORDING_CHECKLIST.md").exists()
    assert (output_dir / "acquisition" / "HANDOFF_TEMPLATE.md").exists()

    checklist = (output_dir / "acquisition" / "staging_checklist.tsv").read_text(encoding="utf-8")
    assert "collection_id\ttitle\tcategory" in checklist
    assert "librispeech_dev_clean" in checklist
    assert "runs/final/turn_train.jsonl" in checklist

    acquisition = (output_dir / "acquisition" / "DATA_ACQUISITION.md").read_text(encoding="utf-8")
    assert "Stable-ASR Final Data Acquisition" in acquisition
    assert "SmartTurn and EasyTurn raw prediction exports" in acquisition

    license_review = (output_dir / "acquisition" / "LICENSE_REVIEW.md").read_text(encoding="utf-8")
    assert "project_or_recording_consent" in license_review
    assert "see_upstream" in license_review
