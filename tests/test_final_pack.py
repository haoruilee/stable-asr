from pathlib import Path

from stable_asr.paper.final_pack import build_final_pack


def test_build_final_pack_writes_runbook_without_fake_evidence(tmp_path: Path) -> None:
    report = build_final_pack(tmp_path / "final_pack")

    assert report.ok
    assert not report.final_ready
    assert report.missing_required
    assert report.config_ok
    assert report.input_collections_ok
    assert report.final_experiments_ok
    assert report.scaffold_entries > 0

    output_dir = Path(report.output_dir)
    assert (output_dir / "README.md").exists()
    assert (output_dir / "COMMANDS.md").exists()
    assert (output_dir / "commands.sh").exists()
    assert (output_dir / "configs" / "final" / "paper_final.json").exists()
    assert (output_dir / "configs" / "final" / "input_collections.json").exists()
    assert (output_dir / "configs" / "final" / "asr_command_compare.json").exists()
    assert (output_dir / "configs" / "paper" / "final_experiments.json").exists()
    assert (output_dir / "reports" / "FINAL_RUN_ACTION_PLAN.md").exists()
    assert (output_dir / "reports" / "FINAL_RUN_FILE_AUDIT.md").exists()
    assert (output_dir / "reports" / "FINAL_EVIDENCE_MATRIX.md").exists()
    assert (output_dir / "reports" / "FINAL_INPUT_COLLECTIONS.md").exists()
    assert (output_dir / "NEXT_COMMANDS.md").exists()
    assert (output_dir / "runs" / "final" / "README.md").exists()

    assert not (output_dir / "runs" / "final" / "turn_train.jsonl").exists()
    assert not (output_dir / "runs" / "final" / "nanoturn" / "checkpoint.pt").exists()
    assert not (output_dir / "data" / "voiceworld" / "metadata.tsv").exists()
    assert "NOT_READY" in (output_dir / "README.md").read_text(encoding="utf-8")
    assert "--check-files" in (output_dir / "COMMANDS.md").read_text(encoding="utf-8")
    assert "stable-asr final-results" in (output_dir / "NEXT_COMMANDS.md").read_text(encoding="utf-8")
