from pathlib import Path

from stable_asr.paper.contributor_pack import build_contributor_pack


def test_build_contributor_pack_writes_all_starter_tracks(tmp_path: Path) -> None:
    report = build_contributor_pack(tmp_path / "contributor_pack")

    assert report.ok
    assert set(report.pack_statuses) == {
        "adapter_pack",
        "benchmark_pack",
        "final_acquisition_pack",
        "final_pack",
        "scenario_pack",
    }
    assert all(report.pack_statuses.values())
    assert len(report.template_files) >= 6

    output_dir = Path(report.output_dir)
    assert (output_dir / "README.md").exists()
    assert (output_dir / "COMMANDS.md").exists()
    assert (output_dir / "CONTRIBUTION_TRACKS.md").exists()
    assert (output_dir / "commands.sh").exists()
    assert (output_dir / "github_templates" / "PULL_REQUEST_TEMPLATE.md").exists()
    assert (output_dir / "github_templates" / "ISSUE_TEMPLATE" / "final_data_acquisition.yml").exists()
    assert (output_dir / "packs" / "benchmark_pack" / "README.md").exists()
    assert (output_dir / "packs" / "adapter_pack" / "README.md").exists()
    assert (output_dir / "packs" / "scenario_pack" / "README.md").exists()
    assert (output_dir / "packs" / "final_pack" / "README.md").exists()
    assert (output_dir / "packs" / "final_acquisition_pack" / "README.md").exists()
    assert (output_dir / "references" / "REFERENCE_WORKQUEUE.md").exists()
    assert (output_dir / "references" / "reference_workqueue.json").exists()
    assert (output_dir / "references" / "REFERENCE_EVIDENCE_AUDIT.md").exists()
    assert (output_dir / "references" / "reference_evidence_audit.json").exists()
    assert (output_dir / "references" / "REFERENCE_ASSIGNMENTS.md").exists()
    assert (output_dir / "references" / "reference_assignments.tsv").exists()

    tracks = (output_dir / "CONTRIBUTION_TRACKS.md").read_text(encoding="utf-8")
    assert "Reference collection and license review" in tracks
    assert "Benchmark submission" in tracks
    assert "External ASR adapter" in tracks
    assert "Final input acquisition" in tracks
    assert "final-assignment-audit" in tracks

    commands = (output_dir / "COMMANDS.md").read_text(encoding="utf-8")
    assert "packs/benchmark_pack" in commands
    assert "packs/final_acquisition_pack" in commands
    assert "reference-workqueue" in commands
    assert "--audit-evidence" in commands
    assert "reference-assignment-audit" in commands
    assert "asr:funasr" in (output_dir / "references" / "REFERENCE_WORKQUEUE.md").read_text(encoding="utf-8")
    assert "Reference Evidence Audit" in (output_dir / "references" / "REFERENCE_EVIDENCE_AUDIT.md").read_text(
        encoding="utf-8"
    )
    assert "Owner Workflow" in (output_dir / "references" / "REFERENCE_ASSIGNMENTS.md").read_text(encoding="utf-8")
