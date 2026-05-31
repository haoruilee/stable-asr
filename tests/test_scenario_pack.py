from pathlib import Path

from stable_asr.data.manifest import load_manifest
from stable_asr.paper.scenario_pack import build_scenario_pack


def test_build_scenario_pack_writes_voiceworld_starter_files(tmp_path: Path) -> None:
    report = build_scenario_pack(tmp_path / "scenario_pack")

    assert report.ok
    assert report.scenario_suite_ok
    assert report.sample_manifest_ok
    assert report.sample_records == 9

    output_dir = Path(report.output_dir)
    assert (output_dir / "README.md").exists()
    assert (output_dir / "COMMANDS.md").exists()
    assert (output_dir / "commands.sh").exists()
    assert (output_dir / "configs" / "scenario_suite.json").exists()
    assert (output_dir / "configs" / "SCENARIO_SUITE.md").exists()
    assert (output_dir / "data" / "voiceworld_metadata.tsv").exists()
    assert (output_dir / "data" / "voiceworld_manifest.jsonl").exists()

    records = load_manifest(output_dir / "data" / "voiceworld_manifest.jsonl")
    assert len(records) == 9
    assert {record.scenario for record in records} == {
        "ambient_speech",
        "backchannel",
        "code_switching",
        "incomplete_pause",
        "noisy_farfield",
        "normal_question",
        "side_conversation",
        "user_interruption",
        "wait_stop",
    }
    assert "stable-asr eval-scenario" in (output_dir / "COMMANDS.md").read_text(encoding="utf-8")
