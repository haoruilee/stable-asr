from pathlib import Path

from stable_asr.data.bootstrap import BootstrapTurnDataConfig, bootstrap_turn_data
from stable_asr.data.profile import profile_turn_records
from stable_asr.data.registry import load_turn_records
from stable_asr.data.turn_from_asr import ASRToTurnConfig


def test_profile_turn_records_reports_distribution_and_warnings() -> None:
    records = load_turn_records("examples/data/turn_demo.jsonl")

    profile = profile_turn_records(records, require_all_turn_labels=True)

    assert profile.records == 4
    assert profile.ok
    assert profile.turn_labels == {"backchannel": 1, "complete": 1, "incomplete": 1, "wait": 1}
    assert profile.duration_stats["median"] == 1.75
    assert "normal_question" in profile.scenarios
    assert "Stable-ASR Turn Data Profile" in profile.to_markdown()


def test_profile_turn_records_warns_on_single_label(tmp_path: Path) -> None:
    result = bootstrap_turn_data(
        "examples/data/asr_metadata.tsv",
        config=BootstrapTurnDataConfig(output_dir=tmp_path),
        audio_root="examples/data",
        asr_to_turn_config=ASRToTurnConfig(include_incomplete=False),
    )
    records = load_turn_records(result.turn_manifest_path)

    profile = profile_turn_records(records, require_all_turn_labels=True)

    assert not profile.ok
    assert "single_turn_label:complete" in profile.warnings
    assert any(warning.startswith("missing_turn_labels:") for warning in profile.warnings)
