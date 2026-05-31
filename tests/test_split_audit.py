from pathlib import Path

from stable_asr.data.bootstrap import BootstrapTurnDataConfig, bootstrap_turn_data
from stable_asr.data.registry import load_turn_records
from stable_asr.data.split_audit import audit_turn_splits
from stable_asr.data.turn_from_asr import ASRToTurnConfig


def test_audit_turn_splits_passes_bootstrap_grouped_splits(tmp_path: Path) -> None:
    result = bootstrap_turn_data(
        "examples/data/asr_metadata.tsv",
        config=BootstrapTurnDataConfig(output_dir=tmp_path),
        audio_root="examples/data",
        asr_to_turn_config=ASRToTurnConfig(include_incomplete=True),
    )

    report = audit_turn_splits(
        {
            "train": load_turn_records(result.split_paths["train"]),
            "dev": load_turn_records(result.split_paths["dev"]),
            "test": load_turn_records(result.split_paths["test"]),
        }
    )

    assert report.ok
    assert report.records_by_split["train"] > 0


def test_audit_turn_splits_detects_asr_group_leakage(tmp_path: Path) -> None:
    result = bootstrap_turn_data(
        "examples/data/asr_metadata.tsv",
        config=BootstrapTurnDataConfig(output_dir=tmp_path / "bootstrap"),
        audio_root="examples/data",
        asr_to_turn_config=ASRToTurnConfig(include_incomplete=True),
    )
    records = load_turn_records(result.turn_manifest_path)
    leaking_pair = [
        record
        for record in records
        if record.metadata.get("asr_record_id") == "asr_demo_0001"
    ]

    report = audit_turn_splits(
        {
            "train": [leaking_pair[0]],
            "dev": [leaking_pair[1]],
            "test": load_turn_records(result.split_paths["test"]),
        }
    )

    assert not report.ok
    assert any(leak.field in {"audio", "metadata.asr_record_id"} for leak in report.leaks)
