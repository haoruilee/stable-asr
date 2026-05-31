from pathlib import Path

from stable_asr.data.bootstrap import BootstrapTurnDataConfig, bootstrap_turn_data
from stable_asr.data.registry import load_turn_records
from stable_asr.data.split_audit import audit_turn_splits
from stable_asr.data.turn_from_asr import ASRToTurnConfig


def test_bootstrap_turn_data_writes_manifests_splits_and_report(tmp_path: Path) -> None:
    result = bootstrap_turn_data(
        "examples/data/asr_metadata.tsv",
        config=BootstrapTurnDataConfig(output_dir=tmp_path),
        audio_root="examples/data",
        default_sample_rate=16000,
        asr_to_turn_config=ASRToTurnConfig(include_incomplete=True),
    )

    assert Path(result.asr_manifest_path).exists()
    assert Path(result.turn_manifest_path).exists()
    assert Path(result.summary_path).exists()
    assert Path(result.report_path).exists()
    assert set(result.split_paths) == {"train", "dev", "test"}
    assert all(Path(path).exists() for path in result.split_paths.values())
    assert result.to_dict()["turn"]["output_records"] == 6
    assert result.to_dict()["splits"]["train"]["records"] > 0

    turn_records = load_turn_records(result.turn_manifest_path)
    assert len(turn_records) == 6
    assert {record.turn_label for record in turn_records} == {"complete", "incomplete"}
    assert "train-turn --dataset" in Path(result.report_path).read_text(encoding="utf-8")

    split_audit = audit_turn_splits({name: load_turn_records(path) for name, path in result.split_paths.items()})
    assert split_audit.ok
