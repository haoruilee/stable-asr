from pathlib import Path

from stable_asr.data.recipes import prepare_asr_manifest
from stable_asr.data.turn_from_asr import ASRToTurnConfig, asr_records_to_turn_records


def test_asr_records_to_turn_records_emits_complete_windows(tmp_path: Path) -> None:
    asr_records = _example_asr_records(tmp_path)

    result = asr_records_to_turn_records(asr_records)

    assert result.input_records == 3
    assert len(result.records) == 3
    assert all(record.turn_label == "complete" for record in result.records)
    assert all(record.action_label == "take_turn" for record in result.records)
    assert result.records[0].start == 0.1
    assert result.records[0].end == 2.1
    assert result.records[0].metadata["asr_record_id"] == "asr_demo_0001"


def test_asr_records_to_turn_records_can_emit_incomplete_negatives(tmp_path: Path) -> None:
    asr_records = _example_asr_records(tmp_path)

    result = asr_records_to_turn_records(
        asr_records,
        config=ASRToTurnConfig(include_incomplete=True, incomplete_ratio=0.5),
    )

    labels = [record.turn_label for record in result.records]
    assert labels.count("complete") == 3
    assert labels.count("incomplete") == 3

    incomplete = [record for record in result.records if record.turn_label == "incomplete"][0]
    assert incomplete.action_label == "keep_listening"
    assert incomplete.text is None
    assert incomplete.metadata["label_strategy"] == "asr_truncated_incomplete"
    assert incomplete.metadata["truncation_ratio"] == 0.5


def _example_asr_records(tmp_path: Path):
    return prepare_asr_manifest(
        "examples/data/asr_metadata.tsv",
        tmp_path / "asr_manifest.jsonl",
        audio_root="examples/data",
        default_sample_rate=16000,
    )
