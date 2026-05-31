from stable_asr.data.registry import (
    TURN_FORMATS,
    convert_turn_manifest,
    load_turn_records,
    summarize_records,
)


def test_registry_detects_jsonl() -> None:
    data_format = TURN_FORMATS.detect("examples/data/turn_demo.jsonl")

    assert data_format.name == "jsonl"


def test_registry_detects_lance() -> None:
    data_format = TURN_FORMATS.detect("data/turn_demo.lance")

    assert data_format.name == "lance"


def test_load_and_summarize_records() -> None:
    records = load_turn_records("examples/data/turn_demo.jsonl")
    summary = summarize_records(records)

    assert summary["records"] == 4
    assert summary["turn_labels"]["complete"] == 1
    assert summary["scenarios"]["backchannel"] == 1


def test_convert_turn_manifest(tmp_path) -> None:
    dest = tmp_path / "converted.jsonl"
    count = convert_turn_manifest("examples/data/turn_demo.jsonl", dest)

    assert count == 4
    assert len(load_turn_records(dest)) == 4
