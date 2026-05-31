from pathlib import Path

import pytest

from stable_asr.data.formats.jsonl import write_jsonl
from stable_asr.data.manifest import load_manifest
from stable_asr.models.adapters import (
    TurnPredictionManifestAdapter,
    convert_turn_prediction_jsonl,
    export_turn_predictions_jsonl,
    load_turn_prediction_jsonl,
)
from stable_asr.models.baselines import TextTurnBaseline


def test_turn_prediction_manifest_adapter_loads_probs_and_labels() -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    adapter = load_turn_prediction_jsonl("tests/fixtures/turn_predictions_sample.jsonl")

    assert adapter.predict(records[0]).label == "complete"
    assert adapter.predict(records[1]).label == "incomplete"
    assert adapter.predict(records[2]).label == "backchannel"
    assert adapter.predict(records[3]).label == "wait"


def test_turn_prediction_manifest_adapter_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    write_jsonl(
        path,
        [
            {"id": "dup", "label": "complete"},
            {"id": "dup", "label": "incomplete"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate prediction id"):
        TurnPredictionManifestAdapter.from_jsonl(path)


def test_turn_prediction_manifest_adapter_requires_matching_record_id() -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    adapter = TurnPredictionManifestAdapter.from_jsonl("tests/fixtures/turn_predictions_sample.jsonl")
    missing = records[0].__class__(
        **{
            **records[0].to_dict(),
            "id": "missing",
        }
    )

    with pytest.raises(KeyError, match="missing prediction"):
        adapter.predict(missing)


def test_convert_smart_turn_predictions_to_stable_schema(tmp_path: Path) -> None:
    output = tmp_path / "smart_turn_converted.jsonl"
    count = convert_turn_prediction_jsonl(
        "tests/fixtures/smart_turn_predictions_sample.jsonl",
        output,
        schema="smart_turn",
    )
    records = load_manifest("examples/data/turn_demo.jsonl")
    adapter = load_turn_prediction_jsonl(output)

    assert count == 4
    assert adapter.predict(records[0]).label == "complete"
    assert adapter.predict(records[1]).label == "incomplete"


def test_convert_easyturn_predictions_to_stable_schema(tmp_path: Path) -> None:
    output = tmp_path / "easyturn_converted.jsonl"
    count = convert_turn_prediction_jsonl(
        "tests/fixtures/easyturn_predictions_sample.jsonl",
        output,
        schema="easyturn",
    )
    records = load_manifest("examples/data/turn_demo.jsonl")
    adapter = load_turn_prediction_jsonl(output)

    assert count == 4
    assert adapter.predict(records[0]).label == "complete"
    assert adapter.predict(records[1]).label == "incomplete"
    assert adapter.predict(records[2]).label == "backchannel"
    assert adapter.predict(records[3]).label == "wait"


def test_export_turn_predictions_jsonl_roundtrip(tmp_path: Path) -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    output = tmp_path / "text_turn_predictions.jsonl"

    rows = export_turn_predictions_jsonl(records, TextTurnBaseline(), output)
    adapter = load_turn_prediction_jsonl(output)

    assert len(rows) == 4
    assert output.exists()
    assert adapter.predict(records[0]).label == "complete"
    assert adapter.predict(records[2]).label == "backchannel"
