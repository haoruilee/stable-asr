import json
from pathlib import Path

from stable_asr.schema_validation import validate_schema_file


def test_validate_schema_file_accepts_turn_manifest_fixture() -> None:
    report = validate_schema_file(
        "examples/data/turn_demo.jsonl",
        schema_id="stable_asr.turn_manifest_record.v0",
    )

    assert report.ok
    assert report.records == 4
    assert report.issues == []


def test_validate_schema_file_accepts_prediction_and_streaming_fixtures() -> None:
    predictions = validate_schema_file(
        "tests/fixtures/turn_predictions_sample.jsonl",
        schema_id="stable_asr.turn_prediction_record.v0",
    )
    streaming = validate_schema_file(
        "tests/fixtures/streaming_asr_sample.jsonl",
        schema_id="stable_asr.streaming_asr_record.v0",
    )

    assert predictions.ok
    assert predictions.records == 4
    assert streaming.ok
    assert streaming.records == 2


def test_validate_schema_file_accepts_json_registry() -> None:
    report = validate_schema_file(
        "configs/models/stable_asr_models.json",
        schema_id="stable_asr.model_registry.v0",
    )

    assert report.ok
    assert report.format == "json"
    assert report.records == 1


def test_validate_schema_file_accepts_nanoturn_training_config() -> None:
    report = validate_schema_file(
        "configs/nanoturn_nano.json",
        schema_id="stable_asr.nanoturn_train_config.v0",
    )

    assert report.ok
    assert report.format == "json"
    assert report.records == 1


def test_validate_schema_file_reports_record_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad_turn.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "bad",
                "audio": "",
                "sample_rate": 0,
                "start": -1,
                "end": 0,
                "turn_label": "done",
                "action_label": "take_turn",
                "assistant_speaking": "no",
                "overlap": False,
                "language": "zh",
                "source": "unit",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_schema_file(path, schema_id="stable_asr.turn_manifest_record.v0")
    text = report.to_text()

    assert not report.ok
    assert report.records == 1
    assert "$.turn_label" in text
    assert "$.assistant_speaking" in text
    assert "schema_file_validation: FAILED" in text
