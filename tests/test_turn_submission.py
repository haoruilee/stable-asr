import json
from pathlib import Path

from stable_asr.paper.leaderboard import validate_leaderboard_jsonl
from stable_asr.paper.submissions import build_streaming_submission, build_turn_submission


def test_build_turn_submission_package(tmp_path: Path) -> None:
    report = build_turn_submission(
        dataset="examples/data/turn_demo.jsonl",
        predictions="tests/fixtures/turn_predictions_sample.jsonl",
        output_dir=tmp_path / "submission",
        system="oracle_fixture",
    )

    assert report.ok
    assert report.records == 4
    assert report.schema_validation.ok
    assert report.prediction_validation.ok
    assert report.evaluation is not None
    assert report.evaluation.classification.accuracy == 1.0

    artifacts = report.artifacts
    assert Path(artifacts.manifest).exists()
    assert Path(artifacts.summary_markdown).exists()
    assert Path(artifacts.schema_validation["json"]).exists()
    assert Path(artifacts.prediction_validation["markdown"]).exists()
    assert Path(artifacts.evaluation["json"]).exists()
    assert Path(artifacts.leaderboard["jsonl"]).exists()
    assert Path(artifacts.leaderboard_validation["markdown"]).exists()

    manifest = json.loads(Path(artifacts.manifest).read_text(encoding="utf-8"))
    assert manifest["system"] == "oracle_fixture"
    assert manifest["ok"] is True
    assert "Stable-ASR Turn Submission" in Path(artifacts.summary_markdown).read_text(encoding="utf-8")

    leaderboard_report = validate_leaderboard_jsonl(artifacts.leaderboard["jsonl"])
    assert leaderboard_report.ok
    assert leaderboard_report.rows == 4
    assert leaderboard_report.tasks == {"turn_quality": 4}


def test_build_turn_submission_reports_invalid_predictions(tmp_path: Path) -> None:
    predictions = tmp_path / "bad_predictions.jsonl"
    predictions.write_text('{"id":"zh_turn_000001","label":"complete"}\n', encoding="utf-8")

    report = build_turn_submission(
        dataset="examples/data/turn_demo.jsonl",
        predictions=predictions,
        output_dir=tmp_path / "submission",
        system="bad_fixture",
    )

    assert not report.ok
    assert report.schema_validation.ok
    assert not report.prediction_validation.ok
    assert report.evaluation is None
    assert report.prediction_validation.missing_ids == [
        "zh_turn_000002",
        "zh_turn_000003",
        "zh_turn_000004",
    ]
    assert Path(report.artifacts.leaderboard["jsonl"]).read_text(encoding="utf-8") == ""
    assert "validation failed" in Path(report.artifacts.evaluation["markdown"]).read_text(encoding="utf-8")


def test_build_streaming_submission_package(tmp_path: Path) -> None:
    report = build_streaming_submission(
        input_path="tests/fixtures/streaming_asr_sample.jsonl",
        output_dir=tmp_path / "streaming_submission",
        system="streaming_fixture",
        slice_name="adapter",
    )

    assert report.ok
    assert report.records == 2
    assert report.schema_validation.ok
    assert report.evaluation is not None
    assert report.evaluation.records == 2

    artifacts = report.artifacts
    assert Path(artifacts.manifest).exists()
    assert Path(artifacts.summary_markdown).exists()
    assert Path(artifacts.schema_validation["json"]).exists()
    assert Path(artifacts.evaluation["markdown"]).exists()
    assert Path(artifacts.leaderboard["jsonl"]).exists()
    assert Path(artifacts.leaderboard_validation["markdown"]).exists()

    manifest = json.loads(Path(artifacts.manifest).read_text(encoding="utf-8"))
    assert manifest["version"] == "streaming_submission_v0"
    assert manifest["system"] == "streaming_fixture"
    assert "Stable-ASR Streaming Submission" in Path(artifacts.summary_markdown).read_text(encoding="utf-8")

    leaderboard_report = validate_leaderboard_jsonl(artifacts.leaderboard["jsonl"])
    assert leaderboard_report.ok
    assert leaderboard_report.rows == 9
    assert leaderboard_report.tasks == {"streaming_asr": 9}


def test_build_streaming_submission_reports_schema_errors(tmp_path: Path) -> None:
    input_path = tmp_path / "bad_streaming.jsonl"
    input_path.write_text('{"id":"bad"}\n', encoding="utf-8")

    report = build_streaming_submission(
        input_path=input_path,
        output_dir=tmp_path / "streaming_submission",
        system="bad_streaming",
    )

    assert not report.ok
    assert not report.schema_validation.ok
    assert report.evaluation is None
    assert Path(report.artifacts.leaderboard["jsonl"]).read_text(encoding="utf-8") == ""
    assert "schema validation failed" in Path(report.artifacts.evaluation["markdown"]).read_text(encoding="utf-8")
