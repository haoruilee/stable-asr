import json
from pathlib import Path

from stable_asr.paper.benchmark_pack import build_benchmark_pack
from stable_asr.paper.submissions import build_streaming_submission, build_turn_submission, index_submission_directory
from stable_asr.schema_validation import validate_schema_file


def test_build_benchmark_pack_writes_contributor_starter_files(tmp_path: Path) -> None:
    report = build_benchmark_pack(tmp_path / "pack")

    assert report.ok
    assert report.schema_registry_ok
    assert report.benchmark_suite_ok
    assert report.sample_validations == {
        "streaming_asr": True,
        "turn_manifest": True,
        "turn_predictions": True,
    }

    output_dir = Path(report.output_dir)
    assert (output_dir / "README.md").exists()
    assert (output_dir / "COMMANDS.md").exists()
    assert (output_dir / "commands.sh").exists()
    assert (output_dir / "configs" / "schema_registry.json").exists()
    assert (output_dir / "configs" / "benchmark_suite.json").exists()
    assert (output_dir / "data" / "turn_demo.jsonl").exists()
    assert (output_dir / "data" / "turn_predictions_sample.jsonl").exists()
    assert (output_dir / "data" / "streaming_asr_sample.jsonl").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "benchmark_pack_v0"
    assert manifest["ok"] is True
    assert "stable-asr turn-submission" in (output_dir / "COMMANDS.md").read_text(encoding="utf-8")
    assert "stable-asr submission-index" in (output_dir / "COMMANDS.md").read_text(encoding="utf-8")

    turn_validation = validate_schema_file(
        output_dir / "data" / "turn_demo.jsonl",
        schema_id="stable_asr.turn_manifest_record.v0",
        registry_path=output_dir / "configs" / "schema_registry.json",
    )
    assert turn_validation.ok


def test_benchmark_pack_samples_run_submission_builders(tmp_path: Path) -> None:
    pack = build_benchmark_pack(tmp_path / "pack")
    output_dir = Path(pack.output_dir)

    turn_report = build_turn_submission(
        dataset=output_dir / "data" / "turn_demo.jsonl",
        predictions=output_dir / "data" / "turn_predictions_sample.jsonl",
        output_dir=output_dir / "submissions" / "turn_oracle",
        system="oracle_fixture",
        suite_path=output_dir / "configs" / "benchmark_suite.json",
    )
    streaming_report = build_streaming_submission(
        input_path=output_dir / "data" / "streaming_asr_sample.jsonl",
        output_dir=output_dir / "submissions" / "streaming_fixture",
        system="streaming_fixture",
        slice_name="adapter",
        suite_path=output_dir / "configs" / "benchmark_suite.json",
    )

    assert turn_report.ok
    assert streaming_report.ok
    assert (output_dir / "submissions" / "turn_oracle" / "leaderboard.jsonl").exists()
    assert (output_dir / "submissions" / "streaming_fixture" / "leaderboard.jsonl").exists()

    leaderboard = index_submission_directory(
        output_dir / "submissions",
        output_dir / "leaderboard",
    )
    assert leaderboard.ok
    assert (output_dir / "leaderboard" / "leaderboard.jsonl").exists()
    assert (output_dir / "leaderboard" / "submissions_index.json").exists()
