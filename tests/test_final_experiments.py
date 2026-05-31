from pathlib import Path

from stable_asr.paper.final_config import load_final_run_config
from stable_asr.paper.final_experiments import (
    final_experiments_markdown,
    load_final_experiments,
    validate_final_experiments,
    write_final_experiments_json,
)


def test_default_final_experiments_match_config() -> None:
    registry = load_final_experiments()
    config_registry = load_final_experiments("configs/paper/final_experiments.json")

    assert validate_final_experiments(registry).ok
    assert validate_final_experiments(config_registry).ok
    assert registry["id"] == "stable_asr_final_experiments_v0"
    assert [item["id"] for item in registry["experiments"]] == [item["id"] for item in config_registry["experiments"]]


def test_final_experiments_markdown_and_json_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "final_experiments.json"
    write_final_experiments_json(output)
    registry = load_final_experiments(output)
    markdown = final_experiments_markdown(registry)

    assert "Stable-ASR Final-Scale Experiment Plan" in markdown
    assert "real_data_layer_benchmark" in markdown
    assert "real_streaming_asr_systems" in markdown
    assert "stable-asr benchmark-data" in markdown


def test_external_turn_commands_match_final_config_prediction_paths() -> None:
    registry = load_final_experiments()
    config = load_final_run_config()
    experiment = next(item for item in registry["experiments"] if item["id"] == "external_turn_baselines")
    commands = "\n".join(experiment["commands"])

    for prediction in config["external_turn_predictions"]:
        assert prediction["raw"] in commands
        assert prediction["converted"] in commands
    assert "runs/final/smartturn_raw.jsonl" not in commands
    assert "runs/final/easyturn_raw.jsonl" not in commands


def test_final_experiments_validation_rejects_duplicate_id() -> None:
    registry = load_final_experiments()
    registry["experiments"].append(dict(registry["experiments"][0]))

    report = validate_final_experiments(registry)

    assert not report.ok
    assert "duplicate experiment id" in report.to_text()
