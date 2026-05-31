from pathlib import Path

from stable_asr.paper.final_config import (
    audit_final_run_files,
    final_run_file_audit_markdown,
    final_run_config_markdown,
    load_final_run_config,
    scaffold_final_run,
    validate_final_run_config,
    write_final_run_config_json,
)


def test_default_final_run_config_matches_config() -> None:
    config = load_final_run_config()
    file_config = load_final_run_config("configs/final/paper_final.json")

    assert validate_final_run_config(config).ok
    assert validate_final_run_config(file_config).ok
    assert config["id"] == "stable_asr_final_run_v0"
    assert [item["id"] for item in config["public_corpora"]] == [item["id"] for item in file_config["public_corpora"]]


def test_final_run_config_markdown_and_json_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "paper_final.json"
    write_final_run_config_json(output)
    config = load_final_run_config(output)
    markdown = final_run_config_markdown(config)

    assert "Stable-ASR Final Paper Run Configuration" in markdown
    assert "librispeech_dev_clean" in markdown
    assert "stable-asr paper-parity-audit" in markdown


def test_final_run_config_validation_rejects_missing_split() -> None:
    config = load_final_run_config()
    config["turn_splits"].pop("test")

    report = validate_final_run_config(config)

    assert not report.ok
    assert "turn_splits missing" in report.to_text()


def test_final_run_file_audit_reports_missing_default_inputs() -> None:
    report = audit_final_run_files(load_final_run_config())

    assert not report.ok
    assert "missing required input" in report.to_text()
    assert "librispeech_dev_clean" in final_run_file_audit_markdown(report)


def test_final_run_file_audit_accepts_existing_required_inputs(tmp_path: Path) -> None:
    for relative in [
        "data/corpus/metadata.tsv",
        "runs/final/turn_train.jsonl",
        "runs/final/turn_dev.jsonl",
        "runs/final/turn_test.jsonl",
        "runs/final/voiceworld_real.jsonl",
        "runs/final/external/smartturn_raw.jsonl",
        "configs/final/asr_command_compare.json",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "data/corpus/audio").mkdir(parents=True)
    config = load_final_run_config()
    config["public_corpora"] = [
        {
            "id": "corpus",
            "language": "en",
            "metadata": "data/corpus/metadata.tsv",
            "audio_root": "data/corpus/audio",
            "manifest": "runs/final/corpus/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "test",
        }
    ]
    config["external_turn_predictions"] = [
        {
            "id": "smart_turn",
            "schema": "smart_turn",
            "raw": "runs/final/external/smartturn_raw.jsonl",
            "converted": "runs/final/external/smartturn_predictions.jsonl",
        }
    ]

    report = audit_final_run_files(config, repo_root=tmp_path)

    assert report.ok
    assert "final_run_file_audit: READY" in report.to_text()


def test_scaffold_final_run_creates_directories_without_input_files(tmp_path: Path) -> None:
    config = load_final_run_config()
    report = scaffold_final_run(config, repo_root=tmp_path)

    assert (tmp_path / "runs/final/README.md").exists()
    assert (tmp_path / "runs/final/TURN_SPLITS_README.md").exists()
    assert (tmp_path / "data/librispeech/LibriSpeech/dev-clean/README.md").exists()
    assert not (tmp_path / "runs/final/turn_train.jsonl").exists()
    assert not (tmp_path / "data/librispeech/LibriSpeech/dev-clean/84").exists()
    assert "final_run_scaffold:" in report.to_text()
