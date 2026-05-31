from pathlib import Path

from stable_asr.paper.final_config import (
    audit_final_voiceworld_real,
    audit_final_run_files,
    bootstrap_final_turn_splits,
    final_run_file_audit_markdown,
    final_run_config_markdown,
    load_final_run_config,
    prepare_final_external_predictions,
    prepare_final_corpora,
    prepare_final_inputs,
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


def test_prepare_final_corpora_writes_existing_inputs_and_skips_missing(tmp_path: Path) -> None:
    chapter = tmp_path / "data/librispeech/LibriSpeech/dev-clean/84/121123"
    chapter.mkdir(parents=True)
    (chapter / "84-121123.trans.txt").write_text(
        "84-121123-0000 WHAT IS THE WEATHER\n",
        encoding="utf-8",
    )
    (chapter / "84-121123-0000.flac").write_bytes(b"")
    config = load_final_run_config()
    config["public_corpora"] = [
        {
            "id": "librispeech_dev_clean",
            "language": "en",
            "corpus": "librispeech",
            "input_dir": "data/librispeech/LibriSpeech/dev-clean",
            "manifest": "runs/final/librispeech_dev_clean/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "test",
        },
        {
            "id": "missing_common_voice",
            "language": "en",
            "corpus": "common_voice",
            "input_dir": "data/common_voice/en",
            "split": "dev",
            "manifest": "runs/final/common_voice_en_dev/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "test",
        },
    ]

    report = prepare_final_corpora(config, repo_root=tmp_path)

    assert report.ok
    assert report.prepared_count == 1
    assert report.skipped_count == 1
    assert (tmp_path / "runs/final/librispeech_dev_clean/asr_manifest.jsonl").exists()
    assert "final_corpora_prepare: READY" in report.to_text()


def test_prepare_final_corpora_require_all_fails_on_missing_input(tmp_path: Path) -> None:
    config = load_final_run_config()
    config["public_corpora"] = [
        {
            "id": "missing_common_voice",
            "language": "en",
            "corpus": "common_voice",
            "input_dir": "data/common_voice/en",
            "split": "dev",
            "manifest": "runs/final/common_voice_en_dev/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "test",
        }
    ]

    report = prepare_final_corpora(config, repo_root=tmp_path, require_all=True)

    assert not report.ok
    assert report.prepared_count == 0
    assert report.skipped_count == 1
    assert "final_corpora_prepare: FAILED" in report.to_text()


def test_bootstrap_final_turn_splits_from_prepared_asr_manifest(tmp_path: Path) -> None:
    chapter = tmp_path / "data/librispeech/LibriSpeech/dev-clean/84/121123"
    chapter.mkdir(parents=True)
    (chapter / "84-121123.trans.txt").write_text(
        "84-121123-0000 WHAT IS THE WEATHER\n"
        "84-121123-0001 TURN ON THE LIGHTS\n"
        "84-121123-0002 OPEN THE DOOR\n"
        "84-121123-0003 CLOSE THE WINDOW\n",
        encoding="utf-8",
    )
    for index in range(4):
        (chapter / f"84-121123-000{index}.flac").write_bytes(b"")
    config = load_final_run_config()
    config["public_corpora"] = [
        {
            "id": "librispeech_dev_clean",
            "language": "en",
            "corpus": "librispeech",
            "input_dir": "data/librispeech/LibriSpeech/dev-clean",
            "manifest": "runs/final/librispeech_dev_clean/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "test",
        }
    ]

    prepare_report = prepare_final_corpora(config, repo_root=tmp_path)
    report = bootstrap_final_turn_splits(config, repo_root=tmp_path)

    assert prepare_report.ok
    assert report.ok
    assert report.asr_records == 4
    assert report.turn_records == 8
    assert report.split_counts["train"] > 0
    assert (tmp_path / "runs/final/turn_train.jsonl").exists()
    assert (tmp_path / "runs/final/final_turn_bootstrap_summary.json").exists()
    assert "voiceworld_real remains" in report.to_text()


def test_prepare_final_external_predictions_converts_and_validates(tmp_path: Path) -> None:
    test_path = tmp_path / "runs/final/turn_test.jsonl"
    raw_path = tmp_path / "runs/final/external/smartturn_raw.jsonl"
    test_path.parent.mkdir(parents=True)
    raw_path.parent.mkdir(parents=True)
    test_path.write_text(
        (
            '{"id":"turn1","audio":"audio/1.wav","sample_rate":16000,"start":0.0,"end":1.0,'
            '"turn_label":"complete","action_label":"take_turn","assistant_speaking":false,'
            '"overlap":false,"language":"en","source":"unit"}\n'
            '{"id":"turn2","audio":"audio/2.wav","sample_rate":16000,"start":0.0,"end":1.0,'
            '"turn_label":"incomplete","action_label":"keep_listening","assistant_speaking":false,'
            '"overlap":false,"language":"en","source":"unit"}\n'
        ),
        encoding="utf-8",
    )
    raw_path.write_text(
        (
            '{"id":"turn1","complete_probability":0.9}\n'
            '{"id":"turn2","complete_probability":0.2}\n'
        ),
        encoding="utf-8",
    )
    config = load_final_run_config()
    config["external_turn_predictions"] = [
        {
            "id": "smart_turn",
            "schema": "smart_turn",
            "raw": "runs/final/external/smartturn_raw.jsonl",
            "converted": "runs/final/external/smartturn_predictions.jsonl",
        },
        {
            "id": "missing_easy_turn",
            "schema": "easyturn",
            "raw": "runs/final/external/easyturn_raw.jsonl",
            "converted": "runs/final/external/easyturn_predictions.jsonl",
        },
    ]

    report = prepare_final_external_predictions(config, repo_root=tmp_path)

    assert report.ok
    assert report.dataset_records == 2
    assert report.prepared_count == 1
    assert report.skipped_count == 1
    assert report.entries[0].coverage_checked
    assert report.entries[0].missing_ids == 0
    assert (tmp_path / "runs/final/external/smartturn_predictions.jsonl").exists()
    assert "final_external_predictions_prepare: READY" in report.to_text()


def test_prepare_final_inputs_runs_sequence_and_audit(tmp_path: Path) -> None:
    chapter = tmp_path / "data/librispeech/LibriSpeech/dev-clean/84/121123"
    chapter.mkdir(parents=True)
    (chapter / "84-121123.trans.txt").write_text(
        "84-121123-0000 WHAT IS THE WEATHER\n"
        "84-121123-0001 TURN ON THE LIGHTS\n"
        "84-121123-0002 OPEN THE DOOR\n",
        encoding="utf-8",
    )
    for index in range(3):
        (chapter / f"84-121123-000{index}.flac").write_bytes(b"")
    voiceworld = tmp_path / "runs/final/voiceworld_real.jsonl"
    voiceworld.parent.mkdir(parents=True)
    factor_payload = (
        '"pause_ms":900,"vad_pause_ms":920,"duration_ms":1000,"snr_db":20,"reverb":"none",'
        '"speaking_rate":1.0,"overlap_offset_ms":0,"network_jitter_ms":0,'
        '"farfield_distance_m":0.5,"code_switch_ratio":0.0,"accent":"standard"'
    )
    rows = []
    scenarios = [
        ("normal_question", "complete", "take_turn", False, False),
        ("incomplete_pause", "incomplete", "keep_listening", False, False),
        ("backchannel", "backchannel", "continue_speaking", True, True),
        ("wait_stop", "wait", "hold", False, False),
        ("user_interruption", "complete", "stop_tts_and_listen", True, True),
        ("side_conversation", "wait", "ignore", False, True),
        ("ambient_speech", "wait", "ignore", False, True),
        ("noisy_farfield", "complete", "take_turn", False, False),
        ("code_switching", "complete", "take_turn", False, False),
    ]
    for index, (scenario, turn_label, action_label, assistant_speaking, overlap) in enumerate(scenarios):
        rows.append(
            f'{{"id":"real_{index}","audio":"audio/{index}.wav","sample_rate":16000,'
            f'"start":0.0,"end":1.0,"turn_label":"{turn_label}","action_label":"{action_label}",'
            f'"assistant_speaking":{str(assistant_speaking).lower()},"overlap":{str(overlap).lower()},'
            f'"language":"en","source":"real","scenario":"{scenario}","metadata":{{{factor_payload}}}}}'
        )
    voiceworld.write_text("\n".join(rows) + "\n", encoding="utf-8")
    asr_config = tmp_path / "configs/final/asr_command_compare.json"
    asr_config.parent.mkdir(parents=True)
    asr_config.write_text('{"systems":[]}\n', encoding="utf-8")
    config = load_final_run_config()
    config["public_corpora"] = [
        {
            "id": "librispeech_dev_clean",
            "language": "en",
            "corpus": "librispeech",
            "input_dir": "data/librispeech/LibriSpeech/dev-clean",
            "manifest": "runs/final/librispeech_dev_clean/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "test",
        }
    ]
    config["external_turn_predictions"] = []

    report = prepare_final_inputs(config, repo_root=tmp_path)

    assert report.ok
    assert report.corpora.prepared_count == 1
    assert report.turn_splits.turn_records == 6
    assert report.external_predictions.prepared_count == 0
    assert report.voiceworld_real.ok
    assert report.missing_required == []
    assert (tmp_path / "runs/final/turn_train.jsonl").exists()
    assert "final_inputs_prepare: READY" in report.to_text()


def test_audit_final_voiceworld_real_checks_scenario_and_factor_coverage(tmp_path: Path) -> None:
    voiceworld = tmp_path / "runs/final/voiceworld_real.jsonl"
    voiceworld.parent.mkdir(parents=True)
    factor_payload = (
        '"pause_ms":900,"vad_pause_ms":920,"duration_ms":1000,"snr_db":20,"reverb":"none",'
        '"speaking_rate":1.0,"overlap_offset_ms":0,"network_jitter_ms":0,'
        '"farfield_distance_m":0.5,"code_switch_ratio":0.0,"accent":"standard"'
    )
    rows = []
    scenarios = [
        ("normal_question", "complete", "take_turn", False, False),
        ("incomplete_pause", "incomplete", "keep_listening", False, False),
        ("backchannel", "backchannel", "continue_speaking", True, True),
        ("wait_stop", "wait", "hold", False, False),
        ("user_interruption", "complete", "stop_tts_and_listen", True, True),
        ("side_conversation", "wait", "ignore", False, True),
        ("ambient_speech", "wait", "ignore", False, True),
        ("noisy_farfield", "complete", "take_turn", False, False),
        ("code_switching", "complete", "take_turn", False, False),
    ]
    for index, (scenario, turn_label, action_label, assistant_speaking, overlap) in enumerate(scenarios):
        rows.append(
            f'{{"id":"real_{index}","audio":"audio/{index}.wav","sample_rate":16000,'
            f'"start":0.0,"end":1.0,"turn_label":"{turn_label}","action_label":"{action_label}",'
            f'"assistant_speaking":{str(assistant_speaking).lower()},"overlap":{str(overlap).lower()},'
            f'"language":"en","source":"real","scenario":"{scenario}","metadata":{{{factor_payload}}}}}'
        )
    voiceworld.write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = load_final_run_config()

    report = audit_final_voiceworld_real(config, repo_root=tmp_path)

    assert report.ok
    assert report.records == len(scenarios)
    assert report.missing_scenarios == []
    assert report.missing_factor_fields == []
    assert "final_voiceworld_real_audit: READY" in report.to_text()


def test_audit_final_voiceworld_real_reports_missing_manifest(tmp_path: Path) -> None:
    report = audit_final_voiceworld_real(load_final_run_config(), repo_root=tmp_path)

    assert not report.ok
    assert report.records == 0
    assert "voiceworld_real manifest is missing" in report.to_text()
