import json
import sys
from pathlib import Path

from stable_asr.paper.final_config import (
    audit_final_voiceworld_real,
    audit_final_run_files,
    build_final_run_action_plan,
    bootstrap_final_turn_splits,
    final_run_file_audit_markdown,
    final_run_config_markdown,
    load_final_run_config,
    prepare_final_asr_eval_manifest,
    prepare_final_asr_transcript_conversions,
    prepare_final_external_predictions,
    prepare_final_corpora,
    prepare_final_inputs,
    prepare_final_voiceworld_real,
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
    assert [item["id"] for item in config["external_turn_predictions"]] == [
        item["id"] for item in file_config["external_turn_predictions"]
    ]
    assert "vap" in {item["schema"] for item in config["external_turn_predictions"]}
    assert config["artifacts"]["assignment_audit"] == file_config["artifacts"]["assignment_audit"]


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


def test_final_run_action_plan_maps_missing_inputs_to_commands() -> None:
    report = build_final_run_action_plan(load_final_run_config())

    markdown = report.to_markdown()
    assert not report.ok
    assert report.missing_required
    assert "Stable-ASR Final Run Action Plan" in markdown
    assert "stage_public_corpora" in markdown
    assert "prepare-public-asr --corpus librispeech" in markdown
    assert "collect_voiceworld_real" in markdown
    assert "prepare-voiceworld" in markdown
    assert "prepare-external-predictions" in markdown
    assert "final-assignment-audit" in markdown
    assert "paper-parity-audit" in markdown


def test_final_run_file_audit_accepts_existing_required_inputs(tmp_path: Path) -> None:
    for relative in [
        "data/corpus/metadata.tsv",
        "runs/final/turn_train.jsonl",
        "runs/final/turn_dev.jsonl",
        "runs/final/turn_test.jsonl",
        "runs/final/voiceworld_real.jsonl",
        "data/voiceworld/metadata.tsv",
        "runs/final/external/smartturn_raw.jsonl",
        "configs/final/asr_command_compare.json",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "data/corpus/audio").mkdir(parents=True)
    (tmp_path / "data/voiceworld/audio").mkdir(parents=True)
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


def test_prepare_final_asr_eval_manifest_combines_prepared_manifests(tmp_path: Path) -> None:
    manifest_a = tmp_path / "runs/final/a/asr_manifest.jsonl"
    manifest_b = tmp_path / "runs/final/b/asr_manifest.jsonl"
    manifest_a.parent.mkdir(parents=True)
    manifest_b.parent.mkdir(parents=True)
    manifest_a.write_text(
        '{"id":"utt1","audio":"audio/utt1.wav","sample_rate":16000,"text":"hello","language":"en","source":"a"}\n',
        encoding="utf-8",
    )
    manifest_b.write_text(
        '{"id":"utt2","audio":"audio/utt2.wav","sample_rate":16000,"text":"你好","language":"zh","source":"b"}\n',
        encoding="utf-8",
    )
    config = load_final_run_config()
    config["public_corpora"] = [
        {
            "id": "a",
            "language": "en",
            "corpus": "librispeech",
            "input_dir": "data/a",
            "manifest": "runs/final/a/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "test",
        },
        {
            "id": "b",
            "language": "zh",
            "corpus": "aishell1",
            "input_dir": "data/b",
            "manifest": "runs/final/b/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "test",
        },
    ]

    report = prepare_final_asr_eval_manifest(config, repo_root=tmp_path)

    assert report.ok
    assert report.records == 2
    assert (tmp_path / "runs/final/asr_eval_manifest.jsonl").exists()
    assert "final_asr_eval_manifest: READY" in report.to_text()


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
    voiceworld_metadata = tmp_path / "data/voiceworld/metadata.tsv"
    voiceworld_audio = tmp_path / "data/voiceworld/audio"
    voiceworld_audio.mkdir(parents=True)
    voiceworld_rows = [
        "id\taudio\ttext\tscenario\tturn_label\taction_label\tassistant_speaking\toverlap\t"
        "start\tend\tpause_ms\tvad_pause_ms\tduration_ms\tsnr_db\treverb\tspeaking_rate\t"
        "overlap_offset_ms\tnetwork_jitter_ms\tfarfield_distance_m\tcode_switch_ratio\taccent"
    ]
    for index, (scenario, turn_label, action_label, assistant_speaking, overlap) in enumerate(scenarios):
        audio = f"{index}.wav"
        (voiceworld_audio / audio).write_bytes(b"")
        voiceworld_rows.append(
            f"real_{index}\t{audio}\ttext {index}\t{scenario}\t{turn_label}\t{action_label}\t"
            f"{str(assistant_speaking).lower()}\t{str(overlap).lower()}\t0.0\t1.0\t"
            "900\t920\t1000\t20\tnone\t1.0\t0\t0\t0.5\t0.0\tstandard"
        )
    voiceworld_metadata.write_text("\n".join(voiceworld_rows) + "\n", encoding="utf-8")
    asr_config = tmp_path / "configs/final/asr_command_compare.json"
    asr_config.parent.mkdir(parents=True)
    asr_config.write_text(
        (
            '{"input_manifest":"runs/final/asr_eval_manifest.jsonl","adapters":['
            '{"name":"cmd_a","command":["'
            + sys.executable
            + '","-c","print(1)","{input_manifest}","{output}"],'
            '"output":"runs/final/asr_commands/a.jsonl"},'
            '{"name":"cmd_b","command":["'
            + sys.executable
            + '","-c","print(1)","{input_manifest}","{output}"],'
            '"output":"runs/final/asr_commands/b.jsonl"}]}\n'
        ),
        encoding="utf-8",
    )
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
    assert report.asr_eval_manifest.records == 3
    assert report.turn_splits.turn_records == 6
    assert report.external_predictions.prepared_count == 0
    assert report.voiceworld_prepare.ok
    assert report.voiceworld_real.ok
    assert report.asr_command_config.ok
    assert report.missing_required == []
    assert (tmp_path / "runs/final/turn_train.jsonl").exists()
    assert "final_inputs_prepare: READY" in report.to_text()


def test_prepare_final_voiceworld_real_writes_and_audits_manifest(tmp_path: Path) -> None:
    voiceworld_dir = tmp_path / "data/voiceworld"
    audio_root = voiceworld_dir / "audio"
    audio_root.mkdir(parents=True)
    metadata = voiceworld_dir / "metadata.tsv"
    factor_header = (
        "pause_ms\tvad_pause_ms\tduration_ms\tsnr_db\treverb\tspeaking_rate\t"
        "overlap_offset_ms\tnetwork_jitter_ms\tfarfield_distance_m\tcode_switch_ratio\taccent"
    )
    factor_values = "900\t920\t1000\t20\tnone\t1.0\t0\t0\t0.5\t0.0\tstandard"
    scenarios = [
        ("normal_question", "complete", "take_turn", "false", "false"),
        ("incomplete_pause", "incomplete", "keep_listening", "false", "false"),
        ("backchannel", "backchannel", "continue_speaking", "true", "true"),
        ("wait_stop", "wait", "hold", "false", "false"),
        ("user_interruption", "complete", "stop_tts_and_listen", "true", "true"),
        ("side_conversation", "wait", "ignore", "false", "true"),
        ("ambient_speech", "wait", "ignore", "false", "true"),
        ("noisy_farfield", "complete", "take_turn", "false", "false"),
        ("code_switching", "complete", "take_turn", "false", "false"),
    ]
    rows = [
        "id\taudio\ttext\tscenario\tturn_label\taction_label\tassistant_speaking\toverlap\tstart\tend\t"
        + factor_header
    ]
    for index, (scenario, turn_label, action_label, assistant_speaking, overlap) in enumerate(scenarios):
        audio = f"{index}.wav"
        (audio_root / audio).write_bytes(b"")
        rows.append(
            f"real_{index}\t{audio}\ttext {index}\t{scenario}\t{turn_label}\t{action_label}\t"
            f"{assistant_speaking}\t{overlap}\t0.0\t1.0\t{factor_values}"
        )
    metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = load_final_run_config()

    report = prepare_final_voiceworld_real(config, repo_root=tmp_path)

    assert report.ok
    assert report.records == len(scenarios)
    assert not report.skipped
    assert report.audit is not None and report.audit.ok
    assert (tmp_path / "runs/final/voiceworld_real.jsonl").exists()
    assert "final_voiceworld_real_prepare: READY" in report.to_text()


def test_prepare_final_asr_transcript_conversions_writes_result_input(tmp_path: Path) -> None:
    fixture_a = Path("tests/fixtures/streaming_asr_sample.jsonl").read_text(encoding="utf-8")
    fixture_b = Path("tests/fixtures/streaming_asr_fast_unstable_sample.jsonl").read_text(encoding="utf-8")
    whisper = tmp_path / "runs/final/asr_commands/whisper_streaming.jsonl"
    funasr = tmp_path / "runs/final/asr_commands/funasr_streaming.jsonl"
    whisper.parent.mkdir(parents=True)
    whisper.write_text(fixture_a, encoding="utf-8")
    funasr.write_text(fixture_b, encoding="utf-8")
    command_config = tmp_path / "configs/final/asr_command_compare.json"
    command_config.parent.mkdir(parents=True)
    command_config.write_text(
        json.dumps(
            {
                "input_manifest": "runs/final/asr_eval_manifest.jsonl",
                "adapters": [
                    {
                        "name": "whisper_final",
                        "command": [sys.executable, "-c", "print(1)", "{input_manifest}", "{output}"],
                        "output": "runs/final/asr_commands/whisper_streaming.jsonl",
                    },
                    {
                        "name": "funasr_final",
                        "command": [sys.executable, "-c", "print(1)", "{input_manifest}", "{output}"],
                        "output": "runs/final/asr_commands/funasr_streaming.jsonl",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_final_run_config()

    report = prepare_final_asr_transcript_conversions(config, repo_root=tmp_path)

    output = tmp_path / "runs/final/reports/asr_transcript_conversions.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert report.ok
    assert report.records_by_adapter == {"whisper": 2, "funasr": 2}
    assert {row["adapter"] for row in payload["rows"]} == {"whisper", "funasr"}
    assert "final_asr_transcript_conversions: READY" in report.to_text()


def test_prepare_final_asr_transcript_conversions_reports_missing_outputs(tmp_path: Path) -> None:
    command_config = tmp_path / "configs/final/asr_command_compare.json"
    command_config.parent.mkdir(parents=True)
    command_config.write_text(
        json.dumps(
            {
                "adapters": [
                    {
                        "name": "whisper_final",
                        "command": [sys.executable, "-c", "print(1)", "{input_manifest}", "{output}"],
                        "output": "runs/final/asr_commands/whisper_streaming.jsonl",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = prepare_final_asr_transcript_conversions(load_final_run_config(), repo_root=tmp_path)

    assert not report.ok
    assert "whisper" in report.missing_inputs
    assert "final_asr_transcript_conversions: NOT_READY" in report.to_text()


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
