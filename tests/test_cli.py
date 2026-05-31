import json
import importlib.util
import sys

import pytest

from stable_asr.cli import main


def test_validate_manifest_cli_ok(capsys) -> None:
    code = main(["validate-manifest", "examples/data/turn_demo.jsonl"])

    captured = capsys.readouterr()
    assert code == 0
    assert "OK:" in captured.out
    assert "4 valid record" in captured.out


def test_doctor_cli(capsys) -> None:
    code = main(["doctor"])

    captured = capsys.readouterr()
    assert code == 0
    assert "stable_asr_doctor: OK" in captured.out
    assert "release_environment_ready:" in captured.out
    assert "config/benchmark_suite" in captured.out
    assert "config/roadmap" in captured.out
    assert "config/asr_collections" in captured.out


def test_doctor_cli_with_final_file_check(capsys) -> None:
    code = main(["doctor", "--check-final-files"])

    captured = capsys.readouterr()
    assert code == 0
    assert "final_inputs_ready: NO" in captured.out


def test_doctor_cli_with_release_env_check(capsys) -> None:
    code = main(["doctor", "--check-release-env"])
    expected = 0 if _has_import("torch") and _has_working_lance() else 1

    captured = capsys.readouterr()
    assert code == expected
    assert "release/environment" in captured.out


def test_paper_status_cli(capsys) -> None:
    code = main(["paper-status"])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Paper Status" in captured.out
    assert "final_inputs_ready" in captured.out


def test_paper_evidence_matrix_cli_writes_markdown(tmp_path, capsys) -> None:
    output = tmp_path / "FINAL_EVIDENCE_MATRIX.md"
    code = main(["paper-evidence-matrix", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Final Evidence Matrix" in captured.out
    assert "real_data_layer_benchmark" in captured.out
    assert output.exists()
    assert "data/librispeech/LibriSpeech/dev-clean" in output.read_text(encoding="utf-8")


def test_roadmap_status_cli(capsys, tmp_path) -> None:
    code = main(["roadmap-status", "--validate-only"])

    captured = capsys.readouterr()
    assert code == 0
    assert "stable_asr_roadmap_v0" in captured.out

    output = tmp_path / "ROADMAP_STATUS.md"
    code = main(["roadmap-status", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Platform Roadmap" in captured.out
    assert output.exists()
    assert "m2_data_reference_layer" in output.read_text(encoding="utf-8")


def test_labels_cli(capsys) -> None:
    code = main(["labels"])

    captured = capsys.readouterr()
    assert code == 0
    assert "complete" in captured.out
    assert "take_turn" in captured.out


def test_eval_turn_cli_json(capsys) -> None:
    code = main(["eval-turn", "--dataset", "examples/data/turn_demo.jsonl", "--json"])

    captured = capsys.readouterr()
    assert code == 0
    assert '"classification"' in captured.out
    assert '"interaction"' in captured.out


def test_eval_turn_cli_writes_report(tmp_path, capsys) -> None:
    report_path = tmp_path / "turn_eval.md"
    code = main(
        [
            "eval-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--baseline",
            "vad_pause",
            "--report",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "macro_f1:" in captured.out
    assert report_path.exists()
    assert "Stable-ASR Turn Evaluation" in report_path.read_text(encoding="utf-8")


def test_eval_turn_cli_text_baseline(capsys) -> None:
    code = main(
        [
            "eval-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--baseline",
            "text_turn",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "macro_f1:" in captured.out


def test_eval_turn_cli_external_predictions(capsys) -> None:
    code = main(
        [
            "eval-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--predictions",
            "tests/fixtures/turn_predictions_sample.jsonl",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "accuracy: 1.0000" in captured.out


def test_predict_turn_cli_writes_prediction_manifest(tmp_path, capsys) -> None:
    output = tmp_path / "text_turn_predictions.jsonl"
    code = main(
        [
            "predict-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--baseline",
            "text_turn",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "records: 4" in captured.out
    assert output.exists()

    eval_code = main(["eval-turn", "--dataset", "examples/data/turn_demo.jsonl", "--predictions", str(output)])
    eval_output = capsys.readouterr()
    assert eval_code == 0
    assert "accuracy: 1.0000" in eval_output.out


def test_validate_turn_predictions_cli(tmp_path, capsys) -> None:
    report = tmp_path / "prediction_validation.md"
    code = main(
        [
            "validate-turn-predictions",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--predictions",
            "tests/fixtures/turn_predictions_sample.jsonl",
            "--report",
            str(report),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "OK: turn prediction manifest validation" in captured.out
    assert report.exists()
    assert "Stable-ASR Turn Prediction Validation" in report.read_text(encoding="utf-8")

    bad_predictions = tmp_path / "bad_predictions.jsonl"
    bad_predictions.write_text(json.dumps({"id": "zh_turn_000001", "label": "complete"}) + "\n", encoding="utf-8")
    bad_code = main(
        [
            "validate-turn-predictions",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--predictions",
            str(bad_predictions),
        ]
    )

    bad_output = capsys.readouterr()
    assert bad_code == 1
    assert "missing_ids: 3" in bad_output.out


def test_compare_turn_cli_with_baselines_and_predictions(tmp_path, capsys) -> None:
    report = tmp_path / "turn_compare.md"
    code = main(
        [
            "compare-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--baseline",
            "vad_pause",
            "--baseline",
            "text_turn",
            "--predictions",
            "oracle=tests/fixtures/turn_predictions_sample.jsonl",
            "--report",
            str(report),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "name=oracle" in captured.out
    assert "macro_f1=1.0000" in captured.out
    assert report.exists()
    assert "Stable-ASR Turn Comparison" in report.read_text(encoding="utf-8")


def test_compare_turn_splits_cli(tmp_path, capsys) -> None:
    output_dir = tmp_path / "splits"
    code = main(
        [
            "split-turn-data",
            "--input",
            "examples/data/turn_demo.jsonl",
            "--output-dir",
            str(output_dir),
            "--train-ratio",
            "0.5",
            "--dev-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
            "--seed",
            "3",
        ]
    )
    assert code == 0
    capsys.readouterr()

    report = tmp_path / "turn_split_compare.md"
    code = main(
        [
            "compare-turn-splits",
            "--train",
            str(output_dir / "turn_train.jsonl"),
            "--dev",
            str(output_dir / "turn_dev.jsonl"),
            "--test",
            str(output_dir / "turn_test.jsonl"),
            "--baseline",
            "vad_pause",
            "--baseline",
            "text_turn",
            "--report",
            str(report),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "split=train" in captured.out
    assert "name=text_turn" in captured.out
    assert report.exists()
    assert "Stable-ASR Turn Split Comparison" in report.read_text(encoding="utf-8")


def test_benchmark_turn_cli(tmp_path, capsys) -> None:
    report_path = tmp_path / "turn_benchmark.md"
    code = main(
        [
            "benchmark-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--baseline",
            "text_turn",
            "--warmup",
            "0",
            "--repeat",
            "2",
            "--report",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "avg_latency_ms:" in captured.out
    assert "rtf:" in captured.out
    assert report_path.exists()
    assert "Stable-ASR Turn Benchmark" in report_path.read_text(encoding="utf-8")


def test_benchmark_turn_cli_external_predictions(capsys) -> None:
    code = main(
        [
            "benchmark-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--predictions",
            "tests/fixtures/turn_predictions_sample.jsonl",
            "--warmup",
            "0",
            "--repeat",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "artifact_bytes[tests/fixtures/turn_predictions_sample.jsonl]:" in captured.out


def test_split_turn_data_cli(tmp_path, capsys) -> None:
    output_dir = tmp_path / "splits"
    code = main(
        [
            "split-turn-data",
            "--input",
            "examples/data/turn_demo.jsonl",
            "--output-dir",
            str(output_dir),
            "--train-ratio",
            "0.5",
            "--dev-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
            "--seed",
            "9",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "turn_split:" in captured.out
    assert (output_dir / "turn_train.jsonl").exists()
    assert (output_dir / "turn_dev.jsonl").exists()
    assert (output_dir / "turn_test.jsonl").exists()


def test_asr_collections_cli_validate(capsys) -> None:
    code = main(["asr-collections", "--registry", "configs/references/asr_collections.json", "--validate-only"])

    captured = capsys.readouterr()
    assert code == 0
    assert "stable_asr_reference_collections_v0" in captured.out


def test_asr_collections_cli_writes_markdown(tmp_path, capsys) -> None:
    output = tmp_path / "ASR_COLLECTIONS.md"
    code = main(["asr-collections", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Reference Collections" in captured.out
    assert output.exists()
    assert "OpenAI Whisper" in output.read_text(encoding="utf-8")


def test_asr_collections_cli_writes_bibtex(tmp_path, capsys) -> None:
    output = tmp_path / "ASR_REFERENCES.bib"
    code = main(["asr-collections", "--format", "bibtex", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "@misc{stableasr_ref_funasr" in captured.out
    assert output.exists()
    assert "\\url{https://github.com/modelscope/FunASR}" in output.read_text(encoding="utf-8")


def test_asr_collections_cli_writes_paper_markdown(tmp_path, capsys) -> None:
    output = tmp_path / "ASR_REFERENCES.md"
    code = main(["asr-collections", "--format", "paper-markdown", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Paper Reference Notes" in captured.out
    assert output.exists()
    assert "stableasr_ref_funasr" in output.read_text(encoding="utf-8")


def test_asr_collections_cli_audits_p0_coverage(tmp_path, capsys) -> None:
    output = tmp_path / "ASR_COLLECTION_COVERAGE.md"
    code = main(["asr-collections", "--audit-coverage", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "ASR Collection Coverage" in captured.out
    assert "missing_required: `0`" in captured.out
    assert output.exists()


def test_audit_audio_cli_with_generated_turn_wavs(tmp_path, capsys) -> None:
    manifest = tmp_path / "turn.jsonl"
    code = main(["make-synthetic-turn-data", "--output", str(manifest), "--episodes", "2", "--write-audio"])
    assert code == 0
    capsys.readouterr()

    report = tmp_path / "audio_audit.txt"
    code = main(["audit-audio", "--kind", "turn", "--manifest", str(manifest), "--report", str(report)])

    captured = capsys.readouterr()
    assert code == 0
    assert "audio_audit: OK" in captured.out
    assert report.exists()


def test_asr_to_turn_cli(tmp_path, capsys) -> None:
    asr_manifest = tmp_path / "asr_manifest.jsonl"
    turn_manifest = tmp_path / "turn.jsonl"
    code = main(
        [
            "prepare-asr-manifest",
            "--input",
            "examples/data/asr_metadata.tsv",
            "--output",
            str(asr_manifest),
            "--audio-root",
            "examples/data",
            "--sample-rate",
            "16000",
        ]
    )
    assert code == 0
    capsys.readouterr()

    code = main(["asr-to-turn", "--input", str(asr_manifest), "--output", str(turn_manifest), "--include-incomplete"])

    captured = capsys.readouterr()
    assert code == 0
    assert "asr_to_turn:" in captured.out
    assert "output_records: 6" in captured.out
    assert turn_manifest.exists()


def test_prepare_public_asr_librispeech_cli(tmp_path, capsys) -> None:
    subset = tmp_path / "LibriSpeech" / "dev-clean"
    chapter = subset / "84" / "121123"
    chapter.mkdir(parents=True)
    (chapter / "84-121123.trans.txt").write_text(
        "84-121123-0000 WHAT IS THE WEATHER\n",
        encoding="utf-8",
    )
    (chapter / "84-121123-0000.flac").write_bytes(b"")
    output = tmp_path / "librispeech.jsonl"

    code = main(
        [
            "prepare-public-asr",
            "--corpus",
            "librispeech",
            "--input-dir",
            str(subset),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "wrote 1 librispeech ASR record" in captured.out
    assert '"dev-clean": 1' in captured.out
    assert output.exists()


def test_prepare_public_asr_common_voice_cli(tmp_path, capsys) -> None:
    root = tmp_path / "common_voice" / "en"
    clips = root / "clips"
    clips.mkdir(parents=True)
    (root / "test.tsv").write_text(
        "client_id\tpath\tsentence\tlocale\n"
        "speaker-a\tcommon_voice_en_0001.mp3\topen the door\ten\n",
        encoding="utf-8",
    )
    (clips / "common_voice_en_0001.mp3").write_bytes(b"")
    output = tmp_path / "common_voice.jsonl"

    code = main(
        [
            "prepare-public-asr",
            "--corpus",
            "common_voice",
            "--input-dir",
            str(root),
            "--split",
            "test",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "wrote 1 common_voice ASR record" in captured.out
    assert '"test": 1' in captured.out
    assert output.exists()


def test_prepare_voiceworld_cli(tmp_path, capsys) -> None:
    audio_root = tmp_path / "voiceworld" / "audio"
    audio_root.mkdir(parents=True)
    (audio_root / "interrupt.wav").write_bytes(b"")
    metadata = tmp_path / "voiceworld" / "metadata.tsv"
    output = tmp_path / "voiceworld.jsonl"
    metadata.write_text(
        (
            "id\taudio\ttext\tscenario\tturn_label\taction_label\tassistant_speaking\toverlap\t"
            "start_ms\tduration_ms\tsnr_db\treverb\tspeaking_rate\toverlap_offset_ms\t"
            "network_jitter_ms\tfarfield_distance_m\tcode_switch_ratio\taccent\n"
            "vw1\tinterrupt.wav\twait a second\tuser_interruption\tcomplete\tstop_tts_and_listen\t"
            "true\ttrue\t100\t900\t10\tnone\t1.0\t100\t30\t1.0\t0.0\tstandard\n"
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "prepare-voiceworld",
            "--input",
            str(metadata),
            "--output",
            str(output),
            "--audio-root",
            str(audio_root),
            "--language",
            "en",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "wrote 1 VoiceWorld turn record" in captured.out
    assert "user_interruption" in captured.out
    assert output.exists()


def test_prepare_public_asr_wenetspeech_cli(tmp_path, capsys) -> None:
    root = tmp_path / "WenetSpeech"
    audio_dir = root / "audio" / "dev" / "third_party" / "B00000"
    audio_dir.mkdir(parents=True)
    (audio_dir / "DEV_T0000000000.opus").write_bytes(b"")
    (root / "WenetSpeech.jsonl").write_text(
        (
            '{"utt_id":"DEV_T0000000000_S00000","audio_path":"audio/dev/third_party/B00000/DEV_T0000000000.opus",'
            '"text":"对我做了介绍啊","begin_time":0.0,"end_time":5.61,"aid":"DEV_T0000000000","subsets":["dev"]}\n'
        ),
        encoding="utf-8",
    )
    output = tmp_path / "wenetspeech.jsonl"

    code = main(
        [
            "prepare-public-asr",
            "--corpus",
            "wenetspeech",
            "--input-dir",
            str(root),
            "--split",
            "dev",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "wrote 1 wenetspeech ASR record" in captured.out
    assert '"dev": 1' in captured.out
    assert output.exists()


def test_bootstrap_turn_data_cli(tmp_path, capsys) -> None:
    output_dir = tmp_path / "bootstrap"
    code = main(
        [
            "bootstrap-turn-data",
            "--input",
            "examples/data/asr_metadata.tsv",
            "--output-dir",
            str(output_dir),
            "--audio-root",
            "examples/data",
            "--sample-rate",
            "16000",
            "--include-incomplete",
            "--train-ratio",
            "0.5",
            "--dev-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
            "--seed",
            "5",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "bootstrap_turn_data:" in captured.out
    assert (output_dir / "asr_manifest.jsonl").exists()
    assert (output_dir / "turn_manifest.jsonl").exists()
    assert (output_dir / "splits" / "turn_train.jsonl").exists()
    assert (output_dir / "BOOTSTRAP_TURN_DATA.md").exists()


def test_audit_turn_splits_cli(tmp_path, capsys) -> None:
    output_dir = tmp_path / "bootstrap"
    code = main(
        [
            "bootstrap-turn-data",
            "--input",
            "examples/data/asr_metadata.tsv",
            "--output-dir",
            str(output_dir),
            "--audio-root",
            "examples/data",
            "--sample-rate",
            "16000",
            "--include-incomplete",
        ]
    )
    assert code == 0
    capsys.readouterr()

    code = main(
        [
            "audit-turn-splits",
            "--train",
            str(output_dir / "splits" / "turn_train.jsonl"),
            "--dev",
            str(output_dir / "splits" / "turn_dev.jsonl"),
            "--test",
            str(output_dir / "splits" / "turn_test.jsonl"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "turn_split_audit: OK" in captured.out


def test_convert_predictions_cli(tmp_path, capsys) -> None:
    output = tmp_path / "easyturn_predictions.jsonl"
    code = main(
        [
            "convert-predictions",
            "--schema",
            "easyturn",
            "--input",
            "tests/fixtures/easyturn_predictions_sample.jsonl",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "converted 4 prediction record" in captured.out
    assert output.exists()

    eval_code = main(
        [
            "eval-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--predictions",
            str(output),
        ]
    )
    eval_output = capsys.readouterr()
    assert eval_code == 0
    assert "accuracy: 1.0000" in eval_output.out


def test_convert_asr_transcript_cli(tmp_path, capsys) -> None:
    output = tmp_path / "whisper_streaming.jsonl"
    code = main(
        [
            "convert-asr-transcript",
            "--schema",
            "whisper",
            "--input",
            "tests/fixtures/whisper_transcript_sample.jsonl",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "converted 2 ASR transcript record" in captured.out
    assert output.exists()


def test_make_synthetic_turn_data_cli(tmp_path, capsys) -> None:
    output = tmp_path / "synthetic.jsonl"
    code = main(
        [
            "make-synthetic-turn-data",
            "--output",
            str(output),
            "--episodes",
            "5",
            "--seed",
            "7",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "wrote 5 record" in captured.out
    assert output.exists()


def test_make_synthetic_turn_data_cli_writes_audio(tmp_path, capsys) -> None:
    output = tmp_path / "synthetic.jsonl"
    code = main(
        [
            "make-synthetic-turn-data",
            "--output",
            str(output),
            "--episodes",
            "2",
            "--write-audio",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "wrote 2 record" in captured.out
    assert len(list((tmp_path / "audio").glob("*.wav"))) == 2


def test_inspect_manifest_cli(capsys) -> None:
    code = main(["inspect-manifest", "examples/data/turn_demo.jsonl"])

    captured = capsys.readouterr()
    assert code == 0
    assert "records: 4" in captured.out
    assert "normal_question: 1" in captured.out


def test_profile_turn_data_cli(tmp_path, capsys) -> None:
    report = tmp_path / "profile.md"
    code = main(["profile-turn-data", "--dataset", "examples/data/turn_demo.jsonl", "--report", str(report)])

    captured = capsys.readouterr()
    assert code == 0
    assert "turn_data_profile:" in captured.out
    assert "records: 4" in captured.out
    assert report.exists()
    assert "Stable-ASR Turn Data Profile" in report.read_text(encoding="utf-8")


def test_convert_cli(tmp_path, capsys) -> None:
    dest = tmp_path / "converted.jsonl"
    code = main(["convert", "examples/data/turn_demo.jsonl", str(dest)])

    captured = capsys.readouterr()
    assert code == 0
    assert "converted 4 record" in captured.out
    assert dest.exists()


def test_convert_lance_cli(tmp_path, capsys) -> None:
    dest = tmp_path / "turn_demo.lance"
    code = main(["convert", "examples/data/turn_demo.jsonl", str(dest)])

    captured = capsys.readouterr()
    if code == 0:
        assert "converted 4 record" in captured.out
        assert dest.exists()
    else:
        assert code == 1
        assert "Lance support requires" in captured.err


def test_convert_external_cli(tmp_path, capsys) -> None:
    output = tmp_path / "easyturn.jsonl"
    code = main(
        [
            "convert-external",
            "--schema",
            "easyturn",
            "--input",
            "tests/fixtures/easyturn_sample.jsonl",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "converted 3 external record" in captured.out
    assert output.exists()


def test_benchmark_data_cli(tmp_path, capsys) -> None:
    pytest.importorskip("pyarrow")

    code = main(
        [
            "benchmark-data",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--output-dir",
            str(tmp_path),
            "--formats",
            "jsonl",
            "parquet",
            "--sample-count",
            "5",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "format=jsonl" in captured.out
    assert "format=parquet" in captured.out
    assert "sample_count=5" in captured.out
    assert "samples_per_second=" in captured.out


def test_data_sources_cli(tmp_path, capsys) -> None:
    output = tmp_path / "DATA_SOURCES.md"
    code = main(["data-sources", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Data Source Registry" in captured.out
    assert output.exists()
    assert "synthetic_voiceworld" in output.read_text(encoding="utf-8")


def test_data_sources_cli_validate_config(capsys) -> None:
    code = main(["data-sources", "--registry", "configs/datasets/stable_asr_sources.json", "--validate-only"])

    captured = capsys.readouterr()
    assert code == 0
    assert "OK: stable_asr_sources_v0" in captured.out


def test_paper_parity_cli_validate_config(capsys) -> None:
    code = main(["paper-parity-audit", "--checklist", "configs/paper/paper_parity_checklist.json", "--validate-only"])

    captured = capsys.readouterr()
    assert code == 0
    assert "OK: stable_asr_paper_parity_v0" in captured.out


def test_final_experiments_cli_validate_config(capsys) -> None:
    code = main(["final-experiments", "--registry", "configs/paper/final_experiments.json", "--validate-only"])

    captured = capsys.readouterr()
    assert code == 0
    assert "OK: stable_asr_final_experiments_v0" in captured.out


def test_final_experiments_cli_writes_markdown(tmp_path, capsys) -> None:
    output = tmp_path / "FINAL_EXPERIMENTS.md"
    code = main(["final-experiments", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Final-Scale Experiment Plan" in captured.out
    assert output.exists()
    assert "real_streaming_asr_systems" in output.read_text(encoding="utf-8")


def test_final_config_cli_validate_config(capsys) -> None:
    code = main(["final-config", "--config", "configs/final/paper_final.json", "--validate-only"])

    captured = capsys.readouterr()
    assert code == 0
    assert "OK: stable_asr_final_run_v0" in captured.out


def test_final_config_cli_writes_markdown(tmp_path, capsys) -> None:
    output = tmp_path / "FINAL_RUN_CONFIG.md"
    code = main(["final-config", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Final Paper Run Configuration" in captured.out
    assert output.exists()
    assert "librispeech_dev_clean" in output.read_text(encoding="utf-8")


def test_final_config_cli_check_files_reports_missing_inputs(capsys) -> None:
    code = main(["final-config", "--config", "configs/final/paper_final.json", "--check-files"])

    captured = capsys.readouterr()
    assert code == 1
    assert "NOT_READY" in captured.out
    assert "missing required input" in captured.out


def test_final_config_cli_plan_missing_outputs_action_plan(tmp_path, capsys) -> None:
    output = tmp_path / "FINAL_RUN_ACTION_PLAN.md"
    code = main(["final-config", "--config", "configs/final/paper_final.json", "--plan-missing", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Final Run Action Plan" in captured.out
    assert "stage_public_corpora" in captured.out
    assert output.exists()
    assert "collect_voiceworld_real" in output.read_text(encoding="utf-8")


def test_final_config_cli_scaffold(tmp_path, capsys) -> None:
    code = main(["final-config", "--repo-root", str(tmp_path), "--scaffold"])

    captured = capsys.readouterr()
    assert code == 0
    assert "final_run_scaffold:" in captured.out
    assert (tmp_path / "runs/final/README.md").exists()
    assert not (tmp_path / "runs/final/turn_train.jsonl").exists()


def test_final_config_cli_prepare_corpora(tmp_path, capsys) -> None:
    chapter = tmp_path / "data/librispeech/LibriSpeech/dev-clean/84/121123"
    chapter.mkdir(parents=True)
    (chapter / "84-121123.trans.txt").write_text(
        "84-121123-0000 WHAT IS THE WEATHER\n",
        encoding="utf-8",
    )
    (chapter / "84-121123-0000.flac").write_bytes(b"")
    config = tmp_path / "paper_final.json"
    config.write_text(
        (
            '{"id":"stable_asr_final_run_v0","version":"0.1.0","title":"Final",'
            '"output_dir":"runs/final","seed":0,'
            '"public_corpora":[{"id":"librispeech_dev_clean","language":"en","corpus":"librispeech",'
            '"input_dir":"data/librispeech/LibriSpeech/dev-clean",'
            '"manifest":"runs/final/librispeech_dev_clean/asr_manifest.jsonl",'
            '"sample_rate":16000,"license":"test"}],'
            '"turn_splits":{"train":"runs/final/turn_train.jsonl","dev":"runs/final/turn_dev.jsonl",'
            '"test":"runs/final/turn_test.jsonl","voiceworld_real":"runs/final/voiceworld_real.jsonl"},'
            '"external_turn_predictions":[],'
            '"asr_eval_manifest":"runs/final/asr_eval_manifest.jsonl",'
            '"asr_command_config":"configs/final/asr_command_compare.json",'
            '"nanoturn":{"model":"nanoturn_pico","checkpoint":"runs/final/nanoturn/checkpoint.pt",'
            '"metrics":"runs/final/nanoturn/metrics.json","onnx":"runs/final/nanoturn/nanoturn.onnx"},'
            '"artifacts":{"paper_results":"runs/final/paper_results.json","bundle_dir":"runs/final/artifacts",'
            '"markdown_draft":"runs/final/PAPER_DRAFT.md","latex_draft":"runs/final/paper.tex",'
            '"dataset_card":"runs/final/DATASET_CARD.md","experiment_card":"runs/final/EXPERIMENT_CARD.md"},'
            '"commands":["stable-asr final-config --validate-only"]}\n'
        ),
        encoding="utf-8",
    )

    code = main(["final-config", "--config", str(config), "--repo-root", str(tmp_path), "--prepare-corpora"])

    captured = capsys.readouterr()
    assert code == 0
    assert "final_corpora_prepare: READY" in captured.out
    assert (tmp_path / "runs/final/librispeech_dev_clean/asr_manifest.jsonl").exists()


def test_final_config_cli_bootstrap_turn_splits(tmp_path, capsys) -> None:
    manifest_dir = tmp_path / "runs/final/librispeech_dev_clean"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "asr_manifest.jsonl").write_text(
        (
            '{"id":"utt1","audio":"audio/utt1.wav","sample_rate":16000,"text":"hello","language":"en",'
            '"source":"unit","duration":1.0}\n'
            '{"id":"utt2","audio":"audio/utt2.wav","sample_rate":16000,"text":"world","language":"en",'
            '"source":"unit","duration":1.2}\n'
            '{"id":"utt3","audio":"audio/utt3.wav","sample_rate":16000,"text":"open door","language":"en",'
            '"source":"unit","duration":1.4}\n'
        ),
        encoding="utf-8",
    )
    config = tmp_path / "paper_final.json"
    config.write_text(
        (
            '{"id":"stable_asr_final_run_v0","version":"0.1.0","title":"Final",'
            '"output_dir":"runs/final","seed":0,'
            '"public_corpora":[{"id":"librispeech_dev_clean","language":"en","corpus":"librispeech",'
            '"input_dir":"data/librispeech/LibriSpeech/dev-clean",'
            '"manifest":"runs/final/librispeech_dev_clean/asr_manifest.jsonl",'
            '"sample_rate":16000,"license":"test"}],'
            '"turn_splits":{"train":"runs/final/turn_train.jsonl","dev":"runs/final/turn_dev.jsonl",'
            '"test":"runs/final/turn_test.jsonl","voiceworld_real":"runs/final/voiceworld_real.jsonl"},'
            '"external_turn_predictions":[],'
            '"asr_eval_manifest":"runs/final/asr_eval_manifest.jsonl",'
            '"asr_command_config":"configs/final/asr_command_compare.json",'
            '"nanoturn":{"model":"nanoturn_pico","checkpoint":"runs/final/nanoturn/checkpoint.pt",'
            '"metrics":"runs/final/nanoturn/metrics.json","onnx":"runs/final/nanoturn/nanoturn.onnx"},'
            '"artifacts":{"paper_results":"runs/final/paper_results.json","bundle_dir":"runs/final/artifacts",'
            '"markdown_draft":"runs/final/PAPER_DRAFT.md","latex_draft":"runs/final/paper.tex",'
            '"dataset_card":"runs/final/DATASET_CARD.md","experiment_card":"runs/final/EXPERIMENT_CARD.md"},'
            '"commands":["stable-asr final-config --validate-only"]}\n'
        ),
        encoding="utf-8",
    )

    code = main(["final-config", "--config", str(config), "--repo-root", str(tmp_path), "--bootstrap-turn-splits"])

    captured = capsys.readouterr()
    assert code == 0
    assert "final_turn_bootstrap: READY" in captured.out
    assert (tmp_path / "runs/final/turn_train.jsonl").exists()


def test_final_config_cli_prepare_external_predictions(tmp_path, capsys) -> None:
    test_path = tmp_path / "runs/final/turn_test.jsonl"
    raw_path = tmp_path / "runs/final/external/smartturn_raw.jsonl"
    test_path.parent.mkdir(parents=True)
    raw_path.parent.mkdir(parents=True)
    test_path.write_text(
        (
            '{"id":"turn1","audio":"audio/1.wav","sample_rate":16000,"start":0.0,"end":1.0,'
            '"turn_label":"complete","action_label":"take_turn","assistant_speaking":false,'
            '"overlap":false,"language":"en","source":"unit"}\n'
        ),
        encoding="utf-8",
    )
    raw_path.write_text('{"id":"turn1","complete_probability":0.9}\n', encoding="utf-8")
    config = tmp_path / "paper_final.json"
    config.write_text(
        (
            '{"id":"stable_asr_final_run_v0","version":"0.1.0","title":"Final",'
            '"output_dir":"runs/final","seed":0,'
            '"public_corpora":[{"id":"librispeech_dev_clean","language":"en","corpus":"librispeech",'
            '"input_dir":"data/librispeech/LibriSpeech/dev-clean",'
            '"manifest":"runs/final/librispeech_dev_clean/asr_manifest.jsonl",'
            '"sample_rate":16000,"license":"test"}],'
            '"turn_splits":{"train":"runs/final/turn_train.jsonl","dev":"runs/final/turn_dev.jsonl",'
            '"test":"runs/final/turn_test.jsonl","voiceworld_real":"runs/final/voiceworld_real.jsonl"},'
            '"external_turn_predictions":[{"id":"smart_turn","schema":"smart_turn",'
            '"raw":"runs/final/external/smartturn_raw.jsonl",'
            '"converted":"runs/final/external/smartturn_predictions.jsonl"}],'
            '"asr_eval_manifest":"runs/final/asr_eval_manifest.jsonl",'
            '"asr_command_config":"configs/final/asr_command_compare.json",'
            '"nanoturn":{"model":"nanoturn_pico","checkpoint":"runs/final/nanoturn/checkpoint.pt",'
            '"metrics":"runs/final/nanoturn/metrics.json","onnx":"runs/final/nanoturn/nanoturn.onnx"},'
            '"artifacts":{"paper_results":"runs/final/paper_results.json","bundle_dir":"runs/final/artifacts",'
            '"markdown_draft":"runs/final/PAPER_DRAFT.md","latex_draft":"runs/final/paper.tex",'
            '"dataset_card":"runs/final/DATASET_CARD.md","experiment_card":"runs/final/EXPERIMENT_CARD.md"},'
            '"commands":["stable-asr final-config --validate-only"]}\n'
        ),
        encoding="utf-8",
    )

    code = main(
        ["final-config", "--config", str(config), "--repo-root", str(tmp_path), "--prepare-external-predictions"]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "final_external_predictions_prepare: READY" in captured.out
    assert (tmp_path / "runs/final/external/smartturn_predictions.jsonl").exists()


def test_final_config_cli_prepare_inputs_reports_missing_required(tmp_path, capsys) -> None:
    config = tmp_path / "paper_final.json"
    config.write_text(
        (
            '{"id":"stable_asr_final_run_v0","version":"0.1.0","title":"Final",'
            '"output_dir":"runs/final","seed":0,'
            '"public_corpora":[{"id":"librispeech_dev_clean","language":"en","corpus":"librispeech",'
            '"input_dir":"data/librispeech/LibriSpeech/dev-clean",'
            '"manifest":"runs/final/librispeech_dev_clean/asr_manifest.jsonl",'
            '"sample_rate":16000,"license":"test"}],'
            '"turn_splits":{"train":"runs/final/turn_train.jsonl","dev":"runs/final/turn_dev.jsonl",'
            '"test":"runs/final/turn_test.jsonl","voiceworld_real":"runs/final/voiceworld_real.jsonl"},'
            '"external_turn_predictions":[],'
            '"asr_eval_manifest":"runs/final/asr_eval_manifest.jsonl",'
            '"asr_command_config":"configs/final/asr_command_compare.json",'
            '"nanoturn":{"model":"nanoturn_pico","checkpoint":"runs/final/nanoturn/checkpoint.pt",'
            '"metrics":"runs/final/nanoturn/metrics.json","onnx":"runs/final/nanoturn/nanoturn.onnx"},'
            '"artifacts":{"paper_results":"runs/final/paper_results.json","bundle_dir":"runs/final/artifacts",'
            '"markdown_draft":"runs/final/PAPER_DRAFT.md","latex_draft":"runs/final/paper.tex",'
            '"dataset_card":"runs/final/DATASET_CARD.md","experiment_card":"runs/final/EXPERIMENT_CARD.md"},'
            '"commands":["stable-asr final-config --validate-only"]}\n'
        ),
        encoding="utf-8",
    )

    code = main(["final-config", "--config", str(config), "--repo-root", str(tmp_path), "--prepare-inputs"])

    captured = capsys.readouterr()
    assert code == 1
    assert "final_inputs_prepare: NOT_READY" in captured.out
    assert "missing_required" in captured.out


def test_final_config_cli_audit_voiceworld_real_missing(tmp_path, capsys) -> None:
    config = tmp_path / "paper_final.json"
    config.write_text(
        (
            '{"id":"stable_asr_final_run_v0","version":"0.1.0","title":"Final",'
            '"output_dir":"runs/final","seed":0,'
            '"public_corpora":[{"id":"librispeech_dev_clean","language":"en","corpus":"librispeech",'
            '"input_dir":"data/librispeech/LibriSpeech/dev-clean",'
            '"manifest":"runs/final/librispeech_dev_clean/asr_manifest.jsonl",'
            '"sample_rate":16000,"license":"test"}],'
            '"turn_splits":{"train":"runs/final/turn_train.jsonl","dev":"runs/final/turn_dev.jsonl",'
            '"test":"runs/final/turn_test.jsonl","voiceworld_real":"runs/final/voiceworld_real.jsonl"},'
            '"external_turn_predictions":[],'
            '"asr_eval_manifest":"runs/final/asr_eval_manifest.jsonl",'
            '"asr_command_config":"configs/final/asr_command_compare.json",'
            '"nanoturn":{"model":"nanoturn_pico","checkpoint":"runs/final/nanoturn/checkpoint.pt",'
            '"metrics":"runs/final/nanoturn/metrics.json","onnx":"runs/final/nanoturn/nanoturn.onnx"},'
            '"artifacts":{"paper_results":"runs/final/paper_results.json","bundle_dir":"runs/final/artifacts",'
            '"markdown_draft":"runs/final/PAPER_DRAFT.md","latex_draft":"runs/final/paper.tex",'
            '"dataset_card":"runs/final/DATASET_CARD.md","experiment_card":"runs/final/EXPERIMENT_CARD.md"},'
            '"commands":["stable-asr final-config --validate-only"]}\n'
        ),
        encoding="utf-8",
    )

    code = main(["final-config", "--config", str(config), "--repo-root", str(tmp_path), "--audit-voiceworld-real"])

    captured = capsys.readouterr()
    assert code == 1
    assert "final_voiceworld_real_audit: NOT_READY" in captured.out
    assert "voiceworld_real manifest is missing" in captured.out


def test_final_config_cli_prepare_voiceworld_real_reports_missing_inputs(tmp_path, capsys) -> None:
    code = main(["final-config", "--repo-root", str(tmp_path), "--prepare-voiceworld-real"])

    captured = capsys.readouterr()
    assert code == 1
    assert "final_voiceworld_real_prepare: SKIPPED" in captured.out
    assert "missing metadata and audio_root" in captured.out


def test_adapter_registry_cli(tmp_path, capsys) -> None:
    output = tmp_path / "ADAPTERS.md"
    code = main(["adapter-registry", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Adapter Registry" in captured.out
    assert output.exists()
    assert "command_streaming_asr" in output.read_text(encoding="utf-8")


def test_adapter_registry_cli_validate_config(capsys) -> None:
    code = main(["adapter-registry", "--registry", "configs/adapters/stable_asr_adapters.json", "--validate-only"])

    captured = capsys.readouterr()
    assert code == 0
    assert "OK: stable_asr_adapters_v0" in captured.out


def test_prepare_validate_and_inspect_asr_manifest_cli(tmp_path, capsys) -> None:
    output = tmp_path / "asr_manifest.jsonl"
    code = main(
        [
            "prepare-asr-manifest",
            "--input",
            "examples/data/asr_metadata.tsv",
            "--output",
            str(output),
            "--audio-root",
            "examples/data",
            "--sample-rate",
            "16000",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "wrote 3 ASR record" in captured.out
    assert output.exists()

    validate_code = main(["validate-asr-manifest", str(output)])
    validate_output = capsys.readouterr()
    assert validate_code == 0
    assert "valid ASR record" in validate_output.out

    inspect_code = main(["inspect-asr-manifest", str(output)])
    inspect_output = capsys.readouterr()
    assert inspect_code == 0
    assert "records: 3" in inspect_output.out
    assert "librispeech: 2" in inspect_output.out


def test_eval_streaming_asr_cli(capsys) -> None:
    code = main(["eval-streaming-asr", "--input", "tests/fixtures/streaming_asr_sample.jsonl"])

    captured = capsys.readouterr()
    assert code == 0
    assert "wer:" in captured.out
    assert "partial_revision_rate:" in captured.out
    assert "streaming_failures:" in captured.out


def test_compare_streaming_asr_cli(tmp_path, capsys) -> None:
    report = tmp_path / "streaming_compare.md"
    code = main(
        [
            "compare-streaming-asr",
            "--input",
            "balanced=tests/fixtures/streaming_asr_sample.jsonl",
            "--input",
            "fast_unstable=tests/fixtures/streaming_asr_fast_unstable_sample.jsonl",
            "--report",
            str(report),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "adapter=balanced" in captured.out
    assert "adapter=fast_unstable" in captured.out
    assert report.exists()
    assert "Stable-ASR Streaming ASR Adapter Comparison" in report.read_text(encoding="utf-8")


def test_compare_streaming_asr_cli_rejects_bad_input(capsys) -> None:
    code = main(["compare-streaming-asr", "--input", "missing_separator"])

    captured = capsys.readouterr()
    assert code == 1
    assert "ADAPTER=PATH" in captured.err


def test_sweep_streaming_asr_cli(tmp_path, capsys) -> None:
    report = tmp_path / "streaming_sweep.md"
    code = main(
        [
            "sweep-streaming-asr",
            "--input",
            "tests/fixtures/streaming_asr_sample.jsonl",
            "--chunks-ms",
            "160",
            "320",
            "--lookahead-ms",
            "0",
            "160",
            "--report",
            str(report),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "chunk_ms=160" in captured.out
    assert "lookahead_ms=160" in captured.out
    assert report.exists()
    assert "Stable-ASR Streaming ASR Schedule Sweep" in report.read_text(encoding="utf-8")


def test_eval_scenario_cli(tmp_path, capsys) -> None:
    report_path = tmp_path / "scenario.md"
    code = main(
        [
            "eval-scenario",
            "--episodes",
            "10",
            "--seed",
            "0",
            "--baseline",
            "vad_pause",
            "--report",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "scenarios:" in captured.out
    assert report_path.exists()
    assert "Scenario Breakdown" in report_path.read_text(encoding="utf-8")


def test_eval_scenario_cli_accepts_manifest_dataset(tmp_path, capsys) -> None:
    dataset = tmp_path / "voiceworld.jsonl"
    dataset.write_text(
        (
            '{"id":"real1","audio":"audio/1.wav","sample_rate":16000,"start":0.0,"end":1.0,'
            '"turn_label":"complete","action_label":"take_turn","assistant_speaking":false,'
            '"overlap":false,"language":"en","source":"unit","scenario":"normal_question",'
            '"metadata":{"snr_db":20,"reverb":"none","speaking_rate":1.0,'
            '"overlap_offset_ms":0,"network_jitter_ms":0,"farfield_distance_m":0.5,'
            '"code_switch_ratio":0.0,"accent":"standard"}}\n'
            '{"id":"real2","audio":"audio/2.wav","sample_rate":16000,"start":0.0,"end":1.0,'
            '"turn_label":"wait","action_label":"ignore","assistant_speaking":false,'
            '"overlap":true,"language":"en","source":"unit","scenario":"ambient_speech",'
            '"metadata":{"snr_db":5,"reverb":"room","speaking_rate":1.0,'
            '"overlap_offset_ms":100,"network_jitter_ms":20,"farfield_distance_m":2.0,'
            '"code_switch_ratio":0.0,"accent":"standard"}}\n'
        ),
        encoding="utf-8",
    )
    output = tmp_path / "scenarios.json"

    code = main(
        [
            "eval-scenario",
            "--dataset",
            str(dataset),
            "--baseline",
            "vad_pause",
            "--json-output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "ambient_speech" in captured.out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert sorted(payload["by_scenario"]) == ["ambient_speech", "normal_question"]
    assert "snr_db" in payload["factor_summary"]


def test_optimize_policy_cli(tmp_path, capsys) -> None:
    output = tmp_path / "policy_search.json"
    code = main(
        [
            "optimize-policy",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--baseline",
            "vad_pause",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "best_score:" in captured.out
    assert output.exists()


def test_train_turn_cli_and_eval_checkpoint(tmp_path, capsys) -> None:
    pytest.importorskip("torch")

    output_dir = tmp_path / "nanoturn"
    code = main(
        [
            "train-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--output-dir",
            str(output_dir),
            "--epochs",
            "5",
            "--seed",
            "0",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "checkpoint:" in captured.out

    checkpoint = output_dir / "checkpoint.pt"
    code = main(
        [
            "eval-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--checkpoint",
            str(checkpoint),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "macro_f1:" in captured.out


def test_train_turn_cli_audio_features(tmp_path, capsys) -> None:
    pytest.importorskip("torch")

    manifest = tmp_path / "synthetic.jsonl"
    main(
        [
            "make-synthetic-turn-data",
            "--output",
            str(manifest),
            "--episodes",
            "5",
            "--seed",
            "4",
            "--write-audio",
        ]
    )
    output_dir = tmp_path / "nanoturn_audio"
    code = main(
        [
            "train-turn",
            "--dataset",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--feature-source",
            "audio",
            "--epochs",
            "3",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "checkpoint:" in captured.out


def test_export_turn_onnx_cli(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("onnx")

    output_dir = tmp_path / "nanoturn"
    main(
        [
            "train-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--output-dir",
            str(output_dir),
            "--epochs",
            "3",
        ]
    )
    onnx_path = tmp_path / "nanoturn.onnx"
    code = main(
        [
            "export-turn-onnx",
            "--checkpoint",
            str(output_dir / "checkpoint.pt"),
            "--output",
            str(onnx_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "onnx:" in captured.out
    assert onnx_path.exists()


def test_reproduce_paper_cli(tmp_path, capsys) -> None:
    code = main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "8",
            "--seed",
            "3",
            "--skip-train",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "results:" in captured.out
    assert "report:" in captured.out


def test_reproduce_paper_cli_config(tmp_path, capsys) -> None:
    config = tmp_path / "paper.json"
    config.write_text(
        (
            "{"
            f"\"output_dir\":\"{tmp_path / 'paper_config'}\","
            "\"episodes\":6,"
            "\"seed\":9,"
            "\"train_model\":false"
            "}"
        ),
        encoding="utf-8",
    )
    code = main(["reproduce-paper", "--config", str(config)])

    captured = capsys.readouterr()
    assert code == 0
    assert "results:" in captured.out
    assert (tmp_path / "paper_config" / "paper_results.json").exists()


def test_paper_table_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "8",
            "--skip-train",
        ]
    )
    output = tmp_path / "baseline_table.md"
    code = main(
        [
            "paper-table",
            "baselines",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "rule_endpoint" in captured.out
    assert output.exists()


def test_leaderboard_export_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "9",
            "--skip-train",
        ]
    )
    output = tmp_path / "leaderboard.jsonl"
    code = main(
        [
            "leaderboard-export",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output",
            str(output),
            "--format",
            "jsonl",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "leaderboard:" in captured.out
    assert output.exists()
    assert "turn_quality" in output.read_text(encoding="utf-8")


def test_leaderboard_validate_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "9",
            "--skip-train",
        ]
    )
    output = tmp_path / "leaderboard.jsonl"
    main(
        [
            "leaderboard-export",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output",
            str(output),
        ]
    )
    report = tmp_path / "LEADERBOARD_VALIDATION.md"
    code = main(["leaderboard-validate", "--input", str(output), "--output", str(report)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Leaderboard Validation" in captured.out
    assert "status: `OK`" in captured.out
    assert report.exists()


def test_leaderboard_validate_cli_rejects_bad_submission(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        (
            '{"suite":"bad","task":"turn_quality","system":"x","slice":"overall",'
            '"metric":"macro_f1","value":"nan","unit":"rate","higher_is_better":true,"source":"unit"}\n'
        ),
        encoding="utf-8",
    )

    code = main(["leaderboard-validate", "--input", str(bad)])

    captured = capsys.readouterr()
    assert code == 1
    assert "FAILED" in captured.out
    assert "expected stable_asr_v0" in captured.out


def test_paper_case_studies_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "8",
            "--skip-train",
        ]
    )
    output_dir = tmp_path / "case_studies"
    code = main(
        [
            "paper-case-studies",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "case_studies_markdown:" in captured.out
    assert (output_dir / "CASE_STUDIES.md").exists()


def test_paper_claim_audit_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "8",
            "--skip-train",
        ]
    )
    artifact_dir = tmp_path / "artifacts"
    main(
        [
            "paper-bundle",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output-dir",
            str(artifact_dir),
        ]
    )
    output_dir = tmp_path / "claims"
    code = main(
        [
            "paper-claim-audit",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--artifacts-dir",
            str(artifact_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "claim_audit: OK" in captured.out
    assert "claims_markdown:" in captured.out
    assert (output_dir / "CLAIMS.md").exists()


def test_benchmark_suite_cli(tmp_path, capsys) -> None:
    output = tmp_path / "BENCHMARK_SUITE.md"
    code = main(["benchmark-suite", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR v0 Paper Benchmark Suite" in captured.out
    assert output.exists()
    assert "asr_transcript_conversion" in output.read_text(encoding="utf-8")


def test_benchmark_suite_cli_validate_config(capsys) -> None:
    code = main(
        [
            "benchmark-suite",
            "--suite",
            "configs/benchmarks/stable_asr_v0.json",
            "--validate-only",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "OK: stable_asr_v0" in captured.out


def test_scenario_suite_cli(tmp_path, capsys) -> None:
    output = tmp_path / "SCENARIO_SUITE.md"
    code = main(["scenario-suite", "--output", str(output)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR VoiceWorld v0 Scenario Suite" in captured.out
    assert output.exists()
    assert "user_interruption" in output.read_text(encoding="utf-8")


def test_scenario_suite_cli_validate_config(capsys) -> None:
    code = main(
        [
            "scenario-suite",
            "--suite",
            "configs/scenarios/stable_asr_voiceworld_v0.json",
            "--validate-only",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "OK: stable_asr_voiceworld_v0" in captured.out


def test_eval_asr_command_cli(tmp_path, capsys) -> None:
    script = tmp_path / "copy_transcript.py"
    output = tmp_path / "command_output.jsonl"
    script.write_text(
        "\n".join(
            [
                "import shutil",
                "import sys",
                "shutil.copyfile(sys.argv[1], sys.argv[2])",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    code = main(
        [
            "eval-asr-command",
            "--name",
            "copy_fixture",
            "--command",
            f"{sys.executable} {script} tests/fixtures/streaming_asr_sample.jsonl {{output}}",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "adapter: copy_fixture" in captured.out
    assert "records: 2" in captured.out
    assert output.exists()


def test_compare_asr_commands_cli(tmp_path, capsys) -> None:
    script = tmp_path / "copy_transcript.py"
    script.write_text(
        "\n".join(
            [
                "import shutil",
                "import sys",
                "shutil.copyfile(sys.argv[1], sys.argv[2])",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "adapters": [
            {
                "name": "balanced_cmd",
                "command": [
                    sys.executable,
                    str(script),
                    "tests/fixtures/streaming_asr_sample.jsonl",
                    "{output}",
                ],
                "output": str(tmp_path / "balanced.jsonl"),
            },
            {
                "name": "fast_cmd",
                "command": [
                    sys.executable,
                    str(script),
                    "tests/fixtures/streaming_asr_fast_unstable_sample.jsonl",
                    "{output}",
                ],
                "output": str(tmp_path / "fast.jsonl"),
            },
        ]
    }
    config_path = tmp_path / "commands.json"
    report_path = tmp_path / "commands.md"
    json_path = tmp_path / "commands.json.out"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    code = main(
        [
            "compare-asr-commands",
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--json-output",
            str(json_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "adapter=balanced_cmd" in captured.out
    assert "adapter=fast_cmd" in captured.out
    assert report_path.exists()
    assert len(json.loads(json_path.read_text(encoding="utf-8"))["rows"]) == 2


def test_compare_asr_commands_validate_only_cli(tmp_path, capsys) -> None:
    manifest = tmp_path / "runs/final/asr_eval_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"id":"utt1","audio":"audio/utt1.wav","sample_rate":16000,"text":"hello","language":"en","source":"unit"}\n',
        encoding="utf-8",
    )
    config = {
        "input_manifest": str(manifest),
        "adapters": [
            {
                "name": "cmd_a",
                "command": [sys.executable, "-c", "print(1)", "{input_manifest}", "{output}"],
                "output": str(tmp_path / "a.jsonl"),
            },
            {
                "name": "cmd_b",
                "command": [sys.executable, "-c", "print(1)", "{input_manifest}", "{output}"],
                "output": str(tmp_path / "b.jsonl"),
            },
        ],
    }
    config_path = tmp_path / "commands.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    code = main(
        [
            "compare-asr-commands",
            "--config",
            str(config_path),
            "--validate-only",
            "--repo-root",
            str(tmp_path),
            "--min-adapters",
            "2",
            "--require-input-manifest",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "asr_command_config_audit: READY" in captured.out


def test_core_eval_commands_write_json_output(tmp_path, capsys) -> None:
    eval_output = tmp_path / "eval_turn.json"
    code = main(
        [
            "eval-turn",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--json-output",
            str(eval_output),
        ]
    )
    assert code == 0
    assert json.loads(eval_output.read_text(encoding="utf-8"))["classification"]["accuracy"] >= 0.0

    data_output = tmp_path / "data_benchmark.json"
    code = main(
        [
            "benchmark-data",
            "--dataset",
            "examples/data/turn_demo.jsonl",
            "--output-dir",
            str(tmp_path / "data_bench"),
            "--formats",
            "jsonl",
            "--json-output",
            str(data_output),
        ]
    )
    assert code == 0
    assert json.loads(data_output.read_text(encoding="utf-8"))[0]["format"] == "jsonl"

    streaming_output = tmp_path / "streaming_compare.json"
    code = main(
        [
            "compare-streaming-asr",
            "--input",
            "balanced=tests/fixtures/streaming_asr_sample.jsonl",
            "--input",
            "fast=tests/fixtures/streaming_asr_fast_unstable_sample.jsonl",
            "--json-output",
            str(streaming_output),
        ]
    )
    assert code == 0
    assert len(json.loads(streaming_output.read_text(encoding="utf-8"))["rows"]) == 2
    capsys.readouterr()


def test_final_config_cli_prepare_asr_eval_manifest(tmp_path, capsys) -> None:
    manifest = tmp_path / "runs/final/librispeech_dev_clean/asr_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"id":"utt1","audio":"audio/utt1.wav","sample_rate":16000,"text":"hello","language":"en","source":"unit"}\n',
        encoding="utf-8",
    )
    config = {
        "id": "stable_asr_final_run_v0",
        "version": "0.1.0",
        "title": "Final",
        "output_dir": "runs/final",
        "seed": 0,
        "public_corpora": [
            {
                "id": "librispeech_dev_clean",
                "language": "en",
                "corpus": "librispeech",
                "input_dir": "data/librispeech/LibriSpeech/dev-clean",
                "manifest": "runs/final/librispeech_dev_clean/asr_manifest.jsonl",
                "sample_rate": 16000,
                "license": "test",
            }
        ],
        "asr_eval_manifest": "runs/final/asr_eval_manifest.jsonl",
        "turn_splits": {
            "train": "runs/final/turn_train.jsonl",
            "dev": "runs/final/turn_dev.jsonl",
            "test": "runs/final/turn_test.jsonl",
            "voiceworld_real": "runs/final/voiceworld_real.jsonl",
        },
        "external_turn_predictions": [],
        "asr_command_config": "configs/final/asr_command_compare.json",
        "nanoturn": {
            "model": "nanoturn_pico",
            "checkpoint": "runs/final/nanoturn/checkpoint.pt",
            "metrics": "runs/final/nanoturn/metrics.json",
            "onnx": "runs/final/nanoturn/nanoturn.onnx",
        },
        "artifacts": {
            "paper_results": "runs/final/paper_results.json",
            "bundle_dir": "runs/final/artifacts",
            "markdown_draft": "runs/final/PAPER_DRAFT.md",
            "latex_draft": "runs/final/paper.tex",
            "dataset_card": "runs/final/DATASET_CARD.md",
            "experiment_card": "runs/final/EXPERIMENT_CARD.md",
        },
        "commands": ["stable-asr final-config --validate-only"],
    }
    config_path = tmp_path / "paper_final.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    code = main(["final-config", "--config", str(config_path), "--repo-root", str(tmp_path), "--prepare-asr-eval-manifest"])

    captured = capsys.readouterr()
    assert code == 0
    assert "final_asr_eval_manifest: READY" in captured.out
    assert (tmp_path / "runs/final/asr_eval_manifest.jsonl").exists()

    command_config = tmp_path / "configs/final/asr_command_compare.json"
    command_config.parent.mkdir(parents=True)
    command_config.write_text(
        json.dumps(
            {
                "input_manifest": "runs/final/asr_eval_manifest.jsonl",
                "adapters": [
                    {
                        "name": "cmd_a",
                        "command": [sys.executable, "-c", "print(1)", "{input_manifest}", "{output}"],
                        "output": "runs/final/asr_commands/a.jsonl",
                    },
                    {
                        "name": "cmd_b",
                        "command": [sys.executable, "-c", "print(1)", "{input_manifest}", "{output}"],
                        "output": "runs/final/asr_commands/b.jsonl",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    code = main(["final-config", "--config", str(config_path), "--repo-root", str(tmp_path), "--audit-asr-commands"])

    captured = capsys.readouterr()
    assert code == 0
    assert "asr_command_config_audit: READY" in captured.out

    fixture_a = open("tests/fixtures/streaming_asr_sample.jsonl", encoding="utf-8").read()
    fixture_b = open("tests/fixtures/streaming_asr_fast_unstable_sample.jsonl", encoding="utf-8").read()
    output_a = tmp_path / "runs/final/asr_commands/a.jsonl"
    output_b = tmp_path / "runs/final/asr_commands/b.jsonl"
    output_a.parent.mkdir(parents=True, exist_ok=True)
    output_a.write_text(fixture_a, encoding="utf-8")
    output_b.write_text(fixture_b, encoding="utf-8")

    code = main(
        [
            "final-config",
            "--config",
            str(config_path),
            "--repo-root",
            str(tmp_path),
            "--prepare-asr-transcript-conversions",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "final_asr_transcript_conversions: READY" in captured.out
    assert (tmp_path / "runs/final/reports/asr_transcript_conversions.json").exists()


def test_benchmark_suite_cli_validates_result_coverage(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "9",
            "--skip-train",
        ]
    )
    suite = {
        "id": "stable_asr_v0",
        "version": "test",
        "title": "Test Suite",
        "leaderboard_suite": "stable_asr_v0",
        "tasks": [
            {
                "id": "turn_quality",
                "title": "Turn Quality",
                "coverage": "system_slice_metric",
                "systems": ["rule_endpoint", "vad_pause", "text_turn", "prediction_manifest"],
                "slices": ["overall"],
                "metrics": [
                    {"name": "accuracy", "unit": "rate", "higher_is_better": True},
                    {"name": "macro_f1", "unit": "rate", "higher_is_better": True},
                ],
            }
        ],
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    code = main(
        [
            "benchmark-suite",
            "--suite",
            str(suite_path),
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--validate-only",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "coverage=OK" in captured.out


def test_paper_table_turn_benchmark_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "8",
            "--skip-train",
        ]
    )
    code = main(
        [
            "paper-table",
            "turn_benchmark",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "avg_latency_ms" in captured.out
    assert "prediction_manifest" in captured.out


def test_paper_table_streaming_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "8",
            "--skip-train",
        ]
    )
    code = main(
        [
            "paper-table",
            "streaming",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "partial_revision_rate" in captured.out


def test_paper_table_asr_transcript_conversions_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "8",
            "--skip-train",
        ]
    )
    code = main(
        [
            "paper-table",
            "asr_transcript_conversions",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "whisper" in captured.out
    assert "funasr" in captured.out


def test_paper_table_scenarios_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "10",
            "--skip-train",
        ]
    )
    code = main(
        [
            "paper-table",
            "scenarios",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "user_interruption" in captured.out


def test_paper_table_policy_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "10",
            "--skip-train",
        ]
    )
    code = main(
        [
            "paper-table",
            "policy",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "complete_threshold" in captured.out


def test_paper_figure_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "10",
            "--skip-train",
        ]
    )
    output = tmp_path / "baseline.svg"
    code = main(
        [
            "paper-figure",
            "baselines",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "figure:" in captured.out
    assert output.exists()
    assert "Baseline Macro F1" in output.read_text(encoding="utf-8")


def test_paper_figure_latency_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "10",
            "--skip-train",
        ]
    )
    output = tmp_path / "latency.svg"
    code = main(
        [
            "paper-figure",
            "latency",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "figure:" in captured.out
    assert output.exists()
    assert "Turn Predictor Latency" in output.read_text(encoding="utf-8")


def test_paper_bundle_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "10",
            "--skip-train",
        ]
    )
    output_dir = tmp_path / "artifacts"
    code = main(
        [
            "paper-bundle",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "artifact_index:" in captured.out
    assert "results:" in captured.out
    assert "benchmark_suite:" in captured.out
    assert "provenance:" in captured.out
    assert "data_sources:" in captured.out
    assert "leaderboard_validation:" in captured.out
    assert "asr_collections:" in captured.out
    assert "scenario_suite:" in captured.out
    assert "case_studies:" in captured.out
    assert "final_evidence_matrix:" in captured.out
    assert "roadmap_status:" in captured.out
    assert "claims:" in captured.out
    assert "artifact_integrity:" in captured.out
    assert (output_dir / "ARTIFACT_INDEX.md").exists()
    assert (output_dir / "paper_results.json").exists()
    assert (output_dir / "ARTIFACT_HASHES.md").exists()
    assert (output_dir / "artifact_hashes.json").exists()
    assert (output_dir / "PROVENANCE.md").exists()
    assert (output_dir / "provenance.json").exists()
    assert (output_dir / "tables" / "baselines.md").exists()
    assert (output_dir / "figures" / "baselines.svg").exists()
    assert (output_dir / "LEADERBOARD_VALIDATION.md").exists()
    assert (output_dir / "BENCHMARK_SUITE.md").exists()
    assert (output_dir / "SCENARIO_SUITE.md").exists()
    assert (output_dir / "CASE_STUDIES.md").exists()
    assert (output_dir / "FINAL_EVIDENCE_MATRIX.md").exists()
    assert (output_dir / "CLAIMS.md").exists()


def test_paper_artifact_integrity_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "10",
            "--skip-train",
        ]
    )
    output_dir = tmp_path / "artifacts"
    main(
        [
            "paper-bundle",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output-dir",
            str(output_dir),
        ]
    )
    code = main(
        [
            "paper-artifact-integrity",
            "--manifest",
            str(output_dir / "artifact_hashes.json"),
            "--root",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "Stable-ASR Artifact Integrity" in captured.out
    assert "status: `OK`" in captured.out

    (output_dir / "tables" / "baselines.md").write_text("tampered\n", encoding="utf-8")
    code = main(
        [
            "paper-artifact-integrity",
            "--manifest",
            str(output_dir / "artifact_hashes.json"),
            "--root",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "Mismatched" in captured.out


def test_paper_audit_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "10",
            "--skip-train",
        ]
    )
    output_dir = tmp_path / "artifacts"
    main(
        [
            "paper-bundle",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output-dir",
            str(output_dir),
        ]
    )
    code = main(
        [
            "paper-audit",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--artifacts-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "paper_audit: OK" in captured.out


def test_paper_draft_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "10",
            "--skip-train",
        ]
    )
    main(
        [
            "paper-bundle",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    output = tmp_path / "PAPER_DRAFT.md"
    code = main(
        [
            "paper-draft",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "draft:" in captured.out
    assert output.exists()
    assert "Stable-ASR" in output.read_text(encoding="utf-8")


def test_paper_latex_cli(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "10",
            "--skip-train",
        ]
    )
    main(
        [
            "paper-bundle",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    output = tmp_path / "paper.tex"
    code = main(
        [
            "paper-latex",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "latex:" in captured.out
    assert output.exists()
    assert "\\documentclass" in output.read_text(encoding="utf-8")


def test_paper_release_audit_cli_reports_not_ready(tmp_path, capsys) -> None:
    main(
        [
            "reproduce-paper",
            "--output-dir",
            str(tmp_path / "paper"),
            "--episodes",
            "10",
            "--skip-train",
        ]
    )
    main(
        [
            "paper-bundle",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    main(
        [
            "paper-draft",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output",
            str(tmp_path / "PAPER_DRAFT.md"),
        ]
    )
    main(
        [
            "paper-latex",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output",
            str(tmp_path / "paper.tex"),
        ]
    )
    code = main(
        [
            "paper-release-audit",
            "--repo-root",
            ".",
            "--results",
            str(tmp_path / "paper" / "paper_results.json"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--markdown-draft",
            str(tmp_path / "PAPER_DRAFT.md"),
            "--latex-draft",
            str(tmp_path / "paper.tex"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "paper_release_audit: NOT_READY" in captured.out
    assert "baseline/nanoturn_release_baseline" in captured.out


def test_paper_release_smoke_cli(tmp_path, capsys) -> None:
    code = main(
        [
            "paper-release-smoke",
            "--output-dir",
            str(tmp_path / "release_smoke"),
            "--episodes",
            "9",
            "--seed",
            "6",
            "--skip-train",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "paper_release_smoke: NOT_READY" in captured.out
    assert "release_audit_json:" in captured.out
    assert (tmp_path / "release_smoke" / "release_audit.json").exists()
    assert (tmp_path / "release_smoke" / "RELEASE_AUDIT.md").exists()


def test_paper_release_smoke_default_trains_nanoturn_when_torch_available(tmp_path, capsys) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = main(
        [
            "paper-release-smoke",
            "--output-dir",
            str(tmp_path / "release_smoke_train"),
            "--episodes",
            "9",
            "--seed",
            "6",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    expected_status = "READY" if _has_working_lance() else "NOT_READY"
    assert f"paper_release_smoke: {expected_status}" in captured.out
    audit = json.loads((tmp_path / "release_smoke_train" / "release_audit.json").read_text(encoding="utf-8"))
    failed = {f"{check['gate']}/{check['name']}" for check in audit["checks"] if not check["ok"]}
    assert "baseline/nanoturn_release_baseline" not in failed
    if _has_working_lance():
        assert "data/lance_data_layer" not in failed
    else:
        assert "data/lance_data_layer" in failed


def test_make_card_cli(tmp_path, capsys) -> None:
    output = tmp_path / "DATASET_CARD.md"
    code = main(
        [
            "make-card",
            "dataset",
            "--input",
            "examples/data/turn_demo.jsonl",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "card:" in captured.out
    assert output.exists()


def _has_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_working_lance() -> bool:
    if importlib.util.find_spec("lance") is None:
        return False
    try:
        import lance
    except Exception:
        return False
    return hasattr(lance, "dataset") and hasattr(lance, "write_dataset")
