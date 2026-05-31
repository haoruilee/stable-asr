from pathlib import Path

import pytest

from stable_asr.data.asr_manifest import (
    ASRManifestError,
    ASRManifestRecord,
    load_asr_manifest,
    summarize_asr_records,
    validate_asr_manifest,
)
from stable_asr.data.recipes import prepare_asr_manifest
from stable_asr.data.recipes.public_corpora import prepare_public_asr_manifest


def test_asr_manifest_record_validation() -> None:
    record = ASRManifestRecord.from_dict(
        {
            "id": "utt_001",
            "audio": "audio/utt_001.wav",
            "sample_rate": "16000",
            "text": "what is the weather",
            "language": "en",
            "source": "unit",
            "duration": "2.1",
        }
    )

    assert record.sample_rate == 16000
    assert record.duration == pytest.approx(2.1)


def test_asr_manifest_rejects_bad_duration() -> None:
    with pytest.raises(ASRManifestError, match="duration must be positive"):
        ASRManifestRecord.from_dict(
            {
                "id": "bad",
                "audio": "audio/bad.wav",
                "sample_rate": 16000,
                "text": "bad",
                "language": "en",
                "source": "unit",
                "duration": 0,
            }
        )


def test_prepare_asr_manifest_from_tsv(tmp_path: Path) -> None:
    output = tmp_path / "asr_manifest.jsonl"
    records = prepare_asr_manifest(
        "examples/data/asr_metadata.tsv",
        output,
        audio_root="examples/data",
        default_sample_rate=16000,
    )

    assert len(records) == 3
    assert records[0].id == "asr_demo_0001"
    assert records[0].audio == "examples/data/audio/asr_demo_0001.wav"
    assert records[0].metadata["domain"] == "assistant_query"
    assert output.exists()

    report = validate_asr_manifest(output)
    loaded = load_asr_manifest(output)
    summary = summarize_asr_records(loaded)

    assert report.ok
    assert summary["records"] == 3
    assert summary["languages"] == {"en": 2, "zh": 1}
    assert summary["sources"] == {"aishell1": 1, "librispeech": 2}


def test_prepare_asr_manifest_from_jsonl_aliases(tmp_path: Path) -> None:
    input_path = tmp_path / "metadata.jsonl"
    input_path.write_text(
        (
            '{"key":"utt_a","wav":"a.wav","sentence":"hello","sr":8000,"lang":"en"}\n'
            '{"key":"utt_b","wav":"b.wav","sentence":"你好","sr":16000,"lang":"zh"}\n'
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "manifest.jsonl"

    records = prepare_asr_manifest(input_path, output_path, default_source="fixture")

    assert [record.id for record in records] == ["utt_a", "utt_b"]
    assert records[0].sample_rate == 8000
    assert records[1].source == "fixture"


def test_prepare_librispeech_public_recipe(tmp_path: Path) -> None:
    subset = tmp_path / "LibriSpeech" / "dev-clean"
    chapter = subset / "84" / "121123"
    chapter.mkdir(parents=True)
    (chapter / "84-121123.trans.txt").write_text(
        "84-121123-0000 WHAT IS THE WEATHER\n"
        "84-121123-0001 TURN ON THE LIGHTS\n",
        encoding="utf-8",
    )
    (chapter / "84-121123-0000.flac").write_bytes(b"")
    (chapter / "84-121123-0001.flac").write_bytes(b"")
    output = tmp_path / "librispeech.jsonl"

    records = prepare_public_asr_manifest(
        corpus="librispeech",
        input_dir=subset,
        output_path=output,
        sample_rate=16000,
    )

    assert len(records) == 2
    assert records[0].id == "84-121123-0000"
    assert records[0].audio.endswith("84-121123-0000.flac")
    assert records[0].language == "en"
    assert records[0].source == "librispeech"
    assert records[0].split == "dev-clean"
    assert records[0].speaker_id == "84"
    assert records[0].metadata["chapter_id"] == "121123"
    assert validate_asr_manifest(output).ok


def test_prepare_aishell_public_recipe(tmp_path: Path) -> None:
    root = tmp_path / "data_aishell"
    transcript_dir = root / "transcript"
    wav_dir = root / "wav" / "dev" / "S0724"
    transcript_dir.mkdir(parents=True)
    wav_dir.mkdir(parents=True)
    (transcript_dir / "aishell_transcript_v0.8.txt").write_text(
        "BAC009S0724W0121 今天天气不错\n"
        "BAC009S0724W0122 打开客厅灯\n",
        encoding="utf-8",
    )
    (wav_dir / "BAC009S0724W0121.wav").write_bytes(b"")
    (wav_dir / "BAC009S0724W0122.wav").write_bytes(b"")
    output = tmp_path / "aishell.jsonl"

    records = prepare_public_asr_manifest(
        corpus="aishell1",
        input_dir=root,
        output_path=output,
        split="dev",
    )

    assert len(records) == 2
    assert records[0].id == "BAC009S0724W0121"
    assert records[0].language == "zh"
    assert records[0].source == "aishell1"
    assert records[0].split == "dev"
    assert records[0].speaker_id == "S0724"
    assert records[0].metadata["corpus_recipe"] == "aishell1"
    assert validate_asr_manifest(output).ok
