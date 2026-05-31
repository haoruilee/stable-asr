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
