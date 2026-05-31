from pathlib import Path

import pytest

from stable_asr.data.converters import (
    convert_external_jsonl,
    convert_rows,
    convert_streaming_asr_jsonl,
    convert_streaming_asr_rows,
)
from stable_asr.data.manifest import load_manifest, validate_manifest
from stable_asr.models.adapters.transcript import load_streaming_transcript_jsonl


def test_convert_easyturn_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "easyturn.jsonl"
    count = convert_external_jsonl(
        "tests/fixtures/easyturn_sample.jsonl",
        output,
        schema="easyturn",
    )
    records = load_manifest(output)

    assert count == 3
    assert validate_manifest(output).ok
    assert records[0].turn_label == "complete"
    assert records[1].action_label == "keep_listening"
    assert records[2].turn_label == "backchannel"
    assert records[2].metadata["source_schema"] == "easyturn"


def test_convert_full_duplex_bench_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "fdb.jsonl"
    count = convert_external_jsonl(
        "tests/fixtures/full_duplex_bench_sample.jsonl",
        output,
        schema="full_duplex_bench",
    )
    records = load_manifest(output)

    assert count == 3
    assert records[0].action_label == "stop_tts_and_listen"
    assert records[0].assistant_speaking is True
    assert records[1].turn_label == "backchannel"
    assert records[2].action_label == "ignore"


def test_convert_smart_turn_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "smart_turn.jsonl"
    count = convert_external_jsonl(
        "tests/fixtures/smart_turn_manifest_sample.jsonl",
        output,
        schema="smart_turn",
    )
    records = load_manifest(output)

    assert count == 3
    assert validate_manifest(output).ok
    assert records[0].turn_label == "complete"
    assert records[1].turn_label == "incomplete"
    assert records[2].scenario == "side_conversation"
    assert records[2].action_label == "ignore"


def test_convert_rows_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="cannot normalize turn label"):
        convert_rows(
            [{"audio": "audio/bad.wav", "label": "not_a_label"}],
            schema="easyturn",
        )


def test_convert_whisper_streaming_asr_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "whisper_streaming.jsonl"
    count = convert_streaming_asr_jsonl(
        "tests/fixtures/whisper_transcript_sample.jsonl",
        output,
        schema="whisper",
    )
    records = load_streaming_transcript_jsonl(output)

    assert count == 2
    assert records[0].id == "whisper_001"
    assert records[0].final_text == "what is the weather"
    assert records[0].partials[-1].is_final is True
    assert records[0].partials[-1].text == "what is the weather"
    assert len(records[0].word_timestamps) == 4
    assert records[0].metadata["source_schema"] == "whisper"
    assert records[1].endpoint_time == pytest.approx(2.3)


def test_convert_funasr_streaming_asr_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "funasr_streaming.jsonl"
    count = convert_streaming_asr_jsonl(
        "tests/fixtures/funasr_transcript_sample.jsonl",
        output,
        schema="funasr",
    )
    records = load_streaming_transcript_jsonl(output)

    assert count == 2
    assert records[0].id == "funasr_001"
    assert records[0].audio_duration == pytest.approx(2.0)
    assert records[0].processing_time == pytest.approx(0.36)
    assert records[0].speech_end_time == pytest.approx(1.8)
    assert records[0].partials[-1].time == pytest.approx(1.7)
    assert len(records[1].word_timestamps) == 3
    assert records[1].word_timestamps[0].word == "turn"
    assert records[1].endpoint_time == pytest.approx(2.3)


def test_convert_streaming_asr_rows_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unknown ASR transcript schema"):
        convert_streaming_asr_rows([], schema="unknown")
