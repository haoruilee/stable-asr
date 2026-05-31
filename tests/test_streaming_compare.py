from pathlib import Path

import pytest

from stable_asr.models.adapters import TranscriptJSONLAdapter
from stable_asr.streaming.compare import compare_streaming_adapters, compare_streaming_transcript_jsonl


def test_compare_streaming_transcript_jsonl() -> None:
    report = compare_streaming_transcript_jsonl(
        [
            ("balanced", "tests/fixtures/streaming_asr_sample.jsonl"),
            ("fast_unstable", "tests/fixtures/streaming_asr_fast_unstable_sample.jsonl"),
        ]
    )

    rows = report.to_dict()["rows"]
    assert len(rows) == 2
    assert rows[0]["adapter"] == "balanced"
    assert rows[1]["adapter"] == "fast_unstable"
    assert rows[1]["rtf"] < rows[0]["rtf"]
    assert rows[1]["wer"] > rows[0]["wer"]
    assert "fast_unstable" in report.to_markdown()


def test_compare_streaming_adapters_accepts_adapter_objects() -> None:
    report = compare_streaming_adapters(
        [
            TranscriptJSONLAdapter("balanced", "tests/fixtures/streaming_asr_sample.jsonl"),
            TranscriptJSONLAdapter("fast_unstable", "tests/fixtures/streaming_asr_fast_unstable_sample.jsonl"),
        ]
    )

    rows = report.to_dict()["rows"]
    assert rows[0]["adapter"] == "balanced"
    assert rows[0]["input_path"] == "tests/fixtures/streaming_asr_sample.jsonl"
    assert rows[1]["wer"] > rows[0]["wer"]


def test_compare_streaming_transcript_jsonl_rejects_duplicate_adapter(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate adapter"):
        compare_streaming_transcript_jsonl(
            [
                ("same", "tests/fixtures/streaming_asr_sample.jsonl"),
                ("same", str(path)),
            ]
        )
