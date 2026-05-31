import pytest

from stable_asr.models.adapters.transcript import load_streaming_transcript_jsonl
from stable_asr.streaming.sweep import sweep_streaming_schedule


def test_sweep_streaming_schedule() -> None:
    records = load_streaming_transcript_jsonl("tests/fixtures/streaming_asr_sample.jsonl")
    report = sweep_streaming_schedule(records, chunk_ms_values=[160, 320], lookahead_ms_values=[0, 160])

    rows = report.to_dict()["rows"]
    assert len(rows) == 4
    assert rows[0]["chunk_ms"] == 160
    assert rows[0]["lookahead_ms"] == 0
    assert rows[1]["lookahead_ms"] == 160
    assert rows[1]["first_partial_latency"] > rows[0]["first_partial_latency"]
    assert "chunk_ms" in report.to_markdown()


def test_sweep_streaming_schedule_rejects_bad_config() -> None:
    records = load_streaming_transcript_jsonl("tests/fixtures/streaming_asr_sample.jsonl")

    with pytest.raises(ValueError, match="positive"):
        sweep_streaming_schedule(records, chunk_ms_values=[0], lookahead_ms_values=[0])
