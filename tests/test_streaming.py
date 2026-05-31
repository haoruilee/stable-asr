from stable_asr.models.adapters.transcript import load_streaming_transcript_jsonl
from stable_asr.streaming.metrics import evaluate_streaming_records


def test_load_streaming_transcript_jsonl() -> None:
    records = load_streaming_transcript_jsonl("tests/fixtures/streaming_asr_sample.jsonl")

    assert len(records) == 2
    assert records[0].partials[0].text == "what"
    assert records[0].endpoint_time == 2.1
    assert records[0].word_timestamps[0].word == "what"


def test_evaluate_streaming_records() -> None:
    records = load_streaming_transcript_jsonl("tests/fixtures/streaming_asr_sample.jsonl")
    report = evaluate_streaming_records(records)

    assert report.records == 2
    assert report.wer > 0.0
    assert report.cer >= 0.0
    assert report.rtf > 0.0
    assert report.endpoint_delay > 0.0
    assert report.partial_revision_rate > 0.0
    assert report.timestamp_drift > 0.0
    assert report.failure_analysis.total_failures > 0
    assert "failure_analysis" in report.to_dict()
