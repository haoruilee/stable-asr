from stable_asr.models.adapters.transcript import load_streaming_transcript_jsonl
from stable_asr.streaming.failures import StreamingFailureThresholds, mine_streaming_failures
from stable_asr.streaming.metrics import evaluate_streaming_records


def test_mine_streaming_failures_from_fixture() -> None:
    records = load_streaming_transcript_jsonl("tests/fixtures/streaming_asr_fast_unstable_sample.jsonl")
    summary = mine_streaming_failures(records)

    assert summary.total_failures > 0
    assert "word_error" in summary.category_counts
    assert any(case.category == "partial_revision" for case in summary.cases)
    assert "Representative Streaming Failures" in summary.to_markdown()


def test_streaming_report_includes_failure_analysis() -> None:
    records = load_streaming_transcript_jsonl("tests/fixtures/streaming_asr_sample.jsonl")
    report = evaluate_streaming_records(records)
    payload = report.to_dict()

    assert "failure_analysis" in payload
    assert payload["failure_analysis"]["total_failures"] >= 0


def test_streaming_failure_thresholds_can_be_relaxed() -> None:
    records = load_streaming_transcript_jsonl("tests/fixtures/streaming_asr_sample.jsonl")
    summary = mine_streaming_failures(
        records,
        thresholds=StreamingFailureThresholds(
            wer=1.0,
            endpoint_delay=10.0,
            first_partial_latency=10.0,
            partial_revision_rate=1.0,
            stable_prefix_ratio=0.0,
            timestamp_drift=10.0,
            rtf=10.0,
        ),
    )

    assert summary.total_failures == 0
