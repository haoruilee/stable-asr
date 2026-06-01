from stable_asr.models.adapters.transcript import load_streaming_transcript_jsonl
from stable_asr.streaming.metrics import evaluate_streaming_records
from stable_asr.streaming.text_normalization import asr_char_tokens, asr_word_tokens, normalize_asr_text
from stable_asr.streaming.types import StreamingASRRecord


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


def test_streaming_metrics_normalize_case_punctuation_and_cjk() -> None:
    records = [
        StreamingASRRecord(
            id="case",
            reference="Hello, WORLD!",
            final_text="hello world",
            audio_duration=1.0,
            processing_time=0.2,
        ),
        StreamingASRRecord(
            id="zh",
            reference="今天天气不错。",
            final_text="今天天气不错",
            audio_duration=1.0,
            processing_time=0.2,
        ),
    ]
    report = evaluate_streaming_records(records)

    assert normalize_asr_text("Hello, WORLD!") == "hello world"
    assert asr_word_tokens("今天天气不错。") == ["今", "天", "天", "气", "不", "错"]
    assert asr_char_tokens("Hello, WORLD!") == list("helloworld")
    assert report.wer == 0.0
    assert report.cer == 0.0
