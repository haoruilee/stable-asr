"""Streaming ASR metrics beyond WER/CER."""

from __future__ import annotations

from dataclasses import dataclass

from stable_asr.streaming.failures import StreamingFailureSummary, mine_streaming_failures
from stable_asr.streaming.types import StreamingASRRecord, WordTimestamp


@dataclass(frozen=True)
class StreamingASRReport:
    records: int
    wer: float
    cer: float
    rtf: float
    first_partial_latency: float
    final_latency: float
    endpoint_delay: float
    partial_revision_rate: float
    stable_prefix_ratio: float
    timestamp_drift: float
    failure_analysis: StreamingFailureSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "wer": self.wer,
            "cer": self.cer,
            "rtf": self.rtf,
            "first_partial_latency": self.first_partial_latency,
            "final_latency": self.final_latency,
            "endpoint_delay": self.endpoint_delay,
            "partial_revision_rate": self.partial_revision_rate,
            "stable_prefix_ratio": self.stable_prefix_ratio,
            "timestamp_drift": self.timestamp_drift,
            "failure_analysis": self.failure_analysis.to_dict(),
        }


def evaluate_streaming_records(records: list[StreamingASRRecord]) -> StreamingASRReport:
    if not records:
        raise ValueError("records must not be empty")

    total_word_edits = 0
    total_words = 0
    total_char_edits = 0
    total_chars = 0
    total_processing = 0.0
    total_audio = 0.0
    first_latencies: list[float] = []
    final_latencies: list[float] = []
    endpoint_delays: list[float] = []
    revision_rates: list[float] = []
    stable_prefix_ratios: list[float] = []
    timestamp_drifts: list[float] = []

    for record in records:
        ref_words = _words(record.reference)
        hyp_words = _words(record.final_text)
        total_word_edits += _edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
        total_char_edits += _edit_distance(list(record.reference), list(record.final_text))
        total_chars += len(record.reference)
        total_processing += record.processing_time
        total_audio += record.audio_duration

        if record.partials:
            first_latencies.append(record.partials[0].time)
            final_latencies.append(record.partials[-1].time)
            revision_rates.append(_partial_revision_rate([partial.text for partial in record.partials]))
            stable_prefix_ratios.append(_stable_prefix_ratio(record.final_text, [partial.text for partial in record.partials]))
        else:
            first_latencies.append(0.0)
            final_latencies.append(record.audio_duration)
            revision_rates.append(0.0)
            stable_prefix_ratios.append(1.0)
        endpoint_delays.append(_endpoint_delay(record))
        timestamp_drifts.append(_timestamp_drift(record.reference_word_timestamps, record.word_timestamps))

    return StreamingASRReport(
        records=len(records),
        wer=_safe_div(total_word_edits, total_words),
        cer=_safe_div(total_char_edits, total_chars),
        rtf=_safe_div(total_processing, total_audio),
        first_partial_latency=sum(first_latencies) / len(first_latencies),
        final_latency=sum(final_latencies) / len(final_latencies),
        endpoint_delay=sum(endpoint_delays) / len(endpoint_delays),
        partial_revision_rate=sum(revision_rates) / len(revision_rates),
        stable_prefix_ratio=sum(stable_prefix_ratios) / len(stable_prefix_ratios),
        timestamp_drift=sum(timestamp_drifts) / len(timestamp_drifts),
        failure_analysis=mine_streaming_failures(records),
    )


def _partial_revision_rate(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    revisions = 0
    for previous, current in zip(texts, texts[1:]):
        if not current.startswith(previous):
            revisions += 1
    return revisions / (len(texts) - 1)


def _stable_prefix_ratio(final_text: str, partials: list[str]) -> float:
    if not final_text:
        return 1.0
    prefix = ""
    for partial in partials:
        prefix = _common_prefix(prefix or partial, partial)
    stable = _common_prefix(prefix, final_text)
    return len(stable) / len(final_text)


def _endpoint_delay(record: StreamingASRRecord) -> float:
    speech_end = record.speech_end_time if record.speech_end_time is not None else record.audio_duration
    if record.endpoint_time is not None:
        endpoint_time = record.endpoint_time
    elif record.partials:
        final_partials = [partial for partial in record.partials if partial.is_final]
        endpoint_time = (final_partials[-1] if final_partials else record.partials[-1]).time
    else:
        endpoint_time = record.audio_duration
    return max(0.0, endpoint_time - speech_end)


def _timestamp_drift(reference: list[WordTimestamp], hypothesis: list[WordTimestamp]) -> float:
    pairs = list(zip(reference, hypothesis))
    if not pairs:
        return 0.0
    total = 0.0
    for ref_word, hyp_word in pairs:
        total += (abs(ref_word.start - hyp_word.start) + abs(ref_word.end - hyp_word.end)) / 2.0
    return total / len(pairs)


def _common_prefix(a: str, b: str) -> str:
    index = 0
    limit = min(len(a), len(b))
    while index < limit and a[index] == b[index]:
        index += 1
    return a[:index]


def _edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _words(text: str) -> list[str]:
    return [token for token in text.strip().split() if token]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
