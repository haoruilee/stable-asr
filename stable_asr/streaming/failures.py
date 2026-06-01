"""Failure mining for streaming ASR records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from stable_asr.eval.report import dict_table
from stable_asr.streaming.text_normalization import asr_word_tokens, normalize_asr_text
from stable_asr.streaming.types import StreamingASRRecord, WordTimestamp


@dataclass(frozen=True)
class StreamingFailureThresholds:
    wer: float = 0.0
    endpoint_delay: float = 0.3
    first_partial_latency: float = 0.8
    partial_revision_rate: float = 0.0
    stable_prefix_ratio: float = 0.8
    timestamp_drift: float = 0.1
    rtf: float = 0.5


@dataclass(frozen=True)
class StreamingFailureCase:
    id: str
    category: str
    severity: int
    value: float
    threshold: float
    reference: str
    final_text: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "category": self.category,
            "severity": self.severity,
            "value": round(self.value, 6),
            "threshold": round(self.threshold, 6),
            "reference": self.reference,
            "final_text": self.final_text,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StreamingFailureSummary:
    total_failures: int
    category_counts: dict[str, int]
    cases: list[StreamingFailureCase]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_failures": self.total_failures,
            "category_counts": self.category_counts,
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_markdown(self, *, max_cases: int = 20) -> str:
        sections = []
        if self.category_counts:
            sections.append(
                "### Streaming Failure Taxonomy\n\n"
                + dict_table(
                    [
                        {"category": category, "count": count}
                        for category, count in self.category_counts.items()
                    ]
                )
            )
        if self.cases:
            sections.append(
                "### Representative Streaming Failures\n\n"
                + dict_table([case.to_dict() for case in self.cases[:max_cases]])
            )
        return "\n\n".join(sections)


def mine_streaming_failures(
    records: Iterable[StreamingASRRecord],
    *,
    thresholds: StreamingFailureThresholds | None = None,
    max_cases: int = 50,
) -> StreamingFailureSummary:
    thresholds = thresholds or StreamingFailureThresholds()
    failures: list[StreamingFailureCase] = []
    for record in records:
        metrics = _record_metrics(record)
        failures.extend(_record_failures(record, metrics, thresholds))

    failures.sort(key=lambda case: (-case.severity, -case.value, case.category, case.id))
    return StreamingFailureSummary(
        total_failures=len(failures),
        category_counts=_counts(case.category for case in failures),
        cases=failures[:max_cases],
    )


def _record_failures(
    record: StreamingASRRecord,
    metrics: dict[str, float],
    thresholds: StreamingFailureThresholds,
) -> list[StreamingFailureCase]:
    checks = [
        (
            "word_error",
            metrics["wer"],
            thresholds.wer,
            5,
            "final transcript differs from the reference at word level",
            True,
        ),
        (
            "endpoint_delay",
            metrics["endpoint_delay"],
            thresholds.endpoint_delay,
            4,
            "endpoint finalization lags behind speech end",
            True,
        ),
        (
            "partial_revision",
            metrics["partial_revision_rate"],
            thresholds.partial_revision_rate,
            3,
            "partial hypotheses revise previous text",
            True,
        ),
        (
            "timestamp_drift",
            metrics["timestamp_drift"],
            thresholds.timestamp_drift,
            3,
            "word timestamps drift from reference timing",
            True,
        ),
        (
            "first_partial_latency",
            metrics["first_partial_latency"],
            thresholds.first_partial_latency,
            2,
            "first partial hypothesis arrives too late",
            True,
        ),
        (
            "low_stable_prefix",
            metrics["stable_prefix_ratio"],
            thresholds.stable_prefix_ratio,
            2,
            "partial hypotheses expose little stable prefix of the final text",
            False,
        ),
        (
            "slow_rtf",
            metrics["rtf"],
            thresholds.rtf,
            2,
            "processing time is high relative to audio duration",
            True,
        ),
    ]

    failures: list[StreamingFailureCase] = []
    for category, value, threshold, severity, reason, higher_is_bad in checks:
        failed = value > threshold if higher_is_bad else value < threshold
        if not failed:
            continue
        failures.append(
            StreamingFailureCase(
                id=record.id,
                category=category,
                severity=severity,
                value=value,
                threshold=threshold,
                reference=record.reference,
                final_text=record.final_text,
                reason=reason,
            )
        )
    return failures


def _record_metrics(record: StreamingASRRecord) -> dict[str, float]:
    ref_words = asr_word_tokens(record.reference)
    hyp_words = asr_word_tokens(record.final_text)
    partial_texts = [normalize_asr_text(partial.text) for partial in record.partials]
    first_partial_latency = record.partials[0].time if record.partials else 0.0
    return {
        "wer": _safe_div(_edit_distance(ref_words, hyp_words), len(ref_words)),
        "rtf": _safe_div(record.processing_time, record.audio_duration),
        "first_partial_latency": first_partial_latency,
        "endpoint_delay": _endpoint_delay(record),
        "partial_revision_rate": _partial_revision_rate(partial_texts),
        "stable_prefix_ratio": _stable_prefix_ratio(normalize_asr_text(record.final_text), partial_texts),
        "timestamp_drift": _timestamp_drift(record.reference_word_timestamps, record.word_timestamps),
    }


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
    if not partials:
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
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
