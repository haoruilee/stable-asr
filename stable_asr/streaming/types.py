"""Streaming ASR data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PartialHypothesis:
    time: float
    text: str
    is_final: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartialHypothesis":
        return cls(
            time=float(data["time"]),
            text=str(data.get("text", "")),
            is_final=bool(data.get("is_final", False)),
        )

    def to_dict(self) -> dict[str, object]:
        return {"time": self.time, "text": self.text, "is_final": self.is_final}


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start: float
    end: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordTimestamp":
        return cls(
            word=str(data.get("word", data.get("text", ""))),
            start=float(data["start"]),
            end=float(data["end"]),
        )

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0

    def to_dict(self) -> dict[str, object]:
        return {"word": self.word, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class StreamingASRRecord:
    id: str
    reference: str
    final_text: str
    audio_duration: float
    processing_time: float
    speech_end_time: float | None = None
    endpoint_time: float | None = None
    word_timestamps: list[WordTimestamp] = field(default_factory=list)
    reference_word_timestamps: list[WordTimestamp] = field(default_factory=list)
    partials: list[PartialHypothesis] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamingASRRecord":
        partials = [PartialHypothesis.from_dict(item) for item in data.get("partials", [])]
        word_timestamps = [
            WordTimestamp.from_dict(item)
            for item in data.get("word_timestamps", data.get("words", []))
        ]
        reference_word_timestamps = [
            WordTimestamp.from_dict(item)
            for item in data.get("reference_word_timestamps", data.get("reference_words", []))
        ]
        final_text = str(data.get("final_text", data.get("hypothesis", "")))
        return cls(
            id=str(data["id"]),
            reference=str(data.get("reference", "")),
            final_text=final_text,
            audio_duration=float(data.get("audio_duration", data.get("duration", 0.0))),
            processing_time=float(data.get("processing_time", data.get("runtime", 0.0))),
            speech_end_time=_optional_float(data, "speech_end_time", "end_of_speech", "reference_endpoint_time"),
            endpoint_time=_optional_float(data, "endpoint_time", "finalized_at", "finalization_time"),
            word_timestamps=word_timestamps,
            reference_word_timestamps=reference_word_timestamps,
            partials=partials,
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "reference": self.reference,
            "final_text": self.final_text,
            "audio_duration": self.audio_duration,
            "processing_time": self.processing_time,
            "speech_end_time": self.speech_end_time,
            "endpoint_time": self.endpoint_time,
            "word_timestamps": [word.to_dict() for word in self.word_timestamps],
            "reference_word_timestamps": [word.to_dict() for word in self.reference_word_timestamps],
            "partials": [partial.to_dict() for partial in self.partials],
            "metadata": self.metadata,
        }


def _optional_float(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return None
