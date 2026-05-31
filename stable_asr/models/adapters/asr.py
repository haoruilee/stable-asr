"""ASR adapter protocols and result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from stable_asr.streaming.types import PartialHypothesis, StreamingASRRecord, WordTimestamp


@dataclass(frozen=True)
class ASRResult:
    text: str
    segments: list[str] = field(default_factory=list)
    words: list[WordTimestamp] = field(default_factory=list)
    rtf: float = 0.0
    latency_ms: float = 0.0
    confidence: float | None = None


@dataclass(frozen=True)
class PartialASRResult:
    time: float
    text: str
    is_final: bool = False

    def to_hypothesis(self) -> PartialHypothesis:
        return PartialHypothesis(time=self.time, text=self.text, is_final=self.is_final)


class ASRModel(Protocol):
    """Minimal interface for optional real ASR adapters."""

    name: str

    def transcribe(self, audio: str | Path) -> ASRResult:
        ...

    def stream(self, audio_chunks: Iterable[bytes]) -> Iterator[PartialASRResult]:
        ...


class StreamingASRAdapter(Protocol):
    """Evaluation adapter that returns normalized streaming ASR records."""

    name: str

    def load_records(self) -> list[StreamingASRRecord]:
        ...
