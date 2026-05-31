"""Transcript fixture adapter for streaming ASR evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stable_asr.data.formats.jsonl import iter_jsonl
from stable_asr.models.adapters.asr import StreamingASRAdapter
from stable_asr.streaming.types import StreamingASRRecord


@dataclass(frozen=True)
class TranscriptJSONLAdapter:
    """Streaming ASR adapter backed by a normalized transcript JSONL file."""

    name: str
    path: str | Path

    def load_records(self) -> list[StreamingASRRecord]:
        return load_streaming_transcript_jsonl(self.path)


def load_streaming_transcript_jsonl(path: str | Path) -> list[StreamingASRRecord]:
    return [StreamingASRRecord.from_dict(row) for _, row in iter_jsonl(path)]


def transcript_jsonl_adapter(name: str, path: str | Path) -> StreamingASRAdapter:
    return TranscriptJSONLAdapter(name=name, path=path)
