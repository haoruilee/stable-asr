"""Core turn-taking dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TurnActionName = Literal[
    "take_turn",
    "keep_listening",
    "continue_speaking",
    "stop_tts_and_listen",
    "ignore",
    "hold",
    "light_ack",
]


@dataclass(frozen=True)
class TurnWindow:
    audio: str
    start: float
    end: float
    sample_rate: int
    label: str
    text: str | None = None
    assistant_speaking: bool = False
    overlap: bool = False
    scenario: str | None = None
    factors: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class TurnPrediction:
    probs: dict[str, float]
    timestamp: float
    embedding: object | None = None

    @property
    def label(self) -> str:
        if not self.probs:
            raise ValueError("cannot choose a label from an empty probability map")
        return max(self.probs, key=self.probs.__getitem__)

    @property
    def confidence(self) -> float:
        return float(self.probs[self.label])


@dataclass(frozen=True)
class TurnAction:
    action: TurnActionName
    confidence: float
    reason: str | None = None

