"""Rule-based endpointing baseline for early evaluation plumbing."""

from __future__ import annotations

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.turn.types import TurnPrediction


class RuleEndpointBaseline:
    """Map a silence duration to a turn-complete probability.

    This baseline does not inspect audio. It exists to make policy and metric
    plumbing testable before adding VAD and neural models.
    """

    def __init__(self, complete_pause_ms: int = 700) -> None:
        if complete_pause_ms <= 0:
            raise ValueError("complete_pause_ms must be positive")
        self.complete_pause_ms = complete_pause_ms

    def predict(self, record: TurnManifestRecord) -> TurnPrediction:
        pause_ms = int(record.metadata.get("pause_ms", 0))
        return self.predict_from_pause(pause_ms=pause_ms, timestamp=record.end)

    def predict_from_pause(self, pause_ms: int, timestamp: float = 0.0) -> TurnPrediction:
        complete = 1.0 if pause_ms >= self.complete_pause_ms else 0.0
        return TurnPrediction(
            probs={
                "complete": complete,
                "incomplete": 1.0 - complete,
                "backchannel": 0.0,
                "wait": 0.0,
            },
            timestamp=timestamp,
        )
