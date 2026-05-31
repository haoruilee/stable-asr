"""VAD pause baseline for endpointing and turn-taking evaluation."""

from __future__ import annotations

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.turn.types import TurnPrediction


class VADPauseBaseline:
    """Predict complete/incomplete from a VAD-derived trailing pause.

    The v0 implementation reads `vad_pause_ms` from record metadata. If that is
    absent, it falls back to `pause_ms`. This keeps evaluation deterministic and
    avoids pulling in an audio/VAD dependency before the interfaces are stable.
    """

    def __init__(self, complete_pause_ms: int = 700) -> None:
        if complete_pause_ms <= 0:
            raise ValueError("complete_pause_ms must be positive")
        self.complete_pause_ms = complete_pause_ms

    def predict(self, record: TurnManifestRecord) -> TurnPrediction:
        pause_ms = int(record.metadata.get("vad_pause_ms", record.metadata.get("pause_ms", 0)))
        complete = min(max(pause_ms / self.complete_pause_ms, 0.0), 1.0)
        if pause_ms >= self.complete_pause_ms:
            complete = max(complete, 0.95)
        else:
            complete = min(complete, 0.49)

        return TurnPrediction(
            probs={
                "complete": complete,
                "incomplete": 1.0 - complete,
                "backchannel": 0.0,
                "wait": 0.0,
            },
            timestamp=record.end,
        )

