"""Turn-action policies that convert model probabilities into system actions."""

from __future__ import annotations

from dataclasses import dataclass, field

from stable_asr.turn.types import TurnAction, TurnPrediction


@dataclass(frozen=True)
class TurnPolicyConfig:
    complete_threshold: float = 0.75
    backchannel_threshold: float = 0.70
    wait_threshold: float = 0.60
    complete_hysteresis: int = 1
    interrupt_min_confidence: float = 0.75

    def validate(self) -> None:
        for name, value in (
            ("complete_threshold", self.complete_threshold),
            ("backchannel_threshold", self.backchannel_threshold),
            ("wait_threshold", self.wait_threshold),
            ("interrupt_min_confidence", self.interrupt_min_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.complete_hysteresis < 1:
            raise ValueError("complete_hysteresis must be at least 1")


class TurnPolicy:
    """Threshold and hysteresis policy for turn-taking decisions."""

    def __init__(self, config: TurnPolicyConfig | None = None) -> None:
        self.config = config or TurnPolicyConfig()
        self.config.validate()
        self._complete_streak = 0

    def reset(self) -> None:
        self._complete_streak = 0

    def decide(
        self,
        prediction: TurnPrediction,
        *,
        assistant_speaking: bool = False,
    ) -> TurnAction:
        probs = prediction.probs
        complete = probs.get("complete", 0.0)
        backchannel = probs.get("backchannel", 0.0)
        wait = probs.get("wait", 0.0)

        if wait >= self.config.wait_threshold:
            self._complete_streak = 0
            return TurnAction("hold", wait, "wait probability crossed threshold")

        if assistant_speaking and backchannel >= self.config.backchannel_threshold:
            self._complete_streak = 0
            return TurnAction(
                "continue_speaking",
                backchannel,
                "backchannel during assistant speech",
            )

        if complete >= self.config.complete_threshold:
            self._complete_streak += 1
        else:
            self._complete_streak = 0

        if self._complete_streak >= self.config.complete_hysteresis:
            if assistant_speaking and complete >= self.config.interrupt_min_confidence:
                return TurnAction("stop_tts_and_listen", complete, "user interruption")
            return TurnAction("take_turn", complete, "complete probability crossed threshold")

        return TurnAction("keep_listening", 1.0 - complete, "no action threshold crossed")

