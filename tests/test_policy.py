from stable_asr.turn.policy import TurnPolicy, TurnPolicyConfig
from stable_asr.turn.types import TurnPrediction


def test_turn_policy_take_turn_after_hysteresis() -> None:
    policy = TurnPolicy(TurnPolicyConfig(complete_threshold=0.7, complete_hysteresis=2))
    prediction = TurnPrediction({"complete": 0.8}, timestamp=1.0)

    first = policy.decide(prediction)
    second = policy.decide(prediction)

    assert first.action == "keep_listening"
    assert second.action == "take_turn"


def test_turn_policy_backchannel_continues_speaking() -> None:
    policy = TurnPolicy()
    prediction = TurnPrediction({"backchannel": 0.9, "complete": 0.1}, timestamp=1.0)

    action = policy.decide(prediction, assistant_speaking=True)

    assert action.action == "continue_speaking"


def test_turn_policy_interrupts_assistant() -> None:
    policy = TurnPolicy(TurnPolicyConfig(complete_threshold=0.7, interrupt_min_confidence=0.7))
    prediction = TurnPrediction({"complete": 0.8}, timestamp=1.0)

    action = policy.decide(prediction, assistant_speaking=True)

    assert action.action == "stop_tts_and_listen"

