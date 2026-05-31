"""Canonical v0 labels for turn state and system action."""

TURN_LABELS = frozenset(
    {
        "complete",
        "incomplete",
        "backchannel",
        "wait",
    }
)

ACTION_LABELS = frozenset(
    {
        "take_turn",
        "keep_listening",
        "continue_speaking",
        "stop_tts_and_listen",
        "ignore",
        "hold",
        "light_ack",
    }
)

