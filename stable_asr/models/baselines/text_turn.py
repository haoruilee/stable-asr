"""Text-only turn baseline using ASR transcript cues."""

from __future__ import annotations

import re

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.turn.types import TurnPrediction


_BACKCHANNEL_TEXT = {
    "嗯",
    "嗯嗯",
    "嗯哼",
    "对",
    "对的",
    "好的",
    "好",
    "是",
    "是的",
    "继续",
    "可以",
    "ok",
    "okay",
    "yes",
    "yeah",
    "yep",
    "uh huh",
    "mhm",
}

_WAIT_PATTERNS = (
    "先别说",
    "别说",
    "等一下",
    "等下",
    "稍等",
    "我想一下",
    "让我想想",
    "hold on",
    "wait",
    "give me a second",
    "let me think",
)

_INTERRUPT_PATTERNS = (
    "不是",
    "不对",
    "错了",
    "停一下",
    "打断一下",
    "not that",
    "that's wrong",
    "stop",
)

_INCOMPLETE_SUFFIXES = (
    "然后",
    "但是",
    "不过",
    "因为",
    "如果",
    "就是",
    "比如",
    "还有",
    "以及",
    "北京",
    "的",
    "and",
    "but",
    "because",
    "if",
    "so",
    "then",
)


class TextTurnBaseline:
    """Predict turn state from transcript text only.

    This baseline intentionally ignores audio and pause metadata. It gives the
    platform a deterministic semantic baseline that can be compared against
    VAD-only and audio turn models under the same evaluator.
    """

    def __init__(self, *, prefer_asr_text: bool = True) -> None:
        self.prefer_asr_text = prefer_asr_text

    def predict(self, record: TurnManifestRecord) -> TurnPrediction:
        text = self._record_text(record)
        normalized = _normalize_text(text)

        if not normalized:
            return _prediction("incomplete", timestamp=record.end, confidence=0.55)
        if record.assistant_speaking and _contains_interrupt_cue(normalized):
            return _prediction("complete", timestamp=record.end, confidence=0.84)
        if _contains_wait_cue(normalized):
            return _prediction("wait", timestamp=record.end, confidence=0.88)
        if _is_backchannel(normalized):
            return _prediction("backchannel", timestamp=record.end, confidence=0.86)
        if _looks_incomplete(normalized):
            return _prediction("incomplete", timestamp=record.end, confidence=0.74)
        return _prediction("complete", timestamp=record.end, confidence=0.78)

    def _record_text(self, record: TurnManifestRecord) -> str:
        if self.prefer_asr_text:
            return record.asr_text or record.text or ""
        return record.text or record.asr_text or ""


def _prediction(label: str, *, timestamp: float, confidence: float) -> TurnPrediction:
    floor = (1.0 - confidence) / 3.0
    probs = {
        "complete": floor,
        "incomplete": floor,
        "backchannel": floor,
        "wait": floor,
    }
    probs[label] = confidence
    return TurnPrediction(probs=probs, timestamp=timestamp)


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s，。！？,.!?;；:：\"'“”‘’（）()]+", " ", text)
    return text.strip()


def _contains_wait_cue(text: str) -> bool:
    return any(pattern in text for pattern in _WAIT_PATTERNS)


def _contains_interrupt_cue(text: str) -> bool:
    return any(pattern in text for pattern in _INTERRUPT_PATTERNS)


def _is_backchannel(text: str) -> bool:
    compact = text.replace(" ", "")
    return compact in _BACKCHANNEL_TEXT or text in _BACKCHANNEL_TEXT


def _looks_incomplete(text: str) -> bool:
    compact = text.replace(" ", "")
    if compact.endswith(("……", "...")):
        return True
    return any(compact.endswith(suffix) for suffix in _INCOMPLETE_SUFFIXES)
