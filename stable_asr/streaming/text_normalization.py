"""Text normalization helpers for streaming ASR metrics."""

from __future__ import annotations

import unicodedata


def normalize_asr_text(text: str) -> str:
    """Normalize ASR text for metric comparison.

    The goal is deliberately conservative: fold case/width, remove punctuation
    and symbols as token boundaries, and collapse whitespace. This keeps
    benchmark scores focused on recognition errors rather than presentation
    differences such as commas, periods, or title case.
    """

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    chars: list[str] = []
    previous_space = True
    for char in normalized:
        category = unicodedata.category(char)
        if char.isspace() or category[0] in {"P", "S"}:
            if not previous_space:
                chars.append(" ")
                previous_space = True
            continue
        chars.append(char)
        previous_space = False
    return "".join(chars).strip()


def asr_word_tokens(text: str) -> list[str]:
    """Return word-like ASR tokens, using character tokens for CJK text."""

    normalized = normalize_asr_text(text)
    tokens: list[str] = []
    current: list[str] = []
    for char in normalized:
        if char.isspace():
            _flush_current(tokens, current)
            continue
        if _is_cjk(char):
            _flush_current(tokens, current)
            tokens.append(char)
            continue
        current.append(char)
    _flush_current(tokens, current)
    return tokens


def asr_char_tokens(text: str) -> list[str]:
    return [char for char in normalize_asr_text(text) if not char.isspace()]


def _flush_current(tokens: list[str], current: list[str]) -> None:
    if current:
        tokens.append("".join(current))
        current.clear()


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
    )
