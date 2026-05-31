"""Small WAV utilities for demo audio and early audio-feature tests."""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WavInfo:
    path: str
    sample_rate: int
    channels: int
    sample_width: int
    frames: int

    @property
    def duration_sec(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def inspect_wav(path: str | Path) -> WavInfo:
    """Read basic WAV container metadata without loading samples."""

    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        return WavInfo(
            path=str(path),
            sample_rate=handle.getframerate(),
            channels=handle.getnchannels(),
            sample_width=handle.getsampwidth(),
            frames=handle.getnframes(),
        )


def load_wav_mono(path: str | Path) -> tuple[list[float], int]:
    """Load a PCM WAV file as mono float samples in [-1, 1]."""

    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise ValueError(f"only 16-bit PCM WAV is supported, got sample width {sample_width}")
    values = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    if channels == 1:
        samples = [value / 32768.0 for value in values]
    else:
        samples = []
        for index in range(0, len(values), channels):
            samples.append(sum(values[index : index + channels]) / channels / 32768.0)
    return samples, sample_rate


def write_wav_mono(path: str | Path, samples: list[float], sample_rate: int = 16000) -> None:
    """Write mono float samples in [-1, 1] as 16-bit PCM WAV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = [max(-1.0, min(1.0, sample)) for sample in samples]
    payload = b"".join(struct.pack("<h", int(sample * 32767.0)) for sample in clipped)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


def synth_tone(
    duration_sec: float,
    *,
    sample_rate: int = 16000,
    frequency: float = 220.0,
    amplitude: float = 0.25,
    noise: float = 0.0,
    seed: int = 0,
) -> list[float]:
    """Create a deterministic tone plus simple pseudo-noise."""

    samples: list[float] = []
    state = seed or 1
    for index in range(max(1, int(duration_sec * sample_rate))):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        pseudo_noise = ((state / 0x7FFFFFFF) * 2.0 - 1.0) * noise
        tone = amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate)
        samples.append(tone + pseudo_noise)
    return samples
