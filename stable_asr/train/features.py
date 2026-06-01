"""Feature extraction for NanoTurn training."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from stable_asr.data.audio import load_audio_mono
from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.turn.nanoturn import require_torch, torch

FEATURE_NAMES = (
    "pause_ms",
    "vad_pause_ms",
    "duration_ms",
    "assistant_speaking",
    "overlap",
    "snr_db",
    "speaking_rate",
    "network_jitter_ms",
)
AUDIO_FEATURE_NAMES = tuple(f"logmel_{index:02d}" for index in range(32))
FEATURE_SOURCE_ALIASES = {
    "manifest_metadata_v0": "metadata",
    "metadata_v0": "metadata",
    "logmel_v0": "audio",
    "audio_logmel_v0": "audio",
}


def records_to_features(
    records: list[TurnManifestRecord],
    *,
    feature_source: str = "metadata",
    audio_root: str | Path | None = None,
    feature_cache: str | Path | None = None,
    feature_cache_format: str | None = None,
    feature_cache_mode: str = "auto",
):
    require_torch()
    feature_source = normalize_feature_source(feature_source)
    if feature_source == "metadata":
        return torch.tensor([record_to_features(record) for record in records], dtype=torch.float32)
    if feature_source == "audio":
        if feature_cache:
            from stable_asr.train.feature_cache import ensure_logmel_feature_cache, load_logmel_feature_cache

            ensure_logmel_feature_cache(
                records,
                feature_cache,
                format=feature_cache_format,
                mode=feature_cache_mode,
                audio_root=audio_root,
            )
            return load_logmel_feature_cache(
                feature_cache,
                format=feature_cache_format,
                record_ids=[record.id for record in records],
            )
        audio_cache: dict[Path, tuple[list[float], int]] = {}
        return torch.stack(
            [record_to_logmel_features(record, audio_root=audio_root, audio_cache=audio_cache) for record in records]
        )
    raise ValueError(f"unknown feature_source: {feature_source}")


def feature_names(feature_source: str) -> tuple[str, ...]:
    feature_source = normalize_feature_source(feature_source)
    if feature_source == "metadata":
        return FEATURE_NAMES
    if feature_source == "audio":
        return AUDIO_FEATURE_NAMES
    raise ValueError(f"unknown feature_source: {feature_source}")


def normalize_feature_source(feature_source: str) -> str:
    return FEATURE_SOURCE_ALIASES.get(feature_source, feature_source)


def record_to_logmel_features(
    record: TurnManifestRecord,
    *,
    audio_root: str | Path | None = None,
    audio_cache: dict[Path, tuple[list[float], int]] | None = None,
):
    """Compute fixed-size log spectral features from a WAV file.

    This is a lightweight v0 audio frontend. It intentionally returns a pooled
    feature vector so it can plug into the existing NanoTurn MLP while the
    streaming log-mel CNN/TCN frontend is developed.
    """

    require_torch()
    path = Path(record.audio)
    if not path.is_absolute() and audio_root is not None and not path.exists():
        path = Path(audio_root) / path
    if audio_cache is not None and path in audio_cache:
        samples, sample_rate = audio_cache[path]
    else:
        samples, sample_rate = load_audio_mono(path, target_sample_rate=record.sample_rate)
        if audio_cache is not None:
            audio_cache[path] = (samples, sample_rate)
    if sample_rate != record.sample_rate:
        raise ValueError(f"sample rate mismatch for {path}: audio={sample_rate}, manifest={record.sample_rate}")
    start = max(0, int(round(record.start * sample_rate)))
    end = min(len(samples), int(round(record.end * sample_rate)))
    if end <= start:
        raise ValueError(f"empty audio window for {record.id}: start={record.start} end={record.end}")
    waveform = torch.tensor(samples[start:end], dtype=torch.float32)
    if waveform.numel() < 400:
        waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.numel()))
    spectrum = torch.stft(
        waveform,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=_hann_window(400),
        return_complex=True,
    ).abs()
    bands = _pool_frequency_bands(spectrum, bands=len(AUDIO_FEATURE_NAMES))
    return torch.log1p(bands.mean(dim=1))


@lru_cache(maxsize=8)
def _hann_window(win_length: int):
    return torch.hann_window(win_length)


def _pool_frequency_bands(spectrum, *, bands: int):
    freq_bins = spectrum.shape[0]
    edges = torch.linspace(0, freq_bins, steps=bands + 1).round().long()
    pooled = []
    for index in range(bands):
        start = int(edges[index].item())
        end = max(start + 1, int(edges[index + 1].item()))
        pooled.append(spectrum[start:end].mean(dim=0))
    return torch.stack(pooled)


def record_to_features(record: TurnManifestRecord) -> list[float]:
    metadata = record.metadata
    duration_ms = float(metadata.get("duration_ms", record.duration * 1000.0))
    return [
        _scale(float(metadata.get("pause_ms", 0.0)), 0.0, 2000.0),
        _scale(float(metadata.get("vad_pause_ms", metadata.get("pause_ms", 0.0))), 0.0, 2000.0),
        _scale(duration_ms, 0.0, 3000.0),
        1.0 if record.assistant_speaking else 0.0,
        1.0 if record.overlap else 0.0,
        _scale(float(metadata.get("snr_db", 20.0)), -5.0, 20.0),
        _scale(float(metadata.get("speaking_rate", 1.0)), 0.5, 1.8),
        _scale(float(metadata.get("network_jitter_ms", 0.0)), 0.0, 300.0),
    ]


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("high must be greater than low")
    return max(0.0, min(1.0, (value - low) / (high - low)))
