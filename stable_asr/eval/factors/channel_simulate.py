"""F2' — synthetic channel simulation (telephone band, codec degradation).

The "real" F2 channel factor is corpus selection over LDC channels
(Switchboard phone, Mixer cellular, AMI meeting, HUB broadcast). This
module provides a *synthetic* version usable today on any audio:

    * telephone — bandpass 300–3400 Hz + downsample to 8 kHz
    * cellular  — telephone + light μ-law-style nonlinearity
    * narrowband — bandpass 300–3400 Hz at original SR (no downsample)

These do not replace the real LDC channels but give us a controlled
"channel" factor we can exercise on AMI / LibriSpeech today, then
compare against the LDC-real channel results once those land.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.eval.scenario_record import ScenarioRecord

ChannelKind = Literal["telephone", "cellular", "narrowband", "clean"]


@dataclass(frozen=True)
class ChannelSimulateConfig:
    kind: ChannelKind
    output_dir: Path
    seed: int = 0
    level_label: str | None = None

    @property
    def label(self) -> str:
        return self.level_label or f"channel_{self.kind}"


def _butter_bandpass(low: float, high: float, sr: int, order: int = 4):
    """Bandpass coefficients using a Butterworth filter.

    Falls back to a simple two-pole approximation when scipy is missing.
    """
    try:
        from scipy.signal import butter
    except ImportError:
        # crude fallback: two-pole high+low; not flat in passband but
        # qualitatively similar for pilot work. Document this in the
        # config so the paper can flag synthetic-channel results.
        return None

    nyq = 0.5 * sr
    return butter(order, [low / nyq, high / nyq], btype="band")


def _bandpass(samples: "np.ndarray", sr: int, low: float, high: float):
    import numpy as np

    coeffs = _butter_bandpass(low, high, sr)
    if coeffs is None:
        # Simple FFT-based bandpass fallback.
        n = len(samples)
        if n == 0:
            return samples
        F = np.fft.rfft(samples)
        freqs = np.fft.rfftfreq(n, d=1 / sr)
        mask = (freqs >= low) & (freqs <= high)
        F = F * mask
        return np.fft.irfft(F, n=n).astype("float32")
    b, a = coeffs
    try:
        from scipy.signal import filtfilt
        return filtfilt(b, a, samples).astype("float32")
    except ImportError:
        return samples


def _mu_law_round_trip(samples: "np.ndarray", mu: int = 255) -> "np.ndarray":
    """Approximate cellular vocoder degradation via μ-law encode/decode."""

    import numpy as np

    s = np.clip(samples, -1.0, 1.0).astype("float32")
    sgn = np.sign(s)
    encoded = sgn * np.log1p(mu * np.abs(s)) / np.log1p(mu)
    quantized = np.round(encoded * (mu // 2)) / (mu // 2)
    decoded = sgn * (np.expm1(np.abs(quantized) * np.log1p(mu)) / mu)
    return decoded.astype("float32")


def _resample(samples, src_sr, dst_sr):
    if src_sr == dst_sr:
        return samples
    import torch
    import torchaudio.functional as F

    wf = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    return F.resample(wf, src_sr, dst_sr).squeeze(0).numpy().astype("float32")


def apply_channel_simulate(
    record: TurnManifestRecord,
    config: ChannelSimulateConfig,
) -> ScenarioRecord:
    """Apply a synthetic channel transformation."""

    import numpy as np
    import soundfile as sf

    speech, sr = sf.read(str(Path(record.audio)), always_2d=False)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    speech = speech.astype("float32", copy=False)

    out_sr = sr
    if config.kind == "clean":
        processed = speech
    elif config.kind == "narrowband":
        processed = _bandpass(speech, sr, 300.0, 3400.0)
    elif config.kind == "telephone":
        # Bandpass at original SR, then downsample to 8 kHz, then
        # upsample back to keep the manifest sample rate stable.
        bp = _bandpass(speech, sr, 300.0, 3400.0)
        ds = _resample(bp, sr, 8000)
        processed = _resample(ds, 8000, sr)
        out_sr = sr  # we keep the original SR for downstream consistency
    elif config.kind == "cellular":
        bp = _bandpass(speech, sr, 300.0, 3400.0)
        ds = _resample(bp, sr, 8000)
        deg = _mu_law_round_trip(ds)
        processed = _resample(deg, 8000, sr)
        out_sr = sr
    else:  # pragma: no cover — guarded by Literal
        raise ValueError(f"unknown channel kind: {config.kind}")

    peak = float(np.max(np.abs(processed))) if len(processed) else 0.0
    norm = 1.0
    if peak > 0.99:
        norm = 0.99 / peak
        processed = processed * norm

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record.id}__{config.label}.wav"
    sf.write(str(out_path), processed, out_sr)

    return ScenarioRecord.from_record(
        record,
        audio=str(out_path),
        factor="channel_simulate",
        factor_level=config.label,
        factor_params={
            "kind": str(config.kind),
            "input_sample_rate": int(sr),
            "output_sample_rate": int(out_sr),
            "post_norm_factor": float(norm),
        },
        sample_rate=int(out_sr),
    )
