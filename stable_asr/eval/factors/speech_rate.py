"""F3 — controlled speech-rate perturbation via torchaudio resampling.

We change the *playback rate* of the audio (faster speech = shorter
duration) by resampling to ``int(orig_sr * rate)`` and then re-tagging
the result as ``orig_sr``. This is the simplest controlled perturbation
that preserves spectral content while compressing/dilating time, which
is exactly what we want for a speech-rate factor: the words are the
same, the rate is different.

Caveats:

* This is *not* pitch-preserving (a 1.3× speedup raises pitch). For an
  empirical-findings paper that is fine — pitch-preserving stretch
  (PSOLA / phase vocoder) introduces its own artefacts that confound the
  factor. Document the choice in the paper Methods section.
* The resulting audio's "true" sample rate is ``orig_sr`` again (we lie
  about the rate to make the playback faster). The duration shrinks by
  ``1 / rate``.
* Rates are required to be in the range [0.5, 2.0] to avoid extreme
  artefacts; 0.7 / 1.0 / 1.3 are the recommended pilot levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.eval.scenario_record import ScenarioRecord


@dataclass(frozen=True)
class SpeechRateConfig:
    rate: float           # 1.0 = unchanged, 0.7 = slower, 1.3 = faster
    output_dir: Path
    level_label: str | None = None  # default: f"rate_{rate}x"

    def __post_init__(self) -> None:
        if not 0.5 <= self.rate <= 2.0:
            raise ValueError(
                f"speech-rate {self.rate} outside [0.5, 2.0]; choose a tighter "
                f"range to avoid extreme artefacts that confound the factor"
            )

    @property
    def label(self) -> str:
        return self.level_label or f"rate_{self.rate:.2f}x"


def apply_speech_rate(
    record: TurnManifestRecord,
    config: SpeechRateConfig,
) -> ScenarioRecord:
    """Apply a controlled speech-rate perturbation, returning a ScenarioRecord."""

    import torch
    import torchaudio
    import torchaudio.functional as F

    src = Path(record.audio)
    waveform, orig_sr = torchaudio.load(str(src))
    # mono-mix if multichannel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to orig_sr * rate; the perceived effect is faster/slower playback
    # because we keep the metadata sample_rate at orig_sr afterward.
    new_internal_sr = int(round(orig_sr * config.rate))
    resampled = F.resample(waveform, orig_sr, new_internal_sr)

    # Save with the *original* sample_rate tag so playback is at the new rate.
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record.id}__{config.label}.wav"
    torchaudio.save(str(out_path), resampled, orig_sr)

    new_duration = (record.end - record.start) / config.rate
    return ScenarioRecord.from_record(
        record,
        audio=str(out_path),
        factor="speech_rate",
        factor_level=config.label,
        factor_params={
            "rate": config.rate,
            "orig_sample_rate": int(orig_sr),
            "internal_resample_sr": int(new_internal_sr),
        },
        sample_rate=int(orig_sr),
        start=0.0,
        end=round(new_duration, 6),
    )
