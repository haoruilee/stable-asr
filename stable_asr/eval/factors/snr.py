"""F4 — controlled SNR perturbation via additive noise.

Mixes a noise source (MUSAN / DEMAND / WHAM! / synthetic gaussian) into
the speech at a controlled signal-to-noise ratio. The SNR is computed
over the speech region only and is exact by construction — the noise
amplitude is solved from the measured speech power, so the resulting
SNR is the parameter we hand in, not a measurement.

Noise source priority:

1. ``noise_dir`` directory of WAV files — use a deterministic per-record
   pick keyed by record id (so the same record always gets the same
   noise sample regardless of run order).
2. ``noise_path`` single noise WAV — used for every record.
3. Synthetic fallback — pink noise generated from numpy. Suitable for
   unit tests and pilot smoke runs; not for paper-quality results.

Loudness model:

* Speech power ``P_s`` = mean(speech**2) over the entire clip.
* Noise sample is rescaled to ``P_n = P_s / (10 ** (snr_db / 10))``.
* Mixed = speech + scaled_noise. Resulting SNR equals ``snr_db`` exactly
  (modulo floating-point round-off).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.eval.scenario_record import ScenarioRecord


@dataclass(frozen=True)
class SNRConfig:
    snr_db: float
    output_dir: Path
    noise_dir: Path | None = None    # directory of noise WAVs (e.g. MUSAN/noise)
    noise_path: Path | None = None   # single noise WAV (overrides noise_dir for all records)
    seed: int = 0                    # for deterministic noise selection
    level_label: str | None = None

    @property
    def label(self) -> str:
        return self.level_label or f"snr_{self.snr_db:+.0f}db".replace("+", "p").replace("-", "m")


def _deterministic_noise_pick(noise_files: list[Path], record_id: str, seed: int) -> Path:
    h = hashlib.sha1(f"{seed}:{record_id}".encode()).digest()
    idx = int.from_bytes(h[:4], "big") % len(noise_files)
    return noise_files[idx]


def _list_noise_files(noise_dir: Path) -> list[Path]:
    files = sorted(
        list(noise_dir.rglob("*.wav"))
        + list(noise_dir.rglob("*.flac"))
        + list(noise_dir.rglob("*.WAV"))
    )
    if not files:
        raise FileNotFoundError(f"no .wav/.flac files under {noise_dir}")
    return files


def _load_audio(path: Path) -> tuple["np.ndarray", int]:
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32, copy=False), int(sr)


def _resample(samples: "np.ndarray", src_sr: int, dst_sr: int) -> "np.ndarray":
    if src_sr == dst_sr:
        return samples
    import numpy as np
    import torch
    import torchaudio.functional as F

    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    out = F.resample(waveform, src_sr, dst_sr).squeeze(0).numpy()
    return out.astype(np.float32, copy=False)


def _tile_or_crop(noise: "np.ndarray", target_len: int, rng) -> "np.ndarray":
    import numpy as np

    n = len(noise)
    if n == 0:
        return np.zeros(target_len, dtype=np.float32)
    if n >= target_len:
        start = rng.integers(0, n - target_len + 1) if n > target_len else 0
        return noise[start : start + target_len]
    # tile, with random rotation per tile to avoid deterministic seam
    reps = (target_len + n - 1) // n
    tiled = np.tile(noise, reps)[:target_len]
    return tiled


def apply_snr(
    record: TurnManifestRecord,
    config: SNRConfig,
) -> ScenarioRecord:
    """Add noise at the configured SNR and write a perturbed WAV."""

    import numpy as np
    import soundfile as sf

    speech, sr = _load_audio(Path(record.audio))
    n_samples = len(speech)
    if n_samples == 0:
        raise ValueError(f"empty audio: {record.audio}")

    # Pick noise source.
    rng = np.random.default_rng(
        seed=int(hashlib.sha1(f"{config.seed}:{record.id}".encode()).hexdigest()[:8], 16)
    )
    noise_source: str
    if config.noise_path is not None:
        noise, n_sr = _load_audio(config.noise_path)
        noise_source = f"file:{config.noise_path}"
    elif config.noise_dir is not None:
        files = _list_noise_files(Path(config.noise_dir))
        picked = _deterministic_noise_pick(files, record.id, config.seed)
        noise, n_sr = _load_audio(picked)
        noise_source = f"dir:{config.noise_dir}:{picked.name}"
    else:
        # Synthetic pink-noise fallback (1/f spectrum approximation)
        white = rng.standard_normal(n_samples).astype(np.float32)
        # one-pole pink filter
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float32)
        a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400], dtype=np.float32)
        try:
            from scipy.signal import lfilter
            noise = lfilter(b, a, white).astype(np.float32)
        except ImportError:
            noise = white  # plain white noise fallback
        n_sr = sr
        noise_source = f"synthetic:rng_seed={config.seed}"

    if n_sr != sr:
        noise = _resample(noise, n_sr, sr)
    noise = _tile_or_crop(noise, n_samples, rng)

    # Compute powers and rescale noise to hit exact SNR.
    eps = 1e-10
    p_speech = float(np.mean(speech**2)) + eps
    p_noise_raw = float(np.mean(noise**2)) + eps
    target_p_noise = p_speech / (10.0 ** (config.snr_db / 10.0))
    scale = float(np.sqrt(target_p_noise / p_noise_raw))
    mixed = speech + scale * noise

    # Avoid clipping by scaling down if needed; record the post-scaling factor
    # so the SNR is preserved (both speech and noise scale together).
    peak = float(np.max(np.abs(mixed)))
    norm_factor = 1.0
    if peak > 0.99:
        norm_factor = 0.99 / peak
        mixed = mixed * norm_factor

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record.id}__{config.label}.wav"
    sf.write(str(out_path), mixed, sr)

    return ScenarioRecord.from_record(
        record,
        audio=str(out_path),
        factor="snr",
        factor_level=config.label,
        factor_params={
            "snr_db": float(config.snr_db),
            "noise_source": noise_source,
            "noise_scale": scale,
            "post_norm_factor": norm_factor,
            "speech_power": p_speech,
            "noise_power_raw": p_noise_raw,
            "sample_rate": int(sr),
        },
        sample_rate=int(sr),
    )
