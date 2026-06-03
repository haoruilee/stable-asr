"""F5 — controlled-overlap two-speaker mixing.

Mixes a target speaker's clip with a competing speaker's clip such that
exactly ``overlap_ratio`` of the target's duration is covered by the
competing speaker. The competing audio is drawn from a pool (typically
the same dataset, different speaker) and amplitude-aligned to the target
so the overlap ratio is the only varying factor.

Definition (the one we report in the paper):

    overlap_ratio = (samples where both speakers active) / (target duration)

Layout:

    target   |==================================|
    other         |======|       |======|
                  ^^^^^^^         ^^^^^^^

If ``overlap_ratio`` is e.g. 0.30, two non-contiguous overlap windows of
combined length ``0.30 * target_dur`` are placed inside the target clip.
The other speaker is silent outside those windows. Mixing is simple
linear addition with per-source rescale to maintain the target speaker's
RMS as the dominant.

Caveats:

* This is *synthetic overlap*. Real overlap (AMI multi-channel, LDC
  Mixer) carries channel and acoustic effects this won't reproduce.
  Document in the paper Methods that synthetic-overlap results are
  controllable but not naturalistic; LDC channels supply the
  naturalistic complement.
* The competing speaker pool must contain audio at the same sample rate
  as the target, or it gets resampled.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.eval.scenario_record import ScenarioRecord


@dataclass(frozen=True)
class OverlapConfig:
    overlap_ratio: float       # 0.0–1.0
    output_dir: Path
    competitor_pool: Path      # directory of competing speaker audio
    target_dominance_db: float = 3.0  # target speaker is +3 dB louder than competitor
    n_overlap_windows: int = 2  # break the overlap into N windows
    seed: int = 0
    level_label: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.overlap_ratio <= 1.0:
            raise ValueError(f"overlap_ratio {self.overlap_ratio} out of [0, 1]")

    @property
    def label(self) -> str:
        return self.level_label or f"overlap_{self.overlap_ratio:.2f}"


def _list_audio(pool: Path) -> list[Path]:
    files = sorted(
        list(pool.rglob("*.wav"))
        + list(pool.rglob("*.flac"))
        + list(pool.rglob("*.WAV"))
        + list(pool.rglob("*.FLAC"))
    )
    if not files:
        raise FileNotFoundError(f"no audio files under {pool}")
    return files


def _load(path: Path) -> tuple["np.ndarray", int]:
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32, copy=False), int(sr)


def _resample(samples: "np.ndarray", src_sr: int, dst_sr: int) -> "np.ndarray":
    if src_sr == dst_sr:
        return samples
    import torch
    import torchaudio.functional as F

    wf = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    return F.resample(wf, src_sr, dst_sr).squeeze(0).numpy().astype("float32")


def apply_overlap(
    record: TurnManifestRecord,
    config: OverlapConfig,
    *,
    competitor_blocklist: set[str] | None = None,
) -> ScenarioRecord:
    """Mix the record's audio with a competing speaker at controlled overlap.

    ``competitor_blocklist`` (optional) holds record ids or audio basenames
    that must not be drawn as competitors (e.g. to avoid mixing the same
    speaker with themselves). Pass a set of disallowed substrings.
    """

    import numpy as np
    import soundfile as sf

    target, sr = _load(Path(record.audio))
    target_len = len(target)
    if target_len == 0:
        raise ValueError(f"empty target audio: {record.audio}")

    rng = np.random.default_rng(
        seed=int(hashlib.sha1(f"{config.seed}:{record.id}".encode()).hexdigest()[:8], 16)
    )

    overlap_samples = int(round(config.overlap_ratio * target_len))
    if overlap_samples == 0 or config.n_overlap_windows == 0:
        # Degenerate case: no overlap. Still write a copy so downstream
        # treats this consistently as a ScenarioRecord with overlap=0.
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{record.id}__{config.label}.wav"
        sf.write(str(out_path), target, sr)
        return ScenarioRecord.from_record(
            record,
            audio=str(out_path),
            factor="overlap",
            factor_level=config.label,
            factor_params={
                "overlap_ratio": float(config.overlap_ratio),
                "actual_overlap_ratio": 0.0,
                "n_overlap_windows": int(config.n_overlap_windows),
                "actual_overlap_samples": 0,
                "competitor": None,
                "sample_rate": int(sr),
            },
            sample_rate=int(sr),
        )

    # Pick a competitor. Avoid the target's own audio.
    pool = _list_audio(Path(config.competitor_pool))
    blocked = set(competitor_blocklist or set())
    blocked.add(Path(record.audio).name)
    blocked.add(record.id)

    # Try a few picks until we find one not in the blocklist.
    competitor_path: Path | None = None
    for _ in range(16):
        idx = int(rng.integers(0, len(pool)))
        candidate = pool[idx]
        if candidate.name in blocked or any(b in candidate.name for b in blocked):
            continue
        competitor_path = candidate
        break
    if competitor_path is None:
        competitor_path = pool[int(rng.integers(0, len(pool)))]

    competitor, c_sr = _load(competitor_path)
    if c_sr != sr:
        competitor = _resample(competitor, c_sr, sr)

    if len(competitor) == 0:
        competitor = np.zeros(target_len, dtype="float32")

    # Layout the overlap windows.
    n_windows = max(1, int(config.n_overlap_windows))
    per_window = overlap_samples // n_windows
    if per_window <= 0:
        per_window = overlap_samples
        n_windows = 1

    # Tile competitor to per_window length pieces with random offsets.
    pieces: list[tuple[int, "np.ndarray"]] = []  # (target-start, piece)
    cursor = 0
    gap_total = max(0, target_len - per_window * n_windows)
    gap_each = gap_total // (n_windows + 1) if n_windows > 0 else 0
    for w in range(n_windows):
        # start position of this overlap window inside the target
        start = gap_each * (w + 1) + per_window * w
        # source position inside competitor
        if len(competitor) > per_window:
            src_start = int(rng.integers(0, len(competitor) - per_window + 1))
            piece = competitor[src_start : src_start + per_window]
        else:
            # short competitor: tile to fill
            reps = (per_window + len(competitor) - 1) // max(1, len(competitor))
            piece = np.tile(competitor, reps)[:per_window]
        pieces.append((start, piece.astype("float32")))
        cursor = start + per_window

    # Loudness model: scale competitor so target is +target_dominance_db louder.
    eps = 1e-10
    p_target = float(np.mean(target**2)) + eps
    competitor_concat = np.concatenate([p[1] for p in pieces])
    p_competitor = float(np.mean(competitor_concat**2)) + eps
    target_p_competitor = p_target / (10.0 ** (config.target_dominance_db / 10.0))
    scale = float(np.sqrt(target_p_competitor / p_competitor))

    mixed = target.copy()
    actual_overlap = 0
    for start, piece in pieces:
        end = min(start + len(piece), target_len)
        actual = end - start
        mixed[start:end] = mixed[start:end] + scale * piece[:actual]
        actual_overlap += actual

    peak = float(np.max(np.abs(mixed)))
    norm = 1.0
    if peak > 0.99:
        norm = 0.99 / peak
        mixed = mixed * norm

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record.id}__{config.label}.wav"
    sf.write(str(out_path), mixed, sr)

    return ScenarioRecord.from_record(
        record,
        audio=str(out_path),
        factor="overlap",
        factor_level=config.label,
        factor_params={
            "overlap_ratio": float(config.overlap_ratio),
            "actual_overlap_ratio": float(actual_overlap / target_len),
            "n_overlap_windows": int(n_windows),
            "competitor": str(competitor_path),
            "competitor_scale": scale,
            "post_norm_factor": norm,
            "target_dominance_db": float(config.target_dominance_db),
            "sample_rate": int(sr),
        },
        sample_rate=int(sr),
    )
