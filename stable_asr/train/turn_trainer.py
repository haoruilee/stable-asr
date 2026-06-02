"""NanoTurn training entry points."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.train.features import records_to_features
from stable_asr.train.framework import NanoTurnRunConfig, fit_nanoturn
from stable_asr.turn.nanoturn import (
    NanoTurnConfig,
    build_nanoturn_model,
    require_torch,
    torch,
)
from stable_asr.turn.types import TurnPrediction


@dataclass(frozen=True)
class TrainTurnResult:
    checkpoint_path: str
    metrics_path: str
    metrics: dict[str, Any]


class NanoTurnCheckpointPredictor:
    def __init__(self, checkpoint_path: str | Path, *, audio_root: str | Path | None = None) -> None:
        model, config = load_nanoturn_checkpoint(checkpoint_path)
        self.model = model
        self.config = config
        self.audio_root = audio_root
        self.model.eval()
        self.is_sequence = getattr(config, "model_type", "") == "nanoturn_micro"

    def predict(self, record: TurnManifestRecord) -> TurnPrediction:
        require_torch()
        with torch.no_grad():
            features = records_to_features(
                [record],
                feature_source=self.config.feature_source,
                audio_root=self.audio_root,
            )
            if self.is_sequence:
                # features is a list of (T, n_mels) tensors; stack with batch dim
                from stable_asr.train.framework import _sequence_collate_fn
                import torch as _t
                batch, _ = _sequence_collate_fn([(features[0], _t.tensor(0))])
                batch = batch.to(next(self.model.parameters()).device)
                logits = self.model(batch)
            else:
                batch = features.to(next(self.model.parameters()).device)
                logits = self.model(batch)
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return TurnPrediction(
            probs={label: float(probs[index]) for index, label in enumerate(self.config.labels)},
            timestamp=record.end,
        )


def train_nanoturn(
    records: list[TurnManifestRecord],
    *,
    output_dir: str | Path,
    model_type: str = "nanoturn_pico",
    epochs: int = 100,
    lr: float = 1e-2,
    seed: int = 0,
    feature_source: str = "metadata",
    audio_root: str | Path | None = None,
    feature_cache: str | Path | None = None,
    feature_cache_format: str | None = None,
    feature_cache_mode: str = "auto",
    val_records: list[TurnManifestRecord] | None = None,
    batch_size: int = 128,
    validation_split: float = 0.0,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    gradient_clip_norm: float | None = None,
    checkpoint_interval: int = 1,
    resume_from: str | Path | None = None,
    device: str = "auto",
    validation_group_by: str | None = "auto",
    tensorboard_log_dir: str | Path | None = None,
) -> TrainTurnResult:
    require_torch()
    if not records:
        raise ValueError("records must not be empty")

    _seed_everything(seed)
    config = NanoTurnRunConfig(
        model_type=model_type,
        epochs=epochs,
        lr=lr,
        seed=seed,
        feature_source=feature_source,
        batch_size=batch_size,
        validation_split=validation_split,
        optimizer=optimizer,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        checkpoint_interval=checkpoint_interval,
        device=device,
        feature_cache=str(feature_cache) if feature_cache else None,
        feature_cache_format=feature_cache_format,
        feature_cache_mode=feature_cache_mode,
        audio_root=str(audio_root) if audio_root else None,
        resume_from=str(resume_from) if resume_from else None,
        validation_group_by=validation_group_by,
        tensorboard_log_dir=str(tensorboard_log_dir) if tensorboard_log_dir else None,
    )
    result = fit_nanoturn(records, output_dir=output_dir, config=config, val_records=val_records)
    return TrainTurnResult(
        checkpoint_path=result.artifacts.checkpoint_path,
        metrics_path=result.artifacts.metrics_path,
        metrics=result.metrics,
    )


def load_nanoturn_checkpoint(checkpoint_path: str | Path):
    """Load a NanoTurn checkpoint (MLP or Micro) and return (model, config)."""
    require_torch()
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = payload["config"]
    model_type = str(cfg_dict.get("model_type", "nanoturn_pico"))
    if model_type == "nanoturn_micro":
        from stable_asr.turn.nanoturn_micro import NanoTurnMicroConfig, NanoTurnMicro
        config = NanoTurnMicroConfig.from_dict(cfg_dict)
        from stable_asr.turn.nanoturn_micro import build_nanoturn_micro
        model = build_nanoturn_micro(
            labels=config.labels,
            n_mels=config.n_mels,
            hidden_dim=config.hidden_dim,
            n_blocks=config.n_blocks,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        model.load_state_dict(payload["state_dict"])
        return model, config
    config = NanoTurnConfig.from_dict(cfg_dict)
    model = build_nanoturn_model(
        config.model_type,
        labels=config.labels,
        input_dim=config.input_dim,
    )
    model.load_state_dict(payload["state_dict"])
    return model, config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
