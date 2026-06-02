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
    """Run NanoTurn inference via PyTorch (default) or ONNX Runtime.

    PyTorch path: standard torch.no_grad() forward pass.
    ONNX path: uses onnxruntime for ~2-5x faster CPU inference at identical
               numerical output (ORT runs the same computation graph).

    Parameters
    ----------
    checkpoint_path:
        Path to a .pt checkpoint saved by NanoTurnTrainer.
    onnx_path:
        Optional path to a .onnx export. When provided and onnxruntime is
        installed, inference uses ORT instead of PyTorch. Produces identical
        predictions; ORT is typically 2-5x faster on CPU.
    audio_root:
        Base directory for relative audio paths.
    batch_size:
        Number of records to process per forward pass in predict_batch().
        predict() always uses batch_size=1.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        onnx_path: str | Path | None = None,
        audio_root: str | Path | None = None,
        batch_size: int = 256,
    ) -> None:
        model, config = load_nanoturn_checkpoint(checkpoint_path)
        self.model = model
        self.config = config
        self.audio_root = audio_root
        self.batch_size = batch_size
        self.model.eval()
        self.is_sequence = getattr(config, "model_type", "") == "nanoturn_micro"
        self._ort_session = None
        if onnx_path is not None:
            self._ort_session = _load_ort_session(onnx_path)

    def predict(self, record: TurnManifestRecord) -> TurnPrediction:
        return self.predict_batch([record])[0]

    def predict_batch(self, records: list[TurnManifestRecord]) -> list[TurnPrediction]:
        """Run batched inference over a list of records.

        For non-sequence models (MLP): all records processed in one forward pass.
        For sequence models (TCN): records padded and processed in mini-batches
        of self.batch_size to bound memory usage.
        """
        require_torch()
        features = records_to_features(
            records,
            feature_source=self.config.feature_source,
            audio_root=self.audio_root,
        )

        if self._ort_session is not None:
            return self._predict_ort(records, features)

        with torch.no_grad():
            if self.is_sequence:
                return self._predict_sequence_batched(records, features)
            # MLP: features is a (N, D) tensor — single forward pass
            device = next(self.model.parameters()).device
            batch = features.to(device)
            logits = self.model(batch)
            probs_all = torch.softmax(logits, dim=-1).cpu().tolist()
        return [
            TurnPrediction(
                probs={label: float(p) for label, p in zip(self.config.labels, probs)},
                timestamp=record.end,
            )
            for record, probs in zip(records, probs_all)
        ]

    def _predict_sequence_batched(
        self,
        records: list[TurnManifestRecord],
        features: list,
    ) -> list[TurnPrediction]:
        """Batched TCN inference with padding."""
        from stable_asr.train.framework import _sequence_collate_fn
        device = next(self.model.parameters()).device
        results = []
        for i in range(0, len(features), self.batch_size):
            chunk_feats = features[i: i + self.batch_size]
            chunk_records = records[i: i + self.batch_size]
            dummy_targets = [torch.tensor(0)] * len(chunk_feats)
            padded, _ = _sequence_collate_fn(list(zip(chunk_feats, dummy_targets)))
            padded = padded.to(device)
            logits = self.model(padded)
            probs_all = torch.softmax(logits, dim=-1).cpu().tolist()
            for record, probs in zip(chunk_records, probs_all):
                results.append(TurnPrediction(
                    probs={label: float(p) for label, p in zip(self.config.labels, probs)},
                    timestamp=record.end,
                ))
        return results

    def _predict_ort(
        self,
        records: list[TurnManifestRecord],
        features,
    ) -> list[TurnPrediction]:
        """Inference via ONNX Runtime (CPU/CUDA, identical to PyTorch)."""
        import numpy as np

        if self.is_sequence:
            # ORT sequence: run one-by-one (variable length; batching would
            # require padding which is already done in the PyTorch path)
            results = []
            for record, feat in zip(records, features):
                x = feat.unsqueeze(0).numpy().astype(np.float32)  # (1, T, n_mels)
                logits = self._ort_session.run(None, {"mel_frames": x})[0]
                probs = _softmax(logits[0])
                results.append(TurnPrediction(
                    probs={label: float(p) for label, p in zip(self.config.labels, probs)},
                    timestamp=record.end,
                ))
            return results
        # MLP: batch all at once
        x = features.numpy().astype(np.float32)  # (N, D)
        logits = self._ort_session.run(None, {"features": x})[0]  # (N, C)
        return [
            TurnPrediction(
                probs={label: float(p) for label, p in zip(self.config.labels, _softmax(logits[i]))},
                timestamp=record.end,
            )
            for i, record in enumerate(records)
        ]


def _softmax(x):
    import numpy as np
    e = np.exp(x - x.max())
    return e / e.sum()


def _load_ort_session(onnx_path: str | Path):
    """Load an ONNX Runtime InferenceSession with best available provider."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime inference requires the onnxruntime package.\n"
            "Install: pip install onnxruntime   (CPU)\n"
            "      or pip install onnxruntime-gpu (CUDA)"
        ) from exc
    providers = ort.get_available_providers()
    # Prefer CUDA > TensorRT > CPU
    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    ordered = [p for p in preferred if p in providers]
    session = ort.InferenceSession(str(onnx_path), providers=ordered)
    return session


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
    # acceleration flags
    amp: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    lr_schedule: str | None = None,
    lr_min: float = 1e-6,
    early_stopping_patience: int | None = None,
    depthwise: bool = False,
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
        amp=amp,
        num_workers=num_workers,
        pin_memory=pin_memory,
        lr_schedule=lr_schedule,
        lr_min=lr_min,
        early_stopping_patience=early_stopping_patience,
        depthwise=depthwise,
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
