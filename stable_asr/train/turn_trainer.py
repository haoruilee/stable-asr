"""Minimal NanoTurn training loop for v0 reproducibility."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.train.features import FEATURE_NAMES, feature_names, normalize_feature_source, records_to_features
from stable_asr.turn.nanoturn import (
    DEFAULT_LABELS,
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

    def predict(self, record: TurnManifestRecord) -> TurnPrediction:
        require_torch()
        with torch.no_grad():
            features = records_to_features(
                [record],
                feature_source=self.config.feature_source,
                audio_root=self.audio_root,
            ).to(next(self.model.parameters()).device)
            logits = self.model(features)
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
) -> TrainTurnResult:
    require_torch()
    if not records:
        raise ValueError("records must not be empty")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    _seed_everything(seed)
    feature_source = normalize_feature_source(feature_source)
    labels = DEFAULT_LABELS
    names = feature_names(feature_source)
    model = build_nanoturn_model(model_type, labels=labels, input_dim=len(names))
    model.config = NanoTurnConfig(
        input_dim=len(names),
        hidden_dim=model.config.hidden_dim,
        depth=model.config.depth,
        dropout=model.config.dropout,
        labels=model.config.labels,
        model_type=model.config.model_type,
        feature_source=feature_source,
    )
    features = records_to_features(records, feature_source=feature_source, audio_root=audio_root)
    targets = torch.tensor([labels.index(record.turn_label) for record in records], dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
        history.append({"epoch": epoch, "loss": float(loss.item()), "accuracy": float(accuracy)})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "metrics.json"

    final_metrics = {
        "model_type": model_type,
        "records": len(records),
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "feature_source": feature_source,
        "feature_names": list(names),
        "labels": list(labels),
        "final_loss": history[-1]["loss"],
        "final_accuracy": history[-1]["accuracy"],
        "history": history,
    }
    torch.save(
        {
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "metrics": final_metrics,
        },
        checkpoint_path,
    )
    metrics_path.write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainTurnResult(
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        metrics=final_metrics,
    )


def load_nanoturn_checkpoint(checkpoint_path: str | Path):
    require_torch()
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = NanoTurnConfig.from_dict(payload["config"])
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
