"""NanoTurn baseline model definitions.

The first implementation is intentionally small and dependency-light at import
time. It uses metadata-derived feature vectors for v0 training/evaluation so the
platform can exercise the full train/eval/checkpoint path before the audio
frontend lands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from stable_asr.turn.labels import TURN_LABELS

try:  # pragma: no cover - exercised in torch-enabled environments.
    import torch
    from torch import nn
except Exception:  # pragma: no cover - import guard for base installs.
    torch = None
    nn = None


DEFAULT_LABELS = tuple(sorted(TURN_LABELS))


@dataclass(frozen=True)
class NanoTurnConfig:
    input_dim: int = 8
    hidden_dim: int = 32
    depth: int = 2
    dropout: float = 0.0
    labels: tuple[str, ...] = DEFAULT_LABELS
    model_type: str = "nanoturn_pico"
    feature_source: str = "metadata"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = list(self.labels)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NanoTurnConfig":
        values = dict(data)
        if "labels" in values:
            values["labels"] = tuple(values["labels"])
        return cls(**values)


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("NanoTurn training requires PyTorch. Install with: pip install 'stable-asr[train]'")


class NanoTurnMLP(nn.Module if nn is not None else object):
    """Small MLP classifier used by NanoTurnPico/NanoTurnNano v0."""

    def __init__(self, config: NanoTurnConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.depth):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, len(config.labels)))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)

    @property
    def labels(self) -> tuple[str, ...]:
        return self.config.labels


def NanoTurnPico(labels: tuple[str, ...] = DEFAULT_LABELS, input_dim: int = 8) -> NanoTurnMLP:
    config = NanoTurnConfig(
        input_dim=input_dim,
        hidden_dim=16,
        depth=1,
        dropout=0.0,
        labels=labels,
        model_type="nanoturn_pico",
    )
    return NanoTurnMLP(config)


def NanoTurnNano(labels: tuple[str, ...] = DEFAULT_LABELS, input_dim: int = 8) -> NanoTurnMLP:
    config = NanoTurnConfig(
        input_dim=input_dim,
        hidden_dim=64,
        depth=2,
        dropout=0.05,
        labels=labels,
        model_type="nanoturn_nano",
    )
    return NanoTurnMLP(config)


def NanoTurnPicoV1(labels: tuple[str, ...] = DEFAULT_LABELS, input_dim: int = 160) -> NanoTurnMLP:
    """NanoTurnPico trained on logmel_v1 160-dim features."""
    config = NanoTurnConfig(
        input_dim=input_dim,
        hidden_dim=64,
        depth=2,
        dropout=0.05,
        labels=labels,
        model_type="nanoturn_pico_v1",
        feature_source="audio_v1",
    )
    return NanoTurnMLP(config)


def NanoTurnNanoV1(labels: tuple[str, ...] = DEFAULT_LABELS, input_dim: int = 160) -> NanoTurnMLP:
    """NanoTurnNano trained on logmel_v1 160-dim features."""
    config = NanoTurnConfig(
        input_dim=input_dim,
        hidden_dim=256,
        depth=3,
        dropout=0.1,
        labels=labels,
        model_type="nanoturn_nano_v1",
        feature_source="audio_v1",
    )
    return NanoTurnMLP(config)


def build_nanoturn_model(model_type: str, *, labels: tuple[str, ...] = DEFAULT_LABELS, input_dim: int = 8) -> NanoTurnMLP:
    if model_type == "nanoturn_pico":
        return NanoTurnPico(labels=labels, input_dim=input_dim)
    if model_type == "nanoturn_nano":
        return NanoTurnNano(labels=labels, input_dim=input_dim)
    if model_type == "nanoturn_pico_v1":
        return NanoTurnPicoV1(labels=labels, input_dim=input_dim)
    if model_type == "nanoturn_nano_v1":
        return NanoTurnNanoV1(labels=labels, input_dim=input_dim)
    raise ValueError(f"unknown NanoTurn model type: {model_type}")
