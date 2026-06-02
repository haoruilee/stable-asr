"""NanoTurnMicro: TCN sequence model for streaming turn-taking.

Unlike the NanoTurnMLP family (which operates on a single fixed-size feature
vector per utterance), NanoTurnMicro operates on a variable-length sequence of
log-mel frames (T, n_mels). A stack of dilated causal temporal convolutions
captures short- and long-range dynamics without requiring the audio to be
summarised into a single vector beforehand.

Architecture:
  - Input projection: Linear(n_mels → hidden_dim)
  - N TCN residual blocks with exponentially growing dilation
  - Global average pooling over time
  - Output head: Linear(hidden_dim → n_classes)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from stable_asr.turn.labels import TURN_LABELS
from stable_asr.turn.nanoturn import require_torch, torch

try:
    from torch import nn
except Exception:  # pragma: no cover
    nn = None

DEFAULT_LABELS = tuple(sorted(TURN_LABELS))


@dataclass(frozen=True)
class NanoTurnMicroConfig:
    n_mels: int = 80
    hidden_dim: int = 64
    n_blocks: int = 4
    kernel_size: int = 3
    dropout: float = 0.1
    labels: tuple[str, ...] = DEFAULT_LABELS
    model_type: str = "nanoturn_micro"
    feature_source: str = "audio_seq"
    # depthwise separable convolution: ~kernel_size× fewer FLOPs per block,
    # same receptive field. Reduces model size and speeds inference on CPU.
    # Does NOT change the output shape or interface. Safe for ablation.
    depthwise: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = list(self.labels)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NanoTurnMicroConfig":
        values = dict(data)
        if "labels" in values:
            values["labels"] = tuple(values["labels"])
        return cls(**values)

    @property
    def input_dim(self) -> int:
        return self.n_mels


class _CausalConv1d(nn.Module if nn is not None else object):
    """Causal convolution: pads left so output[t] depends only on input[:t+1].

    depthwise=True uses depthwise-separable convolution (depthwise + pointwise),
    which reduces FLOPs by ~kernel_size× at the same receptive field.
    Numerically equivalent in terms of what can be learned; slightly different
    weight count. Safe to use for speed experiments.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        if depthwise and in_channels == out_channels:
            # Depthwise separable: depthwise conv + pointwise 1×1
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels, in_channels, kernel_size,
                    dilation=dilation, padding=0, groups=in_channels,
                ),
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                dilation=dilation, padding=0,
            )

    def forward(self, x):
        # x: (B, C, T)
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class _TCNBlock(nn.Module if nn is not None else object):
    """Residual TCN block with two dilated causal convolutions."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation, depthwise=depthwise)
        self.conv2 = _CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation, depthwise=depthwise)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, hidden_dim, T)
        residual = x
        # conv1
        h = self.conv1(x)                          # (B, H, T)
        h = h.transpose(1, 2)                      # (B, T, H) for LayerNorm
        h = self.norm1(h).transpose(1, 2)          # back to (B, H, T)
        h = nn.functional.gelu(h)
        h = self.drop(h)
        # conv2
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.norm2(h).transpose(1, 2)
        h = nn.functional.gelu(h)
        h = self.drop(h)
        return h + residual


class NanoTurnMicro(nn.Module if nn is not None else object):
    """TCN-based sequence model for frame-level turn prediction.

    Input:  (B, T, n_mels) log-mel spectrogram frames
    Output: (B, n_classes) logits (utterance-level, via global avg pool)
    """

    def __init__(self, config: NanoTurnMicroConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.n_mels, config.hidden_dim)
        self.blocks = nn.ModuleList([
            _TCNBlock(
                hidden_dim=config.hidden_dim,
                kernel_size=config.kernel_size,
                dilation=2 ** i,
                dropout=config.dropout,
                depthwise=config.depthwise,
            )
            for i in range(config.n_blocks)
        ])
        self.output_head = nn.Linear(config.hidden_dim, len(config.labels))

    @property
    def labels(self) -> tuple[str, ...]:
        return self.config.labels

    def forward(self, x):
        # x: (B, T, n_mels)
        h = self.input_proj(x)          # (B, T, hidden_dim)
        h = h.transpose(1, 2)           # (B, hidden_dim, T)
        for block in self.blocks:
            h = block(h)
        h = h.mean(dim=2)               # global avg pool: (B, hidden_dim)
        return self.output_head(h)      # (B, n_classes)


def build_nanoturn_micro(
    labels: tuple[str, ...] = DEFAULT_LABELS,
    n_mels: int = 80,
    hidden_dim: int = 64,
    n_blocks: int = 4,
    kernel_size: int = 3,
    dropout: float = 0.1,
    depthwise: bool = False,
) -> NanoTurnMicro:
    config = NanoTurnMicroConfig(
        n_mels=n_mels,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        kernel_size=kernel_size,
        dropout=dropout,
        labels=labels,
        model_type="nanoturn_micro",
        feature_source="audio_seq",
        depthwise=depthwise,
    )
    return NanoTurnMicro(config)
