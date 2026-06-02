"""Export helpers for trained NanoTurn checkpoints."""

from __future__ import annotations

from pathlib import Path

from stable_asr.train.turn_trainer import load_nanoturn_checkpoint
from stable_asr.turn.nanoturn import require_torch, torch


def export_nanoturn_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    opset_version: int = 18,
) -> str:
    """Export a NanoTurn checkpoint to ONNX."""

    require_torch()
    model, config = load_nanoturn_checkpoint(checkpoint_path)
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_type = getattr(config, "model_type", "nanoturn_pico")
    if model_type == "nanoturn_micro":
        # TCN: input is (B, T, n_mels); use a fixed T=32 for export
        n_mels = getattr(config, "n_mels", 80)
        dummy = torch.zeros(1, 32, n_mels, dtype=torch.float32)
        input_names = ["mel_frames"]
        dynamic_axes = {"mel_frames": {0: "batch", 1: "time"}, "logits": {0: "batch"}}
    else:
        dummy = torch.zeros(1, config.input_dim, dtype=torch.float32)
        input_names = ["features"]
        dynamic_axes = {"features": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )
    return str(output_path)
