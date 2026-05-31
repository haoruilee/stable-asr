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
    dummy = torch.zeros(1, config.input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["features"],
        output_names=["logits"],
        opset_version=opset_version,
    )
    return str(output_path)
