from pathlib import Path

import pytest

from stable_asr.data.manifest import load_manifest
from stable_asr.train.export import export_nanoturn_onnx
from stable_asr.train.turn_trainer import train_nanoturn

pytest.importorskip("torch")
pytest.importorskip("onnx")


def test_export_nanoturn_onnx(tmp_path: Path) -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    result = train_nanoturn(records, output_dir=tmp_path / "run", epochs=3, seed=0)
    output = tmp_path / "nanoturn.onnx"

    export_nanoturn_onnx(result.checkpoint_path, output)

    assert output.exists()
    assert output.stat().st_size > 0

