from pathlib import Path

import pytest

from stable_asr.data.manifest import load_manifest
from stable_asr.train.turn_trainer import (
    NanoTurnCheckpointPredictor,
    train_nanoturn,
)
from stable_asr.turn.nanoturn import NanoTurnPico

torch = pytest.importorskip("torch")


def test_nanoturn_pico_forward() -> None:
    model = NanoTurnPico()
    logits = model(torch.zeros(2, 8))

    assert logits.shape == (2, 4)


def test_train_nanoturn_checkpoint(tmp_path: Path) -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    result = train_nanoturn(
        records,
        output_dir=tmp_path,
        model_type="nanoturn_pico",
        epochs=5,
        lr=0.01,
        seed=0,
    )

    assert Path(result.checkpoint_path).exists()
    assert Path(result.metrics_path).exists()
    assert result.metrics["records"] == 4
    assert "final_accuracy" in result.metrics


def test_nanoturn_checkpoint_predictor(tmp_path: Path) -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    result = train_nanoturn(records, output_dir=tmp_path, epochs=5, seed=0)
    predictor = NanoTurnCheckpointPredictor(result.checkpoint_path)

    prediction = predictor.predict(records[0])

    assert set(prediction.probs) == {"backchannel", "complete", "incomplete", "wait"}
    assert abs(sum(prediction.probs.values()) - 1.0) < 1e-5

