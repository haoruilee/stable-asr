import json
from pathlib import Path

import pytest

from stable_asr.data.manifest import load_manifest
from stable_asr.train.framework import NanoTurnRunConfig, fit_nanoturn
from stable_asr.train.turn_trainer import train_nanoturn

pytest.importorskip("torch")


def test_fit_nanoturn_writes_run_artifacts(tmp_path: Path) -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    result = fit_nanoturn(
        records,
        output_dir=tmp_path,
        config=NanoTurnRunConfig(
            epochs=2,
            seed=0,
            batch_size=2,
            validation_split=0.25,
            optimizer="adamw",
            checkpoint_interval=1,
        ),
    )

    assert Path(result.artifacts.config_path).exists()
    assert Path(result.artifacts.checkpoint_path).exists()
    assert Path(result.artifacts.best_checkpoint_path).exists()
    assert Path(result.artifacts.history_path).exists()
    assert Path(result.artifacts.summary_path).exists()
    assert (tmp_path / "checkpoints" / "weights_epoch_1.pt").exists()
    assert result.metrics["train_records"] == 3
    assert result.metrics["val_records"] == 1
    assert result.metrics["final_val_accuracy"] is not None

    run_config = json.loads(Path(result.artifacts.config_path).read_text(encoding="utf-8"))
    assert run_config["framework"] == "stable_asr.nanoturn_trainer.v1"
    assert run_config["config"]["optimizer"] == "adamw"


def test_explicit_validation_records_disable_random_split(tmp_path: Path) -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    result = fit_nanoturn(
        records[1:],
        val_records=records[:1],
        output_dir=tmp_path,
        config=NanoTurnRunConfig(epochs=1, validation_split=0.5, batch_size=2),
    )

    assert result.metrics["records"] == 4
    assert result.metrics["train_records"] == 3
    assert result.metrics["val_records"] == 1


def test_train_nanoturn_resume_from_checkpoint(tmp_path: Path) -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    first = train_nanoturn(
        records,
        output_dir=tmp_path,
        epochs=1,
        seed=1,
        batch_size=2,
        checkpoint_interval=1,
    )
    resumed = train_nanoturn(
        records,
        output_dir=tmp_path,
        epochs=3,
        seed=1,
        batch_size=2,
        checkpoint_interval=1,
        resume_from=first.checkpoint_path,
    )

    assert Path(resumed.checkpoint_path).exists()
    assert resumed.metrics["epochs"] == 3
    assert resumed.metrics["history"][-1]["epoch"] == 3.0
    assert (tmp_path / "checkpoints" / "weights_epoch_3.pt").exists()
