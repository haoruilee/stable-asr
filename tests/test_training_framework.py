import json
from pathlib import Path

import pytest

from stable_asr.data.manifest import load_manifest
from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.train import framework as train_framework
from stable_asr.train.framework import NanoTurnRunConfig, _split_validation, fit_nanoturn
from stable_asr.train.turn_trainer import train_nanoturn

torch = pytest.importorskip("torch")


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
    assert run_config["config"]["validation_group_by"] == "auto"


def test_fit_nanoturn_writes_tensorboard_scalars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writers = []

    class DummyWriter:
        def __init__(self) -> None:
            self.scalars = []
            self.text = []
            self.closed = False

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars.append((tag, value, step))

        def add_text(self, tag: str, text: str, global_step: int = 0) -> None:
            self.text.append((tag, text, global_step))

        def close(self) -> None:
            self.closed = True

    def make_writer(log_dir: Path) -> DummyWriter:
        assert log_dir == tmp_path / "tb"
        writer = DummyWriter()
        writers.append(writer)
        return writer

    monkeypatch.setattr(train_framework, "_make_tensorboard_writer", make_writer)
    records = load_manifest("examples/data/turn_demo.jsonl")
    result = fit_nanoturn(
        records,
        output_dir=tmp_path,
        config=NanoTurnRunConfig(epochs=1, batch_size=2, tensorboard_log_dir="tb"),
    )

    assert result.metrics["tensorboard_log_dir"] == str(tmp_path / "tb")
    assert writers[0].closed
    assert ("train/loss", result.metrics["final_train_loss"], 1) in writers[0].scalars
    assert any(tag == "run/config" for tag, _, _ in writers[0].text)


def test_internal_validation_split_keeps_asr_windows_grouped() -> None:
    records = []
    for index in range(8):
        asr_id = f"utt_{index // 2}"
        records.append(
            TurnManifestRecord.from_dict(
                {
                    "id": f"{asr_id}_{index % 2}",
                    "audio": f"{asr_id}.wav",
                    "sample_rate": 16000,
                    "start": 0.0,
                    "end": 1.0,
                    "turn_label": "complete" if index % 2 == 0 else "incomplete",
                    "action_label": "take_turn" if index % 2 == 0 else "keep_listening",
                    "assistant_speaking": False,
                    "overlap": False,
                    "language": "en",
                    "source": "unit",
                    "metadata": {"asr_record_id": asr_id},
                }
            )
        )

    train, val = _split_validation(
        records,
        config=NanoTurnRunConfig(validation_split=0.25, seed=3),
    )

    train_groups = {record.metadata["asr_record_id"] for record in train}
    val_groups = {record.metadata["asr_record_id"] for record in val}
    assert train_groups.isdisjoint(val_groups)
    assert len(val) == 2


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


def test_train_nanoturn_resume_matches_uninterrupted_epoch_run(tmp_path: Path) -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"

    uninterrupted = train_nanoturn(
        records,
        output_dir=uninterrupted_dir,
        epochs=3,
        seed=11,
        batch_size=2,
        checkpoint_interval=1,
        optimizer="adamw",
    )
    first = train_nanoturn(
        records,
        output_dir=resumed_dir,
        epochs=1,
        seed=11,
        batch_size=2,
        checkpoint_interval=1,
        optimizer="adamw",
    )
    resumed = train_nanoturn(
        records,
        output_dir=resumed_dir,
        epochs=3,
        seed=11,
        batch_size=2,
        checkpoint_interval=1,
        optimizer="adamw",
        resume_from=first.checkpoint_path,
    )

    uninterrupted_state = torch.load(uninterrupted.checkpoint_path, map_location="cpu")["state_dict"]
    resumed_state = torch.load(resumed.checkpoint_path, map_location="cpu")["state_dict"]
    assert uninterrupted.metrics["history"] == resumed.metrics["history"]
    for key, tensor in uninterrupted_state.items():
        assert torch.equal(tensor, resumed_state[key])
