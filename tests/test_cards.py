from pathlib import Path

from stable_asr.paper.cards import dataset_card, experiment_card
from stable_asr.paper.experiments import run_paper_smoke


def test_dataset_card(tmp_path: Path) -> None:
    output = tmp_path / "DATASET_CARD.md"

    dataset_card("examples/data/turn_demo.jsonl", output)

    text = output.read_text(encoding="utf-8")
    assert "Stable-ASR Dataset Card" in text
    assert "Turn Labels" in text


def test_experiment_card(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=6, seed=1, train_model=False)
    output = tmp_path / "EXPERIMENT_CARD.md"

    experiment_card(result.results_path, output)

    text = output.read_text(encoding="utf-8")
    assert "Stable-ASR Experiment Card" in text
    assert "Streaming ASR" in text

