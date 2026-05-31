from pathlib import Path

from stable_asr.paper.cards import dataset_card, experiment_card, model_card
from stable_asr.paper.experiments import run_paper_smoke


def test_dataset_card(tmp_path: Path) -> None:
    output = tmp_path / "DATASET_CARD.md"

    dataset_card("examples/data/turn_demo.jsonl", output)

    text = output.read_text(encoding="utf-8")
    assert "Stable-ASR Dataset Card" in text
    assert "Turn Labels" in text


def test_dataset_card_resolves_platform_asset_root(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "DATASET_CARD.md"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STABLE_ASR_ASSET_ROOT", str(Path(__file__).resolve().parents[1]))

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


def test_model_card_from_registry(tmp_path: Path) -> None:
    output = tmp_path / "MODEL_CARD.md"

    model_card("configs/models/stable_asr_models.json", output, model_id="nanoturn_pico")

    text = output.read_text(encoding="utf-8")
    assert "Stable-ASR Model Card: NanoTurn Pico" in text
    assert "TurnPredictor" in text
    assert "Limitations" in text


def test_model_card_from_config_with_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        (
            '{"model_type":"nanoturn_pico","records":4,"epochs":2,'
            '"lr":0.01,"seed":0,"feature_source":"metadata","final_accuracy":0.5}'
        ),
        encoding="utf-8",
    )
    output = tmp_path / "MODEL_CARD.md"

    model_card("configs/nanoturn_pico.json", output, metrics_path=metrics)

    text = output.read_text(encoding="utf-8")
    assert "Stable-ASR Model Card: Nanoturn Pico" in text
    assert "final_accuracy" in text
    assert "metadata" in text
