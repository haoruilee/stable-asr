from pathlib import Path

import pytest

from stable_asr.paper.experiments import run_paper_smoke


def test_run_paper_smoke_skip_train(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path, episodes=8, seed=3, train_model=False)

    assert Path(result.results_path).exists()
    assert Path(result.report_path).exists()
    assert result.results["meta"]["episodes"] == 8
    assert "rule_endpoint" in result.results["baselines"]
    assert "text_turn" in result.results["baselines"]
    assert "prediction_manifest" in result.results["baselines"]
    assert "turn_benchmarks" in result.results
    assert "prediction_manifest" in result.results["turn_benchmarks"]
    assert "benchmark" in result.results["data"]
    assert "turn_prediction_fixture_path" in result.results["data"]
    assert result.results["data"]["external_conversion"]["records"] == 2
    assert len(result.results["data"]["external_conversions"]) == 3
    assert {item["schema"] for item in result.results["data"]["external_conversions"]} == {
        "easyturn",
        "full_duplex_bench",
        "smart_turn",
    }
    assert "by_scenario" in result.results["scenarios"]
    assert "best" in result.results["policy_search"]
    assert result.results["streaming_asr"]["metrics"]["records"] == 2
    assert len(result.results["streaming_asr"]["asr_transcript_conversions"]) == 4
    assert {item["schema"] for item in result.results["streaming_asr"]["asr_transcript_conversions"]} == {
        "whisper",
        "funasr",
        "qwen3_asr",
        "firered_asr2s",
    }
    assert result.results["streaming_asr"]["command_adapter"]["adapter"] == "command_fixture"
    assert result.results["streaming_asr"]["command_adapter"]["metrics"]["records"] == 2
    assert result.results["nanoturn"]["status"] == "skipped"


def test_run_paper_smoke_with_nanoturn_baseline(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    result = run_paper_smoke(tmp_path, episodes=8, seed=3, train_model=True)

    assert result.results["nanoturn"]["status"] == "completed"
    assert "nanoturn_pico" in result.results["baselines"]
    assert "nanoturn_pico" in result.results["turn_benchmarks"]
    assert result.results["turn_benchmarks"]["nanoturn_pico"]["artifact_bytes"]
