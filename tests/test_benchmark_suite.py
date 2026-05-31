from pathlib import Path

from stable_asr.paper.suites import (
    audit_benchmark_suite_coverage,
    benchmark_suite_markdown,
    load_benchmark_suite,
    validate_benchmark_suite,
    write_benchmark_suite_json,
)
from stable_asr.paper.experiments import run_paper_smoke


def test_default_benchmark_suite_validates() -> None:
    suite = load_benchmark_suite()
    config_suite = load_benchmark_suite("configs/benchmarks/stable_asr_v0.json")
    report = validate_benchmark_suite(suite)
    config_report = validate_benchmark_suite(config_suite)

    assert report.ok
    assert config_report.ok
    assert suite["id"] == "stable_asr_v0"
    assert [task["id"] for task in config_suite["tasks"]] == [task["id"] for task in suite["tasks"]]
    assert {task["id"] for task in suite["tasks"]}.issuperset(
        {"turn_quality", "asr_manifest_recipe", "voiceworld", "streaming_asr", "asr_transcript_conversion"}
    )


def test_benchmark_suite_markdown_and_json_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "benchmark_suite.json"
    write_benchmark_suite_json(output)
    suite = load_benchmark_suite(output)
    markdown = benchmark_suite_markdown(suite)

    assert validate_benchmark_suite(suite).ok
    assert "Stable-ASR v0 Paper Benchmark Suite" in markdown
    assert "asr_manifest_recipe" in markdown
    assert "asr_transcript_conversion" in markdown


def test_benchmark_suite_validation_rejects_bad_metric() -> None:
    suite = load_benchmark_suite()
    suite["tasks"][0]["metrics"][0].pop("higher_is_better")

    report = validate_benchmark_suite(suite)

    assert not report.ok
    assert "higher_is_better" in report.to_text()


def test_benchmark_suite_coverage_accepts_matching_result_subset(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=2, train_model=False)
    suite = load_benchmark_suite()
    suite["tasks"] = [task for task in suite["tasks"] if task["id"] == "turn_quality"]
    suite["tasks"][0]["systems"] = ["rule_endpoint", "vad_pause", "text_turn", "prediction_manifest"]

    report = audit_benchmark_suite_coverage(result.results, suite=suite)

    assert report.ok
    assert report.rows > 0


def test_benchmark_suite_coverage_rejects_missing_required_system(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=2, train_model=False)

    report = audit_benchmark_suite_coverage(result.results, suite=load_benchmark_suite())

    assert not report.ok
    assert "nanoturn_pico" in report.to_text()
