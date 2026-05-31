from pathlib import Path

from stable_asr.scenarios import (
    SCENARIO_NAMES,
    load_scenario_suite,
    scenario_suite_markdown,
    validate_scenario_suite,
    write_scenario_suite_json,
)


def test_default_scenario_suite_validates_against_generated_scenarios() -> None:
    suite = load_scenario_suite()
    config_suite = load_scenario_suite("configs/scenarios/stable_asr_voiceworld_v0.json")

    assert validate_scenario_suite(suite).ok
    assert validate_scenario_suite(config_suite).ok
    assert {item["id"] for item in suite["scenarios"]} == set(SCENARIO_NAMES)
    assert [item["id"] for item in config_suite["scenarios"]] == [item["id"] for item in suite["scenarios"]]


def test_scenario_suite_markdown_and_json_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "scenario_suite.json"
    write_scenario_suite_json(output)
    suite = load_scenario_suite(output)
    markdown = scenario_suite_markdown(suite)

    assert "Stable-ASR VoiceWorld v0 Scenario Suite" in markdown
    assert "user_interruption" in markdown
    assert "network_jitter_ms" in markdown


def test_scenario_suite_validation_rejects_missing_generated_scenario() -> None:
    suite = load_scenario_suite()
    suite["scenarios"] = [item for item in suite["scenarios"] if item["id"] != "ambient_speech"]

    report = validate_scenario_suite(suite)

    assert not report.ok
    assert "ambient_speech" in report.to_text()
