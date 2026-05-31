from stable_asr.models.baselines import VADPauseBaseline
from stable_asr.scenarios.voice_world import evaluate_voice_world


def test_evaluate_voice_world() -> None:
    report = evaluate_voice_world(VADPauseBaseline(), episodes=10, seed=0)

    assert report.suite == "zh_turn_mini_v0"
    assert len(report.overall.examples) == 10
    assert "backchannel" in report.by_scenario
    assert "side_conversation" in report.by_scenario
    assert "ambient_speech" in report.by_scenario
    assert "noisy_farfield" in report.by_scenario
    assert "code_switching" in report.by_scenario
    assert "snr_db" in report.factor_summary
    assert "farfield_distance_m" in report.factor_summary
    assert "code_switch_ratio" in report.factor_summary
    assert "Scenario Breakdown" in report.to_markdown()
