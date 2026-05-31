from stable_asr.data.manifest import load_manifest
from stable_asr.models.baselines import VADPauseBaseline
from stable_asr.turn.solver import threshold_search


def test_threshold_search() -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    result = threshold_search(
        records,
        VADPauseBaseline(),
        complete_thresholds=[0.5, 0.75],
        backchannel_thresholds=[0.7],
        wait_thresholds=[0.6],
        interrupt_thresholds=[0.75],
    )

    assert len(result.trials) == 2
    assert result.best.score >= 0.0
    assert "false_complete_rate" in result.best.interaction

