from stable_asr.data.manifest import load_manifest
from stable_asr.eval.turn_eval import evaluate_turn_records
from stable_asr.models.baselines import VADPauseBaseline


def test_evaluate_turn_records() -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    report = evaluate_turn_records(records, VADPauseBaseline())

    assert report.classification.support["complete"] == 1
    assert report.classification.support["backchannel"] == 1
    assert report.interaction["false_complete_rate"] >= 0.0
    assert report.examples[0].pred_label == "complete"
    assert report.failure_analysis.total_failures >= 0
    assert "failure_analysis" in report.to_dict()
    assert "Stable-ASR Turn Evaluation" in report.to_markdown()
