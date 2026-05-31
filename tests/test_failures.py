from stable_asr.data.manifest import load_manifest
from stable_asr.eval.failures import mine_turn_failures
from stable_asr.eval.turn_eval import TurnEvalExample, evaluate_turn_records
from stable_asr.models.baselines import RuleEndpointBaseline


def test_mine_turn_failures_prioritizes_interaction_failures() -> None:
    examples = [
        TurnEvalExample(
            id="interrupt",
            true_label="incomplete",
            pred_label="incomplete",
            true_action="stop_tts_and_listen",
            pred_action="keep_listening",
            confidence=0.6,
            scenario="user_interruption",
        ),
        TurnEvalExample(
            id="false_complete",
            true_label="incomplete",
            pred_label="complete",
            true_action="keep_listening",
            pred_action="take_turn",
            confidence=0.9,
            scenario="incomplete_pause",
        ),
        TurnEvalExample(
            id="ok",
            true_label="complete",
            pred_label="complete",
            true_action="take_turn",
            pred_action="take_turn",
            confidence=0.8,
            scenario="normal_question",
        ),
    ]

    summary = mine_turn_failures(examples)

    assert summary.total_failures == 2
    assert summary.category_counts["missed_interrupt"] == 1
    assert summary.category_counts["false_complete"] == 1
    assert summary.cases[0].category == "missed_interrupt"
    assert "Representative Failures" in summary.to_markdown()


def test_turn_eval_report_includes_failure_analysis() -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    report = evaluate_turn_records(records, RuleEndpointBaseline())

    payload = report.to_dict()

    assert "failure_analysis" in payload
    assert "category_counts" in payload["failure_analysis"]
