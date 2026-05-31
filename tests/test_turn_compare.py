from stable_asr.data.registry import load_turn_records
from stable_asr.eval.turn_compare import compare_turn_predictors
from stable_asr.models.adapters import TurnPredictionManifestAdapter
from stable_asr.models.baselines import TextTurnBaseline, VADPauseBaseline


def test_compare_turn_predictors_ranks_and_reports() -> None:
    records = load_turn_records("examples/data/turn_demo.jsonl")
    report = compare_turn_predictors(
        records,
        [
            ("vad_pause", "baseline", VADPauseBaseline()),
            ("text_turn", "baseline", TextTurnBaseline()),
            (
                "oracle_predictions",
                "predictions",
                TurnPredictionManifestAdapter.from_jsonl("tests/fixtures/turn_predictions_sample.jsonl"),
            ),
        ],
        dataset="examples/data/turn_demo.jsonl",
    )

    assert report.rows[0].name == "oracle_predictions"
    assert report.rows[0].macro_f1 == 1.0
    assert "Stable-ASR Turn Comparison" in report.to_markdown()
    assert "oracle_predictions" in report.to_dict()["reports"]


def test_compare_turn_predictors_rejects_duplicate_names() -> None:
    records = load_turn_records("examples/data/turn_demo.jsonl")

    try:
        compare_turn_predictors(
            records,
            [
                ("dup", "baseline", VADPauseBaseline()),
                ("dup", "baseline", TextTurnBaseline()),
            ],
            dataset="demo",
        )
    except ValueError as exc:
        assert "duplicate predictor name" in str(exc)
    else:
        raise AssertionError("duplicate predictor names should fail")
