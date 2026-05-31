from stable_asr.eval.turn_metrics import classification_report


def test_classification_report() -> None:
    report = classification_report(
        ["complete", "complete", "incomplete", "backchannel"],
        ["complete", "incomplete", "incomplete", "backchannel"],
        labels=["complete", "incomplete", "backchannel"],
    )

    assert report.accuracy == 0.75
    assert report.support["complete"] == 2
    assert report.precision["complete"] == 1.0
    assert report.recall["complete"] == 0.5
    assert 0.0 < report.macro_f1 <= 1.0

