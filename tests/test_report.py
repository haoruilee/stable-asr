from stable_asr.eval.report import MarkdownReport, dict_table


def test_markdown_report() -> None:
    report = MarkdownReport("Turn Eval")
    report.add_section("Metrics", "macro_f1: 1.0")

    text = report.to_markdown()

    assert "# Turn Eval" in text
    assert "## Metrics" in text
    assert "macro_f1: 1.0" in text


def test_dict_table() -> None:
    table = dict_table([{"metric": "macro_f1", "value": 1.0}])

    assert "| metric | value |" in table
    assert "| macro_f1 | 1.0 |" in table

