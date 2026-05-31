from pathlib import Path

from stable_asr.paper.case_studies import build_case_studies, case_studies_markdown, paper_case_studies
from stable_asr.paper.experiments import run_paper_smoke


def test_build_case_studies_links_failures_to_source_records(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=4, train_model=False)
    payload = build_case_studies(result.results)

    assert payload["summary"]["turn_cases"] > 0
    assert payload["summary"]["streaming_cases"] > 0
    turn_case = payload["turn_cases"][0]
    streaming_case = payload["streaming_cases"][0]

    assert "audio" in turn_case
    assert "text" in turn_case
    assert "scenario" in turn_case
    assert "reference" in streaming_case
    assert "final_text" in streaming_case


def test_paper_case_studies_writes_json_and_markdown(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=4, train_model=False)
    artifacts = paper_case_studies(result.results_path, tmp_path / "case_studies")

    assert Path(artifacts.json_path).exists()
    assert Path(artifacts.markdown_path).exists()
    markdown = Path(artifacts.markdown_path).read_text(encoding="utf-8")

    assert "Stable-ASR Case Studies" in markdown
    assert "Turn-Taking Failures" in markdown
    assert "Streaming ASR Failures" in markdown


def test_case_studies_markdown_handles_empty_payload() -> None:
    markdown = case_studies_markdown(
        {"turn_cases": [], "streaming_cases": [], "summary": {"turn_cases": 0, "streaming_cases": 0}},
        results_path="paper_results.json",
    )

    assert "turn_cases" in markdown
