import csv
import json
from pathlib import Path

from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.leaderboard import (
    export_leaderboard,
    leaderboard_report,
    leaderboard_rows,
    load_paper_results,
    validate_leaderboard_jsonl,
)


def test_leaderboard_rows_cover_major_tasks(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=2, train_model=False)
    rows = leaderboard_rows(load_paper_results(result.results_path), source=result.results_path)
    tasks = {row.task for row in rows}

    assert {
        "turn_quality",
        "turn_latency",
        "data_layer",
        "asr_manifest_recipe",
        "voiceworld",
        "policy_search",
        "streaming_asr",
        "asr_transcript_conversion",
    }.issubset(tasks)
    assert any(row.system == "rule_endpoint" and row.metric == "macro_f1" for row in rows)
    assert any(row.system == "balanced_fixture" and row.metric == "wer" for row in rows)
    assert any(row.system == "schedule_sweep" and row.metric == "endpoint_delay" for row in rows)
    assert any(row.system == "command_fixture" and row.metric == "wer" for row in rows)
    assert any(row.system == "whisper" and row.task == "asr_transcript_conversion" for row in rows)
    assert any(row.system == "funasr" and row.task == "asr_transcript_conversion" for row in rows)
    assert any(row.system == "qwen3_asr" and row.task == "asr_transcript_conversion" for row in rows)
    assert any(row.system == "firered_asr2s" and row.task == "asr_transcript_conversion" for row in rows)
    assert any(row.task == "asr_manifest_recipe" and row.metric == "records" for row in rows)


def test_export_leaderboard_jsonl_and_csv(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=2, train_model=False)
    jsonl = Path(export_leaderboard(result.results_path, tmp_path / "leaderboard.jsonl", format="jsonl"))
    csv_path = Path(export_leaderboard(result.results_path, tmp_path / "leaderboard.csv", format="csv"))

    first = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert {"suite", "task", "system", "metric", "value", "higher_is_better"}.issubset(first)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "task" in rows[0]


def test_validate_leaderboard_jsonl_accepts_exported_rows(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=2, train_model=False)
    jsonl = Path(export_leaderboard(result.results_path, tmp_path / "leaderboard.jsonl", format="jsonl"))

    report = validate_leaderboard_jsonl(jsonl)

    assert report.ok
    assert report.rows > 0
    assert "turn_quality" in report.tasks
    assert "leaderboard_validation: OK" in report.to_text()


def test_leaderboard_report_ranks_metric_groups(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path / "paper", episodes=9, seed=2, train_model=False)
    jsonl = Path(export_leaderboard(result.results_path, tmp_path / "leaderboard.jsonl", format="jsonl"))

    report = leaderboard_report(jsonl, top_k=2)

    assert report.ok
    assert report.groups > 0
    assert report.ranked_rows
    assert all(row.rank <= 2 for row in report.ranked_rows)
    assert any(row.task == "turn_quality" and row.metric == "macro_f1" for row in report.ranked_rows)
    assert "Stable-ASR Leaderboard Report" in report.to_markdown()


def test_validate_leaderboard_jsonl_rejects_bad_unit_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "bad_leaderboard.jsonl"
    row = {
        "suite": "stable_asr_v0",
        "task": "turn_quality",
        "system": "rule_endpoint",
        "slice": "overall",
        "metric": "macro_f1",
        "value": 0.5,
        "unit": "ms",
        "higher_is_better": False,
        "source": "unit",
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    report = validate_leaderboard_jsonl(path)

    assert not report.ok
    text = report.to_text()
    assert "unit" in text
    assert "higher_is_better" in text
    assert "duplicate" in text
