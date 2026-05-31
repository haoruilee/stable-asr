import csv
import json
from pathlib import Path

from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.leaderboard import export_leaderboard, leaderboard_rows, load_paper_results


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
