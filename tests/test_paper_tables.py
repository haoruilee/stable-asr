from pathlib import Path

from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.tables import paper_table


def test_paper_table_baselines_and_data(tmp_path: Path) -> None:
    result = run_paper_smoke(tmp_path, episodes=8, seed=0, train_model=False)

    baselines = paper_table(result.results_path, "baselines")
    turn_benchmark = paper_table(result.results_path, "turn_benchmark")
    data = paper_table(result.results_path, "data")
    asr_manifest = paper_table(result.results_path, "asr_manifest_recipe")
    failure_cases = paper_table(result.results_path, "failure_cases")
    streaming = paper_table(result.results_path, "streaming")
    streaming_failures = paper_table(result.results_path, "streaming_failures")
    streaming_sweep = paper_table(result.results_path, "streaming_sweep")
    asr_transcripts = paper_table(result.results_path, "asr_transcript_conversions")
    scenarios = paper_table(result.results_path, "scenarios")
    policy = paper_table(result.results_path, "policy")

    assert "| baseline | accuracy |" in baselines
    assert "rule_endpoint" in baselines
    assert "text_turn" in baselines
    assert "prediction_manifest" in baselines
    assert "| baseline | avg_latency_ms |" in turn_benchmark
    assert "prediction_manifest" in turn_benchmark
    assert "| format | records |" in data or "benchmark" in data
    assert "| records | valid | languages |" in asr_manifest
    assert "librispeech" in asr_manifest
    assert "| baseline | category | count |" in failure_cases
    assert "vad_pause" in failure_cases
    assert "| records | wer | cer |" in streaming
    assert "| source | category | count |" in streaming_failures
    assert "overall" in streaming_failures
    assert "| chunk_ms | lookahead_ms |" in streaming_sweep
    assert "| schema | records | wer |" in asr_transcripts
    assert "whisper" in asr_transcripts
    assert "funasr" in asr_transcripts
    assert "| scenario | records | accuracy |" in scenarios
    assert "| score | complete_threshold |" in policy
