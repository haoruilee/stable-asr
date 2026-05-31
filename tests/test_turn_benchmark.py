from pathlib import Path

from stable_asr.data.manifest import load_manifest
from stable_asr.eval.turn_benchmark import benchmark_turn_predictor
from stable_asr.models.baselines import TextTurnBaseline


def test_benchmark_turn_predictor_reports_latency_and_artifact_size(tmp_path: Path) -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"stable-asr")

    report = benchmark_turn_predictor(
        records,
        TextTurnBaseline(),
        warmup=0,
        repeat=2,
        artifact_paths=[artifact],
    )

    assert report.records == 4
    assert report.predictions == 8
    assert report.avg_latency_ms >= 0.0
    assert report.p95_latency_ms >= report.p50_latency_ms
    assert report.throughput_predictions_per_sec > 0.0
    assert report.rtf >= 0.0
    assert report.artifact_bytes[str(artifact)] == len(b"stable-asr")
    assert "Stable-ASR Turn Benchmark" in report.to_markdown()
