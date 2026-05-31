import json
from pathlib import Path

from stable_asr.cli import main
from stable_asr.paper.final_results import assemble_final_paper_results
from stable_asr.paper.tables import paper_table


def test_assemble_final_paper_results_writes_table_compatible_schema(tmp_path: Path) -> None:
    config = _write_final_result_inputs(tmp_path)

    report = assemble_final_paper_results(config, repo_root=tmp_path)
    output = tmp_path / "runs/final/paper_results.json"

    assert report.ok
    assert report.wrote
    assert output.exists()
    assert "rule_endpoint" in paper_table(output, "baselines")
    assert "whisper_final" in paper_table(output, "streaming")


def test_final_results_cli_reports_missing_inputs(tmp_path: Path, capsys) -> None:
    config = _minimal_config()
    config_path = tmp_path / "paper_final.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    code = main(["final-results", "--config", str(config_path), "--repo-root", str(tmp_path), "--validate-only"])

    captured = capsys.readouterr()
    assert code == 1
    assert "final_results_assembly: NOT_READY" in captured.out
    assert "data_benchmark" in captured.out


def _write_final_result_inputs(tmp_path: Path) -> dict[str, object]:
    config = _minimal_config()
    _write_jsonl(
        tmp_path / "runs/final/turn_test.jsonl",
        [
            {
                "id": "turn1",
                "audio": "audio/turn1.wav",
                "sample_rate": 16000,
                "start": 0.0,
                "end": 1.0,
                "turn_label": "complete",
                "action_label": "take_turn",
                "assistant_speaking": False,
                "overlap": False,
                "language": "en",
                "source": "unit",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "runs/final/asr_eval_manifest.jsonl",
        [
            {
                "id": "utt1",
                "audio": "audio/utt1.wav",
                "sample_rate": 16000,
                "text": "hello",
                "language": "en",
                "source": "unit",
                "duration": 1.0,
            }
        ],
    )
    _write_json(tmp_path / "runs/final/reports/data_benchmark.json", [_data_row()])
    _write_json(tmp_path / "runs/final/reports/baselines.json", {"rule_endpoint": _turn_eval()})
    _write_json(tmp_path / "runs/final/reports/turn_benchmarks.json", {"rule_endpoint": _turn_benchmark()})
    _write_json(tmp_path / "runs/final/reports/scenarios.json", _scenarios())
    _write_json(tmp_path / "runs/final/reports/policy_search.json", _policy_search())
    _write_json(tmp_path / "runs/final/reports/asr_command_compare.json", _streaming_comparison())
    _write_json(tmp_path / "runs/final/reports/whisper_sweep.json", _streaming_sweep())
    _write_json(tmp_path / "runs/final/nanoturn/metrics.json", {"accuracy": 1.0})
    return config


def _minimal_config() -> dict[str, object]:
    return {
        "id": "stable_asr_final_run_v0",
        "version": "0.1.0",
        "title": "Final",
        "output_dir": "runs/final",
        "seed": 0,
        "public_corpora": [
            {
                "id": "unit",
                "language": "en",
                "corpus": "librispeech",
                "input_dir": "data/unit",
                "manifest": "runs/final/unit/asr_manifest.jsonl",
                "sample_rate": 16000,
                "license": "test",
            }
        ],
        "asr_eval_manifest": "runs/final/asr_eval_manifest.jsonl",
        "turn_splits": {
            "train": "runs/final/turn_train.jsonl",
            "dev": "runs/final/turn_dev.jsonl",
            "test": "runs/final/turn_test.jsonl",
            "voiceworld_real": "runs/final/voiceworld_real.jsonl",
        },
        "external_turn_predictions": [],
        "asr_command_config": "configs/final/asr_command_compare.json",
        "nanoturn": {
            "model": "nanoturn_pico",
            "checkpoint": "runs/final/nanoturn/checkpoint.pt",
            "metrics": "runs/final/nanoturn/metrics.json",
            "onnx": "runs/final/nanoturn/nanoturn.onnx",
        },
        "artifacts": {
            "paper_results": "runs/final/paper_results.json",
            "bundle_dir": "runs/final/artifacts",
            "markdown_draft": "runs/final/PAPER_DRAFT.md",
            "latex_draft": "runs/final/paper.tex",
            "dataset_card": "runs/final/DATASET_CARD.md",
            "experiment_card": "runs/final/EXPERIMENT_CARD.md",
        },
        "result_inputs": {
            "data_benchmark": "runs/final/reports/data_benchmark.json",
            "baselines": "runs/final/reports/baselines.json",
            "turn_benchmarks": "runs/final/reports/turn_benchmarks.json",
            "scenarios": "runs/final/reports/scenarios.json",
            "policy_search": "runs/final/reports/policy_search.json",
            "streaming_comparison": "runs/final/reports/asr_command_compare.json",
            "streaming_sweep": "runs/final/reports/whisper_sweep.json",
            "nanoturn": "runs/final/nanoturn/metrics.json",
        },
        "commands": ["stable-asr final-results --config configs/final/paper_final.json"],
    }


def _turn_eval() -> dict[str, object]:
    return {
        "classification": {"accuracy": 1.0, "macro_f1": 1.0},
        "interaction": {
            "false_complete_rate": 0.0,
            "premature_response_rate": 0.0,
            "missed_interrupt_rate": 0.0,
        },
        "failure_analysis": {"category_counts": {}},
    }


def _turn_benchmark() -> dict[str, object]:
    return {
        "avg_latency_ms": 1.0,
        "p50_latency_ms": 1.0,
        "p95_latency_ms": 1.0,
        "throughput_predictions_per_sec": 1000.0,
        "rtf": 0.001,
        "artifact_bytes": {},
    }


def _scenarios() -> dict[str, object]:
    return {
        "by_scenario": {
            "normal_question": {
                "examples": [{"id": "turn1"}],
                "classification": {"accuracy": 1.0, "macro_f1": 1.0},
                "interaction": {"false_complete_rate": 0.0, "missed_interrupt_rate": 0.0},
            }
        }
    }


def _policy_search() -> dict[str, object]:
    return {
        "best": {
            "score": 1.0,
            "config": {
                "complete_threshold": 0.75,
                "backchannel_threshold": 0.7,
                "wait_threshold": 0.6,
                "interrupt_min_confidence": 0.8,
            },
        },
        "trials": [{"score": 1.0}],
    }


def _streaming_comparison() -> dict[str, object]:
    return {
        "rows": [
            {
                "adapter": "whisper_final",
                "records": 1,
                "wer": 0.0,
                "cer": 0.0,
                "rtf": 0.2,
                "first_partial_latency": 0.1,
                "final_latency": 0.2,
                "endpoint_delay": 0.0,
                "partial_revision_rate": 0.0,
                "stable_prefix_ratio": 1.0,
                "timestamp_drift": 0.0,
                "failure_analysis": {"category_counts": {}},
            }
        ]
    }


def _streaming_sweep() -> dict[str, object]:
    return {
        "rows": [
            {
                "chunk_ms": 320,
                "lookahead_ms": 160,
                "first_partial_latency": 0.1,
                "final_latency": 0.2,
                "endpoint_delay": 0.0,
                "partial_revision_rate": 0.0,
                "timestamp_drift": 0.0,
            }
        ]
    }


def _data_row() -> dict[str, object]:
    return {
        "format": "jsonl",
        "records": 1,
        "write_seconds": 0.001,
        "read_seconds": 0.001,
        "size_bytes": 100,
        "sample_count": 1,
        "sample_seconds": 0.001,
        "samples_per_second": 1000.0,
        "sample_strategy": "random",
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
