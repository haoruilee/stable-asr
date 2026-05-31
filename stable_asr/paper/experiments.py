"""Reproducible experiment bundles for paper-facing reports."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.converters import convert_external_jsonl, convert_streaming_asr_jsonl
from stable_asr.data.formats.jsonl import write_jsonl
from stable_asr.data.recipes import prepare_asr_manifest
from stable_asr.data.asr_manifest import load_asr_manifest, summarize_asr_records, validate_asr_manifest
from stable_asr.data.registry import convert_turn_manifest, load_turn_records, summarize_records
from stable_asr.data.benchmark import benchmark_data_formats
from stable_asr.eval.report import MarkdownReport, dict_table
from stable_asr.eval.turn_benchmark import benchmark_turn_predictor
from stable_asr.eval.turn_eval import evaluate_turn_records
from stable_asr.models.adapters import load_streaming_transcript_jsonl
from stable_asr.models.adapters.command import CommandStreamingASRAdapter
from stable_asr.models.adapters.turn_prediction import TurnPredictionManifestAdapter
from stable_asr.models.baselines import RuleEndpointBaseline, TextTurnBaseline, VADPauseBaseline
from stable_asr.scenarios.synthetic_turn import write_synthetic_turn_manifest
from stable_asr.scenarios.voice_world import evaluate_voice_world_records
from stable_asr.streaming.compare import compare_streaming_transcript_jsonl
from stable_asr.streaming.metrics import evaluate_streaming_records
from stable_asr.streaming.sweep import sweep_streaming_schedule
from stable_asr.train.turn_trainer import NanoTurnCheckpointPredictor, train_nanoturn
from stable_asr.turn.nanoturn import torch
from stable_asr.turn.solver import threshold_search


@dataclass(frozen=True)
class PaperRunResult:
    output_dir: str
    results_path: str
    report_path: str
    results: dict[str, Any]


def run_paper_smoke(
    output_dir: str | Path,
    *,
    episodes: int = 25,
    seed: int = 0,
    train_model: bool = True,
) -> PaperRunResult:
    """Run a small deterministic experiment bundle.

    This is not the final paper experiment. It establishes the artifact shape
    that future paper tables and figures should follow.
    """

    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "synthetic_turn.jsonl"
    converted_path = data_dir / "synthetic_turn_copy.jsonl"
    turn_prediction_fixture_path = output_dir / "turn_prediction_fixture.jsonl"
    streaming_fixture_path = output_dir / "streaming_asr_fixture.jsonl"
    streaming_fast_fixture_path = output_dir / "streaming_asr_fast_unstable_fixture.jsonl"
    records = write_synthetic_turn_manifest(manifest_path, episodes=episodes, seed=seed)

    convert_start = time.perf_counter()
    converted_count = convert_turn_manifest(manifest_path, converted_path)
    convert_seconds = time.perf_counter() - convert_start
    try:
        benchmark_rows = [
            row.to_dict()
            for row in benchmark_data_formats(
                records,
                output_dir=output_dir / "data_benchmark",
                formats=_available_data_benchmark_formats(),
                sample_count=max(16, episodes),
                seed=seed,
            )
        ]
        benchmark_status: dict[str, Any] = {"status": "completed", "rows": benchmark_rows}
    except RuntimeError as exc:
        benchmark_status = {"status": "skipped", "reason": str(exc), "rows": []}

    external_conversions = _convert_external_fixtures(output_dir, data_dir)
    asr_manifest_recipe = _prepare_asr_manifest_fixture(output_dir, data_dir)

    predictors = {
        "rule_endpoint": RuleEndpointBaseline(),
        "vad_pause": VADPauseBaseline(),
        "text_turn": TextTurnBaseline(),
    }
    baseline_results = {
        name: evaluate_turn_records(records, predictor).to_dict()
        for name, predictor in predictors.items()
    }
    _write_turn_prediction_fixture(turn_prediction_fixture_path, records)
    predictors["prediction_manifest"] = TurnPredictionManifestAdapter.from_jsonl(turn_prediction_fixture_path)
    baseline_results["prediction_manifest"] = evaluate_turn_records(
        records,
        predictors["prediction_manifest"],
    ).to_dict()
    turn_benchmarks = {
        name: benchmark_turn_predictor(
            records,
            predictor,
            warmup=0,
            repeat=2,
            artifact_paths=[turn_prediction_fixture_path] if name == "prediction_manifest" else [],
        ).to_dict()
        for name, predictor in predictors.items()
    }

    nanoturn_result: dict[str, Any]
    if train_model and torch is not None:
        run = train_nanoturn(
            records,
            output_dir=output_dir / "nanoturn_pico",
            model_type="nanoturn_pico",
            epochs=30,
            seed=seed,
        )
        nanoturn_result = {
            "status": "completed",
            "checkpoint_path": run.checkpoint_path,
            "metrics_path": run.metrics_path,
            "metrics": run.metrics,
        }
        nanoturn_name = str(run.metrics["model_type"])
        nanoturn_predictor = NanoTurnCheckpointPredictor(run.checkpoint_path)
        baseline_results[nanoturn_name] = evaluate_turn_records(records, nanoturn_predictor).to_dict()
        turn_benchmarks[nanoturn_name] = benchmark_turn_predictor(
            records,
            nanoturn_predictor,
            warmup=0,
            repeat=2,
            artifact_paths=[run.checkpoint_path, run.metrics_path],
        ).to_dict()
    else:
        nanoturn_result = {
            "status": "skipped",
            "reason": "PyTorch unavailable or train_model=false",
        }

    scenario_results = evaluate_voice_world_records(
        records,
        VADPauseBaseline(),
        seed=seed,
        suite="paper_smoke_turn_suite",
    ).to_dict()
    policy_search = threshold_search(records, VADPauseBaseline()).to_dict()
    _write_streaming_fixture(streaming_fixture_path)
    _write_streaming_fast_unstable_fixture(streaming_fast_fixture_path)
    streaming_records = load_streaming_transcript_jsonl(streaming_fixture_path)
    streaming_report = evaluate_streaming_records(streaming_records)
    streaming_comparison = compare_streaming_transcript_jsonl(
        [
            ("balanced_fixture", streaming_fixture_path),
            ("fast_unstable_fixture", streaming_fast_fixture_path),
        ]
    )
    streaming_sweep = sweep_streaming_schedule(
        streaming_records,
        chunk_ms_values=[160, 320, 640],
        lookahead_ms_values=[0, 160],
    )
    asr_transcript_conversions = _convert_streaming_asr_fixtures(output_dir, data_dir)
    command_adapter = _run_streaming_command_adapter(output_dir, data_dir, streaming_fixture_path)

    results = {
        "meta": {
            "episodes": episodes,
            "seed": seed,
            "artifact_version": "paper_smoke_v0",
        },
        "data": {
            "manifest_path": str(manifest_path),
            "converted_path": str(converted_path),
            "turn_prediction_fixture_path": str(turn_prediction_fixture_path),
            "converted_count": converted_count,
            "convert_seconds": convert_seconds,
            "summary": summarize_records(load_turn_records(manifest_path)),
            "benchmark": benchmark_status,
            "asr_manifest_recipe": asr_manifest_recipe,
            "external_conversion": external_conversions[0],
            "external_conversions": external_conversions,
        },
        "baselines": baseline_results,
        "turn_benchmarks": turn_benchmarks,
        "scenarios": scenario_results,
        "policy_search": policy_search,
        "streaming_asr": {
            "fixture_path": str(streaming_fixture_path),
            "adapter_fixture_paths": {
                "balanced_fixture": str(streaming_fixture_path),
                "fast_unstable_fixture": str(streaming_fast_fixture_path),
            },
            "metrics": streaming_report.to_dict(),
            "adapter_comparison": streaming_comparison.to_dict(),
            "schedule_sweep": streaming_sweep.to_dict(),
            "asr_transcript_conversions": asr_transcript_conversions,
            "command_adapter": command_adapter,
        },
        "nanoturn": nanoturn_result,
    }

    results_path = output_dir / "paper_results.json"
    report_path = reports_dir / "paper_smoke.md"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_paper_smoke_markdown(results), encoding="utf-8")
    return PaperRunResult(
        output_dir=str(output_dir),
        results_path=str(results_path),
        report_path=str(report_path),
        results=results,
    )


def _paper_smoke_markdown(results: dict[str, Any]) -> str:
    report = MarkdownReport("Stable-ASR Paper Smoke Run")
    report.add_section(
        "Run Metadata",
        dict_table(
            [
                {
                    "episodes": results["meta"]["episodes"],
                    "seed": results["meta"]["seed"],
                    "artifact_version": results["meta"]["artifact_version"],
                }
            ]
        ),
    )
    report.add_section(
        "Data Summary",
        "```json\n" + json.dumps(results["data"]["summary"], ensure_ascii=False, indent=2) + "\n```",
    )
    report.add_section(
        "External Conversion",
        "```json\n"
        + json.dumps(results["data"]["external_conversions"], ensure_ascii=False, indent=2)
        + "\n```",
    )
    report.add_section(
        "ASR Manifest Recipe",
        "```json\n"
        + json.dumps(results["data"]["asr_manifest_recipe"], ensure_ascii=False, indent=2)
        + "\n```",
    )
    baseline_rows = []
    for name, payload in results["baselines"].items():
        baseline_rows.append(
            {
                "baseline": name,
                "accuracy": f"{payload['classification']['accuracy']:.4f}",
                "macro_f1": f"{payload['classification']['macro_f1']:.4f}",
                "false_complete_rate": f"{payload['interaction']['false_complete_rate']:.4f}",
                "missed_interrupt_rate": f"{payload['interaction']['missed_interrupt_rate']:.4f}",
            }
        )
    report.add_section("Baseline Comparison", dict_table(baseline_rows))
    failure_rows = []
    for name, payload in results["baselines"].items():
        analysis = payload.get("failure_analysis", {})
        category_counts = analysis.get("category_counts", {}) if isinstance(analysis, dict) else {}
        if isinstance(category_counts, dict) and category_counts:
            for category, count in category_counts.items():
                failure_rows.append({"baseline": name, "category": category, "count": count})
        else:
            failure_rows.append({"baseline": name, "category": "none", "count": 0})
    report.add_section("Failure Analysis", dict_table(failure_rows))
    benchmark_rows = []
    for name, payload in results["turn_benchmarks"].items():
        benchmark_rows.append(
            {
                "baseline": name,
                "avg_latency_ms": f"{payload['avg_latency_ms']:.4f}",
                "p95_latency_ms": f"{payload['p95_latency_ms']:.4f}",
                "throughput": f"{payload['throughput_predictions_per_sec']:.2f}",
                "rtf": f"{payload['rtf']:.6f}",
                "artifact_bytes": sum(payload["artifact_bytes"].values()),
            }
        )
    report.add_section("Turn Benchmark", dict_table(benchmark_rows))
    scenario_rows = []
    for scenario, payload in results["scenarios"]["by_scenario"].items():
        scenario_rows.append(
            {
                "scenario": scenario,
                "records": len(payload["examples"]),
                "accuracy": f"{payload['classification']['accuracy']:.4f}",
                "macro_f1": f"{payload['classification']['macro_f1']:.4f}",
                "false_complete_rate": f"{payload['interaction']['false_complete_rate']:.4f}",
                "missed_interrupt_rate": f"{payload['interaction']['missed_interrupt_rate']:.4f}",
            }
        )
    report.add_section("Scenario Robustness", dict_table(scenario_rows))
    best_policy = results["policy_search"]["best"]
    report.add_section(
        "Policy Search",
        dict_table(
            [
                {
                    "score": f"{best_policy['score']:.4f}",
                    "complete_threshold": best_policy["config"]["complete_threshold"],
                    "backchannel_threshold": best_policy["config"]["backchannel_threshold"],
                    "wait_threshold": best_policy["config"]["wait_threshold"],
                    "interrupt_min_confidence": best_policy["config"]["interrupt_min_confidence"],
                    "trials": len(results["policy_search"]["trials"]),
                }
            ]
        ),
    )
    streaming = results["streaming_asr"]["metrics"]
    report.add_section(
        "Streaming ASR Metrics",
        dict_table(
            [
                {
                    "wer": f"{streaming['wer']:.4f}",
                    "cer": f"{streaming['cer']:.4f}",
                    "rtf": f"{streaming['rtf']:.4f}",
                    "first_partial_latency": f"{streaming['first_partial_latency']:.4f}",
                    "final_latency": f"{streaming['final_latency']:.4f}",
                    "endpoint_delay": f"{streaming['endpoint_delay']:.4f}",
                    "partial_revision_rate": f"{streaming['partial_revision_rate']:.4f}",
                    "stable_prefix_ratio": f"{streaming['stable_prefix_ratio']:.4f}",
                    "timestamp_drift": f"{streaming['timestamp_drift']:.4f}",
                }
            ]
        ),
    )
    streaming_failures = streaming.get("failure_analysis", {})
    if isinstance(streaming_failures, dict):
        category_counts = streaming_failures.get("category_counts", {})
        rows = []
        if isinstance(category_counts, dict):
            rows = [
                {"category": category, "count": count}
                for category, count in category_counts.items()
            ]
        report.add_section("Streaming ASR Failure Analysis", dict_table(rows))
    comparison = results["streaming_asr"].get("adapter_comparison", {})
    if isinstance(comparison, dict):
        rows = []
        for row in comparison.get("rows", []):
            if isinstance(row, dict):
                rows.append(
                    {
                        "adapter": row["adapter"],
                        "wer": f"{float(row['wer']):.4f}",
                        "rtf": f"{float(row['rtf']):.4f}",
                        "endpoint_delay": f"{float(row['endpoint_delay']):.4f}",
                        "timestamp_drift": f"{float(row['timestamp_drift']):.4f}",
                    }
                )
        report.add_section("Streaming ASR Adapter Comparison", dict_table(rows))
    sweep = results["streaming_asr"].get("schedule_sweep", {})
    if isinstance(sweep, dict):
        rows = []
        for row in sweep.get("rows", []):
            if isinstance(row, dict):
                rows.append(
                    {
                        "chunk_ms": row["chunk_ms"],
                        "lookahead_ms": row["lookahead_ms"],
                        "first_partial_latency": f"{float(row['first_partial_latency']):.4f}",
                        "final_latency": f"{float(row['final_latency']):.4f}",
                        "endpoint_delay": f"{float(row['endpoint_delay']):.4f}",
                    }
                )
        report.add_section("Streaming ASR Schedule Sweep", dict_table(rows))
    conversions = results["streaming_asr"].get("asr_transcript_conversions", [])
    if isinstance(conversions, list):
        rows = []
        for conversion in conversions:
            if isinstance(conversion, dict):
                metrics = conversion.get("metrics", {})
                rows.append(
                    {
                        "schema": conversion["schema"],
                        "records": conversion["records"],
                        "wer": f"{float(metrics.get('wer', 0.0)):.4f}",
                        "rtf": f"{float(metrics.get('rtf', 0.0)):.4f}",
                        "endpoint_delay": f"{float(metrics.get('endpoint_delay', 0.0)):.4f}",
                    }
                )
        report.add_section("Streaming ASR Transcript Conversions", dict_table(rows))
    command_adapter = results["streaming_asr"].get("command_adapter", {})
    if isinstance(command_adapter, dict):
        metrics = command_adapter.get("metrics", {})
        if isinstance(metrics, dict):
            report.add_section(
                "Streaming ASR Command Adapter",
                dict_table(
                    [
                        {
                            "adapter": command_adapter.get("adapter", "command_adapter"),
                            "records": metrics.get("records", 0),
                            "wer": f"{float(metrics.get('wer', 0.0)):.4f}",
                            "rtf": f"{float(metrics.get('rtf', 0.0)):.4f}",
                            "endpoint_delay": f"{float(metrics.get('endpoint_delay', 0.0)):.4f}",
                        }
                    ]
                ),
            )
    benchmark = results["data"]["benchmark"]
    if benchmark["status"] == "completed":
        report.add_section(
            "Data Benchmark",
            dict_table(
                [
                    {
                        "format": row["format"],
                        "records": row["records"],
                        "write_seconds": f"{row['write_seconds']:.6f}",
                        "read_seconds": f"{row['read_seconds']:.6f}",
                        "size_bytes": row["size_bytes"],
                        "sample_count": row.get("sample_count", 0),
                        "samples_per_second": f"{float(row.get('samples_per_second', 0.0)):.2f}",
                        "sample_strategy": row.get("sample_strategy", "disabled"),
                    }
                    for row in benchmark["rows"]
                ]
            ),
        )
    else:
        report.add_section("Data Benchmark", benchmark["reason"])
    if results["nanoturn"]["status"] == "completed":
        metrics = results["nanoturn"]["metrics"]
        report.add_section(
            "NanoTurn",
            dict_table(
                [
                    {
                        "model_type": metrics["model_type"],
                        "epochs": metrics["epochs"],
                        "final_loss": f"{metrics['final_loss']:.4f}",
                        "final_accuracy": f"{metrics['final_accuracy']:.4f}",
                    }
                ]
            ),
        )
    else:
        report.add_section("NanoTurn", results["nanoturn"]["reason"])
    return report.to_markdown()


def _convert_external_fixtures(output_dir: Path, data_dir: Path) -> list[dict[str, Any]]:
    conversions: list[dict[str, Any]] = []
    for schema in ("easyturn", "full_duplex_bench", "smart_turn"):
        fixture_path = output_dir / f"external_{schema}_fixture.jsonl"
        converted_path = data_dir / f"external_{schema}_converted.jsonl"
        _write_external_fixture(fixture_path, schema=schema)
        count = convert_external_jsonl(
            fixture_path,
            converted_path,
            schema=schema,
            default_language="en",
        )
        conversions.append(
            {
                "schema": schema,
                "input_path": str(fixture_path),
                "output_path": str(converted_path),
                "records": count,
                "summary": summarize_records(load_turn_records(converted_path)),
            }
        )
    return conversions


def _convert_streaming_asr_fixtures(output_dir: Path, data_dir: Path) -> list[dict[str, Any]]:
    conversions: list[dict[str, Any]] = []
    for schema in ("whisper", "funasr"):
        fixture_path = output_dir / f"external_{schema}_transcript_fixture.jsonl"
        converted_path = data_dir / f"external_{schema}_streaming_asr.jsonl"
        _write_asr_transcript_fixture(fixture_path, schema=schema)
        count = convert_streaming_asr_jsonl(fixture_path, converted_path, schema=schema)
        metrics = evaluate_streaming_records(load_streaming_transcript_jsonl(converted_path)).to_dict()
        conversions.append(
            {
                "schema": schema,
                "input_path": str(fixture_path),
                "output_path": str(converted_path),
                "records": count,
                "metrics": metrics,
            }
        )
    return conversions


def _prepare_asr_manifest_fixture(output_dir: Path, data_dir: Path) -> dict[str, Any]:
    fixture_path = output_dir / "asr_metadata_fixture.tsv"
    manifest_path = data_dir / "asr_manifest_fixture.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                "utt_id\taudio_path\ttranscript\tduration_sec\tspeaker_id\tsplit\tsource\tlanguage\tdomain",
                "paper_asr_001\taudio/paper_asr_001.wav\twhat is the weather\t2.10\tspk_a\tdev\tlibrispeech\ten\tassistant_query",
                "paper_asr_002\taudio/paper_asr_002.wav\tturn on the lights\t2.40\tspk_b\tdev\tlibrispeech\ten\tsmart_home",
                "paper_asr_003\taudio/paper_asr_003.wav\t我想问一下今天北京的天气\t2.80\tspk_c\ttest\taishell1\tzh\tassistant_query",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records = prepare_asr_manifest(
        fixture_path,
        manifest_path,
        default_sample_rate=16000,
        default_source="paper_asr_fixture",
    )
    validation = validate_asr_manifest(manifest_path)
    return {
        "input_path": str(fixture_path),
        "output_path": str(manifest_path),
        "records": len(records),
        "validation": validation.to_dict(),
        "summary": summarize_asr_records(load_asr_manifest(manifest_path)),
    }


def _run_streaming_command_adapter(output_dir: Path, data_dir: Path, fixture_path: Path) -> dict[str, Any]:
    script_path = output_dir / "command_asr_fixture.py"
    output_path = data_dir / "command_adapter_streaming_asr.jsonl"
    script_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import shutil",
                "import sys",
                "src = Path(sys.argv[1])",
                "dst = Path(sys.argv[2])",
                "dst.parent.mkdir(parents=True, exist_ok=True)",
                "shutil.copyfile(src, dst)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = CommandStreamingASRAdapter(
        name="command_fixture",
        command=[sys.executable, str(script_path), str(fixture_path), "{output}"],
        output_path=output_path,
        timeout_sec=30.0,
    )
    metrics = evaluate_streaming_records(adapter.load_records()).to_dict()
    return {
        "adapter": adapter.name,
        "command": adapter.command_args(),
        "output_path": str(output_path),
        "records": metrics["records"],
        "metrics": metrics,
    }


def _available_data_benchmark_formats() -> list[str]:
    formats = ["jsonl"]
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return formats
    formats.append("parquet")
    try:
        import lance  # noqa: F401
    except Exception:
        return formats
    formats.append("lance")
    return formats


def _write_external_fixture(path: Path, *, schema: str) -> None:
    if schema == "easyturn":
        rows = [
            {
                "id": "paper_easy_001",
                "audio_path": "audio/paper_easy_001.wav",
                "sample_rate": 16000,
                "duration": 1.8,
                "text": "what is the weather",
                "label": "complete",
                "language": "en",
                "metadata": {"pause_ms": 850},
            },
            {
                "id": "paper_easy_002",
                "audio_path": "audio/paper_easy_002.wav",
                "sample_rate": 16000,
                "duration": 1.1,
                "text": "I want to ask",
                "label": "incomplete",
                "language": "en",
                "metadata": {"pause_ms": 250},
            },
        ]
    elif schema == "full_duplex_bench":
        rows = [
            {
                "episode_id": "paper_fdb_001",
                "audio": "audio/paper_fdb_001.wav",
                "scenario": "user_interruption",
                "duration_sec": 1.2,
                "transcript": "wait that is not it",
                "expected_action": "stop_tts_and_listen",
                "lang": "en",
            },
            {
                "episode_id": "paper_fdb_002",
                "audio": "audio/paper_fdb_002.wav",
                "scenario": "side_conversation",
                "duration_sec": 2.0,
                "transcript": "talking to someone else",
                "lang": "en",
            },
        ]
    elif schema == "smart_turn":
        rows = [
            {
                "id": "paper_smart_001",
                "audio": "audio/paper_smart_001.wav",
                "duration_sec": 1.6,
                "text": "what is the weather",
                "completion_probability": 0.92,
                "language": "en",
            },
            {
                "id": "paper_smart_002",
                "audio": "audio/paper_smart_002.wav",
                "duration_sec": 1.0,
                "text": "I was wondering if",
                "completion_probability": 0.18,
                "language": "en",
            },
        ]
    else:
        raise ValueError(f"unknown external fixture schema: {schema}")
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _write_asr_transcript_fixture(path: Path, *, schema: str) -> None:
    if schema == "whisper":
        rows = [
            {
                "id": "paper_whisper_001",
                "audio": "audio/paper_whisper_001.wav",
                "reference": "what is the weather",
                "text": "what is the weather",
                "duration": 2.0,
                "processing_time": 0.42,
                "speech_end_time": 1.8,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.8,
                        "text": "what is",
                        "words": [
                            {"word": "what", "start": 0.10, "end": 0.34},
                            {"word": "is", "start": 0.40, "end": 0.56},
                        ],
                    },
                    {
                        "start": 0.8,
                        "end": 1.7,
                        "text": "the weather",
                        "words": [
                            {"word": "the", "start": 0.84, "end": 0.98},
                            {"word": "weather", "start": 1.05, "end": 1.46},
                        ],
                    },
                ],
                "reference_word_timestamps": [
                    {"word": "what", "start": 0.10, "end": 0.35},
                    {"word": "is", "start": 0.40, "end": 0.55},
                    {"word": "the", "start": 0.60, "end": 0.75},
                    {"word": "weather", "start": 0.85, "end": 1.35},
                ],
            },
            {
                "utt_id": "paper_whisper_002",
                "audio_path": "audio/paper_whisper_002.wav",
                "reference": "turn on the lights",
                "text": "turn on lights",
                "audio_duration": 2.4,
                "runtime_sec": 0.55,
                "finalized_at": 2.3,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.7,
                        "text": "turn on",
                        "words": [
                            {"word": "turn", "start": 0.18, "end": 0.42},
                            {"word": "on", "start": 0.50, "end": 0.68},
                        ],
                    },
                    {
                        "start": 0.7,
                        "end": 1.5,
                        "text": "lights",
                        "words": [{"word": "lights", "start": 1.00, "end": 1.38}],
                    },
                ],
                "reference_word_timestamps": [
                    {"word": "turn", "start": 0.20, "end": 0.45},
                    {"word": "on", "start": 0.52, "end": 0.70},
                    {"word": "the", "start": 0.76, "end": 0.90},
                    {"word": "lights", "start": 1.00, "end": 1.45},
                ],
            },
        ]
    elif schema == "funasr":
        rows = [
            {
                "key": "paper_funasr_001",
                "wav": "audio/paper_funasr_001.wav",
                "target": "what is the weather",
                "text": "what is the weather",
                "duration_ms": 2000,
                "runtime_ms": 360,
                "end_of_speech": 1800,
                "sentence_info": [
                    {"start": 0, "end": 800, "text": "what is"},
                    {"start": 800, "end": 1700, "text": "the weather"},
                ],
                "words": [
                    {"text": "what", "begin": 100, "end": 340},
                    {"text": "is", "begin": 400, "end": 560},
                    {"text": "the", "begin": 840, "end": 980},
                    {"text": "weather", "begin": 1050, "end": 1460},
                ],
                "reference_words": [
                    {"word": "what", "start": 0.10, "end": 0.35},
                    {"word": "is", "start": 0.40, "end": 0.55},
                    {"word": "the", "start": 0.60, "end": 0.75},
                    {"word": "weather", "start": 0.85, "end": 1.35},
                ],
            },
            {
                "audio_id": "paper_funasr_002",
                "path": "audio/paper_funasr_002.wav",
                "ref": "turn on the lights",
                "pred": "turn on lights",
                "duration": 2400,
                "latency_ms": 420,
                "finalization_time": 2300,
                "sentence_info": [
                    {"start": 0, "end": 700, "text": "turn on"},
                    {"start": 700, "end": 1500, "text": "lights"},
                ],
                "timestamp": [[180, 420], [500, 680], [1000, 1380]],
                "reference_words": [
                    {"word": "turn", "start": 0.20, "end": 0.45},
                    {"word": "on", "start": 0.52, "end": 0.70},
                    {"word": "the", "start": 0.76, "end": 0.90},
                    {"word": "lights", "start": 1.00, "end": 1.45},
                ],
            },
        ]
    else:
        raise ValueError(f"unknown ASR transcript fixture schema: {schema}")
    write_jsonl(path, rows)


def _write_streaming_fixture(path: Path) -> None:
    rows = [
        {
            "id": "paper_stream_001",
            "reference": "what is the weather",
            "final_text": "what is the weather",
            "audio_duration": 2.0,
            "processing_time": 0.5,
            "speech_end_time": 1.8,
            "endpoint_time": 2.1,
            "reference_word_timestamps": [
                {"word": "what", "start": 0.10, "end": 0.35},
                {"word": "is", "start": 0.40, "end": 0.55},
                {"word": "the", "start": 0.60, "end": 0.75},
                {"word": "weather", "start": 0.85, "end": 1.35},
            ],
            "word_timestamps": [
                {"word": "what", "start": 0.12, "end": 0.36},
                {"word": "is", "start": 0.43, "end": 0.57},
                {"word": "the", "start": 0.63, "end": 0.78},
                {"word": "weather", "start": 0.91, "end": 1.41},
            ],
            "partials": [
                {"time": 0.4, "text": "what"},
                {"time": 0.8, "text": "what is"},
                {"time": 2.1, "text": "what is the weather", "is_final": True},
            ],
        },
        {
            "id": "paper_stream_002",
            "reference": "turn on the lights",
            "final_text": "turn on lights",
            "audio_duration": 2.4,
            "processing_time": 0.6,
            "speech_end_time": 2.0,
            "endpoint_time": 2.45,
            "reference_word_timestamps": [
                {"word": "turn", "start": 0.20, "end": 0.45},
                {"word": "on", "start": 0.52, "end": 0.70},
                {"word": "the", "start": 0.76, "end": 0.90},
                {"word": "lights", "start": 1.00, "end": 1.45},
            ],
            "word_timestamps": [
                {"word": "turn", "start": 0.24, "end": 0.49},
                {"word": "on", "start": 0.58, "end": 0.74},
                {"word": "lights", "start": 1.08, "end": 1.50},
            ],
            "partials": [
                {"time": 0.5, "text": "turn"},
                {"time": 0.9, "text": "turn off"},
                {"time": 2.45, "text": "turn on lights", "is_final": True},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _write_streaming_fast_unstable_fixture(path: Path) -> None:
    rows = [
        {
            "id": "paper_stream_001",
            "reference": "what is the weather",
            "final_text": "what is weather",
            "audio_duration": 2.0,
            "processing_time": 0.3,
            "speech_end_time": 1.8,
            "endpoint_time": 1.9,
            "reference_word_timestamps": [
                {"word": "what", "start": 0.10, "end": 0.35},
                {"word": "is", "start": 0.40, "end": 0.55},
                {"word": "the", "start": 0.60, "end": 0.75},
                {"word": "weather", "start": 0.85, "end": 1.35},
            ],
            "word_timestamps": [
                {"word": "what", "start": 0.20, "end": 0.44},
                {"word": "is", "start": 0.50, "end": 0.64},
                {"word": "weather", "start": 1.02, "end": 1.55},
            ],
            "partials": [
                {"time": 0.25, "text": "what"},
                {"time": 0.55, "text": "what was"},
                {"time": 1.9, "text": "what is weather", "is_final": True},
            ],
        },
        {
            "id": "paper_stream_002",
            "reference": "turn on the lights",
            "final_text": "turn off lights",
            "audio_duration": 2.4,
            "processing_time": 0.35,
            "speech_end_time": 2.0,
            "endpoint_time": 2.1,
            "reference_word_timestamps": [
                {"word": "turn", "start": 0.20, "end": 0.45},
                {"word": "on", "start": 0.52, "end": 0.70},
                {"word": "the", "start": 0.76, "end": 0.90},
                {"word": "lights", "start": 1.00, "end": 1.45},
            ],
            "word_timestamps": [
                {"word": "turn", "start": 0.30, "end": 0.55},
                {"word": "off", "start": 0.70, "end": 0.92},
                {"word": "lights", "start": 1.22, "end": 1.68},
            ],
            "partials": [
                {"time": 0.25, "text": "turn"},
                {"time": 0.60, "text": "turn on"},
                {"time": 0.95, "text": "turn off"},
                {"time": 2.1, "text": "turn off lights", "is_final": True},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _write_turn_prediction_fixture(path: Path, records: list[Any]) -> None:
    predictor = TextTurnBaseline()
    rows = []
    for record in records:
        prediction = predictor.predict(record)
        rows.append(
            {
                "id": record.id,
                "probs": prediction.probs,
                "timestamp": prediction.timestamp,
                "source": "text_turn_serialized",
            }
        )
    write_jsonl(path, rows)
