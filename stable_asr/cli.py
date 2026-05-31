"""Command-line interface for Stable-ASR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stable_asr import __version__
from stable_asr.data.asr_manifest import load_asr_manifest, summarize_asr_records, validate_asr_manifest
from stable_asr.data.audio_audit import audit_audio_records
from stable_asr.data.bootstrap import BootstrapTurnDataConfig, bootstrap_turn_data
from stable_asr.data.manifest import load_manifest, validate_manifest
from stable_asr.data.profile import profile_turn_records
from stable_asr.data.benchmark import benchmark_data_formats
from stable_asr.data.split_audit import DEFAULT_LEAKAGE_FIELDS, audit_turn_splits
from stable_asr.data.converters import (
    ASR_TRANSCRIPT_SCHEMAS,
    EXTERNAL_SCHEMAS,
    convert_external_jsonl,
    convert_streaming_asr_jsonl,
)
from stable_asr.data.sources import data_sources_markdown, load_data_sources, validate_data_sources
from stable_asr.data.recipes import prepare_asr_manifest
from stable_asr.data.registry import (
    TURN_FORMATS,
    convert_turn_manifest,
    load_turn_records,
    summarize_records,
    write_turn_records,
)
from stable_asr.data.split import TurnSplitConfig, split_turn_records
from stable_asr.data.turn_from_asr import ASRToTurnConfig, asr_records_to_turn_records
from stable_asr.doctor import run_doctor
from stable_asr.eval.turn_benchmark import benchmark_turn_predictor
from stable_asr.eval.turn_compare import compare_turn_predictors, compare_turn_predictors_on_splits
from stable_asr.eval.turn_eval import evaluate_turn_records
from stable_asr.models.baselines import RuleEndpointBaseline, TextTurnBaseline, VADPauseBaseline
from stable_asr.models.adapters import (
    CommandStreamingASRAdapter,
    PREDICTION_SCHEMAS,
    TurnPredictionManifestAdapter,
    adapter_registry_markdown,
    convert_turn_prediction_jsonl,
    export_turn_predictions_jsonl,
    load_adapter_registry,
    load_streaming_transcript_jsonl,
    validate_adapter_registry,
    validate_turn_prediction_jsonl,
)
from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.audit import audit_paper_artifacts, audit_paper_release
from stable_asr.paper.cards import dataset_card, experiment_card
from stable_asr.paper.case_studies import paper_case_studies
from stable_asr.paper.claims import audit_claims, paper_claims
from stable_asr.paper.draft import paper_draft
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.final_config import (
    audit_final_run_files,
    final_run_file_audit_markdown,
    final_run_config_markdown,
    load_final_run_config,
    scaffold_final_run,
    validate_final_run_config,
)
from stable_asr.paper.final_experiments import (
    final_experiments_markdown,
    load_final_experiments,
    validate_final_experiments,
)
from stable_asr.paper.figures import PAPER_FIGURES, paper_figure
from stable_asr.paper.leaderboard import export_leaderboard
from stable_asr.paper.latex import paper_latex
from stable_asr.paper.parity import (
    audit_paper_parity,
    load_paper_parity_checklist,
    paper_parity_markdown,
    validate_paper_parity_checklist,
)
from stable_asr.paper.status import paper_status, write_paper_status_markdown
from stable_asr.paper.suites import (
    audit_benchmark_suite_coverage,
    benchmark_suite_markdown,
    load_benchmark_suite,
    validate_benchmark_suite,
)
from stable_asr.paper.tables import PAPER_TABLES, load_paper_results, paper_table
from stable_asr.references import (
    asr_collections_markdown,
    audit_asr_collection_coverage,
    load_asr_collections,
    validate_asr_collections,
)
from stable_asr.roadmap import load_roadmap, roadmap_status, validate_roadmap
from stable_asr.scenarios.voice_world import evaluate_voice_world
from stable_asr.scenarios.synthetic_turn import generate_synthetic_turn_records, write_synthetic_turn_manifest
from stable_asr.scenarios.suites import (
    load_scenario_suite,
    scenario_suite_markdown,
    validate_scenario_suite,
)
from stable_asr.streaming.command_compare import compare_asr_commands_from_config
from stable_asr.streaming.compare import compare_streaming_transcript_jsonl
from stable_asr.streaming.metrics import evaluate_streaming_records
from stable_asr.streaming.sweep import sweep_streaming_schedule
from stable_asr.train.export import export_nanoturn_onnx
from stable_asr.train.turn_trainer import NanoTurnCheckpointPredictor, train_nanoturn
from stable_asr.turn.labels import ACTION_LABELS, TURN_LABELS
from stable_asr.turn.policy import TurnPolicy, TurnPolicyConfig
from stable_asr.turn.solver import threshold_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stable-asr",
        description="Stable-ASR research platform utilities.",
    )
    parser.add_argument("--version", action="version", version=f"stable-asr {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check environment, optional dependencies, configs, and final input readiness.",
    )
    doctor_parser.add_argument("--repo-root", type=Path, default=Path("."))
    doctor_parser.add_argument("--check-final-files", action="store_true")
    doctor_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate-manifest",
        help="Validate a Stable-ASR JSONL turn manifest.",
    )
    validate_parser.add_argument("path", type=Path, help="Path to a JSONL manifest.")
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validation report as JSON.",
    )

    subparsers.add_parser(
        "labels",
        help="Print supported v0 turn and action labels.",
    )

    eval_parser = subparsers.add_parser(
        "eval-turn",
        help="Evaluate a turn-taking baseline on a JSONL manifest.",
    )
    eval_parser.add_argument("--dataset", type=Path, required=True, help="Path to a JSONL manifest.")
    eval_parser.add_argument(
        "--baseline",
        choices=["rule_endpoint", "vad_pause", "text_turn"],
        default="vad_pause",
        help="Baseline to evaluate.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional NanoTurn checkpoint to evaluate instead of a pause baseline.",
    )
    eval_parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional external turn prediction JSONL to evaluate instead of a built-in baseline.",
    )
    eval_parser.add_argument(
        "--audio-root",
        type=Path,
        help="Base directory for relative audio paths when evaluating audio-feature checkpoints.",
    )
    eval_parser.add_argument(
        "--complete-pause-ms",
        type=int,
        default=700,
        help="Trailing pause threshold used by pause baselines.",
    )
    eval_parser.add_argument(
        "--report",
        type=Path,
        help="Optional Markdown report output path.",
    )
    eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the evaluation report as JSON.",
    )

    predict_turn_parser = subparsers.add_parser(
        "predict-turn",
        help="Run a turn baseline or NanoTurn checkpoint and write prediction JSONL.",
    )
    predict_turn_parser.add_argument("--dataset", type=Path, required=True, help="Path to a turn manifest.")
    predict_turn_parser.add_argument("--output", type=Path, required=True, help="Output prediction JSONL path.")
    predict_turn_parser.add_argument(
        "--baseline",
        choices=["rule_endpoint", "vad_pause", "text_turn"],
        default="vad_pause",
        help="Built-in baseline to run when --checkpoint is omitted.",
    )
    predict_turn_parser.add_argument("--checkpoint", type=Path, help="Optional NanoTurn checkpoint.")
    predict_turn_parser.add_argument("--audio-root", type=Path)
    predict_turn_parser.add_argument("--complete-pause-ms", type=int, default=700)
    predict_turn_parser.add_argument("--json", action="store_true")

    validate_predictions_parser = subparsers.add_parser(
        "validate-turn-predictions",
        help="Validate turn prediction JSONL coverage, duplicate IDs, and row schema.",
    )
    validate_predictions_parser.add_argument("--dataset", type=Path, required=True, help="Path to a turn manifest.")
    validate_predictions_parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSONL path.")
    validate_predictions_parser.add_argument("--format", choices=TURN_FORMATS.names(), help="Optional dataset format.")
    validate_predictions_parser.add_argument("--allow-extra", action="store_true", help="Allow predictions for IDs absent from the dataset.")
    validate_predictions_parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    validate_predictions_parser.add_argument("--json", action="store_true")

    compare_turn_parser = subparsers.add_parser(
        "compare-turn",
        help="Compare multiple turn baselines, checkpoints, or prediction manifests on one dataset.",
    )
    compare_turn_parser.add_argument("--dataset", type=Path, required=True, help="Path to a turn manifest.")
    compare_turn_parser.add_argument(
        "--baseline",
        action="append",
        choices=["rule_endpoint", "vad_pause", "text_turn"],
        help="Built-in baseline to include. May be repeated. Defaults to all built-in baselines.",
    )
    compare_turn_parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named NanoTurn checkpoint to include. May be repeated.",
    )
    compare_turn_parser.add_argument(
        "--predictions",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named Stable-ASR turn prediction JSONL to include. May be repeated.",
    )
    compare_turn_parser.add_argument("--audio-root", type=Path)
    compare_turn_parser.add_argument("--complete-pause-ms", type=int, default=700)
    compare_turn_parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    compare_turn_parser.add_argument("--json", action="store_true")

    compare_turn_splits_parser = subparsers.add_parser(
        "compare-turn-splits",
        help="Compare turn predictors across train/dev/test split manifests.",
    )
    compare_turn_splits_parser.add_argument("--train", type=Path, required=True)
    compare_turn_splits_parser.add_argument("--dev", type=Path, required=True)
    compare_turn_splits_parser.add_argument("--test", type=Path, required=True)
    compare_turn_splits_parser.add_argument(
        "--baseline",
        action="append",
        choices=["rule_endpoint", "vad_pause", "text_turn"],
        help="Built-in baseline to include. May be repeated. Defaults to all built-in baselines.",
    )
    compare_turn_splits_parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    compare_turn_splits_parser.add_argument("--predictions", action="append", default=[], metavar="NAME=PATH")
    compare_turn_splits_parser.add_argument("--audio-root", type=Path)
    compare_turn_splits_parser.add_argument("--complete-pause-ms", type=int, default=700)
    compare_turn_splits_parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    compare_turn_splits_parser.add_argument("--json", action="store_true")

    benchmark_turn_parser = subparsers.add_parser(
        "benchmark-turn",
        help="Benchmark turn predictor latency, throughput, RTF, and artifact size.",
    )
    benchmark_turn_parser.add_argument("--dataset", type=Path, required=True, help="Path to a JSONL manifest.")
    benchmark_turn_parser.add_argument(
        "--baseline",
        choices=["rule_endpoint", "vad_pause", "text_turn"],
        default="vad_pause",
        help="Baseline to benchmark.",
    )
    benchmark_turn_parser.add_argument("--checkpoint", type=Path, help="Optional NanoTurn checkpoint.")
    benchmark_turn_parser.add_argument("--predictions", type=Path, help="Optional external turn prediction JSONL.")
    benchmark_turn_parser.add_argument("--audio-root", type=Path)
    benchmark_turn_parser.add_argument("--complete-pause-ms", type=int, default=700)
    benchmark_turn_parser.add_argument("--warmup", type=int, default=1)
    benchmark_turn_parser.add_argument("--repeat", type=int, default=5)
    benchmark_turn_parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="Additional artifact file path to include in the size report.",
    )
    benchmark_turn_parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    benchmark_turn_parser.add_argument("--json", action="store_true")

    synth_parser = subparsers.add_parser(
        "make-synthetic-turn-data",
        help="Generate a seedable synthetic turn-taking JSONL manifest.",
    )
    synth_parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    synth_parser.add_argument("--episodes", type=int, default=20, help="Number of records to write.")
    synth_parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    synth_parser.add_argument("--language", default="zh", help="Language tag.")
    synth_parser.add_argument("--write-audio", action="store_true", help="Write deterministic WAV demo audio.")

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a turn manifest between registered formats.",
    )
    convert_parser.add_argument("source", type=Path, help="Source manifest path.")
    convert_parser.add_argument("dest", type=Path, help="Destination manifest path.")
    convert_parser.add_argument("--source-format", choices=TURN_FORMATS.names())
    convert_parser.add_argument("--dest-format", choices=TURN_FORMATS.names())

    external_parser = subparsers.add_parser(
        "convert-external",
        help="Convert EasyTurn/Full-Duplex-Bench/SmartTurn-style JSONL into Stable-ASR JSONL.",
    )
    external_parser.add_argument("--schema", choices=EXTERNAL_SCHEMAS, required=True)
    external_parser.add_argument("--input", type=Path, required=True)
    external_parser.add_argument("--output", type=Path, required=True)
    external_parser.add_argument("--sample-rate", type=int, default=16000)
    external_parser.add_argument("--language", default="unknown")

    prediction_parser = subparsers.add_parser(
        "convert-predictions",
        help="Convert external turn prediction JSONL into Stable-ASR prediction JSONL.",
    )
    prediction_parser.add_argument("--schema", choices=PREDICTION_SCHEMAS, required=True)
    prediction_parser.add_argument("--input", type=Path, required=True)
    prediction_parser.add_argument("--output", type=Path, required=True)

    asr_transcript_parser = subparsers.add_parser(
        "convert-asr-transcript",
        help="Convert Whisper/FunASR-style transcript JSONL into Stable-ASR streaming ASR JSONL.",
    )
    asr_transcript_parser.add_argument("--schema", choices=ASR_TRANSCRIPT_SCHEMAS, required=True)
    asr_transcript_parser.add_argument("--input", type=Path, required=True)
    asr_transcript_parser.add_argument("--output", type=Path, required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-manifest",
        help="Print a summary of a turn manifest.",
    )
    inspect_parser.add_argument("path", type=Path, help="Manifest path.")
    inspect_parser.add_argument("--format", choices=TURN_FORMATS.names())
    inspect_parser.add_argument("--json", action="store_true", help="Print summary as JSON.")

    profile_parser = subparsers.add_parser(
        "profile-turn-data",
        help="Profile turn data distributions, durations, and training-readiness warnings.",
    )
    profile_parser.add_argument("--dataset", type=Path, required=True)
    profile_parser.add_argument("--format", choices=TURN_FORMATS.names())
    profile_parser.add_argument("--min-records", type=int, default=1)
    profile_parser.add_argument("--warn-label-imbalance", type=float, default=0.85)
    profile_parser.add_argument("--require-all-turn-labels", action="store_true")
    profile_parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    profile_parser.add_argument("--json", action="store_true")

    split_parser = subparsers.add_parser(
        "split-turn-data",
        help="Split a turn manifest into deterministic train/dev/test manifests.",
    )
    split_parser.add_argument("--input", type=Path, required=True, help="Input turn manifest.")
    split_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    split_parser.add_argument("--prefix", default="turn", help="Output filename prefix.")
    split_parser.add_argument("--format", choices=TURN_FORMATS.names(), default="jsonl", help="Output format.")
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--dev-ratio", type=float, default=0.1)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)
    split_parser.add_argument("--seed", type=int, default=0)
    split_parser.add_argument(
        "--stratify-by",
        action="append",
        default=None,
        help="Record field used for stratified splitting. May be repeated. Use --no-stratify to disable.",
    )
    split_parser.add_argument("--no-stratify", action="store_true")
    split_parser.add_argument("--group-by", help="Optional record field kept together across splits.")
    split_parser.add_argument("--allow-empty", action="store_true", help="Do not rebalance tiny datasets to fill all splits.")
    split_parser.add_argument("--json", action="store_true")

    split_audit_parser = subparsers.add_parser(
        "audit-turn-splits",
        help="Audit train/dev/test turn manifests for ID, audio, or group leakage.",
    )
    split_audit_parser.add_argument("--train", type=Path, required=True)
    split_audit_parser.add_argument("--dev", type=Path, required=True)
    split_audit_parser.add_argument("--test", type=Path, required=True)
    split_audit_parser.add_argument(
        "--field",
        action="append",
        default=None,
        help="Field that must not appear in multiple splits. Defaults to id, audio, metadata.asr_record_id, metadata.conversation_id.",
    )
    split_audit_parser.add_argument("--report", type=Path, help="Optional text report output path.")
    split_audit_parser.add_argument("--json", action="store_true")

    benchmark_parser = subparsers.add_parser(
        "benchmark-data",
        help="Benchmark registered manifest formats on a dataset.",
    )
    benchmark_parser.add_argument("--dataset", type=Path, required=True)
    benchmark_parser.add_argument("--output-dir", type=Path, required=True)
    benchmark_parser.add_argument(
        "--formats",
        nargs="+",
        choices=TURN_FORMATS.names(),
        default=["jsonl", "parquet"],
    )
    benchmark_parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Optional random record samples to benchmark after writing each format.",
    )
    benchmark_parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling benchmark.")
    benchmark_parser.add_argument("--json", action="store_true")

    data_sources_parser = subparsers.add_parser(
        "data-sources",
        help="Print or validate the Stable-ASR data source registry.",
    )
    data_sources_parser.add_argument("--registry", type=Path, help="Optional data source registry JSON path.")
    data_sources_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    data_sources_parser.add_argument("--json", action="store_true", help="Print registry as JSON.")
    data_sources_parser.add_argument("--validate-only", action="store_true")

    adapter_registry_parser = subparsers.add_parser(
        "adapter-registry",
        help="Print or validate the Stable-ASR external adapter registry.",
    )
    adapter_registry_parser.add_argument("--registry", type=Path, help="Optional adapter registry JSON path.")
    adapter_registry_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    adapter_registry_parser.add_argument("--json", action="store_true", help="Print registry as JSON.")
    adapter_registry_parser.add_argument("--validate-only", action="store_true")

    asr_collections_parser = subparsers.add_parser(
        "asr-collections",
        help="Print or validate the curated upstream ASR reference collection.",
    )
    asr_collections_parser.add_argument("--registry", type=Path, help="Optional ASR collections JSON path.")
    asr_collections_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    asr_collections_parser.add_argument("--json", action="store_true", help="Print registry as JSON.")
    asr_collections_parser.add_argument("--validate-only", action="store_true")
    asr_collections_parser.add_argument("--audit-coverage", action="store_true")
    asr_collections_parser.add_argument("--adapter-registry", type=Path, help="Adapter registry used by --audit-coverage.")
    asr_collections_parser.add_argument(
        "--require-priority",
        action="append",
        default=None,
        choices=["p0", "p1", "p2"],
        help="Reference priority required by --audit-coverage. Defaults to p0.",
    )

    roadmap_parser = subparsers.add_parser(
        "roadmap-status",
        help="Validate and summarize the machine-readable Stable-ASR roadmap.",
    )
    roadmap_parser.add_argument("--roadmap", type=Path, help="Optional roadmap registry JSON path.")
    roadmap_parser.add_argument("--repo-root", type=Path, default=Path("."))
    roadmap_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    roadmap_parser.add_argument("--json", action="store_true")
    roadmap_parser.add_argument("--validate-only", action="store_true")

    prepare_asr_parser = subparsers.add_parser(
        "prepare-asr-manifest",
        help="Normalize ASR corpus metadata TSV/CSV/JSONL into a Stable-ASR ASR manifest.",
    )
    prepare_asr_parser.add_argument("--input", type=Path, required=True, help="Input metadata table path.")
    prepare_asr_parser.add_argument("--output", type=Path, required=True, help="Output ASR manifest JSONL path.")
    prepare_asr_parser.add_argument("--audio-root", type=Path, help="Optional root joined to relative audio paths.")
    prepare_asr_parser.add_argument("--sample-rate", type=int, default=16000, help="Default sample rate.")
    prepare_asr_parser.add_argument("--language", default="unknown", help="Default language tag.")
    prepare_asr_parser.add_argument("--source", default="asr_manifest", help="Default source/corpus name.")
    prepare_asr_parser.add_argument("--split", help="Default split name.")
    prepare_asr_parser.add_argument("--id-field", help="Override input id column/key.")
    prepare_asr_parser.add_argument("--audio-field", help="Override input audio column/key.")
    prepare_asr_parser.add_argument("--text-field", help="Override input transcript column/key.")
    prepare_asr_parser.add_argument("--duration-field", help="Override input duration column/key.")
    prepare_asr_parser.add_argument("--speaker-field", help="Override input speaker column/key.")
    prepare_asr_parser.add_argument("--json", action="store_true", help="Print summary as JSON.")

    validate_asr_parser = subparsers.add_parser(
        "validate-asr-manifest",
        help="Validate a Stable-ASR ASR manifest JSONL file.",
    )
    validate_asr_parser.add_argument("path", type=Path)
    validate_asr_parser.add_argument("--json", action="store_true")

    inspect_asr_parser = subparsers.add_parser(
        "inspect-asr-manifest",
        help="Print a summary of an ASR manifest.",
    )
    inspect_asr_parser.add_argument("path", type=Path)
    inspect_asr_parser.add_argument("--json", action="store_true")

    asr_to_turn_parser = subparsers.add_parser(
        "asr-to-turn",
        help="Convert an ASR utterance manifest into weakly labeled turn windows.",
    )
    asr_to_turn_parser.add_argument("--input", type=Path, required=True, help="Input ASR manifest JSONL.")
    asr_to_turn_parser.add_argument("--output", type=Path, required=True, help="Output turn manifest.")
    asr_to_turn_parser.add_argument("--format", choices=TURN_FORMATS.names(), default="jsonl", help="Output format.")
    asr_to_turn_parser.add_argument("--window-sec", type=float, default=2.0)
    asr_to_turn_parser.add_argument("--no-complete", action="store_true", help="Do not emit complete windows.")
    asr_to_turn_parser.add_argument("--include-incomplete", action="store_true", help="Also emit truncated incomplete windows.")
    asr_to_turn_parser.add_argument("--incomplete-ratio", type=float, default=0.65)
    asr_to_turn_parser.add_argument("--min-incomplete-sec", type=float, default=0.4)
    asr_to_turn_parser.add_argument("--complete-pause-ms", type=int, default=900)
    asr_to_turn_parser.add_argument("--incomplete-pause-ms", type=int, default=250)
    asr_to_turn_parser.add_argument("--source", default="asr_weak_turn_v0")
    asr_to_turn_parser.add_argument(
        "--keep-incomplete-text",
        action="store_true",
        help="Keep full reference text on weak incomplete windows. Default drops it to avoid text-label leakage.",
    )
    asr_to_turn_parser.add_argument("--json", action="store_true")

    bootstrap_turn_parser = subparsers.add_parser(
        "bootstrap-turn-data",
        help="Prepare ASR metadata, derive weak turn windows, and write train/dev/test splits.",
    )
    bootstrap_turn_parser.add_argument("--input", type=Path, required=True, help="Input ASR metadata TSV/CSV/JSONL.")
    bootstrap_turn_parser.add_argument("--output-dir", type=Path, required=True)
    bootstrap_turn_parser.add_argument("--audio-root", type=Path)
    bootstrap_turn_parser.add_argument("--sample-rate", type=int, default=16000)
    bootstrap_turn_parser.add_argument("--language", default="unknown")
    bootstrap_turn_parser.add_argument("--source", default="asr_manifest")
    bootstrap_turn_parser.add_argument("--split")
    bootstrap_turn_parser.add_argument("--id-field")
    bootstrap_turn_parser.add_argument("--audio-field")
    bootstrap_turn_parser.add_argument("--text-field")
    bootstrap_turn_parser.add_argument("--duration-field")
    bootstrap_turn_parser.add_argument("--speaker-field")
    bootstrap_turn_parser.add_argument("--turn-format", choices=TURN_FORMATS.names(), default="jsonl")
    bootstrap_turn_parser.add_argument("--window-sec", type=float, default=2.0)
    bootstrap_turn_parser.add_argument("--include-incomplete", action="store_true")
    bootstrap_turn_parser.add_argument("--incomplete-ratio", type=float, default=0.65)
    bootstrap_turn_parser.add_argument("--train-ratio", type=float, default=0.8)
    bootstrap_turn_parser.add_argument("--dev-ratio", type=float, default=0.1)
    bootstrap_turn_parser.add_argument("--test-ratio", type=float, default=0.1)
    bootstrap_turn_parser.add_argument("--seed", type=int, default=0)
    bootstrap_turn_parser.add_argument("--split-prefix", default="turn")
    bootstrap_turn_parser.add_argument(
        "--group-by",
        default="metadata.asr_record_id",
        help="Field kept together across train/dev/test splits. Defaults to metadata.asr_record_id.",
    )
    bootstrap_turn_parser.add_argument("--no-group-by", action="store_true")
    bootstrap_turn_parser.add_argument("--json", action="store_true")

    audit_audio_parser = subparsers.add_parser(
        "audit-audio",
        help="Check manifest audio paths, WAV sample rates, and WAV durations.",
    )
    audit_audio_parser.add_argument("--kind", choices=["turn", "asr"], required=True)
    audit_audio_parser.add_argument("--manifest", type=Path, required=True)
    audit_audio_parser.add_argument("--audio-root", type=Path)
    audit_audio_parser.add_argument("--duration-tolerance-sec", type=float, default=0.05)
    audit_audio_parser.add_argument(
        "--require-inspectable",
        action="store_true",
        help="Fail non-WAV files whose metadata cannot be inspected by the built-in WAV reader.",
    )
    audit_audio_parser.add_argument("--report", type=Path, help="Optional text report output path.")
    audit_audio_parser.add_argument("--json", action="store_true")

    streaming_parser = subparsers.add_parser(
        "eval-streaming-asr",
        help="Evaluate streaming ASR transcript JSONL fixtures.",
    )
    streaming_parser.add_argument("--input", type=Path, required=True)
    streaming_parser.add_argument("--json", action="store_true")

    streaming_compare_parser = subparsers.add_parser(
        "compare-streaming-asr",
        help="Compare multiple streaming ASR transcript JSONL adapters.",
    )
    streaming_compare_parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="ADAPTER=PATH",
        help="Adapter name and transcript JSONL path. May be repeated.",
    )
    streaming_compare_parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    streaming_compare_parser.add_argument("--json", action="store_true")

    streaming_sweep_parser = subparsers.add_parser(
        "sweep-streaming-asr",
        help="Sweep chunk size and lookahead settings over a streaming ASR transcript JSONL.",
    )
    streaming_sweep_parser.add_argument("--input", type=Path, required=True)
    streaming_sweep_parser.add_argument("--chunks-ms", type=int, nargs="+", default=[160, 320, 640])
    streaming_sweep_parser.add_argument("--lookahead-ms", type=int, nargs="+", default=[0, 160, 320])
    streaming_sweep_parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    streaming_sweep_parser.add_argument("--json", action="store_true")

    command_asr_parser = subparsers.add_parser(
        "eval-asr-command",
        help="Run an external command that writes Stable-ASR streaming transcript JSONL, then evaluate it.",
    )
    command_asr_parser.add_argument("--name", default="command_asr")
    command_asr_parser.add_argument(
        "--command",
        dest="asr_command",
        required=True,
        help="Command string. Use {output} as the output JSONL placeholder.",
    )
    command_asr_parser.add_argument("--output", type=Path, required=True, help="Expected transcript JSONL written by the command.")
    command_asr_parser.add_argument("--cwd", type=Path)
    command_asr_parser.add_argument("--timeout", type=float, default=300.0)
    command_asr_parser.add_argument("--json", action="store_true")

    command_compare_parser = subparsers.add_parser(
        "compare-asr-commands",
        help="Compare multiple command-backed ASR adapters from a JSON config.",
    )
    command_compare_parser.add_argument("--config", type=Path, required=True)
    command_compare_parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    command_compare_parser.add_argument("--json", action="store_true")

    scenario_parser = subparsers.add_parser(
        "eval-scenario",
        help="Evaluate a turn predictor on the seedable VoiceWorld mini-suite.",
    )
    scenario_parser.add_argument("--episodes", type=int, default=25)
    scenario_parser.add_argument("--seed", type=int, default=0)
    scenario_parser.add_argument(
        "--baseline",
        choices=["rule_endpoint", "vad_pause", "text_turn"],
        default="vad_pause",
    )
    scenario_parser.add_argument("--checkpoint", type=Path)
    scenario_parser.add_argument("--report", type=Path)
    scenario_parser.add_argument("--json", action="store_true")

    scenario_suite_parser = subparsers.add_parser(
        "scenario-suite",
        help="Print or validate the Stable-ASR VoiceWorld scenario suite definition.",
    )
    scenario_suite_parser.add_argument("--suite", type=Path, help="Optional scenario suite JSON path.")
    scenario_suite_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    scenario_suite_parser.add_argument("--json", action="store_true", help="Print the suite as JSON.")
    scenario_suite_parser.add_argument("--validate-only", action="store_true")

    optimize_parser = subparsers.add_parser(
        "optimize-policy",
        help="Grid-search turn policy thresholds for a baseline predictor.",
    )
    optimize_parser.add_argument("--dataset", type=Path, help="Manifest path. If omitted, synthetic scenarios are generated.")
    optimize_parser.add_argument("--episodes", type=int, default=25)
    optimize_parser.add_argument("--seed", type=int, default=0)
    optimize_parser.add_argument(
        "--baseline",
        choices=["rule_endpoint", "vad_pause", "text_turn"],
        default="vad_pause",
    )
    optimize_parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    optimize_parser.add_argument("--json", action="store_true")

    train_parser = subparsers.add_parser(
        "train-turn",
        help="Train a NanoTurn baseline on a JSONL manifest.",
    )
    train_parser.add_argument("--dataset", type=Path, required=True, help="Training manifest path.")
    train_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    train_parser.add_argument(
        "--model",
        choices=["nanoturn_pico", "nanoturn_nano"],
        default="nanoturn_pico",
        help="NanoTurn model size.",
    )
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--lr", type=float, default=1e-2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument(
        "--feature-source",
        choices=["metadata", "audio"],
        default="metadata",
        help="Feature source used by NanoTurn v0.",
    )
    train_parser.add_argument(
        "--audio-root",
        type=Path,
        help="Base directory for relative audio paths. Defaults to the dataset parent.",
    )
    train_parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")

    export_parser = subparsers.add_parser(
        "export-turn-onnx",
        help="Export a NanoTurn checkpoint to ONNX.",
    )
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--opset", type=int, default=18)

    paper_parser = subparsers.add_parser(
        "reproduce-paper",
        help="Run a small reproducible paper-facing experiment bundle.",
    )
    paper_parser.add_argument("--config", type=Path, help="JSON config for the paper smoke run.")
    paper_parser.add_argument("--output-dir", type=Path)
    paper_parser.add_argument("--episodes", type=int)
    paper_parser.add_argument("--seed", type=int)
    paper_parser.add_argument("--skip-train", action="store_true")
    paper_parser.add_argument("--json", action="store_true", help="Print result paths as JSON.")

    paper_table_parser = subparsers.add_parser(
        "paper-table",
        help="Extract a Markdown table from paper_results.json.",
    )
    paper_table_parser.add_argument("table", choices=PAPER_TABLES)
    paper_table_parser.add_argument("--results", type=Path, required=True)
    paper_table_parser.add_argument("--output", type=Path)

    paper_figure_parser = subparsers.add_parser(
        "paper-figure",
        help="Generate an SVG figure from paper_results.json.",
    )
    paper_figure_parser.add_argument("figure", choices=PAPER_FIGURES)
    paper_figure_parser.add_argument("--results", type=Path, required=True)
    paper_figure_parser.add_argument("--output", type=Path, required=True)

    paper_bundle_parser = subparsers.add_parser(
        "paper-bundle",
        help="Generate all paper smoke tables, figures, and an artifact index.",
    )
    paper_bundle_parser.add_argument("--results", type=Path, required=True)
    paper_bundle_parser.add_argument("--output-dir", type=Path, required=True)
    paper_bundle_parser.add_argument("--json", action="store_true", help="Print bundle paths as JSON.")

    paper_status_parser = subparsers.add_parser(
        "paper-status",
        help="Summarize smoke, structural, and final paper readiness in one report.",
    )
    paper_status_parser.add_argument("--repo-root", type=Path, default=Path("."))
    paper_status_parser.add_argument("--results", type=Path)
    paper_status_parser.add_argument("--artifacts-dir", type=Path)
    paper_status_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    paper_status_parser.add_argument("--json", action="store_true")

    case_studies_parser = subparsers.add_parser(
        "paper-case-studies",
        help="Generate case-study JSON/Markdown artifacts from paper_results.json.",
    )
    case_studies_parser.add_argument("--results", type=Path, required=True)
    case_studies_parser.add_argument("--output-dir", type=Path, required=True)
    case_studies_parser.add_argument("--json", action="store_true")

    claim_parser = subparsers.add_parser(
        "paper-claim-audit",
        help="Audit paper claims against repo files, paper results, and artifact files.",
    )
    claim_parser.add_argument("--repo-root", type=Path, default=Path("."))
    claim_parser.add_argument("--results", type=Path)
    claim_parser.add_argument("--artifacts-dir", type=Path)
    claim_parser.add_argument("--output-dir", type=Path, help="Optional directory for CLAIMS.md and claims.json.")
    claim_parser.add_argument("--json", action="store_true")
    claim_parser.add_argument("--validate-only", action="store_true")

    parity_parser = subparsers.add_parser(
        "paper-parity-audit",
        help="Audit stable-worldmodel-style paper parity evidence and final-scale gaps.",
    )
    parity_parser.add_argument("--checklist", type=Path, help="Optional paper parity checklist JSON path.")
    parity_parser.add_argument("--repo-root", type=Path, default=Path("."))
    parity_parser.add_argument("--results", type=Path)
    parity_parser.add_argument("--artifacts-dir", type=Path)
    parity_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    parity_parser.add_argument("--json", action="store_true")
    parity_parser.add_argument("--validate-only", action="store_true")
    parity_parser.add_argument("--require-final", action="store_true", help="Fail if final-scale requirements remain.")

    final_experiments_parser = subparsers.add_parser(
        "final-experiments",
        help="Print or validate the final-scale platform-paper experiment runbook.",
    )
    final_experiments_parser.add_argument("--registry", type=Path, help="Optional final experiment registry JSON path.")
    final_experiments_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    final_experiments_parser.add_argument("--json", action="store_true", help="Print registry as JSON.")
    final_experiments_parser.add_argument("--validate-only", action="store_true")

    final_config_parser = subparsers.add_parser(
        "final-config",
        help="Print or validate the final paper run configuration template.",
    )
    final_config_parser.add_argument("--config", type=Path, help="Optional final run config JSON path.")
    final_config_parser.add_argument("--repo-root", type=Path, default=Path("."), help="Base directory for --check-files.")
    final_config_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    final_config_parser.add_argument("--json", action="store_true", help="Print config as JSON.")
    final_config_parser.add_argument("--validate-only", action="store_true")
    final_config_parser.add_argument("--check-files", action="store_true", help="Check required input/config paths exist.")
    final_config_parser.add_argument("--scaffold", action="store_true", help="Create final-run directories and README hints.")

    leaderboard_parser = subparsers.add_parser(
        "leaderboard-export",
        help="Export leaderboard-ready JSONL or CSV rows from paper_results.json.",
    )
    leaderboard_parser.add_argument("--results", type=Path, required=True)
    leaderboard_parser.add_argument("--output", type=Path, required=True)
    leaderboard_parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")

    benchmark_suite_parser = subparsers.add_parser(
        "benchmark-suite",
        help="Print or validate the Stable-ASR benchmark suite definition.",
    )
    benchmark_suite_parser.add_argument("--suite", type=Path, help="Optional benchmark suite JSON path.")
    benchmark_suite_parser.add_argument("--results", type=Path, help="Optional paper_results.json coverage input.")
    benchmark_suite_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    benchmark_suite_parser.add_argument("--json", action="store_true", help="Print the suite as JSON.")
    benchmark_suite_parser.add_argument("--validate-only", action="store_true", help="Only validate the suite.")

    paper_audit_parser = subparsers.add_parser(
        "paper-audit",
        help="Audit paper_results.json and optionally a paper artifact bundle.",
    )
    paper_audit_parser.add_argument("--results", type=Path, required=True)
    paper_audit_parser.add_argument("--artifacts-dir", type=Path)
    paper_audit_parser.add_argument("--json", action="store_true", help="Print audit report as JSON.")

    paper_release_audit_parser = subparsers.add_parser(
        "paper-release-audit",
        help="Run a stricter release-readiness audit for the platform paper.",
    )
    paper_release_audit_parser.add_argument("--repo-root", type=Path, default=Path("."))
    paper_release_audit_parser.add_argument("--results", type=Path)
    paper_release_audit_parser.add_argument("--artifacts-dir", type=Path)
    paper_release_audit_parser.add_argument("--markdown-draft", type=Path)
    paper_release_audit_parser.add_argument("--latex-draft", type=Path)
    paper_release_audit_parser.add_argument("--dataset-card", type=Path)
    paper_release_audit_parser.add_argument("--experiment-card", type=Path)
    paper_release_audit_parser.add_argument("--json", action="store_true", help="Print release audit as JSON.")

    paper_draft_parser = subparsers.add_parser(
        "paper-draft",
        help="Generate an editable Markdown paper draft from paper_results.json.",
    )
    paper_draft_parser.add_argument("--results", type=Path, required=True)
    paper_draft_parser.add_argument("--output", type=Path, required=True)
    paper_draft_parser.add_argument("--artifacts-dir", type=Path)

    paper_latex_parser = subparsers.add_parser(
        "paper-latex",
        help="Generate an arXiv-style LaTeX draft from paper_results.json.",
    )
    paper_latex_parser.add_argument("--results", type=Path, required=True)
    paper_latex_parser.add_argument("--output", type=Path, required=True)
    paper_latex_parser.add_argument("--artifacts-dir", type=Path)

    card_parser = subparsers.add_parser(
        "make-card",
        help="Generate dataset or experiment Markdown cards.",
    )
    card_parser.add_argument("kind", choices=["dataset", "experiment"])
    card_parser.add_argument("--input", type=Path, required=True)
    card_parser.add_argument("--output", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-manifest":
        report = validate_manifest(args.path)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
        return 0 if report.ok else 1

    if args.command == "doctor":
        report = run_doctor(repo_root=args.repo_root, check_final_files=args.check_final_files)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
        return 0 if report.ok else 1

    if args.command == "labels":
        print("turn_labels:")
        for label in sorted(TURN_LABELS):
            print(f"  - {label}")
        print("action_labels:")
        for label in sorted(ACTION_LABELS):
            print(f"  - {label}")
        return 0

    if args.command == "eval-turn":
        records = load_manifest(args.dataset)
        predictor = _build_turn_predictor(args, dataset_parent=args.dataset.parent)
        policy = TurnPolicy(TurnPolicyConfig(complete_threshold=0.75))
        report = evaluate_turn_records(records, predictor=predictor, policy=policy)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(
                "\n".join(
                    [
                        f"accuracy: {report.classification.accuracy:.4f}",
                        f"macro_f1: {report.classification.macro_f1:.4f}",
                        f"false_complete_rate: {report.interaction['false_complete_rate']:.4f}",
                        f"premature_response_rate: {report.interaction['premature_response_rate']:.4f}",
                        f"missed_interrupt_rate: {report.interaction['missed_interrupt_rate']:.4f}",
                    ]
                )
            )
            if args.report:
                print(f"report: {args.report}")
        return 0

    if args.command == "predict-turn":
        try:
            records = load_manifest(args.dataset)
            predictor = _build_turn_predictor(args, dataset_parent=args.dataset.parent)
            rows = export_turn_predictions_jsonl(records, predictor, args.output)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        payload = {"dataset": str(args.dataset), "output": str(args.output), "records": len(rows)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"predictions: {args.output}")
            print(f"records: {len(rows)}")
        return 0

    if args.command == "validate-turn-predictions":
        try:
            report = validate_turn_prediction_jsonl(
                load_turn_records(args.dataset, format=args.format),
                args.predictions,
                allow_extra=args.allow_extra,
                dataset_path=args.dataset,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
            if args.report:
                print(f"report: {args.report}")
        return 0 if report.ok else 1

    if args.command == "compare-turn":
        try:
            records = load_manifest(args.dataset)
            predictors = _build_turn_comparison_predictors(args, dataset_parent=args.dataset.parent)
            report = compare_turn_predictors(
                records,
                predictors,
                dataset=str(args.dataset),
                policy=TurnPolicy(TurnPolicyConfig(complete_threshold=0.75)),
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            for row in report.rows:
                print(
                    " ".join(
                        [
                            f"name={row.name}",
                            f"kind={row.kind}",
                            f"records={row.records}",
                            f"accuracy={row.accuracy:.4f}",
                            f"macro_f1={row.macro_f1:.4f}",
                            f"false_complete_rate={row.false_complete_rate:.4f}",
                            f"failures={row.failures}",
                        ]
                    )
                )
            if args.report:
                print(f"report: {args.report}")
        return 0

    if args.command == "compare-turn-splits":
        try:
            split_records = {
                "train": load_manifest(args.train),
                "dev": load_manifest(args.dev),
                "test": load_manifest(args.test),
            }
            predictors = _build_turn_comparison_predictors(args, dataset_parent=args.train.parent)
            report = compare_turn_predictors_on_splits(
                split_records,
                predictors,
                policy=TurnPolicy(TurnPolicyConfig(complete_threshold=0.75)),
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            for row in report.rows():
                print(
                    " ".join(
                        [
                            f"split={row['split']}",
                            f"name={row['name']}",
                            f"kind={row['kind']}",
                            f"records={row['records']}",
                            f"accuracy={float(row['accuracy']):.4f}",
                            f"macro_f1={float(row['macro_f1']):.4f}",
                            f"false_complete_rate={float(row['false_complete_rate']):.4f}",
                            f"failures={row['failures']}",
                        ]
                    )
                )
            if args.report:
                print(f"report: {args.report}")
        return 0

    if args.command == "benchmark-turn":
        records = load_manifest(args.dataset)
        predictor = _build_turn_predictor(args, dataset_parent=args.dataset.parent)
        artifact_paths = list(args.artifact)
        if args.checkpoint:
            artifact_paths.append(args.checkpoint)
        if args.predictions:
            artifact_paths.append(args.predictions)
        try:
            report = benchmark_turn_predictor(
                records,
                predictor,
                warmup=args.warmup,
                repeat=args.repeat,
                artifact_paths=artifact_paths,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(
                "\n".join(
                    [
                        f"records: {report.records}",
                        f"predictions: {report.predictions}",
                        f"avg_latency_ms: {report.avg_latency_ms:.4f}",
                        f"p50_latency_ms: {report.p50_latency_ms:.4f}",
                        f"p95_latency_ms: {report.p95_latency_ms:.4f}",
                        f"throughput_predictions_per_sec: {report.throughput_predictions_per_sec:.2f}",
                        f"rtf: {report.rtf:.6f}",
                    ]
                )
            )
            for path, size in report.artifact_bytes.items():
                print(f"artifact_bytes[{path}]: {size}")
            if args.report:
                print(f"report: {args.report}")
        return 0

    if args.command == "make-synthetic-turn-data":
        records = write_synthetic_turn_manifest(
            args.output,
            episodes=args.episodes,
            seed=args.seed,
            language=args.language,
            write_audio=args.write_audio,
        )
        print(f"wrote {len(records)} record(s) to {args.output}")
        return 0

    if args.command == "convert":
        try:
            count = convert_turn_manifest(
                args.source,
                args.dest,
                source_format=args.source_format,
                dest_format=args.dest_format,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"converted {count} record(s) from {args.source} to {args.dest}")
        return 0

    if args.command == "convert-external":
        try:
            count = convert_external_jsonl(
                args.input,
                args.output,
                schema=args.schema,
                default_sample_rate=args.sample_rate,
                default_language=args.language,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"converted {count} external record(s) from {args.input} to {args.output}")
        return 0

    if args.command == "convert-predictions":
        try:
            count = convert_turn_prediction_jsonl(
                args.input,
                args.output,
                schema=args.schema,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"converted {count} prediction record(s) from {args.input} to {args.output}")
        return 0

    if args.command == "convert-asr-transcript":
        try:
            count = convert_streaming_asr_jsonl(args.input, args.output, schema=args.schema)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"converted {count} ASR transcript record(s) from {args.input} to {args.output}")
        return 0

    if args.command == "inspect-manifest":
        summary = summarize_records(load_turn_records(args.path, format=args.format))
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"records: {summary['records']}")
            for key in ("turn_labels", "action_labels", "scenarios", "languages"):
                print(f"{key}:")
                values = summary[key]
                if isinstance(values, dict):
                    for name, count in values.items():
                        print(f"  {name}: {count}")
        return 0

    if args.command == "profile-turn-data":
        try:
            profile = profile_turn_records(
                load_turn_records(args.dataset, format=args.format),
                min_records=args.min_records,
                warn_label_imbalance=args.warn_label_imbalance,
                require_all_turn_labels=args.require_all_turn_labels,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(profile.to_markdown(), encoding="utf-8")
        if args.json:
            print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(profile.to_text())
            if args.report:
                print(f"report: {args.report}")
        return 0

    if args.command == "split-turn-data":
        try:
            config = TurnSplitConfig(
                train_ratio=args.train_ratio,
                dev_ratio=args.dev_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                stratify_by=() if args.no_stratify else tuple(args.stratify_by or ["turn_label"]),
                group_by=args.group_by,
                ensure_non_empty=not args.allow_empty,
            )
            result = split_turn_records(load_turn_records(args.input), config=config)
            suffix = ".jsonl" if args.format == "jsonl" else ".parquet" if args.format == "parquet" else ".lance"
            output_paths = {
                name: args.output_dir / f"{args.prefix}_{name}{suffix}"
                for name in ("train", "dev", "test")
            }
            for name, path in output_paths.items():
                write_turn_records(path, result.split(name), format=args.format)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        payload = result.to_dict()
        payload["outputs"] = {name: str(path) for name, path in output_paths.items()}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(result.to_text())
            for name, path in output_paths.items():
                print(f"{name}_path: {path}")
        return 0

    if args.command == "audit-turn-splits":
        try:
            report = audit_turn_splits(
                {
                    "train": load_turn_records(args.train),
                    "dev": load_turn_records(args.dev),
                    "test": load_turn_records(args.test),
                },
                leakage_fields=tuple(args.field or DEFAULT_LEAKAGE_FIELDS),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_text() + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
            if args.report:
                print(f"report: {args.report}")
        return 0 if report.ok else 1

    if args.command == "benchmark-data":
        try:
            rows = benchmark_data_formats(
                load_turn_records(args.dataset),
                output_dir=args.output_dir,
                formats=args.formats,
                sample_count=args.sample_count,
                seed=args.seed,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        payload = [row.to_dict() for row in rows]
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                print(
                    " ".join(
                        [
                            f"format={row.format}",
                            f"records={row.records}",
                            f"write_seconds={row.write_seconds:.6f}",
                            f"read_seconds={row.read_seconds:.6f}",
                            f"size_bytes={row.size_bytes}",
                            f"sample_count={row.sample_count}",
                            f"sample_seconds={row.sample_seconds:.6f}",
                            f"samples_per_second={row.samples_per_second:.2f}",
                            f"sample_strategy={row.sample_strategy}",
                            f"path={row.output_path}",
                        ]
                    )
                )
        return 0

    if args.command == "data-sources":
        try:
            registry = load_data_sources(args.registry)
            validation = validate_data_sources(registry)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.validate_only:
                print(f"OK: {registry['id']} ({len(registry['sources'])} source(s))")
                return 0
            text = json.dumps(registry, ensure_ascii=False, indent=2) if args.json else data_sources_markdown(registry)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "adapter-registry":
        try:
            registry = load_adapter_registry(args.registry)
            validation = validate_adapter_registry(registry)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.validate_only:
                print(f"OK: {registry['id']} ({len(registry['adapters'])} adapter(s))")
                return 0
            text = json.dumps(registry, ensure_ascii=False, indent=2) if args.json else adapter_registry_markdown(registry)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "asr-collections":
        try:
            registry = load_asr_collections(args.registry)
            validation = validate_asr_collections(registry)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.audit_coverage:
                coverage = audit_asr_collection_coverage(
                    registry,
                    load_adapter_registry(args.adapter_registry),
                    required_priorities=tuple(args.require_priority or ["p0"]),
                )
                text = json.dumps(coverage.to_dict(), ensure_ascii=False, indent=2) if args.json else coverage.to_markdown()
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
                print(text)
                return 0 if coverage.ok else 1
            if args.validate_only:
                print(f"OK: {registry['id']} ({len(registry['entries'])} reference(s))")
                return 0
            text = json.dumps(registry, ensure_ascii=False, indent=2) if args.json else asr_collections_markdown(registry)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "roadmap-status":
        try:
            roadmap = load_roadmap(args.roadmap)
            validation = validate_roadmap(roadmap)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.validate_only:
                print(f"OK: {roadmap['id']} ({len(roadmap['milestones'])} milestone(s))")
                return 0
            report = roadmap_status(roadmap, repo_root=args.repo_root)
            text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0 if report.ok else 1

    if args.command == "prepare-asr-manifest":
        try:
            records = prepare_asr_manifest(
                args.input,
                args.output,
                audio_root=args.audio_root,
                default_sample_rate=args.sample_rate,
                default_language=args.language,
                default_source=args.source,
                default_split=args.split,
                id_field=args.id_field,
                audio_field=args.audio_field,
                text_field=args.text_field,
                duration_field=args.duration_field,
                speaker_field=args.speaker_field,
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        summary = summarize_asr_records(records)
        if args.json:
            print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))
        else:
            print(f"wrote {len(records)} ASR record(s) to {args.output}")
            print(f"languages: {json.dumps(summary['languages'], ensure_ascii=False, sort_keys=True)}")
            print(f"sources: {json.dumps(summary['sources'], ensure_ascii=False, sort_keys=True)}")
        return 0

    if args.command == "validate-asr-manifest":
        report = validate_asr_manifest(args.path)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
        return 0 if report.ok else 1

    if args.command == "inspect-asr-manifest":
        try:
            summary = summarize_asr_records(load_asr_manifest(args.path))
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"records: {summary['records']}")
            print(f"total_duration_sec: {summary['total_duration_sec']}")
            for key in ("sample_rates", "languages", "sources", "splits"):
                print(f"{key}:")
                values = summary[key]
                if isinstance(values, dict):
                    for name, count in values.items():
                        print(f"  {name}: {count}")
            print(f"speakers: {summary['speakers']}")
        return 0

    if args.command == "asr-to-turn":
        try:
            result = asr_records_to_turn_records(
                load_asr_manifest(args.input),
                config=ASRToTurnConfig(
                    window_sec=args.window_sec,
                    include_complete=not args.no_complete,
                    include_incomplete=args.include_incomplete,
                    incomplete_ratio=args.incomplete_ratio,
                    min_incomplete_sec=args.min_incomplete_sec,
                    complete_pause_ms=args.complete_pause_ms,
                    incomplete_pause_ms=args.incomplete_pause_ms,
                    source=args.source,
                    drop_incomplete_text=not args.keep_incomplete_text,
                ),
            )
            write_turn_records(args.output, result.records, format=args.format)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        payload = result.to_dict()
        payload["output"] = str(args.output)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(result.to_text())
            print(f"output: {args.output}")
        return 0

    if args.command == "bootstrap-turn-data":
        try:
            result = bootstrap_turn_data(
                args.input,
                config=BootstrapTurnDataConfig(
                    output_dir=args.output_dir,
                    turn_format=args.turn_format,
                    split_prefix=args.split_prefix,
                ),
                audio_root=args.audio_root,
                default_sample_rate=args.sample_rate,
                default_language=args.language,
                default_source=args.source,
                default_split=args.split,
                id_field=args.id_field,
                audio_field=args.audio_field,
                text_field=args.text_field,
                duration_field=args.duration_field,
                speaker_field=args.speaker_field,
                asr_to_turn_config=ASRToTurnConfig(
                    window_sec=args.window_sec,
                    include_incomplete=args.include_incomplete,
                    incomplete_ratio=args.incomplete_ratio,
                ),
                split_config=TurnSplitConfig(
                    train_ratio=args.train_ratio,
                    dev_ratio=args.dev_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.seed,
                    stratify_by=("turn_label",),
                    group_by=None if args.no_group_by else args.group_by,
                ),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(result.to_text())
        return 0

    if args.command == "audit-audio":
        try:
            records = load_manifest(args.manifest) if args.kind == "turn" else load_asr_manifest(args.manifest)
            report = audit_audio_records(
                records,
                kind=args.kind,
                audio_root=args.audio_root,
                manifest_path=args.manifest,
                duration_tolerance_sec=args.duration_tolerance_sec,
                require_inspectable=args.require_inspectable,
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_text() + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
            if args.report:
                print(f"report: {args.report}")
        return 0 if report.ok else 1

    if args.command == "eval-streaming-asr":
        report = evaluate_streaming_records(load_streaming_transcript_jsonl(args.input))
        payload = report.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for key, value in payload.items():
                if key == "failure_analysis" and isinstance(value, dict):
                    print(f"streaming_failures: {value.get('total_failures', 0)}")
                    print(
                        "streaming_failure_categories: "
                        + json.dumps(value.get("category_counts", {}), ensure_ascii=False, sort_keys=True)
                    )
                    continue
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        return 0

    if args.command == "compare-streaming-asr":
        try:
            report = compare_streaming_transcript_jsonl([_parse_named_path(item) for item in args.input])
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        payload = report.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for row in payload["rows"]:
                if isinstance(row, dict):
                    print(
                        " ".join(
                            [
                                f"adapter={row['adapter']}",
                                f"records={row['records']}",
                                f"wer={float(row['wer']):.4f}",
                                f"rtf={float(row['rtf']):.4f}",
                                f"endpoint_delay={float(row['endpoint_delay']):.4f}",
                                f"timestamp_drift={float(row['timestamp_drift']):.4f}",
                            ]
                        )
                    )
            if args.report:
                print(f"report: {args.report}")
        return 0

    if args.command == "sweep-streaming-asr":
        try:
            report = sweep_streaming_schedule(
                load_streaming_transcript_jsonl(args.input),
                chunk_ms_values=args.chunks_ms,
                lookahead_ms_values=args.lookahead_ms,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        payload = report.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for row in payload["rows"]:
                if isinstance(row, dict):
                    print(
                        " ".join(
                            [
                                f"chunk_ms={row['chunk_ms']}",
                                f"lookahead_ms={row['lookahead_ms']}",
                                f"wer={float(row['wer']):.4f}",
                                f"first_partial_latency={float(row['first_partial_latency']):.4f}",
                                f"endpoint_delay={float(row['endpoint_delay']):.4f}",
                            ]
                        )
                    )
            if args.report:
                print(f"report: {args.report}")
        return 0

    if args.command == "eval-asr-command":
        adapter = CommandStreamingASRAdapter(
            name=args.name,
            command=args.asr_command,
            output_path=args.output,
            cwd=args.cwd,
            timeout_sec=args.timeout,
        )
        try:
            report = evaluate_streaming_records(adapter.load_records())
        except (RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        payload = {"adapter": args.name, "output": str(args.output), **report.to_dict()}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for key, value in payload.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        return 0

    if args.command == "compare-asr-commands":
        try:
            report = compare_asr_commands_from_config(args.config)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        payload = report.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for row in payload["rows"]:
                if isinstance(row, dict):
                    print(
                        " ".join(
                            [
                                f"adapter={row['adapter']}",
                                f"records={row['records']}",
                                f"wer={float(row['wer']):.4f}",
                                f"rtf={float(row['rtf']):.4f}",
                                f"endpoint_delay={float(row['endpoint_delay']):.4f}",
                                f"timestamp_drift={float(row['timestamp_drift']):.4f}",
                            ]
                        )
                    )
            if args.report:
                print(f"report: {args.report}")
        return 0

    if args.command == "eval-scenario":
        predictor = (
            NanoTurnCheckpointPredictor(args.checkpoint)
            if args.checkpoint
            else _build_baseline(args.baseline, complete_pause_ms=700)
        )
        report = evaluate_voice_world(
            predictor,
            episodes=args.episodes,
            seed=args.seed,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"suite: {report.suite}")
            print(f"records: {len(report.overall.examples)}")
            print(f"accuracy: {report.overall.classification.accuracy:.4f}")
            print(f"macro_f1: {report.overall.classification.macro_f1:.4f}")
            print("scenarios:")
            for scenario, scenario_report in report.by_scenario.items():
                print(
                    f"  {scenario}: records={len(scenario_report.examples)} "
                    f"accuracy={scenario_report.classification.accuracy:.4f} "
                    f"macro_f1={scenario_report.classification.macro_f1:.4f}"
                )
            if args.report:
                print(f"report: {args.report}")
        return 0

    if args.command == "scenario-suite":
        try:
            suite = load_scenario_suite(args.suite)
            validation = validate_scenario_suite(suite)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.validate_only:
                print(f"OK: {suite['id']} ({len(suite['scenarios'])} scenario(s))")
                return 0
            text = json.dumps(suite, ensure_ascii=False, indent=2) if args.json else scenario_suite_markdown(suite)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "optimize-policy":
        records = (
            load_manifest(args.dataset)
            if args.dataset
            else generate_synthetic_turn_records(
                episodes=args.episodes,
                seed=args.seed,
            )
        )
        predictor = _build_baseline(args.baseline, complete_pause_ms=700)
        result = threshold_search(records, predictor)
        payload = result.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            best = payload["best"]
            print(f"best_score: {best['score']:.4f}")
            print(f"best_config: {json.dumps(best['config'], ensure_ascii=False, sort_keys=True)}")
            print(f"trials: {len(payload['trials'])}")
            if args.output:
                print(f"output: {args.output}")
        return 0

    if args.command == "train-turn":
        result = train_nanoturn(
            load_manifest(args.dataset),
            output_dir=args.output_dir,
            model_type=args.model,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            feature_source=args.feature_source,
            audio_root=args.audio_root or args.dataset.parent,
        )
        if args.json:
            print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
        else:
            print(f"checkpoint: {result.checkpoint_path}")
            print(f"metrics: {result.metrics_path}")
            print(f"final_loss: {result.metrics['final_loss']:.6f}")
            print(f"final_accuracy: {result.metrics['final_accuracy']:.4f}")
        return 0

    if args.command == "export-turn-onnx":
        output = export_nanoturn_onnx(
            args.checkpoint,
            args.output,
            opset_version=args.opset,
        )
        print(f"onnx: {output}")
        return 0

    if args.command == "reproduce-paper":
        paper_config = _load_paper_config(args.config)
        output_dir = args.output_dir or _required_config_path(paper_config, "output_dir")
        episodes = args.episodes if args.episodes is not None else int(paper_config.get("episodes", 25))
        seed = args.seed if args.seed is not None else int(paper_config.get("seed", 0))
        train_model = bool(paper_config.get("train_model", True)) and not args.skip_train
        result = run_paper_smoke(
            output_dir,
            episodes=episodes,
            seed=seed,
            train_model=train_model,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "output_dir": result.output_dir,
                        "results_path": result.results_path,
                        "report_path": result.report_path,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"results: {result.results_path}")
            print(f"report: {result.report_path}")
        return 0

    if args.command == "paper-table":
        table = paper_table(args.results, args.table)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(table + "\n", encoding="utf-8")
        print(table)
        return 0

    if args.command == "paper-figure":
        output = paper_figure(args.results, args.figure, args.output)
        print(f"figure: {output}")
        return 0

    if args.command == "paper-bundle":
        bundle = paper_artifact_bundle(args.results, args.output_dir)
        if args.json:
            print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"artifact_index: {bundle.index_path}")
            print(f"manifest: {bundle.manifest_path}")
            print(f"tables: {len(bundle.tables)}")
            print(f"figures: {len(bundle.figures)}")
            print(f"leaderboards: {len(bundle.leaderboards)}")
            print(f"benchmark_suite: {len(bundle.benchmark_suite)}")
            print(f"data_sources: {len(bundle.data_sources)}")
            print(f"adapter_registry: {len(bundle.adapter_registry)}")
            print(f"scenario_suite: {len(bundle.scenario_suite)}")
            print(f"case_studies: {len(bundle.case_studies)}")
            print(f"paper_parity: {len(bundle.paper_parity)}")
            print(f"final_experiments: {len(bundle.final_experiments)}")
            print(f"final_run_config: {len(bundle.final_run_config)}")
            print(f"final_run_file_audit: {len(bundle.final_run_file_audit)}")
            print(f"paper_status: {len(bundle.paper_status)}")
            print(f"claims: {len(bundle.claims)}")
        return 0

    if args.command == "paper-status":
        try:
            report = paper_status(
                repo_root=args.repo_root,
                results_path=args.results,
                artifacts_dir=args.artifacts_dir,
            )
            if args.output:
                write_paper_status_markdown(report, args.output)
            if args.json:
                print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(report.to_markdown())
                if args.output:
                    print(f"paper_status_markdown: {args.output}")
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0 if report.ok else 1

    if args.command == "paper-case-studies":
        artifacts = paper_case_studies(args.results, args.output_dir)
        payload = artifacts.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"case_studies_json: {artifacts.json_path}")
            print(f"case_studies_markdown: {artifacts.markdown_path}")
        return 0

    if args.command == "paper-claim-audit":
        try:
            report = audit_claims(
                repo_root=args.repo_root,
                results_path=args.results,
                artifacts_dir=args.artifacts_dir,
            )
            artifacts = None
            if args.output_dir:
                if args.results is None:
                    raise ValueError("--output-dir requires --results")
                artifacts = paper_claims(args.results, args.output_dir, repo_root=args.repo_root)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            payload = report.to_dict()
            if artifacts is not None:
                payload["outputs"] = artifacts.to_dict()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
            if artifacts is not None:
                print(f"claims_json: {artifacts.json_path}")
                print(f"claims_markdown: {artifacts.markdown_path}")
        return 0 if report.ok else 1

    if args.command == "paper-parity-audit":
        try:
            checklist = load_paper_parity_checklist(args.checklist)
            validation = validate_paper_parity_checklist(checklist)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.validate_only:
                print(f"OK: {checklist['id']} ({len(checklist['items'])} item(s))")
                return 0
            report = audit_paper_parity(
                checklist=checklist,
                repo_root=args.repo_root,
                results_path=args.results,
                artifacts_dir=args.artifacts_dir,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(paper_parity_markdown(report), encoding="utf-8")
            if args.json:
                print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(report.to_text())
                if args.output:
                    print(f"paper_parity_markdown: {args.output}")
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0 if report.ok and (report.final_ready or not args.require_final) else 1

    if args.command == "final-experiments":
        try:
            registry = load_final_experiments(args.registry)
            validation = validate_final_experiments(registry)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.validate_only:
                print(f"OK: {registry['id']} ({len(registry['experiments'])} experiment(s))")
                return 0
            text = json.dumps(registry, ensure_ascii=False, indent=2) if args.json else final_experiments_markdown(registry)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "final-config":
        try:
            config = load_final_run_config(args.config)
            validation = validate_final_run_config(config)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.scaffold:
                report = scaffold_final_run(config, repo_root=args.repo_root)
                if args.json:
                    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(report.to_text())
                return 0
            if args.check_files:
                report = audit_final_run_files(config, repo_root=args.repo_root)
                text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else final_run_file_audit_markdown(report)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
                print(text)
                return 0 if report.ok else 1
            if args.validate_only:
                print(f"OK: {config['id']} ({len(config['public_corpora'])} corpora)")
                return 0
            text = json.dumps(config, ensure_ascii=False, indent=2) if args.json else final_run_config_markdown(config)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "leaderboard-export":
        try:
            output = export_leaderboard(args.results, args.output, format=args.format)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"leaderboard: {output}")
        return 0

    if args.command == "benchmark-suite":
        try:
            suite = load_benchmark_suite(args.suite)
            validation = validate_benchmark_suite(suite)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            coverage = None
            if args.results:
                coverage = audit_benchmark_suite_coverage(load_paper_results(args.results), suite=suite)
                if not coverage.ok:
                    print(coverage.to_text(), file=sys.stderr)
                    return 1
            if args.validate_only:
                suffix = f"; coverage=OK ({coverage.rows} row(s))" if coverage is not None else ""
                print(f"OK: {suite['id']} ({len(suite['tasks'])} task(s)){suffix}")
                return 0
            if args.json:
                text = json.dumps(suite, ensure_ascii=False, indent=2)
            else:
                text = benchmark_suite_markdown(suite)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "paper-audit":
        report = audit_paper_artifacts(args.results, args.artifacts_dir)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
        return 0 if report.ok else 1

    if args.command == "paper-release-audit":
        report = audit_paper_release(
            repo_root=args.repo_root,
            results_path=args.results,
            artifacts_dir=args.artifacts_dir,
            markdown_draft=args.markdown_draft,
            latex_draft=args.latex_draft,
            dataset_card=args.dataset_card,
            experiment_card=args.experiment_card,
        )
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
        return 0 if report.ok else 1

    if args.command == "paper-draft":
        output = paper_draft(args.results, args.output, artifacts_dir=args.artifacts_dir)
        print(f"draft: {output}")
        return 0

    if args.command == "paper-latex":
        output = paper_latex(args.results, args.output, artifacts_dir=args.artifacts_dir)
        print(f"latex: {output}")
        return 0

    if args.command == "make-card":
        if args.kind == "dataset":
            output = dataset_card(args.input, args.output)
        else:
            output = experiment_card(args.input, args.output)
        print(f"card: {output}")
        return 0

    parser.print_help(sys.stderr)
    return 2


def _build_baseline(name: str, *, complete_pause_ms: int):
    if name == "rule_endpoint":
        return RuleEndpointBaseline(complete_pause_ms=complete_pause_ms)
    if name == "vad_pause":
        return VADPauseBaseline(complete_pause_ms=complete_pause_ms)
    if name == "text_turn":
        return TextTurnBaseline()
    raise ValueError(f"unknown baseline: {name}")


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--input must use ADAPTER=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("--input must use ADAPTER=PATH")
    return name, Path(path)


def _build_turn_predictor(args, *, dataset_parent: Path):
    if getattr(args, "predictions", None) and getattr(args, "checkpoint", None):
        raise ValueError("--predictions and --checkpoint are mutually exclusive")
    if getattr(args, "predictions", None):
        return TurnPredictionManifestAdapter.from_jsonl(args.predictions)
    if getattr(args, "checkpoint", None):
        return NanoTurnCheckpointPredictor(
            args.checkpoint,
            audio_root=getattr(args, "audio_root", None) or dataset_parent,
        )
    return _build_baseline(args.baseline, complete_pause_ms=args.complete_pause_ms)


def _build_turn_comparison_predictors(args, *, dataset_parent: Path):
    predictors = []
    baselines = args.baseline or ["rule_endpoint", "vad_pause", "text_turn"]
    for baseline in baselines:
        predictors.append(
            (
                baseline,
                "baseline",
                _build_baseline(baseline, complete_pause_ms=args.complete_pause_ms),
            )
        )
    for item in args.predictions:
        name, path = _parse_named_path(item)
        predictors.append((name, "predictions", TurnPredictionManifestAdapter.from_jsonl(path)))
    for item in args.checkpoint:
        name, path = _parse_named_path(item)
        predictors.append(
            (
                name,
                "checkpoint",
                NanoTurnCheckpointPredictor(
                    path,
                    audio_root=args.audio_root or dataset_parent,
                ),
            )
        )
    return predictors


def _load_paper_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("paper config must be a JSON object")
    return payload


def _required_config_path(config: dict[str, object], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing required --{key.replace('_', '-')} or config field {key!r}")
    return Path(value)


if __name__ == "__main__":
    raise SystemExit(main())
