"""Command-line interface for Stable-ASR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stable_asr import __version__
from stable_asr.catalog import build_platform_catalog, write_platform_catalog_json, write_platform_catalog_markdown
from stable_asr.data.asr_manifest import load_asr_manifest, summarize_asr_records, validate_asr_manifest
from stable_asr.data.audio_audit import audit_audio_records
from stable_asr.data.bootstrap import BootstrapTurnDataConfig, bootstrap_turn_data
from stable_asr.data.manifest import load_manifest, validate_manifest
from stable_asr.data.profile import profile_turn_records
from stable_asr.data.audio_window_cache import WINDOW_CACHE_FORMATS, benchmark_audio_window_formats
from stable_asr.data.benchmark import benchmark_data_formats
from stable_asr.data.split_audit import DEFAULT_LEAKAGE_FIELDS, audit_turn_splits
from stable_asr.data.converters import (
    ASR_TRANSCRIPT_SCHEMAS,
    EXTERNAL_SCHEMAS,
    convert_external_jsonl,
    convert_streaming_asr_jsonl,
)
from stable_asr.data.sources import data_sources_markdown, load_data_sources, validate_data_sources
from stable_asr.data.recipes import (
    DEFAULT_VOICEWORLD_FACTOR_FIELDS,
    PUBLIC_ASR_CORPORA,
    prepare_asr_manifest,
    prepare_public_asr_manifest,
    prepare_voiceworld_manifest,
)
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
from stable_asr.models.registry import (
    audit_model_registry_configs,
    load_model_registry,
    model_registry_markdown,
    validate_model_registry,
)
from stable_asr.paper.artifacts import paper_artifact_bundle
from stable_asr.paper.audit import audit_paper_artifacts, audit_paper_release
from stable_asr.paper.cards import dataset_card, experiment_card, model_card
from stable_asr.paper.case_studies import paper_case_studies
from stable_asr.paper.claims import audit_claims, paper_claims
from stable_asr.paper.completion import completion_audit, write_completion_audit_markdown
from stable_asr.paper.draft import paper_draft
from stable_asr.paper.evidence import final_evidence_matrix
from stable_asr.paper.experiments import run_paper_smoke
from stable_asr.paper.final_config import (
    audit_final_voiceworld_real,
    audit_final_run_files,
    build_final_run_action_plan,
    bootstrap_final_turn_splits,
    final_run_file_audit_markdown,
    final_run_config_markdown,
    load_final_run_config,
    prepare_final_asr_eval_manifest,
    prepare_final_asr_transcript_conversions,
    prepare_final_external_predictions,
    prepare_final_corpora,
    prepare_final_inputs,
    prepare_final_voiceworld_real,
    scaffold_final_run,
    validate_final_run_config,
)
from stable_asr.paper.final_experiments import (
    final_experiments_markdown,
    load_final_experiments,
    validate_final_experiments,
)
from stable_asr.paper.final_inputs import (
    final_input_collection_report,
    load_final_input_collections,
    validate_final_input_collections,
)
from stable_asr.paper.final_pack import build_final_pack
from stable_asr.paper.final_results import assemble_final_paper_results
from stable_asr.paper.handoff import audit_final_handoff, final_handoff_template, populate_final_handoff_checksums
from stable_asr.paper.figures import PAPER_FIGURES, paper_figure
from stable_asr.paper.archive import paper_artifact_archive, verify_paper_artifact_archive
from stable_asr.paper.integrity import verify_artifact_integrity
from stable_asr.paper.leaderboard import (
    export_leaderboard,
    leaderboard_report,
    merge_leaderboard_jsonl,
    validate_leaderboard_jsonl,
)
from stable_asr.paper.latex import paper_latex
from stable_asr.paper.acquisition_pack import audit_acquisition_assignments, build_final_acquisition_pack
from stable_asr.paper.adapter_pack import build_adapter_pack
from stable_asr.paper.benchmark_pack import build_benchmark_pack
from stable_asr.paper.contributor_pack import build_contributor_pack
from stable_asr.paper.release_smoke import run_paper_release_smoke
from stable_asr.paper.parity import (
    audit_paper_parity,
    load_paper_parity_checklist,
    paper_parity_markdown,
    validate_paper_parity_checklist,
)
from stable_asr.paper.platform_parity import (
    audit_platform_parity,
    load_platform_parity,
    validate_platform_parity,
)
from stable_asr.paper.scenario_pack import build_scenario_pack
from stable_asr.paper.status import paper_status, write_paper_status_markdown
from stable_asr.paper.submissions import build_streaming_submission, build_turn_submission, index_submission_directory
from stable_asr.paper.suites import (
    audit_benchmark_required_artifacts,
    audit_benchmark_suite_coverage,
    benchmark_suite_markdown,
    load_benchmark_suite,
    validate_benchmark_suite,
)
from stable_asr.paper.tables import PAPER_TABLES, load_paper_results, paper_table
from stable_asr.references import (
    asr_collections_acquisition_markdown,
    asr_collections_bibtex,
    asr_collections_markdown,
    asr_collections_reference_markdown,
    asr_collections_source_manifest,
    audit_reference_assignments,
    audit_reference_workqueue_evidence,
    audit_asr_collection_coverage,
    audit_asr_collection_licenses,
    audit_asr_collection_readiness,
    audit_turn_collection_coverage,
    load_asr_collections,
    load_turn_collections,
    reference_workqueue_assignments,
    reference_workqueue_assignments_markdown,
    reference_workqueue_assignments_tsv,
    reference_workqueue_evidence_markdown,
    reference_workqueue_from_registries,
    reference_workqueue_issues_markdown,
    reference_workqueue_license_review_markdown,
    reference_workqueue_jsonl,
    reference_workqueue_markdown,
    turn_collections_acquisition_markdown,
    turn_collections_markdown,
    turn_collections_source_manifest,
    validate_asr_collections,
    validate_turn_collections,
)
from stable_asr.resources import resolve_platform_path
from stable_asr.roadmap import load_roadmap, roadmap_status, validate_roadmap
from stable_asr.scenarios.voice_world import evaluate_voice_world, evaluate_voice_world_records
from stable_asr.scenarios.synthetic_turn import generate_synthetic_turn_records, write_synthetic_turn_manifest
from stable_asr.scenarios.suites import (
    load_scenario_suite,
    scenario_suite_markdown,
    validate_scenario_suite,
)
from stable_asr.schemas import (
    get_schema_entry,
    load_schema_registry,
    schema_entry_markdown,
    schema_registry_markdown,
    validate_schema_registry,
)
from stable_asr.schema_validation import JSON_FORMATS, validate_schema_file
from stable_asr.streaming.command_compare import audit_asr_command_config, compare_asr_commands_from_config
from stable_asr.streaming.compare import compare_streaming_transcript_jsonl
from stable_asr.streaming.metrics import evaluate_streaming_records
from stable_asr.streaming.sweep import sweep_streaming_schedule
from stable_asr.train.export import export_nanoturn_onnx
from stable_asr.train.feature_cache import TRAIN_FEATURE_BENCHMARK_FORMATS, benchmark_train_feature_cache
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
    doctor_parser.add_argument(
        "--check-release-env",
        action="store_true",
        help="Fail if optional Torch and Lance dependencies needed for READY release smoke are missing.",
    )
    doctor_parser.add_argument("--json", action="store_true")

    catalog_parser = subparsers.add_parser(
        "catalog",
        help="Print a one-page catalog of Stable-ASR platform assets.",
    )
    catalog_parser.add_argument("--repo-root", type=Path, default=Path("."))
    catalog_parser.add_argument("--output", type=Path, help="Optional Markdown or JSON output path.")
    catalog_parser.add_argument("--json", action="store_true")

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
    eval_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

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
    validate_predictions_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

    turn_submission_parser = subparsers.add_parser(
        "turn-submission",
        help="Build an auditable external turn prediction submission package.",
    )
    turn_submission_parser.add_argument("--dataset", type=Path, required=True, help="Turn benchmark manifest.")
    turn_submission_parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSONL to submit.")
    turn_submission_parser.add_argument("--system", required=True, help="System name shown in leaderboard rows.")
    turn_submission_parser.add_argument("--output-dir", type=Path, required=True)
    turn_submission_parser.add_argument("--format", choices=TURN_FORMATS.names(), help="Optional dataset format.")
    turn_submission_parser.add_argument("--allow-extra", action="store_true", help="Allow prediction IDs outside the dataset.")
    turn_submission_parser.add_argument("--complete-threshold", type=float, default=0.75)
    turn_submission_parser.add_argument("--suite", type=Path, help="Optional benchmark suite JSON path.")
    turn_submission_parser.add_argument("--json", action="store_true")

    submission_index_parser = subparsers.add_parser(
        "submission-index",
        help="Discover submission packages, write an index, and merge their leaderboard rows.",
    )
    submission_index_parser.add_argument("--root", type=Path, required=True, help="Directory containing submission packages.")
    submission_index_parser.add_argument("--output-dir", type=Path, required=True)
    submission_index_parser.add_argument("--suite", type=Path, help="Optional benchmark suite JSON path.")
    submission_index_parser.add_argument("--top-k", type=int, default=3)
    submission_index_parser.add_argument("--json", action="store_true")
    submission_index_parser.add_argument("--require-known-systems", action="store_true")
    submission_index_parser.add_argument("--require-known-slices", action="store_true")
    submission_index_parser.add_argument("--require-complete-suite", action="store_true")

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
    compare_turn_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

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
    compare_turn_splits_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

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
    benchmark_turn_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

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
    benchmark_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

    audio_window_benchmark_parser = subparsers.add_parser(
        "benchmark-audio-windows",
        help="Benchmark source WAV windows against materialized Parquet/Lance audio-window caches.",
    )
    audio_window_benchmark_parser.add_argument("--dataset", type=Path, required=True)
    audio_window_benchmark_parser.add_argument("--format", choices=TURN_FORMATS.names())
    audio_window_benchmark_parser.add_argument("--output-dir", type=Path, required=True)
    audio_window_benchmark_parser.add_argument(
        "--formats",
        nargs="+",
        choices=WINDOW_CACHE_FORMATS,
        default=list(WINDOW_CACHE_FORMATS),
    )
    audio_window_benchmark_parser.add_argument("--sample-count", type=int, default=1000)
    audio_window_benchmark_parser.add_argument("--seed", type=int, default=0)
    audio_window_benchmark_parser.add_argument("--max-records", type=int)
    audio_window_benchmark_parser.add_argument("--audio-root", type=Path)
    audio_window_benchmark_parser.add_argument(
        "--correctness-sample-count",
        type=int,
        default=32,
        help="Number of materialized random windows to reload from source WAV for allclose validation.",
    )
    audio_window_benchmark_parser.add_argument(
        "--correctness-tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for materialized audio-window correctness checks.",
    )
    audio_window_benchmark_parser.add_argument("--json", action="store_true")
    audio_window_benchmark_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

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

    model_registry_parser = subparsers.add_parser(
        "model-registry",
        help="Print or validate the Stable-ASR built-in model registry.",
    )
    model_registry_parser.add_argument("--registry", type=Path, help="Optional model registry JSON path.")
    model_registry_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    model_registry_parser.add_argument("--json", action="store_true", help="Print registry as JSON.")
    model_registry_parser.add_argument("--validate-only", action="store_true")
    model_registry_parser.add_argument(
        "--audit-configs",
        action="store_true",
        help="Audit trainable model config paths referenced by the registry.",
    )

    schema_registry_parser = subparsers.add_parser(
        "schema-registry",
        help="Print or validate the Stable-ASR JSON Schema registry.",
    )
    schema_registry_parser.add_argument("--registry", type=Path, help="Optional schema registry JSON path.")
    schema_registry_parser.add_argument("--schema-id", help="Print one schema entry instead of the full registry.")
    schema_registry_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    schema_registry_parser.add_argument("--json", action="store_true", help="Print registry or schema entry as JSON.")
    schema_registry_parser.add_argument("--validate-only", action="store_true")

    schema_file_parser = subparsers.add_parser(
        "validate-schema-file",
        help="Validate a JSON or JSONL file against one Stable-ASR schema registry entry.",
    )
    schema_file_parser.add_argument("--input", type=Path, required=True)
    schema_file_parser.add_argument("--schema-id", required=True)
    schema_file_parser.add_argument("--registry", type=Path, help="Optional schema registry JSON path.")
    schema_file_parser.add_argument("--format", choices=sorted(JSON_FORMATS), default="auto")
    schema_file_parser.add_argument("--max-errors", type=int, default=50)
    schema_file_parser.add_argument("--output", type=Path, help="Optional Markdown or JSON report path.")
    schema_file_parser.add_argument("--json", action="store_true")

    asr_collections_parser = subparsers.add_parser(
        "asr-collections",
        help="Print or validate the curated upstream ASR reference collection.",
    )
    asr_collections_parser.add_argument("--registry", type=Path, help="Optional ASR collections JSON path.")
    asr_collections_parser.add_argument("--output", type=Path, help="Optional output path.")
    asr_collections_parser.add_argument("--json", action="store_true", help="Print registry as JSON.")
    asr_collections_parser.add_argument(
        "--format",
        choices=[
            "registry-markdown",
            "paper-markdown",
            "bibtex",
            "acquisition-markdown",
            "license-markdown",
            "source-manifest",
        ],
        default="registry-markdown",
        help="Output format when not using --json or audit flags.",
    )
    asr_collections_parser.add_argument("--validate-only", action="store_true")
    asr_collections_parser.add_argument("--audit-coverage", action="store_true")
    asr_collections_parser.add_argument("--audit-licenses", action="store_true")
    asr_collections_parser.add_argument("--audit-readiness", action="store_true")
    asr_collections_parser.add_argument(
        "--adapter-registry",
        type=Path,
        help="Adapter registry used by --audit-coverage or --audit-readiness.",
    )
    asr_collections_parser.add_argument(
        "--max-review-age-days",
        type=int,
        default=3650,
        help="Maximum registry review age for --audit-readiness. Use a large value to disable practical freshness failure.",
    )
    asr_collections_parser.add_argument(
        "--require-priority",
        action="append",
        default=None,
        choices=["p0", "p1", "p2"],
        help="Reference priority required by collection audits. Defaults to p0 for coverage and p0+p1 for readiness/license audits.",
    )
    asr_collections_parser.add_argument(
        "--require-license-reviewed",
        action="store_true",
        help="Fail --audit-licenses when required references still need manual license review.",
    )

    turn_collections_parser = subparsers.add_parser(
        "turn-collections",
        help="Print or validate the curated turn-taking and full-duplex reference collection.",
    )
    turn_collections_parser.add_argument("--registry", type=Path, help="Optional turn collections JSON path.")
    turn_collections_parser.add_argument("--output", type=Path, help="Optional output path.")
    turn_collections_parser.add_argument("--json", action="store_true", help="Print registry as JSON.")
    turn_collections_parser.add_argument(
        "--format",
        choices=["registry-markdown", "acquisition-markdown", "source-manifest"],
        default="registry-markdown",
        help="Output format when not using --json or audit flags.",
    )
    turn_collections_parser.add_argument("--validate-only", action="store_true")
    turn_collections_parser.add_argument("--audit-coverage", action="store_true")
    turn_collections_parser.add_argument("--data-sources", type=Path, help="Data source registry for coverage audit.")
    turn_collections_parser.add_argument("--adapter-registry", type=Path, help="Adapter registry for coverage audit.")
    turn_collections_parser.add_argument(
        "--require-priority",
        action="append",
        default=None,
        choices=["p0", "p1", "p2"],
        help="Reference priority required by --audit-coverage. Defaults to p0.",
    )

    reference_workqueue_parser = subparsers.add_parser(
        "reference-workqueue",
        help="Merge ASR and turn reference source manifests into a contributor work queue.",
    )
    reference_workqueue_parser.add_argument("--asr-registry", type=Path, help="Optional ASR collections JSON path.")
    reference_workqueue_parser.add_argument("--turn-registry", type=Path, help="Optional turn collections JSON path.")
    reference_workqueue_parser.add_argument("--output", type=Path, help="Optional output path.")
    reference_workqueue_parser.add_argument("--json", action="store_true", help="Print the work queue or evidence audit as JSON.")
    reference_workqueue_parser.add_argument(
        "--format",
        choices=[
            "markdown",
            "json",
            "jsonl",
            "assignments-json",
            "assignments-tsv",
            "assignments-markdown",
            "evidence-markdown",
            "issues-markdown",
            "license-review-markdown",
        ],
        default="markdown",
        help="Output format for the unified reference work queue.",
    )
    reference_workqueue_parser.add_argument(
        "--require-priority",
        action="append",
        default=None,
        choices=["p0", "p1", "p2"],
        help="Reference priorities to include. Defaults to p0 and p1.",
    )
    reference_workqueue_parser.add_argument(
        "--audit-evidence",
        action="store_true",
        help="Audit whether workqueue evidence targets and required license-review files exist.",
    )
    reference_workqueue_parser.add_argument(
        "--require-content",
        action="store_true",
        help="With --audit-evidence, require present evidence and license-review files to contain usable content.",
    )
    reference_workqueue_parser.add_argument("--repo-root", type=Path, default=Path("."))

    reference_assignment_audit_parser = subparsers.add_parser(
        "reference-assignment-audit",
        help="Audit a filled reference assignment tracker for owners, due dates, blockers, evidence, and license reviews.",
    )
    reference_assignment_audit_parser.add_argument("--input", type=Path, required=True)
    reference_assignment_audit_parser.add_argument("--repo-root", type=Path, default=Path("."))
    reference_assignment_audit_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    reference_assignment_audit_parser.add_argument("--require-owner", action="store_true")
    reference_assignment_audit_parser.add_argument("--require-due-date", action="store_true")
    reference_assignment_audit_parser.add_argument("--require-ready", action="store_true")
    reference_assignment_audit_parser.add_argument("--json", action="store_true")

    roadmap_parser = subparsers.add_parser(
        "roadmap-status",
        help="Validate and summarize the machine-readable Stable-ASR roadmap.",
    )
    roadmap_parser.add_argument("--roadmap", type=Path, help="Optional roadmap registry JSON path.")
    roadmap_parser.add_argument("--repo-root", type=Path, default=Path("."))
    roadmap_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    roadmap_parser.add_argument("--json", action="store_true")
    roadmap_parser.add_argument("--validate-only", action="store_true")
    roadmap_parser.add_argument(
        "--require-final-ready",
        action="store_true",
        help="Fail when final-scale inputs, experiments, or artifact evidence are still missing.",
    )

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

    public_asr_parser = subparsers.add_parser(
        "prepare-public-asr",
        help="Prepare an ASR manifest from a supported public corpus directory.",
    )
    public_asr_parser.add_argument("--corpus", choices=PUBLIC_ASR_CORPORA, required=True)
    public_asr_parser.add_argument("--input-dir", type=Path, required=True)
    public_asr_parser.add_argument("--output", type=Path, required=True)
    public_asr_parser.add_argument("--split", help="Optional corpus split/subset filter, for example dev-clean or dev.")
    public_asr_parser.add_argument("--sample-rate", type=int, default=16000)
    public_asr_parser.add_argument("--json", action="store_true", help="Print summary as JSON.")

    voiceworld_prepare_parser = subparsers.add_parser(
        "prepare-voiceworld",
        help="Normalize real VoiceWorld TSV/CSV/JSONL annotations into a Stable-ASR turn manifest.",
    )
    voiceworld_prepare_parser.add_argument("--input", type=Path, required=True, help="Input VoiceWorld metadata table.")
    voiceworld_prepare_parser.add_argument("--output", type=Path, required=True, help="Output turn manifest JSONL path.")
    voiceworld_prepare_parser.add_argument("--audio-root", type=Path, help="Optional root joined to relative audio paths.")
    voiceworld_prepare_parser.add_argument("--sample-rate", type=int, default=16000, help="Default sample rate.")
    voiceworld_prepare_parser.add_argument("--language", default="unknown", help="Default language tag.")
    voiceworld_prepare_parser.add_argument("--source", default="voiceworld_real", help="Default source name.")
    voiceworld_prepare_parser.add_argument("--id-field", help="Override input id column/key.")
    voiceworld_prepare_parser.add_argument("--audio-field", help="Override input audio column/key.")
    voiceworld_prepare_parser.add_argument("--text-field", help="Override input text column/key.")
    voiceworld_prepare_parser.add_argument("--scenario-field", help="Override input scenario column/key.")
    voiceworld_prepare_parser.add_argument("--turn-label-field", help="Override input turn label column/key.")
    voiceworld_prepare_parser.add_argument("--action-label-field", help="Override input action label column/key.")
    voiceworld_prepare_parser.add_argument(
        "--factor-field",
        action="append",
        default=None,
        help="Metadata/factor field to preserve. Defaults to the VoiceWorld v0 factor set.",
    )
    voiceworld_prepare_parser.add_argument("--json", action="store_true", help="Print summary as JSON.")

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
    streaming_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

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
    streaming_compare_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

    streaming_submission_parser = subparsers.add_parser(
        "streaming-submission",
        help="Build an auditable external streaming ASR submission package.",
    )
    streaming_submission_parser.add_argument("--input", type=Path, required=True, help="Stable-ASR streaming ASR JSONL.")
    streaming_submission_parser.add_argument("--system", required=True, help="System name shown in leaderboard rows.")
    streaming_submission_parser.add_argument("--output-dir", type=Path, required=True)
    streaming_submission_parser.add_argument("--slice", default="submission", help="Leaderboard slice name.")
    streaming_submission_parser.add_argument("--suite", type=Path, help="Optional benchmark suite JSON path.")
    streaming_submission_parser.add_argument("--json", action="store_true")

    streaming_sweep_parser = subparsers.add_parser(
        "sweep-streaming-asr",
        help="Sweep chunk size and lookahead settings over a streaming ASR transcript JSONL.",
    )
    streaming_sweep_parser.add_argument("--input", type=Path, required=True)
    streaming_sweep_parser.add_argument("--chunks-ms", type=int, nargs="+", default=[160, 320, 640])
    streaming_sweep_parser.add_argument("--lookahead-ms", type=int, nargs="+", default=[0, 160, 320])
    streaming_sweep_parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    streaming_sweep_parser.add_argument("--json", action="store_true")
    streaming_sweep_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

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
    command_asr_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

    command_compare_parser = subparsers.add_parser(
        "compare-asr-commands",
        help="Compare multiple command-backed ASR adapters from a JSON config.",
    )
    command_compare_parser.add_argument("--config", type=Path, required=True)
    command_compare_parser.add_argument("--report", type=Path, help="Optional Markdown report output path.")
    command_compare_parser.add_argument("--json", action="store_true")
    command_compare_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")
    command_compare_parser.add_argument("--validate-only", action="store_true", help="Audit config without executing commands.")
    command_compare_parser.add_argument("--repo-root", type=Path, default=Path("."))
    command_compare_parser.add_argument("--min-adapters", type=int, default=1)
    command_compare_parser.add_argument("--require-input-manifest", action="store_true")

    scenario_parser = subparsers.add_parser(
        "eval-scenario",
        help="Evaluate a turn predictor on a VoiceWorld manifest or the seedable mini-suite.",
    )
    scenario_parser.add_argument("--dataset", type=Path, help="Optional VoiceWorld turn manifest to evaluate.")
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
    scenario_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

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
    train_parser.add_argument("--dev-dataset", type=Path, help="Optional validation manifest path.")
    train_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    train_parser.add_argument("--config", type=Path, help="Optional NanoTurn training config JSON.")
    train_parser.add_argument(
        "--model",
        choices=["nanoturn_pico", "nanoturn_nano", "nanoturn_pico_v1", "nanoturn_nano_v1", "nanoturn_micro"],
        default=None,
        help="NanoTurn model size. v1 variants use logmel_v1 160-dim features. nanoturn_micro is a TCN sequence model.",
    )
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--lr", type=float)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--validation-split", type=float)
    train_parser.add_argument(
        "--validation-group-by",
        default=None,
        help=(
            "Field kept together for internal validation_split. Defaults to auto, "
            "which uses metadata.asr_record_id, metadata.conversation_id, or audio when duplicated. "
            "Use 'none' to disable grouping."
        ),
    )
    train_parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"])
    train_parser.add_argument("--weight-decay", type=float)
    train_parser.add_argument("--gradient-clip-norm", type=float)
    train_parser.add_argument("--checkpoint-interval", type=int)
    train_parser.add_argument("--resume-from", type=Path)
    train_parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        help="Optional TensorBoard log directory. Relative paths are resolved inside --output-dir.",
    )
    train_parser.add_argument(
        "--device",
        help="Torch device for training. Defaults to auto, which uses CUDA when available.",
    )
    train_parser.add_argument(
        "--feature-source",
        choices=["metadata", "audio", "manifest_metadata_v0", "metadata_v0", "logmel_v0", "audio_logmel_v0", "logmel_v1", "audio_logmel_v1", "audio_v1", "audio_seq", "logmel_seq", "audio_logmel_seq", "metadata_no_duration", "metadata_no_pause", "metadata_no_duration_no_pause", "metadata_content_only"],
        default=None,
        help="Feature source for NanoTurn. Ablation variants: metadata_no_duration, metadata_no_pause, metadata_no_duration_no_pause, metadata_content_only.",
    )
    train_parser.add_argument(
        "--audio-root",
        type=Path,
        help="Base directory for relative audio paths. Defaults to the dataset parent.",
    )
    train_parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Optional Parquet/Lance log-mel feature cache for --feature-source audio.",
    )
    train_parser.add_argument(
        "--feature-cache-format",
        choices=["parquet", "lance"],
        help="Feature cache format. Defaults to the cache path suffix.",
    )
    train_parser.add_argument(
        "--feature-cache-mode",
        choices=["auto", "read", "write", "off"],
        default=None,
        help="auto builds a missing cache and reuses an existing one.",
    )
    train_parser.add_argument("--json", action="store_true", help="Print metrics as JSON.")

    train_feature_benchmark_parser = subparsers.add_parser(
        "benchmark-train-features",
        help="Benchmark raw audio feature extraction against cached log-mel feature stores.",
    )
    train_feature_benchmark_parser.add_argument("--dataset", type=Path, required=True)
    train_feature_benchmark_parser.add_argument("--format", choices=TURN_FORMATS.names())
    train_feature_benchmark_parser.add_argument("--output-dir", type=Path, required=True)
    train_feature_benchmark_parser.add_argument(
        "--formats",
        nargs="+",
        choices=TRAIN_FEATURE_BENCHMARK_FORMATS,
        default=list(TRAIN_FEATURE_BENCHMARK_FORMATS),
    )
    train_feature_benchmark_parser.add_argument("--sample-count", type=int, default=1000)
    train_feature_benchmark_parser.add_argument("--seed", type=int, default=0)
    train_feature_benchmark_parser.add_argument("--max-records", type=int)
    train_feature_benchmark_parser.add_argument("--audio-root", type=Path)
    train_feature_benchmark_parser.add_argument(
        "--correctness-sample-count",
        type=int,
        default=32,
        help="Number of cached random samples to recompute from source audio for allclose validation.",
    )
    train_feature_benchmark_parser.add_argument(
        "--correctness-tolerance",
        type=float,
        default=1e-6,
        help="Absolute/relative tolerance for cached feature correctness checks.",
    )
    train_feature_benchmark_parser.add_argument("--json", action="store_true")
    train_feature_benchmark_parser.add_argument("--json-output", type=Path, help="Optional JSON report output path.")

    export_parser = subparsers.add_parser(
        "export-turn-onnx",
        help="Export a NanoTurn checkpoint to ONNX.",
    )
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--opset", type=int, default=18)

    vap_inference_parser = subparsers.add_parser(
        "run-vap-inference",
        help="Run VAP model inference on a turn manifest and save predictions to JSONL.",
    )
    vap_inference_parser.add_argument("--dataset", type=Path, required=True, help="Turn manifest JSONL.")
    vap_inference_parser.add_argument("--output", type=Path, required=True, help="Output predictions JSONL.")
    vap_inference_parser.add_argument(
        "--checkpoint",
        default="ErikEkstedt/VAP",
        help="VAP checkpoint: HuggingFace model ID or local path (default: ErikEkstedt/VAP).",
    )
    vap_inference_parser.add_argument("--device", help="Torch device (default: auto).")
    vap_inference_parser.add_argument(
        "--context-sec",
        type=float,
        default=10.0,
        help="Seconds of audio context to feed VAP (default: 10.0).",
    )
    vap_inference_parser.add_argument("--audio-root", type=Path, help="Base dir for relative audio paths.")

        "upload-dataset",
        help="Upload a turn manifest (JSONL/Parquet) to a HuggingFace dataset repo.",
    )
    upload_dataset_parser.add_argument("--manifest", type=Path, required=True, help="Path to JSONL or Parquet manifest.")
    upload_dataset_parser.add_argument("--repo-id", required=True, help="HuggingFace repo id, e.g. myuser/my-dataset.")
    upload_dataset_parser.add_argument("--split", default="train", help="Dataset split name (default: train).")
    upload_dataset_parser.add_argument("--private", action="store_true", help="Create a private repo.")
    upload_dataset_parser.add_argument("--token", help="HuggingFace API token (defaults to HF_TOKEN env var).")
    upload_dataset_parser.add_argument("--message", help="Commit message.")

    upload_model_parser = subparsers.add_parser(
        "upload-model",
        help="Upload a NanoTurn checkpoint to a HuggingFace model repo.",
    )
    upload_model_parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    upload_model_parser.add_argument("--repo-id", required=True, help="HuggingFace repo id, e.g. myuser/nanoturn.")
    upload_model_parser.add_argument("--onnx", type=Path, help="Optional ONNX export to also upload.")
    upload_model_parser.add_argument("--metrics", type=Path, help="Optional metrics.json to include in model card.")
    upload_model_parser.add_argument("--private", action="store_true", help="Create a private repo.")
    upload_model_parser.add_argument("--token", help="HuggingFace API token (defaults to HF_TOKEN env var).")
    upload_model_parser.add_argument("--message", help="Commit message.")

    upload_experiment_parser = subparsers.add_parser(
        "upload-experiment",
        help="Upload an entire experiment output directory to a HuggingFace model repo.",
    )
    upload_experiment_parser.add_argument("--dir", type=Path, required=True, help="Experiment output directory.")
    upload_experiment_parser.add_argument("--repo-id", required=True, help="HuggingFace repo id.")
    upload_experiment_parser.add_argument("--private", action="store_true")
    upload_experiment_parser.add_argument("--token", help="HuggingFace API token.")
    upload_experiment_parser.add_argument("--message", help="Commit message.")

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

    integrity_parser = subparsers.add_parser(
        "paper-artifact-integrity",
        help="Verify a paper bundle artifact_hashes.json manifest.",
    )
    integrity_parser.add_argument("--manifest", type=Path, required=True)
    integrity_parser.add_argument("--root", type=Path, help="Artifact root. Defaults to the root stored in the manifest.")
    integrity_parser.add_argument("--output", type=Path, help="Optional Markdown or JSON report output path.")
    integrity_parser.add_argument("--json", action="store_true")

    archive_parser = subparsers.add_parser(
        "paper-archive",
        help="Create a tar.gz archive and sha256 sidecar from an audited paper artifact bundle.",
    )
    archive_parser.add_argument("--artifacts-dir", type=Path, required=True)
    archive_parser.add_argument("--output", type=Path, required=True)
    archive_parser.add_argument("--root-name", default="stable-asr-artifacts")
    archive_parser.add_argument("--json", action="store_true")

    archive_verify_parser = subparsers.add_parser(
        "paper-archive-verify",
        help="Verify a paper artifact tar.gz archive, sha256 sidecar, and embedded bundle checks.",
    )
    archive_verify_parser.add_argument("--archive", type=Path, required=True)
    archive_verify_parser.add_argument("--sha256", type=Path, help="Optional sha256 sidecar path.")
    archive_verify_parser.add_argument("--root-name", default="stable-asr-artifacts")
    archive_verify_parser.add_argument("--json", action="store_true")

    paper_status_parser = subparsers.add_parser(
        "paper-status",
        help="Summarize smoke, structural, and final paper readiness in one report.",
    )
    paper_status_parser.add_argument("--repo-root", type=Path, default=Path("."))
    paper_status_parser.add_argument(
        "--release-dir",
        type=Path,
        help="Paper release-smoke output directory; infers paper/paper_results.json and artifacts/.",
    )
    paper_status_parser.add_argument("--results", type=Path)
    paper_status_parser.add_argument("--artifacts-dir", type=Path)
    paper_status_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    paper_status_parser.add_argument("--json", action="store_true")

    completion_audit_parser = subparsers.add_parser(
        "completion-audit",
        help="Map the Stable-ASR objective to concrete roadmap, platform, paper, reference, and final-scale evidence.",
    )
    completion_audit_parser.add_argument("--repo-root", type=Path, default=Path("."))
    completion_audit_parser.add_argument(
        "--release-dir",
        type=Path,
        help="Paper release-smoke output directory; infers paper/paper_results.json and artifacts/.",
    )
    completion_audit_parser.add_argument("--results", type=Path)
    completion_audit_parser.add_argument("--artifacts-dir", type=Path)
    completion_audit_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    completion_audit_parser.add_argument("--json", action="store_true")
    completion_audit_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return exit code 0 even when the objective is not complete.",
    )

    evidence_parser = subparsers.add_parser(
        "paper-evidence-matrix",
        help="Audit final-scale experiment evidence, blockers, commands, and expected artifacts.",
    )
    evidence_parser.add_argument("--repo-root", type=Path, default=Path("."))
    evidence_parser.add_argument("--registry", type=Path, help="Optional final experiment registry JSON path.")
    evidence_parser.add_argument("--config", type=Path, help="Optional final run config JSON path.")
    evidence_parser.add_argument("--artifacts-dir", type=Path, help="Optional artifact bundle directory to check.")
    evidence_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    evidence_parser.add_argument("--json", action="store_true")

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

    platform_parity_parser = subparsers.add_parser(
        "platform-parity",
        help="Audit Stable-ASR repository structure against the stable-worldmodel-style platform shape.",
    )
    platform_parity_parser.add_argument("--registry", type=Path, help="Optional platform parity registry JSON path.")
    platform_parity_parser.add_argument("--repo-root", type=Path, default=Path("."))
    platform_parity_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    platform_parity_parser.add_argument("--json", action="store_true")
    platform_parity_parser.add_argument("--validate-only", action="store_true")

    final_experiments_parser = subparsers.add_parser(
        "final-experiments",
        help="Print or validate the final-scale platform-paper experiment runbook.",
    )
    final_experiments_parser.add_argument("--registry", type=Path, help="Optional final experiment registry JSON path.")
    final_experiments_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    final_experiments_parser.add_argument("--json", action="store_true", help="Print registry as JSON.")
    final_experiments_parser.add_argument("--validate-only", action="store_true")

    final_inputs_parser = subparsers.add_parser(
        "final-inputs",
        help="Print, validate, or audit the final-scale input collection plan.",
    )
    final_inputs_parser.add_argument("--registry", type=Path, help="Optional final input collection JSON path.")
    final_inputs_parser.add_argument("--config", type=Path, help="Optional final run config JSON path.")
    final_inputs_parser.add_argument("--repo-root", type=Path, default=Path("."))
    final_inputs_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    final_inputs_parser.add_argument("--json", action="store_true", help="Print report or registry as JSON.")
    final_inputs_parser.add_argument("--validate-only", action="store_true")

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
    final_config_parser.add_argument("--plan-missing", action="store_true", help="Render an actionable final-run plan for missing inputs and remaining commands.")
    final_config_parser.add_argument("--scaffold", action="store_true", help="Create final-run directories and README hints.")
    final_config_parser.add_argument(
        "--prepare-corpora",
        action="store_true",
        help="Prepare configured public ASR corpus manifests for local inputs that exist.",
    )
    final_config_parser.add_argument(
        "--prepare-inputs",
        action="store_true",
        help="Run final corpus, ASR eval manifest, weak split, prediction, VoiceWorld, ASR-command, and file audits.",
    )
    final_config_parser.add_argument(
        "--prepare-asr-eval-manifest",
        action="store_true",
        help="Combine prepared public ASR manifests into the shared final ASR evaluation manifest.",
    )
    final_config_parser.add_argument(
        "--require-all-corpora",
        action="store_true",
        help="With --prepare-corpora, fail if any configured corpus input is missing.",
    )
    final_config_parser.add_argument(
        "--bootstrap-turn-splits",
        action="store_true",
        help="Create weak train/dev/test turn splits from prepared final ASR manifests.",
    )
    final_config_parser.add_argument(
        "--no-incomplete-turns",
        action="store_true",
        help="With --bootstrap-turn-splits, emit complete weak-turn records only.",
    )
    final_config_parser.add_argument(
        "--prepare-external-predictions",
        action="store_true",
        help="Normalize configured external turn prediction exports and validate test coverage when possible.",
    )
    final_config_parser.add_argument(
        "--prepare-voiceworld-real",
        action="store_true",
        help="Prepare the configured real VoiceWorld manifest from metadata/audio inputs and audit scenario coverage.",
    )
    final_config_parser.add_argument(
        "--require-all-predictions",
        action="store_true",
        help="With --prepare-external-predictions, fail if any raw prediction export is missing.",
    )
    final_config_parser.add_argument(
        "--allow-extra-predictions",
        action="store_true",
        help="With --prepare-external-predictions, allow prediction rows outside the test split.",
    )
    final_config_parser.add_argument(
        "--audit-voiceworld-real",
        action="store_true",
        help="Audit final voiceworld_real manifest scenario and factor coverage.",
    )
    final_config_parser.add_argument(
        "--audit-asr-commands",
        action="store_true",
        help="Audit final command-backed ASR comparison config without executing adapters.",
    )
    final_config_parser.add_argument(
        "--prepare-asr-transcript-conversions",
        action="store_true",
        help="Build the final ASR transcript conversion result input from configured adapter outputs.",
    )
    final_config_parser.add_argument("--scenario-suite", type=Path, help="Scenario suite for --audit-voiceworld-real.")
    final_config_parser.add_argument("--min-scenario-records", type=int, default=1)
    final_config_parser.add_argument("--min-asr-command-adapters", type=int, default=4)

    final_results_parser = subparsers.add_parser(
        "final-results",
        help="Assemble final-scale experiment JSON outputs into paper_results.json.",
    )
    final_results_parser.add_argument("--config", type=Path, help="Optional final run config JSON path.")
    final_results_parser.add_argument("--repo-root", type=Path, default=Path("."))
    final_results_parser.add_argument("--output", type=Path, help="Optional paper_results.json output path.")
    final_results_parser.add_argument("--allow-missing", action="store_true", help="Write explicit placeholders for missing inputs.")
    final_results_parser.add_argument("--validate-only", action="store_true", help="Audit inputs without writing paper_results.json.")
    final_results_parser.add_argument("--json", action="store_true")

    leaderboard_parser = subparsers.add_parser(
        "leaderboard-export",
        help="Export leaderboard-ready JSONL or CSV rows from paper_results.json.",
    )
    leaderboard_parser.add_argument("--results", type=Path, required=True)
    leaderboard_parser.add_argument("--output", type=Path, required=True)
    leaderboard_parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")

    leaderboard_validate_parser = subparsers.add_parser(
        "leaderboard-validate",
        help="Validate a leaderboard JSONL submission against the benchmark suite schema.",
    )
    leaderboard_validate_parser.add_argument("--input", type=Path, required=True)
    leaderboard_validate_parser.add_argument("--suite", type=Path, help="Optional benchmark suite JSON path.")
    leaderboard_validate_parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    leaderboard_validate_parser.add_argument("--json", action="store_true")
    leaderboard_validate_parser.add_argument("--require-known-systems", action="store_true")
    leaderboard_validate_parser.add_argument("--require-known-slices", action="store_true")
    leaderboard_validate_parser.add_argument("--require-complete-suite", action="store_true")

    leaderboard_report_parser = subparsers.add_parser(
        "leaderboard-report",
        help="Generate a ranked Markdown or JSON report from leaderboard JSONL rows.",
    )
    leaderboard_report_parser.add_argument("--input", type=Path, required=True)
    leaderboard_report_parser.add_argument("--suite", type=Path, help="Optional benchmark suite JSON path.")
    leaderboard_report_parser.add_argument("--output", type=Path, help="Optional Markdown or JSON report path.")
    leaderboard_report_parser.add_argument("--top-k", type=int, default=3)
    leaderboard_report_parser.add_argument("--json", action="store_true")
    leaderboard_report_parser.add_argument("--require-known-systems", action="store_true")
    leaderboard_report_parser.add_argument("--require-known-slices", action="store_true")
    leaderboard_report_parser.add_argument("--require-complete-suite", action="store_true")

    leaderboard_merge_parser = subparsers.add_parser(
        "leaderboard-merge",
        help="Merge multiple leaderboard JSONL submissions and emit validation plus ranked reports.",
    )
    leaderboard_merge_parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Leaderboard JSONL input path. Repeat for multiple submissions.",
    )
    leaderboard_merge_parser.add_argument("--output", type=Path, required=True)
    leaderboard_merge_parser.add_argument("--suite", type=Path, help="Optional benchmark suite JSON path.")
    leaderboard_merge_parser.add_argument("--validation-output", type=Path, help="Optional Markdown validation report path.")
    leaderboard_merge_parser.add_argument("--report-output", type=Path, help="Optional Markdown or JSON ranked report path.")
    leaderboard_merge_parser.add_argument("--top-k", type=int, default=3)
    leaderboard_merge_parser.add_argument("--json", action="store_true")
    leaderboard_merge_parser.add_argument("--require-known-systems", action="store_true")
    leaderboard_merge_parser.add_argument("--require-known-slices", action="store_true")
    leaderboard_merge_parser.add_argument("--require-complete-suite", action="store_true")

    benchmark_suite_parser = subparsers.add_parser(
        "benchmark-suite",
        help="Print or validate the Stable-ASR benchmark suite definition.",
    )
    benchmark_suite_parser.add_argument("--suite", type=Path, help="Optional benchmark suite JSON path.")
    benchmark_suite_parser.add_argument("--results", type=Path, help="Optional paper_results.json coverage input.")
    benchmark_suite_parser.add_argument("--artifacts-dir", type=Path, help="Optional paper artifact bundle directory.")
    benchmark_suite_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    benchmark_suite_parser.add_argument("--json", action="store_true", help="Print the suite as JSON.")
    benchmark_suite_parser.add_argument("--validate-only", action="store_true", help="Only validate the suite.")

    benchmark_pack_parser = subparsers.add_parser(
        "benchmark-pack",
        help="Create a contributor starter pack with schemas, suite metadata, fixtures, and runnable submission commands.",
    )
    benchmark_pack_parser.add_argument("--output-dir", type=Path, required=True)
    benchmark_pack_parser.add_argument("--suite", type=Path, help="Optional benchmark suite JSON path.")
    benchmark_pack_parser.add_argument("--schema-registry", type=Path, help="Optional schema registry JSON path.")
    benchmark_pack_parser.add_argument("--json", action="store_true")

    adapter_pack_parser = subparsers.add_parser(
        "adapter-pack",
        help="Create an external ASR adapter starter pack with registries, templates, fixtures, and command checks.",
    )
    adapter_pack_parser.add_argument("--output-dir", type=Path, required=True)
    adapter_pack_parser.add_argument("--adapter-registry", type=Path, help="Optional adapter registry JSON path.")
    adapter_pack_parser.add_argument("--asr-collections", type=Path, help="Optional ASR collections JSON path.")
    adapter_pack_parser.add_argument("--schema-registry", type=Path, help="Optional schema registry JSON path.")
    adapter_pack_parser.add_argument("--json", action="store_true")

    scenario_pack_parser = subparsers.add_parser(
        "scenario-pack",
        help="Create a VoiceWorld scenario starter pack with suite metadata, fixtures, and runnable evaluation commands.",
    )
    scenario_pack_parser.add_argument("--output-dir", type=Path, required=True)
    scenario_pack_parser.add_argument("--suite", type=Path, help="Optional scenario suite JSON path.")
    scenario_pack_parser.add_argument("--json", action="store_true")

    final_pack_parser = subparsers.add_parser(
        "final-pack",
        help="Create a final-scale run starter pack with configs, audits, runbooks, and scaffold directories.",
    )
    final_pack_parser.add_argument("--output-dir", type=Path, required=True)
    final_pack_parser.add_argument("--config", type=Path, help="Optional final run config JSON path.")
    final_pack_parser.add_argument("--input-collections", type=Path, help="Optional final input collection JSON path.")
    final_pack_parser.add_argument("--final-experiments", type=Path, help="Optional final experiment registry JSON path.")
    final_pack_parser.add_argument("--scenario-suite", type=Path, help="Optional scenario suite JSON path.")
    final_pack_parser.add_argument("--json", action="store_true")

    final_acquisition_pack_parser = subparsers.add_parser(
        "final-acquisition-pack",
        help="Create a final input acquisition checklist pack for corpora, recordings, predictions, and artifacts.",
    )
    final_acquisition_pack_parser.add_argument("--output-dir", type=Path, required=True)
    final_acquisition_pack_parser.add_argument("--config", type=Path, help="Optional final run config JSON path.")
    final_acquisition_pack_parser.add_argument("--input-collections", type=Path, help="Optional final input collection JSON path.")
    final_acquisition_pack_parser.add_argument("--repo-root", type=Path, default=Path("."))
    final_acquisition_pack_parser.add_argument("--json", action="store_true")

    final_assignment_audit_parser = subparsers.add_parser(
        "final-assignment-audit",
        help="Audit a filled final acquisition assignment tracker for owners, due dates, and release blockers.",
    )
    final_assignment_audit_parser.add_argument("--input", type=Path, required=True)
    final_assignment_audit_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    final_assignment_audit_parser.add_argument("--require-owner", action="store_true")
    final_assignment_audit_parser.add_argument("--require-due-date", action="store_true")
    final_assignment_audit_parser.add_argument("--require-ready", action="store_true")
    final_assignment_audit_parser.add_argument("--json", action="store_true")

    final_handoff_template_parser = subparsers.add_parser(
        "final-handoff-template",
        help="Write a structured JSON template for final input handoff.",
    )
    final_handoff_template_parser.add_argument("--output", type=Path, required=True)
    final_handoff_template_parser.add_argument("--input-collections", type=Path, help="Optional final input collection JSON path.")

    final_handoff_checksums_parser = subparsers.add_parser(
        "final-handoff-checksums",
        help="Populate sha256 and byte-size checksum entries for staged files in a final input handoff JSON.",
    )
    final_handoff_checksums_parser.add_argument("--input", type=Path, required=True)
    final_handoff_checksums_parser.add_argument("--repo-root", type=Path, default=Path("."))
    final_handoff_checksums_parser.add_argument("--output", type=Path, required=True)
    final_handoff_checksums_parser.add_argument("--json", action="store_true")

    final_handoff_audit_parser = subparsers.add_parser(
        "final-handoff-audit",
        help="Audit a structured final input handoff JSON for paths, metadata, and checksums.",
    )
    final_handoff_audit_parser.add_argument("--input", type=Path, required=True)
    final_handoff_audit_parser.add_argument("--repo-root", type=Path, default=Path("."))
    final_handoff_audit_parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    final_handoff_audit_parser.add_argument(
        "--require-checksums",
        action="store_true",
        help="Fail when staged file checksums or byte sizes are missing.",
    )
    final_handoff_audit_parser.add_argument("--json", action="store_true")

    contributor_pack_parser = subparsers.add_parser(
        "contributor-pack",
        help="Create a unified contributor onboarding pack containing all starter packs and GitHub templates.",
    )
    contributor_pack_parser.add_argument("--output-dir", type=Path, required=True)
    contributor_pack_parser.add_argument("--json", action="store_true")

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
    paper_release_audit_parser.add_argument("--model-card", type=Path)
    paper_release_audit_parser.add_argument(
        "--require-final-ready",
        action="store_true",
        help="Fail unless final inputs and final-scale paper parity are READY.",
    )
    paper_release_audit_parser.add_argument("--json", action="store_true", help="Print release audit as JSON.")

    paper_release_smoke_parser = subparsers.add_parser(
        "paper-release-smoke",
        help="Run the smoke paper pipeline, generate drafts/cards, and write a release audit.",
    )
    paper_release_smoke_parser.add_argument("--output-dir", type=Path, default=Path("runs/paper/release_smoke"))
    paper_release_smoke_parser.add_argument("--episodes", type=int, default=9)
    paper_release_smoke_parser.add_argument("--seed", type=int, default=6)
    paper_release_smoke_parser.add_argument("--skip-train", action="store_true")
    paper_release_smoke_parser.add_argument("--repo-root", type=Path, default=Path("."))
    paper_release_smoke_parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("examples/data/turn_demo.jsonl"),
        help="Turn manifest used for the generated dataset card.",
    )
    paper_release_smoke_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if release audit is not READY.",
    )
    paper_release_smoke_parser.add_argument(
        "--require-final-ready",
        action="store_true",
        help="Exit nonzero unless the release audit, final inputs, and final-scale paper parity are READY.",
    )
    paper_release_smoke_parser.add_argument("--json", action="store_true", help="Print smoke result as JSON.")

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
        help="Generate dataset, experiment, or model Markdown cards.",
    )
    card_parser.add_argument("kind", choices=["dataset", "experiment", "model"])
    card_parser.add_argument("--input", type=Path, required=True)
    card_parser.add_argument("--output", type=Path, required=True)
    card_parser.add_argument("--model-id", help="Model id used with kind=model and registry inputs.")
    card_parser.add_argument("--metrics", type=Path, help="Optional NanoTurn metrics JSON used with kind=model.")

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
        report = run_doctor(
            repo_root=args.repo_root,
            check_final_files=args.check_final_files,
            check_release_env=args.check_release_env,
        )
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
        return 0 if report.ok else 1

    if args.command == "catalog":
        report = build_platform_catalog(repo_root=args.repo_root)
        if args.output:
            if args.json:
                write_platform_catalog_json(report, args.output)
            else:
                write_platform_catalog_markdown(report, args.output)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_markdown())
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
        payload = report.to_dict()
        _write_json_output(args.json_output, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
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
        payload = report.to_dict()
        _write_json_output(args.json_output, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
            if args.report:
                print(f"report: {args.report}")
        return 0 if report.ok else 1

    if args.command == "turn-submission":
        try:
            report = build_turn_submission(
                dataset=args.dataset,
                predictions=args.predictions,
                output_dir=args.output_dir,
                system=args.system,
                dataset_format=args.format,
                allow_extra=args.allow_extra,
                complete_threshold=args.complete_threshold,
                suite_path=args.suite,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"turn_submission: {'OK' if report.ok else 'FAILED'}")
            print(f"submission: {report.artifacts.manifest}")
            print(f"summary: {report.artifacts.summary_markdown}")
            print(f"leaderboard: {report.artifacts.leaderboard['jsonl']}")
        return 0 if report.ok else 1

    if args.command == "submission-index":
        try:
            report = index_submission_directory(
                args.root,
                args.output_dir,
                suite=load_benchmark_suite(args.suite),
                top_k=args.top_k,
                require_known_systems=args.require_known_systems,
                require_known_slices=args.require_known_slices,
                require_complete_suite=args.require_complete_suite,
            )
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
            print(f"index: {report.artifacts.index_json}")
            print(f"summary: {report.artifacts.summary_markdown}")
            if report.artifacts.leaderboard:
                print(f"leaderboard: {report.artifacts.leaderboard}")
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
        payload = report.to_dict()
        _write_json_output(args.json_output, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
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
        payload = report.to_dict()
        _write_json_output(args.json_output, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
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
        payload = {"name": _benchmark_turn_name(args), **report.to_dict()}
        _write_json_output(args.json_output, payload)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
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
        _write_json_output(args.json_output, payload)
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

    if args.command == "benchmark-audio-windows":
        try:
            rows = benchmark_audio_window_formats(
                load_turn_records(args.dataset, format=args.format),
                output_dir=args.output_dir,
                formats=args.formats,
                sample_count=args.sample_count,
                seed=args.seed,
                max_records=args.max_records,
                audio_root=args.audio_root,
                correctness_sample_count=args.correctness_sample_count,
                correctness_tolerance=args.correctness_tolerance,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        payload = [row.to_dict() for row in rows]
        _write_json_output(args.json_output, payload)
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
                            f"sample_count={row.sample_count}",
                            f"sample_seconds={row.sample_seconds:.6f}",
                            f"samples_per_second={row.samples_per_second:.2f}",
                            f"speedup_vs_source_wav={row.speedup_vs_source_wav:.2f}",
                            f"size_bytes={row.size_bytes}",
                            f"sample_strategy={row.sample_strategy}",
                            f"correctness_sample_count={row.correctness_sample_count}",
                            f"max_abs_error_vs_source={row.max_abs_error_vs_source:.8g}",
                            f"allclose_to_source={row.allclose_to_source}",
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

    if args.command == "model-registry":
        try:
            registry = load_model_registry(args.registry)
            validation = validate_model_registry(registry)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.audit_configs:
                report = audit_model_registry_configs(registry)
                text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
                print(text)
                return 0 if report.ok else 1
            if args.validate_only:
                print(f"OK: {registry['id']} ({len(registry['models'])} model(s))")
                return 0
            text = json.dumps(registry, ensure_ascii=False, indent=2) if args.json else model_registry_markdown(registry)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "schema-registry":
        try:
            registry = load_schema_registry(args.registry)
            validation = validate_schema_registry(registry)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.validate_only:
                print(f"OK: {registry['id']} ({len(registry['schemas'])} schema(s))")
                return 0
            payload = get_schema_entry(registry, args.schema_id) if args.schema_id else registry
            if args.json:
                text = json.dumps(payload, ensure_ascii=False, indent=2)
            else:
                text = schema_entry_markdown(payload) if args.schema_id else schema_registry_markdown(registry)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "validate-schema-file":
        try:
            report = validate_schema_file(
                args.input,
                schema_id=args.schema_id,
                registry_path=args.registry,
                format=args.format,
                max_errors=args.max_errors,
            )
        except (OSError, ValueError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(text)
        return 0 if report.ok else 1

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
            if args.audit_readiness:
                readiness = audit_asr_collection_readiness(
                    registry,
                    load_adapter_registry(args.adapter_registry),
                    required_priorities=tuple(args.require_priority or ["p0", "p1"]),
                    max_review_age_days=args.max_review_age_days,
                )
                text = (
                    json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2)
                    if args.json
                    else readiness.to_markdown()
                )
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
                print(text)
                return 0 if readiness.ok else 1
            if args.audit_licenses:
                licenses = audit_asr_collection_licenses(
                    registry,
                    required_priorities=tuple(args.require_priority or ["p0", "p1"]),
                    require_resolved=args.require_license_reviewed,
                )
                text = (
                    json.dumps(licenses.to_dict(), ensure_ascii=False, indent=2)
                    if args.json
                    else licenses.to_markdown()
                )
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
                print(text)
                return 0 if licenses.ok else 1
            if args.validate_only:
                print(f"OK: {registry['id']} ({len(registry['entries'])} reference(s))")
                return 0
            if args.json:
                text = json.dumps(registry, ensure_ascii=False, indent=2)
            elif args.format == "paper-markdown":
                text = asr_collections_reference_markdown(registry)
            elif args.format == "bibtex":
                text = asr_collections_bibtex(registry)
            elif args.format == "acquisition-markdown":
                text = asr_collections_acquisition_markdown(registry)
            elif args.format == "license-markdown":
                text = audit_asr_collection_licenses(
                    registry,
                    required_priorities=tuple(args.require_priority or ["p0", "p1"]),
                    require_resolved=args.require_license_reviewed,
                ).to_markdown()
            elif args.format == "source-manifest":
                text = json.dumps(asr_collections_source_manifest(registry), ensure_ascii=False, indent=2)
            else:
                text = asr_collections_markdown(registry)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "turn-collections":
        try:
            registry = load_turn_collections(args.registry)
            validation = validate_turn_collections(registry)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.audit_coverage:
                coverage = audit_turn_collection_coverage(
                    registry,
                    load_data_sources(args.data_sources),
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
            if args.json:
                text = json.dumps(registry, ensure_ascii=False, indent=2)
            elif args.format == "acquisition-markdown":
                text = turn_collections_acquisition_markdown(registry)
            elif args.format == "source-manifest":
                text = json.dumps(turn_collections_source_manifest(registry), ensure_ascii=False, indent=2)
            else:
                text = turn_collections_markdown(registry)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "reference-workqueue":
        try:
            workqueue = reference_workqueue_from_registries(
                asr_registry=load_asr_collections(args.asr_registry),
                turn_registry=load_turn_collections(args.turn_registry),
                required_priorities=tuple(args.require_priority or ["p0", "p1"]),
            )
            if args.audit_evidence:
                report = audit_reference_workqueue_evidence(
                    workqueue,
                    repo_root=args.repo_root,
                    require_content=args.require_content,
                )
                text = (
                    json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
                    if args.json or args.format == "json"
                    else report.to_markdown()
                )
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
                print(text)
                return 0 if report.ok else 1
            if args.json or args.format == "json":
                text = json.dumps(workqueue, ensure_ascii=False, indent=2)
            elif args.format == "jsonl":
                text = reference_workqueue_jsonl(workqueue)
            elif args.format == "assignments-json":
                text = json.dumps(reference_workqueue_assignments(workqueue), ensure_ascii=False, indent=2)
            elif args.format == "assignments-tsv":
                text = reference_workqueue_assignments_tsv(reference_workqueue_assignments(workqueue))
            elif args.format == "assignments-markdown":
                text = reference_workqueue_assignments_markdown(reference_workqueue_assignments(workqueue))
            elif args.format == "evidence-markdown":
                text = reference_workqueue_evidence_markdown(workqueue)
            elif args.format == "issues-markdown":
                text = reference_workqueue_issues_markdown(workqueue)
            elif args.format == "license-review-markdown":
                text = reference_workqueue_license_review_markdown(workqueue)
            else:
                text = reference_workqueue_markdown(workqueue)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "reference-assignment-audit":
        report = audit_reference_assignments(
            args.input,
            repo_root=args.repo_root,
            require_owner=args.require_owner,
            require_due_date=args.require_due_date,
            require_ready=args.require_ready,
        )
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(text)
        return 0 if report.ok else 1

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
        if args.require_final_ready and not report.final_scale_ready:
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

    if args.command == "prepare-public-asr":
        try:
            records = prepare_public_asr_manifest(
                corpus=args.corpus,
                input_dir=args.input_dir,
                output_path=args.output,
                split=args.split,
                sample_rate=args.sample_rate,
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        summary = summarize_asr_records(records)
        if args.json:
            print(json.dumps({"output": str(args.output), "corpus": args.corpus, **summary}, ensure_ascii=False, indent=2))
        else:
            print(f"wrote {len(records)} {args.corpus} ASR record(s) to {args.output}")
            print(f"splits: {json.dumps(summary['splits'], ensure_ascii=False, sort_keys=True)}")
            print(f"languages: {json.dumps(summary['languages'], ensure_ascii=False, sort_keys=True)}")
        return 0

    if args.command == "prepare-voiceworld":
        try:
            records = prepare_voiceworld_manifest(
                args.input,
                args.output,
                audio_root=args.audio_root,
                default_sample_rate=args.sample_rate,
                default_language=args.language,
                default_source=args.source,
                factor_fields=tuple(args.factor_field or DEFAULT_VOICEWORLD_FACTOR_FIELDS),
                id_field=args.id_field,
                audio_field=args.audio_field,
                text_field=args.text_field,
                scenario_field=args.scenario_field,
                turn_label_field=args.turn_label_field,
                action_label_field=args.action_label_field,
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        summary = summarize_records(records)
        if args.json:
            print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))
        else:
            print(f"wrote {len(records)} VoiceWorld turn record(s) to {args.output}")
            print(f"scenarios: {json.dumps(summary['scenarios'], ensure_ascii=False, sort_keys=True)}")
            print(f"turn_labels: {json.dumps(summary['turn_labels'], ensure_ascii=False, sort_keys=True)}")
            print(f"action_labels: {json.dumps(summary['action_labels'], ensure_ascii=False, sort_keys=True)}")
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
        _write_json_output(args.json_output, payload)
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
        _write_json_output(args.json_output, payload)
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

    if args.command == "streaming-submission":
        try:
            report = build_streaming_submission(
                input_path=args.input,
                output_dir=args.output_dir,
                system=args.system,
                slice_name=args.slice,
                suite_path=args.suite,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"streaming_submission: {'OK' if report.ok else 'FAILED'}")
            print(f"submission: {report.artifacts.manifest}")
            print(f"summary: {report.artifacts.summary_markdown}")
            print(f"leaderboard: {report.artifacts.leaderboard['jsonl']}")
        return 0 if report.ok else 1

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
        _write_json_output(args.json_output, payload)
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
        _write_json_output(args.json_output, payload)
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
            if args.validate_only:
                audit = audit_asr_command_config(
                    args.config,
                    repo_root=args.repo_root,
                    min_adapters=args.min_adapters,
                    require_input_manifest=args.require_input_manifest,
                )
                _write_json_output(args.json_output, audit.to_dict())
                if args.json:
                    print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(audit.to_text())
                return 0 if audit.ok else 1
            report = compare_asr_commands_from_config(args.config)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        payload = report.to_dict()
        _write_json_output(args.json_output, payload)
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
        if args.dataset:
            report = evaluate_voice_world_records(
                load_manifest(args.dataset),
                predictor,
                seed=args.seed,
                suite=str(args.dataset),
            )
        else:
            report = evaluate_voice_world(
                predictor,
                episodes=args.episodes,
                seed=args.seed,
            )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report.to_markdown(), encoding="utf-8")
        payload = report.to_dict()
        _write_json_output(args.json_output, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
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
        train_config = _load_train_turn_config(args.config)
        model_type = args.model or str(train_config.get("model_type", "nanoturn_pico"))
        epochs = args.epochs if args.epochs is not None else int(train_config.get("epochs", 100))
        lr = args.lr if args.lr is not None else float(train_config.get("lr", 1e-2))
        seed = args.seed if args.seed is not None else int(train_config.get("seed", 0))
        batch_size = args.batch_size if args.batch_size is not None else int(train_config.get("batch_size", 128))
        validation_split = (
            args.validation_split
            if args.validation_split is not None
            else float(train_config.get("validation_split", 0.0))
        )
        validation_group_by = _validation_group_by_arg(
            args.validation_group_by
            if args.validation_group_by is not None
            else train_config.get("validation_group_by", "auto")
        )
        optimizer = args.optimizer or str(train_config.get("optimizer", "adam"))
        weight_decay = (
            args.weight_decay if args.weight_decay is not None else float(train_config.get("weight_decay", 0.0))
        )
        gradient_clip_norm = (
            args.gradient_clip_norm
            if args.gradient_clip_norm is not None
            else _optional_float(train_config.get("gradient_clip_norm"))
        )
        checkpoint_interval = (
            args.checkpoint_interval
            if args.checkpoint_interval is not None
            else int(train_config.get("checkpoint_interval", 1))
        )
        feature_source = args.feature_source or str(train_config.get("feature_source", "metadata"))
        audio_root = args.audio_root or _optional_path(train_config.get("audio_root")) or args.dataset.parent
        feature_cache = args.feature_cache or _optional_path(train_config.get("feature_cache"))
        feature_cache_format = args.feature_cache_format or _optional_str(train_config.get("feature_cache_format"))
        feature_cache_mode = args.feature_cache_mode or str(train_config.get("feature_cache_mode", "auto"))
        resume_from = args.resume_from or _optional_path(train_config.get("resume_from"))
        tensorboard_log_dir = args.tensorboard_log_dir or _optional_path(train_config.get("tensorboard_log_dir"))
        device = args.device or str(train_config.get("device", "auto"))
        train_records = load_manifest(args.dataset)
        val_records = load_manifest(args.dev_dataset) if args.dev_dataset else None
        result = train_nanoturn(
            train_records,
            output_dir=args.output_dir,
            model_type=model_type,
            epochs=epochs,
            lr=lr,
            seed=seed,
            feature_source=feature_source,
            audio_root=audio_root,
            feature_cache=feature_cache,
            feature_cache_format=feature_cache_format,
            feature_cache_mode=feature_cache_mode,
            val_records=val_records,
            batch_size=batch_size,
            validation_split=validation_split,
            optimizer=optimizer,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            checkpoint_interval=checkpoint_interval,
            resume_from=resume_from,
            device=device,
            validation_group_by=validation_group_by,
            tensorboard_log_dir=tensorboard_log_dir,
        )
        if args.json:
            print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
        else:
            print(f"checkpoint: {result.checkpoint_path}")
            print(f"metrics: {result.metrics_path}")
            print(f"final_loss: {result.metrics['final_loss']:.6f}")
            print(f"final_accuracy: {result.metrics['final_accuracy']:.4f}")
        return 0

    if args.command == "benchmark-train-features":
        try:
            rows = benchmark_train_feature_cache(
                load_turn_records(args.dataset, format=args.format),
                output_dir=args.output_dir,
                formats=args.formats,
                sample_count=args.sample_count,
                seed=args.seed,
                max_records=args.max_records,
                audio_root=args.audio_root or args.dataset.parent,
                correctness_sample_count=args.correctness_sample_count,
                correctness_tolerance=args.correctness_tolerance,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        payload = [row.to_dict() for row in rows]
        _write_json_output(args.json_output, payload)
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
                            f"sample_count={row.sample_count}",
                            f"sample_seconds={row.sample_seconds:.6f}",
                            f"samples_per_second={row.samples_per_second:.2f}",
                            f"speedup_vs_source_audio={row.speedup_vs_source_audio:.2f}",
                            f"size_bytes={row.size_bytes}",
                            f"sample_strategy={row.sample_strategy}",
                            f"correctness_sample_count={row.correctness_sample_count}",
                            f"max_abs_error_vs_source={row.max_abs_error_vs_source:.8g}",
                            f"allclose_to_source={row.allclose_to_source}",
                            f"path={row.output_path}",
                        ]
                    )
                )
        return 0

    if args.command == "export-turn-onnx":
        output = export_nanoturn_onnx(
            args.checkpoint,
            args.output,
            opset_version=args.opset,
        )
        print(f"onnx: {output}")
        return 0

    if args.command == "run-vap-inference":
        from stable_asr.models.baselines.vap import run_vap_inference
        dataset_parent = args.dataset.parent
        output = run_vap_inference(
            args.dataset,
            args.output,
            checkpoint=args.checkpoint,
            device=getattr(args, "device", None),
            context_sec=args.context_sec,
            audio_root=getattr(args, "audio_root", None) or dataset_parent,
        )
        print(f"vap_predictions: {output}")
        return 0


        import os
        from stable_asr.hub.upload import upload_dataset
        token = args.token or os.environ.get("HF_TOKEN")
        url = upload_dataset(
            args.manifest,
            args.repo_id,
            split=args.split,
            private=args.private,
            token=token,
            commit_message=args.message,
        )
        print(f"dataset: {url}")
        return 0

    if args.command == "upload-model":
        import json as _json
        import os
        from stable_asr.hub.upload import upload_model
        token = args.token or os.environ.get("HF_TOKEN")
        metrics = None
        if args.metrics and Path(args.metrics).exists():
            with open(args.metrics, encoding="utf-8") as fh:
                metrics = _json.load(fh)
        url = upload_model(
            args.checkpoint,
            args.repo_id,
            private=args.private,
            token=token,
            commit_message=args.message,
            onnx_path=args.onnx,
            metrics=metrics,
        )
        print(f"model: {url}")
        return 0

    if args.command == "upload-experiment":
        import os
        from stable_asr.hub.upload import upload_experiment_dir
        token = args.token or os.environ.get("HF_TOKEN")
        url = upload_experiment_dir(
            args.dir,
            args.repo_id,
            private=args.private,
            token=token,
            commit_message=args.message,
        )
        print(f"experiment: {url}")
        return 0


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
            print(f"results: {len(bundle.results)}")
            print(f"tables: {len(bundle.tables)}")
            print(f"figures: {len(bundle.figures)}")
            print(f"leaderboards: {len(bundle.leaderboards)}")
            print(f"leaderboard_validation: {len(bundle.leaderboard_validation)}")
            print(f"leaderboard_reports: {len(bundle.leaderboard_reports)}")
            print(f"benchmark_suite: {len(bundle.benchmark_suite)}")
            print(f"starter_packs: {len(bundle.starter_packs)}")
            print(f"provenance: {len(bundle.provenance)}")
            print(f"data_sources: {len(bundle.data_sources)}")
            print(f"adapter_registry: {len(bundle.adapter_registry)}")
            print(f"model_registry: {len(bundle.model_registry)}")
            print(f"model_cards: {len(bundle.model_cards)}")
            print(f"schema_registry: {len(bundle.schema_registry)}")
            print(f"asr_collections: {len(bundle.asr_collections)}")
            print(f"scenario_suite: {len(bundle.scenario_suite)}")
            print(f"case_studies: {len(bundle.case_studies)}")
            print(f"paper_parity: {len(bundle.paper_parity)}")
            print(f"final_experiments: {len(bundle.final_experiments)}")
            print(f"final_input_collections: {len(bundle.final_input_collections)}")
            print(f"final_run_config: {len(bundle.final_run_config)}")
            print(f"final_run_file_audit: {len(bundle.final_run_file_audit)}")
            print(f"final_run_action_plan: {len(bundle.final_run_action_plan)}")
            print(f"final_evidence_matrix: {len(bundle.final_evidence_matrix)}")
            print(f"paper_status: {len(bundle.paper_status)}")
            print(f"roadmap_status: {len(bundle.roadmap_status)}")
            print(f"completion_audit: {len(bundle.completion_audit)}")
            print(f"claims: {len(bundle.claims)}")
            print(f"artifact_integrity: {len(bundle.artifact_integrity)}")
        return 0

    if args.command == "paper-artifact-integrity":
        try:
            report = verify_artifact_integrity(args.manifest, root=args.root)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(text)
        return 0 if report.ok else 1

    if args.command == "paper-archive":
        try:
            report = paper_artifact_archive(args.artifacts_dir, args.output, root_name=args.root_name)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_text())
        return 0 if report.ok else 1

    if args.command == "paper-archive-verify":
        report = verify_paper_artifact_archive(
            args.archive,
            sha256_path=args.sha256,
            expected_root=args.root_name,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_text())
        return 0 if report.ok else 1

    if args.command == "paper-status":
        try:
            report = paper_status(
                repo_root=args.repo_root,
                release_dir=args.release_dir,
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

    if args.command == "completion-audit":
        try:
            report = completion_audit(
                repo_root=args.repo_root,
                release_dir=args.release_dir,
                results_path=args.results,
                artifacts_dir=args.artifacts_dir,
            )
            if args.output:
                write_completion_audit_markdown(report, args.output)
            if args.json:
                print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(report.to_markdown())
                if args.output:
                    print(f"completion_audit_markdown: {args.output}")
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0 if report.ok or args.allow_incomplete else 1

    if args.command == "paper-evidence-matrix":
        try:
            report = final_evidence_matrix(
                repo_root=args.repo_root,
                registry_path=args.registry,
                config_path=args.config,
                artifacts_dir=args.artifacts_dir,
            )
            text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
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

    if args.command == "platform-parity":
        try:
            registry = load_platform_parity(args.registry)
            validation = validate_platform_parity(registry)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.validate_only:
                print(f"OK: {registry['id']} ({len(registry['items'])} item(s))")
                return 0
            report = audit_platform_parity(registry, repo_root=args.repo_root)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(report.to_markdown() + "\n", encoding="utf-8")
            if args.json:
                print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(report.to_text())
                if args.output:
                    print(f"platform_parity_markdown: {args.output}")
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0 if report.ok else 1

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

    if args.command == "final-inputs":
        try:
            registry = load_final_input_collections(args.registry)
            validation = validate_final_input_collections(registry)
            if not validation.ok:
                print(validation.to_text(), file=sys.stderr)
                return 1
            if args.validate_only:
                print(f"OK: {registry['id']} ({len(registry['collections'])} collection(s))")
                return 0
            config = load_final_run_config(args.config)
            report = final_input_collection_report(registry, config=config, repo_root=args.repo_root)
            text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
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
            if args.prepare_inputs:
                report = prepare_final_inputs(
                    config,
                    repo_root=args.repo_root,
                    scenario_suite_path=args.scenario_suite,
                    require_all_corpora=args.require_all_corpora,
                    require_all_predictions=args.require_all_predictions,
                    allow_extra_predictions=args.allow_extra_predictions,
                    include_incomplete=not args.no_incomplete_turns,
                    min_per_scenario=args.min_scenario_records,
                )
                if args.json:
                    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(report.to_text())
                return 0 if report.ok else 1
            if args.prepare_corpora:
                report = prepare_final_corpora(
                    config,
                    repo_root=args.repo_root,
                    require_all=args.require_all_corpora,
                )
                if args.json:
                    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(report.to_text())
                return 0 if report.ok else 1
            if args.prepare_asr_eval_manifest:
                report = prepare_final_asr_eval_manifest(
                    config,
                    repo_root=args.repo_root,
                )
                if args.json:
                    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(report.to_text())
                return 0 if report.ok else 1
            if args.bootstrap_turn_splits:
                report = bootstrap_final_turn_splits(
                    config,
                    repo_root=args.repo_root,
                    include_incomplete=not args.no_incomplete_turns,
                )
                if args.json:
                    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(report.to_text())
                return 0 if report.ok else 1
            if args.prepare_external_predictions:
                report = prepare_final_external_predictions(
                    config,
                    repo_root=args.repo_root,
                    require_all=args.require_all_predictions,
                    allow_extra=args.allow_extra_predictions,
                )
                if args.json:
                    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(report.to_text())
                return 0 if report.ok else 1
            if args.prepare_voiceworld_real:
                report = prepare_final_voiceworld_real(
                    config,
                    repo_root=args.repo_root,
                    scenario_suite_path=args.scenario_suite,
                    min_per_scenario=args.min_scenario_records,
                    require_input=True,
                )
                if args.json:
                    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(report.to_text())
                return 0 if report.ok else 1
            if args.audit_voiceworld_real:
                report = audit_final_voiceworld_real(
                    config,
                    repo_root=args.repo_root,
                    scenario_suite_path=args.scenario_suite,
                    min_per_scenario=args.min_scenario_records,
                )
                if args.json:
                    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(report.to_text())
                return 0 if report.ok else 1
            if args.audit_asr_commands:
                report = audit_asr_command_config(
                    _resolve_config_path(config["asr_command_config"], repo_root=args.repo_root),
                    repo_root=args.repo_root,
                    min_adapters=args.min_asr_command_adapters,
                    require_input_manifest=True,
                )
                if args.json:
                    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(report.to_text())
                return 0 if report.ok else 1
            if args.prepare_asr_transcript_conversions:
                report = prepare_final_asr_transcript_conversions(config, repo_root=args.repo_root)
                if args.json:
                    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(report.to_text())
                return 0 if report.ok else 1
            if args.check_files:
                report = audit_final_run_files(config, repo_root=args.repo_root)
                text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else final_run_file_audit_markdown(report)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
                print(text)
                return 0 if report.ok else 1
            if args.plan_missing:
                report = build_final_run_action_plan(
                    config,
                    repo_root=args.repo_root,
                    config_path=args.config or "configs/final/paper_final.json",
                )
                text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
                print(text)
                return 0
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

    if args.command == "final-results":
        try:
            report = assemble_final_paper_results(
                args.config,
                repo_root=args.repo_root,
                output_path=args.output,
                allow_missing=args.allow_missing,
                write=not args.validate_only,
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
        return 0 if report.ok or args.allow_missing else 1

    if args.command == "leaderboard-export":
        try:
            output = export_leaderboard(args.results, args.output, format=args.format)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"leaderboard: {output}")
        return 0

    if args.command == "leaderboard-validate":
        try:
            report = validate_leaderboard_jsonl(
                args.input,
                suite=load_benchmark_suite(args.suite),
                require_known_systems=args.require_known_systems,
                require_known_slices=args.require_known_slices,
                require_complete_suite=args.require_complete_suite,
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(text)
        return 0 if report.ok else 1

    if args.command == "leaderboard-report":
        try:
            report = leaderboard_report(
                args.input,
                suite=load_benchmark_suite(args.suite),
                top_k=args.top_k,
                require_known_systems=args.require_known_systems,
                require_known_slices=args.require_known_slices,
                require_complete_suite=args.require_complete_suite,
            )
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(text)
        return 0 if report.ok else 1

    if args.command == "leaderboard-merge":
        try:
            report = merge_leaderboard_jsonl(
                args.input,
                args.output,
                suite=load_benchmark_suite(args.suite),
                top_k=args.top_k,
                require_known_systems=args.require_known_systems,
                require_known_slices=args.require_known_slices,
                require_complete_suite=args.require_complete_suite,
            )
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.validation_output:
            text = report.validation.to_markdown()
            args.validation_output.parent.mkdir(parents=True, exist_ok=True)
            args.validation_output.write_text(
                text + ("\n" if not text.endswith("\n") else ""),
                encoding="utf-8",
            )
        if args.report_output:
            if args.report_output.suffix == ".json":
                text = json.dumps(report.report.to_dict(), ensure_ascii=False, indent=2)
            else:
                text = report.report.to_markdown()
            args.report_output.parent.mkdir(parents=True, exist_ok=True)
            args.report_output.write_text(
                text + ("\n" if not text.endswith("\n") else ""),
                encoding="utf-8",
            )
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
        print(text)
        return 0 if report.ok else 1

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
            artifact_audit = None
            if args.artifacts_dir:
                artifact_audit = audit_benchmark_required_artifacts(args.artifacts_dir, suite=suite)
                if not artifact_audit.ok:
                    print(artifact_audit.to_text(), file=sys.stderr)
                    return 1
            if args.validate_only:
                suffix = f"; coverage=OK ({coverage.rows} row(s))" if coverage is not None else ""
                if artifact_audit is not None:
                    suffix += f"; artifacts=OK ({len(artifact_audit.present)} required)"
                print(f"OK: {suite['id']} ({len(suite['tasks'])} task(s)){suffix}")
                return 0
            if args.json:
                payload: dict[str, object] = {"suite": suite}
                if coverage is not None:
                    payload["coverage"] = coverage.to_dict()
                if artifact_audit is not None:
                    payload["artifacts"] = artifact_audit.to_dict()
                text = json.dumps(payload if len(payload) > 1 else suite, ensure_ascii=False, indent=2)
            else:
                text = benchmark_suite_markdown(suite)
                if coverage is not None:
                    text += "\n" + coverage.to_text() + "\n"
                if artifact_audit is not None:
                    text += "\n" + artifact_audit.to_text() + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
            print(text)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "benchmark-pack":
        try:
            report = build_benchmark_pack(
                args.output_dir,
                suite_path=args.suite,
                schema_registry_path=args.schema_registry,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"benchmark_pack: {'OK' if report.ok else 'FAILED'}")
            print(f"output_dir: {report.output_dir}")
            print(f"readme: {report.files.get('readme')}")
            print(f"commands: {report.files.get('commands_markdown')}")
        return 0 if report.ok else 1

    if args.command == "adapter-pack":
        try:
            report = build_adapter_pack(
                args.output_dir,
                adapter_registry_path=args.adapter_registry,
                asr_collections_path=args.asr_collections,
                schema_registry_path=args.schema_registry,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"adapter_pack: {'OK' if report.ok else 'FAILED'}")
            print(f"output_dir: {report.output_dir}")
            print(f"readme: {report.files.get('readme')}")
            print(f"commands: {report.files.get('commands_markdown')}")
        return 0 if report.ok else 1

    if args.command == "scenario-pack":
        try:
            report = build_scenario_pack(
                args.output_dir,
                suite_path=args.suite,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"scenario_pack: {'OK' if report.ok else 'FAILED'}")
            print(f"output_dir: {report.output_dir}")
            print(f"readme: {report.files.get('readme')}")
            print(f"commands: {report.files.get('commands_markdown')}")
        return 0 if report.ok else 1

    if args.command == "final-pack":
        try:
            report = build_final_pack(
                args.output_dir,
                config_path=args.config,
                input_collections_path=args.input_collections,
                final_experiments_path=args.final_experiments,
                scenario_suite_path=args.scenario_suite,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"final_pack: {'OK' if report.ok else 'FAILED'}")
            print(f"final_ready: {'READY' if report.final_ready else 'NOT_READY'}")
            print(f"missing_required: {len(report.missing_required)}")
            print(f"output_dir: {report.output_dir}")
            print(f"readme: {report.files.get('readme')}")
            print(f"commands: {report.files.get('commands_markdown')}")
        return 0 if report.ok else 1

    if args.command == "final-acquisition-pack":
        try:
            report = build_final_acquisition_pack(
                args.output_dir,
                config_path=args.config,
                input_collections_path=args.input_collections,
                repo_root=args.repo_root,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"final_acquisition_pack: {'OK' if report.ok else 'FAILED'}")
            print(f"collections: {report.collections}")
            print(f"checklist_rows: {report.checklist_rows}")
            print(f"assignment_rows: {report.assignment_rows}")
            print(f"missing_required: {len(report.missing_required)}")
            print(f"license_review_items: {report.license_review_items}")
            print(f"output_dir: {report.output_dir}")
            print(f"readme: {report.files.get('readme')}")
            print(f"commands: {report.files.get('commands_markdown')}")
        return 0 if report.ok else 1

    if args.command == "final-assignment-audit":
        report = audit_acquisition_assignments(
            args.input,
            require_owner=args.require_owner,
            require_due_date=args.require_due_date,
            require_ready=args.require_ready,
        )
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(text)
        return 0 if report.ok else 1

    if args.command == "final-handoff-template":
        try:
            registry = load_final_input_collections(args.input_collections)
            payload = final_handoff_template(registry)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"final_handoff_template: {args.output}")
        except (OSError, ValueError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "final-handoff-checksums":
        report = populate_final_handoff_checksums(args.input, repo_root=args.repo_root, output=args.output)
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_text()
        print(text)
        return 0 if report.ok else 1

    if args.command == "final-handoff-audit":
        report = audit_final_handoff(args.input, repo_root=args.repo_root, require_checksums=args.require_checksums)
        text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_markdown()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(text)
        return 0 if report.ok else 1

    if args.command == "contributor-pack":
        try:
            report = build_contributor_pack(args.output_dir)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"contributor_pack: {'OK' if report.ok else 'FAILED'}")
            print(f"packs: {len(report.pack_statuses)}")
            print(f"templates: {len(report.template_files)}")
            print(f"output_dir: {report.output_dir}")
            print(f"readme: {report.files.get('readme')}")
            print(f"commands: {report.files.get('commands_markdown')}")
        return 0 if report.ok else 1

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
            model_card=getattr(args, "model_card", None),
            require_final_ready=args.require_final_ready,
        )
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(report.to_text())
        return 0 if report.ok else 1

    if args.command == "paper-release-smoke":
        result = run_paper_release_smoke(
            output_dir=args.output_dir,
            episodes=args.episodes,
            seed=args.seed,
            train_model=not args.skip_train,
            repo_root=args.repo_root,
            dataset_manifest=args.dataset_manifest,
        )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(result.to_text())
        if args.require_final_ready and (not result.ok or not result.final_ready):
            return 1
        return 0 if result.ok or not args.strict else 1

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
        elif args.kind == "experiment":
            output = experiment_card(args.input, args.output)
        else:
            output = model_card(args.input, args.output, model_id=args.model_id, metrics_path=args.metrics)
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
    if name == "vap":
        from stable_asr.models.baselines.vap import VAPPredictor
        return VAPPredictor()
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
    with resolve_platform_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("paper config must be a JSON object")
    return payload


def _load_train_turn_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with resolve_platform_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("NanoTurn training config must be a JSON object")
    return payload


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _validation_group_by_arg(value: object) -> str | None:
    text = str(value or "auto").strip()
    if not text or text.lower() in {"none", "off", "false"}:
        return None
    return text


def _benchmark_turn_name(args: argparse.Namespace) -> str:
    if getattr(args, "checkpoint", None):
        return "nanoturn"
    if getattr(args, "predictions", None):
        return "prediction_manifest"
    if getattr(args, "baseline", None):
        return str(args.baseline)
    return "system"


def _required_config_path(config: dict[str, object], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing required --{key.replace('_', '-')} or config field {key!r}")
    return Path(value)


def _resolve_config_path(value: object, *, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("config path must be a non-empty string")
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _write_json_output(path: Path | None, payload: object) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
