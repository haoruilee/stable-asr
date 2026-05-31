"""Final-scale experiment registry for the Stable-ASR platform paper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.resources import resolve_platform_path


DEFAULT_FINAL_EXPERIMENTS: dict[str, Any] = {
    "id": "stable_asr_final_experiments_v0",
    "version": "0.1.0",
    "title": "Stable-ASR Final-Scale Experiment Plan",
    "description": (
        "Machine-readable runbook for turning the smoke-level Stable-ASR platform "
        "artifacts into final-scale paper experiments. These entries are planned "
        "experiments, not completed benchmark claims."
    ),
    "experiments": [
        {
            "id": "real_data_layer_benchmark",
            "title": "Real Corpus Data-Layer Benchmark",
            "research_question": "RQ1",
            "paper_section": "Data Layer",
            "status": "planned",
            "priority": "P0",
            "required_inputs": [
                "one public ASR corpus manifest, for example LibriSpeech, AISHELL-1, WenetSpeech, or Common Voice",
                "local audio root on NVMe",
                "optional object-storage mirror for remote-read experiments"
            ],
            "commands": [
                "stable-asr prepare-public-asr --corpus librispeech --input-dir data/librispeech/LibriSpeech/dev-clean --output runs/final/librispeech_dev_clean/asr_manifest.jsonl",
                "stable-asr prepare-public-asr --corpus aishell1 --input-dir data/aishell1/data_aishell --split dev --output runs/final/aishell1_dev/asr_manifest.jsonl",
                "stable-asr prepare-public-asr --corpus wenetspeech --input-dir data/wenetspeech/WenetSpeech --split dev --output runs/final/wenetspeech_dev/asr_manifest.jsonl",
                "stable-asr prepare-public-asr --corpus common_voice --input-dir data/common_voice/en --split dev --output runs/final/common_voice_en_dev/asr_manifest.jsonl",
                "stable-asr benchmark-data --dataset runs/final/turn_train.jsonl --output-dir runs/final/data_bench --formats jsonl parquet lance --sample-count 10000 --json-output runs/final/reports/data_benchmark.json"
            ],
            "metrics": ["write_seconds", "read_seconds", "size_bytes", "samples_per_second", "conversion_seconds"],
            "expected_artifacts": ["tables/data.md", "figures/data.svg", "DATASET_CARD.md"],
            "success_criteria": [
                "JSONL, Parquet, and Lance rows are present",
                "random sampling throughput is reported for at least 10000 sampled windows",
                "storage size and conversion time are reported for the same input corpus"
            ]
        },
        {
            "id": "external_turn_baselines",
            "title": "External Turn-Taking Baseline Comparison",
            "research_question": "RQ2",
            "paper_section": "Baselines and Adapters",
            "status": "planned",
            "priority": "P0",
            "required_inputs": [
                "held-out Stable-ASR turn manifest",
                "SmartTurn-style predictions or command export",
                "EasyTurn-style predictions or command export",
                "VAP-style predictions when available"
            ],
            "commands": [
                "stable-asr convert-predictions --schema smart_turn --input runs/final/external/smartturn_raw.jsonl --output runs/final/external/smartturn_predictions.jsonl",
                "stable-asr convert-predictions --schema easyturn --input runs/final/external/easyturn_raw.jsonl --output runs/final/external/easyturn_predictions.jsonl",
                "stable-asr compare-turn --dataset runs/final/turn_test.jsonl --baseline rule_endpoint --baseline vad_pause --baseline text_turn --predictions smart_turn=runs/final/external/smartturn_predictions.jsonl --predictions easy_turn=runs/final/external/easyturn_predictions.jsonl --checkpoint nanoturn=runs/final/nanoturn/checkpoint.pt --report runs/final/reports/baselines.md --json-output runs/final/reports/baselines.json",
                "stable-asr benchmark-turn --dataset runs/final/turn_test.jsonl --checkpoint runs/final/nanoturn/checkpoint.pt --artifact runs/final/nanoturn/checkpoint.pt --artifact runs/final/nanoturn/metrics.json --report runs/final/reports/turn_benchmarks.md --json-output runs/final/reports/turn_benchmarks.json"
            ],
            "metrics": ["macro_f1", "false_complete_rate", "missed_interrupt_rate", "backchannel_precision", "avg_latency_ms", "p95_latency_ms"],
            "expected_artifacts": ["tables/baselines.md", "tables/turn_benchmark.md", "ADAPTERS.md"],
            "success_criteria": [
                "at least two external turn systems are compared through the shared evaluator",
                "NanoTurn checkpoint-backed results are present on the same split",
                "latency and interaction metrics are reported for every system"
            ]
        },
        {
            "id": "real_voiceworld_scenarios",
            "title": "Real-Audio VoiceWorld Scenario Evaluation",
            "research_question": "RQ4",
            "paper_section": "VoiceWorld Scenario Suite",
            "status": "planned",
            "priority": "P0",
            "required_inputs": [
                "curated real-audio or licensed scenario manifest",
                "scenario tags for interruption, backchannel, side speech, ambient speech, noisy far-field, and code-switching",
                "factor annotations for SNR, reverb, accent, overlap offset, assistant state, and language"
            ],
            "commands": [
                "stable-asr final-config --config configs/final/paper_final.json --prepare-voiceworld-real",
                "stable-asr validate-manifest runs/final/voiceworld_real.jsonl",
                "stable-asr eval-scenario --dataset runs/final/voiceworld_real.jsonl --checkpoint runs/final/nanoturn/checkpoint.pt --seed 0 --report runs/final/reports/scenarios.md --json-output runs/final/reports/scenarios.json",
                "stable-asr paper-table scenarios --results runs/final/paper_results.json --output runs/final/artifacts/tables/scenarios.md"
            ],
            "metrics": ["accuracy", "macro_f1", "false_complete_rate", "missed_interrupt_rate", "premature_response_rate"],
            "expected_artifacts": ["SCENARIO_SUITE.md", "tables/scenarios.md", "figures/robustness_heatmap.svg", "CASE_STUDIES.md"],
            "success_criteria": [
                "all v0 scenario ids have real or licensed examples",
                "scenario metrics are broken down by controllable factors",
                "case studies cite source record ids and failure categories"
            ]
        },
        {
            "id": "policy_transfer",
            "title": "Policy Search And Transfer",
            "research_question": "RQ3",
            "paper_section": "Policy Layer",
            "status": "planned",
            "priority": "P1",
            "required_inputs": [
                "development turn manifest for threshold search",
                "held-out scenario manifest for transfer evaluation",
                "cost matrix covering false complete, missed interruption, false interruption, backchannel break, and latency"
            ],
            "commands": [
                "stable-asr optimize-policy --dataset runs/final/turn_dev.jsonl --baseline vad_pause --output runs/final/reports/policy_search.json",
                "stable-asr eval-turn --dataset runs/final/turn_test.jsonl --checkpoint runs/final/nanoturn/checkpoint.pt --report runs/final/reports/policy_transfer.md"
            ],
            "metrics": ["objective_score", "false_complete_rate", "missed_interrupt_rate", "decision_latency_ms"],
            "expected_artifacts": ["tables/policy.md", "figures/policy_state_machine.svg"],
            "success_criteria": [
                "policy is selected on development data only",
                "transfer metrics are reported on held-out scenarios",
                "threshold/latency trade-offs are plotted or tabulated"
            ]
        },
        {
            "id": "real_streaming_asr_systems",
            "title": "Real Streaming ASR System Comparison",
            "research_question": "RQ5",
            "paper_section": "Streaming ASR Evaluation",
            "status": "planned",
            "priority": "P0",
            "required_inputs": [
                "command-backed exports for at least two ASR systems, for example Whisper, FunASR, WeNet, or NeMo",
                "shared reference transcript manifest",
                "streaming partial hypotheses and timestamp outputs"
            ],
            "commands": [
                "stable-asr final-config --config configs/final/paper_final.json --prepare-asr-eval-manifest",
                "stable-asr final-config --config configs/final/paper_final.json --audit-asr-commands",
                "stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --report runs/final/reports/asr_command_compare.md --json-output runs/final/reports/asr_command_compare.json",
                "stable-asr sweep-streaming-asr --input runs/final/asr_commands/whisper_streaming.jsonl --chunks-ms 160 320 640 --lookahead-ms 0 160 320 --report runs/final/reports/whisper_sweep.md --json-output runs/final/reports/whisper_sweep.json",
                "stable-asr final-config --config configs/final/paper_final.json --prepare-asr-transcript-conversions"
            ],
            "metrics": ["wer", "cer", "rtf", "first_partial_latency", "final_latency", "endpoint_delay", "partial_revision_rate", "stable_prefix_ratio", "timestamp_drift"],
            "expected_artifacts": ["tables/streaming.md", "tables/streaming_sweep.md", "tables/streaming_failures.md", "leaderboard.jsonl"],
            "success_criteria": [
                "at least two real ASR systems have comparable rows",
                "chunk/lookahead sensitivity is reported for at least one system",
                "failure mining identifies record-level streaming failures"
            ]
        },
        {
            "id": "final_reproducibility_bundle",
            "title": "Final Reproducibility Bundle",
            "research_question": "RQ6",
            "paper_section": "Reproducibility",
            "status": "planned",
            "priority": "P0",
            "required_inputs": [
                "final experiment config",
                "final paper_results.json",
                "dataset cards",
                "experiment cards",
                "model cards",
                "exact model checkpoints or adapter prediction manifests"
            ],
            "commands": [
                "stable-asr final-results --config configs/final/paper_final.json --output runs/final/paper_results.json",
                "stable-asr final-inputs --registry configs/final/input_collections.json --output runs/final/FINAL_INPUT_COLLECTIONS.md",
                "stable-asr make-card model --input configs/models/stable_asr_models.json --model-id nanoturn_pico --metrics runs/final/nanoturn/metrics.json --output runs/final/MODEL_CARD.md",
                "stable-asr paper-bundle --results runs/final/paper_results.json --output-dir runs/final/artifacts",
                "stable-asr paper-artifact-integrity --manifest runs/final/artifacts/artifact_hashes.json --root runs/final/artifacts",
                "stable-asr paper-archive --artifacts-dir runs/final/artifacts --output runs/final/artifacts.tar.gz",
                "stable-asr paper-archive-verify --archive runs/final/artifacts.tar.gz",
                "stable-asr paper-parity-audit --results runs/final/paper_results.json --artifacts-dir runs/final/artifacts --require-final",
                "stable-asr paper-release-audit --repo-root . --results runs/final/paper_results.json --artifacts-dir runs/final/artifacts --markdown-draft runs/final/PAPER_DRAFT.md --latex-draft runs/final/paper.tex --dataset-card runs/final/DATASET_CARD.md --experiment-card runs/final/EXPERIMENT_CARD.md --model-card runs/final/MODEL_CARD.md"
            ],
            "metrics": ["paper_release_audit", "paper_parity_audit", "benchmark_suite_coverage", "claim_audit"],
            "expected_artifacts": [
                "ARTIFACT_INDEX.md",
                "paper_results.json",
                "artifact_manifest.json",
                "artifact_hashes.json",
                "PROVENANCE.md",
                "PAPER_PARITY.md",
                "FINAL_INPUT_COLLECTIONS.md",
                "CLAIMS.md",
                "MODEL_CARD.md",
                "MODELS.md",
                "artifacts.tar.gz",
                "paper.tex",
            ],
            "success_criteria": [
                "clean reproduction from a fresh environment succeeds",
                "paper parity audit has no remaining final-scale gaps",
                "all benchmark claims point to archived configs and artifacts"
            ]
        }
    ],
}


@dataclass(frozen=True)
class FinalExperimentValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "final_experiments: OK"
        return "final_experiments: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


def load_final_experiments(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_FINAL_EXPERIMENTS))
    with resolve_platform_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("final experiment registry must be a JSON object")
    return payload


def write_final_experiments_json(path: str | Path, registry: dict[str, Any] | None = None) -> str:
    registry = registry or load_final_experiments()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_final_experiments(registry: dict[str, Any]) -> FinalExperimentValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "experiments"):
        if key not in registry:
            errors.append(f"missing top-level key: {key}")
    experiments = registry.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        errors.append("experiments must be a non-empty list")
        return FinalExperimentValidation(ok=False, errors=errors)

    seen: set[str] = set()
    required = {
        "id",
        "title",
        "research_question",
        "paper_section",
        "status",
        "priority",
        "required_inputs",
        "commands",
        "metrics",
        "expected_artifacts",
        "success_criteria",
    }
    allowed_status = {"planned", "ready_to_run", "running", "completed", "blocked"}
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            errors.append(f"experiment {index} must be an object")
            continue
        experiment_id = experiment.get("id")
        if not isinstance(experiment_id, str) or not experiment_id:
            errors.append(f"experiment {index} missing id")
        elif experiment_id in seen:
            errors.append(f"duplicate experiment id: {experiment_id}")
        else:
            seen.add(experiment_id)
        for key in required:
            if key not in experiment:
                errors.append(f"experiment {experiment_id or index} missing {key}")
        status = experiment.get("status")
        if "status" in experiment and status not in allowed_status:
            errors.append(f"experiment {experiment_id or index} has unknown status: {status}")
        for key in ("required_inputs", "commands", "metrics", "expected_artifacts", "success_criteria"):
            if key in experiment and not isinstance(experiment[key], list):
                errors.append(f"experiment {experiment_id or index} {key} must be a list")
            elif key in experiment and not experiment[key]:
                errors.append(f"experiment {experiment_id or index} {key} must be non-empty")
    return FinalExperimentValidation(ok=not errors, errors=errors)


def final_experiments_markdown(registry: dict[str, Any]) -> str:
    validation = validate_final_experiments(registry)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    lines = [
        f"# {registry['title']}",
        "",
        f"- id: `{registry['id']}`",
        f"- version: `{registry['version']}`",
        "",
        str(registry.get("description", "")),
        "",
        "## Experiments",
        "",
        dict_table(_experiment_rows(registry)),
        "",
        "## Runbook",
        "",
    ]
    for experiment in registry["experiments"]:
        lines.extend(_experiment_markdown(experiment))
    return "\n".join(lines)


def _experiment_rows(registry: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "id": experiment["id"],
            "rq": experiment["research_question"],
            "section": experiment["paper_section"],
            "status": experiment["status"],
            "priority": experiment["priority"],
            "metrics": len(experiment.get("metrics", [])),
            "commands": len(experiment.get("commands", [])),
        }
        for experiment in registry["experiments"]
    ]


def _experiment_markdown(experiment: dict[str, Any]) -> list[str]:
    lines = [
        f"### {experiment['id']}",
        "",
        str(experiment["title"]),
        "",
        f"- research question: `{experiment['research_question']}`",
        f"- paper section: `{experiment['paper_section']}`",
        f"- status: `{experiment['status']}`",
        f"- priority: `{experiment['priority']}`",
        "",
        "Required inputs:",
    ]
    lines.extend(f"- {item}" for item in experiment.get("required_inputs", []))
    lines.extend(["", "Commands:", ""])
    lines.extend(f"```bash\n{command}\n```" for command in experiment.get("commands", []))
    lines.extend(["Metrics:"])
    lines.extend(f"- `{metric}`" for metric in experiment.get("metrics", []))
    lines.extend(["", "Expected artifacts:"])
    lines.extend(f"- `{artifact}`" for artifact in experiment.get("expected_artifacts", []))
    lines.extend(["", "Success criteria:"])
    lines.extend(f"- {criterion}" for criterion in experiment.get("success_criteria", []))
    lines.append("")
    return lines
