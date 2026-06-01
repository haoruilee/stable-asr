"""Final paper run configuration schema and renderer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.asr_manifest import load_asr_manifest, summarize_asr_records, write_asr_manifest
from stable_asr.data.recipes import prepare_asr_manifest, prepare_public_asr_manifest, prepare_voiceworld_manifest
from stable_asr.data.registry import load_turn_records, summarize_records, write_turn_records
from stable_asr.data.split import SPLIT_NAMES, TurnSplitConfig, split_turn_records
from stable_asr.data.turn_from_asr import ASRToTurnConfig, asr_records_to_turn_records
from stable_asr.eval.report import dict_table
from stable_asr.models.adapters import convert_turn_prediction_jsonl, validate_turn_prediction_jsonl
from stable_asr.resources import resolve_platform_path
from stable_asr.scenarios.suites import load_scenario_suite, validate_scenario_suite
from stable_asr.streaming.command_compare import (
    ASRCommandConfigAuditReport,
    audit_asr_command_config,
    load_asr_command_config,
)
from stable_asr.streaming.compare import compare_streaming_transcript_jsonl


DEFAULT_FINAL_RUN_CONFIG: dict[str, Any] = {
    "id": "stable_asr_final_run_v0",
    "version": "0.1.0",
    "title": "Stable-ASR Final Paper Run Configuration",
    "description": (
        "Template configuration for a final-scale Stable-ASR platform paper run. "
        "Paths are intentionally explicit so the final benchmark can be audited "
        "before expensive jobs are launched."
    ),
    "output_dir": "runs/final",
    "seed": 0,
    "public_corpora": [
        {
            "id": "librispeech_dev_clean",
            "language": "en",
            "corpus": "librispeech",
            "input_dir": "data/librispeech/LibriSpeech/dev-clean",
            "manifest": "runs/final/librispeech_dev_clean/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "see_upstream",
            "required": True,
        },
        {
            "id": "aishell1_dev",
            "language": "zh",
            "corpus": "aishell1",
            "input_dir": "data/aishell1/data_aishell",
            "split": "dev",
            "manifest": "runs/final/aishell1_dev/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "see_upstream",
            "required": True,
        },
        {
            "id": "wenetspeech_dev",
            "language": "zh",
            "corpus": "wenetspeech",
            "input_dir": "data/wenetspeech/WenetSpeech",
            "split": "dev",
            "manifest": "runs/final/wenetspeech_dev/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "see_upstream",
            "required": False,
        },
        {
            "id": "common_voice_en_dev",
            "language": "en",
            "corpus": "common_voice",
            "input_dir": "data/common_voice/en",
            "split": "dev",
            "manifest": "runs/final/common_voice_en_dev/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "see_upstream",
            "required": False,
        },
    ],
    "asr_eval_manifest": "runs/final/asr_eval_manifest.jsonl",
    "turn_splits": {
        "train": "runs/final/turn_train.jsonl",
        "dev": "runs/final/turn_dev.jsonl",
        "test": "runs/final/turn_test.jsonl",
        "voiceworld_real": "runs/final/voiceworld_real.jsonl",
    },
    "voiceworld_real": {
        "metadata": "data/voiceworld/metadata.tsv",
        "audio_root": "data/voiceworld/audio",
        "manifest": "runs/final/voiceworld_real.jsonl",
        "sample_rate": 16000,
        "language": "zh",
        "source": "voiceworld_real",
        "required": True,
    },
    "external_turn_predictions": [
        {
            "id": "smart_turn",
            "schema": "smart_turn",
            "raw": "runs/final/external/smartturn_raw.jsonl",
            "converted": "runs/final/external/smartturn_predictions.jsonl",
        },
        {
            "id": "easy_turn",
            "schema": "easyturn",
            "raw": "runs/final/external/easyturn_raw.jsonl",
            "converted": "runs/final/external/easyturn_predictions.jsonl",
        },
        {
            "id": "vap",
            "schema": "vap",
            "raw": "runs/final/external/vap_raw.jsonl",
            "converted": "runs/final/external/vap_predictions.jsonl",
        },
    ],
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
        "artifact_archive": "runs/final/artifacts.tar.gz",
        "markdown_draft": "runs/final/PAPER_DRAFT.md",
        "latex_draft": "runs/final/paper.tex",
        "dataset_card": "runs/final/DATASET_CARD.md",
        "experiment_card": "runs/final/EXPERIMENT_CARD.md",
        "model_card": "runs/final/MODEL_CARD.md",
        "handoff": "runs/final/FINAL_INPUT_HANDOFF.json",
        "assignment_audit": "runs/final/FINAL_ASSIGNMENT_AUDIT.md",
        "handoff_schema_validation": "runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md",
        "handoff_audit": "runs/final/FINAL_HANDOFF_AUDIT.md",
    },
    "result_inputs": {
        "data_benchmark": "runs/final/reports/data_benchmark.json",
        "baselines": "runs/final/reports/baselines.json",
        "turn_benchmarks": "runs/final/reports/turn_benchmarks.json",
        "scenarios": "runs/final/reports/scenarios.json",
        "policy_search": "runs/final/reports/policy_search.json",
        "streaming_comparison": "runs/final/reports/asr_command_compare.json",
        "streaming_sweep": "runs/final/reports/whisper_sweep.json",
        "asr_transcript_conversions": "runs/final/reports/asr_transcript_conversions.json",
        "nanoturn": "runs/final/nanoturn/metrics.json",
    },
    "commands": [
        "stable-asr final-config --config configs/final/paper_final.json --validate-only",
        "stable-asr final-config --config configs/final/paper_final.json --prepare-inputs",
        "stable-asr final-config --config configs/final/paper_final.json --prepare-corpora",
        "stable-asr prepare-public-asr --corpus librispeech --input-dir data/librispeech/LibriSpeech/dev-clean --output runs/final/librispeech_dev_clean/asr_manifest.jsonl",
        "stable-asr prepare-public-asr --corpus aishell1 --input-dir data/aishell1/data_aishell --split dev --output runs/final/aishell1_dev/asr_manifest.jsonl",
        "stable-asr prepare-public-asr --corpus wenetspeech --input-dir data/wenetspeech/WenetSpeech --split dev --output runs/final/wenetspeech_dev/asr_manifest.jsonl",
        "stable-asr prepare-public-asr --corpus common_voice --input-dir data/common_voice/en --split dev --output runs/final/common_voice_en_dev/asr_manifest.jsonl",
        "stable-asr final-config --config configs/final/paper_final.json --prepare-asr-eval-manifest",
        "stable-asr final-config --config configs/final/paper_final.json --bootstrap-turn-splits",
        "stable-asr final-config --config configs/final/paper_final.json --prepare-external-predictions",
        "stable-asr convert-predictions --schema vap --input runs/final/external/vap_raw.jsonl --output runs/final/external/vap_predictions.jsonl",
        "stable-asr final-config --config configs/final/paper_final.json --prepare-voiceworld-real",
        "stable-asr final-config --config configs/final/paper_final.json --audit-voiceworld-real --scenario-suite configs/scenarios/stable_asr_voiceworld_v0.json",
        "stable-asr final-config --config configs/final/paper_final.json --audit-asr-commands",
        "stable-asr final-config --config configs/final/paper_final.json --plan-missing --output runs/final/FINAL_RUN_ACTION_PLAN.md",
        "stable-asr final-assignment-audit --input runs/final_acquisition_pack/acquisition/assignments.json --require-owner --require-due-date --require-ready --output runs/final/FINAL_ASSIGNMENT_AUDIT.md",
        "stable-asr final-handoff-template --output runs/final/FINAL_INPUT_HANDOFF.json",
        "stable-asr final-handoff-checksums --input runs/final/FINAL_INPUT_HANDOFF.json --repo-root . --output runs/final/FINAL_INPUT_HANDOFF.json",
        "stable-asr validate-schema-file --input runs/final/FINAL_INPUT_HANDOFF.json --schema-id stable_asr.final_handoff.v0 --output runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md",
        "stable-asr final-handoff-audit --input runs/final/FINAL_INPUT_HANDOFF.json --repo-root . --require-checksums --output runs/final/FINAL_HANDOFF_AUDIT.md",
        "stable-asr train-turn --dataset runs/final/turn_train.jsonl --output-dir runs/final/nanoturn --model nanoturn_pico --feature-source audio",
        "stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --report runs/final/reports/asr_command_compare.md --json-output runs/final/reports/asr_command_compare.json",
        "stable-asr sweep-streaming-asr --input runs/final/asr_commands/whisper_streaming.jsonl --chunks-ms 160 320 640 --lookahead-ms 0 160 320 --report runs/final/reports/whisper_sweep.md --json-output runs/final/reports/whisper_sweep.json",
        "stable-asr final-config --config configs/final/paper_final.json --prepare-asr-transcript-conversions",
        "stable-asr final-results --config configs/final/paper_final.json --output runs/final/paper_results.json",
        "stable-asr make-card model --input configs/models/stable_asr_models.json --model-id nanoturn_pico --metrics runs/final/nanoturn/metrics.json --output runs/final/MODEL_CARD.md",
        "stable-asr paper-bundle --results runs/final/paper_results.json --output-dir runs/final/artifacts",
        "stable-asr paper-artifact-integrity --manifest runs/final/artifacts/artifact_hashes.json --root runs/final/artifacts",
        "stable-asr paper-archive --artifacts-dir runs/final/artifacts --output runs/final/artifacts.tar.gz",
        "stable-asr paper-archive-verify --archive runs/final/artifacts.tar.gz",
        "stable-asr paper-parity-audit --results runs/final/paper_results.json --artifacts-dir runs/final/artifacts --require-final",
    ],
}


@dataclass(frozen=True)
class FinalRunConfigValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "final_run_config: OK"
        return "final_run_config: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class FinalRunPathCheck:
    name: str
    path: str
    kind: str
    required: bool
    exists: bool
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "required": self.required,
            "exists": self.exists,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FinalRunFileAudit:
    ok: bool
    checks: list[FinalRunPathCheck]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}

    def to_text(self) -> str:
        lines = [f"final_run_file_audit: {'READY' if self.ok else 'NOT_READY'}"]
        for check in self.checks:
            status = "OK" if check.ok else "MISSING"
            required = "required" if check.required else "planned"
            lines.append(f"- {status} {check.kind}/{check.name}: {check.path} ({required}; {check.detail})")
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalRunActionItem:
    id: str
    title: str
    status: str
    blockers: list[str]
    commands: list[str]
    artifacts: list[str]
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "blockers": self.blockers,
            "commands": self.commands,
            "artifacts": self.artifacts,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FinalRunActionPlan:
    ok: bool
    config_path: str
    missing_required: list[FinalRunPathCheck]
    items: list[FinalRunActionItem]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "config_path": self.config_path,
            "missing_required": [check.to_dict() for check in self.missing_required],
            "items": [item.to_dict() for item in self.items],
        }

    def to_text(self) -> str:
        lines = [
            f"final_run_action_plan: {'READY' if self.ok else 'NOT_READY'}",
            f"config: {self.config_path}",
            f"missing_required: {len(self.missing_required)}",
        ]
        for item in self.items:
            blockers = ", ".join(item.blockers) if item.blockers else "none"
            lines.append(f"- {item.status.upper()} {item.id}: {item.title} (blockers: {blockers})")
            lines.extend(f"  command: {command}" for command in item.commands)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        item_rows = [
            {
                "step": index,
                "id": item.id,
                "status": item.status,
                "blockers": len(item.blockers),
                "artifacts": ", ".join(item.artifacts),
            }
            for index, item in enumerate(self.items, start=1)
        ]
        missing_rows = [
            {
                "kind": check.kind,
                "name": check.name,
                "path": check.path,
                "detail": check.detail,
                "suggested_action": _suggest_action_for_missing_check(check),
            }
            for check in self.missing_required
        ]
        lines = [
            "# Stable-ASR Final Run Action Plan",
            "",
            f"- status: `{'READY' if self.ok else 'NOT_READY'}`",
            f"- config: `{self.config_path}`",
            f"- missing_required: `{len(self.missing_required)}`",
            "",
            "## Missing Required Inputs",
            "",
            dict_table(missing_rows) if missing_rows else "No required inputs are missing.",
            "",
            "## Execution Plan",
            "",
            dict_table(item_rows),
        ]
        for index, item in enumerate(self.items, start=1):
            lines.extend(
                [
                    "",
                    f"### {index}. {item.title}",
                    "",
                    f"- id: `{item.id}`",
                    f"- status: `{item.status}`",
                    f"- detail: {item.detail}",
                    "",
                    "Blockers:",
                    "",
                ]
            )
            lines.extend(f"- `{blocker}`" for blocker in item.blockers) if item.blockers else lines.append("- none")
            lines.extend(["", "Commands:", ""])
            lines.extend(f"```bash\n{command}\n```" for command in item.commands) if item.commands else lines.append("- none")
            lines.extend(["", "Artifacts:", ""])
            lines.extend(f"- `{artifact}`" for artifact in item.artifacts) if item.artifacts else lines.append("- none")
        lines.append("")
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalRunScaffoldEntry:
    path: str
    kind: str
    created: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "created": self.created,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FinalRunScaffoldReport:
    output_dir: str
    entries: list[FinalRunScaffoldEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_text(self) -> str:
        lines = [f"final_run_scaffold: {self.output_dir}"]
        for entry in self.entries:
            status = "created" if entry.created else "exists"
            lines.append(f"- {status} {entry.kind}: {entry.path} ({entry.detail})")
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalCorpusPrepareEntry:
    id: str
    corpus: str
    input: str
    manifest: str
    records: int
    ok: bool
    skipped: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "corpus": self.corpus,
            "input": self.input,
            "manifest": self.manifest,
            "records": self.records,
            "ok": self.ok,
            "skipped": self.skipped,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FinalCorpusPrepareReport:
    ok: bool
    require_all: bool
    entries: list[FinalCorpusPrepareEntry]

    @property
    def prepared_count(self) -> int:
        return sum(1 for entry in self.entries if entry.ok and not entry.skipped)

    @property
    def skipped_count(self) -> int:
        return sum(1 for entry in self.entries if entry.skipped)

    @property
    def failed_count(self) -> int:
        return sum(1 for entry in self.entries if not entry.ok and not entry.skipped)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "require_all": self.require_all,
            "prepared_count": self.prepared_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_text(self) -> str:
        if self.ok and self.prepared_count:
            status = "READY"
        elif self.ok:
            status = "NO_INPUTS"
        else:
            status = "FAILED"
        lines = [
            f"final_corpora_prepare: {status}",
            f"- prepared: {self.prepared_count}",
            f"- skipped: {self.skipped_count}",
            f"- failed: {self.failed_count}",
        ]
        for entry in self.entries:
            if entry.ok and not entry.skipped:
                entry_status = "PREPARED"
            elif entry.skipped:
                entry_status = "SKIPPED"
            else:
                entry_status = "FAILED"
            lines.append(
                f"- {entry_status} {entry.id}: {entry.records} record(s) -> {entry.manifest} ({entry.detail})"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalTurnBootstrapReport:
    ok: bool
    input_manifests: list[str]
    skipped_manifests: list[str]
    asr_records: int
    turn_records: int
    split_paths: dict[str, str]
    split_counts: dict[str, int]
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "input_manifests": self.input_manifests,
            "skipped_manifests": self.skipped_manifests,
            "asr_records": self.asr_records,
            "turn_records": self.turn_records,
            "split_paths": self.split_paths,
            "split_counts": self.split_counts,
            "detail": self.detail,
        }

    def to_text(self) -> str:
        status = "READY" if self.ok else "NOT_READY"
        lines = [
            f"final_turn_bootstrap: {status}",
            f"- input_manifests: {len(self.input_manifests)}",
            f"- skipped_manifests: {len(self.skipped_manifests)}",
            f"- asr_records: {self.asr_records}",
            f"- turn_records: {self.turn_records}",
            f"- detail: {self.detail}",
        ]
        for name in SPLIT_NAMES:
            lines.append(f"- {name}: {self.split_counts.get(name, 0)} record(s) -> {self.split_paths.get(name, '')}")
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalASREvalManifestReport:
    ok: bool
    output: str
    input_manifests: list[str]
    skipped_manifests: list[str]
    records: int
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "output": self.output,
            "input_manifests": self.input_manifests,
            "skipped_manifests": self.skipped_manifests,
            "records": self.records,
            "detail": self.detail,
        }

    def to_text(self) -> str:
        status = "READY" if self.ok else "NOT_READY"
        lines = [
            f"final_asr_eval_manifest: {status}",
            f"- output: {self.output}",
            f"- input_manifests: {len(self.input_manifests)}",
            f"- skipped_manifests: {len(self.skipped_manifests)}",
            f"- records: {self.records}",
            f"- detail: {self.detail}",
        ]
        lines.extend(f"  - {path}" for path in self.input_manifests)
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalExternalPredictionEntry:
    id: str
    schema: str
    raw: str
    converted: str
    converted_records: int
    ok: bool
    skipped: bool
    coverage_checked: bool
    missing_ids: int
    extra_ids: int
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "schema": self.schema,
            "raw": self.raw,
            "converted": self.converted,
            "converted_records": self.converted_records,
            "ok": self.ok,
            "skipped": self.skipped,
            "coverage_checked": self.coverage_checked,
            "missing_ids": self.missing_ids,
            "extra_ids": self.extra_ids,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FinalExternalPredictionReport:
    ok: bool
    dataset_path: str
    dataset_records: int
    require_all: bool
    entries: list[FinalExternalPredictionEntry]

    @property
    def prepared_count(self) -> int:
        return sum(1 for entry in self.entries if entry.ok and not entry.skipped)

    @property
    def skipped_count(self) -> int:
        return sum(1 for entry in self.entries if entry.skipped)

    @property
    def failed_count(self) -> int:
        return sum(1 for entry in self.entries if not entry.ok and not entry.skipped)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "dataset_path": self.dataset_path,
            "dataset_records": self.dataset_records,
            "require_all": self.require_all,
            "prepared_count": self.prepared_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_text(self) -> str:
        if self.ok and self.prepared_count:
            status = "READY"
        elif self.ok:
            status = "NO_INPUTS"
        else:
            status = "FAILED"
        lines = [
            f"final_external_predictions_prepare: {status}",
            f"- dataset: {self.dataset_path}",
            f"- dataset_records: {self.dataset_records}",
            f"- prepared: {self.prepared_count}",
            f"- skipped: {self.skipped_count}",
            f"- failed: {self.failed_count}",
        ]
        for entry in self.entries:
            if entry.ok and not entry.skipped:
                entry_status = "PREPARED"
            elif entry.skipped:
                entry_status = "SKIPPED"
            else:
                entry_status = "FAILED"
            coverage = "coverage checked" if entry.coverage_checked else "coverage skipped"
            lines.append(
                f"- {entry_status} {entry.id}: {entry.converted_records} record(s) -> "
                f"{entry.converted} ({coverage}; {entry.detail})"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalVoiceWorldPrepareReport:
    ok: bool
    metadata: str
    audio_root: str
    manifest: str
    records: int
    skipped: bool
    audit: FinalVoiceWorldAuditReport | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "metadata": self.metadata,
            "audio_root": self.audio_root,
            "manifest": self.manifest,
            "records": self.records,
            "skipped": self.skipped,
            "audit": self.audit.to_dict() if self.audit else None,
            "detail": self.detail,
        }

    def to_text(self) -> str:
        if self.ok and not self.skipped:
            status = "READY"
        elif self.skipped:
            status = "SKIPPED"
        else:
            status = "FAILED"
        lines = [
            f"final_voiceworld_real_prepare: {status}",
            f"- metadata: {self.metadata}",
            f"- audio_root: {self.audio_root}",
            f"- manifest: {self.manifest}",
            f"- records: {self.records}",
            f"- detail: {self.detail}",
        ]
        if self.audit:
            lines.extend(["", self.audit.to_text()])
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalASRTranscriptConversionReport:
    ok: bool
    output: str
    input_paths: dict[str, str]
    missing_inputs: dict[str, str]
    records_by_adapter: dict[str, int]
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "output": self.output,
            "input_paths": self.input_paths,
            "missing_inputs": self.missing_inputs,
            "records_by_adapter": self.records_by_adapter,
            "detail": self.detail,
        }

    def to_text(self) -> str:
        status = "READY" if self.ok else "NOT_READY"
        lines = [
            f"final_asr_transcript_conversions: {status}",
            f"- output: {self.output}",
            f"- adapters: {len(self.input_paths)}",
            f"- missing_inputs: {len(self.missing_inputs)}",
            f"- detail: {self.detail}",
        ]
        for name, path in self.input_paths.items():
            records = self.records_by_adapter.get(name, 0)
            marker = "MISSING" if name in self.missing_inputs else "OK"
            lines.append(f"- {marker} {name}: {path} ({records} record(s))")
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalInputPrepareReport:
    ok: bool
    corpora: FinalCorpusPrepareReport
    asr_eval_manifest: FinalASREvalManifestReport
    turn_splits: FinalTurnBootstrapReport
    external_predictions: FinalExternalPredictionReport
    voiceworld_prepare: FinalVoiceWorldPrepareReport
    voiceworld_real: FinalVoiceWorldAuditReport
    asr_command_config: ASRCommandConfigAuditReport
    file_audit: FinalRunFileAudit

    @property
    def missing_required(self) -> list[str]:
        return [
            check.path
            for check in self.file_audit.checks
            if check.required and not check.ok
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "corpora": self.corpora.to_dict(),
            "asr_eval_manifest": self.asr_eval_manifest.to_dict(),
            "turn_splits": self.turn_splits.to_dict(),
            "external_predictions": self.external_predictions.to_dict(),
            "voiceworld_prepare": self.voiceworld_prepare.to_dict(),
            "voiceworld_real": self.voiceworld_real.to_dict(),
            "asr_command_config": self.asr_command_config.to_dict(),
            "file_audit": self.file_audit.to_dict(),
            "missing_required": self.missing_required,
        }

    def to_text(self) -> str:
        status = "READY" if self.ok else "NOT_READY"
        lines = [
            f"final_inputs_prepare: {status}",
            f"- corpora_prepared: {self.corpora.prepared_count}",
            f"- asr_eval_records: {self.asr_eval_manifest.records}",
            f"- turn_records: {self.turn_splits.turn_records}",
            f"- external_predictions_prepared: {self.external_predictions.prepared_count}",
            f"- voiceworld_prepare_ready: {self.voiceworld_prepare.ok}",
            f"- voiceworld_real_ready: {self.voiceworld_real.ok}",
            f"- asr_commands_ready: {self.asr_command_config.ok}",
            f"- missing_required: {len(self.missing_required)}",
        ]
        lines.extend(f"  - {path}" for path in self.missing_required)
        lines.extend(
            [
                "",
                self.corpora.to_text(),
                "",
                self.asr_eval_manifest.to_text(),
                "",
                self.turn_splits.to_text(),
                "",
                self.external_predictions.to_text(),
                "",
                self.voiceworld_prepare.to_text(),
                "",
                self.voiceworld_real.to_text(),
                "",
                self.asr_command_config.to_text(),
                "",
                self.file_audit.to_text(),
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalVoiceWorldAuditReport:
    ok: bool
    manifest: str
    records: int
    min_per_scenario: int
    scenario_counts: dict[str, int]
    missing_scenarios: list[str]
    undercovered_scenarios: dict[str, int]
    factor_coverage: dict[str, int]
    missing_factor_fields: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "manifest": self.manifest,
            "records": self.records,
            "min_per_scenario": self.min_per_scenario,
            "scenario_counts": self.scenario_counts,
            "missing_scenarios": self.missing_scenarios,
            "undercovered_scenarios": self.undercovered_scenarios,
            "factor_coverage": self.factor_coverage,
            "missing_factor_fields": self.missing_factor_fields,
            "errors": self.errors,
        }

    def to_text(self) -> str:
        lines = [
            f"final_voiceworld_real_audit: {'READY' if self.ok else 'NOT_READY'}",
            f"- manifest: {self.manifest}",
            f"- records: {self.records}",
            f"- min_per_scenario: {self.min_per_scenario}",
            f"- missing_scenarios: {len(self.missing_scenarios)}",
            f"- undercovered_scenarios: {len(self.undercovered_scenarios)}",
            f"- missing_factor_fields: {len(self.missing_factor_fields)}",
        ]
        lines.extend(f"  - {scenario}" for scenario in self.missing_scenarios)
        lines.extend(
            f"  - {scenario}: {count}" for scenario, count in sorted(self.undercovered_scenarios.items())
        )
        if self.missing_factor_fields:
            lines.append("- factors missing from every record:")
            lines.extend(f"  - {factor}" for factor in self.missing_factor_fields)
        if self.errors:
            lines.append("- errors:")
            lines.extend(f"  - {error}" for error in self.errors)
        return "\n".join(lines)


def load_final_run_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_FINAL_RUN_CONFIG))
    with resolve_platform_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("final run config must be a JSON object")
    return payload


def write_final_run_config_json(path: str | Path, config: dict[str, Any] | None = None) -> str:
    config = config or load_final_run_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_final_run_config(config: dict[str, Any]) -> FinalRunConfigValidation:
    errors: list[str] = []
    for key in (
        "id",
        "version",
        "title",
        "output_dir",
        "seed",
        "public_corpora",
        "asr_eval_manifest",
        "turn_splits",
        "artifacts",
        "commands",
    ):
        if key not in config:
            errors.append(f"missing top-level key: {key}")

    corpora = config.get("public_corpora")
    if not isinstance(corpora, list) or not corpora:
        errors.append("public_corpora must be a non-empty list")
    else:
        seen: set[str] = set()
        for index, corpus in enumerate(corpora):
            if not isinstance(corpus, dict):
                errors.append(f"corpus {index} must be an object")
                continue
            corpus_id = corpus.get("id")
            if not isinstance(corpus_id, str) or not corpus_id:
                errors.append(f"corpus {index} missing id")
            elif corpus_id in seen:
                errors.append(f"duplicate corpus id: {corpus_id}")
            else:
                seen.add(corpus_id)
            for key in ("language", "manifest", "sample_rate", "license"):
                if key not in corpus:
                    errors.append(f"corpus {corpus_id or index} missing {key}")
            if "required" in corpus and not isinstance(corpus["required"], bool):
                errors.append(f"corpus {corpus_id or index} required must be a boolean")
            has_public_recipe = isinstance(corpus.get("corpus"), str) and isinstance(corpus.get("input_dir"), str)
            has_metadata_recipe = isinstance(corpus.get("metadata"), str) and isinstance(corpus.get("audio_root"), str)
            if not has_public_recipe and not has_metadata_recipe:
                errors.append(
                    f"corpus {corpus_id or index} must define either corpus/input_dir or metadata/audio_root"
                )

    turn_splits = config.get("turn_splits")
    required_splits = {"train", "dev", "test", "voiceworld_real"}
    if not isinstance(turn_splits, dict):
        errors.append("turn_splits must be an object")
    else:
        missing_splits = sorted(required_splits.difference(turn_splits))
        if missing_splits:
            errors.append("turn_splits missing: " + ", ".join(missing_splits))

    voiceworld_real = config.get("voiceworld_real", {})
    if voiceworld_real is not None and not isinstance(voiceworld_real, dict):
        errors.append("voiceworld_real must be an object")
    elif isinstance(voiceworld_real, dict) and voiceworld_real:
        for key in ("metadata", "audio_root", "manifest", "sample_rate", "language", "source"):
            if key not in voiceworld_real:
                errors.append(f"voiceworld_real missing {key}")
        if "required" in voiceworld_real and not isinstance(voiceworld_real["required"], bool):
            errors.append("voiceworld_real required must be a boolean")
        if (
            isinstance(turn_splits, dict)
            and isinstance(turn_splits.get("voiceworld_real"), str)
            and isinstance(voiceworld_real.get("manifest"), str)
            and voiceworld_real["manifest"] != turn_splits["voiceworld_real"]
        ):
            errors.append("voiceworld_real manifest must match turn_splits.voiceworld_real")

    predictions = config.get("external_turn_predictions", [])
    if predictions is not None and not isinstance(predictions, list):
        errors.append("external_turn_predictions must be a list")
    elif isinstance(predictions, list):
        for index, prediction in enumerate(predictions):
            if not isinstance(prediction, dict):
                errors.append(f"external prediction {index} must be an object")
                continue
            prediction_id = prediction.get("id", index)
            for key in ("id", "schema", "raw", "converted"):
                if key not in prediction:
                    errors.append(f"external prediction {prediction_id} missing {key}")

    nanoturn = config.get("nanoturn", {})
    if not isinstance(nanoturn, dict):
        errors.append("nanoturn must be an object")
    else:
        for key in ("model", "checkpoint", "metrics", "onnx"):
            if key not in nanoturn:
                errors.append(f"nanoturn missing {key}")

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
    else:
        for key in ("paper_results", "bundle_dir", "markdown_draft", "latex_draft", "dataset_card", "experiment_card"):
            if key not in artifacts:
                errors.append(f"artifacts missing {key}")

    result_inputs = config.get("result_inputs", {})
    if result_inputs is not None and not isinstance(result_inputs, dict):
        errors.append("result_inputs must be an object")
    elif isinstance(result_inputs, dict):
        for key, value in result_inputs.items():
            if not isinstance(value, str) or not value:
                errors.append(f"result_inputs {key} must be a non-empty string")

    commands = config.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("commands must be a non-empty list")
    elif not all(isinstance(command, str) and command.strip() for command in commands):
        errors.append("commands must contain non-empty strings")

    asr_command_config = config.get("asr_command_config")
    if not isinstance(asr_command_config, str) or not asr_command_config:
        errors.append("asr_command_config must be a non-empty string")

    asr_eval_manifest = config.get("asr_eval_manifest")
    if not isinstance(asr_eval_manifest, str) or not asr_eval_manifest:
        errors.append("asr_eval_manifest must be a non-empty string")

    return FinalRunConfigValidation(ok=not errors, errors=errors)


def audit_final_run_files(config: dict[str, Any], *, repo_root: str | Path = ".") -> FinalRunFileAudit:
    validation = validate_final_run_config(config)
    if not validation.ok:
        checks = [
            FinalRunPathCheck(
                name="schema",
                path="",
                kind="config",
                required=True,
                exists=False,
                ok=False,
                detail="; ".join(validation.errors),
            )
        ]
        return FinalRunFileAudit(ok=False, checks=checks)

    root = Path(repo_root)
    checks: list[FinalRunPathCheck] = []
    for corpus in config.get("public_corpora", []):
        corpus_id = str(corpus["id"])
        required = bool(corpus.get("required", True))
        if "input_dir" in corpus:
            checks.append(_input_check(f"corpus:{corpus_id}:input_dir", corpus["input_dir"], root=root, required=required))
        else:
            checks.append(_input_check(f"corpus:{corpus_id}:metadata", corpus["metadata"], root=root, required=required))
            checks.append(_input_check(f"corpus:{corpus_id}:audio_root", corpus["audio_root"], root=root, required=required))
        checks.append(_planned_check(f"corpus:{corpus_id}:manifest", corpus["manifest"], root=root, kind="output"))

    checks.append(_planned_check("asr_eval_manifest", config["asr_eval_manifest"], root=root, kind="output"))

    for split, path in config.get("turn_splits", {}).items():
        checks.append(_input_check(f"turn_split:{split}", path, root=root))

    voiceworld = config.get("voiceworld_real") or {}
    if isinstance(voiceworld, dict) and voiceworld:
        required = bool(voiceworld.get("required", True))
        checks.append(_input_check("voiceworld_real:metadata", voiceworld["metadata"], root=root, required=required))
        checks.append(_input_check("voiceworld_real:audio_root", voiceworld["audio_root"], root=root, required=required))
        checks.append(_planned_check("voiceworld_real:manifest", voiceworld["manifest"], root=root, kind="output"))

    for prediction in config.get("external_turn_predictions", []):
        prediction_id = str(prediction["id"])
        checks.append(_input_check(f"external_prediction:{prediction_id}:raw", prediction["raw"], root=root))
        checks.append(_planned_check(f"external_prediction:{prediction_id}:converted", prediction["converted"], root=root, kind="output"))

    checks.append(_input_check("asr_command_config", config["asr_command_config"], root=root, kind="config"))

    for name, path in config.get("nanoturn", {}).items():
        if name == "model":
            continue
        checks.append(_planned_check(f"nanoturn:{name}", path, root=root, kind="output"))

    for name, path in config.get("artifacts", {}).items():
        checks.append(_planned_check(f"artifact:{name}", path, root=root, kind="output"))

    for name, path in config.get("result_inputs", {}).items():
        checks.append(_planned_check(f"result_input:{name}", path, root=root, kind="output"))

    return FinalRunFileAudit(ok=all(check.ok for check in checks), checks=checks)


def final_run_config_markdown(config: dict[str, Any]) -> str:
    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    lines = [
        f"# {config['title']}",
        "",
        f"- id: `{config['id']}`",
        f"- version: `{config['version']}`",
        f"- output dir: `{config['output_dir']}`",
        f"- seed: `{config['seed']}`",
        "",
        str(config.get("description", "")),
        "",
        "## Public Corpora",
        "",
        dict_table(_corpus_rows(config)),
        "",
        "## ASR Eval Manifest",
        "",
        f"- `asr_eval_manifest`: `{config['asr_eval_manifest']}`",
        f"- `asr_command_config`: `{config['asr_command_config']}`",
        "",
        "## Turn Splits",
        "",
    ]
    for name, path in config["turn_splits"].items():
        lines.append(f"- `{name}`: `{path}`")
    voiceworld = config.get("voiceworld_real") or {}
    lines.extend(["", "## VoiceWorld Real Input", ""])
    if isinstance(voiceworld, dict) and voiceworld:
        lines.append(dict_table([{
            "metadata": voiceworld.get("metadata", ""),
            "audio_root": voiceworld.get("audio_root", ""),
            "manifest": voiceworld.get("manifest", ""),
            "language": voiceworld.get("language", ""),
            "required": voiceworld.get("required", True),
        }]))
    else:
        lines.append("No real VoiceWorld input recipe configured.")
    lines.extend(["", "## External Turn Predictions", ""])
    if config.get("external_turn_predictions"):
        lines.append(dict_table(_prediction_rows(config)))
    else:
        lines.append("No external turn predictions configured.")
    lines.extend(["", "## Artifacts", ""])
    for name, path in config["artifacts"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Result Inputs", ""])
    for name, path in config.get("result_inputs", {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Commands", ""])
    lines.extend(f"```bash\n{command}\n```" for command in config["commands"])
    lines.append("")
    return "\n".join(lines)


def final_run_file_audit_markdown(report: FinalRunFileAudit) -> str:
    lines = [
        "# Stable-ASR Final Run File Audit",
        "",
        f"- status: `{'READY' if report.ok else 'NOT_READY'}`",
        "",
        dict_table([check.to_dict() for check in report.checks]),
        "",
    ]
    return "\n".join(lines)


def build_final_run_action_plan(
    config: dict[str, Any],
    *,
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/final/paper_final.json",
) -> FinalRunActionPlan:
    """Build an actionable runbook from the current final-run file audit.

    The plan is intentionally operational: it maps missing required inputs to
    the next command or manual data-staging action without creating placeholder
    benchmark evidence.
    """

    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    audit = audit_final_run_files(config, repo_root=repo_root)
    missing_required = [check for check in audit.checks if check.required and not check.ok]
    config_display = str(config_path)
    items = [
        _stage_corpora_action(config, missing_required, config_display),
        _prepare_asr_turn_action(config, missing_required, config_display),
        _voiceworld_action(config, missing_required, config_display),
        _external_predictions_action(config, missing_required, config_display),
        _nanoturn_action(config, missing_required, config_display),
        _streaming_asr_action(config, missing_required, config_display),
        _final_artifacts_action(config, missing_required, config_display),
    ]
    return FinalRunActionPlan(
        ok=audit.ok,
        config_path=config_display,
        missing_required=missing_required,
        items=items,
    )


def scaffold_final_run(config: dict[str, Any], *, repo_root: str | Path = ".") -> FinalRunScaffoldReport:
    """Create final-run directories and README hints without fabricating data."""

    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    root = Path(repo_root)
    entries: list[FinalRunScaffoldEntry] = []
    output_dir = _resolve(str(config["output_dir"]), root=root)
    entries.append(_ensure_dir(output_dir, "output_dir"))
    entries.append(
        _ensure_readme(
            output_dir / "README.md",
            title="Stable-ASR Final Run Workspace",
            body=(
                "This directory is reserved for final-scale paper experiments.\n\n"
                "Do not treat placeholder directories as benchmark evidence. "
                "Run `stable-asr final-config --check-files` before launching final jobs.\n"
            ),
        )
    )

    for corpus in config.get("public_corpora", []):
        corpus_id = str(corpus["id"])
        input_path = _resolve(str(corpus.get("input_dir", corpus.get("metadata"))), root=root)
        input_parent = input_path if "input_dir" in corpus else input_path.parent
        entries.append(_ensure_dir(input_parent, f"corpus:{corpus_id}:input_parent"))
        if "input_dir" in corpus:
            body = (
                f"Place or symlink the `{corpus.get('corpus', corpus_id)}` corpus directory at "
                f"`{corpus['input_dir']}`.\n\n"
                "Then run `stable-asr prepare-public-asr` for this corpus.\n"
            )
        else:
            body = (
                f"Place the metadata table at `{corpus['metadata']}` and audio under "
                f"`{corpus['audio_root']}`.\n\n"
                "Then run `stable-asr prepare-asr-manifest` for this corpus.\n"
            )
        entries.append(
            _ensure_readme(
                input_parent / "README.md",
                title=f"Corpus Input: {corpus_id}",
                body=body + "\nThis scaffold does not create corpus files.\n",
            )
        )
        manifest_parent = _resolve(str(corpus["manifest"]), root=root).parent
        entries.append(_ensure_dir(manifest_parent, f"corpus:{corpus_id}:manifest_parent"))

    split_parent = _resolve(str(config["turn_splits"]["train"]), root=root).parent
    entries.append(_ensure_dir(split_parent, "turn_splits_parent"))
    entries.append(_ensure_dir(_resolve(str(config["asr_eval_manifest"]), root=root).parent, "asr_eval_manifest_parent"))
    entries.append(
        _ensure_readme(
            split_parent / "TURN_SPLITS_README.md",
            title="Turn Split Inputs",
            body="\n".join(f"- `{name}`: `{path}`" for name, path in config["turn_splits"].items())
            + "\n\nThese files must be real Stable-ASR turn manifests.\n",
        )
    )

    voiceworld = config.get("voiceworld_real") or {}
    if isinstance(voiceworld, dict) and voiceworld:
        metadata_path = _resolve(str(voiceworld["metadata"]), root=root)
        audio_root = _resolve(str(voiceworld["audio_root"]), root=root)
        manifest_parent = _resolve(str(voiceworld["manifest"]), root=root).parent
        entries.append(_ensure_dir(metadata_path.parent, "voiceworld_real:metadata_parent"))
        entries.append(_ensure_dir(audio_root, "voiceworld_real:audio_root"))
        entries.append(_ensure_dir(manifest_parent, "voiceworld_real:manifest_parent"))
        entries.append(
            _ensure_readme(
                metadata_path.parent / "README.md",
                title="VoiceWorld Real Inputs",
                body=(
                    f"Place real VoiceWorld annotations at `{voiceworld['metadata']}` and audio under "
                    f"`{voiceworld['audio_root']}`.\n\n"
                    "Then run `stable-asr final-config --prepare-voiceworld-real`.\n"
                    "This scaffold does not create benchmark records.\n"
                ),
            )
        )

    for prediction in config.get("external_turn_predictions", []):
        prediction_id = str(prediction["id"])
        raw_parent = _resolve(str(prediction["raw"]), root=root).parent
        converted_parent = _resolve(str(prediction["converted"]), root=root).parent
        entries.append(_ensure_dir(raw_parent, f"external_prediction:{prediction_id}:raw_parent"))
        entries.append(_ensure_dir(converted_parent, f"external_prediction:{prediction_id}:converted_parent"))
        entries.append(
            _ensure_readme(
                raw_parent / "README.md",
                title="External Turn Predictions",
                body=(
                    "Place raw external prediction exports here, then normalize them with "
                    "`stable-asr convert-predictions`.\n"
                ),
            )
        )

    entries.append(_ensure_dir(_resolve(str(config["asr_command_config"]), root=root).parent, "asr_command_config_parent"))
    for name, path in config.get("nanoturn", {}).items():
        if name != "model":
            entries.append(_ensure_dir(_resolve(str(path), root=root).parent, f"nanoturn:{name}:parent"))
    for name, path in config.get("artifacts", {}).items():
        target = _resolve(str(path), root=root)
        parent = target if target.suffix == "" else target.parent
        entries.append(_ensure_dir(parent, f"artifact:{name}:parent"))
    for name, path in config.get("result_inputs", {}).items():
        entries.append(_ensure_dir(_resolve(str(path), root=root).parent, f"result_input:{name}:parent"))

    return FinalRunScaffoldReport(output_dir=str(output_dir), entries=entries)


def prepare_final_corpora(
    config: dict[str, Any],
    *,
    repo_root: str | Path = ".",
    require_all: bool = False,
) -> FinalCorpusPrepareReport:
    """Prepare configured public ASR corpus manifests when inputs exist.

    Missing local corpora are skipped by default so users can prepare partial
    final runs without fabricating data. Set ``require_all`` to make any missing
    corpus input fail the report.
    """

    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    root = Path(repo_root)
    entries: list[FinalCorpusPrepareEntry] = []
    for corpus in config.get("public_corpora", []):
        corpus_id = str(corpus["id"])
        manifest_path = _resolve(str(corpus["manifest"]), root=root)
        try:
            if "input_dir" in corpus:
                input_path = _resolve(str(corpus["input_dir"]), root=root)
                source_name = str(corpus["corpus"])
                if not input_path.exists():
                    entries.append(
                        _skipped_corpus_entry(
                            corpus_id,
                            corpus=source_name,
                            input_path=input_path,
                            manifest_path=manifest_path,
                            detail="missing input directory",
                            require_all=require_all,
                        )
                    )
                    continue
                records = prepare_public_asr_manifest(
                    corpus=source_name,
                    input_dir=input_path,
                    output_path=manifest_path,
                    split=corpus.get("split"),
                    sample_rate=int(corpus["sample_rate"]),
                )
            else:
                metadata_path = _resolve(str(corpus["metadata"]), root=root)
                audio_root = _resolve(str(corpus["audio_root"]), root=root)
                source_name = "metadata_table"
                if not metadata_path.exists() or not audio_root.exists():
                    missing = []
                    if not metadata_path.exists():
                        missing.append("metadata")
                    if not audio_root.exists():
                        missing.append("audio_root")
                    entries.append(
                        _skipped_corpus_entry(
                            corpus_id,
                            corpus=source_name,
                            input_path=metadata_path,
                            manifest_path=manifest_path,
                            detail="missing " + " and ".join(missing),
                            require_all=require_all,
                        )
                    )
                    continue
                records = prepare_asr_manifest(
                    metadata_path,
                    manifest_path,
                    audio_root=audio_root,
                    default_sample_rate=int(corpus["sample_rate"]),
                    default_language=str(corpus["language"]),
                    default_source=corpus_id,
                    default_split=corpus.get("split"),
                )
            entries.append(
                FinalCorpusPrepareEntry(
                    id=corpus_id,
                    corpus=source_name,
                    input=str(input_path if "input_dir" in corpus else metadata_path),
                    manifest=str(manifest_path),
                    records=len(records),
                    ok=True,
                    skipped=False,
                    detail="manifest written",
                )
            )
        except (OSError, ValueError) as exc:
            entries.append(
                FinalCorpusPrepareEntry(
                    id=corpus_id,
                    corpus=str(corpus.get("corpus", "metadata_table")),
                    input=str(corpus.get("input_dir", corpus.get("metadata", ""))),
                    manifest=str(manifest_path),
                    records=0,
                    ok=False,
                    skipped=False,
                    detail=str(exc),
                )
            )
    ok = all(entry.ok for entry in entries) and (not require_all or not any(entry.skipped for entry in entries))
    return FinalCorpusPrepareReport(ok=ok, require_all=require_all, entries=entries)


def bootstrap_final_turn_splits(
    config: dict[str, Any],
    *,
    repo_root: str | Path = ".",
    include_incomplete: bool = True,
    seed: int | None = None,
) -> FinalTurnBootstrapReport:
    """Bootstrap weak final train/dev/test turn splits from prepared ASR manifests."""

    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    root = Path(repo_root)
    records = []
    input_manifests: list[str] = []
    skipped_manifests: list[str] = []
    for corpus in config.get("public_corpora", []):
        manifest_path = _resolve(str(corpus["manifest"]), root=root)
        if not manifest_path.exists():
            skipped_manifests.append(str(manifest_path))
            continue
        corpus_records = load_asr_manifest(manifest_path)
        if corpus_records:
            input_manifests.append(str(manifest_path))
            records.extend(corpus_records)

    if not records:
        return FinalTurnBootstrapReport(
            ok=False,
            input_manifests=input_manifests,
            skipped_manifests=skipped_manifests,
            asr_records=0,
            turn_records=0,
            split_paths={name: str(_resolve(str(config["turn_splits"][name]), root=root)) for name in SPLIT_NAMES},
            split_counts={name: 0 for name in SPLIT_NAMES},
            detail="no prepared ASR manifests found",
        )

    turn_result = asr_records_to_turn_records(
        records,
        config=ASRToTurnConfig(include_incomplete=include_incomplete, source="final_asr_weak_turn_v0"),
    )
    split_result = split_turn_records(
        turn_result.records,
        config=TurnSplitConfig(seed=int(config.get("seed", 0) if seed is None else seed), group_by="metadata.asr_record_id"),
    )
    split_paths = {
        name: _resolve(str(config["turn_splits"][name]), root=root)
        for name in SPLIT_NAMES
    }
    split_counts: dict[str, int] = {}
    for name, path in split_paths.items():
        split_records = split_result.split(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_turn_records(path, split_records, format="jsonl")
        split_counts[name] = len(split_records)

    summary_path = _resolve(str(config["output_dir"]), root=root) / "final_turn_bootstrap_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "asr_summary": summarize_asr_records(records),
        "turn_summary": summarize_records(turn_result.records),
        "splits": split_result.to_dict()["splits"],
        "input_manifests": input_manifests,
        "skipped_manifests": skipped_manifests,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return FinalTurnBootstrapReport(
        ok=True,
        input_manifests=input_manifests,
        skipped_manifests=skipped_manifests,
        asr_records=len(records),
        turn_records=len(turn_result.records),
        split_paths={name: str(path) for name, path in split_paths.items()},
        split_counts=split_counts,
        detail=f"summary written to {summary_path}; voiceworld_real remains a required real scenario input",
    )


def prepare_final_asr_eval_manifest(
    config: dict[str, Any],
    *,
    repo_root: str | Path = ".",
) -> FinalASREvalManifestReport:
    """Combine prepared public ASR manifests into the shared final ASR eval manifest."""

    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    root = Path(repo_root)
    output_path = _resolve(str(config["asr_eval_manifest"]), root=root)
    records = []
    input_manifests: list[str] = []
    skipped_manifests: list[str] = []
    for corpus in config.get("public_corpora", []):
        manifest_path = _resolve(str(corpus["manifest"]), root=root)
        if not manifest_path.exists():
            skipped_manifests.append(str(manifest_path))
            continue
        corpus_records = load_asr_manifest(manifest_path)
        if corpus_records:
            input_manifests.append(str(manifest_path))
            records.extend(corpus_records)

    if not records:
        return FinalASREvalManifestReport(
            ok=False,
            output=str(output_path),
            input_manifests=input_manifests,
            skipped_manifests=skipped_manifests,
            records=0,
            detail="no prepared ASR manifests found",
        )

    write_asr_manifest(output_path, records)
    summary_path = _resolve(str(config["output_dir"]), root=root) / "final_asr_eval_manifest_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "output": str(output_path),
        "input_manifests": input_manifests,
        "skipped_manifests": skipped_manifests,
        "summary": summarize_asr_records(records),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return FinalASREvalManifestReport(
        ok=True,
        output=str(output_path),
        input_manifests=input_manifests,
        skipped_manifests=skipped_manifests,
        records=len(records),
        detail=f"manifest and summary written; summary={summary_path}",
    )


def prepare_final_external_predictions(
    config: dict[str, Any],
    *,
    repo_root: str | Path = ".",
    require_all: bool = False,
    allow_extra: bool = False,
) -> FinalExternalPredictionReport:
    """Normalize configured external turn prediction exports and validate coverage when possible."""

    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    root = Path(repo_root)
    dataset_path = _resolve(str(config["turn_splits"]["test"]), root=root)
    dataset_records = load_turn_records(dataset_path, format="jsonl") if dataset_path.exists() else []
    entries: list[FinalExternalPredictionEntry] = []

    for prediction in config.get("external_turn_predictions", []):
        prediction_id = str(prediction["id"])
        schema = str(prediction["schema"])
        raw_path = _resolve(str(prediction["raw"]), root=root)
        converted_path = _resolve(str(prediction["converted"]), root=root)
        if not raw_path.exists():
            entries.append(
                FinalExternalPredictionEntry(
                    id=prediction_id,
                    schema=schema,
                    raw=str(raw_path),
                    converted=str(converted_path),
                    converted_records=0,
                    ok=not require_all,
                    skipped=True,
                    coverage_checked=False,
                    missing_ids=0,
                    extra_ids=0,
                    detail="missing raw prediction export",
                )
            )
            continue
        try:
            converted_count = convert_turn_prediction_jsonl(raw_path, converted_path, schema=schema)
            coverage_checked = False
            missing_ids = 0
            extra_ids = 0
            detail = "converted"
            ok = True
            if dataset_records:
                coverage = validate_turn_prediction_jsonl(
                    dataset_records,
                    converted_path,
                    allow_extra=allow_extra,
                    dataset_path=dataset_path,
                )
                coverage_checked = True
                missing_ids = len(coverage.missing_ids)
                extra_ids = len(coverage.extra_ids)
                ok = coverage.ok
                detail = "converted and coverage checked" if coverage.ok else "coverage validation failed"
            else:
                detail = "converted; coverage skipped because turn_test is missing"
            entries.append(
                FinalExternalPredictionEntry(
                    id=prediction_id,
                    schema=schema,
                    raw=str(raw_path),
                    converted=str(converted_path),
                    converted_records=converted_count,
                    ok=ok,
                    skipped=False,
                    coverage_checked=coverage_checked,
                    missing_ids=missing_ids,
                    extra_ids=extra_ids,
                    detail=detail,
                )
            )
        except (OSError, ValueError) as exc:
            entries.append(
                FinalExternalPredictionEntry(
                    id=prediction_id,
                    schema=schema,
                    raw=str(raw_path),
                    converted=str(converted_path),
                    converted_records=0,
                    ok=False,
                    skipped=False,
                    coverage_checked=False,
                    missing_ids=0,
                    extra_ids=0,
                    detail=str(exc),
                )
            )

    ok = all(entry.ok for entry in entries) and (not require_all or not any(entry.skipped for entry in entries))
    return FinalExternalPredictionReport(
        ok=ok,
        dataset_path=str(dataset_path),
        dataset_records=len(dataset_records),
        require_all=require_all,
        entries=entries,
    )


def prepare_final_voiceworld_real(
    config: dict[str, Any],
    *,
    repo_root: str | Path = ".",
    scenario_suite_path: str | Path | None = None,
    min_per_scenario: int = 1,
    require_input: bool = False,
) -> FinalVoiceWorldPrepareReport:
    """Prepare the configured real VoiceWorld manifest when metadata exists."""

    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    root = Path(repo_root)
    voiceworld = config.get("voiceworld_real") or {}
    if not isinstance(voiceworld, dict) or not voiceworld:
        manifest = _resolve(str(config["turn_splits"]["voiceworld_real"]), root=root)
        return FinalVoiceWorldPrepareReport(
            ok=not require_input,
            metadata="",
            audio_root="",
            manifest=str(manifest),
            records=0,
            skipped=True,
            audit=None,
            detail="voiceworld_real recipe is not configured",
        )

    metadata_path = _resolve(str(voiceworld["metadata"]), root=root)
    audio_root = _resolve(str(voiceworld["audio_root"]), root=root)
    manifest_path = _resolve(str(voiceworld["manifest"]), root=root)
    if not metadata_path.exists() or not audio_root.exists():
        missing = []
        if not metadata_path.exists():
            missing.append("metadata")
        if not audio_root.exists():
            missing.append("audio_root")
        return FinalVoiceWorldPrepareReport(
            ok=not require_input and not bool(voiceworld.get("required", True)),
            metadata=str(metadata_path),
            audio_root=str(audio_root),
            manifest=str(manifest_path),
            records=0,
            skipped=True,
            audit=None,
            detail="missing " + " and ".join(missing),
        )

    records = prepare_voiceworld_manifest(
        metadata_path,
        manifest_path,
        audio_root=audio_root,
        default_sample_rate=int(voiceworld["sample_rate"]),
        default_language=str(voiceworld["language"]),
        default_source=str(voiceworld["source"]),
    )
    audit = audit_final_voiceworld_real(
        config,
        repo_root=repo_root,
        scenario_suite_path=scenario_suite_path,
        min_per_scenario=min_per_scenario,
    )
    return FinalVoiceWorldPrepareReport(
        ok=audit.ok,
        metadata=str(metadata_path),
        audio_root=str(audio_root),
        manifest=str(manifest_path),
        records=len(records),
        skipped=False,
        audit=audit,
        detail="manifest written and audited" if audit.ok else "manifest written but audit failed",
    )


def prepare_final_asr_transcript_conversions(
    config: dict[str, Any],
    *,
    repo_root: str | Path = ".",
) -> FinalASRTranscriptConversionReport:
    """Compare final normalized ASR transcript outputs and write the result input JSON."""

    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    root = Path(repo_root)
    command_config_path = _resolve(str(config["asr_command_config"]), root=root)
    output_value = config.get("result_inputs", {}).get(
        "asr_transcript_conversions",
        "runs/final/reports/asr_transcript_conversions.json",
    )
    output_path = _resolve(
        str(output_value),
        root=root,
    )
    try:
        command_config = load_asr_command_config(command_config_path)
    except (OSError, ValueError) as exc:
        return FinalASRTranscriptConversionReport(
            ok=False,
            output=str(output_path),
            input_paths={},
            missing_inputs={},
            records_by_adapter={},
            detail=str(exc),
        )

    inputs: dict[str, str] = {}
    missing: dict[str, str] = {}
    for adapter in command_config.get("adapters", []):
        if not isinstance(adapter, dict):
            continue
        name = str(adapter.get("name", "")).strip()
        output = adapter.get("output", adapter.get("output_path"))
        if not name or not isinstance(output, str) or not output:
            continue
        schema = _asr_transcript_schema_name(name)
        resolved = _resolve(output, root=root)
        inputs[schema] = str(resolved)
        if not resolved.exists():
            missing[schema] = str(resolved)

    if not inputs:
        return FinalASRTranscriptConversionReport(
            ok=False,
            output=str(output_path),
            input_paths={},
            missing_inputs={},
            records_by_adapter={},
            detail="ASR command config has no adapter outputs",
        )
    if missing:
        return FinalASRTranscriptConversionReport(
            ok=False,
            output=str(output_path),
            input_paths=inputs,
            missing_inputs=missing,
            records_by_adapter={},
            detail="normalized ASR transcript output(s) missing",
        )

    report = compare_streaming_transcript_jsonl(list(inputs.items()))
    payload = report.to_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records_by_adapter = {
        str(row.get("adapter", "unknown")): int(row.get("records", 0))
        for row in payload.get("rows", [])
        if isinstance(row, dict)
    }
    return FinalASRTranscriptConversionReport(
        ok=True,
        output=str(output_path),
        input_paths=inputs,
        missing_inputs={},
        records_by_adapter=records_by_adapter,
        detail="comparison result written",
    )


def prepare_final_inputs(
    config: dict[str, Any],
    *,
    repo_root: str | Path = ".",
    scenario_suite_path: str | Path | None = None,
    require_all_corpora: bool = False,
    require_all_predictions: bool = False,
    allow_extra_predictions: bool = False,
    include_incomplete: bool = True,
    min_per_scenario: int = 1,
) -> FinalInputPrepareReport:
    """Run the final input preparation sequence and audit remaining required inputs."""

    corpus_report = prepare_final_corpora(
        config,
        repo_root=repo_root,
        require_all=require_all_corpora,
    )
    asr_eval_report = prepare_final_asr_eval_manifest(
        config,
        repo_root=repo_root,
    )
    turn_report = bootstrap_final_turn_splits(
        config,
        repo_root=repo_root,
        include_incomplete=include_incomplete,
    )
    prediction_report = prepare_final_external_predictions(
        config,
        repo_root=repo_root,
        require_all=require_all_predictions,
        allow_extra=allow_extra_predictions,
    )
    voiceworld_prepare_report = prepare_final_voiceworld_real(
        config,
        repo_root=repo_root,
        scenario_suite_path=scenario_suite_path,
        min_per_scenario=min_per_scenario,
        require_input=True,
    )
    voiceworld_report = audit_final_voiceworld_real(
        config,
        repo_root=repo_root,
        scenario_suite_path=scenario_suite_path,
        min_per_scenario=min_per_scenario,
    )
    asr_command_report = audit_asr_command_config(
        _resolve(str(config["asr_command_config"]), root=Path(repo_root)),
        repo_root=repo_root,
        min_adapters=4,
        require_input_manifest=True,
    )
    file_audit = audit_final_run_files(config, repo_root=repo_root)
    ok = (
        corpus_report.ok
        and asr_eval_report.ok
        and turn_report.ok
        and prediction_report.ok
        and voiceworld_prepare_report.ok
        and voiceworld_report.ok
        and asr_command_report.ok
        and file_audit.ok
    )
    return FinalInputPrepareReport(
        ok=ok,
        corpora=corpus_report,
        asr_eval_manifest=asr_eval_report,
        turn_splits=turn_report,
        external_predictions=prediction_report,
        voiceworld_prepare=voiceworld_prepare_report,
        voiceworld_real=voiceworld_report,
        asr_command_config=asr_command_report,
        file_audit=file_audit,
    )


def audit_final_voiceworld_real(
    config: dict[str, Any],
    *,
    repo_root: str | Path = ".",
    scenario_suite_path: str | Path | None = None,
    min_per_scenario: int = 1,
) -> FinalVoiceWorldAuditReport:
    """Audit the real VoiceWorld final manifest against the configured scenario suite."""

    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    if min_per_scenario <= 0:
        raise ValueError("min_per_scenario must be positive")

    root = Path(repo_root)
    manifest_path = _resolve(str(config["turn_splits"]["voiceworld_real"]), root=root)
    suite = load_scenario_suite(scenario_suite_path)
    suite_validation = validate_scenario_suite(suite)
    if not suite_validation.ok:
        raise ValueError("; ".join(suite_validation.errors))
    required_scenarios = [str(item["id"]) for item in suite["scenarios"]]
    factor_names = [str(item["name"]) for item in suite.get("factors", []) if isinstance(item, dict) and "name" in item]

    errors: list[str] = []
    if not manifest_path.exists():
        return FinalVoiceWorldAuditReport(
            ok=False,
            manifest=str(manifest_path),
            records=0,
            min_per_scenario=min_per_scenario,
            scenario_counts={scenario: 0 for scenario in required_scenarios},
            missing_scenarios=required_scenarios,
            undercovered_scenarios={},
            factor_coverage={factor: 0 for factor in factor_names},
            missing_factor_fields=factor_names,
            errors=["voiceworld_real manifest is missing"],
        )

    try:
        records = load_turn_records(manifest_path, format="jsonl")
    except (OSError, ValueError) as exc:
        return FinalVoiceWorldAuditReport(
            ok=False,
            manifest=str(manifest_path),
            records=0,
            min_per_scenario=min_per_scenario,
            scenario_counts={scenario: 0 for scenario in required_scenarios},
            missing_scenarios=required_scenarios,
            undercovered_scenarios={},
            factor_coverage={factor: 0 for factor in factor_names},
            missing_factor_fields=factor_names,
            errors=[str(exc)],
        )

    scenario_counts = {scenario: 0 for scenario in required_scenarios}
    factor_coverage = {factor: 0 for factor in factor_names}
    unknown_scenarios: dict[str, int] = {}
    for record in records:
        scenario = record.scenario or str(record.metadata.get("scenario", ""))
        if scenario in scenario_counts:
            scenario_counts[scenario] += 1
        elif scenario:
            unknown_scenarios[scenario] = unknown_scenarios.get(scenario, 0) + 1
        for factor in factor_names:
            if factor in record.metadata and record.metadata[factor] not in (None, ""):
                factor_coverage[factor] += 1

    missing_scenarios = [scenario for scenario, count in scenario_counts.items() if count == 0]
    undercovered_scenarios = {
        scenario: count
        for scenario, count in scenario_counts.items()
        if 0 < count < min_per_scenario
    }
    missing_factor_fields = [factor for factor, count in factor_coverage.items() if count == 0]
    if unknown_scenarios:
        errors.append(
            "unknown scenario id(s): "
            + ", ".join(f"{scenario}={count}" for scenario, count in sorted(unknown_scenarios.items()))
        )
    ok = not errors and not missing_scenarios and not undercovered_scenarios and not missing_factor_fields
    return FinalVoiceWorldAuditReport(
        ok=ok,
        manifest=str(manifest_path),
        records=len(records),
        min_per_scenario=min_per_scenario,
        scenario_counts=scenario_counts,
        missing_scenarios=missing_scenarios,
        undercovered_scenarios=undercovered_scenarios,
        factor_coverage=factor_coverage,
        missing_factor_fields=missing_factor_fields,
        errors=errors,
    )


def _stage_corpora_action(
    config: dict[str, Any],
    missing_required: list[FinalRunPathCheck],
    config_path: str,
) -> FinalRunActionItem:
    blockers = _missing_paths(missing_required, "corpus:")
    commands = _corpus_recipe_commands(config)
    commands.append(f"stable-asr final-config --config {config_path} --prepare-corpora --require-all-corpora")
    artifacts = [str(corpus["manifest"]) for corpus in config.get("public_corpora", [])]
    return FinalRunActionItem(
        id="stage_public_corpora",
        title="Stage public ASR corpora and write canonical ASR manifests",
        status=_action_status(blockers),
        blockers=blockers,
        commands=commands,
        artifacts=artifacts,
        detail="Place or symlink real upstream corpus files locally, then normalize them into Stable-ASR ASR manifests.",
    )


def _prepare_asr_turn_action(
    config: dict[str, Any],
    missing_required: list[FinalRunPathCheck],
    config_path: str,
) -> FinalRunActionItem:
    blockers = _missing_paths(missing_required, "turn_split:train", "turn_split:dev", "turn_split:test")
    artifacts = [str(config["asr_eval_manifest"])]
    artifacts.extend(str(config["turn_splits"][name]) for name in SPLIT_NAMES)
    return FinalRunActionItem(
        id="prepare_asr_eval_and_turn_splits",
        title="Assemble shared ASR evaluation manifest and weak turn splits",
        status="blocked" if _missing_paths(missing_required, "corpus:") else _action_status(blockers),
        blockers=blockers,
        commands=[
            f"stable-asr final-config --config {config_path} --prepare-asr-eval-manifest",
            f"stable-asr final-config --config {config_path} --bootstrap-turn-splits",
            "stable-asr audit-turn-splits --train runs/final/turn_train.jsonl --dev runs/final/turn_dev.jsonl --test runs/final/turn_test.jsonl",
        ],
        artifacts=artifacts,
        detail="Build the shared ASR eval set first, then derive leakage-audited weak turn train/dev/test splits.",
    )


def _voiceworld_action(
    config: dict[str, Any],
    missing_required: list[FinalRunPathCheck],
    config_path: str,
) -> FinalRunActionItem:
    voiceworld_path = str(config["turn_splits"]["voiceworld_real"])
    voiceworld = config.get("voiceworld_real") or {}
    blockers = _missing_paths(missing_required, "turn_split:voiceworld_real", "voiceworld_real:")
    if isinstance(voiceworld, dict) and voiceworld:
        prepare_command = f"stable-asr final-config --config {config_path} --prepare-voiceworld-real"
        metadata = str(voiceworld.get("metadata", "data/voiceworld/metadata.tsv"))
        audio_root = str(voiceworld.get("audio_root", "data/voiceworld/audio"))
    else:
        prepare_command = f"stable-asr prepare-voiceworld --input data/voiceworld/metadata.tsv --audio-root data/voiceworld/audio --output {voiceworld_path}"
        metadata = "data/voiceworld/metadata.tsv"
        audio_root = "data/voiceworld/audio"
    return FinalRunActionItem(
        id="collect_voiceworld_real",
        title="Collect or compose the real VoiceWorld scenario manifest",
        status=_action_status(blockers),
        blockers=blockers,
        commands=[
            f"# stage annotations at {metadata} and audio under {audio_root}",
            prepare_command,
            f"stable-asr validate-manifest {voiceworld_path}",
            (
                f"stable-asr final-config --config {config_path} --audit-voiceworld-real "
                "--scenario-suite configs/scenarios/stable_asr_voiceworld_v0.json --min-scenario-records 20"
            ),
            f"stable-asr eval-scenario --dataset {voiceworld_path} --checkpoint {config['nanoturn']['checkpoint']} --json-output {config['result_inputs']['scenarios']}",
        ],
        artifacts=[voiceworld_path, str(config["result_inputs"]["scenarios"])],
        detail="Use real or explicitly composed audio examples for every required VoiceWorld scenario and factor.",
    )


def _external_predictions_action(
    config: dict[str, Any],
    missing_required: list[FinalRunPathCheck],
    config_path: str,
) -> FinalRunActionItem:
    blockers = _missing_paths(missing_required, "external_prediction:")
    commands = [
        f"stable-asr convert-predictions --schema {prediction['schema']} --input {prediction['raw']} --output {prediction['converted']}"
        for prediction in config.get("external_turn_predictions", [])
    ]
    commands.append(f"stable-asr final-config --config {config_path} --prepare-external-predictions --require-all-predictions")
    artifacts = [str(prediction["converted"]) for prediction in config.get("external_turn_predictions", [])]
    artifacts.append(str(config["result_inputs"]["baselines"]))
    return FinalRunActionItem(
        id="normalize_external_turn_predictions",
        title="Export and normalize external turn-system predictions",
        status=_action_status(blockers),
        blockers=blockers,
        commands=commands,
        artifacts=artifacts,
        detail="Run SmartTurn/EasyTurn/VAP-style systems outside Stable-ASR, then normalize and coverage-check their prediction manifests.",
    )


def _nanoturn_action(
    config: dict[str, Any],
    missing_required: list[FinalRunPathCheck],
    config_path: str,
) -> FinalRunActionItem:
    blockers = _missing_paths(missing_required, "turn_split:train")
    nanoturn = config["nanoturn"]
    return FinalRunActionItem(
        id="train_and_export_nanoturn",
        title="Train NanoTurn and produce latency/export artifacts",
        status=_action_status(blockers),
        blockers=blockers,
        commands=[
            (
                f"stable-asr train-turn --dataset {config['turn_splits']['train']} --output-dir "
                f"{Path(str(nanoturn['checkpoint'])).parent} --model {nanoturn['model']} --feature-source audio"
            ),
            f"stable-asr export-turn-onnx --checkpoint {nanoturn['checkpoint']} --output {nanoturn['onnx']}",
            f"stable-asr benchmark-turn --dataset {config['turn_splits']['test']} --checkpoint {nanoturn['checkpoint']} --json-output {config['result_inputs']['turn_benchmarks']}",
        ],
        artifacts=[str(nanoturn["checkpoint"]), str(nanoturn["metrics"]), str(nanoturn["onnx"]), str(config["result_inputs"]["turn_benchmarks"])],
        detail="Create the default trainable baseline and the deployment-facing artifacts needed by the platform paper.",
    )


def _streaming_asr_action(
    config: dict[str, Any],
    missing_required: list[FinalRunPathCheck],
    config_path: str,
) -> FinalRunActionItem:
    blockers = _missing_paths(missing_required, "asr_command_config")
    if any(check.name.startswith("corpus:") for check in missing_required):
        blockers.extend(_missing_paths(missing_required, "corpus:"))
    return FinalRunActionItem(
        id="run_command_backed_streaming_asr",
        title="Run command-backed ASR adapters under one streaming schema",
        status=_action_status(blockers),
        blockers=blockers,
        commands=[
            f"stable-asr final-config --config {config_path} --audit-asr-commands",
            f"stable-asr compare-asr-commands --config {config['asr_command_config']} --report runs/final/reports/asr_command_compare.md --json-output {config['result_inputs']['streaming_comparison']}",
            f"stable-asr sweep-streaming-asr --input runs/final/asr_commands/whisper_streaming.jsonl --chunks-ms 160 320 640 --lookahead-ms 0 160 320 --json-output {config['result_inputs']['streaming_sweep']}",
            f"stable-asr final-config --config {config_path} --prepare-asr-transcript-conversions",
        ],
        artifacts=[
            str(config["result_inputs"]["streaming_comparison"]),
            str(config["result_inputs"]["streaming_sweep"]),
            str(config["result_inputs"]["asr_transcript_conversions"]),
        ],
        detail=(
            "Evaluate Whisper, FunASR, Qwen3-ASR, FireRedASR2S, and other real "
            "ASR systems through command adapters without vendoring heavyweight "
            "upstream toolkits."
        ),
    )


def _final_artifacts_action(
    config: dict[str, Any],
    missing_required: list[FinalRunPathCheck],
    config_path: str,
) -> FinalRunActionItem:
    blockers = [check.path for check in missing_required]
    artifacts = [
        str(config["artifacts"]["paper_results"]),
        str(config["artifacts"]["bundle_dir"]),
        str(config["artifacts"].get("artifact_archive", "runs/final/artifacts.tar.gz")),
        str(config["artifacts"].get("model_card", "runs/final/MODEL_CARD.md")),
        str(config["artifacts"].get("assignment_audit", "runs/final/FINAL_ASSIGNMENT_AUDIT.md")),
        str(config["artifacts"].get("handoff", "runs/final/FINAL_INPUT_HANDOFF.json")),
        str(config["artifacts"].get("handoff_schema_validation", "runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md")),
        str(config["artifacts"].get("handoff_audit", "runs/final/FINAL_HANDOFF_AUDIT.md")),
    ]
    return FinalRunActionItem(
        id="assemble_final_artifacts",
        title="Assemble paper results, bundle artifacts, and run final parity gates",
        status=_action_status(blockers),
        blockers=blockers,
        commands=[
            f"stable-asr final-results --config {config_path} --output {config['artifacts']['paper_results']}",
            f"stable-asr final-assignment-audit --input runs/final_acquisition_pack/acquisition/assignments.json --require-owner --require-due-date --require-ready --output {config['artifacts'].get('assignment_audit', 'runs/final/FINAL_ASSIGNMENT_AUDIT.md')}",
            f"stable-asr final-handoff-checksums --input {config['artifacts'].get('handoff', 'runs/final/FINAL_INPUT_HANDOFF.json')} --repo-root . --output {config['artifacts'].get('handoff', 'runs/final/FINAL_INPUT_HANDOFF.json')}",
            f"stable-asr validate-schema-file --input {config['artifacts'].get('handoff', 'runs/final/FINAL_INPUT_HANDOFF.json')} --schema-id stable_asr.final_handoff.v0 --output {config['artifacts'].get('handoff_schema_validation', 'runs/final/FINAL_HANDOFF_SCHEMA_VALIDATION.md')}",
            f"stable-asr final-handoff-audit --input {config['artifacts'].get('handoff', 'runs/final/FINAL_INPUT_HANDOFF.json')} --repo-root . --require-checksums --output {config['artifacts'].get('handoff_audit', 'runs/final/FINAL_HANDOFF_AUDIT.md')}",
            f"stable-asr paper-bundle --results {config['artifacts']['paper_results']} --output-dir {config['artifacts']['bundle_dir']}",
            f"stable-asr make-card model --input configs/models/stable_asr_models.json --model-id {config.get('nanoturn', {}).get('model', 'nanoturn_pico')} --metrics {config.get('nanoturn', {}).get('metrics', 'runs/final/nanoturn/metrics.json')} --output {config['artifacts'].get('model_card', 'runs/final/MODEL_CARD.md')}",
            f"stable-asr paper-artifact-integrity --manifest {config['artifacts']['bundle_dir']}/artifact_hashes.json --root {config['artifacts']['bundle_dir']}",
            f"stable-asr paper-archive --artifacts-dir {config['artifacts']['bundle_dir']} --output {config['artifacts'].get('artifact_archive', 'runs/final/artifacts.tar.gz')}",
            f"stable-asr paper-archive-verify --archive {config['artifacts'].get('artifact_archive', 'runs/final/artifacts.tar.gz')}",
            f"stable-asr paper-parity-audit --results {config['artifacts']['paper_results']} --artifacts-dir {config['artifacts']['bundle_dir']} --require-final",
            f"stable-asr paper-release-audit --repo-root . --results {config['artifacts']['paper_results']} --artifacts-dir {config['artifacts']['bundle_dir']} --model-card {config['artifacts'].get('model_card', 'runs/final/MODEL_CARD.md')} --require-final-ready",
        ],
        artifacts=artifacts,
        detail="Only run this as a final gate after real corpora, external predictions, VoiceWorld records, and ASR outputs exist.",
    )


def _corpus_recipe_commands(config: dict[str, Any]) -> list[str]:
    commands = []
    for corpus in config.get("public_corpora", []):
        if "input_dir" in corpus:
            command = (
                f"stable-asr prepare-public-asr --corpus {corpus['corpus']} --input-dir {corpus['input_dir']} "
                f"--output {corpus['manifest']}"
            )
            if corpus.get("split"):
                command += f" --split {corpus['split']}"
        else:
            command = (
                f"stable-asr prepare-asr-manifest --input {corpus['metadata']} --audio-root {corpus['audio_root']} "
                f"--output {corpus['manifest']}"
            )
        commands.append(command)
    return commands


def _missing_paths(missing_required: list[FinalRunPathCheck], *prefixes: str) -> list[str]:
    return [
        check.path
        for check in missing_required
        if any(check.name.startswith(prefix) for prefix in prefixes)
    ]


def _action_status(blockers: list[str]) -> str:
    return "blocked" if blockers else "ready"


def _suggest_action_for_missing_check(check: FinalRunPathCheck) -> str:
    if check.name.startswith("corpus:") and check.name.endswith(":input_dir"):
        return "place or symlink the upstream corpus directory, then run final-config --prepare-corpora"
    if check.name.startswith("corpus:") and (check.name.endswith(":metadata") or check.name.endswith(":audio_root")):
        return "stage the metadata table and audio root, then run final-config --prepare-corpora"
    if check.name in {"turn_split:train", "turn_split:dev", "turn_split:test"}:
        return "run final-config --prepare-asr-eval-manifest and --bootstrap-turn-splits after corpus manifests exist"
    if check.name == "turn_split:voiceworld_real":
        return "collect or compose real VoiceWorld scenario records, then run final-config --audit-voiceworld-real"
    if check.name == "voiceworld_real:metadata":
        return "place the real VoiceWorld annotation table, then run final-config --prepare-voiceworld-real"
    if check.name == "voiceworld_real:audio_root":
        return "place the real VoiceWorld audio directory, then run final-config --prepare-voiceworld-real"
    if check.name.startswith("external_prediction:") and check.name.endswith(":raw"):
        return "run the external turn model and save its raw prediction export, then run final-config --prepare-external-predictions"
    if check.name == "asr_command_config":
        return "restore or write the command-backed ASR comparison config"
    return "stage the required file, then rerun final-config --check-files"


def _corpus_rows(config: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for corpus in config["public_corpora"]:
        rows.append(
            {
                "id": corpus["id"],
                "corpus": corpus.get("corpus", "metadata_table"),
                "language": corpus["language"],
                "input": corpus.get("input_dir", corpus.get("metadata", "")),
                "manifest": corpus["manifest"],
                "sample_rate": corpus["sample_rate"],
                "required": corpus.get("required", True),
            }
        )
    return rows


def _prediction_rows(config: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for prediction in config.get("external_turn_predictions", []):
        rows.append(
            {
                "id": prediction["id"],
                "schema": prediction["schema"],
                "raw": prediction["raw"],
                "converted": prediction["converted"],
            }
        )
    return rows


def _input_check(name: str, path: str, *, root: Path, kind: str = "input", required: bool = True) -> FinalRunPathCheck:
    resolved = _resolve(path, root=root)
    exists = resolved.exists()
    ok = exists or not required
    if exists:
        detail = "exists"
    elif required:
        detail = "missing required input"
    else:
        detail = "optional input missing"
    return FinalRunPathCheck(
        name=name,
        path=str(path),
        kind=kind,
        required=required,
        exists=exists,
        ok=ok,
        detail=detail,
    )


def _planned_check(name: str, path: str, *, root: Path, kind: str) -> FinalRunPathCheck:
    resolved = _resolve(path, root=root)
    parent = resolved if resolved.suffix == "" else resolved.parent
    return FinalRunPathCheck(
        name=name,
        path=str(path),
        kind=kind,
        required=False,
        exists=resolved.exists(),
        ok=bool(str(path).strip()),
        detail=f"planned output; parent={parent}",
    )


def _resolve(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _asr_transcript_schema_name(adapter_name: str) -> str:
    name = adapter_name.strip()
    for suffix in ("_final", "_streaming", "_adapter"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _ensure_dir(path: Path, kind: str) -> FinalRunScaffoldEntry:
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    return FinalRunScaffoldEntry(
        path=str(path),
        kind=kind,
        created=not existed,
        detail="directory",
    )


def _ensure_readme(path: Path, *, title: str, body: str) -> FinalRunScaffoldEntry:
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not existed:
        path.write_text(f"# {title}\n\n{body}", encoding="utf-8")
    return FinalRunScaffoldEntry(
        path=str(path),
        kind="readme",
        created=not existed,
        detail="placeholder instructions; no data generated",
    )


def _skipped_corpus_entry(
    corpus_id: str,
    *,
    corpus: str,
    input_path: Path,
    manifest_path: Path,
    detail: str,
    require_all: bool,
) -> FinalCorpusPrepareEntry:
    return FinalCorpusPrepareEntry(
        id=corpus_id,
        corpus=corpus,
        input=str(input_path),
        manifest=str(manifest_path),
        records=0,
        ok=not require_all,
        skipped=True,
        detail=detail,
    )
