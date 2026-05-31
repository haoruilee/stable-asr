"""Final paper run configuration schema and renderer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.asr_manifest import load_asr_manifest, summarize_asr_records
from stable_asr.data.recipes import prepare_asr_manifest, prepare_public_asr_manifest
from stable_asr.data.registry import load_turn_records, summarize_records, write_turn_records
from stable_asr.data.split import SPLIT_NAMES, TurnSplitConfig, split_turn_records
from stable_asr.data.turn_from_asr import ASRToTurnConfig, asr_records_to_turn_records
from stable_asr.eval.report import dict_table
from stable_asr.models.adapters import convert_turn_prediction_jsonl, validate_turn_prediction_jsonl
from stable_asr.resources import resolve_platform_path
from stable_asr.scenarios.suites import load_scenario_suite, validate_scenario_suite


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
    "turn_splits": {
        "train": "runs/final/turn_train.jsonl",
        "dev": "runs/final/turn_dev.jsonl",
        "test": "runs/final/turn_test.jsonl",
        "voiceworld_real": "runs/final/voiceworld_real.jsonl",
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
        "markdown_draft": "runs/final/PAPER_DRAFT.md",
        "latex_draft": "runs/final/paper.tex",
        "dataset_card": "runs/final/DATASET_CARD.md",
        "experiment_card": "runs/final/EXPERIMENT_CARD.md",
    },
    "commands": [
        "stable-asr final-config --config configs/final/paper_final.json --validate-only",
        "stable-asr final-config --config configs/final/paper_final.json --prepare-inputs",
        "stable-asr final-config --config configs/final/paper_final.json --prepare-corpora",
        "stable-asr prepare-public-asr --corpus librispeech --input-dir data/librispeech/LibriSpeech/dev-clean --output runs/final/librispeech_dev_clean/asr_manifest.jsonl",
        "stable-asr prepare-public-asr --corpus aishell1 --input-dir data/aishell1/data_aishell --split dev --output runs/final/aishell1_dev/asr_manifest.jsonl",
        "stable-asr prepare-public-asr --corpus wenetspeech --input-dir data/wenetspeech/WenetSpeech --split dev --output runs/final/wenetspeech_dev/asr_manifest.jsonl",
        "stable-asr prepare-public-asr --corpus common_voice --input-dir data/common_voice/en --split dev --output runs/final/common_voice_en_dev/asr_manifest.jsonl",
        "stable-asr final-config --config configs/final/paper_final.json --bootstrap-turn-splits",
        "stable-asr final-config --config configs/final/paper_final.json --prepare-external-predictions",
        "stable-asr final-config --config configs/final/paper_final.json --audit-voiceworld-real --scenario-suite configs/scenarios/stable_asr_voiceworld_v0.json",
        "stable-asr train-turn --dataset runs/final/turn_train.jsonl --output-dir runs/final/nanoturn --model nanoturn_pico --feature-source audio",
        "stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --report runs/final/reports/asr_command_compare.md",
        "stable-asr paper-bundle --results runs/final/paper_results.json --output-dir runs/final/artifacts",
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
class FinalInputPrepareReport:
    ok: bool
    corpora: FinalCorpusPrepareReport
    turn_splits: FinalTurnBootstrapReport
    external_predictions: FinalExternalPredictionReport
    voiceworld_real: FinalVoiceWorldAuditReport
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
            "turn_splits": self.turn_splits.to_dict(),
            "external_predictions": self.external_predictions.to_dict(),
            "voiceworld_real": self.voiceworld_real.to_dict(),
            "file_audit": self.file_audit.to_dict(),
            "missing_required": self.missing_required,
        }

    def to_text(self) -> str:
        status = "READY" if self.ok else "NOT_READY"
        lines = [
            f"final_inputs_prepare: {status}",
            f"- corpora_prepared: {self.corpora.prepared_count}",
            f"- turn_records: {self.turn_splits.turn_records}",
            f"- external_predictions_prepared: {self.external_predictions.prepared_count}",
            f"- voiceworld_real_ready: {self.voiceworld_real.ok}",
            f"- missing_required: {len(self.missing_required)}",
        ]
        lines.extend(f"  - {path}" for path in self.missing_required)
        lines.extend(
            [
                "",
                self.corpora.to_text(),
                "",
                self.turn_splits.to_text(),
                "",
                self.external_predictions.to_text(),
                "",
                self.voiceworld_real.to_text(),
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
    for key in ("id", "version", "title", "output_dir", "seed", "public_corpora", "turn_splits", "artifacts", "commands"):
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

    commands = config.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("commands must be a non-empty list")
    elif not all(isinstance(command, str) and command.strip() for command in commands):
        errors.append("commands must contain non-empty strings")

    asr_command_config = config.get("asr_command_config")
    if not isinstance(asr_command_config, str) or not asr_command_config:
        errors.append("asr_command_config must be a non-empty string")

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

    for split, path in config.get("turn_splits", {}).items():
        checks.append(_input_check(f"turn_split:{split}", path, root=root))

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
        "## Turn Splits",
        "",
    ]
    for name, path in config["turn_splits"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## External Turn Predictions", ""])
    if config.get("external_turn_predictions"):
        lines.append(dict_table(_prediction_rows(config)))
    else:
        lines.append("No external turn predictions configured.")
    lines.extend(["", "## Artifacts", ""])
    for name, path in config["artifacts"].items():
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
    entries.append(
        _ensure_readme(
            split_parent / "TURN_SPLITS_README.md",
            title="Turn Split Inputs",
            body="\n".join(f"- `{name}`: `{path}`" for name, path in config["turn_splits"].items())
            + "\n\nThese files must be real Stable-ASR turn manifests.\n",
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
    voiceworld_report = audit_final_voiceworld_real(
        config,
        repo_root=repo_root,
        scenario_suite_path=scenario_suite_path,
        min_per_scenario=min_per_scenario,
    )
    file_audit = audit_final_run_files(config, repo_root=repo_root)
    ok = corpus_report.ok and turn_report.ok and prediction_report.ok and voiceworld_report.ok and file_audit.ok
    return FinalInputPrepareReport(
        ok=ok,
        corpora=corpus_report,
        turn_splits=turn_report,
        external_predictions=prediction_report,
        voiceworld_real=voiceworld_report,
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
