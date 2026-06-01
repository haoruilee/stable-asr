"""Machine-readable built-in model registry for Stable-ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.resources import resolve_platform_path
from stable_asr.schema_validation import validate_schema_file
from stable_asr.turn.labels import ACTION_LABELS, TURN_LABELS


DEFAULT_MODEL_REGISTRY_PATH = Path("configs/models/stable_asr_models.json")
DEFAULT_MODEL_REGISTRY: dict[str, Any] = {
    "id": "stable_asr_models_v0",
    "version": "0.1.0",
    "title": "Stable-ASR Model Registry",
    "description": (
        "Built-in model and baseline registry for reproducible turn-taking, "
        "endpointing, interruption handling, and streaming ASR system studies."
    ),
    "models": [
        {
            "id": "nanoturn_pico",
            "title": "NanoTurn Pico",
            "family": "NanoTurn",
            "task": "turn_taking",
            "model_type": "trainable_baseline",
            "status": "implemented",
            "modality": "turn_manifest_metadata_v0",
            "interface": "TurnPredictor",
            "entrypoint": "stable_asr.turn.nanoturn.NanoTurnPico",
            "training_entrypoint": "stable-asr train-turn --model nanoturn_pico",
            "evaluation_entrypoint": "stable-asr eval-turn --checkpoint <checkpoint.pt> --dataset <turn.jsonl>",
            "export_entrypoint": "stable-asr export-turn-onnx --checkpoint <checkpoint.pt> --output <model.onnx>",
            "input_schema": "TurnManifestRecord metadata feature vector",
            "output_schema": "TurnPrediction probabilities over four turn labels",
            "labels": sorted(TURN_LABELS),
            "actions": sorted(ACTION_LABELS),
            "config_path": "configs/nanoturn_pico.json",
            "license": "project_license",
            "intended_use": (
                "Small checkpoint-backed baseline for validating Stable-ASR turn-taking, "
                "endpointing, policy search, latency, ONNX export, and paper artifact plumbing."
            ),
            "limitations": [
                "v0 uses manifest metadata features by default, not a production audio frontend",
                "smoke metrics are fixture-level checks and must not be reported as final model quality",
                "final claims require real train/dev/test splits and external baseline comparisons",
            ],
        },
        {
            "id": "nanoturn_nano",
            "title": "NanoTurn Nano",
            "family": "NanoTurn",
            "task": "turn_taking",
            "model_type": "trainable_baseline",
            "status": "implemented",
            "modality": "turn_manifest_metadata_v0",
            "interface": "TurnPredictor",
            "entrypoint": "stable_asr.turn.nanoturn.NanoTurnNano",
            "training_entrypoint": "stable-asr train-turn --model nanoturn_nano",
            "evaluation_entrypoint": "stable-asr eval-turn --checkpoint <checkpoint.pt> --dataset <turn.jsonl>",
            "export_entrypoint": "stable-asr export-turn-onnx --checkpoint <checkpoint.pt> --output <model.onnx>",
            "input_schema": "TurnManifestRecord metadata feature vector",
            "output_schema": "TurnPrediction probabilities over four turn labels",
            "labels": sorted(TURN_LABELS),
            "actions": sorted(ACTION_LABELS),
            "config_path": "configs/nanoturn_nano.json",
            "license": "project_license",
            "intended_use": (
                "Larger NanoTurn baseline intended for final-scale audio-feature and "
                "metadata-feature turn/action experiments."
            ),
            "limitations": [
                "v0 uses the same manifest metadata feature family as NanoTurn Pico; audio-feature final-scale training remains a paper-scale target",
                "requires real train/dev/test splits and external baseline comparisons before quality claims",
            ],
        },
        {
            "id": "rule_endpoint",
            "title": "Rule Endpoint Baseline",
            "family": "RuleEndpoint",
            "task": "endpointing",
            "model_type": "deterministic_baseline",
            "status": "implemented",
            "modality": "pause_metadata",
            "interface": "TurnPredictor",
            "entrypoint": "stable_asr.models.baselines.RuleEndpointBaseline",
            "training_entrypoint": "not_applicable",
            "evaluation_entrypoint": "stable-asr eval-turn --baseline rule_endpoint --dataset <turn.jsonl>",
            "export_entrypoint": "not_applicable",
            "input_schema": "TurnManifestRecord with pause_ms metadata",
            "output_schema": "TurnPrediction complete/incomplete probabilities",
            "labels": sorted(TURN_LABELS),
            "actions": sorted(ACTION_LABELS),
            "config_path": "constructor argument: complete_pause_ms",
            "license": "project_license",
            "intended_use": "Lowest endpointing baseline for cost-sensitive turn policy comparisons.",
            "limitations": [
                "does not inspect audio or text",
                "cannot distinguish backchannel, wait, side speech, or interruption semantics",
            ],
        },
        {
            "id": "vad_pause",
            "title": "VAD Pause Baseline",
            "family": "VADPause",
            "task": "endpointing",
            "model_type": "deterministic_baseline",
            "status": "implemented",
            "modality": "vad_pause_metadata",
            "interface": "TurnPredictor",
            "entrypoint": "stable_asr.models.baselines.VADPauseBaseline",
            "training_entrypoint": "not_applicable",
            "evaluation_entrypoint": "stable-asr eval-turn --baseline vad_pause --dataset <turn.jsonl>",
            "export_entrypoint": "not_applicable",
            "input_schema": "TurnManifestRecord with vad_pause_ms or pause_ms metadata",
            "output_schema": "TurnPrediction complete/incomplete probabilities",
            "labels": sorted(TURN_LABELS),
            "actions": sorted(ACTION_LABELS),
            "config_path": "constructor argument: complete_pause_ms",
            "license": "project_license",
            "intended_use": "Industrial-style pause threshold baseline for endpointing comparisons.",
            "limitations": [
                "depends on upstream VAD quality",
                "does not model lexical intent, backchannels, or assistant speaking state",
            ],
        },
        {
            "id": "text_turn",
            "title": "Text Turn Baseline",
            "family": "TextTurn",
            "task": "turn_taking",
            "model_type": "deterministic_baseline",
            "status": "implemented",
            "modality": "text",
            "interface": "TurnPredictor",
            "entrypoint": "stable_asr.models.baselines.TextTurnBaseline",
            "training_entrypoint": "not_applicable",
            "evaluation_entrypoint": "stable-asr eval-turn --baseline text_turn --dataset <turn.jsonl>",
            "export_entrypoint": "not_applicable",
            "input_schema": "TurnManifestRecord text or asr_text",
            "output_schema": "TurnPrediction probabilities over four turn labels",
            "labels": sorted(TURN_LABELS),
            "actions": sorted(ACTION_LABELS),
            "config_path": "constructor argument: prefer_asr_text",
            "license": "project_license",
            "intended_use": "Semantic rule baseline for comparing ASR-text cues against pause-only models.",
            "limitations": [
                "rule list is intentionally small",
                "inherits ASR transcript errors when asr_text is used",
                "does not inspect audio prosody",
            ],
        },
        {
            "id": "prediction_manifest",
            "title": "External Turn Prediction Manifest",
            "family": "AdapterBaseline",
            "task": "turn_taking",
            "model_type": "external_prediction_adapter",
            "status": "implemented",
            "modality": "prediction_jsonl",
            "interface": "TurnPredictor",
            "entrypoint": "stable_asr.models.adapters.TurnPredictionManifestAdapter",
            "training_entrypoint": "not_applicable",
            "evaluation_entrypoint": "stable-asr eval-turn --predictions <predictions.jsonl> --dataset <turn.jsonl>",
            "export_entrypoint": "not_applicable",
            "input_schema": "Stable-ASR turn prediction JSONL",
            "output_schema": "TurnPrediction probabilities over four turn labels",
            "labels": sorted(TURN_LABELS),
            "actions": sorted(ACTION_LABELS),
            "config_path": "prediction manifest path",
            "license": "depends_on_input_predictions",
            "intended_use": "Bridge for SmartTurn, EasyTurn, VAP, or vendor turn detector outputs.",
            "limitations": [
                "adapter quality depends entirely on the external prediction file",
                "upstream model licenses and data provenance must be audited separately",
            ],
        },
    ],
}


@dataclass(frozen=True)
class ModelRegistryValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "model_registry: OK"
        return "model_registry: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class ModelConfigAuditRow:
    model_id: str
    config_path: str
    expected: bool
    exists: bool
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "config_path": self.config_path,
            "expected": self.expected,
            "exists": self.exists,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ModelConfigAuditReport:
    ok: bool
    rows: list[ModelConfigAuditRow]
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "rows": [row.to_dict() for row in self.rows],
            "errors": self.errors,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Stable-ASR Model Config Audit",
            "",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- rows: `{len(self.rows)}`",
            "",
            "## Configs",
            "",
            dict_table(
                [
                    {
                        "model": row.model_id,
                        "expected": row.expected,
                        "exists": row.exists,
                        "ok": row.ok,
                        "config": row.config_path,
                        "detail": row.detail,
                    }
                    for row in self.rows
                ]
            ),
            "",
            "## Errors",
            "",
        ]
        if self.errors:
            lines.extend(f"- `{error}`" for error in self.errors)
        else:
            lines.append("- None")
        lines.append("")
        return "\n".join(lines)


def load_model_registry(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        registry_path = resolve_platform_path(DEFAULT_MODEL_REGISTRY_PATH)
        if registry_path.exists():
            with registry_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("model registry must be a JSON object")
            return payload
        return json.loads(json.dumps(DEFAULT_MODEL_REGISTRY))
    with resolve_platform_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("model registry must be a JSON object")
    return payload


def write_model_registry_json(path: str | Path, registry: dict[str, Any] | None = None) -> str:
    registry = registry or load_model_registry()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_model_registry(registry: dict[str, Any]) -> ModelRegistryValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "models"):
        if key not in registry:
            errors.append(f"missing top-level key: {key}")
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        errors.append("models must be a non-empty list")
        return ModelRegistryValidation(ok=False, errors=errors)

    required = {
        "id",
        "title",
        "family",
        "task",
        "model_type",
        "status",
        "modality",
        "interface",
        "entrypoint",
        "input_schema",
        "output_schema",
        "license",
        "intended_use",
        "limitations",
    }
    seen: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"model {index} must be an object")
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            errors.append(f"model {index} missing id")
        elif model_id in seen:
            errors.append(f"duplicate model id: {model_id}")
        else:
            seen.add(model_id)
        for key in required:
            if key not in model:
                errors.append(f"model {model_id or index} missing {key}")
        for key in ("labels", "actions", "limitations"):
            if key in model and (
                not isinstance(model[key], list)
                or not all(isinstance(item, str) and item for item in model[key])
            ):
                errors.append(f"model {model_id or index} {key} must be a string list")
    return ModelRegistryValidation(ok=not errors, errors=errors)


def audit_model_registry_configs(
    registry: dict[str, Any] | None = None,
    *,
    repo_root: str | Path = ".",
) -> ModelConfigAuditReport:
    """Audit config files referenced by trainable built-in model registry entries."""

    registry = registry or load_model_registry()
    validation = validate_model_registry(registry)
    if not validation.ok:
        return ModelConfigAuditReport(ok=False, rows=[], errors=validation.errors)

    repo_root = Path(repo_root)
    rows: list[ModelConfigAuditRow] = []
    errors: list[str] = []
    for model in registry["models"]:
        model_id = str(model["id"])
        config_path = str(model.get("config_path", ""))
        expected = model.get("model_type") == "trainable_baseline"
        if not expected:
            continue
        row = _audit_model_config(model_id, config_path, repo_root=repo_root)
        rows.append(row)
        if not row.ok:
            errors.append(f"{model_id}: {row.detail}")
    return ModelConfigAuditReport(ok=not errors, rows=rows, errors=errors)


def model_registry_markdown(registry: dict[str, Any]) -> str:
    validation = validate_model_registry(registry)
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
        "## Models",
        "",
        dict_table(_model_rows(registry)),
        "",
    ]
    return "\n".join(lines)


def find_model_entry(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    validation = validate_model_registry(registry)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    for model in registry["models"]:
        if model["id"] == model_id:
            return dict(model)
    raise ValueError(f"unknown model id: {model_id}")


def _audit_model_config(model_id: str, config_path: str, *, repo_root: Path) -> ModelConfigAuditRow:
    if not config_path or config_path.startswith("planned:"):
        return ModelConfigAuditRow(
            model_id=model_id,
            config_path=config_path,
            expected=True,
            exists=False,
            ok=False,
            detail="config_path is planned or missing",
        )
    path = Path(config_path)
    if not path.is_absolute():
        repo_path = repo_root / path
        path = repo_path if repo_path.exists() else resolve_platform_path(config_path)
    exists = path.exists()
    if not exists:
        return ModelConfigAuditRow(
            model_id=model_id,
            config_path=config_path,
            expected=True,
            exists=False,
            ok=False,
            detail=f"missing: {config_path}",
        )
    try:
        schema_report = validate_schema_file(path, schema_id="stable_asr.nanoturn_train_config.v0")
        if not schema_report.ok:
            issues = "; ".join(f"{issue.path}: {issue.detail}" for issue in schema_report.issues[:3])
            return ModelConfigAuditRow(
                model_id=model_id,
                config_path=config_path,
                expected=True,
                exists=True,
                ok=False,
                detail=issues,
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ModelConfigAuditRow(
            model_id=model_id,
            config_path=config_path,
            expected=True,
            exists=True,
            ok=False,
            detail=str(exc),
        )
    errors: list[str] = []
    if payload.get("model_type") != model_id:
        errors.append(f"model_type={payload.get('model_type')!r}, expected {model_id!r}")
    for key in ("epochs", "lr", "seed", "feature_source"):
        if key not in payload:
            errors.append(f"missing {key}")
    if errors:
        return ModelConfigAuditRow(
            model_id=model_id,
            config_path=config_path,
            expected=True,
            exists=True,
            ok=False,
            detail="; ".join(errors),
        )
    return ModelConfigAuditRow(
        model_id=model_id,
        config_path=config_path,
        expected=True,
        exists=True,
        ok=True,
        detail="ready",
    )


def _model_rows(registry: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "id": model["id"],
            "family": model["family"],
            "task": model["task"],
            "type": model["model_type"],
            "status": model["status"],
            "modality": model["modality"],
            "interface": model["interface"],
        }
        for model in registry["models"]
    ]
