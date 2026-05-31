"""Machine-readable adapter and baseline registry for Stable-ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.resources import resolve_platform_path


DEFAULT_ADAPTER_REGISTRY_PATH = Path("configs/adapters/stable_asr_adapters.json")
DEFAULT_ADAPTER_REGISTRY: dict[str, Any] = {
    "id": "stable_asr_adapters_v0",
    "version": "0.1.0",
    "title": "Stable-ASR Adapter Registry",
    "description": (
        "Adapter and baseline registry for reproducible turn-taking, streaming "
        "ASR, transcript conversion, and command-backed external system evaluation."
    ),
    "adapters": [
        {
            "id": "rule_endpoint",
            "title": "Rule Endpoint Baseline",
            "task": "turn_taking",
            "modality": "metadata",
            "status": "implemented",
            "interface": "TurnPredictor",
            "entrypoint": "stable_asr.models.baselines.RuleEndpointBaseline",
            "input_schema": "TurnManifestRecord",
            "output_schema": "TurnPrediction",
            "license": "project_license",
            "notes": "Lowest endpointing baseline using manifest metadata.",
        },
        {
            "id": "vad_pause",
            "title": "VAD Pause Baseline",
            "task": "turn_taking",
            "modality": "metadata",
            "status": "implemented",
            "interface": "TurnPredictor",
            "entrypoint": "stable_asr.models.baselines.VADPauseBaseline",
            "input_schema": "TurnManifestRecord",
            "output_schema": "TurnPrediction",
            "license": "project_license",
            "notes": "Industrial-style pause threshold baseline.",
        },
        {
            "id": "text_turn",
            "title": "Text Turn Baseline",
            "task": "turn_taking",
            "modality": "text",
            "status": "implemented",
            "interface": "TurnPredictor",
            "entrypoint": "stable_asr.models.baselines.TextTurnBaseline",
            "input_schema": "TurnManifestRecord",
            "output_schema": "TurnPrediction",
            "license": "project_license",
            "notes": "Simple semantic baseline over reference or ASR text.",
        },
        {
            "id": "prediction_manifest",
            "title": "External Turn Prediction Manifest",
            "task": "turn_taking",
            "modality": "prediction_jsonl",
            "status": "implemented",
            "interface": "TurnPredictor",
            "entrypoint": "stable_asr.models.adapters.TurnPredictionManifestAdapter",
            "input_schema": "Stable-ASR turn prediction JSONL",
            "output_schema": "TurnPrediction",
            "license": "depends_on_input_predictions",
            "notes": "Dependency-light bridge for SmartTurn/EasyTurn/VAP-style outputs.",
        },
        {
            "id": "transcript_jsonl",
            "title": "Transcript JSONL Streaming ASR Adapter",
            "task": "streaming_asr_eval",
            "modality": "streaming_transcript_jsonl",
            "status": "implemented",
            "interface": "StreamingASRAdapter",
            "entrypoint": "stable_asr.models.adapters.TranscriptJSONLAdapter",
            "input_schema": "Stable-ASR StreamingASRRecord JSONL",
            "output_schema": "list[StreamingASRRecord]",
            "license": "depends_on_input_audio",
            "notes": "Adapter used by eval-streaming-asr, compare-streaming-asr, and sweep-streaming-asr.",
        },
        {
            "id": "command_streaming_asr",
            "title": "Command-Backed Streaming ASR Adapter",
            "task": "streaming_asr_eval",
            "modality": "external_command",
            "status": "implemented",
            "interface": "StreamingASRAdapter",
            "entrypoint": "stable_asr.models.adapters.CommandStreamingASRAdapter",
            "input_schema": "command writes Stable-ASR StreamingASRRecord JSONL",
            "output_schema": "list[StreamingASRRecord]",
            "license": "depends_on_external_system",
            "notes": "Runs an external command and evaluates the transcript JSONL it writes.",
        },
        {
            "id": "whisper_transcript",
            "title": "Whisper Transcript Converter",
            "task": "streaming_asr_eval",
            "modality": "external_transcript",
            "status": "converter_implemented",
            "interface": "TranscriptConverter",
            "entrypoint": "stable-asr convert-asr-transcript --schema whisper",
            "input_schema": "Whisper-style segments/words JSONL",
            "output_schema": "Stable-ASR StreamingASRRecord JSONL",
            "license": "depends_on_input_audio",
            "notes": "Normalizes Whisper segment and word timestamp exports.",
        },
        {
            "id": "funasr_transcript",
            "title": "FunASR Transcript Converter",
            "task": "streaming_asr_eval",
            "modality": "external_transcript",
            "status": "converter_implemented",
            "interface": "TranscriptConverter",
            "entrypoint": "stable-asr convert-asr-transcript --schema funasr",
            "input_schema": "FunASR-style sentence_info/timestamp JSONL",
            "output_schema": "Stable-ASR StreamingASRRecord JSONL",
            "license": "depends_on_input_audio",
            "notes": "Normalizes FunASR sentence and timestamp exports.",
        },
        {
            "id": "whisper_command_template",
            "title": "Whisper Command Template",
            "task": "streaming_asr_eval",
            "modality": "external_command",
            "status": "template",
            "interface": "CommandStreamingASRAdapter",
            "entrypoint": "stable-asr eval-asr-command --command '<whisper export script> --output {output}'",
            "input_schema": "local audio or manifest handled by external script",
            "output_schema": "Stable-ASR StreamingASRRecord JSONL",
            "license": "depends_on_external_system",
            "notes": "Template for wrapping Whisper or faster-whisper exporters without adding a hard dependency.",
        },
        {
            "id": "funasr_command_template",
            "title": "FunASR Command Template",
            "task": "streaming_asr_eval",
            "modality": "external_command",
            "status": "template",
            "interface": "CommandStreamingASRAdapter",
            "entrypoint": "stable-asr eval-asr-command --command '<funasr export script> --output {output}'",
            "input_schema": "local audio or manifest handled by external script",
            "output_schema": "Stable-ASR StreamingASRRecord JSONL",
            "license": "depends_on_external_system",
            "notes": "Template for evaluating FunASR-style systems through the command adapter.",
        },
        {
            "id": "wenet_command_template",
            "title": "WeNet Command Template",
            "task": "streaming_asr_eval",
            "modality": "external_command",
            "status": "template",
            "interface": "CommandStreamingASRAdapter",
            "entrypoint": "stable-asr eval-asr-command --command '<wenet export script> --output {output}'",
            "input_schema": "local audio or manifest handled by external script",
            "output_schema": "Stable-ASR StreamingASRRecord JSONL",
            "license": "depends_on_external_system",
            "notes": "Template for evaluating WeNet systems without vendoring WeNet.",
        },
        {
            "id": "sherpa_onnx_command_template",
            "title": "sherpa-onnx Command Template",
            "task": "streaming_asr_eval",
            "modality": "external_command",
            "status": "template",
            "interface": "CommandStreamingASRAdapter",
            "entrypoint": "stable-asr eval-asr-command --command '<sherpa-onnx export script> --output {output}'",
            "input_schema": "local audio or manifest handled by external script",
            "output_schema": "Stable-ASR StreamingASRRecord JSONL",
            "license": "depends_on_external_system",
            "related_references": ["sherpa_onnx"],
            "notes": "Template for evaluating sherpa-onnx runtime exports, including CPU and embedded deployment reports.",
        },
        {
            "id": "lhotse_manifest_bridge_template",
            "title": "Lhotse Manifest Bridge Template",
            "task": "data_preparation",
            "modality": "manifest",
            "status": "template",
            "interface": "ASRManifestBridge",
            "entrypoint": "planned: lhotse cuts/supervisions to Stable-ASR ASR/turn manifests",
            "input_schema": "Lhotse recording/cut/supervision manifests",
            "output_schema": "Stable-ASR ASRManifestRecord or TurnManifestRecord JSONL",
            "license": "depends_on_input_data",
            "related_references": ["lhotse"],
            "notes": "Planning template for interoperating with Lhotse instead of duplicating mature corpus recipes.",
        },
        {
            "id": "smart_turn_prediction",
            "title": "SmartTurn Prediction Converter",
            "task": "turn_taking",
            "modality": "external_prediction_jsonl",
            "status": "converter_implemented",
            "interface": "TurnPredictionManifestAdapter",
            "entrypoint": "stable-asr convert-predictions --schema smart_turn",
            "input_schema": "SmartTurn-style completion probability JSONL",
            "output_schema": "Stable-ASR turn prediction JSONL",
            "license": "see_upstream",
            "notes": "Converts SmartTurn-like probabilities into the shared prediction adapter format.",
        },
        {
            "id": "easy_turn_prediction",
            "title": "EasyTurn Prediction Converter",
            "task": "turn_taking",
            "modality": "external_prediction_jsonl",
            "status": "converter_implemented",
            "interface": "TurnPredictionManifestAdapter",
            "entrypoint": "stable-asr convert-predictions --schema easyturn",
            "input_schema": "EasyTurn-style four-state prediction JSONL",
            "output_schema": "Stable-ASR turn prediction JSONL",
            "license": "see_upstream",
            "notes": "Converts EasyTurn-like predictions into the shared prediction adapter format.",
        },
        {
            "id": "vap_prediction_template",
            "title": "VAP Prediction Template",
            "task": "turn_taking",
            "modality": "external_prediction_jsonl",
            "status": "template",
            "interface": "TurnPredictionManifestAdapter",
            "entrypoint": "stable-asr convert-predictions --schema generic",
            "input_schema": "generic prediction JSONL with probs and timestamp",
            "output_schema": "Stable-ASR turn prediction JSONL",
            "license": "see_upstream",
            "notes": "Template for mapping VAP-style future activity scores to Stable-ASR turn probabilities.",
        },
    ],
}


@dataclass(frozen=True)
class AdapterRegistryValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "adapter_registry: OK"
        return "adapter_registry: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


def load_adapter_registry(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        registry_path = resolve_platform_path(DEFAULT_ADAPTER_REGISTRY_PATH)
        if registry_path.exists():
            with registry_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("adapter registry must be a JSON object")
            return payload
        return json.loads(json.dumps(DEFAULT_ADAPTER_REGISTRY))
    with resolve_platform_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("adapter registry must be a JSON object")
    return payload


def write_adapter_registry_json(path: str | Path, registry: dict[str, Any] | None = None) -> str:
    registry = registry or load_adapter_registry()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_adapter_registry(registry: dict[str, Any]) -> AdapterRegistryValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "adapters"):
        if key not in registry:
            errors.append(f"missing top-level key: {key}")
    adapters = registry.get("adapters")
    if not isinstance(adapters, list) or not adapters:
        errors.append("adapters must be a non-empty list")
        return AdapterRegistryValidation(ok=False, errors=errors)

    seen: set[str] = set()
    required = {
        "id",
        "title",
        "task",
        "modality",
        "status",
        "interface",
        "entrypoint",
        "input_schema",
        "output_schema",
        "license",
    }
    for index, adapter in enumerate(adapters):
        if not isinstance(adapter, dict):
            errors.append(f"adapter {index} must be an object")
            continue
        adapter_id = adapter.get("id")
        if not isinstance(adapter_id, str) or not adapter_id:
            errors.append(f"adapter {index} missing id")
        elif adapter_id in seen:
            errors.append(f"duplicate adapter id: {adapter_id}")
        else:
            seen.add(adapter_id)
        for key in required:
            if key not in adapter:
                errors.append(f"adapter {adapter_id or index} missing {key}")
        related = adapter.get("related_references", [])
        if related is not None and (
            not isinstance(related, list) or not all(isinstance(item, str) and item for item in related)
        ):
            errors.append(f"adapter {adapter_id or index} related_references must be a string list")
    return AdapterRegistryValidation(ok=not errors, errors=errors)


def adapter_registry_markdown(registry: dict[str, Any]) -> str:
    validation = validate_adapter_registry(registry)
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
        "## Adapters",
        "",
        dict_table(_adapter_rows(registry)),
        "",
    ]
    return "\n".join(lines)


def _adapter_rows(registry: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "id": adapter["id"],
            "task": adapter["task"],
            "modality": adapter["modality"],
            "status": adapter["status"],
            "interface": adapter["interface"],
            "references": ", ".join(adapter.get("related_references", [])),
            "entrypoint": adapter["entrypoint"],
        }
        for adapter in registry["adapters"]
    ]
