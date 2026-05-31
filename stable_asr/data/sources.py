"""Machine-readable dataset/source registry for Stable-ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.resources import resolve_platform_path


DEFAULT_DATA_SOURCES: dict[str, Any] = {
    "id": "stable_asr_sources_v0",
    "version": "0.1.0",
    "title": "Stable-ASR Data Source Registry",
    "description": (
        "Dataset and transcript-source registry for reproducible real-time ASR, "
        "turn-taking, endpointing, and full-duplex voice-agent experiments."
    ),
    "sources": [
        {
            "id": "synthetic_voiceworld",
            "title": "Stable-ASR Synthetic VoiceWorld",
            "task": "turn_taking",
            "languages": ["zh", "zh_en"],
            "source_type": "generated",
            "status": "implemented",
            "stable_asr_entrypoint": "stable-asr make-synthetic-turn-data",
            "license": "project_license",
            "notes": "Seedable synthetic turn/action scenarios used for smoke tests and paper artifact shape.",
        },
        {
            "id": "easyturn",
            "title": "Easy Turn-style manifests",
            "task": "turn_taking",
            "languages": ["multilingual"],
            "source_type": "external_manifest",
            "status": "converter_implemented",
            "stable_asr_entrypoint": "stable-asr convert-external --schema easyturn",
            "license": "see_upstream",
            "notes": "Four-state turn-taking style data converted into the Stable-ASR turn manifest.",
        },
        {
            "id": "full_duplex_bench",
            "title": "Full-Duplex-Bench-style manifests",
            "task": "full_duplex_eval",
            "languages": ["multilingual"],
            "source_type": "external_manifest",
            "status": "converter_implemented",
            "stable_asr_entrypoint": "stable-asr convert-external --schema full_duplex_bench",
            "license": "see_upstream",
            "notes": "Overlap and full-duplex interaction cases converted into the Stable-ASR turn manifest.",
        },
        {
            "id": "smart_turn",
            "title": "Smart Turn-style manifests",
            "task": "turn_detection",
            "languages": ["multilingual"],
            "source_type": "external_manifest",
            "status": "converter_implemented",
            "stable_asr_entrypoint": "stable-asr convert-external --schema smart_turn",
            "license": "see_upstream",
            "notes": "Turn completion probability rows converted into turn/action labels.",
        },
        {
            "id": "whisper_transcript",
            "title": "Whisper-style transcript exports",
            "task": "streaming_asr_eval",
            "languages": ["multilingual"],
            "source_type": "external_transcript",
            "status": "converter_implemented",
            "stable_asr_entrypoint": "stable-asr convert-asr-transcript --schema whisper",
            "license": "depends_on_input_audio",
            "notes": "Segments and word timestamps normalized into StreamingASRRecord JSONL.",
        },
        {
            "id": "funasr_transcript",
            "title": "FunASR-style transcript exports",
            "task": "streaming_asr_eval",
            "languages": ["multilingual"],
            "source_type": "external_transcript",
            "status": "converter_implemented",
            "stable_asr_entrypoint": "stable-asr convert-asr-transcript --schema funasr",
            "license": "depends_on_input_audio",
            "notes": "Sentence/timestamp rows normalized into StreamingASRRecord JSONL.",
        },
        {
            "id": "librispeech",
            "title": "LibriSpeech",
            "task": "asr",
            "languages": ["en"],
            "source_type": "public_corpus",
            "status": "recipe_scaffold",
            "stable_asr_entrypoint": "stable-asr prepare-asr-manifest --input <metadata.tsv> --output <manifest.jsonl>",
            "license": "see_upstream",
            "notes": "Canonical English ASR corpus; v0 provides metadata normalization before future MiniASR training recipes.",
        },
        {
            "id": "aishell1",
            "title": "AISHELL-1",
            "task": "asr",
            "languages": ["zh"],
            "source_type": "public_corpus",
            "status": "recipe_scaffold",
            "stable_asr_entrypoint": "stable-asr prepare-asr-manifest --input <metadata.tsv> --output <manifest.jsonl>",
            "license": "see_upstream",
            "notes": "Mandarin ASR corpus; v0 provides metadata normalization before future Chinese ASR recipes.",
        },
        {
            "id": "wenetspeech",
            "title": "WenetSpeech",
            "task": "asr",
            "languages": ["zh"],
            "source_type": "public_corpus",
            "status": "recipe_scaffold",
            "stable_asr_entrypoint": "stable-asr prepare-asr-manifest --input <metadata.tsv> --output <manifest.jsonl>",
            "license": "see_upstream",
            "notes": "Large Mandarin ASR source; v0 normalizes local metadata exports for later scaling experiments.",
        },
        {
            "id": "common_voice",
            "title": "Common Voice",
            "task": "asr",
            "languages": ["multilingual"],
            "source_type": "public_corpus",
            "status": "recipe_scaffold",
            "stable_asr_entrypoint": "stable-asr prepare-asr-manifest --input <metadata.tsv> --output <manifest.jsonl>",
            "license": "see_upstream",
            "notes": "Multilingual source; v0 normalizes local metadata exports for accent and cross-language robustness experiments.",
        },
    ],
}


@dataclass(frozen=True)
class DataSourceRegistryValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "data_sources: OK"
        return "data_sources: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


def load_data_sources(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_DATA_SOURCES))
    with resolve_platform_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("data source registry must be a JSON object")
    return payload


def write_data_sources_json(path: str | Path, registry: dict[str, Any] | None = None) -> str:
    registry = registry or load_data_sources()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_data_sources(registry: dict[str, Any]) -> DataSourceRegistryValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "sources"):
        if key not in registry:
            errors.append(f"missing top-level key: {key}")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("sources must be a non-empty list")
        return DataSourceRegistryValidation(ok=False, errors=errors)

    seen: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            errors.append(f"source {index} must be an object")
            continue
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"source {index} missing id")
        elif source_id in seen:
            errors.append(f"duplicate source id: {source_id}")
        else:
            seen.add(source_id)
        for key in ("title", "task", "languages", "source_type", "status", "stable_asr_entrypoint", "license"):
            if key not in source:
                errors.append(f"source {source_id or index} missing {key}")
        if "languages" in source and not isinstance(source["languages"], list):
            errors.append(f"source {source_id or index} languages must be a list")
    return DataSourceRegistryValidation(ok=not errors, errors=errors)


def data_sources_markdown(registry: dict[str, Any]) -> str:
    validation = validate_data_sources(registry)
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
        "## Sources",
        "",
        dict_table(_source_rows(registry)),
        "",
    ]
    return "\n".join(lines)


def _source_rows(registry: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for source in registry["sources"]:
        rows.append(
            {
                "id": source["id"],
                "task": source["task"],
                "languages": ", ".join(source.get("languages", [])),
                "type": source["source_type"],
                "status": source["status"],
                "entrypoint": source["stable_asr_entrypoint"],
            }
        )
    return rows
