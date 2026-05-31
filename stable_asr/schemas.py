"""Machine-readable JSON Schema registry for Stable-ASR public contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.resources import resolve_platform_path


DEFAULT_SCHEMA_REGISTRY_PATH = Path("configs/schemas/stable_asr_schemas.json")
SUPPORTED_FORMATS = {"json", "jsonl"}


@dataclass(frozen=True)
class SchemaRegistryValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "schema_registry: OK"
        return "schema_registry: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


def load_schema_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Load the checked-in schema registry."""

    registry_path = resolve_platform_path(path or DEFAULT_SCHEMA_REGISTRY_PATH)
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("schema registry must be a JSON object")
    return payload


def write_schema_registry_json(path: str | Path, registry: dict[str, Any] | None = None) -> str:
    registry = registry or load_schema_registry()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_schema_registry(registry: dict[str, Any]) -> SchemaRegistryValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "schemas"):
        if key not in registry:
            errors.append(f"missing top-level key: {key}")

    schemas = registry.get("schemas")
    if not isinstance(schemas, list) or not schemas:
        errors.append("schemas must be a non-empty list")
        return SchemaRegistryValidation(ok=False, errors=errors)

    seen: set[str] = set()
    for index, entry in enumerate(schemas):
        if not isinstance(entry, dict):
            errors.append(f"schema {index} must be an object")
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            errors.append(f"schema {index} missing id")
        elif entry_id in seen:
            errors.append(f"duplicate schema id: {entry_id}")
        else:
            seen.add(entry_id)

        for key in ("version", "title", "description", "format", "schema"):
            if key not in entry:
                errors.append(f"schema {entry_id or index} missing {key}")

        if entry.get("format") not in SUPPORTED_FORMATS:
            errors.append(f"schema {entry_id or index} has unsupported format: {entry.get('format')!r}")

        for list_key in ("file_patterns", "producer_commands"):
            value = entry.get(list_key, [])
            if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                errors.append(f"schema {entry_id or index} {list_key} must be a list of non-empty strings")

        schema = entry.get("schema")
        if not isinstance(schema, dict):
            errors.append(f"schema {entry_id or index} schema must be an object")
            continue
        errors.extend(_validate_json_schema_shape(entry_id or str(index), schema))

    return SchemaRegistryValidation(ok=not errors, errors=errors)


def get_schema_entry(registry: dict[str, Any], schema_id: str) -> dict[str, Any]:
    for entry in registry.get("schemas", []):
        if isinstance(entry, dict) and entry.get("id") == schema_id:
            return entry
    raise KeyError(f"unknown schema id: {schema_id}")


def schema_registry_markdown(registry: dict[str, Any]) -> str:
    rows = []
    for entry in registry.get("schemas", []):
        if not isinstance(entry, dict):
            continue
        schema = entry.get("schema") if isinstance(entry.get("schema"), dict) else {}
        required = schema.get("required", [])
        rows.append(
            {
                "id": entry.get("id", ""),
                "version": entry.get("version", ""),
                "format": entry.get("format", ""),
                "required": ", ".join(required) if isinstance(required, list) else "",
                "description": entry.get("description", ""),
            }
        )

    lines = [
        "# Stable-ASR Schema Registry",
        "",
        f"- id: `{registry.get('id', '')}`",
        f"- version: `{registry.get('version', '')}`",
        f"- schemas: `{len(rows)}`",
        "",
        "## Schemas",
        "",
        dict_table(rows) if rows else "No schemas.",
        "",
        "## Usage",
        "",
        "```bash",
        "stable-asr schema-registry --validate-only",
        "stable-asr schema-registry --schema-id stable_asr.turn_manifest_record.v0 --json",
        "stable-asr schema-registry --output runs/SCHEMAS.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def schema_entry_markdown(entry: dict[str, Any]) -> str:
    schema = entry.get("schema") if isinstance(entry.get("schema"), dict) else {}
    return "\n".join(
        [
            f"# {entry.get('title', entry.get('id', 'Stable-ASR Schema'))}",
            "",
            f"- id: `{entry.get('id', '')}`",
            f"- version: `{entry.get('version', '')}`",
            f"- format: `{entry.get('format', '')}`",
            "",
            str(entry.get("description", "")),
            "",
            "## JSON Schema",
            "",
            "```json",
            json.dumps(schema, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def _validate_json_schema_shape(entry_id: str, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append(f"schema {entry_id} must declare JSON Schema draft 2020-12")
    if schema.get("type") != "object":
        errors.append(f"schema {entry_id} must be an object schema")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        errors.append(f"schema {entry_id} properties must be a non-empty object")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) and item for item in required):
        errors.append(f"schema {entry_id} required must be a list of non-empty strings")
    elif isinstance(properties, dict):
        missing_required_defs = [field for field in required if field not in properties]
        if missing_required_defs:
            errors.append(
                f"schema {entry_id} required fields missing property definitions: "
                + ", ".join(missing_required_defs)
            )
    return errors
