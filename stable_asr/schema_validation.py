"""Validate JSON and JSONL files against Stable-ASR JSON Schema contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.formats.jsonl import iter_jsonl
from stable_asr.eval.report import dict_table
from stable_asr.schemas import get_schema_entry, load_schema_registry, validate_schema_registry


JSON_FORMATS = {"auto", "json", "jsonl"}


@dataclass(frozen=True)
class SchemaValidationIssue:
    line: int | None
    path: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"path": self.path, "detail": self.detail}
        if self.line is not None:
            payload["line"] = self.line
        return payload


@dataclass(frozen=True)
class SchemaFileValidationReport:
    ok: bool
    input_path: str
    schema_id: str
    format: str
    records: int
    issues: list[SchemaValidationIssue]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "input_path": self.input_path,
            "schema_id": self.schema_id,
            "format": self.format,
            "records": self.records,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_text(self) -> str:
        lines = [
            f"schema_file_validation: {'OK' if self.ok else 'FAILED'}",
            f"input: {self.input_path}",
            f"schema_id: {self.schema_id}",
            f"format: {self.format}",
            f"records: {self.records}",
            f"issues: {len(self.issues)}",
        ]
        for issue in self.issues:
            prefix = f"line {issue.line} " if issue.line is not None else ""
            lines.append(f"- {prefix}{issue.path}: {issue.detail}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        rows = [
            {
                "line": "" if issue.line is None else issue.line,
                "path": issue.path,
                "detail": issue.detail,
            }
            for issue in self.issues
        ]
        return "\n".join(
            [
                "# Stable-ASR Schema File Validation",
                "",
                f"- status: `{'OK' if self.ok else 'FAILED'}`",
                f"- input: `{self.input_path}`",
                f"- schema_id: `{self.schema_id}`",
                f"- format: `{self.format}`",
                f"- records: `{self.records}`",
                f"- issues: `{len(self.issues)}`",
                "",
                "## Issues",
                "",
                dict_table(rows) if rows else "No issues.",
                "",
            ]
        )


def validate_schema_file(
    input_path: str | Path,
    *,
    schema_id: str,
    registry_path: str | Path | None = None,
    format: str = "auto",
    max_errors: int = 50,
) -> SchemaFileValidationReport:
    """Validate a JSON or JSONL file against one schema from the registry."""

    if format not in JSON_FORMATS:
        raise ValueError(f"format must be one of {sorted(JSON_FORMATS)}")
    if max_errors <= 0:
        raise ValueError("max_errors must be positive")

    registry = load_schema_registry(registry_path)
    registry_validation = validate_schema_registry(registry)
    if not registry_validation.ok:
        raise ValueError(registry_validation.to_text())

    entry = get_schema_entry(registry, schema_id)
    schema = entry.get("schema")
    if not isinstance(schema, dict):
        raise ValueError(f"schema entry {schema_id!r} does not contain a JSON schema object")

    input_path = Path(input_path)
    resolved_format = _resolve_format(input_path, entry, format)
    records = 0
    issues: list[SchemaValidationIssue] = []

    try:
        if resolved_format == "jsonl":
            for line_number, item in iter_jsonl(input_path):
                records += 1
                issues.extend(_line_issues(line_number, item, schema))
                if len(issues) >= max_errors:
                    break
        else:
            with input_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            records = 1
            issues.extend(_line_issues(None, payload, schema))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(SchemaValidationIssue(None, "$", str(exc)))

    return SchemaFileValidationReport(
        ok=not issues,
        input_path=str(input_path),
        schema_id=schema_id,
        format=resolved_format,
        records=records,
        issues=issues[:max_errors],
    )


def _line_issues(line_number: int | None, item: Any, schema: dict[str, Any]) -> list[SchemaValidationIssue]:
    return [
        SchemaValidationIssue(line_number, path, detail)
        for path, detail in _validate_value(item, schema, schema, "$")
    ]


def _resolve_format(path: Path, entry: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    if path.suffix.lower() == ".json":
        return "json"
    entry_format = entry.get("format")
    return "jsonl" if entry_format == "jsonl" else "json"


def _validate_value(value: Any, schema: dict[str, Any], root_schema: dict[str, Any], path: str) -> list[tuple[str, str]]:
    if "$ref" in schema:
        schema = _resolve_ref(str(schema["$ref"]), root_schema)

    errors: list[tuple[str, str]] = []
    if "anyOf" in schema:
        candidates = schema["anyOf"]
        if not isinstance(candidates, list) or not candidates:
            errors.append((path, "anyOf must be a non-empty list"))
        elif not any(not _validate_value(value, candidate, root_schema, path) for candidate in candidates if isinstance(candidate, dict)):
            errors.append((path, "must match at least one anyOf schema"))

    allowed_types = schema.get("type")
    if allowed_types is not None and not _matches_type(value, allowed_types):
        errors.append((path, f"expected type {_type_text(allowed_types)}, got {_value_type(value)}"))
        return errors

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append((path, "expected one of: " + ", ".join(str(item) for item in enum)))

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append((path, f"must have length >= {min_length}"))

    if _is_number(value):
        minimum = schema.get("minimum")
        if _is_number(minimum) and value < minimum:
            errors.append((path, f"must be >= {minimum}"))
        maximum = schema.get("maximum")
        if _is_number(maximum) and value > maximum:
            errors.append((path, f"must be <= {maximum}"))
        exclusive_minimum = schema.get("exclusiveMinimum")
        if _is_number(exclusive_minimum) and value <= exclusive_minimum:
            errors.append((path, f"must be > {exclusive_minimum}"))
        exclusive_maximum = schema.get("exclusiveMaximum")
        if _is_number(exclusive_maximum) and value >= exclusive_maximum:
            errors.append((path, f"must be < {exclusive_maximum}"))

    if isinstance(value, dict):
        errors.extend(_validate_object(value, schema, root_schema, path))
    if isinstance(value, list):
        errors.extend(_validate_array(value, schema, root_schema, path))

    return errors


def _validate_object(
    value: dict[str, Any],
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str,
) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    required = schema.get("required", [])
    if isinstance(required, list):
        for field in required:
            if isinstance(field, str) and field not in value:
                errors.append((f"{path}.{field}", "missing required field"))

    min_properties = schema.get("minProperties")
    if isinstance(min_properties, int) and len(value) < min_properties:
        errors.append((path, f"must have at least {min_properties} properties"))

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for field, field_schema in properties.items():
            if field in value and isinstance(field_schema, dict):
                errors.extend(_validate_value(value[field], field_schema, root_schema, f"{path}.{field}"))
        additional = schema.get("additionalProperties", True)
        extra_fields = sorted(set(value) - set(properties))
        if additional is False:
            for field in extra_fields:
                errors.append((f"{path}.{field}", "additional property is not allowed"))
        elif isinstance(additional, dict):
            for field in extra_fields:
                errors.extend(_validate_value(value[field], additional, root_schema, f"{path}.{field}"))
    return errors


def _validate_array(
    value: list[Any],
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str,
) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        errors.append((path, f"must contain at least {min_items} item(s)"))

    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            errors.extend(_validate_value(item, item_schema, root_schema, f"{path}[{index}]"))
    return errors


def _resolve_ref(ref: str, root_schema: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported schema ref: {ref}")
    target: Any = root_schema
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or part not in target:
            raise ValueError(f"unresolved schema ref: {ref}")
        target = target[part]
    if not isinstance(target, dict):
        raise ValueError(f"schema ref does not resolve to an object: {ref}")
    return target


def _matches_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_matches_type(value, item) for item in expected)
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return _is_number(value)
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_text(expected: Any) -> str:
    if isinstance(expected, list):
        return " or ".join(str(item) for item in expected)
    return str(expected)


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__
