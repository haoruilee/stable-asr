"""Repository-level parity audit against the stable-worldmodel platform shape."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.resources import resolve_platform_path


DEFAULT_PLATFORM_PARITY_PATH = Path("configs/platform/stable_worldmodel_parity.json")
STABLE_ASR_COMMAND_RE = re.compile(r"^stable-asr\s+([a-z0-9][a-z0-9-]*)")


@dataclass(frozen=True)
class PlatformParityValidation:
    ok: bool
    errors: list[str]

    def to_text(self) -> str:
        if self.ok:
            return "platform_parity: OK"
        return "platform_parity: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class PlatformParityItemCheck:
    item_id: str
    stable_worldmodel_reference: str
    required_paths: int
    required_commands: int
    required_markers: int
    missing_paths: list[str]
    missing_commands: list[str]
    missing_markers: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_paths and not self.missing_commands and not self.missing_markers

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "ok": self.ok,
            "stable_worldmodel_reference": self.stable_worldmodel_reference,
            "required_paths": self.required_paths,
            "required_commands": self.required_commands,
            "required_markers": self.required_markers,
            "missing_paths": self.missing_paths,
            "missing_commands": self.missing_commands,
            "missing_markers": self.missing_markers,
        }


@dataclass(frozen=True)
class PlatformParityReport:
    ok: bool
    repo_root: str
    registry: dict[str, Any]
    checks: list[PlatformParityItemCheck]

    @property
    def missing_count(self) -> int:
        return sum(len(check.missing_paths) + len(check.missing_commands) + len(check.missing_markers) for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "repo_root": self.repo_root,
            "registry_id": self.registry.get("id"),
            "missing_count": self.missing_count,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_text(self) -> str:
        lines = [
            f"platform_parity_audit: {'OK' if self.ok else 'MISSING'}",
            f"registry: {self.registry.get('id', '')}",
            f"missing_count: {self.missing_count}",
        ]
        for check in self.checks:
            status = "OK" if check.ok else "MISSING"
            detail = "covered" if check.ok else ", ".join(
                [*check.missing_paths, *check.missing_commands, *check.missing_markers]
            )
            lines.append(f"- {status} {check.item_id}: {detail}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        rows = [
            {
                "item": check.item_id,
                "status": "OK" if check.ok else "MISSING",
                "paths": check.required_paths - len(check.missing_paths),
                "commands": check.required_commands - len(check.missing_commands),
                "markers": check.required_markers - len(check.missing_markers),
                "missing": len(check.missing_paths) + len(check.missing_commands) + len(check.missing_markers),
            }
            for check in self.checks
        ]
        lines = [
            f"# {self.registry.get('title', 'Stable-ASR Platform Parity')}",
            "",
            str(self.registry.get("description", "")),
            "",
            f"- status: `{'OK' if self.ok else 'MISSING'}`",
            f"- registry id: `{self.registry.get('id', '')}`",
            f"- reviewed_at: `{self.registry.get('reviewed_at', '')}`",
            f"- source_reference: `{self.registry.get('source_reference', '')}`",
            f"- missing_count: `{self.missing_count}`",
            "",
            "## Summary",
            "",
            dict_table(rows),
            "",
            "## Checks",
            "",
        ]
        for check in self.checks:
            lines.extend(_check_markdown(check))
        return "\n".join(lines)


def load_platform_parity(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = resolve_platform_path(Path(path) if path else DEFAULT_PLATFORM_PARITY_PATH)
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("platform parity registry must be a JSON object")
    return payload


def validate_platform_parity(registry: dict[str, Any]) -> PlatformParityValidation:
    errors: list[str] = []
    for key in ("id", "version", "reviewed_at", "title", "description", "source_reference", "items"):
        if key not in registry:
            errors.append(f"missing top-level key: {key}")
    items = registry.get("items")
    if not isinstance(items, list) or not items:
        errors.append("items must be a non-empty list")
        return PlatformParityValidation(ok=False, errors=errors)

    seen: set[str] = set()
    required = {"id", "stable_worldmodel_reference", "required_paths", "required_commands", "required_markers"}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"item {index} must be an object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"item {index} missing id")
        elif item_id in seen:
            errors.append(f"duplicate item id: {item_id}")
        else:
            seen.add(item_id)
        missing = sorted(required.difference(item))
        if missing:
            errors.append(f"item {item_id or index} missing: {', '.join(missing)}")
        for key in ("required_paths", "required_commands", "required_markers"):
            value = item.get(key)
            if not isinstance(value, list):
                errors.append(f"item {item_id or index} {key} must be a list")
        for marker_index, marker in enumerate(item.get("required_markers", [])):
            if not isinstance(marker, dict):
                errors.append(f"item {item_id or index} marker {marker_index} must be an object")
                continue
            if not isinstance(marker.get("path"), str) or not marker.get("path"):
                errors.append(f"item {item_id or index} marker {marker_index} missing path")
            contains = marker.get("contains")
            if not isinstance(contains, list) or not contains or not all(isinstance(value, str) for value in contains):
                errors.append(f"item {item_id or index} marker {marker_index} contains must be a non-empty string list")
    return PlatformParityValidation(ok=not errors, errors=errors)


def audit_platform_parity(
    registry: dict[str, Any] | None = None,
    *,
    repo_root: str | Path = ".",
) -> PlatformParityReport:
    registry = registry or load_platform_parity()
    validation = validate_platform_parity(registry)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    root = Path(repo_root)
    known_commands = _known_cli_subcommands()
    checks = [
        _audit_item(item, repo_root=root, known_commands=known_commands)
        for item in registry["items"]
    ]
    return PlatformParityReport(
        ok=all(check.ok for check in checks),
        repo_root=str(root),
        registry=registry,
        checks=checks,
    )


def _audit_item(item: dict[str, Any], *, repo_root: Path, known_commands: set[str]) -> PlatformParityItemCheck:
    missing_paths = [
        str(relative)
        for relative in item.get("required_paths", [])
        if not _resolve_repo_path(repo_root, str(relative)).exists()
    ]
    missing_commands = [
        str(command)
        for command in item.get("required_commands", [])
        if not _command_is_known(str(command), known_commands=known_commands)
    ]
    missing_markers: list[str] = []
    for marker in item.get("required_markers", []):
        marker_path = str(marker["path"])
        resolved = _resolve_repo_path(repo_root, marker_path)
        if not resolved.exists():
            missing_markers.append(f"{marker_path}:missing")
            continue
        text = resolved.read_text(encoding="utf-8")
        for needle in marker["contains"]:
            if str(needle) not in text:
                missing_markers.append(f"{marker_path}:{needle}")
    return PlatformParityItemCheck(
        item_id=str(item["id"]),
        stable_worldmodel_reference=str(item["stable_worldmodel_reference"]),
        required_paths=len(item.get("required_paths", [])),
        required_commands=len(item.get("required_commands", [])),
        required_markers=sum(len(marker.get("contains", [])) for marker in item.get("required_markers", [])),
        missing_paths=missing_paths,
        missing_commands=missing_commands,
        missing_markers=missing_markers,
    )


def _known_cli_subcommands() -> set[str]:
    cli_path = resolve_platform_path("stable_asr/cli.py")
    if not cli_path.exists():
        return set()
    text = cli_path.read_text(encoding="utf-8")
    return set(re.findall(r"subparsers\.add_parser\(\s*['\"]([^'\"]+)['\"]", text))


def _command_is_known(command: str, *, known_commands: set[str]) -> bool:
    match = STABLE_ASR_COMMAND_RE.match(command.strip())
    return bool(match and match.group(1) in known_commands)


def _resolve_repo_path(repo_root: Path, relative: str) -> Path:
    repo_path = repo_root / relative
    if repo_path.exists():
        return repo_path
    return resolve_platform_path(relative)


def _check_markdown(check: PlatformParityItemCheck) -> list[str]:
    lines = [
        f"### {check.item_id}",
        "",
        f"- status: `{'OK' if check.ok else 'MISSING'}`",
        f"- stable_worldmodel_reference: {check.stable_worldmodel_reference}",
        f"- required_paths: `{check.required_paths}`",
        f"- required_commands: `{check.required_commands}`",
        f"- required_markers: `{check.required_markers}`",
        "",
    ]
    if check.ok:
        lines.append("- covered")
    else:
        if check.missing_paths:
            lines.extend(["Missing paths:", ""])
            lines.extend(f"- `{path}`" for path in check.missing_paths)
        if check.missing_commands:
            lines.extend(["", "Missing commands:", ""])
            lines.extend(f"- `{command}`" for command in check.missing_commands)
        if check.missing_markers:
            lines.extend(["", "Missing markers:", ""])
            lines.extend(f"- `{marker}`" for marker in check.missing_markers)
    lines.append("")
    return lines
