"""Structured final input handoff templates and audits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.paper.final_inputs import load_final_input_collections


FINAL_HANDOFF_VERSION = "stable_asr_final_handoff_v0"


@dataclass(frozen=True)
class FinalHandoffAuditReport:
    ok: bool
    path: str
    entries: int
    checked_paths: list[str]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "path": self.path,
            "entries": self.entries,
            "checked_paths": self.checked_paths,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def to_text(self) -> str:
        lines = [
            f"final_handoff_audit: {'OK' if self.ok else 'FAILED'}",
            f"path: {self.path}",
            f"entries: {self.entries}",
            f"checked_paths: {len(self.checked_paths)}",
            f"errors: {len(self.errors)}",
            f"warnings: {len(self.warnings)}",
        ]
        lines.extend(f"- ERROR {error}" for error in self.errors)
        lines.extend(f"- WARNING {warning}" for warning in self.warnings)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        rows = [
            {"check": "entries", "value": self.entries},
            {"check": "checked_paths", "value": len(self.checked_paths)},
            {"check": "errors", "value": len(self.errors)},
            {"check": "warnings", "value": len(self.warnings)},
        ]
        lines = [
            "# Stable-ASR Final Handoff Audit",
            "",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- handoff: `{self.path}`",
            "",
            dict_table(rows),
            "",
            "## Errors",
            "",
        ]
        lines.extend(f"- `{error}`" for error in self.errors) if self.errors else lines.append("- none")
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in self.warnings) if self.warnings else lines.append("- none")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class FinalHandoffChecksumReport:
    ok: bool
    input_path: str
    output_path: str
    entries: int
    checksums: int
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "entries": self.entries,
            "checksums": self.checksums,
            "errors": self.errors,
        }

    def to_text(self) -> str:
        lines = [
            f"final_handoff_checksums: {'OK' if self.ok else 'FAILED'}",
            f"input_path: {self.input_path}",
            f"output_path: {self.output_path}",
            f"entries: {self.entries}",
            f"checksums: {self.checksums}",
        ]
        lines.extend(f"- ERROR {error}" for error in self.errors)
        return "\n".join(lines)


def final_handoff_template(registry: dict[str, Any] | None = None) -> dict[str, object]:
    """Return a JSON handoff template covering every final input collection."""

    registry = registry or load_final_input_collections()
    entries = []
    for collection in registry.get("collections", []):
        entries.append(
            {
                "collection_id": collection["id"],
                "owner": "",
                "staged_paths": list(collection.get("required_paths", [])),
                "source_urls": list(collection.get("source_urls", [])),
                "license_or_consent_notes": "",
                "commands_run": [],
                "verification_outputs": [],
                "checksums": [
                    {
                        "path": path,
                        "sha256": "",
                        "bytes": None,
                    }
                    for path in collection.get("required_paths", [])
                ],
                "known_gaps": [],
            }
        )
    return {
        "version": FINAL_HANDOFF_VERSION,
        "description": "Fill this after staging real final-scale inputs. Do not use empty fields as release evidence.",
        "entries": entries,
    }


def final_handoff_schema_markdown() -> str:
    rows = [
        {"field": "collection_id", "required": "yes", "meaning": "id from configs/final/input_collections.json"},
        {"field": "owner", "required": "yes", "meaning": "person accountable for the staged inputs"},
        {"field": "staged_paths", "required": "yes", "meaning": "real files or directories staged in the final run tree"},
        {"field": "source_urls", "required": "no", "meaning": "upstream corpus, model, or project URLs"},
        {"field": "license_or_consent_notes", "required": "yes", "meaning": "license review, consent, or redistribution notes"},
        {"field": "commands_run", "required": "yes", "meaning": "commands used to produce or normalize the staged artifacts"},
        {"field": "verification_outputs", "required": "yes", "meaning": "verification commands, logs, reports, or issue links"},
        {
            "field": "checksums",
            "required": "final gates",
            "meaning": "sha256 and byte size for staged files; required when auditing final-ready handoffs",
        },
        {"field": "known_gaps", "required": "no", "meaning": "remaining limitations that block final release claims"},
    ]
    return "\n".join(
        [
            "# Stable-ASR Final Handoff Schema",
            "",
            "Use this schema when handing off real final-scale corpora, VoiceWorld recordings, external predictions, ASR exports, NanoTurn artifacts, or paper bundles.",
            "",
            dict_table(rows),
            "",
        ]
    )


def populate_final_handoff_checksums(
    path: str | Path,
    *,
    repo_root: str | Path = ".",
    output: str | Path | None = None,
) -> FinalHandoffChecksumReport:
    """Fill checksum entries from staged files in a final input handoff JSON."""

    handoff_path = Path(path)
    output_path = Path(output) if output else handoff_path
    root = Path(repo_root)
    errors: list[str] = []
    checksum_count = 0

    try:
        payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return FinalHandoffChecksumReport(False, str(handoff_path), str(output_path), 0, 0, [str(exc)])

    if not isinstance(payload, dict):
        return FinalHandoffChecksumReport(False, str(handoff_path), str(output_path), 0, 0, ["handoff must be a JSON object"])
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        return FinalHandoffChecksumReport(False, str(handoff_path), str(output_path), 0, 0, ["entries must be a non-empty list"])

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entry {index} must be an object")
            continue
        label = str(entry.get("collection_id") or index)
        staged_paths = [str(item) for item in entry.get("staged_paths", []) if isinstance(item, str) and item.strip()]
        if not staged_paths:
            errors.append(f"{label}:staged_paths:missing")
            continue
        checksum_entries: dict[str, dict[str, object]] = {}
        for staged in staged_paths:
            resolved = _resolve_staged_path(staged, root=root)
            if not resolved.exists():
                errors.append(f"{label}:staged_path_missing:{staged}")
                continue
            for file_path in _iter_checksum_files(resolved):
                display_path = _display_checksum_path(file_path, root=root)
                checksum_entries[display_path] = {
                    "path": display_path,
                    "sha256": _sha256_file(file_path),
                    "bytes": file_path.stat().st_size,
                }
        if checksum_entries:
            checksums = [checksum_entries[key] for key in sorted(checksum_entries)]
            entry["checksums"] = checksums
            checksum_count += len(checksums)
        else:
            errors.append(f"{label}:checksums:no_files")

    if errors:
        return FinalHandoffChecksumReport(False, str(handoff_path), str(output_path), len(entries), checksum_count, errors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return FinalHandoffChecksumReport(True, str(handoff_path), str(output_path), len(entries), checksum_count, [])


def audit_final_handoff(
    path: str | Path,
    *,
    repo_root: str | Path = ".",
    require_checksums: bool = False,
) -> FinalHandoffAuditReport:
    handoff_path = Path(path)
    root = Path(repo_root)
    errors: list[str] = []
    warnings: list[str] = []
    checked_paths: list[str] = []

    try:
        payload = json.loads(handoff_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return FinalHandoffAuditReport(
            ok=False,
            path=str(handoff_path),
            entries=0,
            checked_paths=[],
            errors=[str(exc)],
            warnings=[],
        )

    if not isinstance(payload, dict):
        return FinalHandoffAuditReport(False, str(handoff_path), 0, [], ["handoff must be a JSON object"], [])
    if payload.get("version") != FINAL_HANDOFF_VERSION:
        errors.append(f"version must be {FINAL_HANDOFF_VERSION}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("entries must be a non-empty list")
        entries = []

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entry {index} must be an object")
            continue
        _audit_entry(
            index,
            entry,
            root=root,
            require_checksums=require_checksums,
            checked_paths=checked_paths,
            errors=errors,
            warnings=warnings,
        )

    return FinalHandoffAuditReport(
        ok=not errors,
        path=str(handoff_path),
        entries=len(entries),
        checked_paths=checked_paths,
        errors=errors,
        warnings=warnings,
    )


def _audit_entry(
    index: int,
    entry: dict[str, Any],
    *,
    root: Path,
    require_checksums: bool,
    checked_paths: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    label = str(entry.get("collection_id") or index)
    for field in ("collection_id", "owner", "license_or_consent_notes"):
        if not isinstance(entry.get(field), str) or not str(entry.get(field)).strip():
            errors.append(f"{label}:{field}:missing")
    for field in ("staged_paths", "commands_run", "verification_outputs"):
        value = entry.get(field)
        if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
            errors.append(f"{label}:{field}:missing")
    source_urls = entry.get("source_urls", [])
    if not isinstance(source_urls, list) or not all(isinstance(item, str) for item in source_urls):
        errors.append(f"{label}:source_urls:invalid")

    staged_paths = [str(path) for path in entry.get("staged_paths", []) if isinstance(path, str) and path.strip()]
    for staged in staged_paths:
        resolved = _resolve_staged_path(staged, root=root)
        checked_paths.append(staged)
        if not resolved.exists():
            errors.append(f"{label}:staged_path_missing:{staged}")
    _audit_checksums(
        label,
        entry.get("checksums", []),
        root=root,
        staged_paths=set(staged_paths),
        require_checksums=require_checksums,
        errors=errors,
        warnings=warnings,
    )


def _audit_checksums(
    label: str,
    checksums: Any,
    *,
    root: Path,
    staged_paths: set[str],
    require_checksums: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    if checksums in (None, []):
        _checksum_issue(
            f"{label}:checksums:missing",
            require_checksums=require_checksums,
            errors=errors,
            warnings=warnings,
        )
        return
    if not isinstance(checksums, list):
        errors.append(f"{label}:checksums:invalid")
        return
    for checksum in checksums:
        if not isinstance(checksum, dict):
            errors.append(f"{label}:checksum:invalid")
            continue
        path = checksum.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"{label}:checksum_path:missing")
            continue
        if staged_paths and not _checksum_path_is_staged(path, staged_paths=staged_paths, root=root):
            warnings.append(f"{label}:checksum_path_not_staged:{path}")
        resolved = _resolve_staged_path(path, root=root)
        if not resolved.exists():
            errors.append(f"{label}:checksum_path_missing:{path}")
            continue
        if resolved.is_dir():
            warnings.append(f"{label}:checksum_path_is_directory:{path}")
            continue
        expected_sha = checksum.get("sha256")
        if not isinstance(expected_sha, str) or not expected_sha:
            _checksum_issue(
                f"{label}:sha256_missing:{path}",
                require_checksums=require_checksums,
                errors=errors,
                warnings=warnings,
            )
        elif _sha256_file(resolved) != expected_sha:
            errors.append(f"{label}:sha256_mismatch:{path}")
        expected_bytes = checksum.get("bytes")
        if expected_bytes is None:
            _checksum_issue(
                f"{label}:bytes_missing:{path}",
                require_checksums=require_checksums,
                errors=errors,
                warnings=warnings,
            )
            continue
        try:
            bytes_value = int(expected_bytes)
        except (TypeError, ValueError):
            errors.append(f"{label}:bytes_invalid:{path}")
        else:
            if bytes_value != resolved.stat().st_size:
                errors.append(f"{label}:bytes_mismatch:{path}")


def _checksum_issue(
    message: str,
    *,
    require_checksums: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    if require_checksums:
        errors.append(message)
    else:
        warnings.append(message)


def _resolve_staged_path(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _iter_checksum_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def _display_checksum_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _checksum_path_is_staged(path: str, *, staged_paths: set[str], root: Path) -> bool:
    if path in staged_paths:
        return True
    candidate = _resolve_staged_path(path, root=root).resolve(strict=False)
    for staged in staged_paths:
        resolved_staged = _resolve_staged_path(staged, root=root).resolve(strict=False)
        if candidate == resolved_staged:
            return True
        if resolved_staged.exists() and resolved_staged.is_dir():
            try:
                candidate.relative_to(resolved_staged)
            except ValueError:
                continue
            return True
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
