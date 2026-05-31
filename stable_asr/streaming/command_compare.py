"""Config-driven comparison for command-backed ASR adapters."""

from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.asr_manifest import load_asr_manifest
from stable_asr.models.adapters.command import CommandStreamingASRAdapter
from stable_asr.streaming.compare import StreamingASRComparisonReport, compare_streaming_adapters


@dataclass(frozen=True)
class ASRCommandAdapterCheck:
    name: str
    command: list[str]
    output: str
    cwd: str | None
    program: str
    program_exists: bool
    entrypoint: str | None
    entrypoint_exists: bool | None
    has_output_placeholder: bool
    references_input_manifest: bool
    missing_required_inputs: list[str]
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "command": self.command,
            "output": self.output,
            "cwd": self.cwd,
            "program": self.program,
            "program_exists": self.program_exists,
            "entrypoint": self.entrypoint,
            "entrypoint_exists": self.entrypoint_exists,
            "has_output_placeholder": self.has_output_placeholder,
            "references_input_manifest": self.references_input_manifest,
            "missing_required_inputs": self.missing_required_inputs,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ASRCommandConfigAuditReport:
    ok: bool
    config_path: str
    input_manifest: str | None
    input_manifest_exists: bool
    input_records: int
    min_adapters: int
    adapter_count: int
    adapters: list[ASRCommandAdapterCheck]
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "config_path": self.config_path,
            "input_manifest": self.input_manifest,
            "input_manifest_exists": self.input_manifest_exists,
            "input_records": self.input_records,
            "min_adapters": self.min_adapters,
            "adapter_count": self.adapter_count,
            "adapters": [adapter.to_dict() for adapter in self.adapters],
            "errors": self.errors,
        }

    def to_text(self) -> str:
        lines = [
            f"asr_command_config_audit: {'READY' if self.ok else 'NOT_READY'}",
            f"- config: {self.config_path}",
            f"- input_manifest: {self.input_manifest or ''}",
            f"- input_records: {self.input_records}",
            f"- adapters: {self.adapter_count}/{self.min_adapters}",
        ]
        for adapter in self.adapters:
            status = "OK" if adapter.ok else "FAILED"
            lines.append(f"- {status} {adapter.name}: {adapter.output} ({adapter.detail})")
            for missing in adapter.missing_required_inputs:
                lines.append(f"  - missing_required_input: {missing}")
        if self.errors:
            lines.append("- errors:")
            lines.extend(f"  - {error}" for error in self.errors)
        return "\n".join(lines)


def load_asr_command_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("ASR command config must be a JSON object")
    return payload


def command_adapters_from_config(
    config: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> list[CommandStreamingASRAdapter]:
    adapters = config.get("adapters")
    if not isinstance(adapters, list) or not adapters:
        raise ValueError("ASR command config must include a non-empty adapters list")

    base = Path(base_dir) if base_dir is not None else None
    input_manifest = config.get("input_manifest")
    if input_manifest is not None and not isinstance(input_manifest, str):
        raise ValueError("ASR command config input_manifest must be a string when present")
    result: list[CommandStreamingASRAdapter] = []
    for index, item in enumerate(adapters):
        if not isinstance(item, dict):
            raise ValueError(f"adapter {index} must be a JSON object")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"adapter {index} missing name")
        command = item.get("command")
        if not isinstance(command, (str, list)):
            raise ValueError(f"adapter {name} missing command")
        if isinstance(command, list) and not all(isinstance(part, (str, int, float)) for part in command):
            raise ValueError(f"adapter {name} command list must contain scalar values")
        output = item.get("output", item.get("output_path"))
        if not isinstance(output, str) or not output:
            raise ValueError(f"adapter {name} missing output")
        cwd = item.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError(f"adapter {name} cwd must be a string")
        timeout = float(item.get("timeout_sec", item.get("timeout", config.get("timeout_sec", 300.0))))
        output_path = _resolve_path(output, base=base)
        cwd_path = _resolve_path(cwd, base=base) if cwd is not None else None
        command = _expand_command_template(command, input_manifest=input_manifest)
        result.append(
            CommandStreamingASRAdapter(
                name=name,
                command=command,
                output_path=output_path,
                cwd=cwd_path,
                timeout_sec=timeout,
            )
        )
    return result


def compare_asr_commands_from_config(path: str | Path) -> StreamingASRComparisonReport:
    path = Path(path)
    config = load_asr_command_config(path)
    adapters = command_adapters_from_config(config, base_dir=path.parent)
    return compare_streaming_adapters(adapters)


def audit_asr_command_config(
    path: str | Path,
    *,
    repo_root: str | Path = ".",
    min_adapters: int = 1,
    require_input_manifest: bool = False,
) -> ASRCommandConfigAuditReport:
    """Audit a command-backed ASR comparison config without executing adapters."""

    root = Path(repo_root)
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    errors: list[str] = []
    adapters: list[ASRCommandAdapterCheck] = []
    input_manifest_value: str | None = None
    input_manifest_path: Path | None = None
    input_manifest_exists = False
    input_records = 0

    try:
        config = load_asr_command_config(config_path)
    except (OSError, ValueError) as exc:
        return ASRCommandConfigAuditReport(
            ok=False,
            config_path=str(config_path),
            input_manifest=None,
            input_manifest_exists=False,
            input_records=0,
            min_adapters=min_adapters,
            adapter_count=0,
            adapters=[],
            errors=[str(exc)],
        )

    input_manifest = config.get("input_manifest")
    if isinstance(input_manifest, str) and input_manifest:
        input_manifest_value = input_manifest
        input_manifest_path = _resolve_repo_path(input_manifest, root=root)
        input_manifest_exists = input_manifest_path.exists()
        if input_manifest_exists:
            try:
                input_records = len(load_asr_manifest(input_manifest_path))
                if input_records == 0:
                    errors.append(f"input_manifest is empty: {input_manifest_path}")
            except (OSError, ValueError) as exc:
                errors.append(f"input_manifest is invalid: {exc}")
        elif require_input_manifest:
            errors.append(f"input_manifest is missing: {input_manifest_path}")
    elif require_input_manifest:
        errors.append("input_manifest must be a non-empty string")

    raw_adapters = config.get("adapters")
    if not isinstance(raw_adapters, list) or not raw_adapters:
        errors.append("adapters must be a non-empty list")
        raw_adapters = []
    if len(raw_adapters) < min_adapters:
        errors.append(f"at least {min_adapters} adapters are required")

    seen_names: set[str] = set()
    seen_outputs: set[str] = set()
    for index, item in enumerate(raw_adapters):
        if not isinstance(item, dict):
            errors.append(f"adapter {index} must be a JSON object")
            continue
        check = _audit_adapter(
            item,
            index=index,
            root=root,
            input_manifest=input_manifest_value,
            require_input_manifest=require_input_manifest,
        )
        if check.name in seen_names:
            errors.append(f"duplicate adapter name: {check.name}")
            check = _replace_adapter_detail(check, ok=False, detail="duplicate adapter name")
        seen_names.add(check.name)
        if check.output in seen_outputs:
            errors.append(f"duplicate adapter output: {check.output}")
            check = _replace_adapter_detail(check, ok=False, detail="duplicate adapter output")
        seen_outputs.add(check.output)
        adapters.append(check)

    ok = not errors and all(adapter.ok for adapter in adapters)
    return ASRCommandConfigAuditReport(
        ok=ok,
        config_path=str(config_path),
        input_manifest=str(input_manifest_path) if input_manifest_path is not None else input_manifest_value,
        input_manifest_exists=input_manifest_exists,
        input_records=input_records,
        min_adapters=min_adapters,
        adapter_count=len(adapters),
        adapters=adapters,
        errors=errors,
    )


def _resolve_path(value: str | None, *, base: Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or base is None:
        return path
    if str(value).startswith("."):
        return base / path
    return path


def _expand_command_template(command: str | list[Any], *, input_manifest: str | None) -> str | list[Any]:
    if input_manifest is None:
        return command
    if isinstance(command, str):
        return command.replace("{input_manifest}", input_manifest)
    return [
        str(part).replace("{input_manifest}", input_manifest)
        for part in command
    ]


def _audit_adapter(
    item: dict[str, Any],
    *,
    index: int,
    root: Path,
    input_manifest: str | None,
    require_input_manifest: bool,
) -> ASRCommandAdapterCheck:
    name = str(item.get("name", f"adapter_{index}")).strip()
    output = str(item.get("output", item.get("output_path", ""))).strip()
    command_value = item.get("command")
    command = _command_parts(command_value)
    command_text = " ".join(command)
    program = command[0] if command else ""
    program_exists = _program_exists(program, root=root)
    entrypoint = _python_entrypoint(command)
    entrypoint_exists = None if entrypoint is None else _resolve_repo_path(entrypoint, root=root).exists()
    cwd = item.get("cwd")
    cwd_text = str(cwd) if cwd is not None else None
    required_inputs = _required_inputs(item, root=root)
    missing_required_inputs = [str(path) for path in required_inputs if not path.exists()]
    has_output_placeholder = "{output}" in command_text
    references_input_manifest = True
    if require_input_manifest:
        references_input_manifest = bool(input_manifest) and (
            "{input_manifest}" in command_text or str(input_manifest) in command_text
        )

    issues: list[str] = []
    if not name:
        issues.append("missing name")
    if not output:
        issues.append("missing output")
    if not command:
        issues.append("missing command")
    if command and not has_output_placeholder:
        issues.append("command must contain {output}")
    if command and not program_exists:
        issues.append(f"program not found: {program}")
    if entrypoint is not None and not entrypoint_exists:
        issues.append(f"python entrypoint not found: {entrypoint}")
    if require_input_manifest and not references_input_manifest:
        issues.append("command must reference input_manifest")
    if missing_required_inputs:
        issues.append("missing required adapter inputs")

    detail = "; ".join(issues) if issues else "command config is ready"
    return ASRCommandAdapterCheck(
        name=name,
        command=command,
        output=str(_resolve_repo_path(output, root=root)) if output else "",
        cwd=cwd_text,
        program=program,
        program_exists=program_exists,
        entrypoint=entrypoint,
        entrypoint_exists=entrypoint_exists,
        has_output_placeholder=has_output_placeholder,
        references_input_manifest=references_input_manifest,
        missing_required_inputs=missing_required_inputs,
        ok=not issues,
        detail=detail,
    )


def _replace_adapter_detail(
    check: ASRCommandAdapterCheck,
    *,
    ok: bool,
    detail: str,
) -> ASRCommandAdapterCheck:
    return ASRCommandAdapterCheck(
        name=check.name,
        command=check.command,
        output=check.output,
        cwd=check.cwd,
        program=check.program,
        program_exists=check.program_exists,
        entrypoint=check.entrypoint,
        entrypoint_exists=check.entrypoint_exists,
        has_output_placeholder=check.has_output_placeholder,
        references_input_manifest=check.references_input_manifest,
        missing_required_inputs=check.missing_required_inputs,
        ok=ok,
        detail=detail,
    )


def _command_parts(command: Any) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, list) and all(isinstance(part, (str, int, float)) for part in command):
        return [str(part) for part in command]
    return []


def _program_exists(program: str, *, root: Path) -> bool:
    if not program:
        return False
    path = Path(program)
    if path.is_absolute():
        return path.exists()
    if "/" in program:
        return (root / path).exists()
    return shutil.which(program) is not None


def _python_entrypoint(command: list[str]) -> str | None:
    if len(command) < 2:
        return None
    program = Path(command[0]).name.lower()
    if not (program.startswith("python") or program in {"py"}):
        return None
    candidate = command[1]
    return candidate if candidate.endswith(".py") else None


def _required_inputs(item: dict[str, Any], *, root: Path) -> list[Path]:
    raw = item.get("required_inputs", [])
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    result = []
    for value in raw:
        if isinstance(value, str) and value:
            result.append(_resolve_repo_path(value, root=root))
    return result


def _resolve_repo_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path
