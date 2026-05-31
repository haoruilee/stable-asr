"""Config-driven comparison for command-backed ASR adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_asr.models.adapters.command import CommandStreamingASRAdapter
from stable_asr.streaming.compare import StreamingASRComparisonReport, compare_streaming_adapters


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


def _resolve_path(value: str | None, *, base: Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or base is None:
        return path
    if str(value).startswith("."):
        return base / path
    return path
