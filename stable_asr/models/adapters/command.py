"""Command-backed streaming ASR adapter.

This adapter is intentionally dependency-light: external systems such as
Whisper, FunASR, WeNet, or local research scripts can be evaluated as long as a
command writes normalized Stable-ASR streaming transcript JSONL.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from stable_asr.models.adapters.transcript import load_streaming_transcript_jsonl
from stable_asr.streaming.types import StreamingASRRecord


@dataclass(frozen=True)
class CommandStreamingASRAdapter:
    """Run an external command that writes a transcript JSONL, then load it."""

    name: str
    command: str | Sequence[str]
    output_path: str | Path
    cwd: str | Path | None = None
    timeout_sec: float = 300.0
    env: Mapping[str, str] | None = field(default=None, repr=False)

    @property
    def path(self) -> str:
        return str(self.output_path)

    def load_records(self) -> list[StreamingASRRecord]:
        self.run()
        return load_streaming_transcript_jsonl(self.output_path)

    def run(self) -> subprocess.CompletedProcess[str]:
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args = self.command_args()
        env = None if self.env is None else {**os.environ, **dict(self.env)}
        result = subprocess.run(
            args,
            cwd=str(self.cwd) if self.cwd is not None else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ASR command failed with exit code {result.returncode}: {_command_excerpt(result)}"
            )
        if not output_path.exists():
            raise RuntimeError(f"ASR command did not write expected output: {output_path}")
        return result

    def command_args(self) -> list[str]:
        if isinstance(self.command, str):
            args = shlex.split(self.command)
        else:
            args = [str(part) for part in self.command]
        if not args:
            raise ValueError("command must not be empty")
        return [part.replace("{output}", str(self.output_path)) for part in args]


def command_streaming_asr_adapter(
    name: str,
    command: str | Sequence[str],
    output_path: str | Path,
    *,
    cwd: str | Path | None = None,
    timeout_sec: float = 300.0,
) -> CommandStreamingASRAdapter:
    return CommandStreamingASRAdapter(
        name=name,
        command=command,
        output_path=output_path,
        cwd=cwd,
        timeout_sec=timeout_sec,
    )


def _command_excerpt(result: subprocess.CompletedProcess[str]) -> str:
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    detail = stderr or stdout or "no output"
    return detail[:1000]
