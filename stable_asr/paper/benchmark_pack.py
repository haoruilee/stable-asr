"""Build self-contained benchmark starter packs for external contributors."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.paper.suites import (
    benchmark_suite_markdown,
    load_benchmark_suite,
    validate_benchmark_suite,
    write_benchmark_suite_json,
)
from stable_asr.resources import resolve_platform_path
from stable_asr.schema_validation import validate_schema_file
from stable_asr.schemas import (
    load_schema_registry,
    schema_registry_markdown,
    validate_schema_registry,
    write_schema_registry_json,
)


BENCHMARK_PACK_VERSION = "benchmark_pack_v0"

SAMPLE_FILES = {
    "turn_manifest": {
        "source": "examples/data/turn_demo.jsonl",
        "dest": "data/turn_demo.jsonl",
        "schema_id": "stable_asr.turn_manifest_record.v0",
    },
    "turn_predictions": {
        "source": "tests/fixtures/turn_predictions_sample.jsonl",
        "dest": "data/turn_predictions_sample.jsonl",
        "schema_id": "stable_asr.turn_prediction_record.v0",
    },
    "streaming_asr": {
        "source": "tests/fixtures/streaming_asr_sample.jsonl",
        "dest": "data/streaming_asr_sample.jsonl",
        "schema_id": "stable_asr.streaming_asr_record.v0",
    },
}


@dataclass(frozen=True)
class BenchmarkPackReport:
    output_dir: str
    files: dict[str, str]
    commands: list[str]
    schema_registry_ok: bool
    benchmark_suite_ok: bool
    sample_validations: dict[str, bool]

    @property
    def ok(self) -> bool:
        return self.schema_registry_ok and self.benchmark_suite_ok and all(self.sample_validations.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "version": BENCHMARK_PACK_VERSION,
            "output_dir": self.output_dir,
            "files": self.files,
            "commands": self.commands,
            "schema_registry_ok": self.schema_registry_ok,
            "benchmark_suite_ok": self.benchmark_suite_ok,
            "sample_validations": self.sample_validations,
        }

    def to_markdown(self) -> str:
        file_rows = [{"name": name, "path": path} for name, path in sorted(self.files.items())]
        validation_rows = [
            {"sample": name, "ok": "yes" if ok else "no"}
            for name, ok in sorted(self.sample_validations.items())
        ]
        command_block = "\n".join(self.commands)
        return "\n".join(
            [
                "# Stable-ASR Benchmark Starter Pack",
                "",
                f"- status: `{'OK' if self.ok else 'FAILED'}`",
                f"- version: `{BENCHMARK_PACK_VERSION}`",
                f"- output_dir: `{self.output_dir}`",
                "",
                "## What Is Included",
                "",
                dict_table(file_rows),
                "",
                "## Sample Validation",
                "",
                dict_table(validation_rows),
                "",
                "## Run From This Directory",
                "",
                "```bash",
                command_block,
                "```",
                "",
            ]
        )


def build_benchmark_pack(
    output_dir: str | Path,
    *,
    suite_path: str | Path | None = None,
    schema_registry_path: str | Path | None = None,
) -> BenchmarkPackReport:
    """Write a starter benchmark pack with schemas, suite metadata, and fixtures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_schema_registry(schema_registry_path)
    registry_validation = validate_schema_registry(registry)
    suite = load_benchmark_suite(suite_path)
    suite_validation = validate_benchmark_suite(suite)
    if not registry_validation.ok:
        raise ValueError(registry_validation.to_text())
    if not suite_validation.ok:
        raise ValueError(suite_validation.to_text())

    files: dict[str, str] = {}
    _write_json(output_dir / "manifest.json", {"version": BENCHMARK_PACK_VERSION, "status": "building"})

    files["schema_registry_json"] = write_schema_registry_json(
        output_dir / "configs" / "schema_registry.json",
        registry,
    )
    files["schema_registry_markdown"] = _write_text(
        output_dir / "configs" / "SCHEMAS.md",
        schema_registry_markdown(registry),
    )
    files["benchmark_suite_json"] = write_benchmark_suite_json(
        output_dir / "configs" / "benchmark_suite.json",
        suite,
    )
    files["benchmark_suite_markdown"] = _write_text(
        output_dir / "configs" / "BENCHMARK_SUITE.md",
        benchmark_suite_markdown(suite),
    )

    for name, spec in SAMPLE_FILES.items():
        source = resolve_platform_path(str(spec["source"]))
        if not source.exists():
            raise FileNotFoundError(f"sample source not found: {spec['source']}")
        destination = output_dir / str(spec["dest"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[name] = str(destination)

    commands = _starter_commands()
    files["commands_markdown"] = _write_text(output_dir / "COMMANDS.md", _commands_markdown(commands))
    files["commands_script"] = _write_text(output_dir / "commands.sh", _commands_script(commands))
    files["data_readme"] = _write_text(output_dir / "data" / "README.md", _data_readme())

    sample_validations: dict[str, bool] = {}
    for name, spec in SAMPLE_FILES.items():
        report = validate_schema_file(
            output_dir / str(spec["dest"]),
            schema_id=str(spec["schema_id"]),
            registry_path=output_dir / "configs" / "schema_registry.json",
        )
        sample_validations[name] = report.ok

    report = BenchmarkPackReport(
        output_dir=str(output_dir),
        files=files,
        commands=commands,
        schema_registry_ok=registry_validation.ok,
        benchmark_suite_ok=suite_validation.ok,
        sample_validations=sample_validations,
    )
    files["readme"] = _write_text(output_dir / "README.md", report.to_markdown())
    _write_json(output_dir / "manifest.json", report.to_dict())
    return report


def _starter_commands() -> list[str]:
    return [
        "stable-asr validate-schema-file --input data/turn_demo.jsonl --schema-id stable_asr.turn_manifest_record.v0 --registry configs/schema_registry.json",
        "stable-asr validate-schema-file --input data/turn_predictions_sample.jsonl --schema-id stable_asr.turn_prediction_record.v0 --registry configs/schema_registry.json",
        "stable-asr turn-submission --dataset data/turn_demo.jsonl --predictions data/turn_predictions_sample.jsonl --system oracle_fixture --output-dir submissions/turn_oracle --suite configs/benchmark_suite.json",
        "stable-asr validate-schema-file --input data/streaming_asr_sample.jsonl --schema-id stable_asr.streaming_asr_record.v0 --registry configs/schema_registry.json",
        "stable-asr streaming-submission --input data/streaming_asr_sample.jsonl --system streaming_fixture --slice adapter --output-dir submissions/streaming_fixture --suite configs/benchmark_suite.json",
        "stable-asr leaderboard-validate --input submissions/turn_oracle/leaderboard.jsonl --suite configs/benchmark_suite.json --output submissions/turn_oracle/LEADERBOARD_VALIDATION.md",
        "stable-asr leaderboard-report --input submissions/streaming_fixture/leaderboard.jsonl --suite configs/benchmark_suite.json --output submissions/streaming_fixture/LEADERBOARD_REPORT.md",
    ]


def _commands_markdown(commands: list[str]) -> str:
    return "\n".join(
        [
            "# Stable-ASR Benchmark Starter Commands",
            "",
            "Run these commands from the benchmark pack root.",
            "",
            "```bash",
            "\n".join(commands),
            "```",
            "",
        ]
    )


def _commands_script(commands: list[str]) -> str:
    return "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", "", *commands, ""])


def _data_readme() -> str:
    return "\n".join(
        [
            "# Benchmark Fixture Data",
            "",
            "- `turn_demo.jsonl`: canonical turn manifest records.",
            "- `turn_predictions_sample.jsonl`: sample external turn predictions.",
            "- `streaming_asr_sample.jsonl`: normalized streaming ASR trace records.",
            "",
            "These files are smoke fixtures. Replace them with real benchmark inputs before publishing final results.",
            "",
        ]
    )


def _write_text(path: str | Path, text: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return str(path)


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)
