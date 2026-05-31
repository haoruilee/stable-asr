"""Build self-contained external ASR adapter starter packs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.asr_manifest import validate_asr_manifest
from stable_asr.eval.report import dict_table
from stable_asr.models.adapters.registry import (
    adapter_registry_markdown,
    load_adapter_registry,
    validate_adapter_registry,
    write_adapter_registry_json,
)
from stable_asr.references.collections import (
    asr_collections_markdown,
    audit_asr_collection_coverage,
    load_asr_collections,
    validate_asr_collections,
    write_asr_collections_json,
)
from stable_asr.resources import resolve_platform_path
from stable_asr.schema_validation import validate_schema_file
from stable_asr.schemas import load_schema_registry, validate_schema_registry, write_schema_registry_json
from stable_asr.streaming.command_compare import audit_asr_command_config


ADAPTER_PACK_VERSION = "adapter_pack_v0"


@dataclass(frozen=True)
class AdapterPackReport:
    output_dir: str
    files: dict[str, str]
    commands: list[str]
    adapter_registry_ok: bool
    asr_collections_ok: bool
    schema_registry_ok: bool
    asr_manifest_ok: bool
    streaming_fixture_ok: bool
    command_config_ok: bool
    reference_coverage_ok: bool

    @property
    def ok(self) -> bool:
        return all(
            [
                self.adapter_registry_ok,
                self.asr_collections_ok,
                self.schema_registry_ok,
                self.asr_manifest_ok,
                self.streaming_fixture_ok,
                self.command_config_ok,
                self.reference_coverage_ok,
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "version": ADAPTER_PACK_VERSION,
            "output_dir": self.output_dir,
            "files": self.files,
            "commands": self.commands,
            "adapter_registry_ok": self.adapter_registry_ok,
            "asr_collections_ok": self.asr_collections_ok,
            "schema_registry_ok": self.schema_registry_ok,
            "asr_manifest_ok": self.asr_manifest_ok,
            "streaming_fixture_ok": self.streaming_fixture_ok,
            "command_config_ok": self.command_config_ok,
            "reference_coverage_ok": self.reference_coverage_ok,
        }

    def to_markdown(self) -> str:
        file_rows = [{"name": name, "path": path} for name, path in sorted(self.files.items())]
        status_rows = [
            {"check": "adapter_registry", "ok": _yes_no(self.adapter_registry_ok)},
            {"check": "asr_collections", "ok": _yes_no(self.asr_collections_ok)},
            {"check": "schema_registry", "ok": _yes_no(self.schema_registry_ok)},
            {"check": "asr_manifest", "ok": _yes_no(self.asr_manifest_ok)},
            {"check": "streaming_fixture", "ok": _yes_no(self.streaming_fixture_ok)},
            {"check": "command_config", "ok": _yes_no(self.command_config_ok)},
            {"check": "reference_coverage", "ok": _yes_no(self.reference_coverage_ok)},
        ]
        return "\n".join(
            [
                "# Stable-ASR External ASR Adapter Pack",
                "",
                f"- status: `{'OK' if self.ok else 'FAILED'}`",
                f"- version: `{ADAPTER_PACK_VERSION}`",
                f"- output_dir: `{self.output_dir}`",
                "",
                "## Included Files",
                "",
                dict_table(file_rows),
                "",
                "## Readiness Checks",
                "",
                dict_table(status_rows),
                "",
                "## Run From This Directory",
                "",
                "```bash",
                "\n".join(self.commands),
                "```",
                "",
            ]
        )


def build_adapter_pack(
    output_dir: str | Path,
    *,
    adapter_registry_path: str | Path | None = None,
    asr_collections_path: str | Path | None = None,
    schema_registry_path: str | Path | None = None,
) -> AdapterPackReport:
    """Write a starter pack for wrapping external ASR systems via commands."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_registry = load_adapter_registry(adapter_registry_path)
    adapter_validation = validate_adapter_registry(adapter_registry)
    asr_collections = load_asr_collections(asr_collections_path)
    collections_validation = validate_asr_collections(asr_collections)
    schema_registry = load_schema_registry(schema_registry_path)
    schema_validation = validate_schema_registry(schema_registry)
    if not adapter_validation.ok:
        raise ValueError(adapter_validation.to_text())
    if not collections_validation.ok:
        raise ValueError(collections_validation.to_text())
    if not schema_validation.ok:
        raise ValueError(schema_validation.to_text())

    files: dict[str, str] = {}
    _write_json(output_dir / "manifest.json", {"version": ADAPTER_PACK_VERSION, "status": "building"})

    files["adapter_registry_json"] = write_adapter_registry_json(
        output_dir / "configs" / "adapter_registry.json",
        adapter_registry,
    )
    files["adapter_registry_markdown"] = _write_text(
        output_dir / "configs" / "ADAPTERS.md",
        adapter_registry_markdown(adapter_registry),
    )
    files["asr_collections_json"] = write_asr_collections_json(
        output_dir / "configs" / "asr_collections.json",
        asr_collections,
    )
    files["asr_collections_markdown"] = _write_text(
        output_dir / "configs" / "ASR_COLLECTIONS.md",
        asr_collections_markdown(asr_collections),
    )
    files["schema_registry_json"] = write_schema_registry_json(
        output_dir / "configs" / "schema_registry.json",
        schema_registry,
    )
    files["command_config"] = _write_json(
        output_dir / "configs" / "asr_command_compare.json",
        _command_config(),
    )
    files["asr_manifest"] = _write_text(
        output_dir / "data" / "asr_eval_manifest.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in _asr_manifest_rows()),
    )

    _copy_fixture("tests/fixtures/streaming_asr_sample.jsonl", output_dir / "data" / "streaming_asr_sample.jsonl")
    _copy_fixture(
        "tests/fixtures/streaming_asr_fast_unstable_sample.jsonl",
        output_dir / "data" / "streaming_asr_fast_unstable_sample.jsonl",
    )
    files["streaming_fixture"] = str(output_dir / "data" / "streaming_asr_sample.jsonl")
    files["streaming_unstable_fixture"] = str(output_dir / "data" / "streaming_asr_fast_unstable_sample.jsonl")
    files["export_template"] = _write_text(
        output_dir / "scripts" / "export_streaming_template.py",
        _export_streaming_template(),
    )
    files["script_readme"] = _write_text(output_dir / "scripts" / "README.md", _scripts_readme())

    commands = _adapter_commands()
    files["commands_markdown"] = _write_text(output_dir / "COMMANDS.md", _commands_markdown(commands))
    files["commands_script"] = _write_text(output_dir / "commands.sh", _commands_script(commands))

    asr_manifest_report = validate_asr_manifest(output_dir / "data" / "asr_eval_manifest.jsonl")
    streaming_report = validate_schema_file(
        output_dir / "data" / "streaming_asr_sample.jsonl",
        schema_id="stable_asr.streaming_asr_record.v0",
        registry_path=output_dir / "configs" / "schema_registry.json",
    )
    command_audit = audit_asr_command_config(
        output_dir / "configs" / "asr_command_compare.json",
        repo_root=output_dir,
        min_adapters=2,
        require_input_manifest=True,
    )
    coverage = audit_asr_collection_coverage(
        asr_collections,
        adapter_registry,
        required_priorities=("p0", "p1"),
    )

    report = AdapterPackReport(
        output_dir=str(output_dir),
        files=files,
        commands=commands,
        adapter_registry_ok=adapter_validation.ok,
        asr_collections_ok=collections_validation.ok,
        schema_registry_ok=schema_validation.ok,
        asr_manifest_ok=asr_manifest_report.ok,
        streaming_fixture_ok=streaming_report.ok,
        command_config_ok=command_audit.ok,
        reference_coverage_ok=coverage.ok,
    )
    files["readme"] = _write_text(output_dir / "README.md", report.to_markdown())
    _write_json(output_dir / "manifest.json", report.to_dict())
    return report


def _command_config() -> dict[str, Any]:
    return {
        "input_manifest": "data/asr_eval_manifest.jsonl",
        "timeout_sec": 30,
        "adapters": [
            {
                "name": "balanced_template",
                "command": [
                    "python3",
                    "scripts/export_streaming_template.py",
                    "--input-manifest",
                    "{input_manifest}",
                    "--fixture",
                    "data/streaming_asr_sample.jsonl",
                    "--output",
                    "{output}",
                ],
                "cwd": "../",
                "output": "../runs/asr_adapter_pack/balanced_template.jsonl",
                "required_inputs": ["data/streaming_asr_sample.jsonl"],
            },
            {
                "name": "fast_unstable_template",
                "command": [
                    "python3",
                    "scripts/export_streaming_template.py",
                    "--input-manifest",
                    "{input_manifest}",
                    "--fixture",
                    "data/streaming_asr_fast_unstable_sample.jsonl",
                    "--output",
                    "{output}",
                ],
                "cwd": "../",
                "output": "../runs/asr_adapter_pack/fast_unstable_template.jsonl",
                "required_inputs": ["data/streaming_asr_fast_unstable_sample.jsonl"],
            },
        ],
    }


def _adapter_commands() -> list[str]:
    return [
        "stable-asr validate-asr-manifest data/asr_eval_manifest.jsonl",
        "stable-asr validate-schema-file --input data/streaming_asr_sample.jsonl --schema-id stable_asr.streaming_asr_record.v0 --registry configs/schema_registry.json",
        "stable-asr adapter-registry --registry configs/adapter_registry.json --validate-only",
        "stable-asr asr-collections --registry configs/asr_collections.json --audit-coverage --require-priority p0 --require-priority p1",
        "stable-asr compare-asr-commands --config configs/asr_command_compare.json --validate-only --require-input-manifest --min-adapters 2 --repo-root .",
        "stable-asr compare-asr-commands --config configs/asr_command_compare.json --report reports/asr_command_compare.md --json-output reports/asr_command_compare.json",
        "stable-asr streaming-submission --input runs/asr_adapter_pack/balanced_template.jsonl --system balanced_template --slice adapter --output-dir submissions/balanced_template",
    ]


def _asr_manifest_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "utt_001",
            "audio": "audio/utt_001.wav",
            "sample_rate": 16000,
            "text": "what is the weather",
            "language": "en",
            "source": "adapter_pack_fixture",
            "duration": 2.0,
            "split": "adapter_demo",
        },
        {
            "id": "utt_002",
            "audio": "audio/utt_002.wav",
            "sample_rate": 16000,
            "text": "turn on the lights",
            "language": "en",
            "source": "adapter_pack_fixture",
            "duration": 2.4,
            "split": "adapter_demo",
        },
    ]


def _export_streaming_template() -> str:
    return "\n".join(
        [
            '"""Template exporter for Stable-ASR command-backed ASR adapters.',
            "",
            "Replace the fixture copy with calls to a real upstream ASR system. The",
            "script must write Stable-ASR StreamingASRRecord JSONL to --output.",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "import argparse",
            "import shutil",
            "from pathlib import Path",
            "",
            "",
            "def main() -> int:",
            "    parser = argparse.ArgumentParser()",
            "    parser.add_argument('--input-manifest', required=True, type=Path)",
            "    parser.add_argument('--fixture', required=True, type=Path)",
            "    parser.add_argument('--output', required=True, type=Path)",
            "    args = parser.parse_args()",
            "",
            "    if not args.input_manifest.exists():",
            "        raise FileNotFoundError(args.input_manifest)",
            "    if not args.fixture.exists():",
            "        raise FileNotFoundError(args.fixture)",
            "    args.output.parent.mkdir(parents=True, exist_ok=True)",
            "    shutil.copyfile(args.fixture, args.output)",
            "    return 0",
            "",
            "",
            "if __name__ == '__main__':",
            "    raise SystemExit(main())",
            "",
        ]
    )


def _scripts_readme() -> str:
    return "\n".join(
        [
            "# Adapter Scripts",
            "",
            "`export_streaming_template.py` is intentionally dependency-light. Replace the fixture copy with calls to Whisper, FunASR, WeNet, NeMo, or another upstream ASR system, but keep the same input/output contract.",
            "",
            "The command must read `--input-manifest` and write Stable-ASR streaming transcript JSONL to `--output`.",
            "",
        ]
    )


def _commands_markdown(commands: list[str]) -> str:
    return "\n".join(
        [
            "# Stable-ASR Adapter Pack Commands",
            "",
            "Run these commands from the adapter pack root.",
            "",
            "```bash",
            "\n".join(commands),
            "```",
            "",
        ]
    )


def _commands_script(commands: list[str]) -> str:
    return "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", "", *commands, ""])


def _copy_fixture(source: str, destination: Path) -> None:
    resolved = resolve_platform_path(source)
    if not resolved.exists():
        raise FileNotFoundError(f"fixture source not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resolved, destination)


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


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
