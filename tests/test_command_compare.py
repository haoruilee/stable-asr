import json
import sys
from pathlib import Path

import pytest

from stable_asr.streaming.command_compare import (
    audit_asr_command_config,
    command_adapters_from_config,
    compare_asr_commands_from_config,
    load_asr_command_config,
)


def _copy_script(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import shutil",
                "import sys",
                "shutil.copyfile(sys.argv[1], sys.argv[2])",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_compare_asr_commands_from_config(tmp_path: Path) -> None:
    script = tmp_path / "copy.py"
    _copy_script(script)
    config = {
        "adapters": [
            {
                "name": "balanced_cmd",
                "command": [sys.executable, str(script), "tests/fixtures/streaming_asr_sample.jsonl", "{output}"],
                "output": str(tmp_path / "balanced.jsonl"),
            },
            {
                "name": "fast_cmd",
                "command": [
                    sys.executable,
                    str(script),
                    "tests/fixtures/streaming_asr_fast_unstable_sample.jsonl",
                    "{output}",
                ],
                "output": str(tmp_path / "fast.jsonl"),
            },
        ]
    }
    config_path = tmp_path / "commands.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = compare_asr_commands_from_config(config_path)
    rows = report.to_dict()["rows"]

    assert len(rows) == 2
    assert rows[0]["adapter"] == "balanced_cmd"
    assert rows[1]["adapter"] == "fast_cmd"
    assert rows[1]["wer"] > rows[0]["wer"]
    assert "balanced_cmd" in report.to_markdown()


def test_compare_asr_commands_expands_input_manifest_template(tmp_path: Path) -> None:
    script = tmp_path / "copy.py"
    _copy_script(script)
    config = {
        "input_manifest": "tests/fixtures/streaming_asr_sample.jsonl",
        "adapters": [
            {
                "name": "templated_cmd",
                "command": [sys.executable, str(script), "{input_manifest}", "{output}"],
                "output": str(tmp_path / "templated.jsonl"),
            }
        ],
    }
    config_path = tmp_path / "commands.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = compare_asr_commands_from_config(config_path)

    assert report.rows[0].adapter == "templated_cmd"
    assert report.rows[0].report.records == 2


def test_audit_asr_command_config_accepts_ready_config(tmp_path: Path) -> None:
    script = tmp_path / "copy.py"
    _copy_script(script)
    manifest = tmp_path / "asr_eval.jsonl"
    manifest.write_text(
        '{"id":"utt1","audio":"audio/utt1.wav","sample_rate":16000,"text":"hello","language":"en","source":"unit"}\n',
        encoding="utf-8",
    )
    required = tmp_path / "raw.jsonl"
    required.write_text('{"id":"utt1","text":"hello"}\n', encoding="utf-8")
    config = {
        "input_manifest": str(manifest),
        "adapters": [
            {
                "name": "cmd_a",
                "command": [sys.executable, str(script), "{input_manifest}", "{output}"],
                "output": str(tmp_path / "a.jsonl"),
                "required_inputs": [str(required)],
            },
            {
                "name": "cmd_b",
                "command": [sys.executable, str(script), "{input_manifest}", "{output}"],
                "output": str(tmp_path / "b.jsonl"),
            },
        ],
    }
    config_path = tmp_path / "commands.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = audit_asr_command_config(
        config_path,
        repo_root=tmp_path,
        min_adapters=2,
        require_input_manifest=True,
    )

    assert report.ok
    assert report.input_records == 1
    assert "asr_command_config_audit: READY" in report.to_text()


def test_audit_asr_command_config_rejects_missing_input_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    config_path.write_text(
        json.dumps(
            {
                "adapters": [
                    {
                        "name": "cmd",
                        "command": [sys.executable, "missing.py", "{output}"],
                        "output": "runs/out.jsonl",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_asr_command_config(
        config_path,
        repo_root=tmp_path,
        min_adapters=2,
        require_input_manifest=True,
    )

    assert not report.ok
    assert "input_manifest must be a non-empty string" in report.to_text()
    assert "at least 2 adapters are required" in report.to_text()


def test_command_adapters_from_config_rejects_missing_output() -> None:
    with pytest.raises(ValueError, match="missing output"):
        command_adapters_from_config({"adapters": [{"name": "bad", "command": "echo hi"}]})


def test_load_asr_command_config_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_asr_command_config(path)
