import json
import sys
from pathlib import Path

import pytest

from stable_asr.streaming.command_compare import (
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


def test_command_adapters_from_config_rejects_missing_output() -> None:
    with pytest.raises(ValueError, match="missing output"):
        command_adapters_from_config({"adapters": [{"name": "bad", "command": "echo hi"}]})


def test_load_asr_command_config_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_asr_command_config(path)
