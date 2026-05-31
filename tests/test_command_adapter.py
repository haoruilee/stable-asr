import sys
from pathlib import Path

import pytest

from stable_asr.models.adapters.command import CommandStreamingASRAdapter
from stable_asr.streaming.metrics import evaluate_streaming_records


def test_command_streaming_asr_adapter_runs_external_command(tmp_path: Path) -> None:
    script = tmp_path / "copy_transcript.py"
    output = tmp_path / "command_output.jsonl"
    script.write_text(
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
    adapter = CommandStreamingASRAdapter(
        name="copy_fixture",
        command=[sys.executable, str(script), "tests/fixtures/streaming_asr_sample.jsonl", "{output}"],
        output_path=output,
        timeout_sec=10.0,
    )

    records = adapter.load_records()
    report = evaluate_streaming_records(records)

    assert output.exists()
    assert len(records) == 2
    assert report.records == 2


def test_command_streaming_asr_adapter_reports_failed_command(tmp_path: Path) -> None:
    adapter = CommandStreamingASRAdapter(
        name="bad",
        command=[sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(2)"],
        output_path=tmp_path / "missing.jsonl",
        timeout_sec=10.0,
    )

    with pytest.raises(RuntimeError, match="exit code 2"):
        adapter.load_records()
