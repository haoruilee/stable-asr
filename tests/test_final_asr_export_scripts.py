import subprocess
import sys
from pathlib import Path


def test_final_asr_export_scripts_convert_precomputed_outputs(tmp_path: Path) -> None:
    manifest = tmp_path / "asr.jsonl"
    manifest.write_text(
        (
            '{"id":"utt1","audio":"audio/utt1.wav","sample_rate":16000,'
            '"text":"hello world","language":"en","source":"unit","duration":1.2}\n'
        ),
        encoding="utf-8",
    )
    whisper_raw = tmp_path / "whisper_raw.jsonl"
    whisper_raw.write_text(
        '{"id":"utt1","text":"hello world","duration":1.2,"processing_time":0.2}\n',
        encoding="utf-8",
    )
    funasr_raw = tmp_path / "funasr_raw.jsonl"
    funasr_raw.write_text(
        '{"key":"utt1","text":"hello world","duration":1.2,"processing_time":0.2}\n',
        encoding="utf-8",
    )
    qwen3_raw = tmp_path / "qwen3_raw.jsonl"
    qwen3_raw.write_text(
        '{"id":"utt1","text":"hello world","duration":1.2,"runtime_ms":200}\n',
        encoding="utf-8",
    )
    firered_raw = tmp_path / "firered_raw.jsonl"
    firered_raw.write_text(
        '{"utt_id":"utt1","text":"hello world","duration":1.2,"processing_time":0.2}\n',
        encoding="utf-8",
    )
    whisper_output = tmp_path / "whisper.jsonl"
    funasr_output = tmp_path / "funasr.jsonl"
    qwen3_output = tmp_path / "qwen3.jsonl"
    firered_output = tmp_path / "firered.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/export_whisper_streaming.py",
            "--manifest",
            str(manifest),
            "--raw",
            str(whisper_raw),
            "--output",
            str(whisper_output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/export_funasr_streaming.py",
            "--manifest",
            str(manifest),
            "--raw",
            str(funasr_raw),
            "--output",
            str(funasr_output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    for schema, raw, output in (
        ("whisper", whisper_raw, whisper_output),
        ("funasr", funasr_raw, funasr_output),
        ("qwen3_asr", qwen3_raw, qwen3_output),
        ("firered_asr2s", firered_raw, firered_output),
    ):
        subprocess.run(
            [
                sys.executable,
                "scripts/export_streaming_transcript.py",
                "--schema",
                schema,
                "--manifest",
                str(manifest),
                "--raw",
                str(raw),
                "--output",
                str(output),
            ],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )

    assert whisper_output.read_text(encoding="utf-8").count("\n") == 1
    assert funasr_output.read_text(encoding="utf-8").count("\n") == 1
    assert qwen3_output.read_text(encoding="utf-8").count("\n") == 1
    assert firered_output.read_text(encoding="utf-8").count("\n") == 1
