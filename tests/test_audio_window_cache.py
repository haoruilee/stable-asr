from pathlib import Path
import importlib

import pytest

from stable_asr.data.audio import synth_tone, write_wav_mono
from stable_asr.data.audio_window_cache import benchmark_audio_window_formats, materialize_audio_windows
from stable_asr.data.manifest import TurnManifestRecord

pytest.importorskip("pyarrow")


def test_audio_window_cache_benchmarks_source_and_parquet(tmp_path: Path) -> None:
    records = _records(tmp_path)

    formats = ["source_wav", "parquet"]
    if _has_pylance():
        formats.append("lance")

    rows = benchmark_audio_window_formats(
        records,
        output_dir=tmp_path / "bench",
        formats=formats,
        sample_count=8,
        seed=2,
    )

    assert [row.format for row in rows] == formats
    assert all(row.records == 3 for row in rows)
    assert all(row.sample_count == 8 for row in rows)
    assert all(row.samples_per_second > 0 for row in rows)
    assert rows[0].speedup_vs_source_wav == pytest.approx(1.0)
    cached_rows = [row for row in rows if row.format in {"parquet", "lance"}]
    assert all(row.correctness_sample_count == 8 for row in cached_rows)
    assert all(row.allclose_to_source for row in cached_rows)
    assert all(row.max_abs_error_vs_source <= row.correctness_tolerance for row in cached_rows)
    assert (tmp_path / "bench" / "audio_windows.parquet").exists()


def test_audio_window_cache_materializes_lance_when_available(tmp_path: Path) -> None:
    if not _has_pylance():
        pytest.skip("pylance is not installed")
    records = _records(tmp_path)
    output = tmp_path / "audio_windows.lance"

    count = materialize_audio_windows(records, output, format="lance")

    assert count == 3
    assert output.exists()


def _records(tmp_path: Path) -> list[TurnManifestRecord]:
    records = []
    for index in range(3):
        path = tmp_path / f"sample_{index}.wav"
        write_wav_mono(
            path,
            synth_tone(0.5 + index * 0.1, sample_rate=16000, frequency=220 + index * 30),
            sample_rate=16000,
        )
        records.append(
            TurnManifestRecord(
                id=f"sample_{index}",
                audio=str(path),
                sample_rate=16000,
                start=0.0,
                end=0.25,
                turn_label="complete",
                action_label="take_turn",
                assistant_speaking=False,
                overlap=False,
                language="en",
                source="unit",
            )
        )
    return records


def _has_pylance() -> bool:
    spec = importlib.util.find_spec("lance")
    if spec is None:
        return False
    import lance

    return hasattr(lance, "dataset") and hasattr(lance, "write_dataset")
