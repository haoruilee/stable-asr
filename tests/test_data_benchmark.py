from pathlib import Path
import importlib

import pytest

from stable_asr.data.benchmark import benchmark_data_formats
from stable_asr.data.registry import load_turn_records

pytest.importorskip("pyarrow")


def test_benchmark_data_formats(tmp_path: Path) -> None:
    records = load_turn_records("examples/data/turn_demo.jsonl")
    formats = ["jsonl", "parquet"]
    if _has_pylance():
        formats.append("lance")
    rows = benchmark_data_formats(records, output_dir=tmp_path, formats=formats, sample_count=6, seed=3)

    assert [row.format for row in rows] == formats
    assert all(row.records == 4 for row in rows)
    assert all(row.size_bytes > 0 for row in rows)
    assert all(row.sample_count == 6 for row in rows)
    assert all(row.sample_seconds > 0.0 for row in rows)
    assert all(row.samples_per_second > 0.0 for row in rows)


def _has_pylance() -> bool:
    spec = importlib.util.find_spec("lance")
    if spec is None:
        return False
    import lance

    return hasattr(lance, "dataset") and hasattr(lance, "write_dataset")
