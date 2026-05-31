from pathlib import Path

import pytest

from stable_asr.data.registry import convert_turn_manifest, load_turn_records

pytest.importorskip("pyarrow")


def test_parquet_roundtrip(tmp_path: Path) -> None:
    dest = tmp_path / "turn_demo.parquet"

    count = convert_turn_manifest("examples/data/turn_demo.jsonl", dest)
    records = load_turn_records(dest)

    assert count == 4
    assert len(records) == 4
    assert records[0].metadata["pause_ms"] == 900

