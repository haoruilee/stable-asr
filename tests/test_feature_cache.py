from pathlib import Path
import importlib

import pytest

from stable_asr.data.manifest import load_manifest
from stable_asr.scenarios.synthetic_turn import write_synthetic_turn_manifest
from stable_asr.train.feature_cache import (
    benchmark_train_feature_cache,
    load_logmel_feature_cache,
    write_logmel_feature_cache,
)
from stable_asr.train.features import records_to_features

pytest.importorskip("torch")
pytest.importorskip("pyarrow")


def test_logmel_feature_cache_round_trips_parquet(tmp_path: Path) -> None:
    manifest = tmp_path / "synthetic.jsonl"
    write_synthetic_turn_manifest(manifest, episodes=4, seed=5, write_audio=True)
    records = load_manifest(manifest)
    cache = tmp_path / "features.parquet"

    count = write_logmel_feature_cache(records, cache, format="parquet", audio_root=tmp_path)
    cached = load_logmel_feature_cache(cache, record_ids=[record.id for record in records])
    direct = records_to_features(records, feature_source="audio", audio_root=tmp_path)

    assert count == 4
    assert cached.shape == direct.shape
    assert cached.allclose(direct, atol=1e-6)


def test_records_to_features_uses_feature_cache(tmp_path: Path) -> None:
    manifest = tmp_path / "synthetic.jsonl"
    write_synthetic_turn_manifest(manifest, episodes=3, seed=6, write_audio=True)
    records = load_manifest(manifest)
    cache = tmp_path / "features.parquet"

    features = records_to_features(
        records,
        feature_source="audio",
        audio_root=tmp_path,
        feature_cache=cache,
        feature_cache_format="parquet",
    )

    assert features.shape == (3, 32)
    assert cache.exists()


def test_benchmark_train_feature_cache_reports_speedups(tmp_path: Path) -> None:
    manifest = tmp_path / "synthetic.jsonl"
    write_synthetic_turn_manifest(manifest, episodes=4, seed=7, write_audio=True)
    records = load_manifest(manifest)
    formats = ["source_audio", "source_audio_file_cache", "parquet"]
    if _has_pylance():
        formats.append("lance")

    rows = benchmark_train_feature_cache(
        records,
        output_dir=tmp_path / "bench",
        formats=formats,
        sample_count=8,
        seed=3,
        audio_root=tmp_path,
    )

    assert [row.format for row in rows] == formats
    assert all(row.samples_per_second > 0 for row in rows)
    assert rows[0].speedup_vs_source_audio == pytest.approx(1.0)


def _has_pylance() -> bool:
    spec = importlib.util.find_spec("lance")
    if spec is None:
        return False
    import lance

    return hasattr(lance, "dataset") and hasattr(lance, "write_dataset")

