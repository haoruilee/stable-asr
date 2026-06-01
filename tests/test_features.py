import pytest

from stable_asr.data.manifest import load_manifest
from stable_asr.scenarios.synthetic_turn import write_synthetic_turn_manifest
from stable_asr.train.features import AUDIO_FEATURE_NAMES, FEATURE_NAMES, feature_names, records_to_features

pytest.importorskip("torch")


def test_audio_features_from_synthetic_wav(tmp_path) -> None:
    manifest = tmp_path / "synthetic.jsonl"
    write_synthetic_turn_manifest(manifest, episodes=3, seed=2, write_audio=True)
    records = load_manifest(manifest)

    features = records_to_features(records, feature_source="audio", audio_root=tmp_path)

    assert features.shape == (3, len(AUDIO_FEATURE_NAMES))


def test_manifest_metadata_feature_source_alias() -> None:
    records = load_manifest("examples/data/turn_demo.jsonl")

    features = records_to_features(records, feature_source="manifest_metadata_v0")

    assert features.shape == (4, len(FEATURE_NAMES))
    assert feature_names("manifest_metadata_v0") == FEATURE_NAMES
