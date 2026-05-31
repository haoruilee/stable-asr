import pytest

from stable_asr.data.manifest import load_manifest
from stable_asr.scenarios.synthetic_turn import write_synthetic_turn_manifest
from stable_asr.train.features import AUDIO_FEATURE_NAMES, records_to_features

pytest.importorskip("torch")


def test_audio_features_from_synthetic_wav(tmp_path) -> None:
    manifest = tmp_path / "synthetic.jsonl"
    write_synthetic_turn_manifest(manifest, episodes=3, seed=2, write_audio=True)
    records = load_manifest(manifest)

    features = records_to_features(records, feature_source="audio", audio_root=tmp_path)

    assert features.shape == (3, len(AUDIO_FEATURE_NAMES))

