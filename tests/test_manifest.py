from pathlib import Path

import pytest

from stable_asr.data.manifest import ManifestError, TurnManifestRecord, load_manifest, validate_manifest


FIXTURE = Path("examples/data/turn_demo.jsonl")


def test_load_manifest_fixture() -> None:
    records = load_manifest(FIXTURE)

    assert len(records) == 4
    assert records[0].id == "zh_turn_000001"
    assert records[0].duration == pytest.approx(2.0)


def test_validate_manifest_fixture() -> None:
    report = validate_manifest(FIXTURE)

    assert report.ok
    assert report.records == 4
    assert report.errors == []


def test_rejects_unknown_turn_label() -> None:
    data = {
        "id": "bad",
        "audio": "audio/bad.flac",
        "sample_rate": 16000,
        "start": 0.0,
        "end": 1.0,
        "turn_label": "done",
        "action_label": "take_turn",
        "assistant_speaking": False,
        "overlap": False,
        "language": "zh",
        "source": "test",
    }

    with pytest.raises(ManifestError, match="unknown turn_label"):
        TurnManifestRecord.from_dict(data)


def test_rejects_invalid_time_range() -> None:
    data = {
        "id": "bad",
        "audio": "audio/bad.flac",
        "sample_rate": 16000,
        "start": 1.0,
        "end": 1.0,
        "turn_label": "complete",
        "action_label": "take_turn",
        "assistant_speaking": False,
        "overlap": False,
        "language": "zh",
        "source": "test",
    }

    with pytest.raises(ManifestError, match="end must be greater than start"):
        TurnManifestRecord.from_dict(data)

