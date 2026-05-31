from pathlib import Path

import pytest

from stable_asr.data.manifest import load_manifest
from stable_asr.data.recipes import prepare_voiceworld_manifest, prepare_voiceworld_manifest_rows


def test_prepare_voiceworld_manifest_rows_preserves_factors() -> None:
    records = prepare_voiceworld_manifest_rows(
        [
            {
                "id": "vw1",
                "audio": "normal.wav",
                "text": "what is the weather",
                "scenario": "normal_question",
                "turn_label": "complete",
                "action_label": "take_turn",
                "assistant_speaking": "false",
                "overlap": "false",
                "start_ms": "200",
                "duration_ms": "1200",
                "snr_db": "15",
                "reverb": "small_room",
                "speaking_rate": "1.1",
            }
        ],
        audio_root="audio",
        default_language="en",
    )

    assert len(records) == 1
    record = records[0]
    assert record.audio == "audio/normal.wav"
    assert record.start == pytest.approx(0.2)
    assert record.end == pytest.approx(1.4)
    assert record.metadata["snr_db"] == 15
    assert record.metadata["reverb"] == "small_room"
    assert record.metadata["speaking_rate"] == 1.1


def test_prepare_voiceworld_manifest_from_tsv(tmp_path: Path) -> None:
    metadata = tmp_path / "voiceworld.tsv"
    output = tmp_path / "voiceworld.jsonl"
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "ambient.wav").write_bytes(b"")
    metadata.write_text(
        (
            "id\taudio\ttext\tscenario\tturn_label\taction_label\tassistant_speaking\toverlap\t"
            "start\tend\tsnr_db\treverb\toverlap_offset_ms\tnetwork_jitter_ms\t"
            "farfield_distance_m\tcode_switch_ratio\taccent\n"
            "vw2\tambient.wav\tbackground speech\tambient_speech\twait\tignore\tfalse\ttrue\t"
            "0.0\t1.0\t5\troom\t250\t20\t2.5\t0.0\tstandard\n"
        ),
        encoding="utf-8",
    )

    records = prepare_voiceworld_manifest(
        metadata,
        output,
        audio_root=audio_root,
        default_language="en",
    )

    loaded = load_manifest(output)
    assert len(records) == 1
    assert len(loaded) == 1
    assert loaded[0].scenario == "ambient_speech"
    assert loaded[0].action_label == "ignore"
    assert loaded[0].overlap
    assert loaded[0].metadata["farfield_distance_m"] == 2.5
    assert loaded[0].metadata["source_row_index"] == 0


def test_prepare_voiceworld_manifest_requires_scenario() -> None:
    with pytest.raises(ValueError, match="scenario"):
        prepare_voiceworld_manifest_rows(
            [
                {
                    "id": "vw3",
                    "audio": "a.wav",
                    "turn_label": "complete",
                    "action_label": "take_turn",
                    "end": 1.0,
                }
            ]
        )
