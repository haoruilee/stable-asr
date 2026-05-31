from stable_asr.data.manifest import load_manifest, validate_manifest
from stable_asr.scenarios import SCENARIO_NAMES, generate_synthetic_turn_records, write_synthetic_turn_manifest


def test_generate_synthetic_turn_records_is_deterministic() -> None:
    first = generate_synthetic_turn_records(episodes=len(SCENARIO_NAMES), seed=123)
    second = generate_synthetic_turn_records(episodes=len(SCENARIO_NAMES), seed=123)

    assert [record.to_dict() for record in first] == [record.to_dict() for record in second]
    assert {record.scenario for record in first} == set(SCENARIO_NAMES)
    assert next(record for record in first if record.scenario == "code_switching").language == "zh_en"
    assert next(record for record in first if record.scenario == "noisy_farfield").metadata["farfield_distance_m"] > 1.0


def test_write_synthetic_turn_manifest(tmp_path) -> None:
    path = tmp_path / "turn_synthetic.jsonl"
    write_synthetic_turn_manifest(path, episodes=6, seed=1)

    report = validate_manifest(path)
    records = load_manifest(path)

    assert report.ok
    assert report.records == 6
    assert records[0].metadata["snr_db"] in {-12, -8, -5, 0, 5, 10, 20}
    assert "accent" in records[0].metadata


def test_write_synthetic_turn_manifest_with_audio(tmp_path) -> None:
    path = tmp_path / "turn_synthetic.jsonl"
    records = write_synthetic_turn_manifest(path, episodes=2, seed=1, write_audio=True)

    assert records[0].audio.endswith(".wav")
    assert (tmp_path / records[0].audio).exists()
