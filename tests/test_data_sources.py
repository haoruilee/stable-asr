from pathlib import Path

from stable_asr.data.sources import (
    data_sources_markdown,
    load_data_sources,
    validate_data_sources,
    write_data_sources_json,
)


def test_default_data_sources_validate() -> None:
    registry = load_data_sources()
    config_registry = load_data_sources("configs/datasets/stable_asr_sources.json")

    assert validate_data_sources(registry).ok
    assert validate_data_sources(config_registry).ok
    assert registry["id"] == "stable_asr_sources_v0"
    assert [item["id"] for item in registry["sources"]] == [item["id"] for item in config_registry["sources"]]


def test_data_sources_markdown_and_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "data_sources.json"
    write_data_sources_json(path)
    registry = load_data_sources(path)
    markdown = data_sources_markdown(registry)

    assert "Stable-ASR Data Source Registry" in markdown
    assert "synthetic_voiceworld" in markdown
    assert "librispeech" in markdown


def test_data_sources_validation_rejects_duplicate_id() -> None:
    registry = load_data_sources()
    registry["sources"].append(dict(registry["sources"][0]))

    report = validate_data_sources(registry)

    assert not report.ok
    assert "duplicate source id" in report.to_text()
