from pathlib import Path

from stable_asr.models.adapters import (
    adapter_registry_markdown,
    load_adapter_registry,
    validate_adapter_registry,
    write_adapter_registry_json,
)


def test_default_adapter_registry_matches_config() -> None:
    registry = load_adapter_registry()
    config_registry = load_adapter_registry("configs/adapters/stable_asr_adapters.json")

    assert validate_adapter_registry(registry).ok
    assert validate_adapter_registry(config_registry).ok
    assert registry["id"] == "stable_asr_adapters_v0"
    assert [item["id"] for item in registry["adapters"]] == [item["id"] for item in config_registry["adapters"]]


def test_adapter_registry_markdown_and_json_roundtrip(tmp_path: Path) -> None:
    output = tmp_path / "adapter_registry.json"
    write_adapter_registry_json(output)
    registry = load_adapter_registry(output)
    markdown = adapter_registry_markdown(registry)

    assert "Stable-ASR Adapter Registry" in markdown
    assert "command_streaming_asr" in markdown
    assert "smart_turn_prediction" in markdown
    assert "whisper_transcript" in markdown
    assert "espnet_command_template" in markdown
    assert "hf_transformers_asr_template" in markdown


def test_adapter_registry_validation_rejects_duplicate_id() -> None:
    registry = load_adapter_registry()
    registry["adapters"].append(dict(registry["adapters"][0]))

    report = validate_adapter_registry(registry)

    assert not report.ok
    assert "duplicate adapter id" in report.to_text()
