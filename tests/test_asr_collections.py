from stable_asr.references import asr_collections_markdown, load_asr_collections, validate_asr_collections


def test_asr_collections_registry_validates() -> None:
    registry = load_asr_collections()
    report = validate_asr_collections(registry)

    assert report.ok
    assert len(registry["entries"]) >= 10
    assert any(entry["id"] == "lhotse" for entry in registry["entries"])
    assert any(entry["priority"] == "p0" for entry in registry["entries"])


def test_asr_collections_markdown_mentions_core_projects() -> None:
    markdown = asr_collections_markdown(load_asr_collections())

    assert "# Stable-ASR Reference Collections" in markdown
    assert "Kaldi" in markdown
    assert "FunASR" in markdown
    assert "sherpa-onnx" in markdown
