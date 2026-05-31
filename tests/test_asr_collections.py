from stable_asr.models.adapters import load_adapter_registry
from stable_asr.references import (
    asr_collections_markdown,
    audit_asr_collection_coverage,
    load_asr_collections,
    validate_asr_collections,
)


def test_asr_collections_registry_validates() -> None:
    registry = load_asr_collections()
    report = validate_asr_collections(registry)

    assert report.ok
    assert len(registry["entries"]) >= 10
    assert any(entry["id"] == "lhotse" for entry in registry["entries"])
    assert any(entry["priority"] == "p0" for entry in registry["entries"])
    assert {"firered_asr2s", "qwen3_asr", "whisper_cpp"}.issubset({entry["id"] for entry in registry["entries"]})


def test_asr_collections_markdown_mentions_core_projects() -> None:
    markdown = asr_collections_markdown(load_asr_collections())

    assert "# Stable-ASR Reference Collections" in markdown
    assert "Kaldi" in markdown
    assert "FunASR" in markdown
    assert "Qwen3-ASR" in markdown
    assert "FireRedASR2S" in markdown
    assert "sherpa-onnx" in markdown


def test_asr_collection_coverage_requires_p0_references() -> None:
    report = audit_asr_collection_coverage(load_asr_collections(), load_adapter_registry())

    assert report.ok
    required = {check.reference_id: check for check in report.checks if check.required}
    assert {
        "firered_asr2s",
        "funasr",
        "lhotse",
        "qwen3_asr",
        "sherpa_onnx",
        "wenet",
        "whisper",
        "whisper_cpp",
    }.issubset(required)
    assert all(check.covered for check in required.values())


def test_asr_collection_coverage_covers_p0_and_p1_references() -> None:
    report = audit_asr_collection_coverage(
        load_asr_collections(),
        load_adapter_registry(),
        required_priorities=("p0", "p1"),
    )

    assert report.ok
    required = {check.reference_id: check for check in report.checks if check.required}
    assert {
        "espnet",
        "huggingface_transformers_asr",
        "icefall",
        "kaldi",
        "moonshine",
        "nemo",
        "sensevoice",
        "speechbrain",
        "whisperx",
    }.issubset(required)
    assert all(check.covered for check in required.values())


def test_asr_collection_coverage_can_surface_missing_required_reference() -> None:
    registry = load_adapter_registry()
    registry["adapters"] = [
        adapter
        for adapter in registry["adapters"]
        if "espnet" not in adapter.get("id", "")
        and "espnet" not in " ".join(adapter.get("related_references", []))
        and "espnet" not in adapter.get("notes", "").lower()
    ]
    report = audit_asr_collection_coverage(
        load_asr_collections(),
        registry,
        required_priorities=("p0", "p1"),
    )

    assert not report.ok
    missing = {check.reference_id for check in report.checks if check.required and not check.covered}
    assert "espnet" in missing
