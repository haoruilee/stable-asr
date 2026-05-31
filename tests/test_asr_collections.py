from datetime import date

from stable_asr.models.adapters import load_adapter_registry
from stable_asr.references import (
    asr_collections_bibtex,
    asr_collections_markdown,
    asr_collections_reference_markdown,
    audit_asr_collection_coverage,
    audit_asr_collection_readiness,
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


def test_asr_collections_reference_markdown_and_bibtex_render() -> None:
    registry = load_asr_collections()
    markdown = asr_collections_reference_markdown(registry)
    bibtex = asr_collections_bibtex(registry)

    assert "# Stable-ASR Paper Reference Notes" in markdown
    assert "stableasr_ref_funasr" in markdown
    assert "adapter planning" in markdown
    assert "@misc{stableasr_ref_funasr" in bibtex
    assert "\\url{https://github.com/modelscope/FunASR}" in bibtex
    assert "@misc{stableasr_ref_lhotse" in bibtex


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


def test_asr_collection_readiness_reports_adapter_and_license_review() -> None:
    report = audit_asr_collection_readiness(load_asr_collections(), load_adapter_registry())

    assert report.ok
    assert report.reviewed_at == "2026-06-01"
    assert any(row.reference_id == "funasr" and row.license_review_needed for row in report.rows)
    assert any(row.reference_id == "whisper" and "adapter:" in ",".join(row.adapter_evidence) for row in report.rows)
    assert "ASR Collection Readiness" in report.to_markdown()
    assert "license_review_needed" in report.to_text()


def test_asr_collection_readiness_can_fail_on_stale_review() -> None:
    report = audit_asr_collection_readiness(
        load_asr_collections(),
        load_adapter_registry(),
        max_review_age_days=1,
        today=date(2026, 6, 5),
    )

    assert not report.ok
    assert report.stale_review
    assert "reference collection review is stale" in report.to_text()
