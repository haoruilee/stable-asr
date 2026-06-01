import json

import pytest

from stable_asr.paper.handoff import final_handoff_template
from stable_asr.references import (
    asr_collections_source_manifest,
    load_asr_collections,
    reference_workqueue_assignments,
    reference_workqueue_from_registries,
)
from stable_asr.schema_validation import validate_schema_file
from stable_asr.schemas import (
    get_schema_entry,
    load_schema_registry,
    schema_entry_markdown,
    schema_registry_markdown,
    validate_schema_registry,
)


def test_schema_registry_validates_checked_in_contracts() -> None:
    registry = load_schema_registry()
    report = validate_schema_registry(registry)

    assert report.ok
    assert registry["id"] == "stable_asr_schema_registry_v0"
    assert len(registry["schemas"]) >= 8


def test_schema_registry_exposes_core_contracts() -> None:
    registry = load_schema_registry()
    expected = {
        "stable_asr.turn_manifest_record.v0",
        "stable_asr.asr_manifest_record.v0",
        "stable_asr.streaming_asr_record.v0",
        "stable_asr.turn_prediction_record.v0",
        "stable_asr.leaderboard_row.v0",
        "stable_asr.model_registry.v0",
        "stable_asr.nanoturn_train_config.v0",
        "stable_asr.final_input_collection.v0",
        "stable_asr.reference_source_manifest.v0",
        "stable_asr.reference_workqueue.v0",
        "stable_asr.reference_assignments.v0",
        "stable_asr.final_handoff.v0",
    }

    ids = {entry["id"] for entry in registry["schemas"]}
    assert expected.issubset(ids)

    turn_schema = get_schema_entry(registry, "stable_asr.turn_manifest_record.v0")["schema"]
    assert "turn_label" in turn_schema["required"]
    assert turn_schema["properties"]["turn_label"]["enum"] == ["backchannel", "complete", "incomplete", "wait"]


def test_schema_registry_validates_final_handoff_template(tmp_path) -> None:
    handoff = tmp_path / "FINAL_INPUT_HANDOFF.json"
    handoff.write_text(json.dumps(final_handoff_template(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = validate_schema_file(handoff, schema_id="stable_asr.final_handoff.v0")

    assert report.ok
    assert report.records == 1


def test_schema_registry_validates_reference_source_manifest(tmp_path) -> None:
    manifest = tmp_path / "asr_collection_source_manifest.json"
    manifest.write_text(
        json.dumps(asr_collections_source_manifest(load_asr_collections()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = validate_schema_file(manifest, schema_id="stable_asr.reference_source_manifest.v0")

    assert report.ok
    assert report.records == 1


def test_schema_registry_validates_reference_workqueue(tmp_path) -> None:
    workqueue = tmp_path / "reference_workqueue.json"
    workqueue.write_text(
        json.dumps(reference_workqueue_from_registries(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = validate_schema_file(workqueue, schema_id="stable_asr.reference_workqueue.v0")

    assert report.ok
    assert report.records == 1


def test_schema_registry_validates_reference_assignments(tmp_path) -> None:
    assignments = tmp_path / "reference_assignments.json"
    assignments.write_text(
        json.dumps(
            reference_workqueue_assignments(reference_workqueue_from_registries()),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_schema_file(assignments, schema_id="stable_asr.reference_assignments.v0")

    assert report.ok
    assert report.records == 1


def test_schema_registry_validates_nanoturn_configs() -> None:
    for path in ("configs/nanoturn_pico.json", "configs/nanoturn_nano.json"):
        report = validate_schema_file(path, schema_id="stable_asr.nanoturn_train_config.v0")

        assert report.ok
        assert report.records == 1


def test_schema_registry_markdown_rendering() -> None:
    registry = load_schema_registry()
    markdown = schema_registry_markdown(registry)

    assert "Stable-ASR Schema Registry" in markdown
    assert "stable_asr.streaming_asr_record.v0" in markdown
    assert "stable-asr schema-registry --validate-only" in markdown

    entry_markdown = schema_entry_markdown(get_schema_entry(registry, "stable_asr.leaderboard_row.v0"))
    assert "Leaderboard Row" in entry_markdown
    assert '"higher_is_better"' in entry_markdown


def test_schema_registry_validation_reports_bad_registry() -> None:
    report = validate_schema_registry({"id": "bad", "schemas": [{"id": "x", "schema": {"type": "array"}}]})

    assert not report.ok
    assert any("missing top-level key: version" in error for error in report.errors)
    assert any("unsupported format" in error for error in report.errors)
    assert any("must be an object schema" in error for error in report.errors)

    with pytest.raises(KeyError):
        get_schema_entry(load_schema_registry(), "missing")
