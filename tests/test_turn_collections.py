from stable_asr.data.sources import load_data_sources
from stable_asr.models.adapters import load_adapter_registry
from stable_asr.references import (
    audit_turn_collection_coverage,
    load_turn_collections,
    turn_collections_acquisition_markdown,
    turn_collections_markdown,
    validate_turn_collections,
)


def test_turn_collections_registry_validates() -> None:
    registry = load_turn_collections()
    report = validate_turn_collections(registry)

    assert report.ok
    assert {"smart_turn", "easy_turn", "full_duplex_bench", "vap"}.issubset(
        {entry["id"] for entry in registry["entries"]}
    )


def test_turn_collections_markdown_mentions_core_projects() -> None:
    markdown = turn_collections_markdown(load_turn_collections())

    assert "# Stable-ASR Turn And Full-Duplex Reference Collections" in markdown
    assert "Smart Turn" in markdown
    assert "Easy Turn" in markdown
    assert "Full-Duplex-Bench" in markdown
    assert "Voice Activity Projection" in markdown


def test_turn_collections_acquisition_markdown_maps_evidence_targets() -> None:
    markdown = turn_collections_acquisition_markdown(load_turn_collections())

    assert "# Stable-ASR Turn Collection Acquisition Plan" in markdown
    assert "runs/final/external/smartturn_raw.jsonl" in markdown
    assert "runs/final/external/smartturn_predictions.jsonl" in markdown
    assert "runs/final/external/easyturn_raw.jsonl" in markdown
    assert "runs/final/external/easyturn_predictions.jsonl" in markdown
    assert "runs/final/external/vap_raw.jsonl" in markdown
    assert "runs/final/external/vap_predictions.jsonl" in markdown
    assert "runs/collections/full_duplex_bench/SCENARIO_BRIDGE.md" in markdown
    assert "Registry presence alone is not evidence" in markdown
    assert "runs/final/smartturn_raw.jsonl" not in markdown
    assert "runs/final/external/smart_turn_raw.jsonl" not in markdown


def test_turn_collection_coverage_requires_p0_references() -> None:
    report = audit_turn_collection_coverage(
        load_turn_collections(),
        load_data_sources(),
        load_adapter_registry(),
    )

    assert report.ok
    required = {check.reference_id: check for check in report.checks if check.required}
    assert {"smart_turn", "easy_turn", "full_duplex_bench", "vap"}.issubset(required)
    assert all(required[key].covered for key in {"smart_turn", "easy_turn", "full_duplex_bench", "vap"})
    assert any("adapter:vap_prediction" in item for item in required["vap"].evidence)


def test_turn_collection_coverage_can_require_p1_references() -> None:
    report = audit_turn_collection_coverage(
        load_turn_collections(),
        load_data_sources(),
        load_adapter_registry(),
        required_priorities=("p0", "p1"),
    )

    assert report.ok
    required = {check.reference_id: check for check in report.checks if check.required}
    assert all(required[key].covered for key in {"pipecat", "silero_vad", "webrtcvad"})
    assert "adapter:pipecat_voice_agent_bridge_template" in required["pipecat"].evidence
    assert "adapter:silero_vad_endpointing_template" in required["silero_vad"].evidence
    assert "adapter:webrtcvad_endpointing_template" in required["webrtcvad"].evidence


def test_turn_collection_coverage_surfaces_missing_required_reference() -> None:
    sources = load_data_sources()
    sources["sources"] = [source for source in sources["sources"] if source.get("id") != "smart_turn"]
    adapters = load_adapter_registry()
    adapters["adapters"] = [
        adapter
        for adapter in adapters["adapters"]
        if "smart" not in adapter.get("id", "").lower()
        and "smart" not in adapter.get("title", "").lower()
        and "smart" not in adapter.get("notes", "").lower()
    ]

    report = audit_turn_collection_coverage(load_turn_collections(), sources, adapters)

    assert not report.ok
    missing = {check.reference_id for check in report.checks if check.required and not check.covered}
    assert "smart_turn" in missing
