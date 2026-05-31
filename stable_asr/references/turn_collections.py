"""Curated turn-taking and full-duplex reference collection registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.resources import resolve_platform_path


DEFAULT_TURN_COLLECTIONS_PATH = Path("configs/references/turn_collections.json")


@dataclass(frozen=True)
class TurnCollectionsValidation:
    ok: bool
    errors: list[str]

    def to_text(self) -> str:
        if self.ok:
            return "turn_collections: OK"
        return "turn_collections: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class TurnCollectionCoverageCheck:
    reference_id: str
    name: str
    category: str
    priority: str
    required: bool
    covered: bool
    evidence: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "name": self.name,
            "category": self.category,
            "priority": self.priority,
            "required": self.required,
            "covered": self.covered,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TurnCollectionCoverageReport:
    ok: bool
    required_priorities: tuple[str, ...]
    checks: list[TurnCollectionCoverageCheck]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "required_priorities": list(self.required_priorities),
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_markdown(self) -> str:
        rows = [
            {
                "reference": check.reference_id,
                "priority": check.priority,
                "required": "yes" if check.required else "no",
                "covered": "yes" if check.covered else "no",
                "evidence": ", ".join(check.evidence),
            }
            for check in self.checks
        ]
        missing = [check for check in self.checks if check.required and not check.covered]
        lines = [
            "# Turn Collection Coverage",
            "",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- required_priorities: `{', '.join(self.required_priorities)}`",
            f"- missing_required: `{len(missing)}`",
            "",
            dict_table(rows),
        ]
        if missing:
            lines.extend(["", "## Missing Required References", ""])
            lines.extend(f"- `{check.reference_id}` ({check.name})" for check in missing)
        return "\n".join(lines)

    def to_text(self) -> str:
        lines = [
            f"turn_collection_coverage: {'OK' if self.ok else 'FAILED'}",
            f"required_priorities: {', '.join(self.required_priorities)}",
        ]
        for check in self.checks:
            marker = "OK" if check.covered else "MISSING"
            required = "required" if check.required else "optional"
            evidence = ", ".join(check.evidence) if check.evidence else "none"
            lines.append(f"- {marker} {check.reference_id} ({required}; evidence: {evidence})")
        return "\n".join(lines)


def load_turn_collections(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = resolve_platform_path(Path(path) if path else DEFAULT_TURN_COLLECTIONS_PATH)
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("turn collections registry must be a JSON object")
    return payload


def write_turn_collections_json(path: str | Path, registry: dict[str, Any] | None = None) -> str:
    registry = registry or load_turn_collections()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_turn_collections(registry: dict[str, Any]) -> TurnCollectionsValidation:
    errors: list[str] = []
    for key in ("id", "version", "reviewed_at", "title", "entries"):
        if key not in registry:
            errors.append(f"missing top-level key: {key}")

    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("entries must be a non-empty list")
        return TurnCollectionsValidation(ok=False, errors=errors)

    seen: set[str] = set()
    required = {
        "id",
        "name",
        "category",
        "source_url",
        "docs_url",
        "license",
        "focus",
        "reference_use",
        "stable_asr_actions",
        "priority",
    }
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entry {index} must be an object")
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            errors.append(f"entry {index} missing id")
        elif entry_id in seen:
            errors.append(f"duplicate entry id: {entry_id}")
        else:
            seen.add(entry_id)
        missing = sorted(required.difference(entry))
        if missing:
            errors.append(f"entry {entry_id or index} missing: {', '.join(missing)}")
        for url_key in ("source_url", "docs_url"):
            value = entry.get(url_key)
            if not isinstance(value, str) or not value.startswith("https://"):
                errors.append(f"entry {entry_id or index} {url_key} must be an https URL")
        actions = entry.get("stable_asr_actions")
        if not isinstance(actions, list) or not actions or not all(isinstance(item, str) and item for item in actions):
            errors.append(f"entry {entry_id or index} stable_asr_actions must be a non-empty string list")
        if entry.get("priority") not in {"p0", "p1", "p2"}:
            errors.append(f"entry {entry_id or index} priority must be p0, p1, or p2")
    return TurnCollectionsValidation(ok=not errors, errors=errors)


def audit_turn_collection_coverage(
    collections: dict[str, Any],
    data_sources: dict[str, Any],
    adapter_registry: dict[str, Any],
    *,
    required_priorities: tuple[str, ...] = ("p0",),
) -> TurnCollectionCoverageReport:
    validation = validate_turn_collections(collections)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    sources = data_sources.get("sources", [])
    adapters = adapter_registry.get("adapters", [])
    if not isinstance(sources, list):
        raise ValueError("data source registry sources must be a list")
    if not isinstance(adapters, list):
        raise ValueError("adapter registry adapters must be a list")

    checks: list[TurnCollectionCoverageCheck] = []
    for entry in collections["entries"]:
        evidence = _coverage_evidence(entry, sources=sources, adapters=adapters)
        priority = str(entry.get("priority", ""))
        required = priority in required_priorities
        checks.append(
            TurnCollectionCoverageCheck(
                reference_id=str(entry["id"]),
                name=str(entry["name"]),
                category=str(entry["category"]),
                priority=priority,
                required=required,
                covered=bool(evidence),
                evidence=evidence,
            )
        )
    ok = all(check.covered for check in checks if check.required)
    return TurnCollectionCoverageReport(ok=ok, required_priorities=required_priorities, checks=checks)


def turn_collections_markdown(registry: dict[str, Any]) -> str:
    validation = validate_turn_collections(registry)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    rows = [
        {
            "id": entry["id"],
            "category": entry["category"],
            "priority": entry["priority"],
            "project": f"[{entry['name']}]({entry['source_url']})",
            "focus": entry["focus"],
            "stable_asr_actions": ", ".join(entry["stable_asr_actions"]),
        }
        for entry in sorted(registry["entries"], key=lambda item: (item["category"], item["priority"], item["id"]))
    ]
    return "\n".join(
        [
            f"# {registry['title']}",
            "",
            str(registry.get("description", "")),
            "",
            f"- registry id: `{registry['id']}`",
            f"- version: `{registry['version']}`",
            f"- reviewed_at: `{registry['reviewed_at']}`",
            f"- entries: `{len(rows)}`",
            "",
            dict_table(rows),
            "",
            "## Usage Policy",
            "",
            "Use this collection for adapter planning, benchmark attribution, and VoiceWorld coverage review. Do not vendor upstream code, weights, or datasets unless license review explicitly allows it.",
        ]
    )


def turn_collections_acquisition_markdown(registry: dict[str, Any]) -> str:
    validation = validate_turn_collections(registry)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    rows = []
    p0_entries = []
    for entry in sorted(registry["entries"], key=lambda item: (item["priority"], item["category"], item["id"])):
        track = _acquisition_track(entry)
        rows.append(
            {
                "reference": entry["id"],
                "priority": entry["priority"],
                "track": track,
                "license_review": "yes" if str(entry.get("license", "")) == "see_upstream" else "no",
                "evidence_target": _acquisition_evidence_target(entry, track),
                "actions": ", ".join(entry["stable_asr_actions"]),
            }
        )
        if entry["priority"] == "p0":
            p0_entries.append(entry)
    p0_lines = [
        f"1. `{entry['id']}`: {_acquisition_track(entry)}; write `{_acquisition_evidence_target(entry, _acquisition_track(entry))}`."
        for entry in p0_entries
    ]
    return "\n".join(
        [
            "# Stable-ASR Turn Collection Acquisition Plan",
            "",
            "This plan turns turn-taking and full-duplex references into concrete collection work. It is for prediction exports, scenario bridges, license review, and benchmark evidence staging.",
            "",
            f"- registry id: `{registry['id']}`",
            f"- reviewed_at: `{registry['reviewed_at']}`",
            f"- entries: `{len(rows)}`",
            f"- p0_entries: `{len(p0_entries)}`",
            "",
            "## Common Commands",
            "",
            "```bash",
            "stable-asr turn-collections --audit-coverage --require-priority p0 --require-priority p1 --output runs/TURN_COLLECTION_COVERAGE.md",
            "stable-asr turn-collections --format acquisition-markdown --output runs/TURN_COLLECTION_ACQUISITION.md",
            "stable-asr convert-predictions --schema smart_turn --input runs/final/external/smartturn_raw.jsonl --output runs/final/external/smartturn_predictions.jsonl",
            "stable-asr convert-predictions --schema easyturn --input runs/final/external/easyturn_raw.jsonl --output runs/final/external/easyturn_predictions.jsonl",
            "stable-asr convert-predictions --schema vap --input runs/final/external/vap_raw.jsonl --output runs/final/external/vap_predictions.jsonl",
            "stable-asr eval-scenario --dataset runs/final/voiceworld_real.jsonl --checkpoint runs/final/nanoturn/checkpoint.pt --json-output runs/final/reports/scenarios.json",
            "```",
            "",
            "## P0 Acquisition Order",
            "",
            *p0_lines,
            "",
            "## Full Checklist",
            "",
            dict_table(rows),
            "",
            "## Evidence Rule",
            "",
            "A turn/full-duplex reference is useful only after it has a concrete converter, prediction export, scenario bridge, or evaluation artifact. Registry presence alone is not evidence.",
        ]
    )


def _coverage_evidence(entry: dict[str, Any], *, sources: list[Any], adapters: list[Any]) -> list[str]:
    variants = _reference_variants(entry)
    evidence: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        text = _normalized_join(source, "id", "title", "task", "source_type", "stable_asr_entrypoint", "notes")
        if any(variant and variant in text for variant in variants):
            evidence.append(f"source:{source.get('id', '')}")
    for adapter in adapters:
        if not isinstance(adapter, dict):
            continue
        related = adapter.get("related_references", [])
        if isinstance(related, list) and any(_normalize(str(item)) in variants for item in related):
            evidence.append(f"adapter:{adapter.get('id', '')}")
            continue
        text = _normalized_join(adapter, "id", "title", "task", "interface", "entrypoint", "input_schema", "notes")
        if any(variant and variant in text for variant in variants):
            evidence.append(f"adapter:{adapter.get('id', '')}")
    return sorted(set(evidence))


def _reference_variants(entry: dict[str, Any]) -> set[str]:
    reference_id = str(entry.get("id", ""))
    name = str(entry.get("name", ""))
    variants = {
        _normalize(reference_id),
        _normalize(reference_id.replace("_", "")),
        _normalize(reference_id.replace("_", "-")),
        _normalize(name),
        _normalize(name.replace(" ", "")),
    }
    if reference_id == "vap":
        variants.add("voice_activity_projection")
    return {variant for variant in variants if variant}


def _normalized_join(item: dict[str, Any], *keys: str) -> str:
    return _normalize(" ".join(str(item.get(key, "")) for key in keys))


def _normalize(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _acquisition_track(entry: dict[str, Any]) -> str:
    category = str(entry.get("category", ""))
    if category == "full_duplex_benchmark":
        return "scenario benchmark bridge"
    if category == "voice_agent_framework":
        return "voice-agent integration bridge"
    if category == "vad_endpointing_baseline":
        return "VAD endpointing adapter"
    if category == "turn_prediction_objective":
        return "turn prediction objective bridge"
    return "turn prediction export"


def _acquisition_evidence_target(entry: dict[str, Any], track: str) -> str:
    reference_id = str(entry.get("id", "unknown"))
    if track == "scenario benchmark bridge":
        return f"runs/collections/{reference_id}/SCENARIO_BRIDGE.md"
    if track == "voice-agent integration bridge":
        return f"runs/collections/{reference_id}/PIPELINE_BRIDGE.md"
    if track == "VAD endpointing adapter":
        return f"runs/collections/{reference_id}/VAD_ADAPTER.md"
    if track == "turn prediction objective bridge":
        return f"runs/collections/{reference_id}/PREDICTION_OBJECTIVE.md"
    return f"runs/final/external/{_final_prediction_stem(reference_id)}_raw.jsonl"


def _final_prediction_stem(reference_id: str) -> str:
    stems = {
        "smart_turn": "smartturn",
        "easy_turn": "easyturn",
    }
    return stems.get(reference_id, reference_id)
