"""Curated ASR reference collection registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.resources import resolve_platform_path


DEFAULT_ASR_COLLECTIONS_PATH = Path("configs/references/asr_collections.json")


@dataclass(frozen=True)
class ASRCollectionsValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "asr_collections: OK"
        return "asr_collections: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class ASRCollectionCoverageCheck:
    reference_id: str
    name: str
    category: str
    priority: str
    covered: bool
    required: bool
    evidence: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "name": self.name,
            "category": self.category,
            "priority": self.priority,
            "covered": self.covered,
            "required": self.required,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ASRCollectionCoverageReport:
    ok: bool
    required_priorities: tuple[str, ...]
    checks: list[ASRCollectionCoverageCheck]

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
            "# ASR Collection Coverage",
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
            f"asr_collection_coverage: {'OK' if self.ok else 'FAILED'}",
            f"required_priorities: {', '.join(self.required_priorities)}",
        ]
        for check in self.checks:
            marker = "OK" if check.covered else "MISSING"
            required = "required" if check.required else "optional"
            evidence = ", ".join(check.evidence) if check.evidence else "none"
            lines.append(f"- {marker} {check.reference_id} ({required}; evidence: {evidence})")
        return "\n".join(lines)


@dataclass(frozen=True)
class ASRCollectionReadinessRow:
    reference_id: str
    name: str
    category: str
    priority: str
    required: bool
    license: str
    license_review_needed: bool
    action_count: int
    adapter_evidence: list[str]
    adapter_statuses: list[str]
    warnings: list[str]
    ok: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "name": self.name,
            "category": self.category,
            "priority": self.priority,
            "required": self.required,
            "license": self.license,
            "license_review_needed": self.license_review_needed,
            "action_count": self.action_count,
            "adapter_evidence": self.adapter_evidence,
            "adapter_statuses": self.adapter_statuses,
            "warnings": self.warnings,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class ASRCollectionReadinessReport:
    ok: bool
    reviewed_at: str
    review_age_days: int | None
    max_review_age_days: int | None
    stale_review: bool
    required_priorities: tuple[str, ...]
    rows: list[ASRCollectionReadinessRow]
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reviewed_at": self.reviewed_at,
            "review_age_days": self.review_age_days,
            "max_review_age_days": self.max_review_age_days,
            "stale_review": self.stale_review,
            "required_priorities": list(self.required_priorities),
            "rows": [row.to_dict() for row in self.rows],
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def to_markdown(self) -> str:
        rows = [
            {
                "reference": row.reference_id,
                "priority": row.priority,
                "required": "yes" if row.required else "no",
                "license": row.license,
                "license_review": "yes" if row.license_review_needed else "no",
                "adapter_statuses": ", ".join(row.adapter_statuses),
                "evidence": ", ".join(row.adapter_evidence),
                "status": "OK" if row.ok else "MISSING",
                "warnings": ", ".join(row.warnings),
            }
            for row in self.rows
        ]
        lines = [
            "# ASR Collection Readiness",
            "",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- reviewed_at: `{self.reviewed_at}`",
            f"- review_age_days: `{self.review_age_days if self.review_age_days is not None else 'unknown'}`",
            f"- max_review_age_days: `{self.max_review_age_days if self.max_review_age_days is not None else 'disabled'}`",
            f"- stale_review: `{'yes' if self.stale_review else 'no'}`",
            f"- required_priorities: `{', '.join(self.required_priorities)}`",
            f"- errors: `{len(self.errors)}`",
            f"- warnings: `{len(self.warnings)}`",
            "",
            dict_table(rows),
        ]
        if self.errors:
            lines.extend(["", "## Errors", ""])
            lines.extend(f"- {error}" for error in self.errors)
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)

    def to_text(self) -> str:
        lines = [
            f"asr_collection_readiness: {'OK' if self.ok else 'FAILED'}",
            f"reviewed_at: {self.reviewed_at}",
            f"review_age_days: {self.review_age_days if self.review_age_days is not None else 'unknown'}",
            f"stale_review: {self.stale_review}",
            f"required_priorities: {', '.join(self.required_priorities)}",
        ]
        lines.extend(f"- ERROR {error}" for error in self.errors)
        lines.extend(f"- WARN {warning}" for warning in self.warnings)
        for row in self.rows:
            marker = "OK" if row.ok else "MISSING"
            evidence = ", ".join(row.adapter_evidence) if row.adapter_evidence else "none"
            lines.append(f"- {marker} {row.reference_id} ({row.priority}; evidence: {evidence})")
        return "\n".join(lines)


def load_asr_collections(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = resolve_platform_path(Path(path) if path else DEFAULT_ASR_COLLECTIONS_PATH)
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("ASR collections registry must be a JSON object")
    return payload


def write_asr_collections_json(path: str | Path, registry: dict[str, Any] | None = None) -> str:
    registry = registry or load_asr_collections()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_asr_collections(registry: dict[str, Any]) -> ASRCollectionsValidation:
    errors: list[str] = []
    for key in ("id", "version", "reviewed_at", "title", "entries"):
        if key not in registry:
            errors.append(f"missing top-level key: {key}")

    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("entries must be a non-empty list")
        return ASRCollectionsValidation(ok=False, errors=errors)

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
        priority = entry.get("priority")
        if priority not in {"p0", "p1", "p2"}:
            errors.append(f"entry {entry_id or index} priority must be p0, p1, or p2")

    return ASRCollectionsValidation(ok=not errors, errors=errors)


def audit_asr_collection_coverage(
    collections: dict[str, Any],
    adapter_registry: dict[str, Any],
    *,
    required_priorities: tuple[str, ...] = ("p0",),
) -> ASRCollectionCoverageReport:
    validation = validate_asr_collections(collections)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    adapters = adapter_registry.get("adapters", [])
    if not isinstance(adapters, list):
        raise ValueError("adapter registry adapters must be a list")

    checks: list[ASRCollectionCoverageCheck] = []
    for entry in collections["entries"]:
        evidence = _coverage_evidence(entry, adapters)
        priority = str(entry.get("priority", ""))
        required = priority in required_priorities
        checks.append(
            ASRCollectionCoverageCheck(
                reference_id=str(entry["id"]),
                name=str(entry["name"]),
                category=str(entry["category"]),
                priority=priority,
                covered=bool(evidence),
                required=required,
                evidence=evidence,
            )
        )
    ok = all(check.covered for check in checks if check.required)
    return ASRCollectionCoverageReport(ok=ok, required_priorities=required_priorities, checks=checks)


def audit_asr_collection_readiness(
    collections: dict[str, Any],
    adapter_registry: dict[str, Any],
    *,
    required_priorities: tuple[str, ...] = ("p0", "p1"),
    max_review_age_days: int | None = 3650,
    today: date | None = None,
) -> ASRCollectionReadinessReport:
    """Audit whether the curated references are usable for release planning."""

    validation = validate_asr_collections(collections)
    errors = list(validation.errors)
    warnings: list[str] = []
    rows: list[ASRCollectionReadinessRow] = []
    reviewed_at = str(collections.get("reviewed_at", ""))
    review_date = _parse_review_date(reviewed_at)
    review_age_days: int | None = None
    stale_review = False
    if review_date is None:
        errors.append("reviewed_at must be an ISO date")
    elif max_review_age_days is not None:
        current_date = today or date.today()
        review_age_days = (current_date - review_date).days
        stale_review = review_age_days > max_review_age_days
        if stale_review:
            errors.append(
                f"reference collection review is stale: {review_age_days} day(s) old, max {max_review_age_days}"
            )

    adapters = adapter_registry.get("adapters", [])
    if not isinstance(adapters, list):
        errors.append("adapter registry adapters must be a list")
        adapters = []
    adapter_statuses = {
        str(adapter.get("id", "")): str(adapter.get("status", "unknown"))
        for adapter in adapters
        if isinstance(adapter, dict)
    }

    entries = collections.get("entries", [])
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            priority = str(entry.get("priority", ""))
            required = priority in required_priorities
            evidence = _coverage_evidence(entry, adapters)
            statuses = sorted(
                {
                    adapter_statuses.get(item.removeprefix("adapter:"), "unknown")
                    for item in evidence
                    if item.startswith("adapter:")
                }
            )
            actions = entry.get("stable_asr_actions", [])
            action_count = len(actions) if isinstance(actions, list) else 0
            license_name = str(entry.get("license", ""))
            row_warnings: list[str] = []
            if license_name == "see_upstream":
                row_warnings.append("license_review_needed")
            if action_count < 2:
                row_warnings.append("weak_action_plan")
            if not evidence:
                row_warnings.append("no_adapter_or_bridge_evidence")
            row_ok = (not required or bool(evidence)) and action_count >= 2
            rows.append(
                ASRCollectionReadinessRow(
                    reference_id=str(entry.get("id", "")),
                    name=str(entry.get("name", "")),
                    category=str(entry.get("category", "")),
                    priority=priority,
                    required=required,
                    license=license_name,
                    license_review_needed=license_name == "see_upstream",
                    action_count=action_count,
                    adapter_evidence=evidence,
                    adapter_statuses=statuses,
                    warnings=row_warnings,
                    ok=row_ok,
                )
            )
            warnings.extend(f"{entry.get('id', '')}: {warning}" for warning in row_warnings)

    for row in rows:
        if row.required and not row.adapter_evidence:
            errors.append(f"required reference missing adapter evidence: {row.reference_id}")
        if row.action_count < 2:
            errors.append(f"reference has fewer than two Stable-ASR actions: {row.reference_id}")

    ok = not errors and all(row.ok for row in rows if row.required)
    return ASRCollectionReadinessReport(
        ok=ok,
        reviewed_at=reviewed_at,
        review_age_days=review_age_days,
        max_review_age_days=max_review_age_days,
        stale_review=stale_review,
        required_priorities=required_priorities,
        rows=rows,
        errors=errors,
        warnings=warnings,
    )


def asr_collections_markdown(registry: dict[str, Any]) -> str:
    rows = []
    entries = registry.get("entries", [])
    if isinstance(entries, list):
        for entry in sorted(entries, key=lambda item: (item.get("category", ""), item.get("priority", ""), item.get("id", ""))):
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "id": entry.get("id", ""),
                    "category": entry.get("category", ""),
                    "priority": entry.get("priority", ""),
                    "project": f"[{entry.get('name', '')}]({entry.get('source_url', '')})",
                    "focus": entry.get("focus", ""),
                    "stable_asr_actions": ", ".join(entry.get("stable_asr_actions", [])),
                }
            )

    title = str(registry.get("title", "Stable-ASR Reference Collections"))
    reviewed_at = str(registry.get("reviewed_at", "unknown"))
    description = str(registry.get("description", ""))
    lines = [
        f"# {title}",
        "",
        description,
        "",
        f"- registry id: `{registry.get('id', '')}`",
        f"- version: `{registry.get('version', '')}`",
        f"- reviewed_at: `{reviewed_at}`",
        f"- entries: `{len(rows)}`",
        "",
        dict_table(rows),
        "",
        "## Usage Policy",
        "",
        "This collection is for attribution, adapter planning, benchmark planning, and design review. Stable-ASR should not vendor upstream code unless the license is compatible and the copied scope is explicitly documented.",
    ]
    return "\n".join(lines)


def asr_collections_reference_markdown(registry: dict[str, Any]) -> str:
    """Render paper-oriented reference notes from the curated ASR collection."""

    validation = validate_asr_collections(registry)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    rows = []
    for entry in sorted(registry["entries"], key=lambda item: (item["priority"], item["category"], item["id"])):
        rows.append(
            {
                "citation_key": _citation_key(entry),
                "project": f"[{entry['name']}]({entry['source_url']})",
                "priority": entry["priority"],
                "category": entry["category"],
                "license": entry["license"],
                "paper_use": entry["reference_use"],
            }
        )

    lines = [
        "# Stable-ASR Paper Reference Notes",
        "",
        (
            "These references are generated from the curated ASR collections registry. "
            "They are intended for related-work drafting, adapter planning, and artifact attribution; "
            "they are not benchmark claims."
        ),
        "",
        f"- registry id: `{registry['id']}`",
        f"- reviewed_at: `{registry['reviewed_at']}`",
        f"- entries: `{len(rows)}`",
        "",
        dict_table(rows),
    ]
    return "\n".join(lines)


def asr_collections_acquisition_markdown(registry: dict[str, Any]) -> str:
    """Render a contributor-facing acquisition plan for upstream ASR references."""

    validation = validate_asr_collections(registry)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    rows = []
    p0_entries = []
    for entry in sorted(registry["entries"], key=lambda item: (item["priority"], item["category"], item["id"])):
        track = _acquisition_track(entry)
        license_review_needed = str(entry.get("license", "")) == "see_upstream"
        row = {
            "reference": entry["id"],
            "priority": entry["priority"],
            "track": track,
            "license_review": "yes" if license_review_needed else "no",
            "evidence_target": _acquisition_evidence_target(entry, track),
            "actions": ", ".join(entry["stable_asr_actions"]),
        }
        rows.append(row)
        if entry["priority"] == "p0":
            p0_entries.append(entry)

    p0_lines = [
        f"1. `{entry['id']}`: {_acquisition_track(entry)}; write `{_acquisition_evidence_target(entry, _acquisition_track(entry))}`."
        for entry in p0_entries
    ]
    lines = [
        "# Stable-ASR ASR Collection Acquisition Plan",
        "",
        (
            "This plan turns the upstream reference registry into executable collection work. "
            "It is for adapter planning, transcript export, license review, and paper evidence staging; "
            "it does not vendor upstream code or model weights."
        ),
        "",
        f"- registry id: `{registry['id']}`",
        f"- reviewed_at: `{registry['reviewed_at']}`",
        f"- entries: `{len(rows)}`",
        f"- p0_entries: `{len(p0_entries)}`",
        "",
        "## Common Commands",
        "",
        "```bash",
        "stable-asr asr-collections --audit-readiness --output runs/ASR_COLLECTION_READINESS.md",
        "stable-asr asr-collections --format acquisition-markdown --output runs/ASR_COLLECTION_ACQUISITION.md",
        "stable-asr adapter-pack --output-dir runs/adapter_pack",
        "stable-asr final-acquisition-pack --output-dir runs/final_acquisition_pack",
        "stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --validate-only --require-input-manifest --min-adapters 2",
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
        (
            "A reference is useful to Stable-ASR only after it has a concrete artifact: "
            "a command adapter output, transcript converter, recipe bridge note, data bridge note, "
            "or explicit license review. Registry presence alone is not evidence."
        ),
    ]
    return "\n".join(lines)


def asr_collections_bibtex(registry: dict[str, Any]) -> str:
    """Render a lightweight BibTeX file for upstream project attribution."""

    validation = validate_asr_collections(registry)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    year = _review_year(registry)
    entries = []
    for entry in sorted(registry["entries"], key=lambda item: item["id"]):
        note = f"{entry['category']}; priority {entry['priority']}; license: {entry['license']}"
        fields = [
            f"  title = {{{{{_bibtex_escape(str(entry['name']))}}}}},",
            f"  howpublished = {{\\url{{{_bibtex_escape(str(entry['source_url']))}}}}},",
            f"  note = {{{_bibtex_escape(note)}}},",
            f"  year = {{{year}}},",
        ]
        docs_url = str(entry.get("docs_url", ""))
        if docs_url and docs_url != entry["source_url"]:
            fields.insert(2, f"  url = {{{_bibtex_escape(docs_url)}}},")
        entries.append("@misc{" + _citation_key(entry) + ",\n" + "\n".join(fields) + "\n}")
    return "\n\n".join(entries) + "\n"


def _coverage_evidence(entry: dict[str, Any], adapters: list[Any]) -> list[str]:
    evidence: list[str] = []
    variants = _reference_variants(entry)
    for adapter in adapters:
        if not isinstance(adapter, dict):
            continue
        related = adapter.get("related_references", [])
        if isinstance(related, list) and any(str(item) in variants for item in related):
            evidence.append(f"adapter:{adapter.get('id', '')}")
            continue
        text = _normalize_reference_text(
            " ".join(
                str(adapter.get(key, ""))
                for key in ("id", "title", "entrypoint", "notes")
            )
        )
        if any(variant and variant in text for variant in variants):
            evidence.append(f"adapter:{adapter.get('id', '')}")
    return evidence


def _parse_review_date(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _reference_variants(entry: dict[str, Any]) -> set[str]:
    reference_id = str(entry.get("id", ""))
    name = str(entry.get("name", ""))
    variants = {
        _normalize_reference_text(reference_id),
        _normalize_reference_text(reference_id.replace("_", "-")),
        _normalize_reference_text(name),
    }
    return {variant for variant in variants if variant}


def _normalize_reference_text(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _citation_key(entry: dict[str, Any]) -> str:
    return "stableasr_ref_" + _normalize_reference_text(str(entry.get("id", "unknown")))


def _acquisition_track(entry: dict[str, Any]) -> str:
    category = str(entry.get("category", ""))
    if category == "data_layer":
        return "data bridge"
    if category in {"classic_toolkit", "research_training_toolkit"}:
        return "recipe bridge"
    if category == "timestamp_alignment":
        return "timestamp converter"
    if category in {"deployment_runtime", "inference_runtime", "on_device_runtime"}:
        return "runtime command adapter"
    return "ASR command adapter"


def _acquisition_evidence_target(entry: dict[str, Any], track: str) -> str:
    reference_id = str(entry.get("id", "unknown"))
    if track == "data bridge":
        return f"runs/collections/{reference_id}/DATA_BRIDGE.md"
    if track == "recipe bridge":
        return f"runs/collections/{reference_id}/RECIPE_BRIDGE.md"
    if track == "timestamp converter":
        return f"runs/collections/{reference_id}/TIMESTAMP_CONVERTER.md"
    return f"runs/final/asr_commands/raw/{reference_id}_raw.jsonl"


def _review_year(registry: dict[str, Any]) -> str:
    reviewed_at = str(registry.get("reviewed_at", ""))
    if len(reviewed_at) >= 4 and reviewed_at[:4].isdigit():
        return reviewed_at[:4]
    return "2026"


def _bibtex_escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}")
