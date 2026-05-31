"""Curated ASR reference collection registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table


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


def load_asr_collections(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path else DEFAULT_ASR_COLLECTIONS_PATH
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("ASR collections registry must be a JSON object")
    return payload


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
