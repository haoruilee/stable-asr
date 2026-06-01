"""Repository catalog for the Stable-ASR platform surface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from stable_asr.data.sources import load_data_sources, validate_data_sources
from stable_asr.eval.report import dict_table
from stable_asr.models.adapters.registry import load_adapter_registry, validate_adapter_registry
from stable_asr.models.registry import load_model_registry, validate_model_registry
from stable_asr.paper.platform_parity import audit_platform_parity, load_platform_parity, validate_platform_parity
from stable_asr.paper.suites import load_benchmark_suite, validate_benchmark_suite
from stable_asr.references import (
    load_asr_collections,
    load_turn_collections,
    validate_asr_collections,
    validate_turn_collections,
)
from stable_asr.roadmap import load_roadmap, roadmap_status, validate_roadmap
from stable_asr.scenarios.suites import load_scenario_suite, validate_scenario_suite
from stable_asr.schemas import load_schema_registry, validate_schema_registry


@dataclass(frozen=True)
class CatalogSection:
    name: str
    registry_id: str
    count: int
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "registry_id": self.registry_id,
            "count": self.count,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PlatformCatalogReport:
    ok: bool
    repo_root: str
    sections: list[CatalogSection]
    stable_worldmodel_parity: dict[str, object]
    quick_commands: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "repo_root": self.repo_root,
            "sections": [section.to_dict() for section in self.sections],
            "stable_worldmodel_parity": self.stable_worldmodel_parity,
            "quick_commands": self.quick_commands,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Stable-ASR Platform Catalog",
            "",
            f"- status: `{'OK' if self.ok else 'FAILED'}`",
            f"- repo_root: `{self.repo_root}`",
            "",
            "## Registered Assets",
            "",
            dict_table(
                [
                    {
                        "section": section.name,
                        "status": section.status,
                        "count": section.count,
                        "registry": section.registry_id,
                        "detail": section.detail,
                    }
                    for section in self.sections
                ]
            ),
            "",
            "## Stable-WorldModel-Style Parity",
            "",
            f"- status: `{'OK' if self.stable_worldmodel_parity.get('ok') else 'MISSING'}`",
            f"- registry: `{self.stable_worldmodel_parity.get('registry_id', '')}`",
            f"- missing_count: `{self.stable_worldmodel_parity.get('missing_count', 0)}`",
            "",
            "## Quick Commands",
            "",
        ]
        lines.extend(f"- `{command}`" for command in self.quick_commands)
        lines.append("")
        return "\n".join(lines)


def build_platform_catalog(*, repo_root: str | Path = ".") -> PlatformCatalogReport:
    """Build a one-page catalog of the checked-in Stable-ASR platform assets."""

    repo_root = Path(repo_root)
    sections = [
        _section("data_sources", lambda: _data_sources_section(repo_root)),
        _section("models", lambda: _models_section(repo_root)),
        _section("adapters", lambda: _adapters_section(repo_root)),
        _section("scenarios", lambda: _scenarios_section(repo_root)),
        _section("benchmark_suite", lambda: _benchmark_section(repo_root)),
        _section("schemas", lambda: _schemas_section(repo_root)),
        _section("asr_references", lambda: _asr_references_section(repo_root)),
        _section("turn_references", lambda: _turn_references_section(repo_root)),
        _section("roadmap", lambda: _roadmap_section(repo_root)),
    ]
    parity = _platform_parity(repo_root)
    ok = all(section.ok for section in sections) and bool(parity.get("ok"))
    return PlatformCatalogReport(
        ok=ok,
        repo_root=str(repo_root),
        sections=sections,
        stable_worldmodel_parity=parity,
        quick_commands=_quick_commands(),
    )


def write_platform_catalog_json(report: PlatformCatalogReport, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def write_platform_catalog_markdown(report: PlatformCatalogReport, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_markdown(), encoding="utf-8")
    return str(path)


def _section(name: str, factory: Callable[[], CatalogSection]) -> CatalogSection:
    try:
        return factory()
    except (OSError, ValueError, TypeError) as exc:
        return CatalogSection(name=name, registry_id="", count=0, status="FAILED", detail=str(exc))


def _data_sources_section(repo_root: Path) -> CatalogSection:
    registry = load_data_sources(repo_root / "configs" / "datasets" / "stable_asr_sources.json")
    validation = validate_data_sources(registry)
    return CatalogSection(
        name="data_sources",
        registry_id=str(registry.get("id", "")),
        count=len(registry.get("sources", [])),
        status="OK" if validation.ok else "FAILED",
        detail=_validation_detail(validation.errors, "ASR, turn, streaming transcript, and public corpus sources"),
    )


def _models_section(repo_root: Path) -> CatalogSection:
    registry = load_model_registry(repo_root / "configs" / "models" / "stable_asr_models.json")
    validation = validate_model_registry(registry)
    models = registry.get("models", [])
    implemented = _count_by_value(models, "status", "implemented")
    return CatalogSection(
        name="models",
        registry_id=str(registry.get("id", "")),
        count=len(models),
        status="OK" if validation.ok else "FAILED",
        detail=_validation_detail(validation.errors, f"{implemented} implemented model/baseline entries"),
    )


def _adapters_section(repo_root: Path) -> CatalogSection:
    registry = load_adapter_registry(repo_root / "configs" / "adapters" / "stable_asr_adapters.json")
    validation = validate_adapter_registry(registry)
    adapters = registry.get("adapters", [])
    implemented = _count_status_prefix(adapters, "implemented") + _count_status_prefix(adapters, "converter_implemented")
    templates = _count_by_value(adapters, "status", "template")
    return CatalogSection(
        name="adapters",
        registry_id=str(registry.get("id", "")),
        count=len(adapters),
        status="OK" if validation.ok else "FAILED",
        detail=_validation_detail(validation.errors, f"{implemented} implemented/converter entries, {templates} templates"),
    )


def _scenarios_section(repo_root: Path) -> CatalogSection:
    suite = load_scenario_suite(repo_root / "configs" / "scenarios" / "stable_asr_voiceworld_v0.json")
    validation = validate_scenario_suite(suite)
    return CatalogSection(
        name="voiceworld_scenarios",
        registry_id=str(suite.get("id", "")),
        count=len(suite.get("scenarios", [])),
        status="OK" if validation.ok else "FAILED",
        detail=_validation_detail(validation.errors, f"{len(suite.get('factors', []))} controllable factor(s)"),
    )


def _benchmark_section(repo_root: Path) -> CatalogSection:
    suite = load_benchmark_suite(repo_root / "configs" / "benchmarks" / "stable_asr_v0.json")
    validation = validate_benchmark_suite(suite)
    systems = sorted({system for task in suite.get("tasks", []) for system in task.get("systems", [])})
    return CatalogSection(
        name="benchmark_suite",
        registry_id=str(suite.get("id", "")),
        count=len(suite.get("tasks", [])),
        status="OK" if validation.ok else "FAILED",
        detail=_validation_detail(validation.errors, f"{len(systems)} system family target(s)"),
    )


def _schemas_section(repo_root: Path) -> CatalogSection:
    registry = load_schema_registry(repo_root / "configs" / "schemas" / "stable_asr_schemas.json")
    validation = validate_schema_registry(registry)
    return CatalogSection(
        name="schemas",
        registry_id=str(registry.get("id", "")),
        count=len(registry.get("schemas", [])),
        status="OK" if validation.ok else "FAILED",
        detail=_validation_detail(validation.errors, "versioned JSON/JSONL contracts"),
    )


def _asr_references_section(repo_root: Path) -> CatalogSection:
    registry = load_asr_collections(repo_root / "configs" / "references" / "asr_collections.json")
    validation = validate_asr_collections(registry)
    entries = registry.get("entries", [])
    p0_p1 = sum(1 for entry in entries if entry.get("priority") in {"p0", "p1"})
    return CatalogSection(
        name="asr_references",
        registry_id=str(registry.get("id", "")),
        count=len(entries),
        status="OK" if validation.ok else "FAILED",
        detail=_validation_detail(validation.errors, f"{p0_p1} P0/P1 upstream ASR references"),
    )


def _turn_references_section(repo_root: Path) -> CatalogSection:
    registry = load_turn_collections(repo_root / "configs" / "references" / "turn_collections.json")
    validation = validate_turn_collections(registry)
    entries = registry.get("entries", [])
    p0_p1 = sum(1 for entry in entries if entry.get("priority") in {"p0", "p1"})
    return CatalogSection(
        name="turn_references",
        registry_id=str(registry.get("id", "")),
        count=len(entries),
        status="OK" if validation.ok else "FAILED",
        detail=_validation_detail(validation.errors, f"{p0_p1} P0/P1 turn/full-duplex references"),
    )


def _roadmap_section(repo_root: Path) -> CatalogSection:
    roadmap = load_roadmap(repo_root / "configs" / "roadmap" / "stable_asr_roadmap.json")
    validation = validate_roadmap(roadmap)
    status = roadmap_status(roadmap, repo_root=repo_root)
    complete = sum(1 for milestone in status.milestones if milestone.status == "complete")
    planned = sum(1 for milestone in status.milestones if milestone.status == "planned")
    return CatalogSection(
        name="roadmap",
        registry_id=str(roadmap.get("id", "")),
        count=len(status.milestones),
        status="OK" if validation.ok and status.ok else "FAILED",
        detail=_validation_detail(validation.errors, f"{complete} complete milestone(s), {planned} planned milestone(s)"),
    )


def _platform_parity(repo_root: Path) -> dict[str, object]:
    try:
        registry = load_platform_parity(repo_root / "configs" / "platform" / "stable_worldmodel_parity.json")
        validation = validate_platform_parity(registry)
        if not validation.ok:
            return {
                "ok": False,
                "registry_id": registry.get("id", ""),
                "missing_count": 0,
                "errors": validation.errors,
            }
        report = audit_platform_parity(registry, repo_root=repo_root)
        return report.to_dict()
    except (OSError, ValueError, TypeError) as exc:
        return {"ok": False, "registry_id": "", "missing_count": 0, "errors": [str(exc)]}


def _validation_detail(errors: list[str], ok_detail: str) -> str:
    if errors:
        return "; ".join(errors[:3])
    return ok_detail


def _count_by_value(items: Any, key: str, value: str) -> int:
    if not isinstance(items, list):
        return 0
    return sum(1 for item in items if isinstance(item, dict) and item.get(key) == value)


def _count_status_prefix(items: Any, prefix: str) -> int:
    if not isinstance(items, list):
        return 0
    return sum(1 for item in items if isinstance(item, dict) and str(item.get("status", "")).startswith(prefix))


def _quick_commands() -> list[str]:
    return [
        "stable-asr catalog",
        "stable-asr platform-parity",
        "stable-asr roadmap-status",
        "stable-asr data-sources",
        "stable-asr model-registry",
        "stable-asr adapter-registry",
        "stable-asr scenario-suite",
        "stable-asr benchmark-suite",
        "stable-asr schema-registry",
        "stable-asr asr-collections --audit-readiness",
        "stable-asr turn-collections --audit-coverage --require-priority p0 --require-priority p1",
        "stable-asr paper-status",
        "stable-asr final-config --config configs/final/paper_final.json --plan-missing",
    ]
