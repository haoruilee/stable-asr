"""Repository and environment health checks for Stable-ASR."""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stable_asr import __version__
from stable_asr.data.sources import load_data_sources, validate_data_sources
from stable_asr.models.adapters.registry import load_adapter_registry, validate_adapter_registry
from stable_asr.paper.final_config import audit_final_run_files, load_final_run_config, validate_final_run_config
from stable_asr.paper.final_experiments import load_final_experiments, validate_final_experiments
from stable_asr.paper.parity import load_paper_parity_checklist, validate_paper_parity_checklist
from stable_asr.paper.suites import load_benchmark_suite, validate_benchmark_suite
from stable_asr.references import (
    load_asr_collections,
    load_turn_collections,
    validate_asr_collections,
    validate_turn_collections,
)
from stable_asr.resources import resolve_platform_path
from stable_asr.roadmap import load_roadmap, validate_roadmap
from stable_asr.scenarios.suites import load_scenario_suite, validate_scenario_suite
from stable_asr.schemas import load_schema_registry, validate_schema_registry


@dataclass(frozen=True)
class DoctorCheck:
    category: str
    name: str
    ok: bool
    required: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "name": self.name,
            "ok": self.ok,
            "required": self.required,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    final_inputs_ready: bool
    release_environment_ready: bool
    checks: list[DoctorCheck]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "final_inputs_ready": self.final_inputs_ready,
            "release_environment_ready": self.release_environment_ready,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_text(self) -> str:
        lines = [
            f"stable_asr_doctor: {'OK' if self.ok else 'FAILED'}",
            f"final_inputs_ready: {'YES' if self.final_inputs_ready else 'NO'}",
            f"release_environment_ready: {'YES' if self.release_environment_ready else 'NO'}",
        ]
        for check in self.checks:
            status = "OK" if check.ok else "MISSING"
            required = "required" if check.required else "optional"
            lines.append(f"- {status} {check.category}/{check.name}: {check.detail} ({required})")
        return "\n".join(lines)


def run_doctor(
    *,
    repo_root: str | Path = ".",
    check_final_files: bool = False,
    check_release_env: bool = False,
) -> DoctorReport:
    """Run a concise health check for repository setup and optional dependencies."""

    repo_root = Path(repo_root)
    checks: list[DoctorCheck] = [
        DoctorCheck(
            "environment",
            "python",
            sys.version_info >= (3, 10),
            True,
            f"{platform.python_version()} on {platform.system()}",
        ),
        DoctorCheck("package", "stable_asr", True, True, f"version={__version__}"),
    ]
    checks.extend(_optional_dependency_checks())
    checks.extend(_config_checks(repo_root))
    release_environment_ready = _release_environment_ready()
    if check_release_env:
        checks.append(
            DoctorCheck(
                "release",
                "environment",
                release_environment_ready,
                True,
                (
                    "ready for paper-release-smoke --strict"
                    if release_environment_ready
                    else "requires torch and Lance backend; install with: python -m pip install -e '.[lance,train]'"
                ),
            )
        )
    final_inputs_ready = True
    if check_final_files:
        final_file_report = audit_final_run_files(
            load_final_run_config(repo_root / "configs" / "final" / "paper_final.json"),
            repo_root=repo_root,
        )
        final_inputs_ready = final_file_report.ok
        missing_required = [
            check
            for check in final_file_report.checks
            if check.required and not check.ok
        ]
        checks.append(
            DoctorCheck(
                "final",
                "input_files",
                final_file_report.ok,
                False,
                "ready" if final_file_report.ok else f"{len(missing_required)} required input(s) missing",
            )
        )
    required_ok = all(check.ok for check in checks if check.required)
    return DoctorReport(
        ok=required_ok,
        final_inputs_ready=final_inputs_ready,
        release_environment_ready=release_environment_ready,
        checks=checks,
    )


def _optional_dependency_checks() -> list[DoctorCheck]:
    dependencies = [
        ("torch", "NanoTurn training"),
        ("pyarrow", "Parquet data backend"),
        ("lance", "Lance data backend"),
        ("onnx", "ONNX export validation"),
    ]
    return [
        DoctorCheck(
            "dependency",
            name,
            importlib.util.find_spec(name) is not None,
            False,
            purpose,
        )
        for name, purpose in dependencies
    ]


def _release_environment_ready() -> bool:
    """Return whether the local environment can produce a READY smoke audit."""

    return _has_import("torch") and _has_working_lance()


def _has_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_working_lance() -> bool:
    if importlib.util.find_spec("lance") is None:
        return False
    try:
        import lance
    except Exception:
        return False
    return hasattr(lance, "dataset") and hasattr(lance, "write_dataset")


def _config_checks(repo_root: Path) -> list[DoctorCheck]:
    return [
        _schema_check(
            "config",
            "benchmark_suite",
            repo_root / "configs" / "benchmarks" / "stable_asr_v0.json",
            lambda path: validate_benchmark_suite(load_benchmark_suite(path)).to_text(),
        ),
        _schema_check(
            "config",
            "roadmap",
            repo_root / "configs" / "roadmap" / "stable_asr_roadmap.json",
            lambda path: validate_roadmap(load_roadmap(path)).to_text(),
        ),
        _schema_check(
            "config",
            "data_sources",
            repo_root / "configs" / "datasets" / "stable_asr_sources.json",
            lambda path: validate_data_sources(load_data_sources(path)).to_text(),
        ),
        _schema_check(
            "config",
            "adapter_registry",
            repo_root / "configs" / "adapters" / "stable_asr_adapters.json",
            lambda path: validate_adapter_registry(load_adapter_registry(path)).to_text(),
        ),
        _schema_check(
            "config",
            "asr_collections",
            repo_root / "configs" / "references" / "asr_collections.json",
            lambda path: validate_asr_collections(load_asr_collections(path)).to_text(),
        ),
        _schema_check(
            "config",
            "turn_collections",
            repo_root / "configs" / "references" / "turn_collections.json",
            lambda path: validate_turn_collections(load_turn_collections(path)).to_text(),
        ),
        _schema_check(
            "config",
            "scenario_suite",
            repo_root / "configs" / "scenarios" / "stable_asr_voiceworld_v0.json",
            lambda path: validate_scenario_suite(load_scenario_suite(path)).to_text(),
        ),
        _schema_check(
            "config",
            "schema_registry",
            repo_root / "configs" / "schemas" / "stable_asr_schemas.json",
            lambda path: validate_schema_registry(load_schema_registry(path)).to_text(),
        ),
        _schema_check(
            "config",
            "paper_parity",
            repo_root / "configs" / "paper" / "paper_parity_checklist.json",
            lambda path: validate_paper_parity_checklist(load_paper_parity_checklist(path)).to_text(),
        ),
        _schema_check(
            "config",
            "final_experiments",
            repo_root / "configs" / "paper" / "final_experiments.json",
            lambda path: validate_final_experiments(load_final_experiments(path)).to_text(),
        ),
        _schema_check(
            "config",
            "final_run",
            repo_root / "configs" / "final" / "paper_final.json",
            lambda path: validate_final_run_config(load_final_run_config(path)).to_text(),
        ),
    ]


def _schema_check(
    category: str,
    name: str,
    path: Path,
    validator: Callable[[Path], str],
) -> DoctorCheck:
    path = resolve_platform_path(path)
    if not path.exists():
        return DoctorCheck(category, name, False, True, f"missing: {path}")
    try:
        text = validator(path)
    except (OSError, ValueError) as exc:
        return DoctorCheck(category, name, False, True, str(exc))
    ok = "FAILED" not in text
    return DoctorCheck(category, name, ok, True, text.splitlines()[0])
