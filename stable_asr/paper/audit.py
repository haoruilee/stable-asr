"""Audit paper-facing result and artifact bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.sources import load_data_sources, validate_data_sources
from stable_asr.models.adapters.registry import load_adapter_registry, validate_adapter_registry
from stable_asr.models.registry import load_model_registry, validate_model_registry
from stable_asr.paper.final_config import load_final_run_config, validate_final_run_config
from stable_asr.paper.final_experiments import load_final_experiments, validate_final_experiments
from stable_asr.paper.final_inputs import load_final_input_collections, validate_final_input_collections
from stable_asr.paper.figures import PAPER_FIGURES
from stable_asr.paper.integrity import verify_artifact_integrity
from stable_asr.paper.parity import load_paper_parity_checklist, validate_paper_parity_checklist
from stable_asr.paper.suites import (
    audit_benchmark_required_artifacts,
    audit_benchmark_suite_coverage,
    load_benchmark_suite,
    validate_benchmark_suite,
)
from stable_asr.paper.tables import PAPER_TABLES, load_paper_results
from stable_asr.references import (
    audit_asr_collection_coverage,
    audit_asr_collection_readiness,
    load_asr_collections,
    validate_asr_collections,
)
from stable_asr.resources import resolve_platform_path
from stable_asr.scenarios.suites import load_scenario_suite, validate_scenario_suite
from stable_asr.schemas import load_schema_registry, validate_schema_registry


@dataclass(frozen=True)
class PaperAuditCheck:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class PaperAuditReport:
    ok: bool
    checks: list[PaperAuditCheck]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}

    def to_text(self) -> str:
        lines = [f"paper_audit: {'OK' if self.ok else 'FAILED'}"]
        for check in self.checks:
            status = "OK" if check.ok else "FAIL"
            lines.append(f"- {status} {check.name}: {check.detail}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PaperReleaseAuditCheck:
    gate: str
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"gate": self.gate, "name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class PaperReleaseAuditReport:
    ok: bool
    checks: list[PaperReleaseAuditCheck]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}

    def to_text(self) -> str:
        lines = [f"paper_release_audit: {'READY' if self.ok else 'NOT_READY'}"]
        for check in self.checks:
            status = "OK" if check.ok else "MISSING"
            lines.append(f"- {status} {check.gate}/{check.name}: {check.detail}")
        return "\n".join(lines)


def audit_paper_artifacts(results_path: str | Path, artifacts_dir: str | Path | None = None) -> PaperAuditReport:
    """Validate the minimum artifact shape expected by the paper pipeline."""

    checks: list[PaperAuditCheck] = []
    results_path = Path(results_path)
    if not results_path.exists():
        return PaperAuditReport(
            ok=False,
            checks=[PaperAuditCheck("results_file", False, f"missing: {results_path}")],
        )

    try:
        results = load_paper_results(results_path)
        checks.append(PaperAuditCheck("results_file", True, str(results_path)))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return PaperAuditReport(
            ok=False,
            checks=[PaperAuditCheck("results_file", False, str(exc))],
        )

    checks.extend(_results_checks(results))
    if artifacts_dir is not None:
        checks.extend(_artifact_checks(Path(artifacts_dir), results_path=results_path))

    return PaperAuditReport(ok=all(check.ok for check in checks), checks=checks)


def audit_paper_release(
    *,
    repo_root: str | Path = ".",
    results_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    markdown_draft: str | Path | None = None,
    latex_draft: str | Path | None = None,
    dataset_card: str | Path | None = None,
    experiment_card: str | Path | None = None,
    model_card: str | Path | None = None,
) -> PaperReleaseAuditReport:
    """Audit whether the repository has enough evidence for a platform paper release."""

    repo_root = Path(repo_root)
    checks: list[PaperReleaseAuditCheck] = []
    checks.extend(_repo_release_checks(repo_root))

    results: dict[str, object] | None = None
    if results_path is None:
        checks.append(_release_check("paper", "results_file", False, "not provided"))
    else:
        results_path = Path(results_path)
        if not results_path.exists():
            checks.append(_release_check("paper", "results_file", False, f"missing: {results_path}"))
        else:
            try:
                results = load_paper_results(results_path)
                checks.append(_release_check("paper", "results_file", True, str(results_path)))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                checks.append(_release_check("paper", "results_file", False, str(exc)))

    suite: dict[str, Any] | None = None
    suite_path = repo_root / "configs" / "benchmarks" / "stable_asr_v0.json"
    if suite_path.exists():
        try:
            suite = load_benchmark_suite(suite_path)
            validation = validate_benchmark_suite(suite)
            checks.append(
                _release_check(
                    "paper",
                    "benchmark_suite_schema",
                    validation.ok,
                    f"{len(suite.get('tasks', []))} task(s)" if validation.ok else "; ".join(validation.errors[:3]),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("paper", "benchmark_suite_schema", False, str(exc)))

    parity_path = repo_root / "configs" / "paper" / "paper_parity_checklist.json"
    if parity_path.exists():
        try:
            parity_checklist = load_paper_parity_checklist(parity_path)
            parity_validation = validate_paper_parity_checklist(parity_checklist)
            checks.append(
                _release_check(
                    "paper",
                    "paper_parity_schema",
                    parity_validation.ok,
                    f"{len(parity_checklist.get('items', []))} item(s)"
                    if parity_validation.ok
                    else "; ".join(parity_validation.errors[:3]),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("paper", "paper_parity_schema", False, str(exc)))

    final_experiments_path = repo_root / "configs" / "paper" / "final_experiments.json"
    if final_experiments_path.exists():
        try:
            final_experiments = load_final_experiments(final_experiments_path)
            final_validation = validate_final_experiments(final_experiments)
            checks.append(
                _release_check(
                    "paper",
                    "final_experiments_schema",
                    final_validation.ok,
                    f"{len(final_experiments.get('experiments', []))} experiment(s)"
                    if final_validation.ok
                    else "; ".join(final_validation.errors[:3]),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("paper", "final_experiments_schema", False, str(exc)))

    final_run_path = repo_root / "configs" / "final" / "paper_final.json"
    if final_run_path.exists():
        try:
            final_run_config = load_final_run_config(final_run_path)
            final_run_validation = validate_final_run_config(final_run_config)
            checks.append(
                _release_check(
                    "paper",
                    "final_run_config_schema",
                    final_run_validation.ok,
                    f"{len(final_run_config.get('public_corpora', []))} corpora"
                    if final_run_validation.ok
                    else "; ".join(final_run_validation.errors[:3]),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("paper", "final_run_config_schema", False, str(exc)))

    final_input_collections_path = repo_root / "configs" / "final" / "input_collections.json"
    if final_input_collections_path.exists():
        try:
            final_input_collections = load_final_input_collections(final_input_collections_path)
            final_input_validation = validate_final_input_collections(final_input_collections)
            checks.append(
                _release_check(
                    "paper",
                    "final_input_collections_schema",
                    final_input_validation.ok,
                    f"{len(final_input_collections.get('collections', []))} collection(s)"
                    if final_input_validation.ok
                    else "; ".join(final_input_validation.errors[:3]),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("paper", "final_input_collections_schema", False, str(exc)))

    schema_registry_path = repo_root / "configs" / "schemas" / "stable_asr_schemas.json"
    if schema_registry_path.exists():
        try:
            schema_registry = load_schema_registry(schema_registry_path)
            schema_validation = validate_schema_registry(schema_registry)
            checks.append(
                _release_check(
                    "software",
                    "schema_registry",
                    schema_validation.ok,
                    f"{len(schema_registry.get('schemas', []))} schema(s)"
                    if schema_validation.ok
                    else "; ".join(schema_validation.errors[:3]),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("software", "schema_registry", False, str(exc)))

    source_path = repo_root / "configs" / "datasets" / "stable_asr_sources.json"
    if source_path.exists():
        try:
            source_registry = load_data_sources(source_path)
            source_validation = validate_data_sources(source_registry)
            checks.append(
                _release_check(
                    "data",
                    "data_source_registry",
                    source_validation.ok,
                    f"{len(source_registry.get('sources', []))} source(s)"
                    if source_validation.ok
                    else "; ".join(source_validation.errors[:3]),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("data", "data_source_registry", False, str(exc)))

    adapter_registry: dict[str, Any] | None = None
    adapter_registry_path = repo_root / "configs" / "adapters" / "stable_asr_adapters.json"
    if adapter_registry_path.exists():
        try:
            adapter_registry = load_adapter_registry(adapter_registry_path)
            adapter_validation = validate_adapter_registry(adapter_registry)
            checks.append(
                _release_check(
                    "adapter",
                    "adapter_registry_schema",
                    adapter_validation.ok,
                    f"{len(adapter_registry.get('adapters', []))} adapter(s)"
                    if adapter_validation.ok
                    else "; ".join(adapter_validation.errors[:3]),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("adapter", "adapter_registry_schema", False, str(exc)))

    model_registry_path = repo_root / "configs" / "models" / "stable_asr_models.json"
    if model_registry_path.exists():
        try:
            model_registry = load_model_registry(model_registry_path)
            model_validation = validate_model_registry(model_registry)
            checks.append(
                _release_check(
                    "model",
                    "model_registry_schema",
                    model_validation.ok,
                    f"{len(model_registry.get('models', []))} model(s)"
                    if model_validation.ok
                    else "; ".join(model_validation.errors[:3]),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("model", "model_registry_schema", False, str(exc)))

    asr_collections: dict[str, Any] | None = None
    asr_collections_path = repo_root / "configs" / "references" / "asr_collections.json"
    if asr_collections_path.exists():
        try:
            asr_collections = load_asr_collections(asr_collections_path)
            collections_validation = validate_asr_collections(asr_collections)
            checks.append(
                _release_check(
                    "reference",
                    "asr_collections_schema",
                    collections_validation.ok,
                    f"{len(asr_collections.get('entries', []))} reference(s)"
                    if collections_validation.ok
                    else "; ".join(collections_validation.errors[:3]),
                )
            )
            if collections_validation.ok and adapter_registry is not None:
                coverage = audit_asr_collection_coverage(
                    asr_collections,
                    adapter_registry,
                    required_priorities=("p0", "p1"),
                )
                required = [check for check in coverage.checks if check.required]
                covered = [check for check in required if check.covered]
                checks.append(
                    _release_check(
                        "reference",
                        "asr_collections_coverage",
                        coverage.ok,
                        f"{len(covered)}/{len(required)} required reference(s) covered",
                    )
                )
                readiness = audit_asr_collection_readiness(
                    asr_collections,
                    adapter_registry,
                    required_priorities=("p0", "p1"),
                )
                checks.append(
                    _release_check(
                        "reference",
                        "asr_collections_readiness",
                        readiness.ok,
                        f"{len(readiness.rows)} reference(s), {len(readiness.warnings)} warning(s)",
                    )
                )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("reference", "asr_collections_schema", False, str(exc)))

    scenario_suite_path = repo_root / "configs" / "scenarios" / "stable_asr_voiceworld_v0.json"
    if scenario_suite_path.exists():
        try:
            scenario_suite = load_scenario_suite(scenario_suite_path)
            scenario_validation = validate_scenario_suite(scenario_suite)
            checks.append(
                _release_check(
                    "scenario",
                    "scenario_suite_schema",
                    scenario_validation.ok,
                    f"{len(scenario_suite.get('scenarios', []))} scenario(s)"
                    if scenario_validation.ok
                    else "; ".join(scenario_validation.errors[:3]),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(_release_check("scenario", "scenario_suite_schema", False, str(exc)))

    if results is not None:
        checks.extend(_release_result_checks(results, suite=suite))

    if artifacts_dir is None:
        checks.append(_release_check("paper", "artifact_bundle", False, "not provided"))
    else:
        artifact_report = audit_paper_artifacts(results_path or repo_root / "missing_results.json", artifacts_dir)
        checks.append(
            _release_check(
                "paper",
                "artifact_bundle",
                artifact_report.ok,
                "tables and figures passed paper-audit" if artifact_report.ok else "paper-audit failed",
            )
        )

    checks.append(_optional_path_check("paper", "markdown_draft", markdown_draft))
    checks.append(_optional_path_check("paper", "latex_draft", latex_draft))
    checks.append(_optional_path_check("data", "dataset_card", dataset_card))
    checks.append(_optional_path_check("paper", "experiment_card", experiment_card))
    checks.append(_optional_path_check("model", "model_card", model_card))
    citation_path = _repo_or_platform_path(repo_root, "CITATION.cff")
    docs_path = _repo_or_platform_path(repo_root, "docs")
    checks.append(_release_check("paper", "citation", citation_path.exists(), _display_repo_path(repo_root, citation_path)))
    checks.append(_release_check("software", "docs_site", docs_path.exists(), _display_repo_path(repo_root, docs_path)))

    return PaperReleaseAuditReport(ok=all(check.ok for check in checks), checks=checks)


def _results_checks(results: dict[str, object]) -> list[PaperAuditCheck]:
    checks: list[PaperAuditCheck] = []
    required = (
        "meta",
        "data",
        "baselines",
        "turn_benchmarks",
        "scenarios",
        "policy_search",
        "streaming_asr",
    )
    missing = [key for key in required if key not in results]
    checks.append(
        PaperAuditCheck(
            "top_level_sections",
            not missing,
            "present" if not missing else "missing: " + ", ".join(missing),
        )
    )
    if missing:
        return checks

    meta = _dict(results["meta"])
    meta_missing = [key for key in ("episodes", "seed", "artifact_version") if key not in meta]
    checks.append(
        PaperAuditCheck(
            "run_metadata",
            not meta_missing,
            "episodes, seed, artifact_version" if not meta_missing else "missing: " + ", ".join(meta_missing),
        )
    )

    data = _dict(results["data"])
    benchmark = _dict(data.get("benchmark", {}))
    data_ok = "summary" in data and benchmark.get("status") in {"completed", "skipped"}
    checks.append(
        PaperAuditCheck(
            "data_section",
            data_ok,
            "summary and benchmark status present" if data_ok else "requires summary and completed/skipped benchmark",
        )
    )
    asr_recipe = _dict(data.get("asr_manifest_recipe", {}))
    asr_validation = _dict(asr_recipe.get("validation", {}))
    checks.append(
        PaperAuditCheck(
            "asr_manifest_recipe",
            int(asr_recipe.get("records", 0)) > 0 and bool(asr_validation.get("ok")),
            f"{asr_recipe.get('records', 0)} ASR record(s)",
        )
    )

    baselines = _dict(results["baselines"])
    baseline_ok = bool(baselines) and all(
        isinstance(payload, dict) and "classification" in payload and "interaction" in payload
        for payload in baselines.values()
    )
    checks.append(
        PaperAuditCheck(
            "baseline_results",
            baseline_ok,
            f"{len(baselines)} baseline(s)" if baseline_ok else "requires classification and interaction results",
        )
    )
    failure_ok = bool(baselines) and all(
        isinstance(payload, dict)
        and isinstance(payload.get("failure_analysis"), dict)
        and "category_counts" in payload["failure_analysis"]
        for payload in baselines.values()
    )
    checks.append(
        PaperAuditCheck(
            "failure_analysis",
            failure_ok,
            f"{len(baselines)} baseline failure report(s)" if failure_ok else "requires per-baseline failure_analysis",
        )
    )

    turn_benchmarks = _dict(results["turn_benchmarks"])
    benchmark_ok = bool(turn_benchmarks) and all(
        isinstance(payload, dict)
        and "avg_latency_ms" in payload
        and "p95_latency_ms" in payload
        and "rtf" in payload
        for payload in turn_benchmarks.values()
    )
    checks.append(
        PaperAuditCheck(
            "turn_benchmarks",
            benchmark_ok,
            f"{len(turn_benchmarks)} benchmark(s)" if benchmark_ok else "requires latency and rtf metrics",
        )
    )

    scenarios = _dict(results["scenarios"])
    by_scenario = _dict(scenarios.get("by_scenario", {}))
    scenario_ok = bool(by_scenario) and "factor_summary" in scenarios
    checks.append(
        PaperAuditCheck(
            "scenario_results",
            scenario_ok,
            f"{len(by_scenario)} scenario(s)" if scenario_ok else "requires by_scenario and factor_summary",
        )
    )

    policy = _dict(results["policy_search"])
    trials = policy.get("trials", [])
    policy_ok = "best" in policy and isinstance(trials, list) and bool(trials)
    checks.append(
        PaperAuditCheck(
            "policy_search",
            policy_ok,
            f"{len(trials)} trial(s)" if policy_ok else "requires best policy and non-empty trials",
        )
    )

    streaming = _dict(results["streaming_asr"])
    metrics = _dict(streaming.get("metrics", {}))
    required_metrics = (
        "wer",
        "cer",
        "rtf",
        "first_partial_latency",
        "final_latency",
        "endpoint_delay",
        "partial_revision_rate",
        "stable_prefix_ratio",
        "timestamp_drift",
    )
    missing_metrics = [metric for metric in required_metrics if metric not in metrics]
    checks.append(
        PaperAuditCheck(
            "streaming_metrics",
            not missing_metrics,
            "present" if not missing_metrics else "missing: " + ", ".join(missing_metrics),
        )
    )
    conversions = streaming.get("asr_transcript_conversions", [])
    conversion_ok = isinstance(conversions, list) and len(conversions) >= 2
    checks.append(
        PaperAuditCheck(
            "asr_transcript_conversions",
            conversion_ok,
            f"{len(conversions) if isinstance(conversions, list) else 0} conversion(s)",
        )
    )
    streaming_failures = _dict(metrics.get("failure_analysis"))
    checks.append(
        PaperAuditCheck(
            "streaming_failure_analysis",
            "category_counts" in streaming_failures,
            f"{streaming_failures.get('total_failures', 0)} failure(s)",
        )
    )
    return checks


def _repo_release_checks(repo_root: Path) -> list[PaperReleaseAuditCheck]:
    required = {
        "pyproject": "pyproject.toml",
        "manifest_in": "MANIFEST.in",
        "mkdocs_config": "mkdocs.yaml",
        "readme": "README.md",
        "license": "LICENSE",
        "contributing": "CONTRIBUTING.md",
        "security": "SECURITY.md",
        "code_of_conduct": "CODE_OF_CONDUCT.md",
        "roadmap": "ROADMAP.md",
        "roadmap_registry": "configs/roadmap/stable_asr_roadmap.json",
        "ci_workflow": ".github/workflows/tests.yml",
        "issue_template_config": ".github/ISSUE_TEMPLATE/config.yml",
        "issue_template_final_data": ".github/ISSUE_TEMPLATE/final_data_acquisition.yml",
        "issue_template_asr_adapter": ".github/ISSUE_TEMPLATE/asr_adapter.yml",
        "issue_template_voiceworld": ".github/ISSUE_TEMPLATE/voiceworld_scenario.yml",
        "issue_template_benchmark_submission": ".github/ISSUE_TEMPLATE/benchmark_submission.yml",
        "paper_config": "configs/paper/paper_smoke.json",
        "paper_parity_checklist": "configs/paper/paper_parity_checklist.json",
        "final_experiments": "configs/paper/final_experiments.json",
        "final_run_config": "configs/final/paper_final.json",
        "final_asr_command_config": "configs/final/asr_command_compare.json",
        "final_results_assembler": "stable_asr/paper/final_results.py",
        "final_whisper_export_bridge": "scripts/export_whisper_streaming.py",
        "final_funasr_export_bridge": "scripts/export_funasr_streaming.py",
        "benchmark_suite": "configs/benchmarks/stable_asr_v0.json",
        "schema_registry": "configs/schemas/stable_asr_schemas.json",
        "data_sources": "configs/datasets/stable_asr_sources.json",
        "adapter_registry": "configs/adapters/stable_asr_adapters.json",
        "asr_collections": "configs/references/asr_collections.json",
        "scenario_suite": "configs/scenarios/stable_asr_voiceworld_v0.json",
        "asr_manifest_schema": "stable_asr/data/asr_manifest.py",
        "json_schema_registry": "stable_asr/schemas.py",
        "asr_manifest_recipe": "stable_asr/data/recipes/asr_folder.py",
        "paper_script": "scripts/reproduce_paper.py",
    }
    return [
        _repo_path_check(repo_root, name, relative_path)
        for name, relative_path in required.items()
    ] + [
        _source_manifest_content_check(_repo_or_platform_path(repo_root, "MANIFEST.in")),
        _wheel_data_files_check(_repo_or_platform_path(repo_root, "pyproject.toml")),
        _ci_wheel_smoke_check(_repo_or_platform_path(repo_root, ".github/workflows/tests.yml")),
        _ci_lance_smoke_check(_repo_or_platform_path(repo_root, ".github/workflows/tests.yml")),
    ]


def _repo_path_check(repo_root: Path, name: str, relative_path: str) -> PaperReleaseAuditCheck:
    path = _repo_or_platform_path(repo_root, relative_path)
    return _release_check("software", name, path.exists(), _display_repo_path(repo_root, path))


def _repo_or_platform_path(repo_root: Path, relative_path: str | Path) -> Path:
    repo_path = repo_root / relative_path
    if repo_path.exists():
        return repo_path
    return resolve_platform_path(relative_path)


def _display_repo_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _source_manifest_content_check(path: Path) -> PaperReleaseAuditCheck:
    required_patterns = (
        "include README.md",
        "include ROADMAP.md",
        "include LICENSE",
        "include CONTRIBUTING.md",
        "include SECURITY.md",
        "include CODE_OF_CONDUCT.md",
        "include CITATION.cff",
        "include mkdocs.yaml",
        "recursive-include .github/workflows",
        "recursive-include configs",
        "recursive-include docs",
        "recursive-include examples",
        "recursive-include scripts",
        "recursive-include tests/fixtures",
    )
    if not path.exists():
        return _release_check("software", "source_manifest_content", False, f"missing: {path}")
    text = path.read_text(encoding="utf-8")
    missing = [pattern for pattern in required_patterns if pattern not in text]
    return _release_check(
        "software",
        "source_manifest_content",
        not missing,
        "covers platform assets" if not missing else "missing: " + ", ".join(missing),
    )


def _wheel_data_files_check(path: Path) -> PaperReleaseAuditCheck:
    required_patterns = (
        "[tool.setuptools.data-files]",
        "share/stable-asr",
        "mkdocs.yaml",
        "share/stable-asr/.github/workflows",
        "share/stable-asr/configs/adapters",
        "share/stable-asr/configs/benchmarks",
        "share/stable-asr/configs/datasets",
        "share/stable-asr/configs/final",
        "share/stable-asr/configs/paper",
        "share/stable-asr/configs/references",
        "share/stable-asr/configs/roadmap",
        "share/stable-asr/configs/scenarios",
        "share/stable-asr/configs/schemas",
        "share/stable-asr/docs",
        "share/stable-asr/docs/api",
        "share/stable-asr/docs/guides",
        "share/stable-asr/examples",
        "share/stable-asr/scripts",
        "share/stable-asr/tests/fixtures",
    )
    if not path.exists():
        return _release_check("software", "wheel_data_files", False, f"missing: {path}")
    text = path.read_text(encoding="utf-8")
    missing = [pattern for pattern in required_patterns if pattern not in text]
    return _release_check(
        "software",
        "wheel_data_files",
        not missing,
        "covers platform assets" if not missing else "missing: " + ", ".join(missing),
    )


def _ci_wheel_smoke_check(path: Path) -> PaperReleaseAuditCheck:
    required_patterns = (
        "Smoke test wheel install",
        "python -m pip wheel . --no-deps",
        "python -m venv",
        "stable-asr-wheel-venv/bin/stable-asr doctor",
        "stable-asr-wheel-venv/bin/stable-asr roadmap-status --roadmap configs/roadmap/stable_asr_roadmap.json --validate-only",
        "stable-asr-wheel-venv/bin/stable-asr asr-collections --registry configs/references/asr_collections.json --audit-coverage",
        "stable-asr-wheel-venv/bin/stable-asr paper-release-smoke --output-dir /tmp/stable-asr-wheel-release-smoke --episodes 9 --seed 6 --skip-train",
    )
    if not path.exists():
        return _release_check("software", "ci_wheel_smoke", False, f"missing: {path}")
    text = path.read_text(encoding="utf-8")
    missing = [pattern for pattern in required_patterns if pattern not in text]
    return _release_check(
        "software",
        "ci_wheel_smoke",
        not missing,
        "wheel install smoke is covered" if not missing else "missing: " + ", ".join(missing),
    )


def _ci_lance_smoke_check(path: Path) -> PaperReleaseAuditCheck:
    required_patterns = (
        "optional-data-backends",
        "python -m pip install -e \".[lance]\"",
        "stable-asr benchmark-data --dataset examples/data/turn_demo.jsonl --output-dir /tmp/stable-asr-data-backends --formats jsonl parquet lance --sample-count 16",
        "stable-asr paper-release-smoke --output-dir /tmp/stable-asr-lance-release-smoke --episodes 9 --seed 6 --skip-train",
    )
    if not path.exists():
        return _release_check("software", "ci_lance_smoke", False, f"missing: {path}")
    text = path.read_text(encoding="utf-8")
    missing = [pattern for pattern in required_patterns if pattern not in text]
    return _release_check(
        "software",
        "ci_lance_smoke",
        not missing,
        "optional Lance backend smoke is covered" if not missing else "missing: " + ", ".join(missing),
    )


def _release_result_checks(
    results: dict[str, object],
    *,
    suite: dict[str, Any] | None = None,
) -> list[PaperReleaseAuditCheck]:
    checks: list[PaperReleaseAuditCheck] = []
    required_sections = ("meta", "data", "baselines", "turn_benchmarks", "scenarios", "policy_search", "streaming_asr")
    missing = [section for section in required_sections if section not in results]
    checks.append(
        _release_check(
            "paper",
            "result_sections",
            not missing,
            "present" if not missing else "missing: " + ", ".join(missing),
        )
    )
    if missing:
        return checks

    data = _dict(results["data"])
    benchmark = _dict(data.get("benchmark", {}))
    rows = benchmark.get("rows", [])
    formats = {str(row.get("format")) for row in rows if isinstance(row, dict)}
    checks.append(
        _release_check(
            "data",
            "benchmark_formats",
            {"jsonl", "parquet"}.issubset(formats),
            "formats: " + ", ".join(sorted(formats)) if formats else "no benchmark formats",
        )
    )
    checks.append(
        _release_check(
            "data",
            "lance_data_layer",
            "lance" in formats,
            "requires completed Lance benchmark row for the platform paper",
        )
    )
    sampling_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and int(row.get("sample_count", 0)) > 0
        and float(row.get("samples_per_second", 0.0)) > 0.0
    ]
    checks.append(
        _release_check(
            "data",
            "random_sampling_benchmark",
            len(sampling_rows) >= 2,
            f"{len(sampling_rows)} sampled format row(s)",
        )
    )
    external_count = _external_conversion_count(data)
    checks.append(
        _release_check(
            "data",
            "external_data_sources",
            external_count >= 3,
            f"{external_count}/3 converted source(s)",
        )
    )
    asr_recipe = _dict(data.get("asr_manifest_recipe", {}))
    asr_validation = _dict(asr_recipe.get("validation", {}))
    checks.append(
        _release_check(
            "data",
            "asr_manifest_recipe",
            int(asr_recipe.get("records", 0)) > 0 and bool(asr_validation.get("ok")),
            f"{asr_recipe.get('records', 0)} ASR record(s)",
        )
    )

    baselines = _dict(results["baselines"])
    baseline_names = set(baselines)
    minimum_baselines = {"rule_endpoint", "vad_pause", "text_turn", "prediction_manifest"}
    checks.append(
        _release_check(
            "baseline",
            "minimum_baseline_set",
            minimum_baselines.issubset(baseline_names),
            "present: " + ", ".join(sorted(baseline_names)),
        )
    )
    failure_baselines = [
        name
        for name, payload in baselines.items()
        if isinstance(payload, dict)
        and isinstance(payload.get("failure_analysis"), dict)
        and "category_counts" in payload["failure_analysis"]
    ]
    checks.append(
        _release_check(
            "baseline",
            "failure_case_mining",
            minimum_baselines.issubset(set(failure_baselines)),
            f"{len(failure_baselines)} baseline failure report(s)",
        )
    )
    nanoturn = _dict(results.get("nanoturn", {}))
    has_nanoturn = any(name.startswith("nanoturn") for name in baseline_names)
    nanoturn_ok = has_nanoturn and nanoturn.get("status") == "completed"
    checks.append(
        _release_check(
            "baseline",
            "nanoturn_release_baseline",
            nanoturn_ok,
            "NanoTurn checkpoint metrics and baseline row present"
            if nanoturn_ok
            else "requires NanoTurn checkpoint metrics and baseline row in paper_results.json",
        )
    )

    benchmarks = _dict(results["turn_benchmarks"])
    checks.append(
        _release_check(
            "baseline",
            "latency_benchmarks",
            baseline_names.issubset(set(benchmarks)),
            f"{len(benchmarks)} benchmark row(s)",
        )
    )

    scenarios = _dict(results["scenarios"])
    by_scenario = _dict(scenarios.get("by_scenario", {}))
    required_scenarios = {
        "incomplete_pause",
        "backchannel",
        "wait_stop",
        "user_interruption",
        "side_conversation",
        "ambient_speech",
        "noisy_farfield",
        "code_switching",
    }
    missing_scenarios = sorted(required_scenarios.difference(by_scenario))
    checks.append(
        _release_check(
            "scenario",
            "scenario_suite_coverage",
            not missing_scenarios,
            "covered" if not missing_scenarios else "missing: " + ", ".join(missing_scenarios),
        )
    )
    factor_summary = _dict(scenarios.get("factor_summary", {}))
    required_factors = {
        "snr_db",
        "reverb",
        "speaking_rate",
        "overlap_offset_ms",
        "network_jitter_ms",
        "farfield_distance_m",
        "code_switch_ratio",
        "accent",
    }
    missing_factors = sorted(required_factors.difference(factor_summary))
    checks.append(
        _release_check(
            "scenario",
            "factor_summary",
            not missing_factors,
            "present" if not missing_factors else "missing: " + ", ".join(missing_factors),
        )
    )

    policy = _dict(results["policy_search"])
    trials = policy.get("trials", [])
    checks.append(
        _release_check(
            "policy",
            "policy_search_trials",
            isinstance(trials, list) and len(trials) >= 24,
            f"{len(trials) if isinstance(trials, list) else 0} trial(s)",
        )
    )

    streaming = _dict(results["streaming_asr"])
    streaming_metrics = _dict(streaming.get("metrics", {}))
    required_streaming = {
        "wer",
        "cer",
        "rtf",
        "first_partial_latency",
        "final_latency",
        "endpoint_delay",
        "partial_revision_rate",
        "stable_prefix_ratio",
        "timestamp_drift",
    }
    missing_streaming = sorted(required_streaming.difference(streaming_metrics))
    checks.append(
        _release_check(
            "streaming",
            "streaming_metrics",
            not missing_streaming,
            "present" if not missing_streaming else "missing: " + ", ".join(missing_streaming),
        )
    )
    streaming_failures = _dict(streaming_metrics.get("failure_analysis"))
    checks.append(
        _release_check(
            "streaming",
            "streaming_failure_mining",
            "category_counts" in streaming_failures,
            f"{streaming_failures.get('total_failures', 0)} failure(s)",
        )
    )
    comparison = _dict(streaming.get("adapter_comparison", {}))
    comparison_rows = comparison.get("rows", [])
    comparison_count = len(comparison_rows) if isinstance(comparison_rows, list) else 0
    checks.append(
        _release_check(
            "streaming",
            "adapter_comparison",
            comparison_count >= 2,
            f"{comparison_count} adapter row(s)",
        )
    )
    sweep = _dict(streaming.get("schedule_sweep", {}))
    sweep_rows = sweep.get("rows", [])
    sweep_count = len(sweep_rows) if isinstance(sweep_rows, list) else 0
    checks.append(
        _release_check(
            "streaming",
            "schedule_sweep",
            sweep_count >= 4,
            f"{sweep_count} sweep row(s)",
        )
    )
    asr_conversions = streaming.get("asr_transcript_conversions", [])
    asr_conversion_count = len(asr_conversions) if isinstance(asr_conversions, list) else 0
    checks.append(
        _release_check(
            "streaming",
            "asr_transcript_conversions",
            asr_conversion_count >= 2,
            f"{asr_conversion_count} converted ASR transcript schema(s)",
        )
    )
    command_adapter = _dict(streaming.get("command_adapter", {}))
    command_metrics = _dict(command_adapter.get("metrics"))
    checks.append(
        _release_check(
            "streaming",
            "command_adapter",
            int(command_metrics.get("records", 0)) > 0,
            f"{command_metrics.get('records', 0)} record(s)",
        )
    )
    coverage = audit_benchmark_suite_coverage(results, suite=suite)
    detail = (
        f"{coverage.rows} leaderboard row(s)"
        if coverage.ok
        else "; ".join(coverage.missing[:5])
    )
    checks.append(
        _release_check(
            "paper",
            "benchmark_suite_coverage",
            coverage.ok,
            detail,
        )
    )
    return checks


def _artifact_checks(artifacts_dir: Path, *, results_path: Path) -> list[PaperAuditCheck]:
    checks: list[PaperAuditCheck] = []
    checks.append(_exists_check("artifact_index", artifacts_dir / "ARTIFACT_INDEX.md"))
    checks.append(_exists_check("artifact_manifest", artifacts_dir / "artifact_manifest.json"))
    checks.append(_results_copy_check(results_path=Path(results_path), artifacts_dir=artifacts_dir))
    for table in PAPER_TABLES:
        checks.append(_exists_check(f"table:{table}", artifacts_dir / "tables" / f"{table}.md"))
    for figure in PAPER_FIGURES:
        checks.append(_exists_check(f"figure:{figure}", artifacts_dir / "figures" / f"{figure}.svg"))
    checks.append(_exists_check("leaderboard:jsonl", artifacts_dir / "leaderboard.jsonl"))
    checks.append(_exists_check("leaderboard:csv", artifacts_dir / "leaderboard.csv"))
    checks.append(_exists_check("leaderboard_validation:json", artifacts_dir / "leaderboard_validation.json"))
    checks.append(_exists_check("leaderboard_validation:markdown", artifacts_dir / "LEADERBOARD_VALIDATION.md"))
    checks.append(_exists_check("leaderboard_report:json", artifacts_dir / "leaderboard_report.json"))
    checks.append(_exists_check("leaderboard_report:markdown", artifacts_dir / "LEADERBOARD_REPORT.md"))
    checks.append(_exists_check("artifact_integrity:json", artifacts_dir / "artifact_hashes.json"))
    checks.append(_exists_check("artifact_integrity:markdown", artifacts_dir / "ARTIFACT_HASHES.md"))
    checks.append(_integrity_check(artifacts_dir))
    checks.append(_exists_check("provenance:json", artifacts_dir / "provenance.json"))
    checks.append(_exists_check("provenance:markdown", artifacts_dir / "PROVENANCE.md"))
    checks.append(_exists_check("benchmark_suite:json", artifacts_dir / "benchmark_suite.json"))
    checks.append(_exists_check("benchmark_suite:markdown", artifacts_dir / "BENCHMARK_SUITE.md"))
    checks.append(_benchmark_required_artifacts_check(artifacts_dir))
    checks.append(
        _exists_check("starter_pack:benchmark_manifest", artifacts_dir / "starter_packs" / "benchmark_pack" / "manifest.json")
    )
    checks.append(
        _exists_check("starter_pack:benchmark_readme", artifacts_dir / "starter_packs" / "benchmark_pack" / "README.md")
    )
    checks.append(
        _exists_check("starter_pack:adapter_manifest", artifacts_dir / "starter_packs" / "adapter_pack" / "manifest.json")
    )
    checks.append(
        _exists_check("starter_pack:adapter_readme", artifacts_dir / "starter_packs" / "adapter_pack" / "README.md")
    )
    checks.append(
        _exists_check("starter_pack:scenario_manifest", artifacts_dir / "starter_packs" / "scenario_pack" / "manifest.json")
    )
    checks.append(
        _exists_check("starter_pack:scenario_readme", artifacts_dir / "starter_packs" / "scenario_pack" / "README.md")
    )
    checks.append(
        _exists_check("starter_pack:final_manifest", artifacts_dir / "starter_packs" / "final_pack" / "manifest.json")
    )
    checks.append(
        _exists_check("starter_pack:final_readme", artifacts_dir / "starter_packs" / "final_pack" / "README.md")
    )
    checks.append(
        _exists_check(
            "starter_pack:final_acquisition_manifest",
            artifacts_dir / "starter_packs" / "final_acquisition_pack" / "manifest.json",
        )
    )
    checks.append(
        _exists_check(
            "starter_pack:final_acquisition_checklist",
            artifacts_dir / "starter_packs" / "final_acquisition_pack" / "acquisition" / "staging_checklist.tsv",
        )
    )
    checks.append(_exists_check("data_sources:json", artifacts_dir / "data_sources.json"))
    checks.append(_exists_check("data_sources:markdown", artifacts_dir / "DATA_SOURCES.md"))
    checks.append(_exists_check("adapter_registry:json", artifacts_dir / "adapter_registry.json"))
    checks.append(_exists_check("adapter_registry:markdown", artifacts_dir / "ADAPTERS.md"))
    checks.append(_exists_check("model_registry:json", artifacts_dir / "model_registry.json"))
    checks.append(_exists_check("model_registry:markdown", artifacts_dir / "MODELS.md"))
    checks.append(_exists_check("model_card:json", artifacts_dir / "model_card.json"))
    checks.append(_exists_check("model_card:markdown", artifacts_dir / "MODEL_CARD.md"))
    checks.append(_exists_check("schema_registry:json", artifacts_dir / "schema_registry.json"))
    checks.append(_exists_check("schema_registry:markdown", artifacts_dir / "SCHEMAS.md"))
    checks.append(_exists_check("asr_collections:json", artifacts_dir / "asr_collections.json"))
    checks.append(_exists_check("asr_collections:markdown", artifacts_dir / "ASR_COLLECTIONS.md"))
    checks.append(_exists_check("asr_collections:paper_markdown", artifacts_dir / "ASR_REFERENCES.md"))
    checks.append(_exists_check("asr_collections:bibtex", artifacts_dir / "ASR_REFERENCES.bib"))
    checks.append(_exists_check("asr_collection_coverage:json", artifacts_dir / "asr_collection_coverage.json"))
    checks.append(_exists_check("asr_collection_coverage:markdown", artifacts_dir / "ASR_COLLECTION_COVERAGE.md"))
    checks.append(_exists_check("asr_collection_readiness:json", artifacts_dir / "asr_collection_readiness.json"))
    checks.append(_exists_check("asr_collection_readiness:markdown", artifacts_dir / "ASR_COLLECTION_READINESS.md"))
    checks.append(_exists_check("scenario_suite:json", artifacts_dir / "scenario_suite.json"))
    checks.append(_exists_check("scenario_suite:markdown", artifacts_dir / "SCENARIO_SUITE.md"))
    checks.append(_exists_check("case_studies:json", artifacts_dir / "case_studies.json"))
    checks.append(_exists_check("case_studies:markdown", artifacts_dir / "CASE_STUDIES.md"))
    checks.append(_exists_check("paper_parity:json", artifacts_dir / "paper_parity.json"))
    checks.append(_exists_check("paper_parity:markdown", artifacts_dir / "PAPER_PARITY.md"))
    checks.append(_exists_check("final_experiments:json", artifacts_dir / "final_experiments.json"))
    checks.append(_exists_check("final_experiments:markdown", artifacts_dir / "FINAL_EXPERIMENTS.md"))
    checks.append(_exists_check("final_input_collections:json", artifacts_dir / "final_input_collections.json"))
    checks.append(_exists_check("final_input_collections:audit_json", artifacts_dir / "final_input_collection_status.json"))
    checks.append(_exists_check("final_input_collections:markdown", artifacts_dir / "FINAL_INPUT_COLLECTIONS.md"))
    checks.append(_exists_check("final_run_config:json", artifacts_dir / "final_run_config.json"))
    checks.append(_exists_check("final_run_config:markdown", artifacts_dir / "FINAL_RUN_CONFIG.md"))
    checks.append(_exists_check("final_run_file_audit:json", artifacts_dir / "final_run_file_audit.json"))
    checks.append(_exists_check("final_run_file_audit:markdown", artifacts_dir / "FINAL_RUN_FILE_AUDIT.md"))
    checks.append(_exists_check("final_run_action_plan:json", artifacts_dir / "final_run_action_plan.json"))
    checks.append(_exists_check("final_run_action_plan:markdown", artifacts_dir / "FINAL_RUN_ACTION_PLAN.md"))
    checks.append(_exists_check("final_evidence_matrix:json", artifacts_dir / "final_evidence_matrix.json"))
    checks.append(_exists_check("final_evidence_matrix:markdown", artifacts_dir / "FINAL_EVIDENCE_MATRIX.md"))
    checks.append(_exists_check("paper_status:json", artifacts_dir / "paper_status.json"))
    checks.append(_exists_check("paper_status:markdown", artifacts_dir / "PAPER_STATUS.md"))
    checks.append(_exists_check("roadmap_status:json", artifacts_dir / "roadmap_status.json"))
    checks.append(_exists_check("roadmap_status:markdown", artifacts_dir / "ROADMAP_STATUS.md"))
    checks.append(_exists_check("claims:json", artifacts_dir / "claims.json"))
    checks.append(_exists_check("claims:markdown", artifacts_dir / "CLAIMS.md"))
    return checks


def _exists_check(name: str, path: Path) -> PaperAuditCheck:
    return PaperAuditCheck(name, path.exists(), str(path))


def _results_copy_check(*, results_path: Path, artifacts_dir: Path) -> PaperAuditCheck:
    artifact_path = artifacts_dir / "paper_results.json"
    if not artifact_path.exists():
        return PaperAuditCheck("results:json", False, f"missing: {artifact_path}")
    if not results_path.exists():
        return PaperAuditCheck("results:json", False, f"source missing: {results_path}")
    source_sha = _sha256_file(results_path)
    artifact_sha = _sha256_file(artifact_path)
    return PaperAuditCheck(
        "results:json",
        source_sha == artifact_sha,
        str(artifact_path) if source_sha == artifact_sha else f"hash mismatch: {artifact_path}",
    )


def _benchmark_required_artifacts_check(artifacts_dir: Path) -> PaperAuditCheck:
    suite_path = artifacts_dir / "benchmark_suite.json"
    if not suite_path.exists():
        return PaperAuditCheck("benchmark_suite:required_artifacts", False, f"missing: {suite_path}")
    try:
        report = audit_benchmark_required_artifacts(artifacts_dir, suite=load_benchmark_suite(suite_path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return PaperAuditCheck("benchmark_suite:required_artifacts", False, str(exc))
    detail = (
        f"{len(report.present)} required artifact(s)"
        if report.ok
        else "missing: " + ", ".join(report.missing[:5])
    )
    return PaperAuditCheck("benchmark_suite:required_artifacts", report.ok, detail)


def _integrity_check(artifacts_dir: Path) -> PaperAuditCheck:
    manifest = artifacts_dir / "artifact_hashes.json"
    if not manifest.exists():
        return PaperAuditCheck("artifact_integrity:sha256", False, f"missing: {manifest}")
    try:
        report = verify_artifact_integrity(manifest, root=artifacts_dir)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        return PaperAuditCheck("artifact_integrity:sha256", False, str(exc))
    if report.ok:
        detail = f"{len(report.files)} file(s) verified"
    else:
        parts = []
        if report.missing:
            parts.append("missing: " + ", ".join(report.missing[:3]))
        if report.mismatched:
            parts.append("mismatched: " + ", ".join(report.mismatched[:3]))
        detail = "; ".join(parts)
    return PaperAuditCheck("artifact_integrity:sha256", report.ok, detail)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _optional_path_check(gate: str, name: str, path: str | Path | None) -> PaperReleaseAuditCheck:
    if path is None:
        return _release_check(gate, name, False, "not provided")
    path = Path(path)
    return _release_check(gate, name, path.exists(), str(path))


def _release_check(gate: str, name: str, ok: bool, detail: str) -> PaperReleaseAuditCheck:
    return PaperReleaseAuditCheck(gate=gate, name=name, ok=ok, detail=detail)


def _external_conversion_count(data: dict[str, Any]) -> int:
    external = data.get("external_conversions", data.get("external_conversion"))
    if isinstance(external, list):
        return len(external)
    if isinstance(external, dict) and external:
        return 1
    return 0


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
