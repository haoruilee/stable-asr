"""Stable-worldmodel-style paper parity checklist and audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.resources import resolve_platform_path
from stable_asr.paper.tables import load_paper_results


DEFAULT_PARITY_CHECKLIST: dict[str, Any] = {
    "id": "stable_asr_paper_parity_v0",
    "version": "0.1.0",
    "title": "Stable-ASR Platform Paper Parity Checklist",
    "description": (
        "A stable-worldmodel-style paper checklist that maps platform-paper "
        "claims to repository files, result keys, paper artifacts, reproduction "
        "commands, and final-scale experiment gaps."
    ),
    "target_paper": "Stable-ASR: A Platform for Reproducible Real-Time ASR and Full-Duplex Turn-Taking Research and Evaluation",
    "items": [
        {
            "id": "problem_and_scope",
            "title": "Fragmentation Problem And Scope",
            "paper_section": "Introduction",
            "stable_worldmodel_analogy": "fragmented world-model codebases, data pipelines, and evaluation protocols",
            "repo_paths": ["README.md", "ROADMAP.md", "docs/index.md"],
            "result_keys": ["meta.artifact_version"],
            "artifact_paths": ["ARTIFACT_INDEX.md", "paper_results.json", "artifact_manifest.json", "PROVENANCE.md"],
            "commands": ["stable-asr paper-draft --results runs/paper/smoke/paper_results.json --output runs/paper/smoke/PAPER_DRAFT.md"],
            "final_scale_requirements": [
                "write the final related-work comparison against ASR toolkits, streaming ASR evaluation, turn-taking models, and full-duplex benchmarks",
                "include a limitation statement that separates structural smoke evidence from final benchmark claims"
            ]
        },
        {
            "id": "data_layer",
            "title": "Unified Data Layer",
            "paper_section": "Data Layer",
            "stable_worldmodel_analogy": "Lance-backed multi-format data layer for reproducible sequence-window sampling",
            "repo_paths": [
                "stable_asr/data/manifest.py",
                "stable_asr/data/asr_manifest.py",
                "stable_asr/data/registry.py",
                "stable_asr/data/formats/lance.py",
                "configs/datasets/stable_asr_sources.json"
            ],
            "result_keys": ["data.benchmark", "data.asr_manifest_recipe", "data.external_conversions"],
            "artifact_paths": ["tables/data.md", "tables/asr_manifest_recipe.md", "DATA_SOURCES.md", "data_sources.json"],
            "commands": [
                "stable-asr benchmark-data --dataset examples/data/turn_demo.jsonl --output-dir runs/data_bench --formats jsonl parquet lance --sample-count 16",
                "stable-asr prepare-asr-manifest --input examples/data/asr_metadata.tsv --output runs/asr_manifest.jsonl"
            ],
            "final_scale_requirements": [
                "run JSONL, Parquet, and Lance benchmarks on at least one real public ASR corpus",
                "measure random audio-window sampling throughput on NVMe and object storage",
                "report conversion time, storage size, read throughput, and multi-worker dataloader utilization"
            ]
        },
        {
            "id": "baseline_zoo",
            "title": "Baseline Zoo And Adapter Registry",
            "paper_section": "Baselines and Adapters",
            "stable_worldmodel_analogy": "modern baselines exposed through one training/evaluation interface",
            "repo_paths": [
                "stable_asr/models/baselines/rule_endpoint.py",
                "stable_asr/models/baselines/vad_pause.py",
                "stable_asr/models/baselines/text_turn.py",
                "stable_asr/turn/nanoturn.py",
                "stable_asr/models/adapters/registry.py",
                "configs/adapters/stable_asr_adapters.json"
            ],
            "result_keys": ["baselines", "turn_benchmarks", "nanoturn"],
            "artifact_paths": ["tables/baselines.md", "tables/turn_benchmark.md", "ADAPTERS.md", "adapter_registry.json"],
            "commands": [
                "stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --baseline vad_pause",
                "stable-asr adapter-registry --registry configs/adapters/stable_asr_adapters.json --validate-only"
            ],
            "final_scale_requirements": [
                "run at least two external turn systems through prediction-manifest or command-backed adapters",
                "include NanoTurn checkpoint-backed results trained on non-toy data",
                "report CPU and ONNX latency for every paper baseline"
            ]
        },
        {
            "id": "voiceworld_scenarios",
            "title": "Controllable VoiceWorld Scenario Suite",
            "paper_section": "Scenario Suite",
            "stable_worldmodel_analogy": "environment suite with controllable factors of variation",
            "repo_paths": [
                "stable_asr/scenarios/synthetic_turn.py",
                "stable_asr/scenarios/voice_world.py",
                "stable_asr/scenarios/suites.py",
                "configs/scenarios/stable_asr_voiceworld_v0.json"
            ],
            "result_keys": ["scenarios.by_scenario", "scenarios.factor_summary"],
            "artifact_paths": ["tables/scenarios.md", "SCENARIO_SUITE.md", "scenario_suite.json", "figures/robustness_heatmap.svg"],
            "commands": ["stable-asr eval-scenario --episodes 21 --seed 0 --baseline vad_pause"],
            "final_scale_requirements": [
                "compose or collect real audio examples for interruption, backchannel, side speech, ambient speech, far-field noise, and code-switching",
                "run scenario robustness across held-out speakers, languages, accents, SNR, reverb, and overlap offsets",
                "include qualitative case studies with audio snippets or reproducible record ids"
            ]
        },
        {
            "id": "policy_solver",
            "title": "Policy Solver And Cost-Sensitive Decisions",
            "paper_section": "Policy Layer",
            "stable_worldmodel_analogy": "planning and solver layer separated from learned model outputs",
            "repo_paths": ["stable_asr/turn/policy.py", "stable_asr/turn/solver.py"],
            "result_keys": ["policy_search.best", "policy_search.trials"],
            "artifact_paths": ["tables/policy.md", "figures/policy_state_machine.svg"],
            "commands": ["stable-asr optimize-policy --dataset examples/data/turn_demo.jsonl --baseline vad_pause"],
            "final_scale_requirements": [
                "sweep cost matrices for false complete, missed interruption, false interruption, backchannel break, and latency",
                "evaluate policy transfer from synthetic scenarios to held-out real interaction traces",
                "plot false-complete versus latency and missed-interruption versus false-interruption trade-offs"
            ]
        },
        {
            "id": "streaming_asr_eval",
            "title": "Streaming ASR Evaluation Beyond WER",
            "paper_section": "Streaming ASR Evaluation",
            "stable_worldmodel_analogy": "evaluation protocol that exposes behavior hidden by static metrics",
            "repo_paths": [
                "stable_asr/streaming/metrics.py",
                "stable_asr/streaming/compare.py",
                "stable_asr/streaming/sweep.py",
                "stable_asr/streaming/failures.py"
            ],
            "result_keys": [
                "streaming_asr.metrics",
                "streaming_asr.adapter_comparison",
                "streaming_asr.schedule_sweep",
                "streaming_asr.command_adapter"
            ],
            "artifact_paths": ["tables/streaming.md", "tables/streaming_failures.md", "tables/streaming_sweep.md"],
            "commands": [
                "stable-asr eval-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl",
                "stable-asr compare-asr-commands --config examples/configs/asr_command_compare_demo.json"
            ],
            "final_scale_requirements": [
                "evaluate at least two real ASR systems under the same streaming trace schema",
                "run chunk-size and lookahead sweeps on real streaming outputs",
                "report WER/CER together with RTF, first partial latency, final latency, endpoint delay, partial revisions, stable prefix, and timestamp drift"
            ]
        },
        {
            "id": "paper_artifacts",
            "title": "One-Command Paper Artifacts",
            "paper_section": "Reproducibility",
            "stable_worldmodel_analogy": "reproducible platform release with scripts, reports, figures, and benchmark assets",
            "repo_paths": [
                "configs/paper/paper_smoke.json",
                "scripts/reproduce_paper.py",
                "stable_asr/paper/artifacts.py",
                "stable_asr/paper/audit.py",
                "stable_asr/paper/provenance.py",
                "stable_asr/paper/draft.py",
                "stable_asr/paper/latex.py"
            ],
            "result_keys": ["meta.artifact_version"],
            "artifact_paths": [
                "ARTIFACT_INDEX.md",
                "paper_results.json",
                "artifact_manifest.json",
                "artifact_hashes.json",
                "PROVENANCE.md",
                "leaderboard.jsonl",
                "BENCHMARK_SUITE.md"
            ],
            "commands": [
                "stable-asr reproduce-paper --config configs/paper/paper_smoke.json",
                "stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts",
                "stable-asr paper-release-audit --repo-root . --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts"
            ],
            "final_scale_requirements": [
                "archive the final benchmark result bundle and exact configs used for the paper",
                "publish dataset and experiment cards for every non-toy dataset and trained model",
                "run a clean reproduction from a fresh environment before submission"
            ]
        }
    ]
}


@dataclass(frozen=True)
class PaperParityValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "paper_parity_checklist: OK"
        return "paper_parity_checklist: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class PaperParityItemCheck:
    item_id: str
    ok: bool
    missing: list[str]
    final_scale_requirements: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "ok": self.ok,
            "missing": self.missing,
            "final_scale_requirements": self.final_scale_requirements,
        }


@dataclass(frozen=True)
class PaperParityAuditReport:
    ok: bool
    final_ready: bool
    checks: list[PaperParityItemCheck]
    checklist: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "final_ready": self.final_ready,
            "checks": [check.to_dict() for check in self.checks],
            "checklist": self.checklist,
        }

    def to_text(self) -> str:
        lines = [
            f"paper_parity_audit: {'OK' if self.ok else 'MISSING'}",
            f"final_scale_ready: {'YES' if self.final_ready else 'NO'}",
        ]
        for check in self.checks:
            status = "OK" if check.ok else "MISSING"
            missing = "covered" if check.ok else "; ".join(check.missing)
            gap_count = len(check.final_scale_requirements)
            lines.append(f"- {status} {check.item_id}: {missing}; final_gap_count={gap_count}")
        return "\n".join(lines)


def load_paper_parity_checklist(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_PARITY_CHECKLIST))
    with resolve_platform_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("paper parity checklist must be a JSON object")
    return payload


def write_paper_parity_checklist_json(path: str | Path, checklist: dict[str, Any] | None = None) -> str:
    checklist = checklist or load_paper_parity_checklist()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checklist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_paper_parity_checklist(checklist: dict[str, Any]) -> PaperParityValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "items"):
        if key not in checklist:
            errors.append(f"missing top-level key: {key}")
    items = checklist.get("items")
    if not isinstance(items, list) or not items:
        errors.append("items must be a non-empty list")
        return PaperParityValidation(ok=False, errors=errors)

    seen: set[str] = set()
    required = {
        "id",
        "title",
        "paper_section",
        "stable_worldmodel_analogy",
        "repo_paths",
        "result_keys",
        "artifact_paths",
        "commands",
        "final_scale_requirements",
    }
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"item {index} must be an object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"item {index} missing id")
        elif item_id in seen:
            errors.append(f"duplicate item id: {item_id}")
        else:
            seen.add(item_id)
        for key in required:
            if key not in item:
                errors.append(f"item {item_id or index} missing {key}")
        for key in ("repo_paths", "result_keys", "artifact_paths", "commands", "final_scale_requirements"):
            if key in item and not isinstance(item[key], list):
                errors.append(f"item {item_id or index} {key} must be a list")
    return PaperParityValidation(ok=not errors, errors=errors)


def audit_paper_parity(
    *,
    checklist: dict[str, Any] | None = None,
    repo_root: str | Path = ".",
    results_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> PaperParityAuditReport:
    checklist = checklist or load_paper_parity_checklist()
    validation = validate_paper_parity_checklist(checklist)
    if not validation.ok:
        checks = [
            PaperParityItemCheck(
                item_id="checklist_schema",
                ok=False,
                missing=validation.errors,
                final_scale_requirements=[],
            )
        ]
        return PaperParityAuditReport(ok=False, final_ready=False, checks=checks, checklist=checklist)

    repo_root = Path(repo_root)
    results = _load_results(results_path)
    artifacts = Path(artifacts_dir) if artifacts_dir is not None else None
    checks = [
        _audit_item(item, repo_root=repo_root, results=results, artifacts_dir=artifacts)
        for item in checklist["items"]
    ]
    final_ready = all(check.ok and not check.final_scale_requirements for check in checks)
    return PaperParityAuditReport(
        ok=all(check.ok for check in checks),
        final_ready=final_ready,
        checks=checks,
        checklist=checklist,
    )


def paper_parity_markdown(report: PaperParityAuditReport) -> str:
    rows = []
    items_by_id = {str(item["id"]): item for item in report.checklist.get("items", []) if isinstance(item, dict)}
    for check in report.checks:
        item = items_by_id.get(check.item_id, {})
        rows.append(
            {
                "item": check.item_id,
                "section": item.get("paper_section", ""),
                "status": "OK" if check.ok else "MISSING",
                "final_gaps": len(check.final_scale_requirements),
                "missing": "; ".join(check.missing),
            }
        )

    lines = [
        f"# {report.checklist.get('title', 'Stable-ASR Paper Parity Checklist')}",
        "",
        f"- id: `{report.checklist.get('id', '')}`",
        f"- version: `{report.checklist.get('version', '')}`",
        f"- structural audit: `{'OK' if report.ok else 'MISSING'}`",
        f"- final-scale ready: `{'YES' if report.final_ready else 'NO'}`",
        "",
        str(report.checklist.get("description", "")),
        "",
        "## Audit",
        "",
        dict_table(rows),
        "",
        "## Final-Scale Requirements",
        "",
    ]
    for check in report.checks:
        item = items_by_id.get(check.item_id, {})
        lines.extend(
            [
                f"### {check.item_id}",
                "",
                str(item.get("title", "")),
                "",
            ]
        )
        if check.final_scale_requirements:
            lines.extend(f"- {requirement}" for requirement in check.final_scale_requirements)
        else:
            lines.append("- No remaining final-scale requirements recorded.")
        lines.append("")
    return "\n".join(lines)


def _audit_item(
    item: dict[str, Any],
    *,
    repo_root: Path,
    results: dict[str, Any] | None,
    artifacts_dir: Path | None,
) -> PaperParityItemCheck:
    missing: list[str] = []
    for relative in item.get("repo_paths", []):
        if not (repo_root / str(relative)).exists():
            missing.append(f"repo:{relative}")
    for key in item.get("result_keys", []):
        if results is None or not _has_nested_key(results, str(key)):
            missing.append(f"result:{key}")
    for relative in item.get("artifact_paths", []):
        if artifacts_dir is None or not (artifacts_dir / str(relative)).exists():
            missing.append(f"artifact:{relative}")
    return PaperParityItemCheck(
        item_id=str(item["id"]),
        ok=not missing,
        missing=missing,
        final_scale_requirements=[str(value) for value in item.get("final_scale_requirements", [])],
    )


def _load_results(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return load_paper_results(path)


def _has_nested_key(payload: dict[str, Any], dotted_key: str) -> bool:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True
