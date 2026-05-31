"""Claim-to-artifact audit matrix for the Stable-ASR platform paper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.paper.tables import load_paper_results


DEFAULT_CLAIMS: list[dict[str, Any]] = [
    {
        "id": "data_layer",
        "claim": "Stable-ASR standardizes ASR, turn-taking, and streaming-trace data with registered storage backends and public-corpus recipes.",
        "repo_paths": [
            "stable_asr/data/manifest.py",
            "stable_asr/data/asr_manifest.py",
            "stable_asr/data/registry.py",
            "stable_asr/data/formats/lance.py",
            "stable_asr/data/recipes/asr_folder.py",
        ],
        "result_keys": ["data.benchmark", "data.asr_manifest_recipe", "data.external_conversions"],
        "artifact_paths": ["tables/data.md", "tables/asr_manifest_recipe.md", "DATA_SOURCES.md"],
        "commands": [
            "stable-asr prepare-asr-manifest --input examples/data/asr_metadata.tsv --output runs/asr_manifest.jsonl",
            "stable-asr benchmark-data --dataset examples/data/turn_demo.jsonl --output-dir runs/data_bench --formats jsonl parquet lance --sample-count 16",
        ],
    },
    {
        "id": "baseline_zoo",
        "claim": "Stable-ASR makes rule, VAD, text, external prediction, and NanoTurn baselines comparable under one turn-evaluation interface.",
        "repo_paths": [
            "stable_asr/eval/turn_eval.py",
            "stable_asr/models/baselines/rule_endpoint.py",
            "stable_asr/models/baselines/vad_pause.py",
            "stable_asr/models/baselines/text_turn.py",
            "stable_asr/turn/nanoturn.py",
            "stable_asr/train/turn_trainer.py",
            "stable_asr/models/adapters/registry.py",
            "configs/adapters/stable_asr_adapters.json",
        ],
        "result_keys": ["baselines", "turn_benchmarks", "nanoturn"],
        "artifact_paths": [
            "tables/baselines.md",
            "tables/turn_benchmark.md",
            "ADAPTERS.md",
            "adapter_registry.json",
            "figures/latency_quality_pareto.svg",
        ],
        "commands": [
            "stable-asr eval-turn --dataset examples/data/turn_demo.jsonl --baseline vad_pause",
            "stable-asr train-turn --dataset examples/data/turn_demo.jsonl --output-dir runs/nanoturn",
        ],
    },
    {
        "id": "voiceworld_scenarios",
        "claim": "Stable-ASR provides seedable VoiceWorld scenarios and controllable factors for full-duplex robustness evaluation.",
        "repo_paths": [
            "stable_asr/scenarios/synthetic_turn.py",
            "stable_asr/scenarios/voice_world.py",
            "stable_asr/scenarios/suites.py",
            "configs/scenarios/stable_asr_voiceworld_v0.json",
        ],
        "result_keys": ["scenarios.by_scenario", "scenarios.factor_summary"],
        "artifact_paths": [
            "tables/scenarios.md",
            "SCENARIO_SUITE.md",
            "scenario_suite.json",
            "figures/robustness_heatmap.svg",
            "figures/voiceworld_timeline.svg",
        ],
        "commands": ["stable-asr eval-scenario --episodes 21 --seed 0 --baseline vad_pause"],
    },
    {
        "id": "policy_solver",
        "claim": "Stable-ASR separates model probabilities from deployment actions through turn policies and cost-sensitive threshold search.",
        "repo_paths": ["stable_asr/turn/policy.py", "stable_asr/turn/solver.py"],
        "result_keys": ["policy_search.best", "policy_search.trials"],
        "artifact_paths": ["tables/policy.md", "figures/policy_state_machine.svg"],
        "commands": ["stable-asr optimize-policy --dataset examples/data/turn_demo.jsonl --baseline vad_pause"],
    },
    {
        "id": "streaming_asr_eval",
        "claim": "Stable-ASR evaluates streaming ASR with real-time metrics, adapter comparisons, chunk/lookahead sweeps, and failure mining beyond WER/CER.",
        "repo_paths": [
            "stable_asr/streaming/metrics.py",
            "stable_asr/streaming/compare.py",
            "stable_asr/streaming/sweep.py",
            "stable_asr/streaming/failures.py",
        ],
        "result_keys": [
            "streaming_asr.metrics",
            "streaming_asr.adapter_comparison",
            "streaming_asr.schedule_sweep",
        ],
        "artifact_paths": ["tables/streaming.md", "tables/streaming_failures.md", "tables/streaming_sweep.md"],
        "commands": [
            "stable-asr eval-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl",
            "stable-asr sweep-streaming-asr --input tests/fixtures/streaming_asr_sample.jsonl",
        ],
    },
    {
        "id": "external_adapters",
        "claim": "Stable-ASR connects external turn datasets, ASR transcript exports, and command-backed ASR systems without forcing a single model stack.",
        "repo_paths": [
            "stable_asr/data/converters/external.py",
            "stable_asr/data/converters/streaming_asr.py",
            "stable_asr/models/adapters/command.py",
            "stable_asr/models/adapters/registry.py",
            "stable_asr/streaming/command_compare.py",
            "configs/adapters/stable_asr_adapters.json",
        ],
        "result_keys": [
            "data.external_conversions",
            "streaming_asr.asr_transcript_conversions",
            "streaming_asr.command_adapter",
        ],
        "artifact_paths": ["tables/asr_transcript_conversions.md", "ADAPTERS.md", "adapter_registry.json", "leaderboard.jsonl"],
        "commands": [
            "stable-asr convert-external --schema easyturn --input tests/fixtures/easyturn_sample.jsonl --output runs/easyturn.jsonl",
            "stable-asr compare-asr-commands --config examples/configs/asr_command_compare_demo.json",
        ],
    },
    {
        "id": "case_studies",
        "claim": "Stable-ASR turns aggregate failures into paper-ready case studies linked to source manifest and transcript records.",
        "repo_paths": [
            "stable_asr/eval/failures.py",
            "stable_asr/streaming/failures.py",
            "stable_asr/paper/case_studies.py",
        ],
        "result_keys": ["baselines.rule_endpoint.failure_analysis", "streaming_asr.metrics.failure_analysis"],
        "artifact_paths": ["tables/failure_cases.md", "tables/streaming_failures.md", "CASE_STUDIES.md", "case_studies.json"],
        "commands": ["stable-asr paper-case-studies --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts"],
    },
    {
        "id": "paper_reproducibility",
        "claim": "Stable-ASR ships a paper pipeline that regenerates tables, figures, leaderboards, cards, audits, and editable drafts from one result bundle.",
        "repo_paths": [
            "configs/paper/paper_smoke.json",
            "configs/paper/paper_parity_checklist.json",
            "configs/paper/final_experiments.json",
            "configs/final/paper_final.json",
            "configs/final/asr_command_compare.json",
            "configs/references/asr_collections.json",
            "scripts/reproduce_paper.py",
            "stable_asr/paper/artifacts.py",
            "stable_asr/paper/audit.py",
            "stable_asr/paper/integrity.py",
            "stable_asr/paper/draft.py",
            "stable_asr/paper/latex.py",
            "stable_asr/paper/final_config.py",
            "stable_asr/paper/final_experiments.py",
            "stable_asr/paper/parity.py",
            "stable_asr/references/collections.py",
            "stable_asr/roadmap.py",
        ],
        "result_keys": ["meta.artifact_version"],
        "artifact_paths": [
            "ARTIFACT_INDEX.md",
            "artifact_manifest.json",
            "artifact_hashes.json",
            "ARTIFACT_HASHES.md",
            "leaderboard.jsonl",
            "benchmark_suite.json",
            "data_sources.json",
            "adapter_registry.json",
            "asr_collections.json",
            "ASR_COLLECTIONS.md",
            "asr_collection_coverage.json",
            "ASR_COLLECTION_COVERAGE.md",
            "paper_parity.json",
            "PAPER_PARITY.md",
            "final_experiments.json",
            "FINAL_EXPERIMENTS.md",
            "final_run_config.json",
            "FINAL_RUN_CONFIG.md",
            "final_run_file_audit.json",
            "FINAL_RUN_FILE_AUDIT.md",
            "paper_status.json",
            "PAPER_STATUS.md",
            "roadmap_status.json",
            "ROADMAP_STATUS.md",
        ],
        "commands": [
            "stable-asr reproduce-paper --config configs/paper/paper_smoke.json",
            "stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts",
            "stable-asr paper-release-audit --repo-root . --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts",
        ],
    },
]


@dataclass(frozen=True)
class ClaimAuditCheck:
    claim_id: str
    ok: bool
    missing: list[str]
    claim: str

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "ok": self.ok,
            "missing": self.missing,
            "claim": self.claim,
        }


@dataclass(frozen=True)
class ClaimAuditReport:
    ok: bool
    checks: list[ClaimAuditCheck]
    claims: list[dict[str, Any]]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "claims": self.claims,
        }

    def to_text(self) -> str:
        lines = [f"claim_audit: {'OK' if self.ok else 'FAILED'}"]
        for check in self.checks:
            status = "OK" if check.ok else "MISSING"
            detail = "covered" if check.ok else "; ".join(check.missing)
            lines.append(f"- {status} {check.claim_id}: {detail}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PaperClaimArtifacts:
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {"json": self.json_path, "markdown": self.markdown_path}


def audit_claims(
    *,
    repo_root: str | Path = ".",
    results_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    claims: list[dict[str, Any]] | None = None,
) -> ClaimAuditReport:
    repo_root = Path(repo_root)
    results = _load_results(results_path)
    artifacts = Path(artifacts_dir) if artifacts_dir is not None else None
    claim_items = claims or DEFAULT_CLAIMS
    checks = [
        _audit_claim(claim, repo_root=repo_root, results=results, artifacts_dir=artifacts)
        for claim in claim_items
    ]
    return ClaimAuditReport(ok=all(check.ok for check in checks), checks=checks, claims=claim_items)


def paper_claims(
    results_path: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> PaperClaimArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = audit_claims(repo_root=repo_root, results_path=results_path, artifacts_dir=output_dir)
    json_path = output_dir / "claims.json"
    markdown_path = output_dir / "CLAIMS.md"
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(claims_markdown(report), encoding="utf-8")
    return PaperClaimArtifacts(json_path=str(json_path), markdown_path=str(markdown_path))


def claims_markdown(report: ClaimAuditReport) -> str:
    rows = []
    claim_by_id = {str(claim["id"]): claim for claim in report.claims}
    for check in report.checks:
        claim = claim_by_id.get(check.claim_id, {})
        rows.append(
            {
                "claim": check.claim_id,
                "status": "OK" if check.ok else "MISSING",
                "repo_paths": len(claim.get("repo_paths", [])),
                "result_keys": len(claim.get("result_keys", [])),
                "artifacts": len(claim.get("artifact_paths", [])),
                "missing": "; ".join(check.missing),
            }
        )

    lines = [
        "# Stable-ASR Claim Evidence Matrix",
        "",
        "This matrix maps paper claims to repository files, result keys, artifact files, and reproduction commands.",
        "",
        "## Audit",
        "",
        dict_table(rows),
        "",
        "## Claims",
        "",
    ]
    for claim in report.claims:
        lines.extend(_claim_markdown(claim))
    return "\n".join(lines)


def _claim_markdown(claim: dict[str, Any]) -> list[str]:
    lines = [
        f"### {claim['id']}",
        "",
        str(claim["claim"]),
        "",
        "Repository evidence:",
    ]
    lines.extend(f"- `{path}`" for path in claim.get("repo_paths", []))
    lines.append("")
    lines.append("Result keys:")
    lines.extend(f"- `{key}`" for key in claim.get("result_keys", []))
    lines.append("")
    lines.append("Artifact evidence:")
    lines.extend(f"- `{path}`" for path in claim.get("artifact_paths", []))
    lines.append("")
    lines.append("Reproduction commands:")
    lines.extend(f"- `{command}`" for command in claim.get("commands", []))
    lines.append("")
    return lines


def _audit_claim(
    claim: dict[str, Any],
    *,
    repo_root: Path,
    results: dict[str, Any] | None,
    artifacts_dir: Path | None,
) -> ClaimAuditCheck:
    missing: list[str] = []
    for relative in claim.get("repo_paths", []):
        if not (repo_root / relative).exists():
            missing.append(f"repo:{relative}")
    for key in claim.get("result_keys", []):
        if results is None or not _has_nested_key(results, str(key)):
            missing.append(f"result:{key}")
    for relative in claim.get("artifact_paths", []):
        if artifacts_dir is None or not (artifacts_dir / relative).exists():
            missing.append(f"artifact:{relative}")
    return ClaimAuditCheck(
        claim_id=str(claim["id"]),
        ok=not missing,
        missing=missing,
        claim=str(claim["claim"]),
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
