"""Generate an editable paper draft from Stable-ASR paper artifacts."""

from __future__ import annotations

from pathlib import Path

from stable_asr.paper.figures import PAPER_FIGURES
from stable_asr.paper.tables import PAPER_TABLES, load_paper_results, paper_table


def paper_draft(
    results_path: str | Path,
    output_path: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
) -> str:
    """Write a Markdown preprint draft scaffold from ``paper_results.json``."""

    results_path = Path(results_path)
    output_path = Path(output_path)
    results = load_paper_results(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _draft_markdown(results_path, results, Path(artifacts_dir) if artifacts_dir else None),
        encoding="utf-8",
    )
    return str(output_path)


def _draft_markdown(results_path: Path, results: dict[str, object], artifacts_dir: Path | None) -> str:
    meta = results.get("meta", {})
    episodes = meta.get("episodes", "unknown") if isinstance(meta, dict) else "unknown"
    seed = meta.get("seed", "unknown") if isinstance(meta, dict) else "unknown"
    baselines = results.get("baselines", {})
    scenarios = results.get("scenarios", {})
    scenario_count = len(scenarios.get("by_scenario", {})) if isinstance(scenarios, dict) else 0
    baseline_count = len(baselines) if isinstance(baselines, dict) else 0

    lines = [
        "# Stable-ASR: A Platform for Reproducible Real-Time ASR and Full-Duplex Turn-Taking Research",
        "",
        "> Draft generated from Stable-ASR experiment artifacts. Edit before submission.",
        "",
        "## Abstract",
        "",
        (
            "Real-time ASR systems are usually evaluated with recognition accuracy, "
            "but voice agents also depend on endpointing, turn-taking, interruption "
            "handling, streaming stability, and deployment latency. Stable-ASR is an "
            "open-source platform for standardized real-time ASR and full-duplex "
            "turn-taking research. It unifies data manifests, baseline models, "
            "external prediction adapters, scenario evaluation, policy search, "
            "latency benchmarking, and paper artifact generation."
        ),
        "",
        "## 1. Introduction",
        "",
        (
            "ASR toolkits, VAD/endpointing systems, turn-taking models, and "
            "voice-agent orchestration frameworks often expose incompatible data "
            "formats and evaluation protocols. This fragmentation makes it hard to "
            "compare whether a system is accurate, responsive, robust to overlap, "
            "and deployable under realistic latency constraints."
        ),
        "",
        "This draft reports a smoke experiment that exercises the artifact shape of the platform:",
        "",
        f"- episodes: `{episodes}`",
        f"- seed: `{seed}`",
        f"- baselines: `{baseline_count}`",
        f"- scenarios: `{scenario_count}`",
        f"- results source: `{results_path}`",
        "",
        "## 2. Contributions",
        "",
        "- A canonical turn/action manifest and conversion path for real-time ASR and turn-taking data.",
        "- A shared evaluator for rule, VAD, text, NanoTurn, and external prediction-manifest baselines.",
        "- A VoiceWorld scenario suite for incomplete pauses, backchannels, waits, interruptions, side conversations, ambient speech, noisy far-field speech, and code-switching.",
        "- Cost-sensitive policy search and latency benchmarking for interaction-level decisions.",
        "- Reproducible paper tables, figures, bundles, claim audits, paper-parity audits, final-experiment runbooks, final-run configs/action plans, and this draft generator.",
        "",
        "## 3. Platform Overview",
        "",
        "Stable-ASR follows the loop: prepare or convert data, run baselines or adapters, evaluate static and scenario behavior, optimize policies, then generate paper artifacts.",
        "",
        "## 4. Data Layer",
        "",
        "The current artifact includes turn manifest conversion, ASR corpus manifest preparation, and data-format benchmark rows.",
        "",
        _table(results_path, "data"),
        "",
        _table(results_path, "asr_manifest_recipe"),
        "",
        "## 5. Baselines",
        "",
        "All baselines are evaluated through the same turn prediction and policy interface.",
        "",
        _table(results_path, "baselines"),
        "",
        "Failure mining turns aggregate scores into interaction-level case studies for debugging and paper analysis.",
        "",
        _table(results_path, "failure_cases"),
        "",
        "## 6. Turn Latency And Deployment",
        "",
        "Turn-taking quality is reported together with latency, throughput, RTF, and artifact-size measurements.",
        "",
        _table(results_path, "turn_benchmark"),
        "",
        "## 7. VoiceWorld Scenario Evaluation",
        "",
        "Scenario metrics expose interaction failures that are not visible in aggregate accuracy alone.",
        "",
        _table(results_path, "scenarios"),
        "",
        "## 8. Policy Search",
        "",
        "The policy search table summarizes the best cost-sensitive threshold configuration found for the smoke suite.",
        "",
        _table(results_path, "policy"),
        "",
        "## 9. Streaming ASR Evaluation",
        "",
        "Streaming metrics capture partial-result behavior, endpoint delay, and timestamp drift in addition to final recognition quality.",
        "",
        _table(results_path, "streaming"),
        "",
        "Streaming failure mining highlights recognition, endpointing, partial stability, timestamp, and real-time latency failures at the record level.",
        "",
        _table(results_path, "streaming_failures"),
        "",
        "The schedule sweep shows how chunk size and lookahead alter real-time latency metrics while transcript quality remains fixed.",
        "",
        _table(results_path, "streaming_sweep"),
        "",
        "External ASR transcript conversion tables show that Whisper, FunASR, Qwen3-ASR, and FireRedASR2S-style outputs can enter the same evaluator.",
        "",
        _table(results_path, "asr_transcript_conversions"),
        "",
    ]

    if artifacts_dir is not None:
        lines.extend(_figure_references(artifacts_dir))

    lines.extend(
        [
            "## 10. Limitations",
            "",
            "- Current smoke artifacts use small synthetic fixtures and are not final benchmark claims.",
            "- Lance-backed data benchmarking is smoke-scale; final paper claims require larger local and object-storage runs.",
            "- External SmartTurn/EasyTurn/VAP integration is currently represented through prediction-manifest adapters and converters.",
            "- Human user studies and full WebRTC voice-agent deployments remain future work.",
            "",
            "## 11. Reproducibility",
            "",
            "```bash",
            "stable-asr reproduce-paper --config configs/paper/paper_smoke.json --skip-train",
            "stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts",
            "stable-asr paper-audit --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts",
            "stable-asr paper-parity-audit --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts",
            "stable-asr final-experiments --registry configs/paper/final_experiments.json --output runs/paper/smoke/artifacts/FINAL_EXPERIMENTS.md",
            "stable-asr final-config --config configs/final/paper_final.json --output runs/paper/smoke/artifacts/FINAL_RUN_CONFIG.md",
            "# Expected to report NOT_READY until final corpora, splits, and external predictions exist.",
            "stable-asr final-config --config configs/final/paper_final.json --check-files",
            "stable-asr paper-draft --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/PAPER_DRAFT.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _table(results_path: Path, name: str) -> str:
    if name not in PAPER_TABLES:
        raise ValueError(f"unknown paper table: {name}")
    return paper_table(results_path, name)


def _figure_references(artifacts_dir: Path) -> list[str]:
    lines = ["## Figures", ""]
    for figure in PAPER_FIGURES:
        path = artifacts_dir / "figures" / f"{figure}.svg"
        if path.exists():
            lines.append(f"![{figure}]({path})")
            lines.append("")
    index_path = artifacts_dir / "ARTIFACT_INDEX.md"
    if index_path.exists():
        lines.append(f"Artifact index: `{index_path}`")
        lines.append("")
    return lines
