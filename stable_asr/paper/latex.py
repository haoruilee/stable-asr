"""Generate a LaTeX paper draft from Stable-ASR paper artifacts."""

from __future__ import annotations

from pathlib import Path

from stable_asr.paper.figures import PAPER_FIGURES
from stable_asr.paper.tables import PAPER_TABLES, load_paper_results, paper_table


def paper_latex(
    results_path: str | Path,
    output_path: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
) -> str:
    """Write an arXiv-style LaTeX draft scaffold from ``paper_results.json``."""

    results_path = Path(results_path)
    output_path = Path(output_path)
    results = load_paper_results(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _latex_document(results_path, results, Path(artifacts_dir) if artifacts_dir else None),
        encoding="utf-8",
    )
    return str(output_path)


def _latex_document(results_path: Path, results: dict[str, object], artifacts_dir: Path | None) -> str:
    meta = results.get("meta", {})
    episodes = meta.get("episodes", "unknown") if isinstance(meta, dict) else "unknown"
    seed = meta.get("seed", "unknown") if isinstance(meta, dict) else "unknown"
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{hyperref}",
        r"\title{Stable-ASR: A Platform for Reproducible Real-Time ASR and Full-Duplex Turn-Taking Research}",
        r"\author{Haorui Li}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        (
            "Real-time ASR systems are usually evaluated with recognition accuracy, "
            "but voice agents also depend on endpointing, turn-taking, interruption "
            "handling, streaming stability, and deployment latency. Stable-ASR is an "
            "open-source platform for standardized real-time ASR and full-duplex "
            "turn-taking research. It unifies data manifests, baseline models, "
            "external prediction adapters, scenario evaluation, policy search, "
            "latency benchmarking, and paper artifact generation."
        ),
        r"\end{abstract}",
        r"\section{Introduction}",
        (
            "ASR toolkits, VAD and endpointing systems, turn-taking models, and "
            "voice-agent orchestration frameworks often expose incompatible data "
            "formats and evaluation protocols. This fragmentation makes it hard to "
            "compare whether a system is accurate, responsive, robust to overlap, "
            "and deployable under realistic latency constraints."
        ),
        "",
        f"This draft was generated from \\texttt{{{_tex_escape(str(results_path))}}} with episodes={_tex_escape(str(episodes))} and seed={_tex_escape(str(seed))}.",
        r"\section{Contributions}",
        r"\begin{itemize}",
        r"\item A canonical turn/action manifest and conversion path for real-time ASR and turn-taking data.",
        r"\item A shared evaluator for rule, VAD, text, NanoTurn, and external prediction-manifest baselines.",
        r"\item A VoiceWorld scenario suite for incomplete pauses, backchannels, waits, interruptions, side conversations, ambient speech, noisy far-field speech, and code-switching.",
        r"\item Cost-sensitive policy search and latency benchmarking for interaction-level decisions.",
        r"\item Reproducible paper tables, figures, bundles, claim audits, paper-parity audits, final-experiment runbooks, final-run configs, and draft generation.",
        r"\end{itemize}",
        r"\section{Platform Overview}",
        "Stable-ASR follows the loop: prepare or convert data, run baselines or adapters, evaluate static and scenario behavior, optimize policies, then generate paper artifacts.",
        r"\section{Data Layer}",
        "The current artifact includes turn manifest conversion, ASR corpus manifest preparation, and data-format benchmark rows.",
        _latex_table(results_path, "data", "Data format benchmark."),
        _latex_table(results_path, "asr_manifest_recipe", "ASR corpus manifest recipe summary."),
        r"\section{Baselines}",
        "All baselines are evaluated through the same turn prediction and policy interface.",
        _latex_table(results_path, "baselines", "Baseline quality metrics."),
        "Failure mining turns aggregate scores into interaction-level case studies for debugging and paper analysis.",
        _latex_table(results_path, "failure_cases", "Turn-taking failure taxonomy by baseline."),
        r"\section{Turn Latency and Deployment}",
        "Turn-taking quality is reported together with latency, throughput, RTF, and artifact-size measurements.",
        _latex_table(results_path, "turn_benchmark", "Turn predictor latency and artifact size."),
        r"\section{VoiceWorld Scenario Evaluation}",
        "Scenario metrics expose interaction failures that are not visible in aggregate accuracy alone.",
        _latex_table(results_path, "scenarios", "Scenario-level robustness metrics."),
        r"\section{Policy Search}",
        "The policy search table summarizes the best cost-sensitive threshold configuration found for the smoke suite.",
        _latex_table(results_path, "policy", "Cost-sensitive policy search result."),
        r"\section{Streaming ASR Evaluation}",
        "Streaming metrics capture partial-result behavior, endpoint delay, and timestamp drift in addition to final recognition quality.",
        _latex_table(results_path, "streaming", "Streaming ASR metrics."),
        "Streaming failure mining highlights recognition, endpointing, partial stability, timestamp, and real-time latency failures at the record level.",
        _latex_table(results_path, "streaming_failures", "Streaming ASR failure taxonomy."),
        "The schedule sweep shows how chunk size and lookahead alter real-time latency metrics while transcript quality remains fixed.",
        _latex_table(results_path, "streaming_sweep", "Streaming chunk and lookahead sensitivity."),
        "External ASR transcript conversion tables show that Whisper, FunASR, Qwen3-ASR, and FireRedASR2S-style outputs can enter the same evaluator.",
        _latex_table(results_path, "asr_transcript_conversions", "External ASR transcript conversion metrics."),
    ]
    if artifacts_dir is not None:
        lines.extend(_latex_figures(artifacts_dir))
    lines.extend(
        [
            r"\section{Limitations}",
            r"\begin{itemize}",
            r"\item Current smoke artifacts use small synthetic fixtures and are not final benchmark claims.",
            r"\item Lance-backed data benchmarking is smoke-scale; final paper claims require larger local and object-storage runs.",
            r"\item External SmartTurn, EasyTurn, and VAP integration is currently represented through prediction-manifest adapters and converters.",
            r"\item Human user studies and full WebRTC voice-agent deployments remain future work.",
            r"\end{itemize}",
            r"\section{Reproducibility}",
            r"\begin{verbatim}",
            "stable-asr reproduce-paper --config configs/paper/paper_smoke.json --skip-train",
            "stable-asr paper-bundle --results runs/paper/smoke/paper_results.json --output-dir runs/paper/smoke/artifacts",
            "stable-asr paper-audit --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts",
            "stable-asr paper-parity-audit --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts",
            "stable-asr final-experiments --registry configs/paper/final_experiments.json --output runs/paper/smoke/artifacts/FINAL_EXPERIMENTS.md",
            "stable-asr final-config --config configs/final/paper_final.json --output runs/paper/smoke/artifacts/FINAL_RUN_CONFIG.md",
            "# Expected to report NOT_READY until final corpora, splits, and external predictions exist.",
            "stable-asr final-config --config configs/final/paper_final.json --check-files",
            "stable-asr paper-latex --results runs/paper/smoke/paper_results.json --artifacts-dir runs/paper/smoke/artifacts --output runs/paper/smoke/paper.tex",
            r"\end{verbatim}",
            r"\end{document}",
            "",
        ]
    )
    return "\n\n".join(lines)


def _latex_table(results_path: Path, table: str, caption: str) -> str:
    if table not in PAPER_TABLES:
        raise ValueError(f"unknown paper table: {table}")
    rows = _parse_markdown_table(paper_table(results_path, table))
    if not rows:
        return "% table unavailable"
    header = rows[0]
    body = rows[1:]
    align = "|" + "|".join("l" for _ in header) + "|"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(_tex_escape(cell) for cell in header) + r" \\",
        r"\hline",
    ]
    for row in body:
        lines.append(" & ".join(_tex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _parse_markdown_table(table: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for index, line in enumerate(table.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if index == 1 and all(set(cell) <= {"-"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def _latex_figures(artifacts_dir: Path) -> list[str]:
    lines = [r"\section{Generated Figures}"]
    for figure in PAPER_FIGURES:
        path = artifacts_dir / "figures" / f"{figure}.svg"
        if not path.exists():
            continue
        lines.extend(
            [
                r"\begin{figure}[t]",
                r"\centering",
                r"\fbox{\parbox{0.88\linewidth}{\centering Generated figure artifact: "
                + _tex_escape(str(path))
                + r"}}",
                f"\\caption{{{_tex_escape(figure)} figure artifact. Convert the SVG to PDF or PNG before final submission.}}",
                r"\end{figure}",
            ]
        )
    return lines


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)
