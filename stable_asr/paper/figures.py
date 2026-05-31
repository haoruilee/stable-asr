"""Paper figure extraction from structured result artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from pathlib import Path

from stable_asr.paper.tables import load_paper_results


STATIC_PAPER_FIGURES = (
    "architecture",
    "api_flow",
    "data_registry",
    "voiceworld_timeline",
    "policy_state_machine",
    "robustness_heatmap",
    "latency_quality_pareto",
)
CHART_PAPER_FIGURES = ("baselines", "latency", "data", "streaming", "scenarios", "policy")
PAPER_FIGURES = STATIC_PAPER_FIGURES + CHART_PAPER_FIGURES


@dataclass(frozen=True)
class ChartSpec:
    title: str
    y_label: str
    rows: list[tuple[str, float]]
    y_max: float | None = None


def paper_figure(results_path: str | Path, figure: str, output_path: str | Path) -> str:
    """Write a paper-facing SVG figure from ``paper_results.json``."""

    results = load_paper_results(results_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if figure in STATIC_PAPER_FIGURES:
        svg = _static_figure_svg(results, figure)
    else:
        svg = _bar_chart_svg(_chart_spec(results, figure))
    output_path.write_text(svg, encoding="utf-8")
    return str(output_path)


def _chart_spec(results: dict[str, object], figure: str) -> ChartSpec:
    if figure == "baselines":
        return _baseline_spec(results)
    if figure == "latency":
        return _latency_spec(results)
    if figure == "data":
        return _data_spec(results)
    if figure == "streaming":
        return _streaming_spec(results)
    if figure == "scenarios":
        return _scenario_spec(results)
    if figure == "policy":
        return _policy_spec(results)
    raise ValueError(f"unknown paper figure: {figure}")


def _static_figure_svg(results: dict[str, object], figure: str) -> str:
    if figure == "architecture":
        return _architecture_svg(results)
    if figure == "api_flow":
        return _api_flow_svg(results)
    if figure == "data_registry":
        return _data_registry_svg(results)
    if figure == "voiceworld_timeline":
        return _voiceworld_timeline_svg(results)
    if figure == "policy_state_machine":
        return _policy_state_machine_svg(results)
    if figure == "robustness_heatmap":
        return _robustness_heatmap_svg(results)
    if figure == "latency_quality_pareto":
        return _latency_quality_pareto_svg(results)
    raise ValueError(f"unknown paper figure: {figure}")


def _architecture_svg(results: dict[str, object]) -> str:
    meta = results.get("meta", {})
    artifact_version = meta.get("artifact_version", "unknown") if isinstance(meta, dict) else "unknown"
    width = 1120
    height = 700
    parts = _svg_canvas(width, height, "Stable-ASR Platform Architecture")
    parts.extend(
        [
            _svg_text(width / 2, 42, "Stable-ASR Platform Architecture", size=28, weight=700, anchor="middle"),
            _svg_text(
                width / 2,
                70,
                f"Unified data, baseline, policy, scenario, evaluation, and paper artifact pipeline (artifact {artifact_version})",
                size=14,
                anchor="middle",
                color="#4b5563",
            ),
        ]
    )
    layers = [
        (
            "Data Layer",
            ["Turn/ASR Manifest", "JSONL / Parquet / Lance", "EasyTurn / Full-Duplex-Bench converters"],
            92,
            "#dbeafe",
            "#1d4ed8",
        ),
        (
            "Model and Adapter Layer",
            ["NanoTurn", "Rule / VAD / Text baselines", "Prediction and ASR adapters"],
            230,
            "#dcfce7",
            "#047857",
        ),
        (
            "Policy and Scenario Layer",
            ["TurnPolicy", "Threshold search", "VoiceWorld scenarios"],
            368,
            "#fef3c7",
            "#b45309",
        ),
        (
            "Evaluation and Paper Layer",
            ["Turn / streaming / latency metrics", "Tables and figures", "Bundle / audit / Markdown / LaTeX"],
            506,
            "#f3e8ff",
            "#6d28d9",
        ),
    ]
    for title, items, y, fill, stroke in layers:
        parts.append(_svg_round_rect(90, y, 940, 102, fill, stroke))
        parts.append(_svg_text(118, y + 28, title, size=18, weight=700, color="#111827"))
        for index, item in enumerate(items):
            x = 300 + index * 235
            parts.append(_svg_round_rect(x, y + 22, 198, 48, "#ffffff", stroke, radius=8))
            parts.append(_svg_wrapped_label(x + 99, y + 42, item, width=176, size=13))
        if y < 506:
            parts.append(_svg_arrow(width / 2, y + 105, width / 2, y + 132, "#374151"))
    parts.append(_svg_footer(width, height))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _api_flow_svg(results: dict[str, object]) -> str:
    meta = results.get("meta", {})
    seed = meta.get("seed", "unknown") if isinstance(meta, dict) else "unknown"
    width = 1120
    height = 560
    parts = _svg_canvas(width, height, "Stable-ASR Three-Stage API Flow")
    parts.extend(
        [
            _svg_text(width / 2, 44, "Three-Stage API Flow", size=28, weight=700, anchor="middle"),
            _svg_text(width / 2, 72, f"Smoke artifact seed={seed}; every stage emits reusable files.", size=14, anchor="middle", color="#4b5563"),
        ]
    )
    stages = [
        (
            "1. Prepare / Convert",
            ["validate-manifest", "convert-external", "benchmark-data"],
            "Canonical manifests and data benchmark rows",
            86,
            "#dbeafe",
            "#1d4ed8",
        ),
        (
            "2. Train / Adapt",
            ["train-turn", "convert-predictions", "benchmark-turn"],
            "NanoTurn checkpoints or external prediction adapters",
            394,
            "#dcfce7",
            "#047857",
        ),
        (
            "3. Evaluate / Report",
            ["eval-turn", "eval-scenario", "paper-bundle"],
            "Metrics, tables, figures, audits, drafts",
            702,
            "#f3e8ff",
            "#6d28d9",
        ),
    ]
    for title, commands, detail, x, fill, stroke in stages:
        parts.append(_svg_round_rect(x, 128, 250, 290, fill, stroke))
        parts.append(_svg_text(x + 125, 166, title, size=18, weight=700, anchor="middle"))
        parts.append(_svg_wrapped_label(x + 125, 210, detail, width=210, size=13))
        for index, command in enumerate(commands):
            y = 260 + index * 46
            parts.append(_svg_round_rect(x + 36, y, 178, 30, "#ffffff", stroke, radius=7))
            parts.append(_svg_text(x + 125, y + 20, command, size=12, anchor="middle", family="monospace"))
    parts.append(_svg_arrow(336, 274, 386, 274, "#374151"))
    parts.append(_svg_arrow(644, 274, 694, 274, "#374151"))
    parts.append(_svg_footer(width, height))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _data_registry_svg(results: dict[str, object]) -> str:
    data = results.get("data", {})
    benchmark = data.get("benchmark", {}) if isinstance(data, dict) else {}
    rows = benchmark.get("rows", []) if isinstance(benchmark, dict) else []
    formats = [str(row.get("format", "unknown")) for row in rows if isinstance(row, dict)] or ["jsonl"]
    external = data.get("external_conversions", data.get("external_conversion", {})) if isinstance(data, dict) else {}
    if isinstance(external, list):
        external_items = [str(item.get("schema", "external")) for item in external if isinstance(item, dict)]
    elif isinstance(external, dict):
        external_items = [str(external.get("schema", "external"))]
    else:
        external_items = ["external"]

    width = 1120
    height = 640
    parts = _svg_canvas(width, height, "Stable-ASR Data Format Registry")
    parts.extend(
        [
            _svg_text(width / 2, 44, "Data Format Registry", size=28, weight=700, anchor="middle"),
            _svg_text(
                width / 2,
                72,
                "Every converter produces the same manifest abstraction before training, benchmarking, and paper artifact generation.",
                size=14,
                anchor="middle",
                color="#4b5563",
            ),
        ]
    )
    backend_items = list(formats)
    if "lance" not in backend_items:
        backend_items.append("lance optional")
    columns = [
        ("Sources", ["Turn manifest", *external_items[:3], "Prediction JSONL"], 92, "#dbeafe", "#1d4ed8"),
        ("Registry", ["stable_asr.data.registry", "load_records", "write_records", "benchmark-data"], 360, "#fef3c7", "#b45309"),
        ("Backends", backend_items, 628, "#dcfce7", "#047857"),
        ("Consumers", ["train-turn", "eval-turn", "VoiceWorld", "paper bundle"], 896, "#f3e8ff", "#6d28d9"),
    ]
    for title, items, x, fill, stroke in columns:
        parts.append(_svg_round_rect(x, 126, 202, 360, fill, stroke))
        parts.append(_svg_text(x + 101, 164, title, size=18, weight=700, anchor="middle"))
        for index, item in enumerate(items[:5]):
            y = 210 + index * 48
            parts.append(_svg_round_rect(x + 22, y, 158, 32, "#ffffff", stroke, radius=7))
            parts.append(_svg_wrapped_label(x + 101, y + 21, item, width=140, size=12))
    for x in (294, 562, 830):
        parts.append(_svg_arrow(x, 304, x + 58, 304, "#374151"))
    parts.append(_svg_footer(width, height))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _voiceworld_timeline_svg(results: dict[str, object]) -> str:
    scenarios = results.get("scenarios", {})
    count = 0
    if isinstance(scenarios, dict) and isinstance(scenarios.get("by_scenario"), dict):
        count = len(scenarios["by_scenario"])
    width = 1120
    height = 560
    parts = _svg_canvas(width, height, "VoiceWorld Scenario Timeline")
    parts.extend(
        [
            _svg_text(width / 2, 44, "VoiceWorld Scenario Timeline", size=28, weight=700, anchor="middle"),
            _svg_text(width / 2, 72, f"Seedable interaction traces with {count or 'multiple'} scenario family breakdowns.", size=14, anchor="middle", color="#4b5563"),
        ]
    )
    lanes = [("User audio", 150, "#2563eb"), ("Assistant audio", 250, "#059669"), ("Turn policy", 350, "#7c3aed")]
    for label, y, color in lanes:
        parts.append(_svg_text(96, y + 8, label, size=15, weight=700, anchor="end"))
        parts.append(f'<line x1="130" y1="{y}" x2="1010" y2="{y}" stroke="#d1d5db" stroke-width="2"/>')
        parts.append(f'<circle cx="130" cy="{y}" r="4" fill="{color}"/>')
        parts.append(f'<circle cx="1010" cy="{y}" r="4" fill="{color}"/>')
    user_blocks = [
        ("incomplete prefix", 160, 135, "#bfdbfe"),
        ("pause", 305, 92, "#e5e7eb"),
        ("completion", 410, 150, "#bfdbfe"),
        ("interrupt / backchannel", 706, 188, "#fecaca"),
    ]
    for label, x, w, fill in user_blocks:
        parts.append(_svg_round_rect(x, 125, w, 44, fill, "#2563eb", radius=8))
        parts.append(_svg_wrapped_label(x + w / 2, 151, label, width=w - 18, size=12))
    assistant_blocks = [
        ("assistant TTS", 580, 230, "#bbf7d0"),
        ("stop or continue", 828, 146, "#dcfce7"),
    ]
    for label, x, w, fill in assistant_blocks:
        parts.append(_svg_round_rect(x, 225, w, 44, fill, "#059669", radius=8))
        parts.append(_svg_wrapped_label(x + w / 2, 251, label, width=w - 18, size=12))
    decisions = [
        ("keep_listening", 298),
        ("take_turn", 548),
        ("stop_tts_and_listen", 780),
        ("light_ack / hold", 934),
    ]
    for label, x in decisions:
        parts.append(_svg_round_rect(x - 70, 325, 140, 44, "#f3e8ff", "#7c3aed", radius=8))
        parts.append(_svg_wrapped_label(x, 351, label, width=124, size=12))
        parts.append(_svg_arrow(x, 320, x, 282, "#7c3aed"))
    parts.append(_svg_footer(width, height))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _robustness_heatmap_svg(results: dict[str, object]) -> str:
    scenarios = results.get("scenarios", {})
    by_scenario = scenarios.get("by_scenario", {}) if isinstance(scenarios, dict) else {}
    if not isinstance(by_scenario, dict) or not by_scenario:
        raise ValueError("missing scenario breakdown for robustness heatmap")

    metrics = [
        ("accuracy", "higher"),
        ("macro_f1", "higher"),
        ("false_complete", "lower"),
        ("missed_interrupt", "lower"),
    ]
    rows: list[tuple[str, list[float]]] = []
    for scenario, payload in by_scenario.items():
        if not isinstance(payload, dict):
            continue
        classification = payload.get("classification", {})
        interaction = payload.get("interaction", {})
        if not isinstance(classification, dict) or not isinstance(interaction, dict):
            continue
        rows.append(
            (
                str(scenario),
                [
                    float(classification.get("accuracy", 0.0)),
                    float(classification.get("macro_f1", 0.0)),
                    float(interaction.get("false_complete_rate", 0.0)),
                    float(interaction.get("missed_interrupt_rate", 0.0)),
                ],
            )
        )
    if not rows:
        raise ValueError("robustness heatmap has no scenario rows")

    width = 1120
    height = max(560, 180 + len(rows) * 64)
    left = 260
    top = 130
    cell_w = 178
    cell_h = 48
    parts = _svg_canvas(width, height, "VoiceWorld Robustness Heatmap")
    parts.extend(
        [
            _svg_text(width / 2, 44, "VoiceWorld Robustness Heatmap", size=28, weight=700, anchor="middle"),
            _svg_text(
                width / 2,
                72,
                "Scenario-level quality and failure rates from the same reproducible paper result artifact.",
                size=14,
                anchor="middle",
                color="#4b5563",
            ),
        ]
    )
    for col, (metric, _) in enumerate(metrics):
        x = left + col * cell_w
        parts.append(_svg_wrapped_label(x + cell_w / 2, top - 24, metric, width=cell_w - 18, size=12, color="#374151"))
    for row_index, (scenario, values) in enumerate(rows):
        y = top + row_index * cell_h
        parts.append(_svg_wrapped_label(left - 28, y + 29, scenario, width=196, size=12, color="#111827"))
        for col, value in enumerate(values):
            _, direction = metrics[col]
            quality = value if direction == "higher" else 1.0 - value
            color = _heat_color(quality)
            x = left + col * cell_w
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 8:.1f}" height="{cell_h - 8:.1f}" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            parts.append(_svg_text(x + (cell_w - 8) / 2, y + 26, _format_value(value), size=13, weight=700, anchor="middle", color="#111827"))
    parts.append(_svg_text(left, height - 56, "Color encodes interaction quality: green is better, red is worse.", size=12, color="#4b5563"))
    parts.append(_svg_footer(width, height))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _latency_quality_pareto_svg(results: dict[str, object]) -> str:
    baselines = results.get("baselines", {})
    benchmarks = results.get("turn_benchmarks", {})
    if not isinstance(baselines, dict) or not isinstance(benchmarks, dict):
        raise ValueError("missing baselines or turn_benchmarks for Pareto figure")
    points: list[tuple[str, float, float]] = []
    for name, payload in baselines.items():
        benchmark = benchmarks.get(name)
        if not isinstance(payload, dict) or not isinstance(benchmark, dict):
            continue
        classification = payload.get("classification", {})
        if not isinstance(classification, dict):
            continue
        points.append((str(name), float(benchmark.get("avg_latency_ms", 0.0)), float(classification.get("macro_f1", 0.0))))
    if not points:
        raise ValueError("latency-quality Pareto figure has no points")

    width = 960
    height = 620
    margin_left = 96
    margin_right = 56
    margin_top = 92
    margin_bottom = 96
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_latency = max(latency for _, latency, _ in points)
    max_quality = max(quality for _, _, quality in points)
    x_max = max_latency * 1.25 if max_latency > 0 else 1.0
    y_max = max(1.0, max_quality * 1.15)
    axis_y = margin_top + plot_h

    parts = _svg_canvas(width, height, "Latency-Quality Pareto")
    parts.extend(
        [
            _svg_text(width / 2, 42, "Latency-Quality Pareto", size=28, weight=700, anchor="middle"),
            _svg_text(width / 2, 70, "Turn predictor macro F1 versus average decision latency.", size=14, anchor="middle", color="#4b5563"),
            _svg_text(width / 2, height - 36, "avg latency ms", size=14, anchor="middle", color="#374151"),
            _svg_text(26, margin_top + plot_h / 2, "macro F1", size=14, anchor="middle", color="#374151"),
        ]
    )
    for tick in range(6):
        x_value = x_max * tick / 5
        x = margin_left + (x_value / x_max) * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{axis_y}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(_svg_text(x, axis_y + 22, _format_value(x_value), size=11, anchor="middle", color="#6b7280"))
        y_value = y_max * tick / 5
        y = axis_y - (y_value / y_max) * plot_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(_svg_text(margin_left - 12, y + 4, _format_value(y_value), size=11, anchor="end", color="#6b7280"))
    parts.append(f'<line x1="{margin_left}" y1="{axis_y}" x2="{width - margin_right}" y2="{axis_y}" stroke="#111827" stroke-width="1.5"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_y}" stroke="#111827" stroke-width="1.5"/>')
    palette = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#4b5563"]
    for index, (name, latency, quality) in enumerate(points):
        x = margin_left + (latency / x_max) * plot_w
        y = axis_y - (quality / y_max) * plot_h
        color = palette[index % len(palette)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        parts.append(_svg_text(x + 12, y - 10, _short_label(name, 20), size=12, color="#111827"))
    parts.append(_svg_footer(width, height))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _policy_state_machine_svg(results: dict[str, object]) -> str:
    policy = results.get("policy_search", {})
    trials = policy.get("trials", []) if isinstance(policy, dict) else []
    width = 1120
    height = 660
    parts = _svg_canvas(width, height, "TurnPolicy State Machine")
    parts.extend(
        [
            _svg_text(width / 2, 44, "TurnPolicy State Machine", size=28, weight=700, anchor="middle"),
            _svg_text(width / 2, 72, f"Cost-sensitive policy search trials: {len(trials) if isinstance(trials, list) else 0}", size=14, anchor="middle", color="#4b5563"),
        ]
    )
    nodes = {
        "keep_listening": (220, 300, "#dbeafe", "#1d4ed8"),
        "take_turn": (560, 180, "#dcfce7", "#047857"),
        "continue_speaking": (560, 420, "#fef3c7", "#b45309"),
        "stop_tts_and_listen": (900, 300, "#fecaca", "#dc2626"),
        "hold_or_light_ack": (560, 300, "#f3e8ff", "#6d28d9"),
    }
    for label, (x, y, fill, stroke) in nodes.items():
        parts.append(_svg_round_rect(x - 88, y - 38, 176, 76, fill, stroke, radius=10))
        parts.append(_svg_wrapped_label(x, y - 2, label, width=148, size=13))
    arrows = [
        ("complete >= threshold", 308, 276, 472, 205, "#047857"),
        ("incomplete / silence", 308, 310, 472, 310, "#6d28d9"),
        ("backchannel", 560, 338, 560, 382, "#b45309"),
        ("interrupt confidence", 648, 300, 812, 300, "#dc2626"),
        ("assistant resumes", 812, 336, 648, 406, "#374151"),
        ("wait", 472, 344, 308, 344, "#6d28d9"),
    ]
    for label, x1, y1, x2, y2, color in arrows:
        parts.append(_svg_arrow(x1, y1, x2, y2, color))
        parts.append(_svg_text((x1 + x2) / 2, (y1 + y2) / 2 - 8, label, size=11, anchor="middle", color=color))
    parts.append(_svg_footer(width, height))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _baseline_spec(results: dict[str, object]) -> ChartSpec:
    baselines = results["baselines"]
    if not isinstance(baselines, dict):
        raise ValueError("missing baselines object")
    rows: list[tuple[str, float]] = []
    for name, payload in baselines.items():
        if isinstance(payload, dict):
            rows.append((name, float(payload["classification"]["macro_f1"])))
    return ChartSpec("Baseline Macro F1", "macro F1", rows, y_max=1.0)


def _latency_spec(results: dict[str, object]) -> ChartSpec:
    benchmarks = results["turn_benchmarks"]
    if not isinstance(benchmarks, dict):
        raise ValueError("missing turn_benchmarks object")
    rows = []
    for name, payload in benchmarks.items():
        if isinstance(payload, dict):
            rows.append((name, float(payload["avg_latency_ms"])))
    return ChartSpec("Turn Predictor Latency", "avg latency ms", rows)


def _data_spec(results: dict[str, object]) -> ChartSpec:
    data = results["data"]
    if not isinstance(data, dict):
        raise ValueError("missing data object")
    benchmark = data["benchmark"]
    if not isinstance(benchmark, dict) or benchmark.get("status") != "completed":
        reason = benchmark.get("reason", "data benchmark unavailable") if isinstance(benchmark, dict) else "missing"
        raise ValueError(str(reason))
    rows = [(str(row["format"]), float(row["size_bytes"]) / 1024.0) for row in benchmark["rows"]]
    return ChartSpec("Data Format Size", "KB", rows)


def _streaming_spec(results: dict[str, object]) -> ChartSpec:
    streaming = results["streaming_asr"]
    if not isinstance(streaming, dict):
        raise ValueError("missing streaming_asr object")
    comparison = streaming.get("adapter_comparison")
    if isinstance(comparison, dict) and isinstance(comparison.get("rows"), list):
        rows = []
        for row in comparison["rows"]:
            if isinstance(row, dict):
                rows.append((str(row["adapter"]), float(row["wer"])))
        if rows:
            return ChartSpec("Streaming ASR Adapter WER", "WER", rows, y_max=1.0)
    metrics = streaming["metrics"]
    rows = [
        ("WER", float(metrics["wer"])),
        ("CER", float(metrics["cer"])),
        ("RTF", float(metrics["rtf"])),
        ("Endpoint", float(metrics["endpoint_delay"])),
        ("Revision", float(metrics["partial_revision_rate"])),
        ("Stable prefix", float(metrics["stable_prefix_ratio"])),
        ("TS drift", float(metrics["timestamp_drift"])),
    ]
    return ChartSpec("Streaming ASR Metrics", "rate", rows, y_max=1.0)


def _scenario_spec(results: dict[str, object]) -> ChartSpec:
    scenarios = results["scenarios"]
    if not isinstance(scenarios, dict):
        raise ValueError("missing scenarios object")
    rows = []
    for scenario, payload in scenarios["by_scenario"].items():
        rows.append((str(scenario), float(payload["classification"]["accuracy"])))
    return ChartSpec("VoiceWorld Scenario Accuracy", "accuracy", rows, y_max=1.0)


def _policy_spec(results: dict[str, object]) -> ChartSpec:
    policy_search = results["policy_search"]
    if not isinstance(policy_search, dict):
        raise ValueError("missing policy_search object")
    rows = []
    for index, trial in enumerate(policy_search["trials"][:12], start=1):
        rows.append((f"t{index}", float(trial["score"])))
    return ChartSpec("Policy Search Objective", "score, lower is better", rows)


def _svg_canvas(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _svg_round_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str,
    *,
    radius: int = 12,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )


def _svg_text(
    x: float,
    y: float,
    text: object,
    *,
    size: int = 14,
    weight: int = 400,
    anchor: str = "start",
    color: str = "#111827",
    family: str = "Arial, sans-serif",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{escape(family)}" font-size="{size}" font-weight="{weight}" fill="{color}">'
        f"{escape(str(text))}</text>"
    )


def _svg_wrapped_label(
    x: float,
    y: float,
    text: str,
    *,
    width: float,
    size: int = 13,
    color: str = "#111827",
) -> str:
    max_chars = max(8, int(width / max(size * 0.58, 1)))
    lines = _wrap_label(text, max_chars)
    line_height = size + 4
    y_start = y - ((len(lines) - 1) * line_height / 2)
    return "\n".join(
        _svg_text(x, y_start + index * line_height, line, size=size, anchor="middle", color=color)
        for index, line in enumerate(lines)
    )


def _wrap_label(text: str, max_chars: int, max_lines: int = 3) -> list[str]:
    chunks: list[str] = []
    for word in text.split():
        if len(word) <= max_chars:
            chunks.append(word)
            continue
        chunks.extend(word[index : index + max_chars] for index in range(0, len(word), max_chars))

    lines: list[str] = []
    current = ""
    for chunk in chunks:
        candidate = chunk if not current else f"{current} {chunk}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = chunk
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _short_label(lines[-1], max_chars)
    return lines or [""]


def _svg_arrow(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return ""
    ux = dx / length
    uy = dy / length
    size = 10
    base_x = x2 - ux * size
    base_y = y2 - uy * size
    normal_x = -uy
    normal_y = ux
    p1 = (x2, y2)
    p2 = (base_x + normal_x * size * 0.48, base_y + normal_y * size * 0.48)
    p3 = (base_x - normal_x * size * 0.48, base_y - normal_y * size * 0.48)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (p1, p2, p3))
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{base_x:.1f}" y2="{base_y:.1f}" '
        f'stroke="{color}" stroke-width="2.2"/>\n'
        f'<polygon points="{points}" fill="{color}"/>'
    )


def _svg_footer(width: int, height: int) -> str:
    return (
        f'<text x="{width - 36}" y="{height - 20}" text-anchor="end" '
        'font-family="Arial, sans-serif" font-size="11" fill="#6b7280">'
        "Generated by stable-asr paper-figure</text>"
    )


def _heat_color(quality: float) -> str:
    quality = max(0.0, min(1.0, quality))
    if quality >= 0.80:
        return "#86efac"
    if quality >= 0.60:
        return "#bbf7d0"
    if quality >= 0.40:
        return "#fde68a"
    if quality >= 0.20:
        return "#fed7aa"
    return "#fecaca"


def _bar_chart_svg(spec: ChartSpec) -> str:
    if not spec.rows:
        raise ValueError(f"figure has no rows: {spec.title}")

    width = 960
    height = 540
    margin_left = 82
    margin_right = 36
    margin_top = 72
    margin_bottom = 118
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    raw_max = max(value for _, value in spec.rows)
    y_max = spec.y_max if spec.y_max is not None else raw_max * 1.15
    if y_max <= 0:
        y_max = 1.0

    bar_gap = 16
    bar_width = max(18, (plot_width - bar_gap * (len(spec.rows) + 1)) / len(spec.rows))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(spec.title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{escape(spec.title)}</text>',
        f'<text x="22" y="{margin_top + plot_height / 2:.1f}" transform="rotate(-90 22 {margin_top + plot_height / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#374151">{escape(spec.y_label)}</text>',
    ]

    for tick in range(6):
        value = y_max * tick / 5
        y = margin_top + plot_height - (value / y_max) * plot_height
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{_format_value(value)}</text>'
        )

    axis_y = margin_top + plot_height
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_y}" stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{axis_y}" x2="{width - margin_right}" y2="{axis_y}" stroke="#111827" stroke-width="1.5"/>'
    )

    palette = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#4b5563"]
    for index, (label, value) in enumerate(spec.rows):
        x = margin_left + bar_gap + index * (bar_width + bar_gap)
        bar_height = 0 if y_max == 0 else (value / y_max) * plot_height
        y = axis_y - bar_height
        color = palette[index % len(palette)]
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="3"/>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{max(margin_top + 14, y - 8):.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">{_format_value(value)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{axis_y + 22}" transform="rotate(30 {x + bar_width / 2:.1f} {axis_y + 22})" text-anchor="start" font-family="Arial, sans-serif" font-size="12" fill="#374151">{escape(_short_label(label))}</text>'
        )

    parts.append(
        f'<text x="{width - margin_right}" y="{height - 18}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">Generated by stable-asr paper-figure</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _short_label(label: str, limit: int = 22) -> str:
    return label if len(label) <= limit else label[: limit - 1] + "..."


def _format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")
