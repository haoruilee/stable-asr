#!/usr/bin/env python3
"""Generate latency vs. accuracy Pareto curve for NanoTurn variants.

Reads benchmark JSON outputs produced by `stable-asr benchmark-turn` and
compare-turn, then plots:
  - X axis: p50 inference latency (ms)
  - Y axis: accuracy or macro-F1 on test split
  - Each point: one model variant or baseline
  - Pareto frontier highlighted

Usage:
    python scripts/plot_pareto.py \\
        --benchmark-dir runs/eval/benchmark \\
        --compare-json  runs/eval/compare_turn/baselines.json \\
        --output        runs/eval/pareto.png \\
        [--metric accuracy|f1] \\
        [--no-show]

Outputs:
    <output>.png  — publication-quality figure (300 dpi)
    <output>.json — raw data table used for the plot

Dependencies:
    matplotlib, numpy  (both in standard ML envs)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# ── data loading ─────────────────────────────────────────────────────────────

def load_benchmark_dir(bench_dir: Path) -> dict[str, dict]:
    """Load all benchmark-turn JSON files from a directory.

    Returns dict: model_name → {avg_latency_ms, p50_latency_ms, p95_latency_ms, rtf, ...}
    """
    results = {}
    for jf in sorted(bench_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
            if isinstance(data, list):
                data = data[0]
            results[jf.stem] = data
        except Exception as e:
            print(f"  Warning: could not load {jf}: {e}")
    return results


def load_compare_json(compare_json: Path) -> dict[str, dict]:
    """Load compare-turn output JSON.

    Returns dict: system_name → {accuracy, macro_f1, precision, recall, ...}
    """
    data = json.loads(compare_json.read_text())
    results = {}

    # compare-turn can output either a list of rows or a nested dict
    if isinstance(data, list):
        for row in data:
            name = row.get("system") or row.get("name") or row.get("model", "unknown")
            results[name] = row
    elif isinstance(data, dict):
        # Nested: {"system_name": {"classification": {...}, ...}, ...}
        for name, metrics in data.items():
            if isinstance(metrics, dict):
                # Flatten nested classification metrics
                flat = dict(metrics)
                if "classification" in metrics:
                    flat.update(metrics["classification"])
                results[name] = flat
    return results


def merge_latency_accuracy(
    bench_data: dict[str, dict],
    compare_data: dict[str, dict],
    *,
    metric: str = "accuracy",
) -> list[dict]:
    """Merge latency and accuracy data by model name.

    Returns list of dicts with keys: name, latency_ms, metric_value, has_latency.
    """
    all_names = set(bench_data) | set(compare_data)
    rows = []

    for name in sorted(all_names):
        bench = bench_data.get(name, {})
        comp  = compare_data.get(name, {})

        latency_ms = (
            bench.get("p50_latency_ms")
            or bench.get("avg_latency_ms")
            or bench.get("latency_ms")
        )

        acc = (
            comp.get(metric)
            or comp.get("accuracy")
            or comp.get("macro_f1")
            or comp.get("f1")
        )

        # Also check nested classification dict
        if acc is None and "classification" in comp:
            clf = comp["classification"]
            acc = clf.get(metric) or clf.get("accuracy") or clf.get("macro_f1")

        if acc is None and latency_ms is None:
            continue

        rows.append({
            "name": name,
            "latency_ms": float(latency_ms) if latency_ms is not None else None,
            "p95_latency_ms": float(bench.get("p95_latency_ms", 0)) if bench else None,
            "metric_value": float(acc) if acc is not None else None,
            "rtf": float(bench.get("rtf", 0)) if bench else None,
            "has_latency": latency_ms is not None,
            "has_metric": acc is not None,
        })

    return rows


# ── Pareto frontier ────────────────────────────────────────────────────────

def pareto_frontier(rows: list[dict]) -> list[str]:
    """Return names of points on the Pareto frontier (min latency, max metric)."""
    valid = [r for r in rows if r["latency_ms"] is not None and r["metric_value"] is not None]
    if not valid:
        return []

    # Sort by latency ascending
    valid.sort(key=lambda r: r["latency_ms"])
    frontier = []
    best_metric = -math.inf
    for r in valid:
        if r["metric_value"] >= best_metric:
            frontier.append(r["name"])
            best_metric = r["metric_value"]
    return frontier


# ── plotting ────────────────────────────────────────────────────────────────

# Colour scheme: NanoTurn family = blues, baselines = greys, VAP = orange
_COLOUR_MAP = {
    "nanoturn_pico":       "#2196F3",
    "nanoturn_nano":       "#1565C0",
    "nanoturn_pico_v1":    "#42A5F5",
    "nanoturn_nano_v1":    "#0D47A1",
    "nanoturn_micro":      "#00BCD4",
    "vap":                 "#FF6F00",
    "rule_endpoint":       "#9E9E9E",
    "vad_pause":           "#757575",
    "text_turn":           "#616161",
    "smart_turn":          "#E91E63",
    "easy_turn":           "#9C27B0",
}

_MARKER_MAP = {
    "nanoturn_pico":       "o",
    "nanoturn_nano":       "o",
    "nanoturn_pico_v1":    "s",
    "nanoturn_nano_v1":    "s",
    "nanoturn_micro":      "^",
    "vap":                 "D",
    "rule_endpoint":       "x",
    "vad_pause":           "x",
    "text_turn":           "x",
}


def _colour(name: str) -> str:
    for key, colour in _COLOUR_MAP.items():
        if key in name.lower():
            return colour
    return "#4CAF50"


def _marker(name: str) -> str:
    for key, marker in _MARKER_MAP.items():
        if key in name.lower():
            return marker
    return "o"


def plot_pareto(
    rows: list[dict],
    *,
    output: Path,
    metric: str = "accuracy",
    title: str = "Latency vs. Accuracy — NanoTurn Family",
    show: bool = False,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Skipping plot; JSON output still written.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Inference latency — p50 (ms)", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3, linestyle="--")

    frontier_names = set(pareto_frontier(rows))

    # Separate into two groups: have both latency+metric, or only one
    full = [r for r in rows if r["latency_ms"] is not None and r["metric_value"] is not None]
    acc_only = [r for r in rows if r["latency_ms"] is None and r["metric_value"] is not None]

    # Plot points with latency and metric
    for r in full:
        colour = _colour(r["name"])
        marker = _marker(r["name"])
        on_frontier = r["name"] in frontier_names
        ax.scatter(
            r["latency_ms"],
            r["metric_value"],
            c=colour,
            marker=marker,
            s=120 if on_frontier else 70,
            zorder=5,
            edgecolors="black" if on_frontier else "none",
            linewidths=1.2,
        )
        # Error bar for p95 if available
        if r.get("p95_latency_ms") and r["p95_latency_ms"] > r["latency_ms"]:
            ax.errorbar(
                r["latency_ms"],
                r["metric_value"],
                xerr=[[0], [r["p95_latency_ms"] - r["latency_ms"]]],
                fmt="none",
                ecolor=colour,
                alpha=0.4,
                capsize=3,
            )
        # Label
        ax.annotate(
            r["name"].replace("nanoturn_", "NT-"),
            (r["latency_ms"], r["metric_value"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=colour,
        )

    # Draw Pareto frontier line
    frontier_rows = sorted(
        [r for r in full if r["name"] in frontier_names],
        key=lambda r: r["latency_ms"],
    )
    if len(frontier_rows) >= 2:
        xs = [r["latency_ms"] for r in frontier_rows]
        ys = [r["metric_value"] for r in frontier_rows]
        ax.step(xs, ys, where="post", color="#333", linestyle="--", alpha=0.5,
                linewidth=1.2, label="Pareto frontier")

    # Plot accuracy-only baselines as vertical reference lines
    for r in acc_only:
        ax.axhline(
            r["metric_value"],
            color=_colour(r["name"]),
            linestyle=":",
            alpha=0.6,
            linewidth=1,
        )
        ax.text(
            ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 0 else 1,
            r["metric_value"] + 0.002,
            r["name"],
            ha="right",
            fontsize=8,
            color=_colour(r["name"]),
        )

    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {output}")
    if show:
        plt.show()
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("runs/eval/benchmark"),
        help="Directory with benchmark-turn *.json files (default: runs/eval/benchmark)",
    )
    parser.add_argument(
        "--compare-json",
        type=Path,
        default=Path("runs/eval/compare_turn/baselines.json"),
        help="compare-turn output JSON (default: runs/eval/compare_turn/baselines.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/eval/pareto.png"),
        help="Output figure path (default: runs/eval/pareto.png)",
    )
    parser.add_argument(
        "--metric",
        default="accuracy",
        choices=["accuracy", "macro_f1", "f1"],
        help="Metric for Y axis (default: accuracy)",
    )
    parser.add_argument(
        "--title",
        default="Latency vs. Accuracy — NanoTurn Family",
        help="Plot title",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot interactively",
    )
    args = parser.parse_args()

    bench_data: dict[str, dict] = {}
    if args.benchmark_dir.exists():
        bench_data = load_benchmark_dir(args.benchmark_dir)
        print(f"Loaded {len(bench_data)} benchmark entries from {args.benchmark_dir}")
    else:
        print(f"Warning: benchmark dir not found: {args.benchmark_dir}")

    compare_data: dict[str, dict] = {}
    if args.compare_json.exists():
        compare_data = load_compare_json(args.compare_json)
        print(f"Loaded {len(compare_data)} compare entries from {args.compare_json}")
    else:
        print(f"Warning: compare JSON not found: {args.compare_json}")

    if not bench_data and not compare_data:
        print("No data to plot. Run benchmark-turn and compare-turn first.")
        return

    rows = merge_latency_accuracy(bench_data, compare_data, metric=args.metric)
    print(f"\nMerged {len(rows)} data points:")
    for r in rows:
        lat = f"{r['latency_ms']:.2f}ms" if r["latency_ms"] else "no-latency"
        met = f"{r['metric_value']:.4f}" if r["metric_value"] else "no-metric"
        print(f"  {r['name']:<30} latency={lat:<12} {args.metric}={met}")

    # Write JSON table
    json_out = args.output.with_suffix(".json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nData table: {json_out}")

    # Pareto frontier
    frontier = pareto_frontier(rows)
    print(f"Pareto frontier: {frontier}")

    # Plot
    plot_pareto(
        rows,
        output=args.output,
        metric=args.metric,
        title=args.title,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
