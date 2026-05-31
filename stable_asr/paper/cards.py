"""Repository-facing model/data card generation."""

from __future__ import annotations

from pathlib import Path

from stable_asr.data.registry import load_turn_records, summarize_records
from stable_asr.eval.report import MarkdownReport, dict_table
from stable_asr.paper.tables import load_paper_results


def dataset_card(manifest_path: str | Path, output_path: str | Path) -> str:
    records = load_turn_records(manifest_path)
    summary = summarize_records(records)
    report = MarkdownReport("Stable-ASR Dataset Card")
    report.add_section("Source", f"- manifest: `{manifest_path}`")
    report.add_section(
        "Summary",
        dict_table(
            [
                {
                    "records": summary["records"],
                    "languages": ", ".join(summary["languages"].keys()),
                    "scenarios": len(summary["scenarios"]),
                }
            ]
        ),
    )
    report.add_section("Turn Labels", dict_table([{"label": k, "count": v} for k, v in summary["turn_labels"].items()]))
    report.add_section("Actions", dict_table([{"action": k, "count": v} for k, v in summary["action_labels"].items()]))
    text = report.to_markdown()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)


def experiment_card(results_path: str | Path, output_path: str | Path) -> str:
    results = load_paper_results(results_path)
    report = MarkdownReport("Stable-ASR Experiment Card")
    meta = results["meta"]
    report.add_section(
        "Run",
        dict_table(
            [
                {
                    "artifact_version": meta["artifact_version"],
                    "episodes": meta["episodes"],
                    "seed": meta["seed"],
                }
            ]
        ),
    )
    data = results["data"]
    report.add_section(
        "Artifacts",
        "\n".join(
            [
                f"- manifest: `{data['manifest_path']}`",
                f"- converted: `{data['converted_path']}`",
            ]
        ),
    )
    if "streaming_asr" in results:
        streaming = results["streaming_asr"]["metrics"]
        report.add_section(
            "Streaming ASR",
            dict_table(
                [
                    {
                        "wer": f"{streaming['wer']:.4f}",
                        "cer": f"{streaming['cer']:.4f}",
                        "rtf": f"{streaming['rtf']:.4f}",
                        "partial_revision_rate": f"{streaming['partial_revision_rate']:.4f}",
                    }
                ]
            ),
        )
    text = report.to_markdown()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)

