"""Repository-facing model/data/experiment card generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_asr.data.registry import load_turn_records, summarize_records
from stable_asr.eval.report import MarkdownReport, dict_table
from stable_asr.models.registry import find_model_entry, load_model_registry
from stable_asr.paper.tables import load_paper_results
from stable_asr.resources import resolve_platform_path


def dataset_card(manifest_path: str | Path, output_path: str | Path) -> str:
    resolved_manifest_path = resolve_platform_path(manifest_path)
    records = load_turn_records(resolved_manifest_path)
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


def model_card(
    model_source_path: str | Path,
    output_path: str | Path,
    *,
    model_id: str | None = None,
    metrics_path: str | Path | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Write a Markdown model card from a model registry or NanoTurn config."""

    payload = model_card_payload(
        model_source_path,
        model_id=model_id,
        metrics_path=metrics_path,
        metrics=metrics,
    )
    text = model_card_markdown(payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)


def model_card_payload(
    model_source_path: str | Path,
    *,
    model_id: str | None = None,
    metrics_path: str | Path | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_path = resolve_platform_path(model_source_path)
    with source_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    if not isinstance(source, dict):
        raise ValueError("model card source must be a JSON object")

    entry = _model_entry_from_source(source, model_id=model_id, source_path=model_source_path)
    resolved_metrics = metrics if metrics is not None else _load_metrics(metrics_path)
    return {
        "source_path": str(model_source_path),
        "model_id": entry["id"],
        "model": entry,
        "metrics_path": str(metrics_path) if metrics_path is not None else None,
        "metrics": resolved_metrics,
    }


def write_model_card_json(payload: dict[str, Any], output_path: str | Path) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(output_path)


def model_card_markdown(payload: dict[str, Any]) -> str:
    entry = payload["model"]
    metrics = payload.get("metrics") or {}
    report = MarkdownReport(f"Stable-ASR Model Card: {entry['title']}")
    report.add_section(
        "Summary",
        dict_table(
            [
                {
                    "model_id": entry["id"],
                    "family": entry["family"],
                    "task": entry["task"],
                    "type": entry["model_type"],
                    "status": entry["status"],
                    "modality": entry["modality"],
                    "license": entry["license"],
                }
            ]
        ),
    )
    report.add_section("Intended Use", str(entry["intended_use"]))
    report.add_section(
        "Interface",
        dict_table(
            [
                {
                    "interface": entry["interface"],
                    "input": entry["input_schema"],
                    "output": entry["output_schema"],
                    "entrypoint": entry["entrypoint"],
                }
            ]
        ),
    )
    report.add_section(
        "Labels And Actions",
        "\n".join(
            [
                f"- labels: `{', '.join(entry.get('labels', []))}`",
                f"- actions: `{', '.join(entry.get('actions', []))}`",
            ]
        ),
    )
    report.add_section(
        "Training And Evaluation",
        "\n".join(
            [
                f"- config: `{entry.get('config_path', 'not_specified')}`",
                f"- training: `{entry.get('training_entrypoint', 'not_specified')}`",
                f"- evaluation: `{entry.get('evaluation_entrypoint', 'not_specified')}`",
                f"- export: `{entry.get('export_entrypoint', 'not_specified')}`",
            ]
        ),
    )
    if metrics:
        report.add_section("Metrics", dict_table(_metric_rows(metrics)))
    report.add_section("Limitations", "\n".join(f"- {item}" for item in entry.get("limitations", [])))
    report.add_section(
        "Provenance",
        "\n".join(
            [
                f"- model source: `{payload['source_path']}`",
                f"- metrics source: `{payload['metrics_path'] or 'not_provided'}`",
            ]
        ),
    )
    return report.to_markdown()


def _model_entry_from_source(
    source: dict[str, Any],
    *,
    model_id: str | None,
    source_path: str | Path,
) -> dict[str, Any]:
    if "models" in source:
        models = source.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError("model registry must contain a non-empty models list")
        selected = model_id or "nanoturn_pico"
        return find_model_entry(load_model_registry(source_path), selected)
    if "model_type" in source:
        selected = model_id or str(source["model_type"])
        return _entry_from_nanoturn_config(source, selected)
    raise ValueError("model card source must be a model registry or a NanoTurn config JSON")


def _entry_from_nanoturn_config(source: dict[str, Any], model_id: str) -> dict[str, Any]:
    feature_source = str(source.get("feature_source", "metadata"))
    return {
        "id": model_id,
        "title": model_id.replace("_", " ").title(),
        "family": "NanoTurn",
        "task": "turn_taking",
        "model_type": "trainable_baseline",
        "status": "config_only",
        "modality": feature_source,
        "interface": "TurnPredictor",
        "entrypoint": f"stable_asr.turn.nanoturn.{model_id}",
        "training_entrypoint": f"stable-asr train-turn --model {model_id}",
        "evaluation_entrypoint": "stable-asr eval-turn --checkpoint <checkpoint.pt> --dataset <turn.jsonl>",
        "export_entrypoint": "stable-asr export-turn-onnx --checkpoint <checkpoint.pt> --output <model.onnx>",
        "input_schema": f"TurnManifestRecord features from {feature_source}",
        "output_schema": "TurnPrediction probabilities over four turn labels",
        "labels": ["backchannel", "complete", "incomplete", "wait"],
        "actions": [
            "continue_speaking",
            "hold",
            "ignore",
            "keep_listening",
            "light_ack",
            "stop_tts_and_listen",
            "take_turn",
        ],
        "config_path": "inline model config",
        "license": "project_license",
        "intended_use": "NanoTurn configuration card for Stable-ASR turn-taking experiments.",
        "limitations": [
            "configuration alone is not evidence of final model quality",
            "attach metrics from a trained checkpoint before citing performance",
        ],
        "config": source,
    }


def _load_metrics(metrics_path: str | Path | None) -> dict[str, Any] | None:
    if metrics_path is None:
        return None
    with resolve_platform_path(metrics_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("metrics file must be a JSON object")
    return payload


def _metric_rows(metrics: dict[str, Any]) -> list[dict[str, object]]:
    keys = (
        "model_type",
        "records",
        "epochs",
        "lr",
        "seed",
        "feature_source",
        "final_loss",
        "final_accuracy",
    )
    rows = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            value = f"{value:.6f}"
        rows.append({"metric": key, "value": value})
    return rows or [{"metric": "available_keys", "value": ", ".join(sorted(metrics.keys()))}]
