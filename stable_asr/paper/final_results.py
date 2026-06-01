"""Assemble final-scale experiment outputs into ``paper_results.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.asr_manifest import load_asr_manifest, summarize_asr_records
from stable_asr.data.registry import load_turn_records, summarize_records
from stable_asr.paper.final_config import load_final_run_config, validate_final_run_config


DEFAULT_RESULT_INPUTS: dict[str, str] = {
    "data_benchmark": "runs/final/reports/data_benchmark.json",
    "baselines": "runs/final/reports/baselines.json",
    "turn_benchmarks": "runs/final/reports/turn_benchmarks.json",
    "scenarios": "runs/final/reports/scenarios.json",
    "policy_search": "runs/final/reports/policy_search.json",
    "streaming_comparison": "runs/final/reports/asr_command_compare.json",
    "streaming_sweep": "runs/final/reports/whisper_sweep.json",
    "asr_transcript_conversions": "runs/final/reports/asr_transcript_conversions.json",
    "nanoturn": "runs/final/nanoturn/metrics.json",
}


@dataclass(frozen=True)
class FinalResultsInputCheck:
    name: str
    path: str
    exists: bool
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "exists": self.exists,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FinalResultsAssemblyReport:
    ok: bool
    output: str
    wrote: bool
    checks: list[FinalResultsInputCheck]
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "output": self.output,
            "wrote": self.wrote,
            "checks": [check.to_dict() for check in self.checks],
            "errors": self.errors,
        }

    def to_text(self) -> str:
        lines = [
            f"final_results_assembly: {'READY' if self.ok else 'NOT_READY'}",
            f"- output: {self.output}",
            f"- wrote: {self.wrote}",
        ]
        for check in self.checks:
            status = "OK" if check.ok else "MISSING"
            lines.append(f"- {status} {check.name}: {check.path} ({check.detail})")
        if self.errors:
            lines.append("- errors:")
            lines.extend(f"  - {error}" for error in self.errors)
        return "\n".join(lines)


def assemble_final_paper_results(
    config: dict[str, Any] | str | Path | None = None,
    *,
    repo_root: str | Path = ".",
    output_path: str | Path | None = None,
    allow_missing: bool = False,
    write: bool = True,
) -> FinalResultsAssemblyReport:
    """Assemble final JSON outputs into the paper result schema.

    The assembler is intentionally strict by default: it refuses to write a
    final ``paper_results.json`` until all configured result inputs exist and
    parse as JSON. ``allow_missing`` is only for scaffolding/debugging and writes
    explicit placeholders instead of benchmark claims.
    """

    if config is None or isinstance(config, (str, Path)):
        config = load_final_run_config(config)
    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    root = Path(repo_root)
    output = _resolve(str(output_path or config["artifacts"]["paper_results"]), root=root)
    checks: list[FinalResultsInputCheck] = []
    errors: list[str] = []

    turn_test_path = _resolve(str(config["turn_splits"]["test"]), root=root)
    asr_eval_path = _resolve(str(config["asr_eval_manifest"]), root=root)
    result_inputs = _result_inputs(config)

    turn_records = []
    asr_records = []
    payloads: dict[str, Any] = {}

    try:
        turn_records = load_turn_records(turn_test_path, format="jsonl")
        checks.append(_ok_check("turn_test", turn_test_path, f"{len(turn_records)} turn record(s)"))
    except (OSError, ValueError) as exc:
        checks.append(_missing_check("turn_test", turn_test_path, str(exc)))
        errors.append(f"turn_test unavailable: {exc}")

    try:
        asr_records = load_asr_manifest(asr_eval_path)
        checks.append(_ok_check("asr_eval_manifest", asr_eval_path, f"{len(asr_records)} ASR record(s)"))
    except (OSError, ValueError) as exc:
        checks.append(_missing_check("asr_eval_manifest", asr_eval_path, str(exc)))
        errors.append(f"asr_eval_manifest unavailable: {exc}")

    for name, relative_path in result_inputs.items():
        path = _resolve(relative_path, root=root)
        if not path.exists():
            checks.append(_missing_check(name, path, "missing result input"))
            errors.append(f"{name} result input is missing: {path}")
            continue
        try:
            payloads[name] = _load_json(path)
            checks.append(_ok_check(name, path, "loaded JSON"))
        except (OSError, ValueError) as exc:
            checks.append(_missing_check(name, path, str(exc)))
            errors.append(f"{name} result input is invalid: {exc}")

    ok = not errors
    wrote = False
    if ok or allow_missing:
        results = _build_results(config, turn_records, asr_records, payloads, result_inputs)
        if write:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            wrote = True

    return FinalResultsAssemblyReport(
        ok=ok,
        output=str(output),
        wrote=wrote,
        checks=checks,
        errors=[] if allow_missing else errors,
    )


def _result_inputs(config: dict[str, Any]) -> dict[str, str]:
    raw = config.get("result_inputs", DEFAULT_RESULT_INPUTS)
    if not isinstance(raw, dict):
        raise ValueError("result_inputs must be an object")
    result: dict[str, str] = {}
    for key, value in {**DEFAULT_RESULT_INPUTS, **raw}.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"result_inputs.{key} must be a non-empty string")
        result[str(key)] = value
    return result


def _build_results(
    config: dict[str, Any],
    turn_records,
    asr_records,
    payloads: dict[str, Any],
    result_inputs: dict[str, str],
) -> dict[str, Any]:
    data_benchmark = _data_benchmark(payloads.get("data_benchmark"))
    return {
        "meta": {
            "artifact_version": "final_v0",
            "config_id": config["id"],
            "config_version": config["version"],
            "episodes": len(turn_records),
            "seed": config["seed"],
            "result_inputs": result_inputs,
        },
        "data": {
            "summary": summarize_records(turn_records) if turn_records else {},
            "benchmark": data_benchmark,
            "asr_manifest_recipe": {
                "records": len(asr_records),
                "summary": summarize_asr_records(asr_records) if asr_records else {},
                "validation": {"ok": bool(asr_records), "path": config["asr_eval_manifest"]},
            },
            "external_conversion": _external_conversion_summary(payloads.get("asr_transcript_conversions")),
            "external_conversions": _asr_transcript_conversions(
                payloads.get("asr_transcript_conversions")
            ),
        },
        "baselines": _baseline_results(payloads.get("baselines")),
        "turn_benchmarks": _turn_benchmarks(payloads.get("turn_benchmarks")),
        "scenarios": _scenario_results(payloads.get("scenarios")),
        "policy_search": _dict(payloads.get("policy_search")),
        "streaming_asr": {
            "metrics": _first_streaming_metrics(payloads.get("streaming_comparison")),
            "adapter_comparison": _dict(payloads.get("streaming_comparison")),
            "schedule_sweep": _dict(payloads.get("streaming_sweep")),
            "asr_transcript_conversions": _asr_transcript_conversions(
                payloads.get("asr_transcript_conversions")
            ),
            "command_adapter": _command_adapter_summary(payloads.get("streaming_comparison")),
        },
        "nanoturn": {
            "status": "completed" if "nanoturn" in payloads else "missing",
            "checkpoint_path": config["nanoturn"]["checkpoint"],
            "metrics_path": result_inputs["nanoturn"],
            "metrics": _dict(payloads.get("nanoturn")),
        },
    }


def _data_benchmark(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"status": "completed", "rows": payload}
    if isinstance(payload, dict):
        if "status" in payload:
            return dict(payload)
        if isinstance(payload.get("rows"), list):
            return {"status": "completed", **payload}
    return {"status": "missing", "rows": []}


def _first_streaming_metrics(payload: Any) -> dict[str, Any]:
    payload = _dict(payload)
    rows = payload.get("rows", [])
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return dict(rows[0])
    return {}


def _scenario_results(payload: Any) -> dict[str, Any]:
    result = _dict(payload)
    if "factor_summary" not in result:
        result["factor_summary"] = {}
    return result


def _baseline_results(payload: Any) -> dict[str, Any]:
    payload = _dict(payload)
    reports = payload.get("reports")
    if isinstance(reports, dict):
        result = {str(name): _dict(report) for name, report in reports.items()}
    else:
        result = payload
    if "prediction_manifest" not in result:
        for name in ("smart_turn", "easy_turn", "vap"):
            if name in result:
                result["prediction_manifest"] = dict(_dict(result[name]))
                break
    if "nanoturn_pico" not in result and "nanoturn" in result:
        result["nanoturn_pico"] = dict(_dict(result["nanoturn"]))
    return result


def _turn_benchmarks(payload: Any) -> dict[str, Any]:
    payload = _dict(payload)
    if "avg_latency_ms" in payload:
        name = str(payload.get("name", payload.get("baseline", "system")))
        result = {name: payload}
    else:
        result = payload
    if "nanoturn_pico" not in result and "nanoturn" in result:
        result["nanoturn_pico"] = dict(_dict(result["nanoturn"]))
    if "prediction_manifest" not in result:
        for name in ("smart_turn", "easy_turn", "vap"):
            if name in result:
                result["prediction_manifest"] = dict(_dict(result[name]))
                break
    return result


def _external_conversion_summary(payload: Any) -> dict[str, Any]:
    conversions = _asr_transcript_conversions(payload)
    return {
        "records": sum(int(item.get("records", 0)) for item in conversions),
        "schemas": [str(item.get("schema", "unknown")) for item in conversions],
        "count": len(conversions),
    }


def _command_adapter_summary(payload: Any) -> dict[str, Any]:
    payload = _dict(payload)
    rows = payload.get("rows", [])
    if isinstance(rows, list) and rows:
        total_records = sum(int(row.get("records", 0)) for row in rows if isinstance(row, dict))
        metrics = dict(_dict(rows[0]))
        metrics["records"] = total_records
        return {"adapter": "command_fixture", "metrics": metrics, "rows": rows}
    return {"metrics": {"records": 0}, "rows": []}


def _asr_transcript_conversions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    payload = _dict(payload)
    rows = payload.get("rows", [])
    if isinstance(rows, list):
        conversions = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            conversions.append(
                {
                    "schema": str(row.get("schema", row.get("adapter", "unknown"))),
                    "records": int(row.get("records", 0)),
                    "metrics": dict(row),
                }
            )
        return conversions
    return []


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ok_check(name: str, path: Path, detail: str) -> FinalResultsInputCheck:
    return FinalResultsInputCheck(name=name, path=str(path), exists=True, ok=True, detail=detail)


def _missing_check(name: str, path: Path, detail: str) -> FinalResultsInputCheck:
    return FinalResultsInputCheck(name=name, path=str(path), exists=path.exists(), ok=False, detail=detail)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve(value: str, *, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path
