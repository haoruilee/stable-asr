"""End-to-end ASR metadata to weak turn-data bootstrap pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stable_asr.data.asr_manifest import ASRManifestRecord, summarize_asr_records
from stable_asr.data.recipes import prepare_asr_manifest
from stable_asr.data.registry import summarize_records, write_turn_records
from stable_asr.data.split import SPLIT_NAMES, TurnSplitConfig, split_turn_records
from stable_asr.data.turn_from_asr import ASRToTurnConfig, ASRToTurnResult, asr_records_to_turn_records


@dataclass(frozen=True)
class BootstrapTurnDataConfig:
    output_dir: Path
    turn_format: str = "jsonl"
    split_prefix: str = "turn"
    asr_manifest_name: str = "asr_manifest.jsonl"
    turn_manifest_name: str | None = None
    summary_name: str = "bootstrap_summary.json"
    report_name: str = "BOOTSTRAP_TURN_DATA.md"


@dataclass(frozen=True)
class BootstrapTurnDataResult:
    output_dir: str
    asr_manifest_path: str
    turn_manifest_path: str
    split_paths: dict[str, str]
    summary_path: str
    report_path: str
    asr_records: list[ASRManifestRecord]
    turn_result: ASRToTurnResult
    split_summaries: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "asr_manifest_path": self.asr_manifest_path,
            "turn_manifest_path": self.turn_manifest_path,
            "split_paths": self.split_paths,
            "summary_path": self.summary_path,
            "report_path": self.report_path,
            "asr_summary": summarize_asr_records(self.asr_records),
            "turn": self.turn_result.to_dict(),
            "splits": self.split_summaries,
        }

    def to_markdown(self) -> str:
        payload = self.to_dict()
        lines = [
            "# Stable-ASR Turn Data Bootstrap",
            "",
            f"- output_dir: `{self.output_dir}`",
            f"- asr_manifest: `{self.asr_manifest_path}`",
            f"- turn_manifest: `{self.turn_manifest_path}`",
            f"- summary: `{self.summary_path}`",
            "",
            "## ASR Input",
            "",
            f"- records: `{payload['asr_summary']['records']}`",
            f"- total_duration_sec: `{payload['asr_summary']['total_duration_sec']}`",
            f"- languages: `{json.dumps(payload['asr_summary']['languages'], ensure_ascii=False, sort_keys=True)}`",
            f"- sources: `{json.dumps(payload['asr_summary']['sources'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "## Turn Output",
            "",
            f"- records: `{payload['turn']['output_records']}`",
            f"- labels: `{json.dumps(payload['turn']['summary']['turn_labels'], ensure_ascii=False, sort_keys=True)}`",
            f"- actions: `{json.dumps(payload['turn']['summary']['action_labels'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "## Splits",
            "",
        ]
        split_summary = payload.get("splits", {})
        for split_name in SPLIT_NAMES:
            path = self.split_paths[split_name]
            records = ""
            if isinstance(split_summary, dict) and isinstance(split_summary.get(split_name), dict):
                records = f" ({split_summary[split_name].get('records', 0)} record(s))"
            lines.append(f"- `{split_name}`: `{path}`{records}")
        lines.extend(
            [
                "",
                "## Next Commands",
                "",
                "```bash",
                f"stable-asr validate-manifest {self.turn_manifest_path}",
                f"stable-asr inspect-manifest {self.turn_manifest_path}",
                "stable-asr audit-turn-splits "
                f"--train {self.split_paths['train']} "
                f"--dev {self.split_paths['dev']} "
                f"--test {self.split_paths['test']}",
                f"stable-asr train-turn --dataset {self.split_paths['train']} --output-dir {Path(self.output_dir) / 'nanoturn'}",
                "```",
            ]
        )
        return "\n".join(lines) + "\n"

    def to_text(self) -> str:
        turn_summary = summarize_records(self.turn_result.records)
        return "\n".join(
            [
                "bootstrap_turn_data:",
                f"- output_dir: {self.output_dir}",
                f"- asr_records: {len(self.asr_records)}",
                f"- turn_records: {len(self.turn_result.records)}",
                f"- turn_labels: {json.dumps(turn_summary['turn_labels'], ensure_ascii=False, sort_keys=True)}",
                f"- asr_manifest: {self.asr_manifest_path}",
                f"- turn_manifest: {self.turn_manifest_path}",
                f"- train: {self.split_paths['train']}",
                f"- dev: {self.split_paths['dev']}",
                f"- test: {self.split_paths['test']}",
                f"- report: {self.report_path}",
            ]
        )


def bootstrap_turn_data(
    input_path: str | Path,
    *,
    config: BootstrapTurnDataConfig,
    audio_root: str | Path | None = None,
    default_sample_rate: int = 16000,
    default_language: str = "unknown",
    default_source: str = "asr_manifest",
    default_split: str | None = None,
    id_field: str | None = None,
    audio_field: str | None = None,
    text_field: str | None = None,
    duration_field: str | None = None,
    speaker_field: str | None = None,
    asr_to_turn_config: ASRToTurnConfig | None = None,
    split_config: TurnSplitConfig | None = None,
) -> BootstrapTurnDataResult:
    """Create ASR, weak-turn, and train/dev/test manifests from metadata."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asr_manifest_path = output_dir / config.asr_manifest_name
    turn_manifest_path = output_dir / (config.turn_manifest_name or f"turn_manifest{_suffix(config.turn_format)}")
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    asr_records = prepare_asr_manifest(
        input_path,
        asr_manifest_path,
        audio_root=audio_root,
        default_sample_rate=default_sample_rate,
        default_language=default_language,
        default_source=default_source,
        default_split=default_split,
        id_field=id_field,
        audio_field=audio_field,
        text_field=text_field,
        duration_field=duration_field,
        speaker_field=speaker_field,
    )
    turn_result = asr_records_to_turn_records(asr_records, config=asr_to_turn_config)
    write_turn_records(turn_manifest_path, turn_result.records, format=config.turn_format)

    split_result = split_turn_records(
        turn_result.records,
        config=split_config or TurnSplitConfig(group_by="metadata.asr_record_id"),
    )
    split_summaries = split_result.to_dict()["splits"]
    split_paths = {
        name: split_dir / f"{config.split_prefix}_{name}{_suffix(config.turn_format)}"
        for name in SPLIT_NAMES
    }
    for name, path in split_paths.items():
        write_turn_records(path, split_result.split(name), format=config.turn_format)

    result = BootstrapTurnDataResult(
        output_dir=str(output_dir),
        asr_manifest_path=str(asr_manifest_path),
        turn_manifest_path=str(turn_manifest_path),
        split_paths={name: str(path) for name, path in split_paths.items()},
        summary_path=str(output_dir / config.summary_name),
        report_path=str(output_dir / config.report_name),
        asr_records=asr_records,
        turn_result=turn_result,
        split_summaries=split_summaries if isinstance(split_summaries, dict) else {},
    )
    Path(result.summary_path).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(result.report_path).write_text(result.to_markdown(), encoding="utf-8")
    return result


def _suffix(turn_format: str) -> str:
    if turn_format == "jsonl":
        return ".jsonl"
    if turn_format == "parquet":
        return ".parquet"
    if turn_format == "lance":
        return ".lance"
    raise ValueError(f"unknown turn format: {turn_format}")
