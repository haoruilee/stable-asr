"""Final paper run configuration schema and renderer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table


DEFAULT_FINAL_RUN_CONFIG: dict[str, Any] = {
    "id": "stable_asr_final_run_v0",
    "version": "0.1.0",
    "title": "Stable-ASR Final Paper Run Configuration",
    "description": (
        "Template configuration for a final-scale Stable-ASR platform paper run. "
        "Paths are intentionally explicit so the final benchmark can be audited "
        "before expensive jobs are launched."
    ),
    "output_dir": "runs/final",
    "seed": 0,
    "public_corpora": [
        {
            "id": "librispeech_dev_clean",
            "language": "en",
            "metadata": "data/librispeech/dev-clean/metadata.tsv",
            "audio_root": "data/librispeech/dev-clean/audio",
            "manifest": "runs/final/librispeech_dev_clean/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "see_upstream",
        },
        {
            "id": "aishell1_dev",
            "language": "zh",
            "metadata": "data/aishell1/dev/metadata.tsv",
            "audio_root": "data/aishell1/dev/audio",
            "manifest": "runs/final/aishell1_dev/asr_manifest.jsonl",
            "sample_rate": 16000,
            "license": "see_upstream",
        },
    ],
    "turn_splits": {
        "train": "runs/final/turn_train.jsonl",
        "dev": "runs/final/turn_dev.jsonl",
        "test": "runs/final/turn_test.jsonl",
        "voiceworld_real": "runs/final/voiceworld_real.jsonl",
    },
    "external_turn_predictions": [
        {
            "id": "smart_turn",
            "schema": "smart_turn",
            "raw": "runs/final/external/smartturn_raw.jsonl",
            "converted": "runs/final/external/smartturn_predictions.jsonl",
        },
        {
            "id": "easy_turn",
            "schema": "easyturn",
            "raw": "runs/final/external/easyturn_raw.jsonl",
            "converted": "runs/final/external/easyturn_predictions.jsonl",
        },
    ],
    "asr_command_config": "configs/final/asr_command_compare.json",
    "nanoturn": {
        "model": "nanoturn_pico",
        "checkpoint": "runs/final/nanoturn/checkpoint.pt",
        "metrics": "runs/final/nanoturn/metrics.json",
        "onnx": "runs/final/nanoturn/nanoturn.onnx",
    },
    "artifacts": {
        "paper_results": "runs/final/paper_results.json",
        "bundle_dir": "runs/final/artifacts",
        "markdown_draft": "runs/final/PAPER_DRAFT.md",
        "latex_draft": "runs/final/paper.tex",
        "dataset_card": "runs/final/DATASET_CARD.md",
        "experiment_card": "runs/final/EXPERIMENT_CARD.md",
    },
    "commands": [
        "stable-asr final-config --config configs/final/paper_final.json --validate-only",
        "stable-asr prepare-asr-manifest --input data/librispeech/dev-clean/metadata.tsv --output runs/final/librispeech_dev_clean/asr_manifest.jsonl --audio-root data/librispeech/dev-clean/audio --sample-rate 16000 --language en --source librispeech",
        "stable-asr train-turn --dataset runs/final/turn_train.jsonl --output-dir runs/final/nanoturn --model nanoturn_pico --feature-source audio",
        "stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --report runs/final/reports/asr_command_compare.md",
        "stable-asr paper-bundle --results runs/final/paper_results.json --output-dir runs/final/artifacts",
        "stable-asr paper-parity-audit --results runs/final/paper_results.json --artifacts-dir runs/final/artifacts --require-final",
    ],
}


@dataclass(frozen=True)
class FinalRunConfigValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "final_run_config: OK"
        return "final_run_config: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class FinalRunPathCheck:
    name: str
    path: str
    kind: str
    required: bool
    exists: bool
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "required": self.required,
            "exists": self.exists,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FinalRunFileAudit:
    ok: bool
    checks: list[FinalRunPathCheck]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checks": [check.to_dict() for check in self.checks]}

    def to_text(self) -> str:
        lines = [f"final_run_file_audit: {'READY' if self.ok else 'NOT_READY'}"]
        for check in self.checks:
            status = "OK" if check.ok else "MISSING"
            required = "required" if check.required else "planned"
            lines.append(f"- {status} {check.kind}/{check.name}: {check.path} ({required}; {check.detail})")
        return "\n".join(lines)


@dataclass(frozen=True)
class FinalRunScaffoldEntry:
    path: str
    kind: str
    created: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "created": self.created,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FinalRunScaffoldReport:
    output_dir: str
    entries: list[FinalRunScaffoldEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_text(self) -> str:
        lines = [f"final_run_scaffold: {self.output_dir}"]
        for entry in self.entries:
            status = "created" if entry.created else "exists"
            lines.append(f"- {status} {entry.kind}: {entry.path} ({entry.detail})")
        return "\n".join(lines)


def load_final_run_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_FINAL_RUN_CONFIG))
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("final run config must be a JSON object")
    return payload


def write_final_run_config_json(path: str | Path, config: dict[str, Any] | None = None) -> str:
    config = config or load_final_run_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_final_run_config(config: dict[str, Any]) -> FinalRunConfigValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "output_dir", "seed", "public_corpora", "turn_splits", "artifacts", "commands"):
        if key not in config:
            errors.append(f"missing top-level key: {key}")

    corpora = config.get("public_corpora")
    if not isinstance(corpora, list) or not corpora:
        errors.append("public_corpora must be a non-empty list")
    else:
        seen: set[str] = set()
        for index, corpus in enumerate(corpora):
            if not isinstance(corpus, dict):
                errors.append(f"corpus {index} must be an object")
                continue
            corpus_id = corpus.get("id")
            if not isinstance(corpus_id, str) or not corpus_id:
                errors.append(f"corpus {index} missing id")
            elif corpus_id in seen:
                errors.append(f"duplicate corpus id: {corpus_id}")
            else:
                seen.add(corpus_id)
            for key in ("language", "metadata", "audio_root", "manifest", "sample_rate", "license"):
                if key not in corpus:
                    errors.append(f"corpus {corpus_id or index} missing {key}")

    turn_splits = config.get("turn_splits")
    required_splits = {"train", "dev", "test", "voiceworld_real"}
    if not isinstance(turn_splits, dict):
        errors.append("turn_splits must be an object")
    else:
        missing_splits = sorted(required_splits.difference(turn_splits))
        if missing_splits:
            errors.append("turn_splits missing: " + ", ".join(missing_splits))

    predictions = config.get("external_turn_predictions", [])
    if predictions is not None and not isinstance(predictions, list):
        errors.append("external_turn_predictions must be a list")
    elif isinstance(predictions, list):
        for index, prediction in enumerate(predictions):
            if not isinstance(prediction, dict):
                errors.append(f"external prediction {index} must be an object")
                continue
            prediction_id = prediction.get("id", index)
            for key in ("id", "schema", "raw", "converted"):
                if key not in prediction:
                    errors.append(f"external prediction {prediction_id} missing {key}")

    nanoturn = config.get("nanoturn", {})
    if not isinstance(nanoturn, dict):
        errors.append("nanoturn must be an object")
    else:
        for key in ("model", "checkpoint", "metrics", "onnx"):
            if key not in nanoturn:
                errors.append(f"nanoturn missing {key}")

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
    else:
        for key in ("paper_results", "bundle_dir", "markdown_draft", "latex_draft", "dataset_card", "experiment_card"):
            if key not in artifacts:
                errors.append(f"artifacts missing {key}")

    commands = config.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("commands must be a non-empty list")
    elif not all(isinstance(command, str) and command.strip() for command in commands):
        errors.append("commands must contain non-empty strings")

    asr_command_config = config.get("asr_command_config")
    if not isinstance(asr_command_config, str) or not asr_command_config:
        errors.append("asr_command_config must be a non-empty string")

    return FinalRunConfigValidation(ok=not errors, errors=errors)


def audit_final_run_files(config: dict[str, Any], *, repo_root: str | Path = ".") -> FinalRunFileAudit:
    validation = validate_final_run_config(config)
    if not validation.ok:
        checks = [
            FinalRunPathCheck(
                name="schema",
                path="",
                kind="config",
                required=True,
                exists=False,
                ok=False,
                detail="; ".join(validation.errors),
            )
        ]
        return FinalRunFileAudit(ok=False, checks=checks)

    root = Path(repo_root)
    checks: list[FinalRunPathCheck] = []
    for corpus in config.get("public_corpora", []):
        corpus_id = str(corpus["id"])
        checks.append(_input_check(f"corpus:{corpus_id}:metadata", corpus["metadata"], root=root))
        checks.append(_input_check(f"corpus:{corpus_id}:audio_root", corpus["audio_root"], root=root))
        checks.append(_planned_check(f"corpus:{corpus_id}:manifest", corpus["manifest"], root=root, kind="output"))

    for split, path in config.get("turn_splits", {}).items():
        checks.append(_input_check(f"turn_split:{split}", path, root=root))

    for prediction in config.get("external_turn_predictions", []):
        prediction_id = str(prediction["id"])
        checks.append(_input_check(f"external_prediction:{prediction_id}:raw", prediction["raw"], root=root))
        checks.append(_planned_check(f"external_prediction:{prediction_id}:converted", prediction["converted"], root=root, kind="output"))

    checks.append(_input_check("asr_command_config", config["asr_command_config"], root=root, kind="config"))

    for name, path in config.get("nanoturn", {}).items():
        if name == "model":
            continue
        checks.append(_planned_check(f"nanoturn:{name}", path, root=root, kind="output"))

    for name, path in config.get("artifacts", {}).items():
        checks.append(_planned_check(f"artifact:{name}", path, root=root, kind="output"))

    return FinalRunFileAudit(ok=all(check.ok for check in checks), checks=checks)


def final_run_config_markdown(config: dict[str, Any]) -> str:
    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    lines = [
        f"# {config['title']}",
        "",
        f"- id: `{config['id']}`",
        f"- version: `{config['version']}`",
        f"- output dir: `{config['output_dir']}`",
        f"- seed: `{config['seed']}`",
        "",
        str(config.get("description", "")),
        "",
        "## Public Corpora",
        "",
        dict_table(_corpus_rows(config)),
        "",
        "## Turn Splits",
        "",
    ]
    for name, path in config["turn_splits"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## External Turn Predictions", ""])
    if config.get("external_turn_predictions"):
        lines.append(dict_table(_prediction_rows(config)))
    else:
        lines.append("No external turn predictions configured.")
    lines.extend(["", "## Artifacts", ""])
    for name, path in config["artifacts"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Commands", ""])
    lines.extend(f"```bash\n{command}\n```" for command in config["commands"])
    lines.append("")
    return "\n".join(lines)


def final_run_file_audit_markdown(report: FinalRunFileAudit) -> str:
    lines = [
        "# Stable-ASR Final Run File Audit",
        "",
        f"- status: `{'READY' if report.ok else 'NOT_READY'}`",
        "",
        dict_table([check.to_dict() for check in report.checks]),
        "",
    ]
    return "\n".join(lines)


def scaffold_final_run(config: dict[str, Any], *, repo_root: str | Path = ".") -> FinalRunScaffoldReport:
    """Create final-run directories and README hints without fabricating data."""

    validation = validate_final_run_config(config)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))

    root = Path(repo_root)
    entries: list[FinalRunScaffoldEntry] = []
    output_dir = _resolve(str(config["output_dir"]), root=root)
    entries.append(_ensure_dir(output_dir, "output_dir"))
    entries.append(
        _ensure_readme(
            output_dir / "README.md",
            title="Stable-ASR Final Run Workspace",
            body=(
                "This directory is reserved for final-scale paper experiments.\n\n"
                "Do not treat placeholder directories as benchmark evidence. "
                "Run `stable-asr final-config --check-files` before launching final jobs.\n"
            ),
        )
    )

    for corpus in config.get("public_corpora", []):
        corpus_id = str(corpus["id"])
        metadata_parent = _resolve(str(corpus["metadata"]), root=root).parent
        entries.append(_ensure_dir(metadata_parent, f"corpus:{corpus_id}:metadata_parent"))
        entries.append(
            _ensure_readme(
                metadata_parent / "README.md",
                title=f"Corpus Input: {corpus_id}",
                body=(
                    f"Place the metadata table at `{corpus['metadata']}` and audio under "
                    f"`{corpus['audio_root']}`.\n\n"
                    "This scaffold does not create metadata or audio files.\n"
                ),
            )
        )
        manifest_parent = _resolve(str(corpus["manifest"]), root=root).parent
        entries.append(_ensure_dir(manifest_parent, f"corpus:{corpus_id}:manifest_parent"))

    split_parent = _resolve(str(config["turn_splits"]["train"]), root=root).parent
    entries.append(_ensure_dir(split_parent, "turn_splits_parent"))
    entries.append(
        _ensure_readme(
            split_parent / "TURN_SPLITS_README.md",
            title="Turn Split Inputs",
            body="\n".join(f"- `{name}`: `{path}`" for name, path in config["turn_splits"].items())
            + "\n\nThese files must be real Stable-ASR turn manifests.\n",
        )
    )

    for prediction in config.get("external_turn_predictions", []):
        prediction_id = str(prediction["id"])
        raw_parent = _resolve(str(prediction["raw"]), root=root).parent
        converted_parent = _resolve(str(prediction["converted"]), root=root).parent
        entries.append(_ensure_dir(raw_parent, f"external_prediction:{prediction_id}:raw_parent"))
        entries.append(_ensure_dir(converted_parent, f"external_prediction:{prediction_id}:converted_parent"))
        entries.append(
            _ensure_readme(
                raw_parent / "README.md",
                title="External Turn Predictions",
                body=(
                    "Place raw external prediction exports here, then normalize them with "
                    "`stable-asr convert-predictions`.\n"
                ),
            )
        )

    entries.append(_ensure_dir(_resolve(str(config["asr_command_config"]), root=root).parent, "asr_command_config_parent"))
    for name, path in config.get("nanoturn", {}).items():
        if name != "model":
            entries.append(_ensure_dir(_resolve(str(path), root=root).parent, f"nanoturn:{name}:parent"))
    for name, path in config.get("artifacts", {}).items():
        target = _resolve(str(path), root=root)
        parent = target if target.suffix == "" else target.parent
        entries.append(_ensure_dir(parent, f"artifact:{name}:parent"))

    return FinalRunScaffoldReport(output_dir=str(output_dir), entries=entries)


def _corpus_rows(config: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for corpus in config["public_corpora"]:
        rows.append(
            {
                "id": corpus["id"],
                "language": corpus["language"],
                "metadata": corpus["metadata"],
                "manifest": corpus["manifest"],
                "sample_rate": corpus["sample_rate"],
            }
        )
    return rows


def _prediction_rows(config: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for prediction in config.get("external_turn_predictions", []):
        rows.append(
            {
                "id": prediction["id"],
                "schema": prediction["schema"],
                "raw": prediction["raw"],
                "converted": prediction["converted"],
            }
        )
    return rows


def _input_check(name: str, path: str, *, root: Path, kind: str = "input") -> FinalRunPathCheck:
    resolved = _resolve(path, root=root)
    exists = resolved.exists()
    return FinalRunPathCheck(
        name=name,
        path=str(path),
        kind=kind,
        required=True,
        exists=exists,
        ok=exists,
        detail="exists" if exists else "missing required input",
    )


def _planned_check(name: str, path: str, *, root: Path, kind: str) -> FinalRunPathCheck:
    resolved = _resolve(path, root=root)
    parent = resolved if resolved.suffix == "" else resolved.parent
    return FinalRunPathCheck(
        name=name,
        path=str(path),
        kind=kind,
        required=False,
        exists=resolved.exists(),
        ok=bool(str(path).strip()),
        detail=f"planned output; parent={parent}",
    )


def _resolve(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _ensure_dir(path: Path, kind: str) -> FinalRunScaffoldEntry:
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    return FinalRunScaffoldEntry(
        path=str(path),
        kind=kind,
        created=not existed,
        detail="directory",
    )


def _ensure_readme(path: Path, *, title: str, body: str) -> FinalRunScaffoldEntry:
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not existed:
        path.write_text(f"# {title}\n\n{body}", encoding="utf-8")
    return FinalRunScaffoldEntry(
        path=str(path),
        kind="readme",
        created=not existed,
        detail="placeholder instructions; no data generated",
    )
