"""Final-scale input collection registry and readiness report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_asr.eval.report import dict_table
from stable_asr.paper.final_config import load_final_run_config, validate_final_run_config
from stable_asr.resources import resolve_platform_path


DEFAULT_FINAL_INPUT_COLLECTIONS_PATH = Path("configs/final/input_collections.json")
DEFAULT_FINAL_INPUT_COLLECTIONS: dict[str, Any] = {
    "id": "stable_asr_final_input_collections_v0",
    "version": "0.1.0",
    "title": "Stable-ASR Final Input Collection Plan",
    "description": (
        "Operational collection plan for the real corpora, VoiceWorld records, "
        "external turn predictions, ASR adapter outputs, NanoTurn artifacts, and "
        "final paper bundle inputs required by the Stable-ASR platform paper."
    ),
    "collections": [
        {
            "id": "librispeech_dev_clean",
            "title": "LibriSpeech dev-clean",
            "category": "public_asr_corpus",
            "priority": "p0",
            "required": True,
            "license": "see_upstream",
            "source_urls": ["https://www.openslr.org/12"],
            "required_paths": ["data/librispeech/LibriSpeech/dev-clean"],
            "generated_paths": ["runs/final/librispeech_dev_clean/asr_manifest.jsonl"],
            "commands": [
                "stable-asr prepare-public-asr --corpus librispeech --input-dir data/librispeech/LibriSpeech/dev-clean --output runs/final/librispeech_dev_clean/asr_manifest.jsonl"
            ],
            "verification": [
                "stable-asr inspect-asr-manifest runs/final/librispeech_dev_clean/asr_manifest.jsonl"
            ],
            "notes": "Place or symlink the upstream LibriSpeech dev-clean directory before preparing the manifest.",
        },
        {
            "id": "aishell1_dev",
            "title": "AISHELL-1 dev",
            "category": "public_asr_corpus",
            "priority": "p0",
            "required": True,
            "license": "see_upstream",
            "source_urls": ["https://www.openslr.org/33"],
            "required_paths": ["data/aishell1/data_aishell"],
            "generated_paths": ["runs/final/aishell1_dev/asr_manifest.jsonl"],
            "commands": [
                "stable-asr prepare-public-asr --corpus aishell1 --input-dir data/aishell1/data_aishell --split dev --output runs/final/aishell1_dev/asr_manifest.jsonl"
            ],
            "verification": [
                "stable-asr inspect-asr-manifest runs/final/aishell1_dev/asr_manifest.jsonl"
            ],
            "notes": "Required Mandarin public ASR input for bilingual final evidence.",
        },
        {
            "id": "optional_public_corpora",
            "title": "WenetSpeech and Common Voice optional corpora",
            "category": "public_asr_corpus",
            "priority": "p1",
            "required": False,
            "license": "see_upstream",
            "source_urls": ["https://wenet.org.cn/WenetSpeech/", "https://commonvoice.mozilla.org/datasets"],
            "required_paths": ["data/wenetspeech/WenetSpeech", "data/common_voice/en"],
            "generated_paths": [
                "runs/final/wenetspeech_dev/asr_manifest.jsonl",
                "runs/final/common_voice_en_dev/asr_manifest.jsonl",
            ],
            "commands": [
                "stable-asr prepare-public-asr --corpus wenetspeech --input-dir data/wenetspeech/WenetSpeech --split dev --output runs/final/wenetspeech_dev/asr_manifest.jsonl",
                "stable-asr prepare-public-asr --corpus common_voice --input-dir data/common_voice/en --split dev --output runs/final/common_voice_en_dev/asr_manifest.jsonl",
            ],
            "verification": [
                "stable-asr inspect-asr-manifest runs/final/wenetspeech_dev/asr_manifest.jsonl",
                "stable-asr inspect-asr-manifest runs/final/common_voice_en_dev/asr_manifest.jsonl",
            ],
            "notes": "Optional robustness corpora; absence should not block v0 final readiness unless the final config marks them required.",
        },
        {
            "id": "turn_splits",
            "title": "Leakage-audited turn train/dev/test splits",
            "category": "derived_turn_data",
            "priority": "p0",
            "required": True,
            "license": "derived_from_upstream_inputs",
            "source_urls": [],
            "required_paths": [
                "runs/final/turn_train.jsonl",
                "runs/final/turn_dev.jsonl",
                "runs/final/turn_test.jsonl",
            ],
            "generated_paths": ["runs/final/asr_eval_manifest.jsonl"],
            "commands": [
                "stable-asr final-config --config configs/final/paper_final.json --prepare-asr-eval-manifest",
                "stable-asr final-config --config configs/final/paper_final.json --bootstrap-turn-splits",
            ],
            "verification": [
                "stable-asr audit-turn-splits --train runs/final/turn_train.jsonl --dev runs/final/turn_dev.jsonl --test runs/final/turn_test.jsonl"
            ],
            "notes": "Derived split files are required final inputs for NanoTurn training and external baseline comparison.",
        },
        {
            "id": "voiceworld_real",
            "title": "Real VoiceWorld scenario records",
            "category": "full_duplex_scenarios",
            "priority": "p0",
            "required": True,
            "license": "project_or_recording_consent",
            "source_urls": [],
            "required_paths": [
                "data/voiceworld/metadata.tsv",
                "data/voiceworld/audio",
                "runs/final/voiceworld_real.jsonl",
            ],
            "generated_paths": ["runs/final/reports/scenarios.json"],
            "commands": [
                "stable-asr final-config --config configs/final/paper_final.json --prepare-voiceworld-real",
                "stable-asr eval-scenario --dataset runs/final/voiceworld_real.jsonl --checkpoint runs/final/nanoturn/checkpoint.pt --json-output runs/final/reports/scenarios.json",
            ],
            "verification": [
                "stable-asr validate-manifest runs/final/voiceworld_real.jsonl",
                "stable-asr final-config --config configs/final/paper_final.json --audit-voiceworld-real --scenario-suite configs/scenarios/stable_asr_voiceworld_v0.json --min-scenario-records 20",
            ],
            "notes": "Do not use synthetic smoke examples as final VoiceWorld evidence.",
        },
        {
            "id": "external_turn_predictions",
            "title": "SmartTurn, EasyTurn, and VAP raw prediction exports",
            "category": "external_turn_baselines",
            "priority": "p0",
            "required": True,
            "license": "see_upstream",
            "source_urls": [
                "https://github.com/pipecat-ai/smart-turn",
                "https://huggingface.co/ASLP-lab/Easy-Turn",
                "https://github.com/ErikEkstedt/VoiceActivityProjection",
            ],
            "required_paths": [
                "runs/final/external/smartturn_raw.jsonl",
                "runs/final/external/easyturn_raw.jsonl",
                "runs/final/external/vap_raw.jsonl",
            ],
            "generated_paths": [
                "runs/final/external/smartturn_predictions.jsonl",
                "runs/final/external/easyturn_predictions.jsonl",
                "runs/final/external/vap_predictions.jsonl",
                "runs/final/reports/baselines.json",
            ],
            "commands": [
                "stable-asr final-config --config configs/final/paper_final.json --prepare-external-predictions --require-all-predictions",
                "stable-asr compare-turn --dataset runs/final/turn_test.jsonl --baseline rule_endpoint --baseline vad_pause --baseline text_turn --predictions smart_turn=runs/final/external/smartturn_predictions.jsonl --predictions easy_turn=runs/final/external/easyturn_predictions.jsonl --predictions vap=runs/final/external/vap_predictions.jsonl --checkpoint nanoturn=runs/final/nanoturn/checkpoint.pt --json-output runs/final/reports/baselines.json",
            ],
            "verification": [
                "stable-asr validate-turn-predictions --dataset runs/final/turn_test.jsonl --predictions runs/final/external/smartturn_predictions.jsonl",
                "stable-asr validate-turn-predictions --dataset runs/final/turn_test.jsonl --predictions runs/final/external/easyturn_predictions.jsonl",
                "stable-asr validate-turn-predictions --dataset runs/final/turn_test.jsonl --predictions runs/final/external/vap_predictions.jsonl",
            ],
            "notes": "External systems should be run outside Stable-ASR, then normalized and coverage-checked here.",
        },
        {
            "id": "command_backed_asr_outputs",
            "title": "Command-backed streaming ASR outputs",
            "category": "external_asr_systems",
            "priority": "p0",
            "required": True,
            "license": "depends_on_external_system",
            "source_urls": [
                "https://github.com/openai/whisper",
                "https://github.com/modelscope/FunASR",
                "https://github.com/QwenLM/Qwen3-ASR",
                "https://github.com/FireRedTeam/FireRedASR2S",
            ],
            "required_paths": [
                "configs/final/asr_command_compare.json",
                "runs/final/asr_commands/raw/whisper_raw.jsonl",
                "runs/final/asr_commands/raw/funasr_raw.jsonl",
                "runs/final/asr_commands/raw/qwen3_asr_raw.jsonl",
                "runs/final/asr_commands/raw/firered_asr2s_raw.jsonl",
            ],
            "generated_paths": [
                "runs/final/asr_commands/whisper_streaming.jsonl",
                "runs/final/asr_commands/funasr_streaming.jsonl",
                "runs/final/asr_commands/qwen3_asr_streaming.jsonl",
                "runs/final/asr_commands/firered_asr2s_streaming.jsonl",
                "runs/final/reports/asr_command_compare.json",
                "runs/final/reports/whisper_sweep.json",
                "runs/final/reports/asr_transcript_conversions.json",
            ],
            "commands": [
                "stable-asr final-config --config configs/final/paper_final.json --audit-asr-commands",
                "stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --json-output runs/final/reports/asr_command_compare.json",
                "stable-asr sweep-streaming-asr --input runs/final/asr_commands/whisper_streaming.jsonl --chunks-ms 160 320 640 --lookahead-ms 0 160 320 --json-output runs/final/reports/whisper_sweep.json",
                "stable-asr final-config --config configs/final/paper_final.json --prepare-asr-transcript-conversions",
            ],
            "verification": [
                "stable-asr compare-asr-commands --config configs/final/asr_command_compare.json --validate-only --require-input-manifest --min-adapters 4"
            ],
            "notes": "Stable-ASR evaluates normalized outputs and command contracts; it does not vendor upstream ASR stacks.",
        },
        {
            "id": "nanoturn_final_artifacts",
            "title": "NanoTurn final checkpoint, metrics, and export",
            "category": "trainable_turn_model",
            "priority": "p0",
            "required": True,
            "license": "project_license",
            "source_urls": [],
            "required_paths": ["runs/final/nanoturn/checkpoint.pt", "runs/final/nanoturn/metrics.json"],
            "generated_paths": ["runs/final/nanoturn/nanoturn.onnx", "runs/final/MODEL_CARD.md"],
            "commands": [
                "stable-asr train-turn --dataset runs/final/turn_train.jsonl --output-dir runs/final/nanoturn --model nanoturn_pico --feature-source audio",
                "stable-asr export-turn-onnx --checkpoint runs/final/nanoturn/checkpoint.pt --output runs/final/nanoturn/nanoturn.onnx",
                "stable-asr make-card model --input configs/models/stable_asr_models.json --model-id nanoturn_pico --metrics runs/final/nanoturn/metrics.json --output runs/final/MODEL_CARD.md",
            ],
            "verification": [
                "stable-asr benchmark-turn --dataset runs/final/turn_test.jsonl --checkpoint runs/final/nanoturn/checkpoint.pt --json-output runs/final/reports/turn_benchmarks.json"
            ],
            "notes": "Final NanoTurn metrics require real splits and must not reuse smoke fixtures.",
        },
        {
            "id": "final_paper_bundle",
            "title": "Final paper result bundle and release gates",
            "category": "paper_artifacts",
            "priority": "p0",
            "required": True,
            "license": "project_license",
            "source_urls": [],
            "required_paths": ["runs/final/FINAL_INPUT_HANDOFF.json", "runs/final/paper_results.json", "runs/final/artifacts"],
            "generated_paths": [
                "runs/final/FINAL_ASSIGNMENT_AUDIT.md",
                "runs/final/FINAL_HANDOFF_AUDIT.md",
                "runs/final/artifacts.tar.gz",
                "runs/final/artifacts.tar.gz.sha256",
            ],
            "commands": [
                "stable-asr final-assignment-audit --input runs/final_acquisition_pack/acquisition/assignments.json --require-owner --require-due-date --require-ready --output runs/final/FINAL_ASSIGNMENT_AUDIT.md",
                "stable-asr final-handoff-template --output runs/final/FINAL_INPUT_HANDOFF.json",
                "stable-asr final-handoff-checksums --input runs/final/FINAL_INPUT_HANDOFF.json --repo-root . --output runs/final/FINAL_INPUT_HANDOFF.json",
                "stable-asr final-handoff-audit --input runs/final/FINAL_INPUT_HANDOFF.json --repo-root . --require-checksums --output runs/final/FINAL_HANDOFF_AUDIT.md",
                "stable-asr final-results --config configs/final/paper_final.json --output runs/final/paper_results.json",
                "stable-asr paper-bundle --results runs/final/paper_results.json --output-dir runs/final/artifacts",
                "stable-asr paper-archive --artifacts-dir runs/final/artifacts --output runs/final/artifacts.tar.gz",
            ],
            "verification": [
                "stable-asr final-assignment-audit --input runs/final_acquisition_pack/acquisition/assignments.json --require-owner --require-due-date --require-ready",
                "stable-asr final-handoff-checksums --input runs/final/FINAL_INPUT_HANDOFF.json --repo-root . --output runs/final/FINAL_INPUT_HANDOFF.json",
                "stable-asr final-handoff-audit --input runs/final/FINAL_INPUT_HANDOFF.json --repo-root . --require-checksums",
                "stable-asr paper-archive-verify --archive runs/final/artifacts.tar.gz",
                "stable-asr paper-parity-audit --results runs/final/paper_results.json --artifacts-dir runs/final/artifacts --require-final",
                "stable-asr paper-release-audit --repo-root . --results runs/final/paper_results.json --artifacts-dir runs/final/artifacts --model-card runs/final/MODEL_CARD.md --require-final-ready",
            ],
            "notes": "Run only after all real inputs and intermediate reports exist.",
        },
    ],
}


@dataclass(frozen=True)
class FinalInputCollectionValidation:
    ok: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": self.errors}

    def to_text(self) -> str:
        if self.ok:
            return "final_input_collections: OK"
        return "final_input_collections: FAILED\n" + "\n".join(f"- {error}" for error in self.errors)


@dataclass(frozen=True)
class FinalInputCollectionStatus:
    id: str
    title: str
    category: str
    priority: str
    required: bool
    ready: bool
    required_present: list[str]
    required_missing: list[str]
    generated_present: list[str]
    generated_missing: list[str]
    commands: list[str]
    verification: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "priority": self.priority,
            "required": self.required,
            "ready": self.ready,
            "required_present": self.required_present,
            "required_missing": self.required_missing,
            "generated_present": self.generated_present,
            "generated_missing": self.generated_missing,
            "commands": self.commands,
            "verification": self.verification,
        }


@dataclass(frozen=True)
class FinalInputCollectionReport:
    ok: bool
    registry_id: str
    config_id: str
    repo_root: str
    missing_required: list[str]
    collections: list[FinalInputCollectionStatus]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "registry_id": self.registry_id,
            "config_id": self.config_id,
            "repo_root": self.repo_root,
            "missing_required": self.missing_required,
            "collections": [collection.to_dict() for collection in self.collections],
        }

    def to_text(self) -> str:
        lines = [
            f"final_input_collections: {'READY' if self.ok else 'NOT_READY'}",
            f"registry: {self.registry_id}",
            f"config: {self.config_id}",
            f"missing_required: {len(self.missing_required)}",
        ]
        for collection in self.collections:
            status = "READY" if collection.ready else "MISSING"
            required = "required" if collection.required else "optional"
            lines.append(
                f"- {status} {collection.id} ({required}, {collection.priority}): "
                f"{len(collection.required_missing)} missing required path(s)"
            )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        rows = [
            {
                "id": collection.id,
                "priority": collection.priority,
                "required": collection.required,
                "status": "READY" if collection.ready else "MISSING",
                "missing_required": len(collection.required_missing),
                "generated_missing": len(collection.generated_missing),
                "category": collection.category,
            }
            for collection in self.collections
        ]
        lines = [
            "# Stable-ASR Final Input Collections",
            "",
            f"- status: `{'READY' if self.ok else 'NOT_READY'}`",
            f"- registry: `{self.registry_id}`",
            f"- config: `{self.config_id}`",
            f"- repo_root: `{self.repo_root}`",
            f"- missing_required: `{len(self.missing_required)}`",
            "",
            "## Summary",
            "",
            dict_table(rows),
        ]
        if self.missing_required:
            lines.extend(["", "## Missing Required Paths", ""])
            lines.extend(f"- `{path}`" for path in self.missing_required)
        for collection in self.collections:
            lines.extend(
                [
                    "",
                    f"## {collection.title}",
                    "",
                    f"- id: `{collection.id}`",
                    f"- category: `{collection.category}`",
                    f"- priority: `{collection.priority}`",
                    f"- required: `{collection.required}`",
                    "",
                    "Required paths:",
                    "",
                ]
            )
            if collection.required_present:
                lines.extend(f"- present: `{path}`" for path in collection.required_present)
            if collection.required_missing:
                lines.extend(f"- missing: `{path}`" for path in collection.required_missing)
            if not collection.required_present and not collection.required_missing:
                lines.append("- none")
            lines.extend(["", "Generated paths:", ""])
            if collection.generated_present:
                lines.extend(f"- present: `{path}`" for path in collection.generated_present)
            if collection.generated_missing:
                lines.extend(f"- pending: `{path}`" for path in collection.generated_missing)
            if not collection.generated_present and not collection.generated_missing:
                lines.append("- none")
            lines.extend(["", "Commands:", ""])
            lines.extend(f"```bash\n{command}\n```" for command in collection.commands)
            lines.extend(["", "Verification:", ""])
            lines.extend(f"```bash\n{command}\n```" for command in collection.verification)
        lines.append("")
        return "\n".join(lines)


def load_final_input_collections(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        registry_path = resolve_platform_path(DEFAULT_FINAL_INPUT_COLLECTIONS_PATH)
        if registry_path.exists():
            with registry_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("final input collection registry must be a JSON object")
            return payload
        return json.loads(json.dumps(DEFAULT_FINAL_INPUT_COLLECTIONS))
    with resolve_platform_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("final input collection registry must be a JSON object")
    return payload


def write_final_input_collections_json(path: str | Path, registry: dict[str, Any] | None = None) -> str:
    registry = registry or load_final_input_collections()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def validate_final_input_collections(registry: dict[str, Any]) -> FinalInputCollectionValidation:
    errors: list[str] = []
    for key in ("id", "version", "title", "collections"):
        if key not in registry:
            errors.append(f"missing top-level key: {key}")
    collections = registry.get("collections")
    if not isinstance(collections, list) or not collections:
        errors.append("collections must be a non-empty list")
        return FinalInputCollectionValidation(ok=False, errors=errors)

    seen: set[str] = set()
    required_keys = {
        "id",
        "title",
        "category",
        "priority",
        "required",
        "license",
        "source_urls",
        "required_paths",
        "generated_paths",
        "commands",
        "verification",
    }
    for index, collection in enumerate(collections):
        if not isinstance(collection, dict):
            errors.append(f"collection {index} must be an object")
            continue
        collection_id = collection.get("id")
        if not isinstance(collection_id, str) or not collection_id:
            errors.append(f"collection {index} missing id")
        elif collection_id in seen:
            errors.append(f"duplicate collection id: {collection_id}")
        else:
            seen.add(collection_id)
        for key in required_keys:
            if key not in collection:
                errors.append(f"collection {collection_id or index} missing {key}")
        if "required" in collection and not isinstance(collection["required"], bool):
            errors.append(f"collection {collection_id or index} required must be boolean")
        for key in ("source_urls", "required_paths", "generated_paths", "commands", "verification"):
            if key in collection and (
                not isinstance(collection[key], list)
                or not all(isinstance(item, str) and item for item in collection[key])
            ):
                errors.append(f"collection {collection_id or index} {key} must be a string list")
    return FinalInputCollectionValidation(ok=not errors, errors=errors)


def final_input_collection_report(
    registry: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
    repo_root: str | Path = ".",
) -> FinalInputCollectionReport:
    registry = registry or load_final_input_collections()
    config = config or load_final_run_config()
    validation = validate_final_input_collections(registry)
    if not validation.ok:
        raise ValueError("; ".join(validation.errors))
    config_validation = validate_final_run_config(config)
    if not config_validation.ok:
        raise ValueError("; ".join(config_validation.errors))

    root = Path(repo_root)
    statuses = [_collection_status(collection, root=root) for collection in registry["collections"]]
    missing_required = [
        path
        for status in statuses
        if status.required
        for path in status.required_missing
    ]
    return FinalInputCollectionReport(
        ok=not missing_required,
        registry_id=str(registry["id"]),
        config_id=str(config["id"]),
        repo_root=str(root),
        missing_required=missing_required,
        collections=statuses,
    )


def final_input_collections_markdown(
    registry: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
    repo_root: str | Path = ".",
) -> str:
    return final_input_collection_report(registry, config=config, repo_root=repo_root).to_markdown()


def _collection_status(collection: dict[str, Any], *, root: Path) -> FinalInputCollectionStatus:
    required_present, required_missing = _partition_paths(collection.get("required_paths", []), root=root)
    generated_present, generated_missing = _partition_paths(collection.get("generated_paths", []), root=root)
    return FinalInputCollectionStatus(
        id=str(collection["id"]),
        title=str(collection["title"]),
        category=str(collection["category"]),
        priority=str(collection["priority"]),
        required=bool(collection["required"]),
        ready=not required_missing,
        required_present=required_present,
        required_missing=required_missing,
        generated_present=generated_present,
        generated_missing=generated_missing,
        commands=list(collection.get("commands", [])),
        verification=list(collection.get("verification", [])),
    )


def _partition_paths(paths: list[str], *, root: Path) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for path in paths:
        target = Path(path)
        candidate = target if target.is_absolute() else root / target
        if candidate.exists():
            present.append(path)
        else:
            missing.append(path)
    return present, missing
