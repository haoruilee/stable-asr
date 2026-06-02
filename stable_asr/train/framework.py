"""Reusable NanoTurn training framework.

The structure mirrors the stable-worldmodel training scripts in a lightweight
form: config-driven run setup, a data module, DataLoader-based training,
epoch checkpoints, resume support, validation metrics, and reproducible run
artifacts.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stable_asr.data.formats.jsonl import write_jsonl
from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.train.features import feature_names, normalize_feature_source, records_to_features
from stable_asr.turn.nanoturn import DEFAULT_LABELS, NanoTurnConfig, build_nanoturn_model, require_torch, torch


@dataclass(frozen=True)
class NanoTurnRunConfig:
    model_type: str = "nanoturn_pico"
    epochs: int = 100
    lr: float = 1e-2
    seed: int = 0
    feature_source: str = "metadata"
    batch_size: int = 128
    validation_split: float = 0.0
    optimizer: str = "adam"
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    checkpoint_interval: int = 1
    device: str = "auto"
    feature_cache: str | None = None
    feature_cache_format: str | None = None
    feature_cache_mode: str = "auto"
    audio_root: str | None = None
    resume_from: str | None = None
    validation_group_by: str | None = "auto"

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must be in [0.0, 1.0)")
        if self.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError("optimizer must be one of: adam, adamw, sgd")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be at least 1")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when set")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NanoTurnRunArtifacts:
    run_dir: str
    config_path: str
    checkpoint_path: str
    best_checkpoint_path: str
    metrics_path: str
    history_path: str
    summary_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NanoTurnFitResult:
    artifacts: NanoTurnRunArtifacts
    metrics: dict[str, Any]


class NanoTurnTensorDataset:
    def __init__(self, features, targets) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


class NanoTurnDataModule:
    def __init__(
        self,
        train_records: list[TurnManifestRecord],
        *,
        val_records: list[TurnManifestRecord] | None,
        labels: tuple[str, ...],
        config: NanoTurnRunConfig,
    ) -> None:
        self.train_records = train_records
        self.val_records = val_records or []
        self.labels = labels
        self.config = config
        self.train_dataset: NanoTurnTensorDataset | None = None
        self.val_dataset: NanoTurnTensorDataset | None = None

    def setup(self) -> None:
        self.train_dataset = self._make_dataset(
            self.train_records,
            feature_cache=self.config.feature_cache,
            feature_cache_format=self.config.feature_cache_format,
            feature_cache_mode=self.config.feature_cache_mode,
        )
        if self.val_records:
            self.val_dataset = self._make_dataset(
                self.val_records,
                feature_cache=None,
                feature_cache_format=None,
                feature_cache_mode="off",
            )

    def train_dataloader(self):
        require_torch()
        if self.train_dataset is None:
            raise RuntimeError("NanoTurnDataModule.setup() must be called before train_dataloader()")
        generator = torch.Generator().manual_seed(self.config.seed)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )

    def val_dataloader(self):
        require_torch()
        if self.val_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    def _make_dataset(
        self,
        records: list[TurnManifestRecord],
        *,
        feature_cache: str | None,
        feature_cache_format: str | None,
        feature_cache_mode: str,
    ) -> NanoTurnTensorDataset:
        if not records:
            raise ValueError("records must not be empty")
        features = records_to_features(
            records,
            feature_source=self.config.feature_source,
            audio_root=self.config.audio_root,
            feature_cache=feature_cache,
            feature_cache_format=feature_cache_format,
            feature_cache_mode=feature_cache_mode,
        )
        targets = torch.tensor([self.labels.index(record.turn_label) for record in records], dtype=torch.long)
        return NanoTurnTensorDataset(features, targets)


class NanoTurnTrainer:
    def __init__(self, config: NanoTurnRunConfig, *, labels: tuple[str, ...] = DEFAULT_LABELS) -> None:
        require_torch()
        config.validate()
        self.config = config
        self.labels = labels
        self.device = _resolve_device(config.device)
        names = feature_names(config.feature_source)
        self.feature_names = tuple(names)
        self.model = build_nanoturn_model(config.model_type, labels=labels, input_dim=len(names)).to(self.device)
        self.model.config = NanoTurnConfig(
            input_dim=len(names),
            hidden_dim=self.model.config.hidden_dim,
            depth=self.model.config.depth,
            dropout=self.model.config.dropout,
            labels=self.model.config.labels,
            model_type=self.model.config.model_type,
            feature_source=normalize_feature_source(config.feature_source),
        )
        self.optimizer = _build_optimizer(config, self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def fit(
        self,
        data: NanoTurnDataModule,
        *,
        output_dir: str | Path,
    ) -> NanoTurnFitResult:
        output_dir = Path(output_dir)
        artifacts = _prepare_artifacts(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        _write_json(artifacts.config_path, self._run_config_payload(data))

        data.setup()
        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()

        start_epoch = 1
        history: list[dict[str, float]] = []
        best_score = float("-inf")
        best_epoch = 0
        if self.config.resume_from:
            start_epoch, history, best_score, best_epoch = self._load_resume(self.config.resume_from)

        started_at = time.time()
        for epoch in range(start_epoch, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False) if val_loader is not None else {}
            row = {"epoch": float(epoch), **_prefixed("train", train_metrics), **_prefixed("val", val_metrics)}
            history.append(row)
            write_jsonl(artifacts.history_path, history)
            score = float(val_metrics.get("accuracy", train_metrics["accuracy"]))
            is_best = score >= best_score
            if is_best:
                best_score = score
                best_epoch = epoch
            if epoch % self.config.checkpoint_interval == 0 or epoch == self.config.epochs:
                epoch_path = output_dir / "checkpoints" / f"weights_epoch_{epoch}.pt"
                self._save_checkpoint(
                    epoch_path,
                    epoch=epoch,
                    history=history,
                    best_score=best_score,
                    best_epoch=best_epoch,
                )
            if is_best:
                self._save_checkpoint(
                    artifacts.best_checkpoint_path,
                    epoch=epoch,
                    history=history,
                    best_score=best_score,
                    best_epoch=best_epoch,
                )

        self._save_checkpoint(
            artifacts.checkpoint_path,
            epoch=self.config.epochs,
            history=history,
            best_score=best_score,
            best_epoch=best_epoch,
        )
        final = history[-1]
        metrics = {
            "model_type": self.config.model_type,
            "records": len(data.train_records) + len(data.val_records),
            "train_records": len(data.train_records),
            "val_records": len(data.val_records),
            "epochs": self.config.epochs,
            "lr": self.config.lr,
            "seed": self.config.seed,
            "feature_source": normalize_feature_source(self.config.feature_source),
            "feature_cache": self.config.feature_cache,
            "feature_cache_format": self.config.feature_cache_format,
            "feature_cache_mode": self.config.feature_cache_mode if self.config.feature_cache else None,
            "validation_group_by": self.config.validation_group_by,
            "feature_names": list(self.feature_names),
            "labels": list(self.labels),
            "batch_size": self.config.batch_size,
            "optimizer": self.config.optimizer,
            "weight_decay": self.config.weight_decay,
            "gradient_clip_norm": self.config.gradient_clip_norm,
            "device": str(self.device),
            "best_epoch": best_epoch,
            "best_accuracy": best_score,
            "final_loss": final["train_loss"],
            "final_accuracy": final["train_accuracy"],
            "final_train_loss": final["train_loss"],
            "final_train_accuracy": final["train_accuracy"],
            "final_val_loss": final.get("val_loss"),
            "final_val_accuracy": final.get("val_accuracy"),
            "wall_seconds": time.time() - started_at,
            "artifacts": artifacts.to_dict(),
            "history": history,
        }
        _write_json(artifacts.metrics_path, metrics)
        Path(artifacts.summary_path).write_text(_summary_markdown(metrics), encoding="utf-8")
        return NanoTurnFitResult(artifacts=artifacts, metrics=metrics)

    def _run_epoch(self, loader, *, train: bool) -> dict[str, float]:
        if loader is None:
            return {}
        self.model.train(mode=train)
        total_loss = 0.0
        total_correct = 0
        total = 0
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                logits = self.model(features)
                loss = self.criterion(logits, targets)
                if train:
                    loss.backward()
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self.optimizer.step()
            predictions = logits.argmax(dim=-1)
            batch = int(targets.shape[0])
            total += batch
            total_loss += float(loss.item()) * batch
            total_correct += int((predictions == targets).sum().item())
        if total == 0:
            raise RuntimeError("empty dataloader")
        return {"loss": total_loss / total, "accuracy": total_correct / total}

    def _save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        history: list[dict[str, float]],
        best_score: float,
        best_epoch: int,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "history": history,
        }
        torch.save(
            {
                "config": self.model.config.to_dict(),
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "run_config": self.config.to_dict(),
            },
            path,
        )

    def _load_resume(self, path: str | Path) -> tuple[int, list[dict[str, float]], float, int]:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        epoch = int(payload.get("epoch", 0))
        metrics = payload.get("metrics", {})
        history = [dict(row) for row in metrics.get("history", [])] if isinstance(metrics, dict) else []
        best_score = float(metrics.get("best_score", float("-inf"))) if isinstance(metrics, dict) else float("-inf")
        best_epoch = int(metrics.get("best_epoch", 0)) if isinstance(metrics, dict) else 0
        return epoch + 1, history, best_score, best_epoch

    def _run_config_payload(self, data: NanoTurnDataModule) -> dict[str, object]:
        return {
            "framework": "stable_asr.nanoturn_trainer.v1",
            "config": self.config.to_dict(),
            "model": self.model.config.to_dict(),
            "data": {
                "records": len(data.train_records) + len(data.val_records),
                "train_records": len(data.train_records),
                "val_records": len(data.val_records),
            },
        }


def fit_nanoturn(
    train_records: list[TurnManifestRecord],
    *,
    output_dir: str | Path,
    config: NanoTurnRunConfig,
    val_records: list[TurnManifestRecord] | None = None,
    labels: tuple[str, ...] = DEFAULT_LABELS,
) -> NanoTurnFitResult:
    require_torch()
    if not train_records:
        raise ValueError("records must not be empty")
    if val_records is None:
        train_records, split_val_records = _split_validation(train_records, config=config)
        val_records = split_val_records
    else:
        train_records = list(train_records)
        val_records = list(val_records)
    trainer = NanoTurnTrainer(config, labels=labels)
    data = NanoTurnDataModule(train_records, val_records=val_records, labels=labels, config=config)
    return trainer.fit(data, output_dir=output_dir)


def _split_validation(
    records: list[TurnManifestRecord],
    *,
    config: NanoTurnRunConfig,
) -> tuple[list[TurnManifestRecord], list[TurnManifestRecord]]:
    if config.validation_split <= 0.0:
        return list(records), []
    if len(records) < 2:
        raise ValueError("validation_split requires at least two records")
    units = _validation_units(records, group_by=config.validation_group_by)
    if len(units) < 2:
        raise ValueError("validation_split requires at least two validation groups")
    rng = random.Random(config.seed)
    rng.shuffle(units)
    target_val_records = max(1, int(round(len(records) * config.validation_split)))
    val_units: list[list[TurnManifestRecord]] = []
    val_count = 0
    for unit in units[:-1]:
        if val_units and val_count >= target_val_records:
            break
        val_units.append(unit)
        val_count += len(unit)
    val_unit_ids = {id(unit) for unit in val_units}
    train = [record for unit in units if id(unit) not in val_unit_ids for record in unit]
    val = [record for unit in val_units for record in unit]
    if not train:
        raise ValueError("validation_split leaves no training records")
    return train, val


def _validation_units(
    records: list[TurnManifestRecord],
    *,
    group_by: str | None,
) -> list[list[TurnManifestRecord]]:
    if not group_by:
        return [[record] for record in records]
    if group_by == "auto":
        group_by = _auto_validation_group_field(records)
    grouped: dict[str, list[TurnManifestRecord]] = {}
    for record in records:
        key = _validation_field_value(record, group_by)
        grouped.setdefault(key, []).append(record)
    return list(grouped.values())


def _auto_validation_group_field(records: list[TurnManifestRecord]) -> str:
    for field in ("metadata.asr_record_id", "metadata.conversation_id", "audio"):
        values = [_validation_field_value(record, field) for record in records]
        if any(value for value in values) and len(set(values)) < len(records):
            return field
    return "id"


def _validation_field_value(record: TurnManifestRecord, field: str) -> str:
    if field.startswith("metadata."):
        value = record.metadata.get(field.removeprefix("metadata."))
    else:
        value = getattr(record, field, None)
    if value is None:
        return record.id
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return repr(value)


def _build_optimizer(config: NanoTurnRunConfig, parameters):
    if config.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"unknown optimizer: {config.optimizer}")


def _resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(device)


def _prepare_artifacts(output_dir: Path) -> NanoTurnRunArtifacts:
    return NanoTurnRunArtifacts(
        run_dir=str(output_dir),
        config_path=str(output_dir / "run_config.json"),
        checkpoint_path=str(output_dir / "checkpoint.pt"),
        best_checkpoint_path=str(output_dir / "best.pt"),
        metrics_path=str(output_dir / "metrics.json"),
        history_path=str(output_dir / "history.jsonl"),
        summary_path=str(output_dir / "TRAINING_SUMMARY.md"),
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def _summary_markdown(metrics: dict[str, Any]) -> str:
    rows = [
        ("model_type", metrics["model_type"]),
        ("train_records", metrics["train_records"]),
        ("val_records", metrics["val_records"]),
        ("epochs", metrics["epochs"]),
        ("batch_size", metrics["batch_size"]),
        ("optimizer", metrics["optimizer"]),
        ("feature_source", metrics["feature_source"]),
        ("final_train_accuracy", metrics["final_train_accuracy"]),
        ("final_val_accuracy", metrics.get("final_val_accuracy")),
        ("best_epoch", metrics["best_epoch"]),
        ("best_accuracy", metrics["best_accuracy"]),
    ]
    lines = ["# NanoTurn Training Summary", "", "| key | value |", "| --- | --- |"]
    lines.extend(f"| {key} | `{value}` |" for key, value in rows)
    return "\n".join(lines) + "\n"


def copy_best_to_checkpoint(best_path: str | Path, checkpoint_path: str | Path) -> None:
    """Utility for callers that want the best checkpoint as the final checkpoint."""

    shutil.copy2(best_path, checkpoint_path)
