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

_SEQUENCE_MODEL_TYPES = frozenset({"nanoturn_micro"})


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
    tensorboard_log_dir: str | None = None
    # ── acceleration flags ──────────────────────────────────────────────────
    # Mixed-precision training (torch.autocast + GradScaler). Safe for all
    # NanoTurn variants. Falls back to FP32 silently on CPU.
    amp: bool = False
    # DataLoader worker processes. 0 = main-process loading (original behaviour).
    # Recommended: 2-4 for audio feature sources, 0 for cached/metadata.
    num_workers: int = 0
    # Pin DataLoader output tensors to page-locked memory for faster GPU transfer.
    # Only effective when num_workers > 0 and device is CUDA.
    pin_memory: bool = False
    # Cosine annealing LR schedule. None = constant LR (original behaviour).
    # "cosine" decays LR from lr to lr_min over epochs.
    lr_schedule: str | None = None
    # Minimum LR for cosine schedule (absolute value, not relative).
    lr_min: float = 1e-6
    # Early stopping: halt if val accuracy does not improve for this many epochs.
    # None = disabled (original behaviour).
    early_stopping_patience: int | None = None

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
        if self.lr_schedule is not None and self.lr_schedule not in {"cosine"}:
            raise ValueError("lr_schedule must be 'cosine' or None")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1 when set")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")

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
        # features is either a 2D tensor (MLP) or a list of (T, n_mels) tensors (TCN)
        self.is_sequence = isinstance(features, list)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


def _sequence_collate_fn(batch):
    """Pad variable-length (T, n_mels) sequences to the max T in the batch."""
    import torch as _torch
    seqs, targets = zip(*batch)
    max_t = max(s.shape[0] for s in seqs)
    n_mels = seqs[0].shape[1]
    padded = _torch.zeros(len(seqs), max_t, n_mels, dtype=_torch.float32)
    for i, seq in enumerate(seqs):
        padded[i, : seq.shape[0], :] = seq
    return padded, _torch.stack(targets)


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
            # Val set uses the same cache in read-only mode when a cache exists.
            # This avoids decoding audio twice (train cache was built above).
            val_cache_mode = "read" if self.config.feature_cache else "off"
            self.val_dataset = self._make_dataset(
                self.val_records,
                feature_cache=self.config.feature_cache,
                feature_cache_format=self.config.feature_cache_format,
                feature_cache_mode=val_cache_mode,
            )

    def train_dataloader(self, *, epoch: int | None = None):
        require_torch()
        if self.train_dataset is None:
            raise RuntimeError("NanoTurnDataModule.setup() must be called before train_dataloader()")
        generator = torch.Generator().manual_seed(_epoch_seed(self.config.seed, epoch))
        collate_fn = _sequence_collate_fn if self.train_dataset.is_sequence else None
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.config.num_workers > 0,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self):
        require_torch()
        if self.val_dataset is None:
            return None
        collate_fn = _sequence_collate_fn if self.val_dataset.is_sequence else None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and self.config.num_workers > 0,
            persistent_workers=self.config.num_workers > 0,
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
        self.is_sequence_model = config.model_type in _SEQUENCE_MODEL_TYPES
        if self.is_sequence_model:
            from stable_asr.turn.nanoturn_micro import NanoTurnMicroConfig, build_nanoturn_micro
            micro_cfg = NanoTurnMicroConfig(
                n_mels=len(names),
                hidden_dim=64,
                n_blocks=4,
                kernel_size=3,
                dropout=0.1,
                labels=labels,
                model_type="nanoturn_micro",
                feature_source=normalize_feature_source(config.feature_source),
            )
            self.model = build_nanoturn_micro(
                labels=labels,
                n_mels=micro_cfg.n_mels,
                hidden_dim=micro_cfg.hidden_dim,
                n_blocks=micro_cfg.n_blocks,
                kernel_size=micro_cfg.kernel_size,
                dropout=micro_cfg.dropout,
            ).to(self.device)
            # Store micro config for checkpoint serialisation
            self._micro_config = micro_cfg
        else:
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
            self._micro_config = None
        self.optimizer = _build_optimizer(config, self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = _build_scheduler(config, self.optimizer)
        # AMP: GradScaler is a no-op on CPU (scale=1.0), so it's always safe to create.
        # torch.autocast is only activated when config.amp=True and device is CUDA.
        self._use_amp = config.amp and str(self.device).startswith("cuda")
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

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
        val_loader = data.val_dataloader()

        start_epoch = 1
        history: list[dict[str, float]] = []
        best_score = float("-inf")
        best_epoch = 0
        if self.config.resume_from:
            start_epoch, history, best_score, best_epoch = self._load_resume(self.config.resume_from)

        started_at = time.time()
        tensorboard_log_dir = _resolve_tensorboard_log_dir(self.config.tensorboard_log_dir, output_dir)
        writer = _make_tensorboard_writer(tensorboard_log_dir) if tensorboard_log_dir is not None else None
        patience = self.config.early_stopping_patience
        epochs_no_improve = 0
        stopped_early = False
        try:
            if writer is not None:
                _write_tensorboard_run_config(writer, self._run_config_payload(data))
            for epoch in range(start_epoch, self.config.epochs + 1):
                train_loader = data.train_dataloader(epoch=epoch)
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
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                # Step LR scheduler after each epoch (uses val accuracy for ReduceLROnPlateau,
                # or just epoch count for cosine/step schedules).
                if self.scheduler is not None:
                    self.scheduler.step()
                _write_tensorboard_epoch(writer, epoch=epoch, row=row, best_score=best_score, optimizer=self.optimizer)
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
                # Early stopping
                if patience is not None and epochs_no_improve >= patience:
                    stopped_early = True
                    break
        finally:
            if writer is not None:
                writer.close()

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
            "tensorboard_log_dir": str(tensorboard_log_dir) if tensorboard_log_dir is not None else None,
            "best_epoch": best_epoch,
            "best_accuracy": best_score,
            "final_loss": final["train_loss"],
            "final_accuracy": final["train_accuracy"],
            "final_train_loss": final["train_loss"],
            "final_train_accuracy": final["train_accuracy"],
            "final_val_loss": final.get("val_loss"),
            "final_val_accuracy": final.get("val_accuracy"),
            "wall_seconds": time.time() - started_at,
            "stopped_early": stopped_early,
            "amp": self._use_amp,
            "lr_schedule": self.config.lr_schedule,
            "early_stopping_patience": self.config.early_stopping_patience,
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
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        for features, targets in loader:
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=device_type, enabled=self._use_amp):
                    logits = self.model(features)
                    loss = self.criterion(logits, targets)
                if train:
                    self._scaler.scale(loss).backward()
                    if self.config.gradient_clip_norm is not None:
                        self._scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
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
        if self.is_sequence_model:
            model_config_dict = self._micro_config.to_dict()
        else:
            model_config_dict = self.model.config.to_dict()
        torch.save(
            {
                "config": model_config_dict,
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
        if self.is_sequence_model:
            model_info = self._micro_config.to_dict()
        else:
            model_info = self.model.config.to_dict()
        return {
            "framework": "stable_asr.nanoturn_trainer.v1",
            "config": self.config.to_dict(),
            "model": model_info,
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


def _build_scheduler(config: NanoTurnRunConfig, optimizer):
    """Build an optional LR scheduler.

    "cosine" → CosineAnnealingLR: decays LR from config.lr to config.lr_min
               over config.epochs steps.  Safe — does not change the loss
               landscape, only the step size.  Typically saves 10-30% of
               training epochs to the same validation accuracy.
    None → no scheduler (original constant-LR behaviour).
    """
    if config.lr_schedule is None:
        return None
    if config.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.lr_min,
        )
    raise ValueError(f"unknown lr_schedule: {config.lr_schedule}")


def _epoch_seed(seed: int, epoch: int | None) -> int:
    if epoch is None:
        return int(seed)
    return int(seed) + int(epoch)


def _resolve_tensorboard_log_dir(log_dir: str | None, output_dir: Path) -> Path | None:
    if log_dir is None:
        return None
    text = str(log_dir).strip()
    if not text or text.lower() in {"none", "off", "false"}:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return output_dir / path


def _make_tensorboard_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging requires the tensorboard package. "
            "Install with `pip install -e '.[train]'` or `pip install tensorboard`."
        ) from exc
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _write_tensorboard_run_config(writer, payload: dict[str, object]) -> None:
    writer.add_text("run/config", json.dumps(payload, ensure_ascii=False, indent=2), global_step=0)


def _write_tensorboard_epoch(writer, *, epoch: int, row: dict[str, float], best_score: float, optimizer) -> None:
    if writer is None:
        return
    for key, value in row.items():
        if key == "epoch":
            continue
        writer.add_scalar(key.replace("_", "/"), float(value), epoch)
    writer.add_scalar("best/accuracy", float(best_score), epoch)
    if optimizer.param_groups:
        writer.add_scalar("train/lr", float(optimizer.param_groups[0].get("lr", 0.0)), epoch)


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
