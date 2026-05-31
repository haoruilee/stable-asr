"""Deterministic train/dev/test splitting for turn manifests."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.data.registry import summarize_records


SPLIT_NAMES = ("train", "dev", "test")


@dataclass(frozen=True)
class TurnSplitConfig:
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 0
    stratify_by: tuple[str, ...] = ("turn_label",)
    group_by: str | None = None
    ensure_non_empty: bool = True

    def validate(self) -> None:
        ratios = (self.train_ratio, self.dev_ratio, self.test_ratio)
        if any(ratio < 0 for ratio in ratios):
            raise ValueError("split ratios must be non-negative")
        if sum(ratios) <= 0:
            raise ValueError("at least one split ratio must be positive")
        if self.group_by and self.group_by in self.stratify_by:
            raise ValueError("group_by must not also be used as a stratification field")


@dataclass(frozen=True)
class TurnSplitResult:
    train: list[TurnManifestRecord]
    dev: list[TurnManifestRecord]
    test: list[TurnManifestRecord]
    config: TurnSplitConfig

    def split(self, name: str) -> list[TurnManifestRecord]:
        if name == "train":
            return self.train
        if name == "dev":
            return self.dev
        if name == "test":
            return self.test
        raise ValueError(f"unknown split: {name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "config": {
                "train_ratio": self.config.train_ratio,
                "dev_ratio": self.config.dev_ratio,
                "test_ratio": self.config.test_ratio,
                "seed": self.config.seed,
                "stratify_by": list(self.config.stratify_by),
                "group_by": self.config.group_by,
                "ensure_non_empty": self.config.ensure_non_empty,
            },
            "splits": {
                name: summarize_records(self.split(name))
                for name in SPLIT_NAMES
            },
        }

    def to_text(self) -> str:
        payload = self.to_dict()
        lines = ["turn_split:"]
        split_payload = payload["splits"]
        if isinstance(split_payload, dict):
            for name in SPLIT_NAMES:
                summary = split_payload[name]
                if isinstance(summary, dict):
                    lines.append(f"- {name}: {summary['records']} record(s)")
                    labels = summary.get("turn_labels", {})
                    if isinstance(labels, dict):
                        label_text = ", ".join(f"{key}={value}" for key, value in labels.items())
                        lines.append(f"  turn_labels: {label_text or 'none'}")
        return "\n".join(lines)


def split_turn_records(
    records: Iterable[TurnManifestRecord],
    *,
    config: TurnSplitConfig | None = None,
) -> TurnSplitResult:
    """Split records deterministically, optionally preserving label mix per bucket."""

    config = config or TurnSplitConfig()
    config.validate()
    record_list = list(records)
    rng = random.Random(config.seed)
    split_units: dict[str, list[list[TurnManifestRecord]]] = {name: [] for name in SPLIT_NAMES}
    ratios = (config.train_ratio, config.dev_ratio, config.test_ratio)

    buckets = _build_buckets(record_list, config=config)
    for units in buckets.values():
        local_units = list(units)
        rng.shuffle(local_units)
        counts = _allocate_counts(len(local_units), ratios)
        cursor = 0
        for name, count in zip(SPLIT_NAMES, counts, strict=True):
            split_units[name].extend(local_units[cursor : cursor + count])
            cursor += count

    if config.ensure_non_empty:
        _rebalance_empty_units(split_units, ratios)

    splits: dict[str, list[TurnManifestRecord]] = {
        name: [record for unit in split_units[name] for record in unit]
        for name in SPLIT_NAMES
    }
    for name in SPLIT_NAMES:
        splits[name].sort(key=lambda record: record.id)

    return TurnSplitResult(
        train=splits["train"],
        dev=splits["dev"],
        test=splits["test"],
        config=config,
    )


def _build_buckets(
    records: list[TurnManifestRecord],
    *,
    config: TurnSplitConfig,
) -> dict[tuple[str, ...], list[list[TurnManifestRecord]]]:
    if config.group_by:
        grouped: dict[str, list[TurnManifestRecord]] = {}
        for record in records:
            key = _field_value(record, config.group_by)
            grouped.setdefault(key, []).append(record)
        units = list(grouped.values())
    else:
        units = [[record] for record in records]

    buckets: dict[tuple[str, ...], list[list[TurnManifestRecord]]] = {}
    for unit in units:
        anchor = unit[0]
        key = tuple(_field_value(anchor, field) for field in config.stratify_by)
        buckets.setdefault(key, []).append(unit)
    return buckets


def _field_value(record: TurnManifestRecord, field: str) -> str:
    if field.startswith("metadata."):
        key = field.removeprefix("metadata.")
        value = record.metadata.get(key)
    else:
        value = getattr(record, field, None)
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return repr(value)


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total <= 0:
        return (0, 0, 0)
    ratio_sum = sum(ratios)
    exact = [total * ratio / ratio_sum for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: (exact[index] - counts[index], ratios[index]), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return (counts[0], counts[1], counts[2])


def _rebalance_empty_units(
    splits: dict[str, list[list[TurnManifestRecord]]],
    ratios: tuple[float, float, float],
) -> None:
    desired = [name for name, ratio in zip(SPLIT_NAMES, ratios, strict=True) if ratio > 0]
    total = sum(len(splits[name]) for name in SPLIT_NAMES)
    if total < len(desired):
        return

    for name in desired:
        if splits[name]:
            continue
        donors = [candidate for candidate in SPLIT_NAMES if len(splits[candidate]) > 1]
        if not donors:
            return
        donor = max(donors, key=lambda candidate: len(splits[candidate]))
        splits[name].append(splits[donor].pop())
