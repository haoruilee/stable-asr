"""Weak turn-window generation from utterance-level ASR manifests."""

from __future__ import annotations

import random as _random
from dataclasses import dataclass

from stable_asr.data.asr_manifest import ASRManifestRecord
from stable_asr.data.manifest import TurnManifestRecord
from stable_asr.data.registry import summarize_records


@dataclass(frozen=True)
class ASRToTurnConfig:
    window_sec: float = 2.0
    include_complete: bool = True
    include_incomplete: bool = False
    # Randomised truncation range — each incomplete sample gets a ratio drawn
    # uniformly from [incomplete_ratio_min, incomplete_ratio_max].  Using a
    # range instead of a fixed value prevents the model from learning to
    # classify by duration alone (the "duration shortcut" bias).
    incomplete_ratio_min: float = 0.40
    incomplete_ratio_max: float = 0.85
    # Deprecated single-value shorthand kept for backwards compat — ignored
    # when incomplete_ratio_min/max are set to their non-0.65 defaults.
    incomplete_ratio: float = 0.65
    min_incomplete_sec: float = 0.4
    complete_pause_ms: int = 900
    incomplete_pause_ms: int = 250
    source: str = "asr_weak_turn_v0"
    drop_incomplete_text: bool = True
    seed: int = 42

    def validate(self) -> None:
        if self.window_sec <= 0:
            raise ValueError("window_sec must be positive")
        if not self.include_complete and not self.include_incomplete:
            raise ValueError("at least one of include_complete or include_incomplete must be enabled")
        if not (0.0 < self.incomplete_ratio_min <= self.incomplete_ratio_max < 1.0):
            raise ValueError("incomplete_ratio_min/max must satisfy 0 < min <= max < 1")
        if self.min_incomplete_sec <= 0:
            raise ValueError("min_incomplete_sec must be positive")
        if self.complete_pause_ms < 0 or self.incomplete_pause_ms < 0:
            raise ValueError("pause values must be non-negative")


@dataclass(frozen=True)
class ASRToTurnResult:
    records: list[TurnManifestRecord]
    input_records: int
    config: ASRToTurnConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "input_records": self.input_records,
            "output_records": len(self.records),
            "config": {
                "window_sec": self.config.window_sec,
                "include_complete": self.config.include_complete,
                "include_incomplete": self.config.include_incomplete,
                "incomplete_ratio_min": self.config.incomplete_ratio_min,
                "incomplete_ratio_max": self.config.incomplete_ratio_max,
                "min_incomplete_sec": self.config.min_incomplete_sec,
                "complete_pause_ms": self.config.complete_pause_ms,
                "incomplete_pause_ms": self.config.incomplete_pause_ms,
                "source": self.config.source,
                "drop_incomplete_text": self.config.drop_incomplete_text,
            },
            "summary": summarize_records(self.records),
        }

    def to_text(self) -> str:
        summary = summarize_records(self.records)
        lines = [
            "asr_to_turn:",
            f"- input_records: {self.input_records}",
            f"- output_records: {len(self.records)}",
            f"- source: {self.config.source}",
        ]
        labels = summary.get("turn_labels", {})
        if isinstance(labels, dict):
            lines.append("- turn_labels: " + (", ".join(f"{key}={value}" for key, value in labels.items()) or "none"))
        return "\n".join(lines)


def asr_records_to_turn_records(
    records: list[ASRManifestRecord],
    *,
    config: ASRToTurnConfig | None = None,
) -> ASRToTurnResult:
    config = config or ASRToTurnConfig()
    config.validate()
    rng = _random.Random(config.seed)

    output: list[TurnManifestRecord] = []
    for record in records:
        duration = float(record.duration if record.duration is not None else config.window_sec)
        if config.include_complete:
            output.append(_complete_record(record, duration=duration, config=config))
        if config.include_incomplete:
            ratio = rng.uniform(config.incomplete_ratio_min, config.incomplete_ratio_max)
            incomplete_end = min(duration - 0.05, max(config.min_incomplete_sec, duration * ratio))
            if incomplete_end <= 0:
                continue
            output.append(_incomplete_record(record, end=incomplete_end, duration=duration, config=config))
    return ASRToTurnResult(records=output, input_records=len(records), config=config)


def _complete_record(
    record: ASRManifestRecord,
    *,
    duration: float,
    config: ASRToTurnConfig,
) -> TurnManifestRecord:
    start = max(0.0, duration - config.window_sec)
    metadata = _base_metadata(record, full_duration=duration, strategy="asr_complete")
    metadata.update(
        {
            "pause_ms": config.complete_pause_ms,
            "vad_pause_ms": config.complete_pause_ms,
            "duration_ms": round((duration - start) * 1000.0, 3),
            "text_is_full_reference": True,
        }
    )
    return TurnManifestRecord.from_dict(
        {
            "id": f"{record.id}__complete",
            "audio": record.audio,
            "sample_rate": record.sample_rate,
            "start": round(start, 6),
            "end": round(duration, 6),
            "text": record.text,
            "asr_text": record.text,
            "turn_label": "complete",
            "action_label": "take_turn",
            "assistant_speaking": False,
            "overlap": False,
            "scenario": "asr_weak_complete",
            "language": record.language,
            "source": config.source,
            "metadata": metadata,
        }
    )


def _incomplete_record(
    record: ASRManifestRecord,
    *,
    end: float,
    duration: float,
    config: ASRToTurnConfig,
) -> TurnManifestRecord:
    start = max(0.0, end - config.window_sec)
    metadata = _base_metadata(record, full_duration=duration, strategy="asr_truncated_incomplete")
    metadata.update(
        {
            "pause_ms": config.incomplete_pause_ms,
            "vad_pause_ms": config.incomplete_pause_ms,
            "duration_ms": round((end - start) * 1000.0, 3),
            "truncation_ratio": round(end / duration, 6) if duration > 0 else None,
            "text_is_full_reference": not config.drop_incomplete_text,
        }
    )
    text = None if config.drop_incomplete_text else record.text
    return TurnManifestRecord.from_dict(
        {
            "id": f"{record.id}__incomplete",
            "audio": record.audio,
            "sample_rate": record.sample_rate,
            "start": round(start, 6),
            "end": round(end, 6),
            "text": text,
            "asr_text": text,
            "turn_label": "incomplete",
            "action_label": "keep_listening",
            "assistant_speaking": False,
            "overlap": False,
            "scenario": "asr_weak_incomplete",
            "language": record.language,
            "source": config.source,
            "metadata": metadata,
        }
    )


def _base_metadata(record: ASRManifestRecord, *, full_duration: float, strategy: str) -> dict[str, object]:
    metadata = {
        "derived_from": "asr_manifest",
        "asr_record_id": record.id,
        "asr_source": record.source,
        "asr_split": record.split,
        "speaker_id": record.speaker_id,
        "full_duration_sec": round(full_duration, 6),
        "label_strategy": strategy,
        "speaking_rate": 1.0,
        "snr_db": 20.0,
        "network_jitter_ms": 0.0,
    }
    for key, value in record.metadata.items():
        metadata[f"asr_metadata.{key}"] = value
    return {key: value for key, value in metadata.items() if value is not None}
