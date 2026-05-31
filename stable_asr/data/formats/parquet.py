"""Parquet backend for turn manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_asr.data.manifest import TurnManifestRecord


def load_parquet(path: str | Path) -> list[TurnManifestRecord]:
    pq = _require_pyarrow_parquet()
    table = pq.read_table(path)
    columns = table.to_pydict()
    records: list[TurnManifestRecord] = []
    rows = table.num_rows
    for index in range(rows):
        metadata_raw = columns.get("metadata_json", ["{}"] * rows)[index] or "{}"
        row: dict[str, Any] = {
            "id": columns["id"][index],
            "audio": columns["audio"][index],
            "sample_rate": columns["sample_rate"][index],
            "start": columns["start"][index],
            "end": columns["end"][index],
            "turn_label": columns["turn_label"][index],
            "action_label": columns["action_label"][index],
            "assistant_speaking": columns["assistant_speaking"][index],
            "overlap": columns["overlap"][index],
            "language": columns["language"][index],
            "source": columns["source"][index],
            "text": columns.get("text", [None] * rows)[index],
            "asr_text": columns.get("asr_text", [None] * rows)[index],
            "scenario": columns.get("scenario", [None] * rows)[index],
            "metadata": json.loads(metadata_raw),
        }
        records.append(TurnManifestRecord.from_dict(row))
    return records


def write_parquet(path: str | Path, records: list[TurnManifestRecord]) -> None:
    pa, pq = _require_pyarrow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {
        "id": [],
        "audio": [],
        "sample_rate": [],
        "start": [],
        "end": [],
        "turn_label": [],
        "action_label": [],
        "assistant_speaking": [],
        "overlap": [],
        "language": [],
        "source": [],
        "text": [],
        "asr_text": [],
        "scenario": [],
        "metadata_json": [],
    }
    for record in records:
        rows["id"].append(record.id)
        rows["audio"].append(record.audio)
        rows["sample_rate"].append(record.sample_rate)
        rows["start"].append(record.start)
        rows["end"].append(record.end)
        rows["turn_label"].append(record.turn_label)
        rows["action_label"].append(record.action_label)
        rows["assistant_speaking"].append(record.assistant_speaking)
        rows["overlap"].append(record.overlap)
        rows["language"].append(record.language)
        rows["source"].append(record.source)
        rows["text"].append(record.text)
        rows["asr_text"].append(record.asr_text)
        rows["scenario"].append(record.scenario)
        rows["metadata_json"].append(json.dumps(record.metadata, ensure_ascii=False, sort_keys=True))
    table = pa.table(rows)
    pq.write_table(table, path)


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - depends on optional env.
        raise RuntimeError("Parquet support requires pyarrow. Install with: pip install 'stable-asr[data]'") from exc
    return pa, pq


def _require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - depends on optional env.
        raise RuntimeError("Parquet support requires pyarrow. Install with: pip install 'stable-asr[data]'") from exc
    return pq

