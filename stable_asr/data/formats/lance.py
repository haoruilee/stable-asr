"""Lance backend for turn manifests.

The backend is optional so Stable-ASR can keep a zero-dependency core package.
When ``pylance`` is installed, ``.lance`` manifests are stored as native Lance
datasets and can be included in paper-grade data-layer benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_asr.data.manifest import TurnManifestRecord


def load_lance(path: str | Path) -> list[TurnManifestRecord]:
    lance = _require_lance()
    table = lance.dataset(str(path)).to_table()
    return _table_to_records(table)


def take_lance(path: str | Path, indices: list[int]) -> list[TurnManifestRecord]:
    lance = _require_lance()
    if not indices:
        return []
    batch = lance.dataset(str(path)).take(indices)
    return _table_to_records(batch)


def _table_to_records(table: Any) -> list[TurnManifestRecord]:
    columns = table.to_pydict()
    rows = len(next(iter(columns.values()), []))
    records: list[TurnManifestRecord] = []
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


def write_lance(path: str | Path, records: list[TurnManifestRecord]) -> None:
    pa, lance = _require_pyarrow_lance()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = _records_to_columns(records)
    table = pa.table(rows)
    lance.write_dataset(
        table,
        str(path),
        mode="overwrite",
        commit_message="stable-asr turn manifest write",
    )


def _records_to_columns(records: list[TurnManifestRecord]) -> dict[str, list[Any]]:
    rows: dict[str, list[Any]] = {
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
    return rows


def _require_lance():
    try:
        import lance
    except Exception as exc:  # pragma: no cover - depends on optional env.
        raise RuntimeError("Lance support requires pylance. Install with: pip install 'stable-asr[lance]'") from exc
    if not hasattr(lance, "dataset") or not hasattr(lance, "write_dataset"):
        raise RuntimeError(
            "Lance support requires the pylance package that provides the Lance data format. "
            "Install with: pip install 'stable-asr[lance]'"
        )
    return lance


def _require_pyarrow_lance():
    try:
        import pyarrow as pa
    except Exception as exc:  # pragma: no cover - depends on optional env.
        raise RuntimeError("Lance support requires pyarrow. Install with: pip install 'stable-asr[lance]'") from exc
    return pa, _require_lance()
