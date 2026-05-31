"""Normalize a precomputed Whisper JSONL export for Stable-ASR final runs.

This bridge does not run Whisper inference. It validates a shared ASR manifest,
converts a checked Whisper-style raw export to Stable-ASR StreamingASRRecord
JSONL, and fails if record IDs do not cover the manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_asr.data.asr_manifest import ASRManifestRecord, load_asr_manifest
from stable_asr.data.converters.streaming_asr import convert_streaming_asr_rows
from stable_asr.data.formats.jsonl import iter_jsonl, write_jsonl


SCHEMA = "whisper"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_records = load_asr_manifest(args.manifest)
    raw_rows = [row for _, row in iter_jsonl(args.raw)]
    records = convert_streaming_asr_rows(raw_rows, schema=SCHEMA)
    rows = _enrich_records(records, manifest_records, schema=SCHEMA)
    _validate_coverage(rows, manifest_records)
    write_jsonl(args.output, rows)
    print(f"converted {len(rows)} {SCHEMA} record(s) to {args.output}")
    return 0


def _enrich_records(records, manifest_records: list[ASRManifestRecord], *, schema: str) -> list[dict[str, object]]:
    manifest_by_id = {record.id: record for record in manifest_records}
    rows: list[dict[str, object]] = []
    for record in records:
        row = record.to_dict()
        manifest = manifest_by_id.get(record.id)
        if manifest is not None:
            if not row.get("reference"):
                row["reference"] = manifest.text
            if float(row.get("audio_duration", 0.0)) <= 0.0 and manifest.duration is not None:
                row["audio_duration"] = manifest.duration
            metadata = dict(row.get("metadata", {}))
            metadata.update(
                {
                    "asr_schema": schema,
                    "audio": manifest.audio,
                    "language": manifest.language,
                    "source": manifest.source,
                }
            )
            row["metadata"] = metadata
        rows.append(row)
    return rows


def _validate_coverage(rows: list[dict[str, object]], manifest_records: list[ASRManifestRecord]) -> None:
    expected = {record.id for record in manifest_records}
    actual = {str(row.get("id", "")) for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing_ids={missing[:10]}")
        if extra:
            details.append(f"extra_ids={extra[:10]}")
        raise SystemExit("ERROR: raw ASR export does not match manifest coverage: " + "; ".join(details))


if __name__ == "__main__":
    raise SystemExit(main())
