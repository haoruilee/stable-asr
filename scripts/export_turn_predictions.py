"""Normalize and coverage-check external turn prediction exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_asr.data.manifest import load_manifest
from stable_asr.models.adapters import (
    PREDICTION_SCHEMAS,
    convert_turn_prediction_jsonl,
    validate_turn_prediction_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", choices=PREDICTION_SCHEMAS, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-extra", action="store_true")
    args = parser.parse_args()

    records = load_manifest(args.dataset)
    count = convert_turn_prediction_jsonl(args.raw, args.output, schema=args.schema)
    report = validate_turn_prediction_jsonl(
        records,
        args.output,
        allow_extra=args.allow_extra,
        dataset_path=args.dataset,
    )
    print(report.to_text())
    if not report.ok:
        raise SystemExit(1)
    print(f"converted {count} {args.schema} prediction row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
