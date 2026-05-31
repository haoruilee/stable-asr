"""Normalize a vendor ASR transcript export for Stable-ASR command adapters.

This script is intentionally small: a real integration should run the upstream
ASR system first, write a JSONL transcript export, then invoke this normalizer
so Stable-ASR can evaluate it with the shared streaming metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_asr.data.converters import ASR_TRANSCRIPT_SCHEMAS, convert_streaming_asr_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, choices=ASR_TRANSCRIPT_SCHEMAS)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    count = convert_streaming_asr_jsonl(args.input, args.output, schema=args.schema)
    print(f"converted {count} {args.schema} transcript record(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
