"""Demo external ASR command for Stable-ASR command adapter examples.

Real ASR integrations should write the same normalized streaming transcript
JSONL schema to the output path. This demo copies an existing fixture so the
command-comparison path can run without heavyweight ASR dependencies.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
