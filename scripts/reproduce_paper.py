"""Run Stable-ASR paper-facing experiment bundles.

This script is intentionally thin: it delegates to the package CLI so the same
entrypoint works for local development, CI, and future paper reproduction.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Stable-ASR paper smoke artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper/paper_smoke.json"),
        help="Paper experiment JSON config.",
    )
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        "-m",
        "stable_asr",
        "reproduce-paper",
        "--config",
        str(args.config),
    ]
    if args.skip_train:
        command.append("--skip-train")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())

