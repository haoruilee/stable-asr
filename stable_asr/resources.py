"""Path resolution helpers for repository and installed platform assets."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_platform_path(path: str | Path) -> Path:
    """Resolve a repo-relative platform asset path.

    During repository development, commands read paths such as
    ``configs/paper/paper_smoke.json`` directly from the working tree. When the
    project is installed from a wheel, these non-Python assets are installed
    under ``share/stable-asr``. This helper keeps explicit paths working in both
    contexts without changing the documented CLI examples.
    """

    requested = Path(path)
    if requested.exists() or requested.is_absolute():
        return requested

    roots: list[Path] = []
    env_root = os.environ.get("STABLE_ASR_ASSET_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path(__file__).resolve().parents[1])
    roots.extend(
        [
            Path(sys.prefix) / "share" / "stable-asr",
            Path(sys.base_prefix) / "share" / "stable-asr",
        ]
    )
    for root in roots:
        candidate = root / requested
        if candidate.exists():
            return candidate
    return requested
